"""Production GP recommendation with exact physical identities and fail-closed safety.

Small finite spaces are enumerated. Mixed spaces enumerate their finite factors
and optimize continuous subspaces, retaining an acquisition-scored multistart
pool under hard rejection. Larger finite products use a bounded seeded pool;
this is a declared search policy, never a fallback after a numerical failure.
"""
from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from scipy.special import ndtr

from ..domain import ObservationBatch, PendingBatch, Space
from ..doe.design import MAX_ENUMERATION_ROWS, MAX_POOL_ROWS, MAX_REQUESTED_ROWS, candidate_keys
from ..errors import EngineError, ErrorCode
from ..knowledge import Knowledge
from .contracts import RecommendationResult
from .candidates import prepare_catalog
from .knowledge_runtime import compile_runtime
from .linear_search import (
    ContinuousSubspace, compile_linear_search, generate_linear_candidates, linear_seed_pool,
)
from .surrogates import FittedSurrogate, fit_surrogate
from .multiobjective import MultiObjectiveConfig

if TYPE_CHECKING:
    from .objective_geometry import ObjectiveGeometry

_MINIMUM_FEASIBILITY = .95
_MC_SAMPLES = 128


def _invalid(message, **details):
    return EngineError(ErrorCode.CONFIG_INVALID, message, details=details)


def _check_precision(space):
    """Reject known non-injective finite codecs before keys can alias inputs.

    Small finite factors are checked exhaustively. Larger log-integer grids
    are unsupported: sparse probes cannot certify their interior precision.
    Large linear integer grids must satisfy a conservative binary64 cardinality
    bound and endpoint/midpoint round trips. This does not redesign Space's codec.
    """
    for parameter in space.params:
        if parameter.kind == "ordinal" and any(value is None for value in parameter.values):
            raise _invalid("Ordinal null levels are not preserved by the current batch/result copy boundary",
                           parameter=parameter.name, limitation="ordinal_null_copy_coercion")
        if parameter.kind not in {"integer", "discrete"}:
            continue
        count = parameter.cardinality
        if (parameter.kind == "integer" and count > MAX_ENUMERATION_ROWS
                and (parameter.transform == "log" or parameter.log_scale)):
            raise _invalid("Large log-integer domain precision cannot be validated exhaustively",
                           parameter=parameter.name, cardinality=count, limit=MAX_ENUMERATION_ROWS)
        if parameter.kind == "integer" and count > 2**52:
            raise _invalid("Integer domain exceeds certified model-coordinate precision; values may alias",
                           parameter=parameter.name, cardinality=count)
        if count <= MAX_ENUMERATION_ROWS or parameter.values is not None:
            values = parameter.numeric_levels
        else:
            indices = sorted({0, 1, 2, count // 2 - 1, count // 2, count // 2 + 1,
                              count - 3, count - 2, count - 1})
            lo = parameter.bounds[0]
            step = parameter.step or 1
            values = [lo + index * step for index in indices]
        encoded = parameter.encode(values)
        decoded = parameter.decode(encoded)
        if len(set(encoded)) != len(values) or any(a != b for a, b in zip(values, decoded)):
            raise _invalid("Finite parameter codec loses precision or aliases distinct conditions",
                           parameter=parameter.name)


def _validate_frame(space, frame, label):
    if set(frame.columns) != set(space.param_names) or not frame.columns.is_unique:
        raise _invalid(f"{label} columns must match Space parameters")
    try:
        space.physical_to_model(frame)
    except (TypeError, ValueError, OverflowError) as error:
        raise _invalid(f"Invalid {label} physical values: {error}") from error


def _utility(samples, objective, runtime):
    """Evaluate Objective.to_utility on each physical sample, with exact slopes.

    Objective's public implementation is NumPy-based. The equivalent piecewise
    linear tensor expression supplies its derivative without changing its value.
    """
    y = samples[..., 0]
    if objective.direction == "maximize":
        differentiable = y
    elif objective.direction == "minimize":
        differentiable = -y
    elif isinstance(objective.target, tuple):
        lo, hi = objective.target
        differentiable = -torch.maximum((lo - y).clamp_min(0), (y - hi).clamp_min(0))
    else:
        differentiable = -(y - objective.target).abs()
    native = torch.as_tensor(objective.to_utility(y.detach().cpu().numpy()), dtype=y.dtype, device=y.device)
    utility = differentiable + (native - differentiable).detach()
    for interval in runtime.get("outcome_constraints", ()):
        # RuntimeOutcomeRange.margin is positive inside the interval. Only its
        # negative part is a soft physical-unit shortfall, never a rejection.
        lower, upper = interval.payload["lower"], interval.payload["upper"]
        utility = utility - torch.maximum((lower - y).clamp_min(0), (y - upper).clamp_min(0))
    for preference in runtime.get("acquisition_preferences", ()):
        utility = utility * preference({objective.name: 1.})
    return utility


def _outcome_bounds(space):
    lo, hi = -math.inf, math.inf
    for constraint in space.outcome_constraints:
        if not constraint.hard or constraint.operator == "==":
            raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
                "Outcome safety supports hard one-sided inequalities, not soft or equality constraints")
        if constraint.operator == "<=":
            hi = min(hi, constraint.bound)
        else:
            lo = max(lo, constraint.bound)
    return lo, hi


def _safety_probability(prediction, bounds):
    lo, hi = bounds
    mean, variance = prediction.mean[:, 0], prediction.variance[:, 0]
    std = np.sqrt(variance)
    scale = np.where(std > 0, std, 1.)
    probability = np.maximum(0., ndtr((hi - mean) / scale) - ndtr((lo - mean) / scale))
    return np.where(std > 0, probability, ((lo <= mean) & (mean <= hi)).astype(float))


def _make_acquisition(space, fitted: FittedSurrogate, baseline, pending, runtime, seed):
    constraints = []
    for constraint in space.outcome_constraints:
        if constraint.operator == "<=":
            constraints.append(lambda samples, bound=constraint.bound: samples[..., 0] - bound)
        else:
            constraints.append(lambda samples, bound=constraint.bound: bound - samples[..., 0])
    return qLogNoisyExpectedImprovement(
        model=fitted.acquisition_model(), X_baseline=space.physical_to_model(baseline),
        X_pending=None if pending is None or pending.empty else space.physical_to_model(pending),
        sampler=SobolQMCNormalSampler(torch.Size([_MC_SAMPLES]), seed=seed),
        objective=GenericMCObjective(lambda samples, X=None: _utility(samples, space.objective_specs[0], runtime)),
        constraints=constraints or None, prune_baseline=False, cache_root=False,
    )


def _search_pool(space, q, seed, linear_rows=()):
    finite = [i for i, parameter in enumerate(space.params) if parameter.kind != "continuous"]
    continuous = [i for i, parameter in enumerate(space.params) if parameter.kind == "continuous"]
    cardinality = math.prod(space.params[i].cardinality for i in finite)
    bounds = torch.tensor([parameter.model_bounds for parameter in space.params], dtype=torch.double).T
    bounded = cardinality > MAX_ENUMERATION_ROWS
    if bounded:
        if linear_rows:
            raise _invalid("Linear equality search exceeds the finite enumeration limit; "
                           "provide a feasible candidate catalog with candidates= instead",
                           finite_combinations=cardinality, limit=MAX_ENUMERATION_ROWS)
        size = min(MAX_POOL_ROWS, max(2048, q * 32))
        unit = torch.quasirandom.SobolEngine(space.n_dims, scramble=True, seed=seed).draw(size).double()
        coordinates = bounds[0] + unit * (bounds[1] - bounds[0])
        return space.model_to_physical(coordinates), (), continuous, {
            "search_strategy": "seeded_bounded_pool", "finite_combinations": cardinality,
            "degradation_reason": "Finite product exceeds the 100000 enumeration limit; no exhaustive optimum claim",
            "pool_size": size,
        }
    levels = []
    for index in finite:
        parameter = space.params[index]
        physical = parameter.numeric_levels if parameter.kind in {"integer", "discrete"} else parameter.values
        levels.append(parameter.encode(physical))
    combinations = tuple(dict(zip(finite, values)) for values in itertools.product(*levels))
    samples = max(4, min(128, 512 // max(1, cardinality))) if continuous else 1
    if linear_rows:
        samples = min(max(samples, q + 2), MAX_POOL_ROWS // cardinality)
        pool, subspaces, feasible_count = linear_seed_pool(
            space, bounds, continuous, combinations, linear_rows, samples, seed,
        )
        return pool, subspaces, continuous, {
            "search_strategy": "enumerated_mixed_linear_polytope",
            "finite_combinations": cardinality, "linear_subspaces_feasible": feasible_count,
            "degradation_reason": None, "pool_size": len(pool),
        }
    samples = min(samples, MAX_POOL_ROWS // max(1, cardinality))
    unit = torch.quasirandom.SobolEngine(space.n_dims, scramble=True, seed=seed).draw(samples).double()
    initial = bounds[0] + unit * (bounds[1] - bounds[0])
    frames = []
    for fixed in combinations:
        values = initial.clone()
        for index, value in fixed.items():
            values[:, index] = value
        frames.append(space.model_to_physical(values))
    pool = pd.concat(frames, ignore_index=True)
    return pool, tuple(ContinuousSubspace(fixed) for fixed in combinations) if continuous else (), continuous, {
        "search_strategy": "enumerated_mixed_multistart" if continuous else "finite_enumeration",
        "finite_combinations": cardinality, "degradation_reason": None, "pool_size": len(pool),
    }


def _filter(space, frame, excluded, fitted, safety_bounds, *,
            feasibility=None, minimum_probability=_MINIMUM_FEASIBILITY):
    mask = space.feasibility_mask(frame).numpy()
    frame = frame.loc[mask].reset_index(drop=True)
    if len(frame) and (safety_bounds is not None or feasibility is not None):
        probabilities = (feasibility(frame) if feasibility is not None else
                         _safety_probability(fitted.predict(frame), safety_bounds))
        frame = frame.loc[probabilities >= minimum_probability].reset_index(drop=True)
    selected, seen = [], set(excluded)
    for i in range(len(frame)):
        key = next(iter(candidate_keys(frame.iloc[[i]], space)))
        if key not in seen:
            selected.append(i)
            seen.add(key)
    return frame.iloc[selected].reset_index(drop=True)


def _score(acquisition, coordinates):
    with torch.no_grad():
        values = torch.cat([acquisition(part.unsqueeze(-2)).reshape(-1)
                            for part in coordinates.split(128)])
    if not torch.isfinite(values).all():
        raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE, "Acquisition produced non-finite scores")
    return values


def recommend(space: Space, observations: ObservationBatch, *, knowledge: Knowledge | None = None,
              pending: PendingBatch | None = None, q: int = 1, seed: int = 0,
              candidates: pd.DataFrame | None = None,
              candidate_ids: Sequence[str] | None = None,
              multiobjective: MultiObjectiveConfig | None = None) -> RecommendationResult:
    """Return a complete, unique GP-selected batch, or raise a typed error.

    Requires three successful observations. Safety additionally requires an
    observed feasible baseline and, by default, >=0.95 joint interval probability per point
    under the fitted latent GP; this is model-based feasibility, NOT guaranteed
    safety. Greedy conditional qLogNEI uses the pending API for both externally
    pending and already-selected batch points. All returned values are physical.

    With ``candidates``, score only those original rows and return aligned
    ``candidate_ids``. IDs are unique non-empty strings, separate from trial
    IDs and response columns. Duplicate physical conditions are rejected.
    All observed conditions (including failures) and pending conditions remain
    excluded. Sorting by stable ID makes selection independent of catalog order.

    Multiple objectives use fixed observed utility geometry and conditional
    log-HV or NParEGO-style acquisitions. Diagnostics include the geometry,
    per-slot weights and conditional scores, never a mixed-weight joint score.
    """
    return _recommend_resolved(space, observations, knowledge=knowledge, pending=pending,
        q=q, seed=seed, candidates=candidates, candidate_ids=candidate_ids,
        multiobjective=multiobjective, geometry=None)


def _recommend_resolved(space: Space, observations: ObservationBatch, *,
                        knowledge: Knowledge | None = None, pending: PendingBatch | None = None,
                        q: int = 1, seed: int = 0, candidates: pd.DataFrame | None = None,
                        candidate_ids: Sequence[str] | None = None,
                        multiobjective: MultiObjectiveConfig | None = None,
                        geometry: ObjectiveGeometry | None = None) -> RecommendationResult:
    """Shared search entry for Campaign's frozen geometry and standalone calls.

    A supplied geometry is reused, not resolved from later observations. The
    caller owns persistence and commits it only after a successful result.
    """
    if not isinstance(space, Space) or not isinstance(observations, ObservationBatch):
        raise _invalid("Expected Space and ObservationBatch")
    if knowledge is not None and not isinstance(knowledge, Knowledge):
        raise _invalid("knowledge must be Knowledge or None")
    if pending is not None and not isinstance(pending, PendingBatch):
        raise _invalid("pending must be PendingBatch or None")
    if type(q) is not int or not 1 <= q <= MAX_REQUESTED_ROWS or type(seed) is not int or seed < 0:
        raise _invalid("q must be a positive bounded integer and seed a non-negative integer")
    is_multiobjective = space.n_objectives > 1
    if multiobjective is not None:
        if not isinstance(multiobjective, MultiObjectiveConfig):
            raise _invalid("multiobjective must be MultiObjectiveConfig or None")
        multiobjective.validate_for_space(space)
    if geometry is not None:
        from .objective_geometry import ObjectiveGeometry
        if not is_multiobjective or not isinstance(geometry, ObjectiveGeometry):
            raise _invalid("geometry requires multiple objectives and ObjectiveGeometry")
    if is_multiobjective:
        from .acquisition import build_multiobjective_acquisition, joint_feasibility
        from .multiobjective import resolve_noise_mode
        from .objective_geometry import _feasible_baseline, resolve_geometry
        multiobjective = multiobjective if multiobjective is not None else MultiObjectiveConfig()
        multiobjective.validate_for_space(space)
        noise_mode = resolve_noise_mode(observations, multiobjective)
    if candidates is None and candidate_ids is not None:
        raise _invalid("candidate_ids requires candidates")
    _check_precision(space)
    catalog, catalog_ids = None, {}
    _validate_frame(space, observations.X, "observations")
    if pending is not None:
        _validate_frame(space, pending.X, "pending")
    if candidates is not None:
        catalog, catalog_ids = prepare_catalog(
            space, candidates, candidate_ids,
            references=(observations.X,) + (() if pending is None else (pending.X,)),
        )
    knowledge = knowledge if knowledge is not None else Knowledge()
    runtime = compile_runtime(space, knowledge, observations)
    design_space = space.with_constraints(*runtime["parameter_constraints"])
    linear_rows = compile_linear_search(design_space) if catalog is None else ()
    successful = observations.successful()
    safety_bounds = _outcome_bounds(space) if space.outcome_constraints and not is_multiobjective else None
    safe_baseline_count = None
    model_observations = observations
    if is_multiobjective:
        if geometry is None:
            geometry = resolve_geometry(space, observations, runtime, multiobjective)
        safe_baseline_count = int(_feasible_baseline(space, successful, runtime).sum())
        if noise_mode == "noiseless" and observations.Yvar is None:
            # Zero noise is a model setting, never a fabricated measurement.
            model_observations = ObservationBatch(observations.X, observations.Y,
                Yvar=pd.DataFrame(0., index=observations.Y.index, columns=observations.Y.columns),
                status=observations.status, ids=observations.ids)
    if safety_bounds is not None:
        if list(successful.Y.columns) != space.objectives:
            raise EngineError(ErrorCode.INSUFFICIENT_DATA, "Safety requires measured physical outcomes")
        values = successful.Y.iloc[:, 0].to_numpy(dtype=float)
        safe_baseline_count = int(((values >= safety_bounds[0]) & (values <= safety_bounds[1])).sum())
        if len(values) < 3 or safe_baseline_count == 0:
            raise EngineError(ErrorCode.INSUFFICIENT_DATA,
                "Safety requires three successful observations and an observed feasible baseline")
    try:
        fitted = fit_surrogate(space, model_observations, knowledge=knowledge, seed=seed)
    except EngineError:
        raise
    except Exception as error:
        raise EngineError(ErrorCode.MODEL_FIT_FAILED, f"GP fit failed: {error}") from error
    excluded = candidate_keys(observations.X, space)
    original_pending = None if pending is None else pending.X
    if original_pending is not None:
        excluded |= candidate_keys(original_pending, space)
    feasibility = (lambda frame: joint_feasibility(space, fitted, frame)) if is_multiobjective else None
    minimum_probability = multiobjective.min_joint_feasibility if is_multiobjective else _MINIMUM_FEASIBILITY
    filter_options = dict(feasibility=feasibility, minimum_probability=minimum_probability)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if is_multiobjective:
                acquisition, diagnostic = build_multiobjective_acquisition(
                    space, fitted, observations, pending, runtime, multiobjective, geometry,
                    seed=seed, slot=0)
            else:
                acquisition = _make_acquisition(space, fitted, successful.X, original_pending, runtime, seed)
            if catalog is None:
                pool, subspaces, continuous, search = _search_pool(space, q, seed, linear_rows)
            else:
                pool, subspaces, continuous = catalog, (), ()
                search = {"search_strategy": "provided_catalog", "pool_size": len(catalog),
                          "degradation_reason": None, "catalog_order": "stable_id",
                          "duplicate_condition_policy": "reject"}
            pool = _filter(design_space, pool, excluded, fitted, safety_bounds, **filter_options)
            if catalog is not None and len(pool) < q:
                raise EngineError(ErrorCode.NO_FEASIBLE_CANDIDATE,
                    "Candidate catalog cannot supply a complete unique feasible batch",
                    details={"requested": q, "available": len(pool), **search})
            chosen = []
            slot_diagnostics, conditional_values = [], []
            optimized_count = 0
            bounds = torch.tensor([parameter.model_bounds for parameter in space.params], dtype=torch.double).T
            for slot in range(q):
                if is_multiobjective:
                    if slot:
                        acquisition, diagnostic = build_multiobjective_acquisition(
                            space, fitted, observations, pending, runtime, multiobjective, geometry,
                            seed=seed, slot=slot)
                    slot_diagnostics.append(diagnostic)
                    conditioning = ([original_pending] if original_pending is not None else []) + chosen
                    condition = pd.concat(conditioning, ignore_index=True) if conditioning else None
                    acquisition.set_X_pending(None if condition is None or condition.empty
                                              else space.physical_to_model(condition))
                current = pool
                if continuous and subspaces:
                    # Optimize each declared finite subspace; keep feasible seeded
                    # starts in the scored candidate set even when an optimum is
                    # rejected. Numerical failure never switches search policy.
                    optimized = []
                    for subspace in subspaces:
                        optimizer_kwargs = {"num_restarts": 2, "raw_samples": 32}
                        if subspace.seeds is not None:
                            starts = subspace.seeds
                            best = _score(acquisition, starts).topk(min(len(starts), 2)).indices
                            optimizer_kwargs = {
                                "num_restarts": len(best), "raw_samples": None,
                                "batch_initial_conditions": starts[best].unsqueeze(-2),
                                "equality_constraints": list(subspace.equalities) or None,
                                "inequality_constraints": list(subspace.inequalities) or None,
                                "gen_candidates": generate_linear_candidates,
                            }
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", OptimizationWarning)
                            candidate, value = optimize_acqf(acquisition, bounds=bounds, q=1,
                                fixed_features=subspace.fixed, **optimizer_kwargs,
                                options={"maxiter": 80, "batch_limit": 2},
                                retry_on_optimization_warning=False)
                        if not torch.isfinite(value).all():
                            raise ValueError("Continuous optimization produced a non-finite value")
                        optimized.append(space.model_to_physical(candidate))
                        optimized_count += 1
                    current = pd.concat([pool, *optimized], ignore_index=True)
                current = _filter(design_space, current, excluded, fitted, safety_bounds, **filter_options)
                if current.empty:
                    raise EngineError(ErrorCode.NO_FEASIBLE_CANDIDATE,
                        "Search cannot produce a complete unique feasible batch",
                        details={"requested": q, "selected": len(chosen), **search})
                scores = _score(acquisition, space.physical_to_model(current))
                selected = current.iloc[[int(scores.argmax())]].reset_index(drop=True)
                chosen.append(selected)
                if is_multiobjective:
                    conditional_values.append(float(scores.max().item()))
                excluded |= candidate_keys(selected, space)
                if not is_multiobjective:
                    condition = pd.concat(([original_pending] if original_pending is not None else []) + chosen,
                                          ignore_index=True)
                    acquisition.set_X_pending(space.physical_to_model(condition))
            frame = pd.concat(chosen, ignore_index=True)
            if not is_multiobjective:
                acquisition.set_X_pending(None if original_pending is None or original_pending.empty
                                          else space.physical_to_model(original_pending))
                with torch.no_grad():
                    joint_value = float(acquisition(space.physical_to_model(frame).unsqueeze(0)).item())
                if not math.isfinite(joint_value):
                    raise ValueError("Joint acquisition is non-finite")
    except EngineError:
        raise
    except Exception as error:
        raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
            f"Acquisition search failed; no fallback was returned: {error}") from error
    predictions = fitted.predict(frame)
    avoided = candidate_keys(observations.X, space) | (set() if original_pending is None
                                                       else candidate_keys(original_pending, space))
    final_keys = candidate_keys(frame, space)
    probabilities = (feasibility(frame) if feasibility is not None else
                     None if safety_bounds is None else _safety_probability(predictions, safety_bounds))
    if (len(frame) != q or len(final_keys) != q or final_keys & avoided
            or not design_space.feasibility_mask(frame).all()
            or (probabilities is not None and (probabilities < minimum_probability).any())):
        raise EngineError(ErrorCode.NO_FEASIBLE_CANDIDATE, "Final physical candidate validation failed")
    acquisition_diagnostics = ({} if is_multiobjective else {
        "acquisition_name": "qLogNoisyExpectedImprovement",
        "acquisition_value_kind": "joint_batch", "joint_acquisition_value": joint_value,
    })
    diagnostics = {**search, **acquisition_diagnostics,
        "seed": seed, "mc_samples": _MC_SAMPLES, "pending_count": 0 if pending is None else len(pending.X),
        "candidate_keys": [next(iter(candidate_keys(frame.iloc[[i]], space))) for i in range(q)],
        "continuous_subspaces_optimized": optimized_count,
        "parameter_constraint_sources": [item.source_pattern_id for item in runtime["parameter_constraints"]],
        "transforms": fitted.transforms, "exclusions": list(fitted.exclusions),
        "soft_outcome_preference": "additive negative physical interval shortfall; no feasibility rejection",
    }
    if is_multiobjective:
        diagnostics.update({key: value for key, value in slot_diagnostics[0].items()
                            if key not in {"slot", "scalarization_weights"}})
        diagnostics.update({
            "weights_by_slot": [item["scalarization_weights"] for item in slot_diagnostics],
            "conditional_acquisition_values": conditional_values,
            "acquisition_value_kind": "conditional_per_slot",
            "observation_ids": list(observations.ids),
            "pending_ids": [] if pending is None else list(pending.ids),
            "strategy_explanation": (
                "Greedy conditional multiobjective log acquisition uses independent physical GP outputs "
                "and frozen observed utility geometry. Each slot conditions external pending and earlier "
                "selections; conditional scores correspond to that slot's recorded weights, not a common "
                "mixed-weight batch objective. The shared search scores only original rows for a provided "
                "catalog, otherwise enumerates finite subspaces or uses the declared seeded pool and "
                "continuous optimization. Physical constraints, uniqueness and joint model-based "
                "feasibility are rechecked; this is not guaranteed safety. No numerical fallback is used."),
        })
    else:
        diagnostics["strategy_explanation"] = (
            "Greedy conditional joint MC log noisy expected improvement uses physical objective utility "
            "and GP posterior uncertainty, with external pending and selected points conditioned through "
            "the acquisition pending API. Finite enumeration or the declared seeded pool supplies "
            "candidates; continuous subspaces use multistart optimization plus a scored seeded candidate pool. "
            "Hard physical constraints and exact duplicate exclusions are rechecked. "
            "Joint acquisition is reported for the complete batch, not attributed to individual points. "
            "Any outcome safety is model-based feasibility, not guaranteed safety.")
    diagnostics["outcome_safety"] = {"basis": "model_based_feasibility", "guaranteed": False,
        "minimum_probability": minimum_probability,
        "observed_feasible_baseline_count": safe_baseline_count,
        "probabilities": None if probabilities is None else probabilities.tolist()}
    ids = tuple(catalog_ids[key] for key in diagnostics["candidate_keys"]) if catalog is not None else ()
    if catalog is not None and not is_multiobjective:
        diagnostics["strategy_explanation"] = (
            "Greedy conditional joint MC log noisy expected improvement scores only original catalog rows. "
            "Stable-ID ordering resolves exact score ties; IDs and physical values are returned together. "
            "Observed and pending conditions are excluded, with pending points conditioned in the acquisition. "
            "The existing knowledge, physical objective utility, hard constraints and model-based outcome "
            "safety checks apply. No continuous optimization, Cartesian expansion or random fallback is used.")
    return RecommendationResult(frame, predictions, diagnostics, candidate_ids=ids)


__all__ = ["recommend"]
