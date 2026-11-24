"""Constrained log acquisitions over physical posteriors and frozen utilities.

BoTorch multiobjective capabilities are loaded only when building the selected
method. Constraints always see physical samples, never scalarized utilities.
"""
from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import inspect
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtr
import torch

from expdoe_dk.errors import EngineError, ErrorCode
from .contracts import AcquisitionDecision
from .knowledge_runtime import outcome_safety_intervals
from .multiobjective import MultiObjectiveConfig, resolve_noise_mode, resolve_preference_weights
from .objective_geometry import _feasible_baseline, objective_samples

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import AcquisitionFunction
    from expdoe_dk.domain import ObservationBatch, PendingBatch, Space
    from .objective_geometry import ObjectiveGeometry


_MC_SAMPLES = 128


def choose_multiobjective_acquisition(n_objectives: int, config: MultiObjectiveConfig, *,
                                    has_pending: bool, has_preferences: bool) -> AcquisitionDecision:
    """Choose a route; unresolved auto noise is conservatively noisy.

    Builders pass resolved noise/weights and the caller's external pending state.
    Internal greedy selections are applied afterwards via set_X_pending.
    For an explicit HV override, has_preferences=True requires effective
    weights in config so equal weights can be distinguished from a tradeoff.
    """
    if isinstance(n_objectives, bool) or not isinstance(n_objectives, int) or n_objectives < 2:
        raise EngineError(ErrorCode.CONFIG_INVALID, "Multiobjective acquisition requires at least two objectives")
    kind = config.acquisition
    if kind != "auto":
        if kind == "q_log_ehvi" and (config.noise_mode != "noiseless" or has_pending):
            raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
                              "qLogEHVI requires explicit noiseless mode and no external pending points")
        if has_preferences and kind in ("q_log_ehvi", "q_log_nehvi"):
            weights = config.preference_weights
            if (weights is None or len(weights) != n_objectives or not all(
                    math.isclose(w, next(iter(weights.values())), rel_tol=1e-9, abs_tol=0.)
                    for w in weights.values())):
                raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
                                  "HV overrides require resolved equal weights; non-equal preferences "
                                  "require qLogNParEGO-style scalarization")
        return AcquisitionDecision(kind, "explicit_override")
    if has_preferences:
        return AcquisitionDecision("q_log_nparego", "preference_override")
    if n_objectives >= 5:
        return AcquisitionDecision("q_log_nparego", "many_objectives")
    if config.noise_mode == "noiseless" and not has_pending:
        return AcquisitionDecision("q_log_ehvi", "noiseless_without_pending")
    return AcquisitionDecision("q_log_nehvi", "noise_or_pending")


def joint_feasibility(space, fitted, X) -> np.ndarray:
    """Independent-output latent probability of all hard physical outcomes.

    X is a physical DataFrame, as for fitted.predict. Parameter feasibility and
    the configured probability threshold are separate caller-side filters.
    """
    intervals = outcome_safety_intervals(space)
    probability = np.ones(len(X), dtype=float)
    if not intervals or not len(X):
        return probability
    prediction = fitted.predict(X)
    means, variances = prediction.mean, prediction.variance
    if (means.shape != (len(X), len(space.objectives)) or variances.shape != means.shape
            or not np.isfinite(means).all() or not np.isfinite(variances).all()
            or (variances < 0).any()):
        raise EngineError(ErrorCode.MODEL_FIT_FAILED, "Physical outcome marginals must be finite and aligned")
    for name, (lo, hi) in intervals.items():
        index = space.objectives.index(name)
        mean, variance = means[:, index], variances[:, index]
        std = np.sqrt(variance)
        scale = np.where(std > 0, std, 1.)
        lower, upper = (lo - mean) / scale, (hi - mean) / scale
        # Use survival probabilities in the positive tail to avoid 1 - 1.
        mass = np.where(lower > 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))
        mass = np.where(std > 0, mass, ((lo <= mean) & (mean <= hi)).astype(float))
        probability *= np.clip(mass, 0., 1.)
    return probability


def _capability(module, symbol):
    try:
        return getattr(import_module(module), symbol)
    except (ImportError, AttributeError) as error:
        raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
                          f"Multiobjective acquisition requires {module}.{symbol}",
                          hints=["Install BoTorch with the required log-acquisition capabilities."]) from error


def _construct(factory, **kwargs):
    try:
        inspect.signature(factory).bind(**kwargs)
    except (TypeError, ValueError) as error:
        raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
                          f"Multiobjective acquisition requires {factory.__name__} supporting "
                          f"{', '.join(kwargs)}",
                          hints=["Install BoTorch with physical-sample constraints and seeded samplers."]) from error
    return factory(**kwargs)


def build_multiobjective_acquisition(space: Space, fitted, observations: ObservationBatch,
                                    pending: PendingBatch | None, runtime, config: MultiObjectiveConfig,
                                    geometry: ObjectiveGeometry, *, seed: int,
                                    slot: int) -> tuple[AcquisitionFunction, dict]:
    """Build a BoTorch acquisition and JSON diagnostics for one greedy slot.

    Pass ORIGINAL observations and external PendingBatch, even for noiseless
    fitting. Later call set_X_pending with combined external/greedy model-frame
    coordinates without rebuilding the route. Geometry is never refitted here.
    """
    config.validate_for_space(space)
    mode = resolve_noise_mode(observations, config)
    if tuple(space.objectives) != geometry.objective_names:
        raise EngineError(ErrorCode.CONFIG_INVALID, "Geometry objective order must match Space")
    resolved_weights, _ = resolve_preference_weights(space, runtime, config)
    frozen_weights = geometry.preference_weights
    if ((resolved_weights is None) != (frozen_weights is None)
            or (resolved_weights is not None and not all(
                math.isclose(a, b, rel_tol=1e-9, abs_tol=0.)
                for a, b in zip(resolved_weights, frozen_weights, strict=True)))):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Preferences must match the frozen geometry")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (seed, slot)):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Acquisition seed and slot must be non-negative integers")
    intervals = outcome_safety_intervals(space)
    routing_config = replace(config, noise_mode=mode, preference_weights=(
        None if frozen_weights is None else dict(zip(space.objectives, frozen_weights, strict=True))))
    decision = choose_multiobjective_acquisition(len(space.objectives), routing_config,
        has_pending=pending is not None and bool(len(pending.X)),
        has_preferences=geometry.preference_weights is not None)
    successful = observations.successful()
    if len(successful.X) < 3:
        raise EngineError(ErrorCode.INSUFFICIENT_DATA, "At least three successful observations are required")
    if set(successful.Y.columns) != set(space.objectives):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Observation columns must exactly match Space objectives")
    baseline = space.physical_to_model(successful.X)
    X_pending = None if pending is None or not len(pending.X) else space.physical_to_model(pending.X)
    constraints = []
    for name, (lo, hi) in intervals.items():
        index = space.objectives.index(name)
        constraints.append(lambda samples, i=index, lower=lo, upper=hi:
                           torch.maximum(lower - samples[..., i], samples[..., i] - upper))

    def utilities(samples, X=None):
        return objective_samples(samples, space, geometry, runtime, with_preferences=True)

    weights = None
    if decision.kind == "q_log_nparego":
        cls = _capability("botorch.acquisition.logei", "qLogNoisyExpectedImprovement")
        objective_cls = _capability("botorch.acquisition.objective", "GenericMCObjective")
        chebyshev = _capability("botorch.utils.multi_objective.scalarization", "get_chebyshev_scalarization")
        weights = (list(geometry.preference_weights) if geometry.preference_weights is not None else
                   np.random.default_rng(np.random.SeedSequence([seed, slot])).dirichlet(
                       np.ones(len(space.objectives))).tolist())
        # Empty Y disables BoTorch's data-dependent normalization. Centering at
        # the frozen ideal gives min(w * (z - ideal)) + .05 * sum(w * (z - ideal)).
        scalarize = _construct(chebyshev, weights=baseline.new_tensor(weights),
                              Y=baseline.new_empty((0, len(space.objectives))), alpha=.05)

        def scalar_objective(samples, X=None):
            z = utilities(samples, X)
            return scalarize(z - z.new_tensor(geometry.acquisition_ideal))

        objective = _construct(objective_cls, objective=scalar_objective)
        kwargs = dict(X_baseline=baseline, prune_baseline=False, cache_root=False)
    else:
        symbol = ("qLogExpectedHypervolumeImprovement" if decision.kind == "q_log_ehvi"
                  else "qLogNoisyExpectedHypervolumeImprovement")
        cls = _capability("botorch.acquisition.multi_objective.logei", symbol)
        objective_cls = _capability("botorch.acquisition.multi_objective.objective", "GenericMCMultiOutputObjective")
        objective = _construct(objective_cls, objective=utilities)
        kwargs = dict(ref_point=list(geometry.acquisition_reference))
        if decision.kind == "q_log_ehvi":
            partition_cls = _capability("botorch.utils.multi_objective.box_decompositions.non_dominated",
                                        "NondominatedPartitioning")
            feasible = _feasible_baseline(space, successful, runtime)
            physical = baseline.new_tensor(successful.Y.loc[:, space.objectives].to_numpy(dtype=float, copy=True))
            kwargs["partitioning"] = _construct(partition_cls,
                ref_point=baseline.new_tensor(geometry.acquisition_reference), Y=utilities(physical)[feasible])
        else:
            kwargs.update(X_baseline=baseline, prune_baseline=False, cache_root=False)
    acquisition = _construct(cls, model=fitted.acquisition_model(),
        sampler=fitted.acquisition_sampler(sample_shape=torch.Size([_MC_SAMPLES]), seed=seed),
        objective=objective, constraints=constraints or None, X_pending=X_pending, **kwargs)
    diagnostics = {
        "acquisition_kind": decision.kind, "acquisition_name": cls.__name__,
        "acquisition_reason": decision.reason, "noise_mode": mode,
        "objective_names": list(space.objectives), "geometry": geometry.to_dict(),
        "scalarization_weights": weights, "seed": seed, "slot": slot,
        "mc_samples": _MC_SAMPLES,
    }
    return acquisition, diagnostics


__all__ = ["choose_multiobjective_acquisition", "build_multiobjective_acquisition", "joint_feasibility"]
