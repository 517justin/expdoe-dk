"""Capability-aware, deterministic initial design generation."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from ..domain import Space
from ..errors import EngineError, ErrorCode
from .constrained import (
    _physical_frame_from_records,
    _repair_design_physical,
    _unit_to_physical_array,
)
from .lhs import latin_hypercube_sample


METHOD_KINDS = {
    "lhs_maximin": frozenset({"continuous", "integer", "discrete", "ordinal"}),
    "lhs_random": frozenset({"continuous", "integer", "discrete", "ordinal"}),
    "sobol": frozenset({"continuous", "integer", "discrete", "ordinal", "categorical"}),
    "halton": frozenset({"continuous", "integer", "discrete", "ordinal", "categorical"}),
    "d_optimal": frozenset({"continuous", "integer", "discrete"}),
    "random_uniform": frozenset({"continuous", "integer", "discrete", "ordinal", "categorical"}),
}

MAX_REQUESTED_ROWS = 4096
MAX_POOL_ROWS = 100_000
MAX_ENUMERATION_ROWS = 100_000
_MAX_ATTEMPTS = 48
_DEFAULT_LHS_RESTARTS = 10
_EXACT_BALANCE_COMBINATIONS = 100_000


@dataclass(frozen=True)
class DesignDiagnostics:
    requested_method: str
    effective_method: str
    selection_rule: str
    seed: int
    constraint_digest: str
    knowledge_sources: tuple[str, ...]
    factor_encodings: Mapping[str, str]
    level_counts: Mapping[str, Mapping[str, int]]
    minimum_model_distance: float | None
    rejection_counts: Mapping[str, int]
    candidate_keys: tuple[str, ...]


@dataclass(frozen=True, init=False)
class DesignBatch:
    """An immutable diagnostic envelope that exposes defensive frame copies."""

    _frame: pd.DataFrame
    diagnostics: DesignDiagnostics

    def __init__(self, frame: pd.DataFrame, diagnostics: DesignDiagnostics) -> None:
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)


def compatible_design_methods(space: Space) -> tuple[str, ...]:
    _validate_space(space)
    kinds = {parameter.kind for parameter in space.params}
    return tuple(method for method, supported in METHOD_KINDS.items() if kinds <= supported)


def resolve_design_method(space: Space, requested: str) -> tuple[str, str]:
    _validate_space(space)
    if type(requested) is not str:
        raise _config_invalid("Design method must be a string")
    if requested == "auto":
        if any(parameter.kind == "categorical" for parameter in space.params):
            return "sobol", "auto:nominal-categorical->sobol"
        return "lhs_maximin", "auto:no-nominal-categorical->lhs_maximin"
    compatible = compatible_design_methods(space)
    if requested not in compatible:
        raise EngineError(
            ErrorCode.CONFIG_INVALID,
            f"Design method {requested!r} is incompatible with this space",
            details={"requested_method": requested, "compatible_methods": list(compatible)},
        )
    return requested, "explicit"


def suggest_design(
    space: Space,
    n: int,
    method: str = "auto",
    *,
    seed: int = 42,
    parameter_constraints: Sequence[object] = (),
    knowledge_sources: Sequence[str] = (),
    existing: pd.DataFrame | None = None,
    pending: pd.DataFrame | None = None,
    return_diagnostics: bool = False,
    **legacy_options: object,
) -> pd.DataFrame | DesignBatch:
    """Return exactly ``n`` unique feasible physical conditions.

    ``n_iterations``, ``n_restarts``, ``max_resample``, and ``verbose`` remain
    accepted for legacy top-level callers.  Only ``n_restarts`` affects v0.5
    LHS maximin restarts; the other options are validated compatibility no-ops.
    """
    _validate_space(space)
    n = _validate_n(n)
    seed = _validate_seed(seed)
    if type(return_diagnostics) is not bool:
        raise _config_invalid("return_diagnostics must be a boolean")
    legacy = _validate_legacy_options(legacy_options)
    constraints = _validate_constraints(parameter_constraints)
    _validate_hard_constraints(constraints)
    sources = _validate_knowledge_sources(knowledge_sources)
    effective, rule = resolve_design_method(space, method)
    design_space = space.with_constraints(*constraints)

    # Existing/pending conditions are historical conditions: canonicalize their
    # declared parameter values, but do not impose newly supplied safety rules.
    avoided = _concat_condition_frames(existing, pending, space)
    avoided_keys = _candidate_keys_from_canonical(avoided, space)
    finite, finite_rejections = _enumerate_feasible(design_space)
    rejections: dict[str, int] = {"constraint": finite_rejections, "duplicate": 0, "avoided": 0}
    available: int | None = None
    if finite is not None:
        finite_eligible, finite_selection_rejections = _eligible_unique(
            finite, design_space, avoided_keys
        )
        _add_rejections(rejections, finite_selection_rejections)
        available = len(finite_eligible)
        if available < n:
            rejections["finite_cardinality"] = n - available
            raise _space_infeasible(n, available, effective, seed, rejections)
    else:
        finite_eligible = None

    if effective == "d_optimal":
        generated: Mapping[str, int] = {}
        if finite is None:
            universe, generated = _generate_feasible_pool(
                design_space, n=n, method=effective, seed=seed
            )
        else:
            universe = finite
        if finite is None:
            _add_rejections(rejections, generated)
        eligible, rejected = _eligible_unique(universe, design_space, avoided_keys)
        _add_rejections(rejections, rejected)
        selected = _select_d_optimal(eligible, design_space, n)
    elif effective in {"lhs_random", "lhs_maximin"}:
        selected, rejected = _generate_lhs_design(
            design_space,
            n=n,
            method=effective,
            seed=seed,
            avoided_keys=avoided_keys,
            restarts=legacy["n_restarts"],
        )
        _add_rejections(rejections, rejected)
    else:
        pool, generated = _generate_feasible_pool(design_space, n=n, method=effective, seed=seed)
        _add_rejections(rejections, generated)
        eligible, rejected = _eligible_unique(pool, design_space, avoided_keys)
        _add_rejections(rejections, rejected)
        selected = _select_balanced_maximin(eligible, design_space, n, random_ties=(effective == "random_uniform"))

    if len(selected) != n:
        raise _space_infeasible(n, available, effective, seed, rejections)
    selected = _canonical_frame(selected, design_space, "generated design")
    if not bool(design_space.feasibility_mask(selected).all().item()):
        raise _space_infeasible(n, available, effective, seed, {**rejections, "final_infeasible": 1})

    diagnostics = _build_diagnostics(
        design_space, selected, method, effective, rule, seed, sources, rejections
    )
    batch = DesignBatch(selected, diagnostics)
    return batch if return_diagnostics else batch.frame


def candidate_keys(frame: pd.DataFrame, space: Space) -> set[str]:
    """Return identities after declared-parameter canonicalization."""
    canonical = _canonical_frame(frame, space, "condition frame")
    return _candidate_keys_from_canonical(canonical, space)


def _generate_lhs_design(
    space: Space, *, n: int, method: str, seed: int, avoided_keys: set[str], restarts: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    candidates: list[tuple[float, int, pd.DataFrame]] = []
    rejections = {"constraint": 0, "duplicate": 0, "avoided": 0}
    attempts = max(1, restarts)
    for attempt in range(attempts):
        unit = latin_hypercube_sample(n, space.n_dims, seed=_attempt_seed(seed, attempt))
        decoded = _unit_to_physical_array(unit, space)
        feasible = space.feasibility_mask(decoded).cpu().numpy()
        rejections["constraint"] += int((~feasible).sum())
        if not bool(feasible.all()):
            decoded = _repair_design_physical(
                decoded, space, max_swaps=max(200, 25 * n), seed=_attempt_seed(seed, attempt)
            )
            decoded = _repair_lhs_strata(decoded, space)
            feasible = space.feasibility_mask(decoded).cpu().numpy()
            if not bool(feasible.all()):
                continue
        eligible, rejected = _eligible_unique(decoded, space, avoided_keys)
        _add_rejections(rejections, rejected)
        if len(eligible) != n:
            continue
        distance = _minimum_distance(space.physical_to_model(eligible).cpu().numpy())
        candidates.append((distance if distance is not None else math.inf, attempt, eligible))
        if method == "lhs_random":
            return eligible, rejections
    if not candidates:
        # Snapping can make a coupled discrete constraint incompatible with the
        # requested LHS strata.  Recover only after whole-design retries and
        # column-swap repair have failed; unconstrained LHS never takes this path.
        pool, generated = _generate_feasible_pool(
            space, n=n, method="random_uniform", seed=seed
        )
        _add_rejections(rejections, generated)
        eligible, rejected = _eligible_unique(pool, space, avoided_keys)
        _add_rejections(rejections, rejected)
        return _select_balanced_maximin(
            eligible, space, n, random_ties=(method == "lhs_random")
        ), rejections
    _, _, chosen = max(candidates, key=lambda item: (item[0], -item[1]))
    return chosen, rejections


def _repair_lhs_strata(design: pd.DataFrame, space: Space) -> pd.DataFrame:
    """Repair coupled constraints with only column-wise row swaps.

    Swapping retains each original LHS column sample (and therefore its
    requested strata before physical snapping) while deterministically choosing
    the largest feasibility-count improvement.
    """
    current = design.copy()
    for _ in range(max(1, len(current) * space.n_dims)):
        mask = space.feasibility_mask(current).cpu().numpy()
        current_count = int(mask.sum())
        if current_count == len(current):
            return current
        best: pd.DataFrame | None = None
        best_count = current_count
        for left in np.flatnonzero(~mask):
            for right in range(len(current)):
                if left == right:
                    continue
                for column in range(space.n_dims):
                    proposal = current.copy()
                    proposal.iat[left, column], proposal.iat[right, column] = (
                        proposal.iat[right, column], proposal.iat[left, column]
                    )
                    count = int(space.feasibility_mask(proposal).sum().item())
                    if count > best_count:
                        best, best_count = proposal, count
        if best is None:
            return current
        current = best
    return current


def _generate_feasible_pool(
    space: Space, *, n: int, method: str, seed: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    target = min(MAX_POOL_ROWS, max(64, n * 32))
    records: list[dict[str, object]] = []
    rejected = {"constraint": 0}
    for attempt in range(_MAX_ATTEMPTS):
        unit = _draw_unit(method, target, space.n_dims, _attempt_seed(seed, attempt))
        decoded = _unit_to_physical_array(unit, space)
        feasible = space.feasibility_mask(decoded).cpu().numpy()
        rejected["constraint"] += int((~feasible).sum())
        records.extend(decoded.loc[feasible].to_dict(orient="records"))
        if len(records) >= target:
            break
    return _physical_frame_from_records(records, space), rejected


def _draw_unit(method: str, n: int, d: int, seed: int) -> np.ndarray:
    if method == "sobol":
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n).astype(np.float64)
    if method == "halton":
        return qmc.Halton(d=d, scramble=True, seed=seed).random(n).astype(np.float64)
    return np.random.default_rng(seed).uniform(size=(n, d))


def _select_d_optimal(pool: pd.DataFrame, space: Space, n: int) -> pd.DataFrame:
    if len(pool) < n:
        return pool
    model = space.physical_to_model(pool).cpu().numpy()
    matrix = np.column_stack((np.ones(len(model)), model, model**2))
    information = np.eye(matrix.shape[1], dtype=np.float64) * 1e-12
    remaining = np.ones(len(matrix), dtype=bool)
    selected: list[int] = []
    for _ in range(n):
        best, score = -1, -math.inf
        for index in np.flatnonzero(remaining):
            sign, value = np.linalg.slogdet(information + np.outer(matrix[index], matrix[index]))
            if sign > 0 and value > score:
                best, score = int(index), float(value)
        if best < 0:
            break
        selected.append(best)
        information += np.outer(matrix[best], matrix[best])
        remaining[best] = False
    return pool.iloc[selected].reset_index(drop=True)


def _eligible_unique(
    pool: pd.DataFrame, space: Space, avoided_keys: set[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    canonical = _canonical_frame(pool, space, "candidate pool")
    records: list[dict[str, object]] = []
    seen = set(avoided_keys)
    rejected = {"duplicate": 0, "avoided": 0}
    for record in canonical.to_dict(orient="records"):
        key = _candidate_key(record, space)
        if key in avoided_keys:
            rejected["avoided"] += 1
        elif key in seen:
            rejected["duplicate"] += 1
        else:
            seen.add(key)
            records.append(record)
    return _physical_frame_from_records(records, space), rejected


def _select_balanced_maximin(
    pool: pd.DataFrame, space: Space, n: int, *, random_ties: bool
) -> pd.DataFrame:
    if len(pool) < n:
        return pool
    model = space.physical_to_model(pool).cpu().numpy()
    balanced = [p for p in space.params if p.kind in {"categorical", "ordinal"}]
    combinations = math.comb(len(pool), n) if n <= len(pool) else 0
    if balanced and combinations <= _EXACT_BALANCE_COMBINATIONS:
        best_indices: tuple[int, ...] | None = None
        best_score: tuple[float, float, float, tuple[int, ...]] | None = None
        for indices in itertools.combinations(range(len(pool)), n):
            score = _selection_score(indices, pool, model, balanced)
            if best_score is None or score < best_score:
                best_indices, best_score = indices, score
        assert best_indices is not None
        return pool.iloc[list(best_indices)].reset_index(drop=True)

    selected: list[int] = []
    remaining = np.ones(len(pool), dtype=bool)
    nearest = np.full(len(pool), math.inf)
    counts = _empty_level_counts(balanced)
    for _ in range(n):
        candidates = np.flatnonzero(remaining)
        best = min(
            candidates,
            key=lambda index: (
                _projected_balance_score(int(index), pool, balanced, counts, n),
                0.0 if random_ties else -nearest[index],
                int(index),
            ),
        )
        selected.append(int(best))
        remaining[best] = False
        _increment_counts(counts, pool.iloc[int(best)].to_dict(), balanced)
        distances = np.linalg.norm(model - model[best], axis=1)
        nearest = np.minimum(nearest, distances)
    return pool.iloc[selected].reset_index(drop=True)


def _selection_score(
    indices: tuple[int, ...], pool: pd.DataFrame, model: np.ndarray, balanced: list[object]
) -> tuple[float, float, float, tuple[int, ...]]:
    counts = _empty_level_counts(balanced)
    for index in indices:
        _increment_counts(counts, pool.iloc[index].to_dict(), balanced)
    maximum, total = _balance_deviation(counts, len(indices), balanced)
    distance = _minimum_distance(model[np.asarray(indices)])
    return maximum, total, -(distance if distance is not None else math.inf), indices


def _projected_balance_score(
    index: int, pool: pd.DataFrame, balanced: list[object], counts: dict[str, dict[str, int]], n: int
) -> tuple[float, float]:
    trial = {name: dict(levels) for name, levels in counts.items()}
    _increment_counts(trial, pool.iloc[index].to_dict(), balanced)
    return _balance_deviation(trial, n, balanced)


def _empty_level_counts(parameters: list[object]) -> dict[str, dict[str, int]]:
    return {
        parameter.name: {_scalar_token(level): 0 for level in parameter.values or ()}
        for parameter in parameters
    }


def _increment_counts(counts: dict[str, dict[str, int]], record: dict[str, object], parameters: list[object]) -> None:
    for parameter in parameters:
        token = _scalar_token(record[parameter.name])
        counts[parameter.name][token] += 1


def _balance_deviation(counts: dict[str, dict[str, int]], n: int, parameters: list[object]) -> tuple[float, float]:
    values = [
        abs(count - n / len(parameter.values or ()))
        for parameter in parameters
        for count in counts[parameter.name].values()
    ]
    return (max(values), sum(values)) if values else (0.0, 0.0)


def _enumerate_feasible(space: Space) -> tuple[pd.DataFrame | None, int]:
    cardinalities = [parameter.cardinality for parameter in space.params]
    if any(cardinality is None for cardinality in cardinalities):
        return None, 0
    total = math.prod(int(cardinality) for cardinality in cardinalities)
    if total > MAX_ENUMERATION_ROWS:
        _raise_limit("finite cardinality", total, MAX_ENUMERATION_ROWS)
    levels: list[Iterable[object]] = []
    for parameter in space.params:
        levels.append(parameter.numeric_levels if parameter.kind in {"integer", "discrete"} else (parameter.values or ()))
    records = [dict(zip(space.param_names, values, strict=True)) for values in itertools.product(*levels)]
    frame = _physical_frame_from_records(records, space)
    feasible = space.feasibility_mask(frame).cpu().numpy()
    return frame.loc[feasible].reset_index(drop=True), int((~feasible).sum())


def _canonical_frame(frame: object, space: Space, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise _config_invalid(f"{name} must be a pandas DataFrame")
    try:
        space._validate_required_columns(frame)
        model = space.physical_to_model(frame[space.param_names])
        return space.model_to_physical(model).reset_index(drop=True)
    except EngineError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise _config_invalid(f"{name} contains invalid physical values") from error


def _concat_condition_frames(existing: pd.DataFrame | None, pending: pd.DataFrame | None, space: Space) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for name, frame in (("existing", existing), ("pending", pending)):
        if frame is not None:
            records.extend(_canonical_frame(frame, space, name).to_dict(orient="records"))
    return _physical_frame_from_records(records, space)


def _build_diagnostics(
    space: Space, frame: pd.DataFrame, requested: str, effective: str, selection_rule: str,
    seed: int, knowledge_sources: tuple[str, ...], rejections: dict[str, int],
) -> DesignDiagnostics:
    model = space.physical_to_model(frame).cpu().numpy()
    levels = _empty_level_counts([p for p in space.params if p.kind in {"categorical", "ordinal"}])
    _ = [_increment_counts(levels, row, [p for p in space.params if p.kind in {"categorical", "ordinal"}]) for row in frame.to_dict("records")]
    return DesignDiagnostics(
        requested, effective, selection_rule, seed, _constraint_digest(space), knowledge_sources,
        _freeze_mapping({p.name: _factor_encoding(p) for p in space.params}),
        _freeze_mapping(levels), _minimum_distance(model), _freeze_mapping(dict(sorted(rejections.items()))),
        tuple(_candidate_key(row, space) for row in frame.to_dict("records")),
    )


def _minimum_distance(model: np.ndarray) -> float | None:
    return None if len(model) < 2 else float(pdist(model).min())


def _constraint_digest(space: Space) -> str:
    payload: list[object] = []
    for constraint in space.constraints:
        payload.append(constraint.to_dict() if hasattr(constraint, "to_dict") else {
            "name": getattr(constraint, "name", ""), "coeffs": dict(getattr(constraint, "coeffs", {})),
            "lower": getattr(constraint, "lower", None), "upper": getattr(constraint, "upper", None),
        })
    return hashlib.sha256(json.dumps(_digest_json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _candidate_keys_from_canonical(frame: pd.DataFrame, space: Space) -> set[str]:
    return {_candidate_key(record, space) for record in frame.to_dict(orient="records")}


def _candidate_key(record: dict[str, object], space: Space) -> str:
    return json.dumps([(name, _tagged_scalar(record[name])) for name in space.param_names], ensure_ascii=False, separators=(",", ":"))


def _tagged_scalar(value: object) -> list[object]:
    if value is None:
        return ["null", None]
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, (str, np.str_)):
        return ["str", str(value)]
    if isinstance(value, Integral):
        return ["int", int(value)]
    if isinstance(value, (float, np.floating)):
        return ["float", float(value)]
    raise _config_invalid("candidate values must be JSON scalars")


def _scalar_token(value: object) -> str:
    return json.dumps(_tagged_scalar(value), ensure_ascii=False, separators=(",", ":"))


def _factor_encoding(parameter: object) -> str:
    if parameter.kind == "categorical":
        return "categorical:index"
    if parameter.kind == "ordinal":
        return "ordinal:index"
    return f"numeric:{parameter.transform}"


def _digest_json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _digest_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_digest_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "NaN" if math.isnan(float(value)) else ("Infinity" if value > 0 else "-Infinity")
    return value.item() if isinstance(value, np.generic) else value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze_mapping(item) if isinstance(item, Mapping) else item for key, item in value.items()})


def _validate_space(space: object) -> None:
    if not isinstance(space, Space):
        raise _config_invalid("space must be a Space")


def _validate_n(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise _config_invalid("n must be a positive integer")
    if int(value) > MAX_REQUESTED_ROWS:
        _raise_limit("n", int(value), MAX_REQUESTED_ROWS)
    return int(value)


def _validate_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise _config_invalid("seed must be a non-negative integer")
    return int(value)


def _validate_constraints(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _config_invalid("parameter_constraints must be an ordered sequence")
    return tuple(value)


def _validate_hard_constraints(constraints: tuple[object, ...]) -> None:
    if any(not getattr(constraint, "hard", True) for constraint in constraints):
        raise _config_invalid("parameter_constraints used for initial safety must be hard")


def _validate_knowledge_sources(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or any(type(item) is not str for item in value):
        raise _config_invalid("knowledge_sources must be an ordered sequence of strings")
    return tuple(value)


def _validate_legacy_options(options: Mapping[str, object]) -> dict[str, int]:
    permitted = {"n_iterations", "n_restarts", "max_resample", "verbose"}
    unknown = sorted(set(options) - permitted)
    if unknown:
        raise TypeError(f"unexpected legacy option(s): {', '.join(unknown)}")
    for name in ("n_iterations", "n_restarts", "max_resample"):
        if name in options and (isinstance(options[name], bool) or not isinstance(options[name], Integral) or int(options[name]) < 1):
            raise _config_invalid(f"{name} must be a positive integer")
    if "verbose" in options and type(options["verbose"]) is not bool:
        raise _config_invalid("verbose must be a boolean")
    return {"n_restarts": int(options.get("n_restarts", _DEFAULT_LHS_RESTARTS))}


def _attempt_seed(seed: int, attempt: int) -> int:
    return int(np.random.SeedSequence([seed, attempt]).generate_state(1)[0])


def _add_rejections(target: dict[str, int], added: Mapping[str, int]) -> None:
    for name, value in added.items():
        target[name] = target.get(name, 0) + int(value)


def _space_infeasible(requested: int, available: int | None, method: str, seed: int, rejections: Mapping[str, int]) -> EngineError:
    return EngineError(ErrorCode.SPACE_INFEASIBLE, "Unable to generate the requested number of unique feasible design points", details={"requested": requested, "available_cardinality": available, "rejections": dict(sorted(rejections.items())), "method": method, "seed": seed})


def _raise_limit(field: str, requested: int, limit: int) -> None:
    raise EngineError(ErrorCode.CONFIG_INVALID, f"{field} exceeds supported design limit", details={"field": field, "requested": requested, "limit": limit}, hints=("Reduce the requested design size or split the campaign.",))


def _config_invalid(message: str) -> EngineError:
    return EngineError(ErrorCode.CONFIG_INVALID, message)


__all__ = ["METHOD_KINDS", "MAX_ENUMERATION_ROWS", "MAX_POOL_ROWS", "MAX_REQUESTED_ROWS", "DesignBatch", "DesignDiagnostics", "candidate_keys", "compatible_design_methods", "resolve_design_method", "suggest_design"]
