"""Capability-aware, deterministic initial design generation."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from ..domain import Space
from ..errors import EngineError, ErrorCode
from .constrained import _physical_frame_from_records, _unit_to_physical_array
from .lhs import latin_hypercube_sample


METHOD_KINDS = {
    "lhs_maximin": frozenset({"continuous", "integer", "discrete", "ordinal"}),
    "lhs_random": frozenset({"continuous", "integer", "discrete", "ordinal"}),
    "sobol": frozenset(
        {"continuous", "integer", "discrete", "ordinal", "categorical"}
    ),
    "halton": frozenset(
        {"continuous", "integer", "discrete", "ordinal", "categorical"}
    ),
    "d_optimal": frozenset({"continuous", "integer", "discrete"}),
    "random_uniform": frozenset(
        {"continuous", "integer", "discrete", "ordinal", "categorical"}
    ),
}

_FINITE_ENUMERATION_LIMIT = 100_000
_MAX_ATTEMPTS = 48


@dataclass(frozen=True)
class DesignDiagnostics:
    requested_method: str
    effective_method: str
    selection_rule: str
    seed: int
    constraint_digest: str
    knowledge_sources: tuple[str, ...]
    factor_encodings: dict[str, str]
    level_counts: dict[str, dict[str, int]]
    minimum_model_distance: float | None
    rejection_counts: dict[str, int]
    candidate_keys: tuple[str, ...]


@dataclass(frozen=True)
class DesignBatch:
    frame: pd.DataFrame
    diagnostics: DesignDiagnostics


def compatible_design_methods(space: Space) -> tuple[str, ...]:
    """Return methods which explicitly support every factor in ``space``."""
    _validate_space(space)
    kinds = {parameter.kind for parameter in space.params}
    return tuple(
        method for method, supported in METHOD_KINDS.items() if kinds <= supported
    )


def resolve_design_method(space: Space, requested: str) -> tuple[str, str]:
    """Resolve the deterministic automatic policy or validate an explicit choice."""
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
            details={
                "requested_method": requested,
                "compatible_methods": list(compatible),
            },
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
) -> pd.DataFrame | DesignBatch:
    """Return exactly ``n`` unique feasible physical conditions.

    The legacy return remains a DataFrame.  Request diagnostics to receive the
    same frame wrapped in a :class:`DesignBatch`.
    """
    _validate_space(space)
    _validate_n(n)
    _validate_seed(seed)
    if type(return_diagnostics) is not bool:
        raise _config_invalid("return_diagnostics must be a boolean")
    constraints = _validate_constraints(parameter_constraints)
    sources = _validate_knowledge_sources(knowledge_sources)
    effective, rule = resolve_design_method(space, method)
    design_space = space.with_constraints(*constraints)
    avoided = _concat_condition_frames(existing, pending, design_space)
    avoided_keys = candidate_keys(avoided, design_space)

    finite_frame = _enumerate_feasible(design_space)
    available: int | None = None
    rejections: dict[str, int] = {
        "constraint": 0,
        "duplicate": 0,
        "avoided": 0,
    }
    if finite_frame is not None:
        finite_keys = candidate_keys(finite_frame, design_space)
        available = len(finite_keys - avoided_keys)
        if available < n:
            rejections["finite_cardinality"] = n - available
            raise _space_infeasible(n, available, effective, seed, rejections)
        pool = finite_frame
    else:
        pool, generated_rejections = _generate_feasible_pool(
            design_space, n=n, method=effective, seed=seed
        )
        _add_rejections(rejections, generated_rejections)

    if effective == "d_optimal":
        pool = _select_d_optimal_pool(pool, design_space, n=n)

    selected, selection_rejections = _select_unique_balanced_maximin(
        pool, design_space, n=n, avoided_keys=avoided_keys
    )
    _add_rejections(rejections, selection_rejections)
    if len(selected) != n:
        raise _space_infeasible(n, available, effective, seed, rejections)

    selected = _physical_frame_from_records(
        selected.to_dict(orient="records"), design_space
    ).reset_index(drop=True)
    if not bool(design_space.feasibility_mask(selected).all().item()):
        raise _space_infeasible(
            n, available, effective, seed, {**rejections, "final_infeasible": 1}
        )

    diagnostics = _build_diagnostics(
        design_space,
        selected,
        requested=method,
        effective=effective,
        selection_rule=rule,
        seed=seed,
        knowledge_sources=sources,
        rejections=rejections,
    )
    batch = DesignBatch(selected, diagnostics)
    return batch if return_diagnostics else batch.frame


def candidate_keys(frame: pd.DataFrame, space: Space) -> set[str]:
    """Return deterministic physical-row identities without scalar coercion."""
    _validate_frame(frame, space, "condition frame", require_feasible=False)
    return {
        _candidate_key(record, space)
        for record in frame[space.param_names].to_dict(orient="records")
    }


def _validate_space(space: object) -> None:
    if not isinstance(space, Space):
        raise _config_invalid("space must be a Space")


def _validate_n(n: object) -> None:
    if isinstance(n, bool) or not isinstance(n, Integral) or int(n) < 1:
        raise _config_invalid("n must be a positive integer")


def _validate_seed(seed: object) -> None:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise _config_invalid("seed must be a non-negative integer")


def _validate_constraints(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _config_invalid("parameter_constraints must be an ordered sequence")
    return tuple(value)


def _validate_knowledge_sources(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _config_invalid("knowledge_sources must be an ordered sequence of strings")
    if any(type(item) is not str for item in value):
        raise _config_invalid("knowledge_sources must be an ordered sequence of strings")
    return tuple(value)


def _validate_frame(
    frame: object, space: Space, name: str, *, require_feasible: bool
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise _config_invalid(f"{name} must be a pandas DataFrame")
    try:
        space._validate_required_columns(frame)
    except EngineError:
        raise
    except (TypeError, ValueError) as error:
        raise _config_invalid(f"{name} is invalid: {error}") from error
    checked = _physical_frame_from_records(
        frame[space.param_names].to_dict(orient="records"), space
    )
    try:
        mask = space.feasibility_mask(checked)
    except (TypeError, ValueError, OverflowError) as error:
        raise _config_invalid(f"{name} contains invalid physical values") from error
    if require_feasible and not bool(mask.all().item()):
        raise _config_invalid(f"{name} contains infeasible physical values")
    return checked


def _concat_condition_frames(
    existing: pd.DataFrame | None, pending: pd.DataFrame | None, space: Space
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, frame in (("existing", existing), ("pending", pending)):
        if frame is not None:
            frames.append(_validate_frame(frame, space, name, require_feasible=True))
    if not frames:
        return _physical_frame_from_records([], space)
    return _physical_frame_from_records(
        [
            record
            for frame in frames
            for record in frame[space.param_names].to_dict(orient="records")
        ],
        space,
    )


def _enumerate_feasible(space: Space) -> pd.DataFrame | None:
    cardinalities = [parameter.cardinality for parameter in space.params]
    if any(cardinality is None for cardinality in cardinalities):
        return None
    total = math.prod(int(cardinality) for cardinality in cardinalities)
    if total > _FINITE_ENUMERATION_LIMIT:
        return None
    levels: list[Iterable[object]] = []
    for parameter in space.params:
        if parameter.kind in {"integer", "discrete"}:
            levels.append(parameter.numeric_levels)
        else:
            levels.append(parameter.values or ())
    records = [
        dict(zip(space.param_names, values, strict=True))
        for values in itertools.product(*levels)
    ]
    frame = _physical_frame_from_records(records, space)
    return frame.loc[space.feasibility_mask(frame).cpu().numpy()].reset_index(drop=True)


def _generate_feasible_pool(
    space: Space, *, n: int, method: str, seed: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    records: list[dict[str, object]] = []
    rejected = {"constraint": 0}
    target = max(64, n * 32)
    for attempt in range(_MAX_ATTEMPTS):
        attempt_seed = _attempt_seed(seed, attempt)
        unit = _draw_unit(method, target, space.n_dims, attempt_seed)
        decoded = _unit_to_physical_array(unit, space)
        mask = space.feasibility_mask(decoded).cpu().numpy()
        rejected["constraint"] += int((~mask).sum())
        records.extend(decoded.loc[mask].to_dict(orient="records"))
        if len(records) >= target:
            break
    return _physical_frame_from_records(records, space), rejected


def _attempt_seed(seed: int, attempt: int) -> int:
    return int(np.random.SeedSequence([seed, attempt]).generate_state(1)[0])


def _draw_unit(method: str, n: int, d: int, seed: int) -> np.ndarray:
    if method in {"lhs_maximin", "lhs_random"}:
        return latin_hypercube_sample(n, d, seed=seed)
    if method == "sobol":
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        return sampler.random(n).astype(np.float64)
    if method == "halton":
        return qmc.Halton(d=d, scramble=True, seed=seed).random(n).astype(np.float64)
    if method in {"random_uniform", "d_optimal"}:
        return np.random.default_rng(seed).uniform(size=(n, d))
    raise AssertionError(f"unresolved method {method!r}")


def _select_d_optimal_pool(pool: pd.DataFrame, space: Space, *, n: int) -> pd.DataFrame:
    if len(pool) < n:
        return pool
    model = space.physical_to_model(pool).cpu().numpy()
    matrix = np.column_stack((np.ones(len(model)), model, model**2))
    chosen = _greedy_log_determinant(matrix, n)
    return pool.iloc[chosen].reset_index(drop=True)


def _greedy_log_determinant(matrix: np.ndarray, n: int) -> list[int]:
    information = np.eye(matrix.shape[1], dtype=np.float64) * 1e-12
    chosen: list[int] = []
    remaining = np.ones(len(matrix), dtype=bool)
    for _ in range(n):
        best_index = -1
        best_score = -math.inf
        for index in np.flatnonzero(remaining):
            candidate = information + np.outer(matrix[index], matrix[index])
            sign, score = np.linalg.slogdet(candidate)
            if sign > 0 and score > best_score:
                best_index = int(index)
                best_score = float(score)
        if best_index < 0:
            break
        chosen.append(best_index)
        information += np.outer(matrix[best_index], matrix[best_index])
        remaining[best_index] = False
    return chosen


def _select_unique_balanced_maximin(
    pool: pd.DataFrame,
    space: Space,
    *,
    n: int,
    avoided_keys: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    unique: list[dict[str, object]] = []
    seen = set(avoided_keys)
    rejections = {"duplicate": 0, "avoided": 0}
    for record in pool[space.param_names].to_dict(orient="records"):
        key = _candidate_key(record, space)
        if key in avoided_keys:
            rejections["avoided"] += 1
        elif key in seen:
            rejections["duplicate"] += 1
        else:
            seen.add(key)
            unique.append(record)
    if not unique:
        return _physical_frame_from_records([], space), rejections

    unique_frame = _physical_frame_from_records(unique, space)
    model = space.physical_to_model(unique_frame).cpu().numpy()
    balanced = [
        parameter
        for parameter in space.params
        if parameter.kind in {"categorical", "ordinal"}
    ]
    counts = {
        parameter.name: {_scalar_token(level): 0 for level in parameter.values or ()}
        for parameter in balanced
    }
    selected: list[int] = []
    remaining = set(range(len(unique)))
    while remaining and len(selected) < n:
        scored: list[tuple[float, float, int]] = []
        for index in sorted(remaining):
            record = unique[index]
            deficit = sum(
                n / len(parameter.values or ())
                - counts[parameter.name][_scalar_token(record[parameter.name])]
                for parameter in balanced
            )
            if selected:
                distance = float(
                    np.linalg.norm(model[index] - model[np.asarray(selected)], axis=1).min()
                )
            else:
                distance = math.inf
            scored.append((deficit, distance, index))
        _, _, best = max(scored, key=lambda item: (item[0], item[1], -item[2]))
        selected.append(best)
        remaining.remove(best)
        for parameter in balanced:
            counts[parameter.name][_scalar_token(unique[best][parameter.name])] += 1
    return unique_frame.iloc[selected].reset_index(drop=True), rejections


def _build_diagnostics(
    space: Space,
    frame: pd.DataFrame,
    *,
    requested: str,
    effective: str,
    selection_rule: str,
    seed: int,
    knowledge_sources: tuple[str, ...],
    rejections: dict[str, int],
) -> DesignDiagnostics:
    model = space.physical_to_model(frame).cpu().numpy()
    minimum = None if len(model) < 2 else float(pdist(model).min())
    level_counts: dict[str, dict[str, int]] = {}
    for parameter in space.params:
        if parameter.kind in {"categorical", "ordinal"}:
            values = frame[parameter.name].tolist()
            level_counts[parameter.name] = {
                _scalar_token(level): sum(
                    _scalar_token(value) == _scalar_token(level) for value in values
                )
                for level in parameter.values or ()
            }
    return DesignDiagnostics(
        requested_method=requested,
        effective_method=effective,
        selection_rule=selection_rule,
        seed=seed,
        constraint_digest=_constraint_digest(space),
        knowledge_sources=knowledge_sources,
        factor_encodings={
            parameter.name: _factor_encoding(parameter) for parameter in space.params
        },
        level_counts=level_counts,
        minimum_model_distance=minimum,
        rejection_counts=dict(sorted(rejections.items())),
        candidate_keys=tuple(_candidate_key(row, space) for row in frame.to_dict("records")),
    )


def _factor_encoding(parameter: object) -> str:
    kind = parameter.kind  # type: ignore[attr-defined]
    if kind == "categorical":
        return "categorical:index"
    if kind == "ordinal":
        return "ordinal:index"
    return f"numeric:{parameter.transform}"  # type: ignore[attr-defined]


def _constraint_digest(space: Space) -> str:
    payload: list[object] = []
    for constraint in space.constraints:
        if hasattr(constraint, "to_dict"):
            payload.append(constraint.to_dict())
        else:
            payload.append(
                {
                    "name": getattr(constraint, "name", ""),
                    "coeffs": dict(getattr(constraint, "coeffs", {})),
                    "lower": getattr(constraint, "lower", None),
                    "upper": getattr(constraint, "upper", None),
                }
            )
    canonical = json.dumps(
        _digest_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _digest_json_value(value: object) -> object:
    """Represent legacy open bounds canonically before hashing diagnostics."""
    if isinstance(value, dict):
        return {str(key): _digest_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_digest_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        if math.isnan(float(value)):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, np.generic):
        return value.item()
    return value


def _candidate_key(record: dict[str, object], space: Space) -> str:
    return json.dumps(
        [(name, _tagged_scalar(record[name])) for name in space.param_names],
        ensure_ascii=False,
        separators=(",", ":"),
    )


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


def _space_infeasible(
    requested: int,
    available: int | None,
    method: str,
    seed: int,
    rejections: dict[str, int],
) -> EngineError:
    return EngineError(
        ErrorCode.SPACE_INFEASIBLE,
        "Unable to generate the requested number of unique feasible design points",
        details={
            "requested": requested,
            "available_cardinality": available,
            "rejections": dict(sorted(rejections.items())),
            "method": method,
            "seed": seed,
        },
    )


def _add_rejections(target: dict[str, int], added: dict[str, int]) -> None:
    for name, value in added.items():
        target[name] = target.get(name, 0) + int(value)


def _config_invalid(message: str) -> EngineError:
    return EngineError(ErrorCode.CONFIG_INVALID, message)


__all__ = [
    "METHOD_KINDS",
    "DesignBatch",
    "DesignDiagnostics",
    "candidate_keys",
    "compatible_design_methods",
    "resolve_design_method",
    "suggest_design",
]
