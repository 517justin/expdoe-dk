"""Fixed observed utility geometry, independent of fitted physical models."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from expdoe_dk.domain import ObservationBatch, Space
from expdoe_dk.errors import EngineError, ErrorCode
from .knowledge_runtime import RuntimeOutcomeRange, validate_runtime_artifact
from .multiobjective import MultiObjectiveConfig, _finite_number, resolve_preference_weights


def _invalid(message):
    return EngineError(ErrorCode.CONFIG_INVALID, message)


def _sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise _invalid(f"{name} must be an array")
    return tuple(value)


@dataclass(frozen=True, kw_only=True)
class ObjectiveGeometry:
    """Immutable initial geometry, with vectors in ``objective_names`` order.

    Location and scale use native Objective utility. Both references and the
    acquisition ideal use (utility - location) / scale. Only acquisition values
    include target-range shortfalls; neither includes scalarization weights.
    Auto references have no physical inverse. ``physical_reference`` is present
    only for a user-supplied named physical reference.
    """

    objective_names: tuple[str, ...]
    location: tuple[float, ...]
    scale: tuple[float, ...]
    report_reference: tuple[float, ...]
    acquisition_reference: tuple[float, ...]
    acquisition_ideal: tuple[float, ...]
    source_digest: str
    source_row_count: int
    preference_weights: tuple[float, ...] | None
    preference_provenance: tuple[Mapping[str, str], ...]
    physical_reference: Mapping[str, float] | None
    reference_method: str

    def __post_init__(self):
        names = _sequence(self.objective_names, "objective_names")
        if (len(names) < 2 or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names)):
            raise _invalid("objective_names must contain at least two unique nonempty names")
        object.__setattr__(self, "objective_names", names)
        for name in ("location", "scale", "report_reference", "acquisition_reference",
                     "acquisition_ideal", "preference_weights"):
            raw = getattr(self, name)
            if name == "preference_weights" and raw is None:
                continue
            values = tuple(_finite_number(v, name) for v in _sequence(raw, name))
            if len(values) != len(names):
                raise _invalid(f"{name} must match objective_names length")
            if name in {"scale", "preference_weights"} and any(v <= 0 for v in values):
                raise _invalid(f"{name} must be positive")
            if name == "preference_weights" and (any(v > 1 for v in values)
                    or not math.isclose(math.fsum(values), 1., rel_tol=1e-9)):
                raise _invalid("preference_weights must be normalized")
            object.__setattr__(self, name, values)
        if (not isinstance(self.source_digest, str) or len(self.source_digest) != 64
                or any(c not in "0123456789abcdef" for c in self.source_digest)):
            raise _invalid("source_digest must be a lowercase SHA256 digest")
        if type(self.source_row_count) is not int or self.source_row_count < 3:
            raise _invalid("source_row_count must be an integer of at least three")
        provenance = []
        for item in _sequence(self.preference_provenance, "preference_provenance"):
            if not isinstance(item, Mapping):
                raise _invalid("Preference provenance must be an object")
            source = item.get("source")
            if source not in ("config", "knowledge", "objective_priority"):
                raise _invalid("Unknown preference source")
            expected = ({"source", "source_pattern_id", "source_pattern", "source_version"}
                        if source == "knowledge" else {"source"})
            if set(item) != expected or any(not isinstance(v, str) or not v for v in item.values()):
                raise _invalid("Invalid preference provenance fields")
            provenance.append(MappingProxyType(dict(item)))
        if bool(provenance) != (self.preference_weights is not None):
            raise _invalid("Preference weights and provenance must be present together")
        object.__setattr__(self, "preference_provenance", tuple(provenance))
        if self.reference_method not in ("physical", "observed_feasible_min_minus_margin"):
            raise _invalid("Unknown reference_method")
        if (self.physical_reference is not None) != (self.reference_method == "physical"):
            raise _invalid("Only explicit physical references have physical_reference values")
        if self.physical_reference is not None:
            if not isinstance(self.physical_reference, Mapping) or set(self.physical_reference) != set(names):
                raise _invalid("physical_reference must exactly match objective_names")
            object.__setattr__(self, "physical_reference", MappingProxyType({
                name: _finite_number(self.physical_reference[name], "physical_reference") for name in names
            }))

    def to_dict(self) -> dict:
        """Return detached finite JSON values, with no tensors or physical guess."""
        return {
            "objective_names": list(self.objective_names), "location": list(self.location),
            "scale": list(self.scale), "report_reference": list(self.report_reference),
            "acquisition_reference": list(self.acquisition_reference),
            "acquisition_ideal": list(self.acquisition_ideal), "source_digest": self.source_digest,
            "source_row_count": self.source_row_count,
            "preference_weights": None if self.preference_weights is None else list(self.preference_weights),
            "preference_provenance": [dict(p) for p in self.preference_provenance],
            "physical_reference": None if self.physical_reference is None else dict(self.physical_reference),
            "reference_method": self.reference_method,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> ObjectiveGeometry:
        """Reject unknown, missing, malformed and nonfinite checkpoint fields."""
        if (not isinstance(payload, Mapping) or any(not isinstance(k, str) for k in payload)
                or set(payload) != {f.name for f in fields(cls)}):
            raise _invalid("ObjectiveGeometry fields must match exactly")
        for name in ("objective_names", "location", "scale", "report_reference", "acquisition_reference",
                     "acquisition_ideal", "preference_provenance", "preference_weights"):
            if name == "preference_weights" and payload[name] is None:
                continue
            if type(payload[name]) is not list:
                raise _invalid(f"{name} must be a JSON array")
        return cls(**payload)


def _target_ranges(space, runtime):
    ranges = {}
    for interval in runtime.get("outcome_constraints", ()):
        if not isinstance(interval, RuntimeOutcomeRange):
            raise _invalid("Expected compiled RuntimeOutcomeRange")
        validate_runtime_artifact(space, "outcome_constraints", interval.artifact)
        name = interval.payload["objective"]
        bounds = (interval.payload["lower"], interval.payload["upper"])
        if name in ranges and ranges[name] != bounds:
            raise _invalid(f"Conflicting target ranges for {name}")
        ranges[name] = bounds
    return ranges


def _utilities(samples, space, ranges):
    if (not isinstance(samples, Tensor) or not samples.is_floating_point()
            or samples.ndim < 1 or samples.shape[-1] != len(space.objectives)
            or not torch.isfinite(samples).all()):
        raise _invalid("Physical samples must be finite floating tensors with last axis matching objectives")
    columns = []
    for i, objective in enumerate(space.objective_specs):
        y = samples[..., i]
        if objective.direction == "maximize":
            utility = y
        elif objective.direction == "minimize":
            utility = -y
        elif isinstance(objective.target, tuple):
            lo, hi = objective.target
            utility = -torch.maximum((lo - y).clamp_min(0), (y - hi).clamp_min(0))
        else:
            utility = -(y - objective.target).abs()
        if objective.name in ranges:
            lo, hi = ranges[objective.name]
            utility = utility - torch.maximum((lo - y).clamp_min(0), (y - hi).clamp_min(0))
        columns.append(utility)
    result = torch.stack(columns, dim=-1)
    if not torch.isfinite(result).all():
        raise _invalid("Objective utilities overflowed")
    return result


def objective_samples(samples: Tensor, space: Space, geometry: ObjectiveGeometry, runtime: Mapping,
                      *, with_preferences: bool) -> Tensor:
    """Transform physical draws without fitting, clipping, inversion or reduction.

    ``with_preferences`` adds only scoped target-range shortfalls. Explicit
    weights are consumed by scalarization, not multiplied into these outputs.
    Every leading sample/batch/q dimension, dtype, device and gradient is kept.
    """
    if tuple(space.objectives) != geometry.objective_names:
        raise _invalid("Space objective order must match ObjectiveGeometry")
    if type(with_preferences) is not bool:
        raise _invalid("with_preferences must be boolean")
    utility = _utilities(samples, space, _target_ranges(space, runtime) if with_preferences else {})
    result = (utility - samples.new_tensor(geometry.location)) / samples.new_tensor(geometry.scale)
    if not torch.isfinite(result).all():
        raise _invalid("Normalized objective utilities overflowed")
    return result


def _snapshot_value(value):
    """Encode missing/nonfinite failed outcomes explicitly in a strict JSON digest."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA:
        return {"missing": "pd.NA"}
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise _invalid("Observation digest requires scalar physical values")


def _source_digest(space, observations):
    def frame_snapshot(frame, names):
        if frame is None:
            return None
        return {name: [_snapshot_value(v) for v in frame[name].tolist()] for name in sorted(names)}

    semantics = space.to_dict()
    # Runtime/version metadata is not part of physical experiment semantics.
    semantics = {key: sorted(semantics[key], key=lambda item: item["name"])
                 for key in ("params", "objectives", "constraints", "outcome_constraints")}
    status = observations.status
    snapshot = {
        "space": semantics,
        "X": frame_snapshot(observations.X, space.param_names),
        "Y": frame_snapshot(observations.Y, space.objectives),
        "Yvar": frame_snapshot(observations.Yvar, space.objectives),
        "status": ["success"] * len(observations.X) if status is None else status.tolist(),
        "ids": list(observations.ids),
    }
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _feasible_baseline(space, observations, runtime):
    mask = space.feasibility_mask(observations.X)
    for constraint in space.outcome_constraints:
        if not constraint.hard or constraint.operator not in ("<=", ">="):
            raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
                              "Outcome safety requires hard named <= or >= bounds")
    for name in space.objectives:
        bounds = [c for c in space.outcome_constraints if c.objective == name]
        lo = max((c.bound for c in bounds if c.operator == ">="), default=-math.inf)
        hi = min((c.bound for c in bounds if c.operator == "<="), default=math.inf)
        if lo > hi:
            raise _invalid(f"Empty outcome safety interval for {name}")
    x_rows, y_rows = observations.X.to_dict("records"), observations.Y.to_dict("records")
    for i, (x, y) in enumerate(zip(x_rows, y_rows, strict=True)):
        mask[i] = bool(mask[i]) and all(c.satisfied(x) for c in runtime.get("parameter_constraints", ())
                                      if c.hard) and all(c.satisfied(y) for c in space.outcome_constraints)
    if not mask.any():
        raise EngineError(ErrorCode.INSUFFICIENT_DATA, "No measured feasible baseline", details={
            "reason": "no_feasible_baseline",
        })
    return mask


def resolve_geometry(space: Space, observations: ObservationBatch, runtime: Mapping,
                     config: MultiObjectiveConfig) -> ObjectiveGeometry:
    """Resolve once from the initial observed batch; later draws reuse the result."""
    config.validate_for_space(space)
    if not isinstance(observations, ObservationBatch):
        raise _invalid("Geometry requires an ObservationBatch")
    for frame, names in ((observations.X, space.param_names), (observations.Y, space.objectives)):
        if not frame.columns.is_unique or set(frame.columns) != set(names):
            raise _invalid("Observation columns must exactly match Space names")
    successful = observations.successful()
    if len(successful.X) < 3:
        raise EngineError(ErrorCode.INSUFFICIENT_DATA, "At least three successful observations are required")
    weights, provenance = resolve_preference_weights(space, runtime, config)
    feasible = _feasible_baseline(space, successful, runtime)
    ranges = _target_ranges(space, runtime)
    samples = torch.tensor(successful.Y[space.objectives].to_numpy(dtype=float).copy(), dtype=torch.double)
    report = _utilities(samples, space, {})
    acquisition = _utilities(samples, space, ranges)
    location = report.amin(dim=0)
    spread = report.amax(dim=0) - location
    scale = torch.where(spread > 0, spread, location.abs().clamp_min(1.))
    if not torch.isfinite(scale).all():
        raise _invalid("Initial utility scale overflowed")
    if config.reference_point is None:
        report_reference = (report[feasible].amin(dim=0) - location) / scale - .1
        acquisition_reference = (acquisition[feasible].amin(dim=0) - location) / scale - .1
        method = "observed_feasible_min_minus_margin"
    else:
        physical = torch.tensor([config.reference_point[n] for n in space.objectives], dtype=torch.double)
        raw_reference = _utilities(physical, space, {})
        if not (report[feasible] > raw_reference).all(dim=-1).any():
            raise _invalid("A feasible measured row must strictly dominate the physical reference in every objective")
        report_reference = (raw_reference - location) / scale
        acquisition_reference = (_utilities(physical, space, ranges) - location) / scale
        method = "physical"
    ideal = ((acquisition - location) / scale).amax(dim=0)
    return ObjectiveGeometry(
        objective_names=tuple(space.objectives), location=tuple(location.tolist()), scale=tuple(scale.tolist()),
        report_reference=tuple(report_reference.tolist()), acquisition_reference=tuple(acquisition_reference.tolist()),
        acquisition_ideal=tuple(ideal.tolist()), source_digest=_source_digest(space, observations),
        source_row_count=len(observations.X), preference_weights=weights, preference_provenance=provenance,
        physical_reference=config.reference_point, reference_method=method,
    )


__all__ = ["ObjectiveGeometry", "resolve_geometry", "objective_samples"]
