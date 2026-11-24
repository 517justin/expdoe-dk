"""Validated, immutable configuration for multiobjective recommendation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING

from expdoe_dk.errors import EngineError, ErrorCode

if TYPE_CHECKING:
    from expdoe_dk.domain.space import Space


def _finite_number(value: object, name: str) -> float:
    message = f"{name} must be a finite non-boolean real number"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EngineError(ErrorCode.CONFIG_INVALID, message)
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise EngineError(ErrorCode.CONFIG_INVALID, message) from error
    if not math.isfinite(number):
        raise EngineError(ErrorCode.CONFIG_INVALID, message)
    return number


@dataclass(frozen=True, kw_only=True)
class MultiObjectiveConfig:
    """Physical reference values and preferences for at least two objectives.

    Call ``validate_for_space`` before using the configuration with a Space.
    Serialization requires all five fields; constructor defaults are not used
    to silently repair incomplete persisted configurations.
    """

    acquisition: str = "auto"
    noise_mode: str = "auto"
    reference_point: Mapping[str, float] | None = None
    preference_weights: Mapping[str, float] | None = None
    min_joint_feasibility: float = 0.95

    def __post_init__(self) -> None:
        for name, choices in (
            ("acquisition", ("auto", "q_log_ehvi", "q_log_nehvi", "q_log_nparego")),
            ("noise_mode", ("auto", "inferred", "known", "noiseless")),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in choices:
                raise EngineError(ErrorCode.CONFIG_INVALID, f"Invalid {name}: {value!r}")
        for name in ("reference_point", "preference_weights"):
            source = getattr(self, name)
            if source is None:
                continue
            if not isinstance(source, Mapping):
                raise EngineError(ErrorCode.CONFIG_INVALID, f"{name} must be a mapping")
            detached = {}
            for objective, value in source.items():
                if not isinstance(objective, str) or not objective:
                    raise EngineError(
                        ErrorCode.CONFIG_INVALID, f"{name} keys must be non-empty objective names",
                    )
                number = _finite_number(value, f"{name}[{objective!r}]")
                if name == "preference_weights" and number <= 0:
                    raise EngineError(ErrorCode.CONFIG_INVALID, "preference_weights must be positive")
                detached[objective] = number
            object.__setattr__(self, name, MappingProxyType(detached))
        probability = _finite_number(self.min_joint_feasibility, "min_joint_feasibility")
        if not 0 <= probability <= 1:
            raise EngineError(ErrorCode.CONFIG_INVALID, "min_joint_feasibility must be in [0, 1]")
        object.__setattr__(self, "min_joint_feasibility", probability)

    def validate_for_space(self, space: Space) -> None:
        """Reject single-objective use and incomplete or unknown objective names."""
        if len(space.objectives) < 2:
            raise EngineError(
                ErrorCode.CONFIG_INVALID, "MultiObjectiveConfig requires at least two objectives",
            )
        expected = set(space.objectives)
        for name in ("reference_point", "preference_weights"):
            values = getattr(self, name)
            if values is not None and set(values) != expected:
                raise EngineError(
                    ErrorCode.CONFIG_INVALID,
                    f"{name} must exactly match space.objectives; "
                    f"missing={sorted(expected - set(values))!r}, "
                    f"unknown={sorted(set(values) - expected)!r}",
                )

    def to_dict(self) -> dict:
        """Return a detached JSON-compatible configuration snapshot."""
        return {
            "acquisition": self.acquisition,
            "noise_mode": self.noise_mode,
            "reference_point": None if self.reference_point is None else dict(self.reference_point),
            "preference_weights": (
                None if self.preference_weights is None else dict(self.preference_weights)
            ),
            "min_joint_feasibility": self.min_joint_feasibility,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MultiObjectiveConfig:
        """Restore only complete, known fields with the constructor's validation."""
        if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
            raise EngineError(ErrorCode.CONFIG_INVALID, "MultiObjectiveConfig must be a string-keyed mapping")
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise EngineError(
                ErrorCode.CONFIG_INVALID,
                "MultiObjectiveConfig fields must match exactly; "
                f"missing={sorted(expected - set(payload))!r}, "
                f"unknown={sorted(set(payload) - expected)!r}",
            )
        return cls(**payload)


def resolve_preference_weights(space: Space, runtime: Mapping, config: MultiObjectiveConfig):
    """Reconcile config, knowledge, then ranks without silently overriding any.

    Weights follow Space objective order. Equivalent knowledge declarations
    remain separate provenance entries but constitute one numerical preference.
    These weights belong to scalarization, never weighted hypervolume.
    """
    from .knowledge_runtime import RuntimePreference, validate_runtime_artifact

    config.validate_for_space(space)
    candidates = []

    def add(values, provenance):
        numbers = tuple(_finite_number(v, "preference weight") for v in values)
        if len(numbers) != len(space.objectives) or any(v <= 0 for v in numbers):
            raise EngineError(ErrorCode.CONFIG_INVALID, "Preferences require positive weights for all objectives")
        largest = max(numbers)
        scaled = tuple(v / largest for v in numbers)
        total = math.fsum(scaled)
        normalized = tuple(v / total for v in scaled)
        if any(v <= 0 for v in normalized):
            raise EngineError(ErrorCode.CONFIG_INVALID, "Preference normalization underflows")
        candidates.append((normalized, provenance))

    if config.preference_weights is not None:
        add([config.preference_weights[name] for name in space.objectives], {"source": "config"})
    for preference in runtime.get("acquisition_preferences", ()):
        if not isinstance(preference, RuntimePreference):
            raise EngineError(ErrorCode.CONFIG_INVALID, "Expected compiled RuntimePreference")
        validate_runtime_artifact(space, "acquisition_preferences", preference.artifact)
        weights = dict(zip(preference.payload["objectives"], preference.payload["weights"], strict=True))
        add([weights[name] for name in space.objectives], {
            "source": "knowledge", "source_pattern_id": preference.source_pattern_id,
            "source_pattern": preference.source_pattern, "source_version": preference.source_version,
        })
    priorities = tuple(objective.priority for objective in space.objective_specs)
    if len(set(priorities)) > 1:
        add([1 / (1 + rank) for rank in priorities], {"source": "objective_priority"})
    if not candidates:
        return None, ()
    first = candidates[0][0]
    if any(not all(math.isclose(a, b, rel_tol=1e-9, abs_tol=0.)
                   for a, b in zip(first, weights, strict=True)) for weights, _ in candidates[1:]):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Conflicting preference sources", details={
            "sources": [provenance for _, provenance in candidates],
            "objective_names": space.objectives,
        })
    return first, tuple(provenance for _, provenance in candidates)


def resolve_noise_mode(observations, config: MultiObjectiveConfig) -> str:
    """Resolve against original measurements, without synthesizing metadata.

    The caller must fit noiseless models with a temporary zero-Yvar batch when
    original Yvar is absent. Keep the original batch for geometry and history.
    """
    from expdoe_dk.domain import ObservationBatch

    if not isinstance(observations, ObservationBatch):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Noise resolution requires an ObservationBatch")
    variance = observations.Yvar
    mode = config.noise_mode
    if mode == "auto":
        return "inferred" if variance is None else "known"
    if mode == "known" and variance is None:
        raise EngineError(ErrorCode.CONFIG_INVALID, "known noise requires complete Yvar")
    if mode == "inferred" and variance is not None:
        raise EngineError(ErrorCode.CONFIG_INVALID, "inferred noise rejects supplied Yvar")
    if mode == "noiseless" and variance is not None:
        # The batch boundary permits missing variance only on unsuccessful rows.
        # Check every recorded value, including unsuccessful trials, without imputation.
        recorded = variance.to_numpy()[variance.notna().to_numpy()]
        if (recorded != 0).any():
            raise EngineError(ErrorCode.CONFIG_INVALID, "noiseless noise requires absent or all-zero Yvar")
    return mode


__all__ = ["MultiObjectiveConfig", "resolve_preference_weights", "resolve_noise_mode"]
