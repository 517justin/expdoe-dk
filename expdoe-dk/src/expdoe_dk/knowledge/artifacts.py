"""Typed immutable optimization artifacts compiled from knowledge patterns."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .specs import (
    JSONValue,
    _freeze_json_object,
    _require_exact_fields,
    _require_string,
    _thaw_json,
)


ARTIFACT_CATEGORIES = (
    "input_transforms",
    "outcome_transforms",
    "mean_components",
    "kernel_components",
    "priors",
    "virtual_observations",
    "parameter_constraints",
    "outcome_constraints",
    "acquisition_preferences",
    "diagnostics",
)


@dataclass(frozen=True)
class OptimizationArtifact:
    """One compiled artifact with complete source-pattern provenance."""

    kind: str
    payload: dict[str, JSONValue]
    source_pattern_id: str
    source_pattern: str
    source_version: str

    def __post_init__(self) -> None:
        _require_string(self.kind, "OptimizationArtifact.kind", nonempty=True)
        _require_string(
            self.source_pattern_id,
            "OptimizationArtifact.source_pattern_id",
            nonempty=True,
        )
        _require_string(
            self.source_pattern,
            "OptimizationArtifact.source_pattern",
            nonempty=True,
        )
        _require_string(
            self.source_version,
            "OptimizationArtifact.source_version",
            nonempty=True,
        )
        object.__setattr__(
            self,
            "payload",
            _freeze_json_object(self.payload, "OptimizationArtifact.payload"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "kind": self.kind,
            "payload": _thaw_json(self.payload),  # type: ignore[arg-type]
            "source_pattern_id": self.source_pattern_id,
            "source_pattern": self.source_pattern,
            "source_version": self.source_version,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "OptimizationArtifact":
        item = _require_exact_fields(
            payload,
            {
                "kind",
                "payload",
                "source_pattern_id",
                "source_pattern",
                "source_version",
            },
            "OptimizationArtifact",
        )
        return cls(
            kind=item["kind"],  # type: ignore[arg-type]
            payload=item["payload"],  # type: ignore[arg-type]
            source_pattern_id=item["source_pattern_id"],  # type: ignore[arg-type]
            source_pattern=item["source_pattern"],  # type: ignore[arg-type]
            source_version=item["source_version"],  # type: ignore[arg-type]
        )


def _normalize_artifact_tuple(value: object, context: str) -> tuple[OptimizationArtifact, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be an ordered sequence")
    items = tuple(value)
    if not all(isinstance(item, OptimizationArtifact) for item in items):
        raise TypeError(f"{context} must contain OptimizationArtifact instances")
    return items


@dataclass(frozen=True)
class OptimizationArtifacts:
    """The exact ordered artifact categories consumed by optimization."""

    input_transforms: tuple[OptimizationArtifact, ...] = ()
    outcome_transforms: tuple[OptimizationArtifact, ...] = ()
    mean_components: tuple[OptimizationArtifact, ...] = ()
    kernel_components: tuple[OptimizationArtifact, ...] = ()
    priors: tuple[OptimizationArtifact, ...] = ()
    virtual_observations: tuple[OptimizationArtifact, ...] = ()
    parameter_constraints: tuple[OptimizationArtifact, ...] = ()
    outcome_constraints: tuple[OptimizationArtifact, ...] = ()
    acquisition_preferences: tuple[OptimizationArtifact, ...] = ()
    diagnostics: tuple[OptimizationArtifact, ...] = ()

    def __post_init__(self) -> None:
        for category in ARTIFACT_CATEGORIES:
            object.__setattr__(
                self,
                category,
                _normalize_artifact_tuple(
                    getattr(self, category), f"OptimizationArtifacts.{category}"
                ),
            )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            category: [item.to_dict() for item in getattr(self, category)]
            for category in ARTIFACT_CATEGORIES
        }

    @classmethod
    def from_dict(cls, payload: object) -> "OptimizationArtifacts":
        item = _require_exact_fields(
            payload, set(ARTIFACT_CATEGORIES), "OptimizationArtifacts"
        )
        values: dict[str, tuple[OptimizationArtifact, ...]] = {}
        for category in ARTIFACT_CATEGORIES:
            serialized = item[category]
            if type(serialized) is not list:
                raise TypeError(
                    f"OptimizationArtifacts.{category} must be a JSON array"
                )
            values[category] = tuple(
                OptimizationArtifact.from_dict(value) for value in serialized
            )
        return cls(**values)


def merge_artifacts(
    artifact_sets: Iterable[OptimizationArtifacts],
) -> OptimizationArtifacts:
    """Concatenate each category deterministically in input order."""
    merged: dict[str, list[OptimizationArtifact]] = {
        category: [] for category in ARTIFACT_CATEGORIES
    }
    for index, artifacts in enumerate(artifact_sets):
        if not isinstance(artifacts, OptimizationArtifacts):
            raise TypeError(
                f"artifact_sets[{index}] must be an OptimizationArtifacts instance"
            )
        for category in ARTIFACT_CATEGORIES:
            merged[category].extend(getattr(artifacts, category))
    return OptimizationArtifacts(
        **{category: tuple(values) for category, values in merged.items()}
    )


__all__ = [
    "ARTIFACT_CATEGORIES",
    "OptimizationArtifact",
    "OptimizationArtifacts",
    "merge_artifacts",
]
