"""Insertion-independent exact knowledge pattern registry."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from expdoe_dk.errors import EngineError, ErrorCode

from ..artifacts import ARTIFACT_CATEGORIES, OptimizationArtifacts, merge_artifacts
from ..guard import KnowledgeValidationResult
from ..specs import KnowledgePatternSpec
from .definition import KnowledgePatternDefinition

if TYPE_CHECKING:
    from expdoe_dk.domain import ObservationBatch, Space


class PatternRegistry:
    """Own exact ``(pattern, version)`` definitions and dispatch to them."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], KnowledgePatternDefinition] = {}

    def register(self, definition: KnowledgePatternDefinition) -> None:
        if not isinstance(definition, KnowledgePatternDefinition):
            raise TypeError("definition must be a KnowledgePatternDefinition")
        key = (definition.pattern, definition.version)
        if key in self._definitions:
            raise ValueError(f"Pattern {key} already registered")
        self._definitions[key] = definition

    def resolve(self, pattern: str, version: str) -> KnowledgePatternDefinition:
        try:
            return self._definitions[(pattern, version)]
        except KeyError as error:
            raise EngineError(
                ErrorCode.KNOWLEDGE_INVALID,
                f"Unknown pattern {pattern}@{version}",
            ) from error

    def definitions(self) -> tuple[KnowledgePatternDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    @staticmethod
    def _require_spec(spec: object) -> KnowledgePatternSpec:
        if not isinstance(spec, KnowledgePatternSpec):
            raise TypeError("spec must be a KnowledgePatternSpec")
        return spec

    def validate(
        self,
        spec: KnowledgePatternSpec,
        space: "Space",
        observations: "ObservationBatch | None" = None,
    ) -> KnowledgeValidationResult:
        spec = self._require_spec(spec)
        result = self.resolve(spec.pattern, spec.version).validator(
            spec, space, observations
        )
        if not isinstance(result, KnowledgeValidationResult):
            raise TypeError("pattern validator must return KnowledgeValidationResult")
        if result.pattern_id != spec.pattern_id:
            raise EngineError(
                ErrorCode.KNOWLEDGE_INVALID,
                "Knowledge validation provenance mismatch: "
                f"expected pattern_id {spec.pattern_id!r}, "
                f"got {result.pattern_id!r}",
                details={
                    "expected": {"pattern_id": spec.pattern_id},
                    "actual": {"pattern_id": result.pattern_id},
                },
            )
        return result

    def validate_many(
        self,
        specs: Iterable[KnowledgePatternSpec],
        space: "Space",
        observations: "ObservationBatch | None" = None,
    ) -> tuple[KnowledgeValidationResult, ...]:
        return tuple(self.validate(spec, space, observations) for spec in specs)

    def compile_many(
        self,
        specs: Iterable[KnowledgePatternSpec],
        space: "Space",
        observations: "ObservationBatch | None" = None,
    ) -> OptimizationArtifacts:
        def compiled():
            for spec in specs:
                checked = self._require_spec(spec)
                artifacts = self.resolve(checked.pattern, checked.version).compiler(
                    checked, space, observations
                )
                if not isinstance(artifacts, OptimizationArtifacts):
                    raise TypeError(
                        "pattern compiler must return OptimizationArtifacts"
                    )
                self._validate_artifact_provenance(checked, artifacts)
                yield artifacts

        return merge_artifacts(compiled())

    @staticmethod
    def _validate_artifact_provenance(
        spec: KnowledgePatternSpec, artifacts: OptimizationArtifacts
    ) -> None:
        expected = {
            "source_pattern_id": spec.pattern_id,
            "source_pattern": spec.pattern,
            "source_version": spec.version,
        }
        for category in ARTIFACT_CATEGORIES:
            for index, artifact in enumerate(getattr(artifacts, category)):
                actual = {
                    "source_pattern_id": artifact.source_pattern_id,
                    "source_pattern": artifact.source_pattern,
                    "source_version": artifact.source_version,
                }
                if actual != expected:
                    raise EngineError(
                        ErrorCode.KNOWLEDGE_INVALID,
                        "Knowledge artifact provenance mismatch in "
                        f"{category}[{index}] for {spec.pattern}@{spec.version}",
                        details={
                            "category": category,
                            "artifact_index": index,
                            "expected": expected,
                            "actual": actual,
                        },
                    )

    def render(self, spec: KnowledgePatternSpec, space: "Space") -> str:
        spec = self._require_spec(spec)
        rendered = self.resolve(spec.pattern, spec.version).renderer(spec, space)
        if type(rendered) is not str:
            raise TypeError("pattern renderer must return a string")
        return rendered


__all__ = ["PatternRegistry"]
