"""Registered knowledge pattern definition contract."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..specs import JSONValue, _freeze_json_object, _require_string

if TYPE_CHECKING:
    from expdoe_dk.domain import ObservationBatch, Space

    from ..artifacts import OptimizationArtifacts
    from ..guard import CompatibilityResult, KnowledgeValidationResult
    from ..specs import KnowledgePatternSpec


KnowledgeCompiler = Callable[
    ["KnowledgePatternSpec", "Space", "ObservationBatch | None"],
    "OptimizationArtifacts",
]
KnowledgeValidator = Callable[
    ["KnowledgePatternSpec", "Space", "ObservationBatch | None"],
    "KnowledgeValidationResult",
]
KnowledgeRenderer = Callable[["KnowledgePatternSpec", "Space"], str]
CompatibilityRule = Callable[
    ["KnowledgePatternSpec", "Space"], "CompatibilityResult"
]


@dataclass(frozen=True)
class KnowledgePatternDefinition:
    pattern: str
    version: str
    family: str
    schema: dict[str, JSONValue]
    compiler: KnowledgeCompiler
    validator: KnowledgeValidator
    renderer: KnowledgeRenderer
    compatibility: CompatibilityRule

    def __post_init__(self) -> None:
        _require_string(self.pattern, "KnowledgePatternDefinition.pattern", nonempty=True)
        _require_string(self.version, "KnowledgePatternDefinition.version", nonempty=True)
        _require_string(self.family, "KnowledgePatternDefinition.family", nonempty=True)
        object.__setattr__(
            self,
            "schema",
            _freeze_json_object(self.schema, "KnowledgePatternDefinition.schema"),
        )
        for name in ("compiler", "validator", "renderer", "compatibility"):
            if not callable(getattr(self, name)):
                raise TypeError(f"KnowledgePatternDefinition.{name} must be callable")


__all__ = [
    "CompatibilityRule",
    "KnowledgeCompiler",
    "KnowledgePatternDefinition",
    "KnowledgeRenderer",
    "KnowledgeValidator",
]
