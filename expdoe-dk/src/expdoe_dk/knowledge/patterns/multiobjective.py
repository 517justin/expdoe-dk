"""Registered objective ranges and bounded acquisition preferences."""
from __future__ import annotations

import math

from expdoe_dk.errors import EngineError, ErrorCode

from ..artifacts import OptimizationArtifact, OptimizationArtifacts
from ..guard import CompatibilityResult, KnowledgeValidationResult
from ..registry import KnowledgePatternDefinition


def _artifact(spec, kind: str, payload: dict) -> OptimizationArtifact:
    return OptimizationArtifact(
        kind=kind,
        payload=payload,
        source_pattern_id=spec.pattern_id,
        source_pattern=spec.pattern,
        source_version=spec.version,
    )


def _structural_errors(spec, space) -> tuple[str, ...]:
    errors: list[str] = []
    objectives = spec.scope.objectives
    unknown = [objective for objective in objectives if objective not in space.objectives]
    if unknown:
        errors.append(f"Unknown objectives {unknown!r}")
    if spec.pattern == "target_range":
        if len(objectives) != 1:
            errors.append("target_range requires exactly one scoped objective")
        if float(spec.parameters["lower"]) >= float(spec.parameters["upper"]):
            errors.append("target range must be ordered")
    else:
        weights = tuple(float(value) for value in spec.parameters["weights"])
        if len(objectives) < 1 or len(weights) != len(objectives):
            errors.append("Preference weights must match scoped objectives")
        if any(not math.isfinite(value) or value < 0 for value in weights):
            errors.append("Preference weights must be finite and nonnegative")
        if not any(value > 0 for value in weights):
            errors.append("At least one preference weight must be positive")
        if spec.pattern == "tradeoff" and len(objectives) != 2:
            errors.append("tradeoff requires exactly two scoped objectives")
    return tuple(errors)


def _validate(spec, space, observations=None) -> KnowledgeValidationResult:
    errors = _structural_errors(spec, space)
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="invalid" if errors else "valid",
        summary=(f"Invalid {spec.pattern} declaration" if errors else f"{spec.pattern} is valid"),
        errors=errors,
        effective_confidence=spec.confidence,
    )


def _compatible(spec, space) -> CompatibilityResult:
    errors = _structural_errors(spec, space)
    return CompatibilityResult(compatible=not errors, reasons=errors)


def _compile(spec, space, observations=None) -> OptimizationArtifacts:
    objectives = list(spec.scope.objectives)
    if spec.pattern == "target_range":
        artifact = _artifact(
            spec,
            "target_range_constraint",
            {
                "objective": objectives[0],
                "lower": spec.parameters["lower"],
                "upper": spec.parameters["upper"],
                "hard": False,
                "confidence": spec.confidence,
            },
        )
        return OptimizationArtifacts(outcome_constraints=(artifact,))
    weights = [float(value) for value in spec.parameters["weights"]]
    scale = max(weights, default=0.0)
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or any(not math.isfinite(value) or value < 0.0 for value in weights)
    ):
        raise EngineError(
            ErrorCode.KNOWLEDGE_INVALID,
            "Acquisition preference weights must be finite, nonnegative, and not all zero",
            details={
                "pattern_id": spec.pattern_id,
                "pattern": spec.pattern,
                "version": spec.version,
            },
        )
    scaled = [value / scale for value in weights]
    scaled_total = sum(scaled)
    normalized = [value / scaled_total for value in scaled]
    return OptimizationArtifacts(
        acquisition_preferences=(
            _artifact(
                spec,
                f"{spec.pattern}_preference",
                {
                    "objectives": objectives,
                    "weights": normalized,
                    "confidence": spec.confidence,
                },
            ),
        )
    )


def _render(spec, space) -> str:
    return f"{spec.pattern} for objectives {list(spec.scope.objectives)}"


def _definition(pattern: str, properties: dict, required: list[str]):
    return KnowledgePatternDefinition(
        pattern=pattern,
        version="1.0",
        family="multiobjective",
        schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        compiler=_compile,
        validator=_validate,
        renderer=_render,
        compatibility=_compatible,
    )


def target_range_definition() -> KnowledgePatternDefinition:
    return _definition(
        "target_range",
        {"lower": {"type": "number"}, "upper": {"type": "number"}},
        ["lower", "upper"],
    )


def objective_priority_definition() -> KnowledgePatternDefinition:
    return _definition(
        "objective_priority",
        {"weights": {"type": "array", "minItems": 1, "items": {"type": "number", "minimum": 0}}},
        ["weights"],
    )


def tradeoff_definition() -> KnowledgePatternDefinition:
    return _definition(
        "tradeoff",
        {"weights": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "minimum": 0}}},
        ["weights"],
    )


__all__ = ["objective_priority_definition", "target_range_definition", "tradeoff_definition"]
