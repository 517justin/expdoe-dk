"""Registered hard safe and forbidden domain-region knowledge."""
from __future__ import annotations

from expdoe_dk.domain import ExpressionConstraint
from expdoe_dk.errors import EngineError

from ..artifacts import OptimizationArtifact, OptimizationArtifacts
from ..guard import CompatibilityResult, KnowledgeValidationResult
from ..registry import KnowledgePatternDefinition, PatternRegistry


def _constraint(spec, space) -> ExpressionConstraint:
    expression = spec.parameters["expression"]
    if spec.pattern == "forbidden_region":
        expression = f"({expression}) == False"
    return ExpressionConstraint(
        f"knowledge-{spec.pattern_id}",
        expression,
        hard=True,
        allowed_names=spec.scope.factors or space.param_names,
    )


def _validate(spec, space, observations=None) -> KnowledgeValidationResult:
    unknown_scope = [
        factor for factor in spec.scope.factors if factor not in space.param_names
    ]
    if unknown_scope:
        errors = (f"Unknown factors {unknown_scope!r}",)
    else:
        try:
            _constraint(spec, space)
            artifacts = _compile(spec, space)
            PatternRegistry._reject_proven_empty_safety_intersection(
                artifacts, space
            )
        except (EngineError, TypeError, ValueError) as error:
            errors = (str(error),)
        else:
            errors = ()
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="invalid" if errors else "valid",
        summary=(f"Invalid {spec.pattern} declaration" if errors else f"{spec.pattern} is a valid hard region"),
        errors=errors,
        effective_confidence=spec.confidence,
    )


def _compatible(spec, space) -> CompatibilityResult:
    result = _validate(spec, space)
    return CompatibilityResult(compatible=result.state != "invalid", reasons=result.errors)


def _compile(spec, space, observations=None) -> OptimizationArtifacts:
    constraint = _constraint(spec, space)
    return OptimizationArtifacts(
        parameter_constraints=(
            OptimizationArtifact(
                kind="hard_parameter_constraint",
                payload={
                    "constraint": constraint.to_dict(),
                    "source_expression": spec.parameters["expression"],
                    "region_semantics": "allowed" if spec.pattern == "safe_region" else "forbidden",
                },
                source_pattern_id=spec.pattern_id,
                source_pattern=spec.pattern,
                source_version=spec.version,
            ),
        )
    )


def _render(spec, space) -> str:
    semantics = "must satisfy" if spec.pattern == "safe_region" else "must avoid"
    return f"{semantics}: {spec.parameters['expression']}"


def _definition(pattern: str) -> KnowledgePatternDefinition:
    return KnowledgePatternDefinition(
        pattern=pattern,
        version="1.0",
        family="feasibility",
        schema={
            "type": "object",
            "properties": {"expression": {"type": "string", "minLength": 1}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        compiler=_compile,
        validator=_validate,
        renderer=_render,
        compatibility=_compatible,
    )


def safe_region_definition() -> KnowledgePatternDefinition:
    return _definition("safe_region")


def forbidden_region_definition() -> KnowledgePatternDefinition:
    return _definition("forbidden_region")


__all__ = ["forbidden_region_definition", "safe_region_definition"]
