"""Registered hard safe and forbidden domain-region knowledge."""
from __future__ import annotations

import json

from expdoe_dk.domain import ExpressionConstraint, constraint_from_dict
from expdoe_dk.errors import EngineError, ErrorCode

from ..artifacts import OptimizationArtifact, OptimizationArtifacts
from ..guard import CompatibilityResult, KnowledgeValidationResult
from ..registry import KnowledgePatternDefinition, PatternRegistry


def _constraint(spec, space) -> ExpressionConstraint:
    template = PatternRegistry._plain_json(spec.parameters["constraint"])
    if spec.pattern == "forbidden_region":
        template["ast"] = {
            "type": "compare",
            "op": "==",
            "left": template["ast"],
            "right": {"type": "constant", "value": False},
        }
    constraint = constraint_from_dict(
        {"name": f"knowledge-{spec.pattern_id}", **template},
        allowed_names=spec.scope.factors or space.param_names,
    )
    if not isinstance(constraint, ExpressionConstraint):
        raise TypeError("safety constraint must be an expression constraint")
    return constraint


def _validate(spec, space, observations=None) -> KnowledgeValidationResult:
    unknown_scope = [
        factor for factor in spec.scope.factors if factor not in space.param_names
    ]
    if unknown_scope:
        errors = (f"Unknown factors {unknown_scope!r}",)
        warnings = ()
        error_code = None
        error_details = None
    else:
        try:
            constraint = _constraint(spec, space)
            status, details = PatternRegistry._safety_constraint_certificate(
                ((spec.pattern_id, constraint),), space
            )
            if status == "infeasible":
                raise EngineError(
                    ErrorCode.KNOWLEDGE_CONFLICT,
                    "Hard safety regions have a certified empty intersection",
                    details=details,
                )
        except EngineError as error:
            errors = (str(error),)
            warnings = ()
            error_code = error.code
            error_details = error.details
        except (TypeError, ValueError) as error:
            errors = (str(error),)
            warnings = ()
            error_code = None
            error_details = None
        else:
            errors = ()
            warnings = (
                ("Safety feasibility is inconclusive: "
                 f"{details['reason']}",)
                if status == "inconclusive"
                else ()
            )
            error_code = None
            error_details = None
    state = "invalid" if errors else ("insufficient_data" if warnings else "valid")
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state=state,
        summary=(
            f"Invalid {spec.pattern} declaration"
            if errors
            else (
                f"{spec.pattern} safety feasibility is inconclusive"
                if warnings
                else f"{spec.pattern} is a valid hard region"
            )
        ),
        errors=errors,
        warnings=warnings,
        effective_confidence=spec.confidence,
        error_code=error_code,
        error_details=error_details,
    )


def _compatible(spec, space) -> CompatibilityResult:
    unknown_scope = [
        factor for factor in spec.scope.factors if factor not in space.param_names
    ]
    if unknown_scope:
        return CompatibilityResult(
            compatible=False,
            reasons=(f"Unknown factors {unknown_scope!r}",),
        )
    try:
        _constraint(spec, space)
    except (EngineError, TypeError, ValueError) as error:
        return CompatibilityResult(compatible=False, reasons=(str(error),))
    return CompatibilityResult(compatible=True)


def _compile(spec, space, observations=None) -> OptimizationArtifacts:
    constraint = _constraint(spec, space)
    return OptimizationArtifacts(
        parameter_constraints=(
            OptimizationArtifact(
                kind="hard_parameter_constraint",
                payload={
                    "constraint": constraint.to_dict(),
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
    ast = PatternRegistry._plain_json(spec.parameters["constraint"]["ast"])
    return f"{semantics}: {json.dumps(ast, sort_keys=True, separators=(',', ':'))}"


def _definition(pattern: str) -> KnowledgePatternDefinition:
    return KnowledgePatternDefinition(
        pattern=pattern,
        version="1.0",
        family="feasibility",
        schema={
            "type": "object",
            "properties": {
                "constraint": {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "expression"},
                        "hard": {"const": True},
                        "ast": {"type": "object"},
                    },
                    "required": ["kind", "hard", "ast"],
                    "additionalProperties": False,
                }
            },
            "required": ["constraint"],
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
