"""Registered categorical ordering and similarity knowledge."""
from __future__ import annotations

from ..artifacts import OptimizationArtifact, OptimizationArtifacts
from ..guard import CompatibilityResult, KnowledgeValidationResult
from ..registry import KnowledgePatternDefinition


def _artifact(spec, payload: dict) -> OptimizationArtifact:
    return OptimizationArtifact(
        kind=f"{spec.pattern}_descriptor",
        payload=payload,
        source_pattern_id=spec.pattern_id,
        source_pattern=spec.pattern,
        source_version=spec.version,
    )


def _structural_errors(spec, space) -> tuple[str, ...]:
    if len(spec.scope.factors) != 1:
        return (f"{spec.pattern} requires exactly one scoped factor",)
    factor_name = spec.scope.factors[0]
    if factor_name not in space.param_names:
        return (f"Unknown factor {factor_name!r}",)
    factor = space.param_by_name(factor_name)
    required_kind = "ordinal" if spec.pattern == "ordinal_categories" else "categorical"
    errors: list[str] = []
    if factor.kind != required_kind:
        errors.append(f"Factor {factor_name!r} must be {required_kind}")
    declared = list(factor.values or ())
    levels = list(spec.parameters["levels"])
    if any(level not in declared for level in levels):
        errors.append("All referenced levels must exist in the factor")
    if len(set(levels)) != len(levels):
        errors.append("Referenced levels must be unique")
    if spec.pattern == "category_similarity":
        matrix = [list(row) for row in spec.parameters["matrix"]]
        if len(matrix) != len(levels) or any(len(row) != len(levels) for row in matrix):
            errors.append("Similarity matrix dimensions must match levels")
        elif any(
            not 0.0 <= float(matrix[row][column]) <= 1.0
            for row in range(len(levels))
            for column in range(len(levels))
        ):
            errors.append("Similarities must be bounded in [0, 1]")
        elif any(
            float(matrix[row][column]) != float(matrix[column][row])
            for row in range(len(levels))
            for column in range(len(levels))
        ):
            errors.append("Similarity matrix must be symmetric")
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
    factor = spec.scope.factors[0]
    return OptimizationArtifacts(
        kernel_components=(
            _artifact(
                spec,
                {
                    "factor": factor,
                    "dimension": space.param_names.index(factor),
                    "parameters": spec.to_dict()["parameters"],
                    "confidence": spec.confidence,
                },
            ),
        )
    )


def _render(spec, space) -> str:
    return f"{spec.pattern} for {spec.scope.factors[0]} levels {list(spec.parameters['levels'])}"


def _definition(pattern: str, properties: dict, required: list[str]):
    return KnowledgePatternDefinition(
        pattern=pattern,
        version="1.0",
        family="categorical",
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


def ordinal_categories_definition() -> KnowledgePatternDefinition:
    return _definition(
        "ordinal_categories",
        {"levels": {"type": "array", "items": {"type": "string"}, "minItems": 2, "uniqueItems": True}},
        ["levels"],
    )


def category_similarity_definition() -> KnowledgePatternDefinition:
    return _definition(
        "category_similarity",
        {
            "levels": {"type": "array", "items": {"type": "string"}, "minItems": 2, "uniqueItems": True},
            "matrix": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        ["levels", "matrix"],
    )


__all__ = ["category_similarity_definition", "ordinal_categories_definition"]
