"""Registered multi-factor interaction knowledge."""
from __future__ import annotations

from ..artifacts import OptimizationArtifact, OptimizationArtifacts
from ..guard import CompatibilityResult, KnowledgeValidationResult
from ..registry import KnowledgePatternDefinition


_NUMERIC_KINDS = frozenset({"continuous", "integer", "discrete"})


def _numeric_limits(parameter) -> tuple[float, float]:
    if parameter.bounds is not None:
        return float(parameter.bounds[0]), float(parameter.bounds[1])
    levels = tuple(float(value) for value in parameter.numeric_levels)
    return min(levels), max(levels)


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
    unknown_objectives = [
        objective
        for objective in spec.scope.objectives
        if objective not in space.objectives
    ]
    if unknown_objectives:
        errors.append(f"Unknown objectives {unknown_objectives!r}")
    if len(spec.scope.factors) != 2:
        errors.append(f"{spec.pattern} requires exactly two distinct scoped factors")
        return tuple(errors)
    first, second = spec.scope.factors
    if first == second:
        errors.append("Interaction factors must be distinct")
    unknown = [factor for factor in (first, second) if factor not in space.param_names]
    if unknown:
        errors.append(f"Unknown factors {unknown!r}")
        return tuple(errors)
    nonnumeric = [
        factor
        for factor in (first, second)
        if space.param_by_name(factor).kind not in _NUMERIC_KINDS
    ]
    if nonnumeric:
        errors.append(f"Interaction factors must be numeric; got {nonnumeric!r}")
    if spec.pattern == "conditional_effect" and not nonnumeric:
        conditioning_low, conditioning_high = _numeric_limits(
            space.param_by_name(second)
        )
        threshold = float(spec.parameters["value"])
        if not conditioning_low <= threshold <= conditioning_high:
            errors.append(
                "conditional_effect value must lie within the conditioning factor domain"
            )
    if spec.pattern == "ratio_optimum" and not nonnumeric:
        numerator = space.param_by_name(first)
        denominator = space.param_by_name(second)
        numerator_low, numerator_high = _numeric_limits(numerator)
        denominator_low, denominator_high = _numeric_limits(denominator)
        if denominator_low <= 0:
            errors.append("ratio_optimum denominator domain must be positive")
        ratio = float(spec.parameters["ratio"])
        tolerance = float(spec.parameters["tolerance"])
        if ratio <= 0:
            errors.append("ratio must be positive")
        if tolerance <= 0:
            errors.append("ratio tolerance must be positive")
        if denominator_low > 0 and ratio > 0 and tolerance > 0:
            endpoint_ratios = (
                numerator_low / denominator_low,
                numerator_low / denominator_high,
                numerator_high / denominator_low,
                numerator_high / denominator_high,
            )
            feasible_low = min(endpoint_ratios)
            feasible_high = max(endpoint_ratios)
            if ratio + tolerance < feasible_low or ratio - tolerance > feasible_high:
                errors.append("Requested ratio range is infeasible for factor domains")
    return tuple(errors)


def _validate(spec, space, observations=None) -> KnowledgeValidationResult:
    errors = _structural_errors(spec, space)
    if errors:
        return KnowledgeValidationResult(
            pattern_id=spec.pattern_id,
            state="invalid",
            summary=f"Invalid {spec.pattern} declaration",
            errors=errors,
            effective_confidence=spec.confidence,
        )
    empirically_supported = observations is not None
    if observations is not None:
        first, second = spec.scope.factors
        successful = observations.successful().X
        empirically_supported = (
            first in successful
            and second in successful
            and successful[first].nunique() >= 2
            and successful[second].nunique() >= 2
            and len(successful[[first, second]].drop_duplicates()) >= 4
        )
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="valid" if empirically_supported else "insufficient_data",
        summary=(
            f"{spec.pattern} is structurally valid; joint coverage is insufficient"
            if not empirically_supported
            else f"{spec.pattern} is valid"
        ),
        effective_confidence=spec.confidence,
    )


def _compatible(spec, space) -> CompatibilityResult:
    errors = _structural_errors(spec, space)
    return CompatibilityResult(compatible=not errors, reasons=errors)


def _compile(spec, space, observations=None) -> OptimizationArtifacts:
    factors = list(spec.scope.factors)
    artifact = _artifact(
        spec,
        f"{spec.pattern}_descriptor",
        {
            "factors": factors,
            "dimensions": [space.param_names.index(factor) for factor in factors],
            "parameters": spec.to_dict()["parameters"],
            "objectives": list(spec.scope.objectives),
            "confidence": spec.confidence,
        },
    )
    if spec.pattern in {"synergy", "antagonism"}:
        return OptimizationArtifacts(kernel_components=(artifact,))
    return OptimizationArtifacts(mean_components=(artifact,))


def _render(spec, space) -> str:
    return f"{spec.pattern} interaction between {spec.scope.factors[0]} and {spec.scope.factors[1]}"


def _definition(pattern: str, properties: dict, required: list[str]):
    return KnowledgePatternDefinition(
        pattern=pattern,
        version="1.0",
        family="interaction",
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


def synergy_definition() -> KnowledgePatternDefinition:
    return _definition("synergy", {}, [])


def antagonism_definition() -> KnowledgePatternDefinition:
    return _definition("antagonism", {}, [])


def conditional_effect_definition() -> KnowledgePatternDefinition:
    return _definition(
        "conditional_effect",
        {
            "operator": {"enum": ["<", "<=", "==", ">=", ">"]},
            "value": {"type": "number"},
        },
        ["operator", "value"],
    )


def ratio_optimum_definition() -> KnowledgePatternDefinition:
    return _definition(
        "ratio_optimum",
        {
            "ratio": {"type": "number", "exclusiveMinimum": 0},
            "tolerance": {"type": "number", "exclusiveMinimum": 0},
        },
        ["ratio", "tolerance"],
    )


__all__ = [
    "antagonism_definition",
    "conditional_effect_definition",
    "ratio_optimum_definition",
    "synergy_definition",
]
