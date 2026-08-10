"""Registered optimum-shape and random-augmentation knowledge."""
from __future__ import annotations

import math

from ..artifacts import OptimizationArtifact, OptimizationArtifacts
from ..guard import CompatibilityResult, KnowledgeValidationResult
from ..registry import KnowledgePatternDefinition


_NUMERIC_KINDS = frozenset({"continuous", "integer", "discrete"})


def _has_usable_shape_evidence(spec, space, observations) -> bool:
    """Return whether a batch is sufficient to attempt a shape agreement test."""
    if observations is None:
        return False
    successful = observations.successful()
    factor = spec.scope.factors[0]
    objectives = spec.scope.objectives or tuple(space.objectives)
    X = successful.X
    Y = successful.Y
    if (
        len(X) < 4
        or factor not in X.columns
        or not objectives
        or any(objective not in Y.columns for objective in objectives)
    ):
        return False
    try:
        factor_values = tuple(float(value) for value in X[factor])
        objective_values = tuple(
            float(value)
            for objective in objectives
            for value in Y[objective]
        )
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (*factor_values, *objective_values)):
        return False
    if len(set(factor_values)) < 3:
        return False
    parameter = space.param_by_name(factor)
    if parameter.bounds is not None:
        lo, hi = map(float, parameter.bounds)
    else:
        levels = tuple(float(value) for value in parameter.numeric_levels)
        lo, hi = min(levels), max(levels)
    return max(factor_values) - min(factor_values) >= 0.25 * (hi - lo)


def _artifact(spec, kind: str, payload: dict) -> OptimizationArtifact:
    return OptimizationArtifact(
        kind=kind,
        payload=payload,
        source_pattern_id=spec.pattern_id,
        source_pattern=spec.pattern,
        source_version=spec.version,
    )


def _compile_quadratic_peak(spec, space, observations=None) -> OptimizationArtifacts:
    parameters = spec.parameters
    factor = spec.scope.factors[0]
    dim = space.param_names.index(factor)
    lo, hi = space.param_by_name(factor).bounds
    center_unit = (parameters["center"] - lo) / (hi - lo)
    curvature_signs = [0.0] * space.n_dims
    if parameters["direction"] == "peak":
        sign_internal = 1.0 if space.maximize[0] else -1.0
    else:
        sign_internal = -1.0 if space.maximize[0] else 1.0
    curvature_signs[dim] = sign_internal
    centers = [0.5] * space.n_dims
    centers[dim] = float(center_unit)
    return OptimizationArtifacts(
        mean_components=(
            _artifact(
                spec,
                "quadratic_mean",
                {
                    "input_dim": space.n_dims,
                    "curvature_signs": curvature_signs,
                    "centers": centers,
                },
            ),
        )
    )


def _compile_random_augment(spec, space, observations=None) -> OptimizationArtifacts:
    return OptimizationArtifacts(
        virtual_observations=(
            _artifact(spec, "random_augment", {"n": spec.parameters["n"]}),
        )
    )


def _validate_factor(spec, space, observations=None) -> KnowledgeValidationResult:
    errors: list[str] = []
    if len(spec.scope.factors) != 1:
        errors.append(f"{spec.pattern} requires exactly one scoped factor")
        factor = None
    else:
        factor = spec.scope.factors[0]
        if factor not in space.param_names:
            errors.append(f"Unknown factor {factor!r}")
        else:
            parameter = space.param_by_name(factor)
            if parameter.kind not in _NUMERIC_KINDS:
                errors.append(f"Factor {factor!r} must be numeric")
            elif spec.pattern == "quadratic_peak":
                if parameter.bounds is not None:
                    lo, hi = map(float, parameter.bounds)
                else:
                    levels = tuple(float(value) for value in parameter.numeric_levels)
                    lo, hi = min(levels), max(levels)
                if not lo <= float(spec.parameters["center"]) <= hi:
                    errors.append("center must lie within factor bounds")
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="invalid" if errors else "valid",
        summary=(f"Invalid {spec.pattern} declaration" if errors else f"Factor {factor!r} is available"),
        errors=tuple(errors),
        effective_confidence=spec.confidence,
    )


def _validate(spec, space, observations=None) -> KnowledgeValidationResult:
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="valid",
        summary="Random augmentation is valid",
        effective_confidence=spec.confidence,
    )


def _render(spec, space) -> str:
    if spec.scope.factors:
        return f"{spec.pattern} on {spec.scope.factors[0]}"
    return f"random augmentation ({spec.parameters['n']})"


def _compatible_factor(spec, space) -> CompatibilityResult:
    result = _validate_factor(spec, space)
    return CompatibilityResult(compatible=result.state != "invalid", reasons=result.errors)


def _compatible(spec, space) -> CompatibilityResult:
    return CompatibilityResult(compatible=True)


def _new_shape_errors(spec, space) -> tuple[str, ...]:
    errors: list[str] = []
    unknown_objectives = [
        objective
        for objective in spec.scope.objectives
        if objective not in space.objectives
    ]
    if unknown_objectives:
        errors.append(f"Unknown objectives {unknown_objectives!r}")
    if len(spec.scope.factors) != 1:
        errors.append(f"{spec.pattern} requires exactly one scoped factor")
    elif spec.scope.factors[0] not in space.param_names:
        errors.append(f"Unknown factor {spec.scope.factors[0]!r}")
    else:
        factor = space.param_by_name(spec.scope.factors[0])
        if factor.kind not in _NUMERIC_KINDS:
            errors.append(f"Factor {factor.name!r} must be numeric")
        else:
            if factor.bounds is not None:
                lo, hi = (float(value) for value in factor.bounds)
            else:
                levels = tuple(float(value) for value in factor.numeric_levels)
                lo, hi = min(levels), max(levels)
            parameters = spec.parameters
            if spec.pattern in {"saturation", "threshold"}:
                location_name = (
                    "half_response" if spec.pattern == "saturation" else "threshold"
                )
                location = float(parameters[location_name])
                if not lo <= location <= hi:
                    errors.append(f"{location_name} must lie within factor bounds")
            elif spec.pattern == "quadratic_valley":
                if not lo <= float(parameters["center"]) <= hi:
                    errors.append("center must lie within factor bounds")
                if float(parameters["width"]) <= 0:
                    errors.append("width must be positive")
                elif float(parameters["width"]) > hi - lo:
                    errors.append("width must not exceed the factor range")
            elif spec.pattern == "optimum_range":
                lower = float(parameters["lower"])
                upper = float(parameters["upper"])
                if not lo <= lower < upper <= hi:
                    errors.append("optimum range must be ordered within factor bounds")
            elif spec.pattern == "power_law" and lo <= 0:
                errors.append("power_law requires a positive factor domain")
            elif spec.pattern == "periodic":
                if float(parameters["period"]) <= 0:
                    errors.append("period must be positive")
                if hi - lo <= 0:
                    errors.append("factor range must be positive")
    return tuple(errors)


def _new_shape_validation(spec, space, observations=None) -> KnowledgeValidationResult:
    errors = _new_shape_errors(spec, space)
    if errors:
        return KnowledgeValidationResult(
            pattern_id=spec.pattern_id,
            state="invalid",
            summary=f"Invalid {spec.pattern} declaration",
            errors=tuple(errors),
            effective_confidence=spec.confidence,
        )
    empirically_supported = False
    if observations is not None and spec.pattern == "periodic":
        factor_name = spec.scope.factors[0]
        successful = observations.successful().X
        empirically_supported = (
            factor_name in successful
            and len(successful) >= 2
            and float(successful[factor_name].max() - successful[factor_name].min())
            >= float(spec.parameters["period"])
        )
    elif _has_usable_shape_evidence(spec, space, observations):
        # The registry currently records descriptors rather than fitting them.
        # Adequate rows permit a future agreement test but are not themselves
        # evidence that the declared response shape agrees with observations.
        empirically_supported = False
    state = "valid" if empirically_supported else "insufficient_data"
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state=state,
        summary=(
            f"{spec.pattern} is structurally valid; empirical coverage is insufficient"
            if not empirically_supported
            else f"{spec.pattern} is valid"
        ),
        effective_confidence=spec.confidence,
    )


def _new_shape_compatibility(spec, space) -> CompatibilityResult:
    reasons = _new_shape_errors(spec, space)
    return CompatibilityResult(compatible=not reasons, reasons=reasons)


def _compile_shape_descriptor(spec, space, observations=None) -> OptimizationArtifacts:
    factor = spec.scope.factors[0]
    payload = {
        "factor": factor,
        "dimension": space.param_names.index(factor),
        "parameters": spec.to_dict()["parameters"],
        "objectives": list(spec.scope.objectives),
        "confidence": spec.confidence,
    }
    artifact = _artifact(spec, f"{spec.pattern}_descriptor", payload)
    if spec.pattern == "optimum_range":
        return OptimizationArtifacts(virtual_observations=(artifact,))
    return OptimizationArtifacts(mean_components=(artifact,))


def _new_shape_renderer(spec, space) -> str:
    return (
        f"{spec.pattern} response on {spec.scope.factors[0]} "
        f"with {spec.to_dict()['parameters']}"
    )


def _number_schema(*, exclusive_minimum=None):
    schema = {"type": "number"}
    if exclusive_minimum is not None:
        schema["exclusiveMinimum"] = exclusive_minimum
    return schema


def _shape_definition(pattern: str, properties: dict, required: list[str]):
    return KnowledgePatternDefinition(
        pattern=pattern,
        version="1.0",
        family="shape",
        schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        compiler=_compile_shape_descriptor,
        validator=_new_shape_validation,
        renderer=_new_shape_renderer,
        compatibility=_new_shape_compatibility,
    )


def saturation_definition() -> KnowledgePatternDefinition:
    return _shape_definition(
        "saturation",
        {
            "direction": {"enum": ["increasing", "decreasing"]},
            "half_response": {"type": "number"},
        },
        ["direction", "half_response"],
    )


def threshold_definition() -> KnowledgePatternDefinition:
    behavior = {"enum": ["increasing", "decreasing", "flat"]}
    return _shape_definition(
        "threshold",
        {
            "threshold": {"type": "number"},
            "below_behavior": behavior,
            "above_behavior": behavior,
        },
        ["threshold", "below_behavior", "above_behavior"],
    )


def quadratic_valley_definition() -> KnowledgePatternDefinition:
    return _shape_definition(
        "quadratic_valley",
        {"center": {"type": "number"}, "width": _number_schema(exclusive_minimum=0)},
        ["center", "width"],
    )


def optimum_range_definition() -> KnowledgePatternDefinition:
    return _shape_definition(
        "optimum_range",
        {"lower": {"type": "number"}, "upper": {"type": "number"}},
        ["lower", "upper"],
    )


def power_law_definition() -> KnowledgePatternDefinition:
    return _shape_definition(
        "power_law",
        {"exponent": {"type": "number"}, "scale": _number_schema(exclusive_minimum=0)},
        ["exponent", "scale"],
    )


def exponential_definition() -> KnowledgePatternDefinition:
    return _shape_definition(
        "exponential",
        {"rate": {"type": "number"}, "amplitude": _number_schema(exclusive_minimum=0)},
        ["rate", "amplitude"],
    )


def periodic_definition() -> KnowledgePatternDefinition:
    return _shape_definition(
        "periodic",
        {"period": _number_schema(exclusive_minimum=0), "phase": {"type": "number"}},
        ["period", "phase"],
    )


def quadratic_peak_definition() -> KnowledgePatternDefinition:
    return KnowledgePatternDefinition(
        pattern="quadratic_peak",
        version="1.0",
        family="shape",
        schema={
            "type": "object",
            "properties": {
                "center": {"type": "number"},
                "direction": {"enum": ["peak", "valley"]},
                "frozen": {"type": "boolean"},
            },
            "required": ["center", "direction", "frozen"],
            "additionalProperties": False,
        },
        compiler=_compile_quadratic_peak,
        validator=_validate_factor,
        renderer=_render,
        compatibility=_compatible_factor,
    )


def random_augment_definition() -> KnowledgePatternDefinition:
    return KnowledgePatternDefinition(
        pattern="random_augment",
        version="1.0",
        family="experimental",
        schema={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "minimum": 1, "maximum": 4096}
            },
            "required": ["n"],
            "additionalProperties": False,
        },
        compiler=_compile_random_augment,
        validator=_validate,
        renderer=_render,
        compatibility=_compatible,
    )


__all__ = [
    "exponential_definition",
    "optimum_range_definition",
    "periodic_definition",
    "power_law_definition",
    "quadratic_peak_definition",
    "quadratic_valley_definition",
    "random_augment_definition",
    "saturation_definition",
    "threshold_definition",
]
