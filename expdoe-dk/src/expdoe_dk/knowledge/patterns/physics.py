"""Registered physical-law and monotonicity knowledge."""
from __future__ import annotations

import math

from .._frame import flip_for_minimize
from ..artifacts import OptimizationArtifact, OptimizationArtifacts
from ..guard import CompatibilityResult, KnowledgeValidationResult
from ..registry import KnowledgePatternDefinition


def _factor_index(spec, space) -> int:
    return space.param_names.index(spec.scope.factors[0])


def _artifact(spec, kind: str, payload: dict) -> OptimizationArtifact:
    return OptimizationArtifact(
        kind=kind,
        payload=payload,
        source_pattern_id=spec.pattern_id,
        source_pattern=spec.pattern,
        source_version=spec.version,
    )


def _compile_arrhenius(spec, space, observations=None) -> OptimizationArtifacts:
    parameters = spec.parameters
    return OptimizationArtifacts(
        mean_components=(
            _artifact(
                spec,
                "arrhenius_mean",
                {
                    "temp_dim_index": _factor_index(spec, space),
                    "activation_energy": parameters["activation_energy"],
                    "amplitude_init": parameters["amplitude_init"],
                },
            ),
        )
    )


def _compile_monotone(spec, space, observations=None) -> OptimizationArtifacts:
    parameters = spec.parameters
    direction = parameters["direction"]
    effect = (
        "increases_objective"
        if direction == "increasing"
        else "decreases_objective"
    )
    return OptimizationArtifacts(
        virtual_observations=(
            _artifact(
                spec,
                "monotone",
                {
                    "dim": _factor_index(spec, space),
                    "direction": flip_for_minimize(effect, space.maximize[0]),
                    "n_pairs_per_dim": parameters["n_pairs_per_dim"],
                    "epsilon": parameters["epsilon"],
                    "delta_norm": parameters["delta_norm"],
                },
            ),
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
        elif space.param_by_name(factor).kind not in {
            "continuous",
            "integer",
            "discrete",
        }:
            errors.append(f"Factor {factor!r} must be numeric")
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="invalid" if errors else "valid",
        summary=(f"Invalid {spec.pattern} declaration" if errors else f"Factor {factor!r} is available"),
        errors=tuple(errors),
        effective_confidence=spec.confidence,
    )


def _render(spec, space) -> str:
    return f"{spec.pattern} on {spec.scope.factors[0]}"


def _compatible_factor(spec, space) -> CompatibilityResult:
    result = _validate_factor(spec, space)
    return CompatibilityResult(compatible=result.state != "invalid", reasons=result.errors)


_TEMPERATURE_UNITS = frozenset({"", "K", "kelvin", "C", "°C", "celsius", "F", "°F", "fahrenheit"})


def _arrhenius_errors(spec, space) -> tuple[str, ...]:
    if len(spec.scope.factors) != 1:
        return ("arrhenius requires exactly one temperature factor",)
    factor_name = spec.scope.factors[0]
    if factor_name not in space.param_names:
        return (f"Unknown factor {factor_name!r}",)
    factor = space.param_by_name(factor_name)
    errors: list[str] = []
    if factor.kind not in {"continuous", "integer", "discrete"}:
        errors.append(f"Temperature factor {factor_name!r} must be numeric")
    name_is_temperature = "temp" in factor_name.lower() or "temperature" in factor_name.lower()
    unit_is_temperature = factor.unit in _TEMPERATURE_UNITS and factor.unit != ""
    if not name_is_temperature and not unit_is_temperature:
        errors.append(f"Factor {factor_name!r} is not temperature-compatible")
    if factor.unit not in _TEMPERATURE_UNITS:
        errors.append(f"Unrecognized temperature unit {factor.unit!r}")
    if not errors:
        if factor.bounds is not None:
            low = float(factor.bounds[0])
        else:
            low = float(min(factor.numeric_levels))
        if factor.unit in {"C", "°C", "celsius"}:
            absolute_low = low + 273.15
        elif factor.unit in {"F", "°F", "fahrenheit"}:
            absolute_low = (low - 32.0) * 5.0 / 9.0 + 273.15
        else:
            absolute_low = low
        if absolute_low <= 0:
            errors.append("Arrhenius temperature must be positive in Kelvin")
    return tuple(errors)


def _validate_arrhenius(spec, space, observations=None) -> KnowledgeValidationResult:
    errors = _arrhenius_errors(spec, space)
    empirical_evidence_ready = False
    if not errors and observations is not None:
        successful = observations.successful()
        factor_name = spec.scope.factors[0]
        objectives = spec.scope.objectives or tuple(space.objectives)
        X = successful.X
        Y = successful.Y
        if (
            len(X) >= 4
            and factor_name in X.columns
            and objectives
            and all(objective in Y.columns for objective in objectives)
        ):
            try:
                values = tuple(float(value) for value in X[factor_name]) + tuple(
                    float(value)
                    for objective in objectives
                    for value in Y[objective]
                )
            except (TypeError, ValueError):
                values = ()
            empirical_evidence_ready = (
                bool(values)
                and all(math.isfinite(value) for value in values)
                and len(set(float(value) for value in X[factor_name])) >= 3
            )
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="invalid" if errors else "insufficient_data",
        summary=(
            "Invalid arrhenius declaration"
            if errors
            else (
                "arrhenius is structurally valid; empirical agreement has not been established"
                if empirical_evidence_ready
                else "arrhenius is structurally valid; empirical coverage is insufficient"
            )
        ),
        errors=errors,
        effective_confidence=spec.confidence,
    )


def _compatible_arrhenius(spec, space) -> CompatibilityResult:
    errors = _arrhenius_errors(spec, space)
    return CompatibilityResult(compatible=not errors, reasons=errors)


def arrhenius_definition() -> KnowledgePatternDefinition:
    return KnowledgePatternDefinition(
        pattern="arrhenius",
        version="1.0",
        family="physics",
        schema={
            "type": "object",
            "properties": {
                "frozen": {"type": "boolean"},
                "activation_energy": {"type": "number"},
                "amplitude_init": {"type": "number"},
            },
            "required": ["frozen", "activation_energy", "amplitude_init"],
            "additionalProperties": False,
        },
        compiler=_compile_arrhenius,
        validator=_validate_arrhenius,
        renderer=_render,
        compatibility=_compatible_arrhenius,
    )


def monotone_definition() -> KnowledgePatternDefinition:
    return KnowledgePatternDefinition(
        pattern="monotone",
        version="1.0",
        family="shape",
        schema={
            "type": "object",
            "properties": {
                "direction": {"enum": ["increasing", "decreasing"]},
                "n_pairs_per_dim": {"type": "integer"},
                "epsilon": {
                    "anyOf": [
                        {"type": "number"},
                        {"const": "auto"},
                    ]
                },
                "delta_norm": {"type": "number"},
            },
            "required": [
                "direction",
                "n_pairs_per_dim",
                "epsilon",
                "delta_norm",
            ],
            "additionalProperties": False,
        },
        compiler=_compile_monotone,
        validator=_validate_factor,
        renderer=_render,
        compatibility=_compatible_factor,
    )


__all__ = ["arrhenius_definition", "monotone_definition"]
