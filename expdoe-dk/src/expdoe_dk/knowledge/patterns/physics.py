"""Registered physical-law and monotonicity knowledge."""
from __future__ import annotations

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


def _validate(spec, space, observations=None) -> KnowledgeValidationResult:
    factor = spec.scope.factors[0]
    exists = factor in space.param_names
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="valid" if exists else "invalid",
        summary=(f"Factor {factor!r} is available" if exists else f"Unknown factor {factor!r}"),
        errors=() if exists else (f"Unknown factor {factor!r}",),
        effective_confidence=spec.confidence,
    )


def _render(spec, space) -> str:
    return f"{spec.pattern} on {spec.scope.factors[0]}"


def _compatible(spec, space) -> CompatibilityResult:
    factor = spec.scope.factors[0]
    compatible = factor in space.param_names
    return CompatibilityResult(
        compatible=compatible,
        reasons=() if compatible else (f"Unknown factor {factor!r}",),
    )


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
        validator=_validate,
        renderer=_render,
        compatibility=_compatible,
    )


def monotone_definition() -> KnowledgePatternDefinition:
    return KnowledgePatternDefinition(
        pattern="monotone",
        version="1.0",
        family="physics",
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
        validator=_validate,
        renderer=_render,
        compatibility=_compatible,
    )


__all__ = ["arrhenius_definition", "monotone_definition"]
