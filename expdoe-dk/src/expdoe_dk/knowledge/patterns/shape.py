"""Registered optimum-shape and random-augmentation knowledge."""
from __future__ import annotations

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
    factor = spec.scope.factors[0]
    exists = factor in space.param_names
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="valid" if exists else "invalid",
        summary=(f"Factor {factor!r} is available" if exists else f"Unknown factor {factor!r}"),
        errors=() if exists else (f"Unknown factor {factor!r}",),
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
    factor = spec.scope.factors[0]
    compatible = factor in space.param_names
    return CompatibilityResult(
        compatible=compatible,
        reasons=() if compatible else (f"Unknown factor {factor!r}",),
    )


def _compatible(spec, space) -> CompatibilityResult:
    return CompatibilityResult(compatible=True)


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
        family="augmentation",
        schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
            "additionalProperties": False,
        },
        compiler=_compile_random_augment,
        validator=_validate,
        renderer=_render,
        compatibility=_compatible,
    )


__all__ = ["quadratic_peak_definition", "random_augment_definition"]
