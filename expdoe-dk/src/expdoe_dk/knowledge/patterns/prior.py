"""Registered GP hyperparameter-prior knowledge."""
from __future__ import annotations

from typing import Any

from ..artifacts import OptimizationArtifact, OptimizationArtifacts
from ..guard import CompatibilityResult, KnowledgeValidationResult
from ..registry import KnowledgePatternDefinition


GP_PRIOR_PRESETS: dict[str, dict[str, Any]] = {
    "weak": {"ls": (1.5, 0.5), "os": (2.0, 0.5), "noise": (1.0, 50.0)},
    "medium": {"ls": (3.0, 6.0), "os": (3.0, 1.5), "noise": (2.0, 200.0)},
    "strong": {"ls": (6.0, 15.0), "os": (3.0, 1.5), "noise": (3.0, 500.0)},
}


def _compile_gp_prior(spec, space, observations=None) -> OptimizationArtifacts:
    preset = GP_PRIOR_PRESETS[spec.parameters["lengthscale"]]
    artifact = OptimizationArtifact(
        kind="gp_prior",
        payload={
            "lengthscale": list(preset["ls"]),
            "outputscale": list(preset["os"]),
            "noise": list(preset["noise"]),
        },
        source_pattern_id=spec.pattern_id,
        source_pattern=spec.pattern,
        source_version=spec.version,
    )
    return OptimizationArtifacts(priors=(artifact,))


def _validate(spec, space, observations=None) -> KnowledgeValidationResult:
    return KnowledgeValidationResult(
        pattern_id=spec.pattern_id,
        state="valid",
        summary="GP prior preset is valid",
        effective_confidence=spec.confidence,
    )


def _render(spec, space) -> str:
    return f"GP prior ({spec.parameters['lengthscale']})"


def _compatible(spec, space) -> CompatibilityResult:
    return CompatibilityResult(compatible=True)


def gp_prior_definition() -> KnowledgePatternDefinition:
    return KnowledgePatternDefinition(
        pattern="gp_prior",
        version="1.0",
        family="prior",
        schema={
            "type": "object",
            "properties": {
                "lengthscale": {"enum": ["weak", "medium", "strong"]}
            },
            "required": ["lengthscale"],
            "additionalProperties": False,
        },
        compiler=_compile_gp_prior,
        validator=_validate,
        renderer=_render,
        compatibility=_compatible,
    )


__all__ = ["GP_PRIOR_PRESETS", "gp_prior_definition"]
