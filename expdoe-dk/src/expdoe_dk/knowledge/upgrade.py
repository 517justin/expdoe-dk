"""Explicit, non-mutating previews; applying a candidate requires external approval."""
from __future__ import annotations

from expdoe_dk.domain import Space
from expdoe_dk.errors import EngineError, ErrorCode

from .patterns import builtin_pattern_definitions
from .patterns.shape import _v2_is_finite_number
from .registry import PatternRegistry
from .specs import KnowledgePatternSpec


def preview_v2_upgrade(
    spec: KnowledgePatternSpec,
    space: Space,
    *,
    objective: str,
    origin: float | None = None,
    reference: str | None = None,
    response_effect: str | None = None,
) -> dict:
    """Return a detached v2 candidate, never infer semantics from objective utility."""
    if not isinstance(spec, KnowledgePatternSpec):
        raise TypeError("spec must be a KnowledgePatternSpec")
    if spec.version != "1.0" or spec.pattern not in {
        "saturation", "threshold", "synergy", "antagonism"
    }:
        raise ValueError("Only the four supported v1.0 declarations have a v2 upgrade preview")
    if type(objective) is not str or not objective.strip():
        raise ValueError("An explicit objective response is required")
    parameters = spec.to_dict()["parameters"]
    if spec.pattern == "saturation":
        if not _v2_is_finite_number(origin):
            raise ValueError("An explicit finite origin is required")
        if reference is not None or response_effect is not None:
            raise ValueError("Interaction semantics do not belong to saturation")
        parameters["origin"] = origin
    elif spec.pattern == "threshold":
        if any(value is not None for value in (origin, reference, response_effect)):
            raise ValueError("Threshold has no origin or interaction semantic arguments")
    else:
        if origin is not None:
            raise ValueError("origin does not belong to interaction declarations")
        if reference != "anchored_additive":
            raise ValueError("An explicit anchored_additive reference is required")
        if response_effect not in {"higher", "lower"}:
            raise ValueError("An explicit higher or lower response_effect is required")
        parameters.update(reference=reference, response_effect=response_effect)

    original = spec.to_dict()
    candidate_payload = spec.to_dict()
    candidate_payload["version"] = "2.0"
    candidate_payload["parameters"] = parameters
    candidate_payload["scope"]["objectives"] = [objective]
    candidate = KnowledgePatternSpec.from_dict(candidate_payload)
    # Use only built-ins: a preview neither loads providers nor executes a provider
    # selected by a caller-supplied registry. Validate even disabled audit specs.
    registry = PatternRegistry()
    registry.register_many(builtin_pattern_definitions())
    result = registry.validate(candidate, space)
    if result.state == "invalid":
        raise EngineError(ErrorCode.KNOWLEDGE_INVALID, result.summary,
                          details={"pattern_id": candidate.pattern_id, "errors": list(result.errors)})
    return {
        "original": original,
        "candidate": candidate.to_dict(),
        "explanation": registry.render(candidate, space),
        "requires_confirmation": True,
    }


__all__ = ["preview_v2_upgrade"]
