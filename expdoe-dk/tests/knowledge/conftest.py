from __future__ import annotations

import pytest

from expdoe_dk.domain import Parameter, Space
from expdoe_dk.knowledge.artifacts import OptimizationArtifacts
from expdoe_dk.knowledge.guard import CompatibilityResult, KnowledgeValidationResult
from expdoe_dk.knowledge.registry import KnowledgePatternDefinition, PatternRegistry
from expdoe_dk.knowledge.specs import KnowledgePatternSpec, KnowledgeScope


@pytest.fixture
def numeric_space() -> Space:
    return Space([Parameter("x", bounds=(0.0, 1.0))], objectives="y")


@pytest.fixture
def fake_definition() -> KnowledgePatternDefinition:
    def compile_pattern(spec, space, observations=None):
        return OptimizationArtifacts()

    def validate_pattern(spec, space, observations=None):
        return KnowledgeValidationResult(
            pattern_id=spec.pattern_id,
            state="valid",
            summary="valid",
            effective_confidence=spec.confidence,
        )

    def render_pattern(spec, space):
        return "fake"

    def check_compatibility(spec, space):
        return CompatibilityResult(compatible=True)

    return KnowledgePatternDefinition(
        pattern="fake",
        version="1.0",
        family="test",
        schema={"type": "object", "properties": {}},
        compiler=compile_pattern,
        validator=validate_pattern,
        renderer=render_pattern,
        compatibility=check_compatibility,
    )


@pytest.fixture
def builtin_registry(fake_definition) -> PatternRegistry:
    registry = PatternRegistry()
    registry.register(fake_definition)
    return registry


@pytest.fixture
def make_spec():
    def factory(
        pattern: str = "fake",
        *,
        factors: tuple[str, ...] = ("x",),
        objectives: tuple[str, ...] = ("y",),
        parameters: dict | None = None,
        scope: KnowledgeScope | None = None,
        pattern_id: str | None = None,
        version: str = "1.0",
        confidence: float = 0.8,
        evidence: tuple = (),
        enabled: bool = True,
        conditions: dict | None = None,
        region: dict | None = None,
    ) -> KnowledgePatternSpec:
        return KnowledgePatternSpec(
            pattern_id=pattern_id if pattern_id is not None else f"KP-{pattern}",
            pattern=pattern,
            version=version,
            parameters={} if parameters is None else parameters,
            scope=(
                scope
                if scope is not None
                else KnowledgeScope(
                    factors=factors,
                    objectives=objectives,
                    conditions=conditions,
                    region=region,
                )
            ),
            confidence=confidence,
            evidence=evidence,
            enabled=enabled,
        )

    return factory
