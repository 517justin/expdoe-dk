from __future__ import annotations

import pytest

from expdoe_dk.errors import EngineError, ErrorCode
from expdoe_dk.knowledge.artifacts import OptimizationArtifact, OptimizationArtifacts
from expdoe_dk.knowledge.guard import CompatibilityResult, KnowledgeValidationResult
from expdoe_dk.knowledge.registry import KnowledgePatternDefinition, PatternRegistry


def _definition(pattern: str, version: str, calls: list[tuple]) -> KnowledgePatternDefinition:
    def compiler(spec, space, observations=None):
        calls.append(("compile", spec.pattern_id, observations))
        return OptimizationArtifacts(
            diagnostics=(
                OptimizationArtifact(
                    kind="compiled",
                    payload={"pattern_id": spec.pattern_id},
                    source_pattern_id=spec.pattern_id,
                    source_pattern=spec.pattern,
                    source_version=spec.version,
                ),
            )
        )

    def validator(spec, space, observations=None):
        calls.append(("validate", spec.pattern_id, observations))
        return KnowledgeValidationResult(
            pattern_id=spec.pattern_id,
            state="valid",
            summary="valid",
            effective_confidence=spec.confidence,
        )

    def renderer(spec, space):
        calls.append(("render", spec.pattern_id))
        return f"{spec.pattern_id} rendered"

    def compatibility(spec, space):
        return CompatibilityResult(compatible=True)

    return KnowledgePatternDefinition(
        pattern=pattern,
        version=version,
        family="test",
        schema={"type": "object", "properties": {"nested": {"type": "array"}}},
        compiler=compiler,
        validator=validator,
        renderer=renderer,
        compatibility=compatibility,
    )


def test_definition_detaches_and_freezes_schema_and_requires_callables(fake_definition):
    schema = {"properties": {"levels": ["A", "B"]}}
    values = {
        "pattern": "detached",
        "version": "1.0",
        "family": "test",
        "schema": schema,
        "compiler": fake_definition.compiler,
        "validator": fake_definition.validator,
        "renderer": fake_definition.renderer,
        "compatibility": fake_definition.compatibility,
    }
    definition = KnowledgePatternDefinition(**values)
    schema["properties"]["levels"].append("caller mutation")

    assert definition.schema["properties"]["levels"] == ("A", "B")
    with pytest.raises(TypeError):
        definition.schema["new"] = True

    for field in ("compiler", "validator", "renderer", "compatibility"):
        invalid = dict(values)
        invalid[field] = None
        with pytest.raises(TypeError, match=field):
            KnowledgePatternDefinition(**invalid)


def test_make_spec_fixture_supports_pattern_and_scope_overrides(make_spec):
    spec = make_spec(
        "arrhenius",
        factors=("temperature",),
        objectives=("rate",),
        parameters={"temperature_unit": "K"},
    )

    assert spec.pattern == "arrhenius"
    assert spec.pattern_id == "KP-arrhenius"
    assert spec.scope.factors == ("temperature",)
    assert spec.scope.objectives == ("rate",)
    assert spec.parameters["temperature_unit"] == "K"


@pytest.mark.parametrize("field", ["pattern", "version", "family"])
def test_definition_requires_non_empty_identity_fields(fake_definition, field):
    values = {
        "pattern": "fake",
        "version": "1.0",
        "family": "test",
        "schema": {},
        "compiler": fake_definition.compiler,
        "validator": fake_definition.validator,
        "renderer": fake_definition.renderer,
        "compatibility": fake_definition.compatibility,
    }
    values[field] = ""

    with pytest.raises(ValueError, match=field):
        KnowledgePatternDefinition(**values)


def test_registry_rejects_duplicate_pattern_version(fake_definition):
    registry = PatternRegistry()
    registry.register(fake_definition)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(fake_definition)


def test_registry_definitions_are_sorted_independent_of_registration_order():
    calls = []
    registry = PatternRegistry()
    later = _definition("zeta", "1.0", calls)
    earlier_version = _definition("alpha", "2.0", calls)
    earliest = _definition("alpha", "1.0", calls)
    for definition in (later, earlier_version, earliest):
        registry.register(definition)

    assert [(item.pattern, item.version) for item in registry.definitions()] == [
        ("alpha", "1.0"),
        ("alpha", "2.0"),
        ("zeta", "1.0"),
    ]


def test_registry_unknown_exact_resolution_is_a_typed_engine_error():
    registry = PatternRegistry()

    with pytest.raises(EngineError, match="Unknown pattern fake@2.0") as caught:
        registry.resolve("fake", "2.0")

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


def test_registry_validate_validate_many_and_render_dispatch_exactly(
    numeric_space, make_spec
):
    calls = []
    registry = PatternRegistry()
    registry.register(_definition("fake", "1.0", calls))
    first = make_spec(pattern_id="KP-first")
    second = make_spec(pattern_id="KP-second")
    observations = object()

    one = registry.validate(first, numeric_space, observations)
    many = registry.validate_many((first, second), numeric_space, observations)
    rendered = registry.render(second, numeric_space)

    assert one.pattern_id == "KP-first"
    assert [result.pattern_id for result in many] == ["KP-first", "KP-second"]
    assert rendered == "KP-second rendered"
    assert calls == [
        ("validate", "KP-first", observations),
        ("validate", "KP-first", observations),
        ("validate", "KP-second", observations),
        ("render", "KP-second"),
    ]


def test_registry_compile_many_merges_in_spec_order(numeric_space, make_spec):
    calls = []
    registry = PatternRegistry()
    registry.register(_definition("fake", "1.0", calls))
    specs = (make_spec(pattern_id="KP-second"), make_spec(pattern_id="KP-first"))
    observations = object()

    artifacts = registry.compile_many(specs, numeric_space, observations)

    assert [item.source_pattern_id for item in artifacts.diagnostics] == [
        "KP-second",
        "KP-first",
    ]
    assert calls == [
        ("compile", "KP-second", observations),
        ("compile", "KP-first", observations),
    ]


def test_registry_rejects_non_definition_registration():
    registry = PatternRegistry()

    with pytest.raises(TypeError, match="KnowledgePatternDefinition"):
        registry.register(object())
