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


def _provider_definition(fake_definition, **overrides) -> KnowledgePatternDefinition:
    values = {
        "pattern": fake_definition.pattern,
        "version": fake_definition.version,
        "family": fake_definition.family,
        "schema": {"type": "object", "properties": {}},
        "compiler": fake_definition.compiler,
        "validator": fake_definition.validator,
        "renderer": fake_definition.renderer,
        "compatibility": fake_definition.compatibility,
    }
    values.update(overrides)
    return KnowledgePatternDefinition(**values)


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


def test_registry_rejects_validation_result_for_a_different_pattern_id(
    fake_definition, numeric_space, make_spec
):
    def wrong_validator(spec, space, observations=None):
        return KnowledgeValidationResult(
            pattern_id="KP-another-declaration",
            state="valid",
            summary="valid",
            effective_confidence=spec.confidence,
        )

    registry = PatternRegistry()
    registry.register(_provider_definition(fake_definition, validator=wrong_validator))
    spec = make_spec(pattern_id="KP-dispatched")

    with pytest.raises(EngineError, match="provenance mismatch") as caught:
        registry.validate(spec, numeric_space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert caught.value.details["expected"] == {"pattern_id": "KP-dispatched"}
    assert caught.value.details["actual"] == {
        "pattern_id": "KP-another-declaration"
    }


@pytest.mark.parametrize(
    "category,override,actual",
    [
        (
            "input_transforms",
            {"source_pattern_id": "KP-another-declaration"},
            {
                "source_pattern_id": "KP-another-declaration",
                "source_pattern": "fake",
                "source_version": "1.0",
            },
        ),
        (
            "diagnostics",
            {"source_pattern": "another-pattern"},
            {
                "source_pattern_id": "KP-dispatched",
                "source_pattern": "another-pattern",
                "source_version": "1.0",
            },
        ),
        (
            "acquisition_preferences",
            {"source_version": "2.0"},
            {
                "source_pattern_id": "KP-dispatched",
                "source_pattern": "fake",
                "source_version": "2.0",
            },
        ),
    ],
)
def test_registry_rejects_artifact_provenance_mismatch_before_merge(
    fake_definition,
    numeric_space,
    make_spec,
    category,
    override,
    actual,
):
    def wrong_compiler(spec, space, observations=None):
        provenance = {
            "source_pattern_id": spec.pattern_id,
            "source_pattern": spec.pattern,
            "source_version": spec.version,
        }
        provenance.update(override)
        artifact = OptimizationArtifact(
            kind="compiled",
            payload={},
            **provenance,
        )
        return OptimizationArtifacts(**{category: (artifact,)})

    registry = PatternRegistry()
    registry.register(_provider_definition(fake_definition, compiler=wrong_compiler))
    spec = make_spec(pattern_id="KP-dispatched")

    with pytest.raises(EngineError, match="provenance mismatch") as caught:
        registry.compile_many((spec,), numeric_space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert caught.value.details["category"] == category
    assert caught.value.details["artifact_index"] == 0
    assert caught.value.details["expected"] == {
        "source_pattern_id": "KP-dispatched",
        "source_pattern": "fake",
        "source_version": "1.0",
    }
    assert caught.value.details["actual"] == actual


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
        ("validate", "KP-second", observations),
        ("compile", "KP-second", observations),
        ("validate", "KP-first", observations),
        ("compile", "KP-first", observations),
    ]


def test_disabled_specs_never_resolve_validate_or_compile(numeric_space, make_spec):
    """Catches disabled audit declarations reaching any executable registry hook."""
    registry = PatternRegistry()
    disabled = make_spec(
        pattern="provider-not-loaded",
        version="9.0",
        pattern_id="KP-disabled",
        enabled=False,
    )

    assert registry.validate_many((disabled,), numeric_space) == ()
    assert registry.compile_many((disabled,), numeric_space) == OptimizationArtifacts()


def test_compile_gates_compatibility_and_validation_before_compiler(
    fake_definition, numeric_space, make_spec
):
    """Catches structurally incompatible or invalid declarations reaching compilers."""
    calls: list[str] = []

    def compatibility(spec, space):
        calls.append("compatibility")
        return CompatibilityResult(compatible=False, reasons=("wrong factor kind",))

    def validator(spec, space, observations=None):
        calls.append("validator")
        return KnowledgeValidationResult(
            pattern_id=spec.pattern_id, state="valid", summary="valid"
        )

    def compiler(spec, space, observations=None):
        calls.append("compiler")
        return OptimizationArtifacts()

    registry = PatternRegistry()
    registry.register(
        _provider_definition(
            fake_definition,
            compatibility=compatibility,
            validator=validator,
            compiler=compiler,
        )
    )

    with pytest.raises(EngineError, match="incompatible") as caught:
        registry.compile_many((make_spec(),), numeric_space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert calls == ["compatibility"]


def test_invalid_validation_blocks_compiler_but_insufficient_data_compiles_with_diagnostic(
    fake_definition, numeric_space, make_spec
):
    """Catches invalid and insufficient-data states being treated identically."""
    calls: list[tuple[str, str]] = []

    def compatibility(spec, space):
        calls.append((spec.pattern, "compatibility"))
        return CompatibilityResult(compatible=True)

    def validator(spec, space, observations=None):
        calls.append((spec.pattern, "validator"))
        state = "invalid" if spec.pattern == "invalid-pattern" else "insufficient_data"
        return KnowledgeValidationResult(
            pattern_id=spec.pattern_id,
            state=state,
            summary=f"{state} summary",
            errors=("structural contradiction",) if state == "invalid" else (),
            warnings=("more observations required",) if state == "insufficient_data" else (),
            effective_confidence=spec.confidence,
        )

    def compiler(spec, space, observations=None):
        calls.append((spec.pattern, "compiler"))
        return OptimizationArtifacts()

    registry = PatternRegistry()
    for pattern in ("invalid-pattern", "insufficient-pattern"):
        registry.register(
            _provider_definition(
                fake_definition,
                pattern=pattern,
                compatibility=compatibility,
                validator=validator,
                compiler=compiler,
            )
        )

    invalid = make_spec(pattern="invalid-pattern", pattern_id="KP-invalid")
    insufficient = make_spec(
        pattern="insufficient-pattern", pattern_id="KP-insufficient"
    )

    with pytest.raises(EngineError, match="invalid") as caught:
        registry.compile_many((invalid,), numeric_space)
    artifacts = registry.compile_many((insufficient,), numeric_space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert ("invalid-pattern", "compiler") not in calls
    assert ("insufficient-pattern", "compiler") in calls
    assert [item.payload["state"] for item in artifacts.diagnostics] == [
        "insufficient_data"
    ]


def test_registry_rejects_non_definition_registration():
    registry = PatternRegistry()

    with pytest.raises(TypeError, match="KnowledgePatternDefinition"):
        registry.register(object())
