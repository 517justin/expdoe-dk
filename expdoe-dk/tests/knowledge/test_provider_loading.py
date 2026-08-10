from __future__ import annotations

import importlib
import json
from dataclasses import dataclass

import pytest

from expdoe_dk.errors import EngineError, ErrorCode
from expdoe_dk.knowledge.registry import KnowledgePatternDefinition, PatternRegistry


@dataclass(frozen=True)
class FakeDistribution:
    name: str
    version: str

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name, "Version": self.version}


@dataclass(frozen=True)
class FakeEntryPoint:
    name: str
    value: str
    group: str
    dist: FakeDistribution
    factory: object

    @property
    def module(self) -> str:
        return self.value.partition(":")[0]

    @property
    def attr(self) -> str:
        return self.value.partition(":")[2]

    @property
    def extras(self) -> tuple[str, ...]:
        return ()

    def load(self):
        return self.factory()


def _providers_module():
    try:
        return importlib.import_module("expdoe_dk.knowledge.registry.providers")
    except ModuleNotFoundError:
        pytest.fail("explicit pattern-provider loading is unavailable")


def _definition(
    fake_definition,
    pattern: str,
    *,
    version: str = "1.0",
    schema: dict | None = None,
) -> KnowledgePatternDefinition:
    return KnowledgePatternDefinition(
        pattern=pattern,
        version=version,
        family="provider-test",
        schema=(
            {"type": "object", "properties": {}}
            if schema is None
            else schema
        ),
        compiler=fake_definition.compiler,
        validator=fake_definition.validator,
        renderer=fake_definition.renderer,
        compatibility=fake_definition.compatibility,
    )


def fake_entry_points(loaded: list[str], fake_definition):
    def provider(entry_name: str, pattern: str):
        def load_factory():
            loaded.append(entry_name)
            return lambda: (_definition(fake_definition, pattern),)

        return load_factory

    return (
        FakeEntryPoint(
            name="approved-entry",
            value="approved_package.patterns:provide",
            group="expdoe_dk.knowledge_patterns",
            dist=FakeDistribution("approved-distribution", "1.2.3"),
            factory=provider("approved-entry", "approved-pattern"),
        ),
        FakeEntryPoint(
            name="unapproved-entry",
            value="unapproved_package.patterns:provide",
            group="expdoe_dk.knowledge_patterns",
            dist=FakeDistribution("unapproved-distribution", "9.8.7"),
            factory=provider("unapproved-entry", "unapproved-pattern"),
        ),
    )


def test_provider_loader_imports_only_allowed_distributions(
    monkeypatch, fake_definition
):
    providers = _providers_module()
    loaded: list[str] = []
    monkeypatch.setattr(
        providers,
        "entry_points",
        lambda *, group: fake_entry_points(loaded, fake_definition),
    )
    registry = PatternRegistry()

    report = providers.load_pattern_providers({"approved-distribution"}, registry)

    assert loaded == ["approved-entry"]
    assert registry.resolve("approved-pattern", "1.0").family == "provider-test"
    assert report.providers[0].distribution_name == "approved-distribution"
    assert report.providers[0].entry_point_name == "approved-entry"


def test_provider_loader_rejects_unlisted_name(monkeypatch, fake_definition):
    providers = _providers_module()
    monkeypatch.setattr(
        providers,
        "entry_points",
        lambda *, group: fake_entry_points([], fake_definition),
    )

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers({"not-installed"}, PatternRegistry())

    assert caught.value.code is ErrorCode.EXTENSION_NOT_ALLOWED


def test_missing_distribution_is_rejected_before_any_provider_load(
    monkeypatch, fake_definition
):
    providers = _providers_module()
    loaded: list[str] = []
    monkeypatch.setattr(
        providers,
        "entry_points",
        lambda *, group: fake_entry_points(loaded, fake_definition),
    )

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers(
            {"NOT.installed", "Approved_Distribution"}, PatternRegistry()
        )

    assert loaded == []
    assert caught.value.code is ErrorCode.EXTENSION_NOT_ALLOWED
    assert caught.value.details == {
        "requested_distributions": [
            "approved-distribution",
            "not-installed",
        ],
        "missing_distributions": ["not-installed"],
    }


def test_allowed_names_are_canonicalized_and_selected_once(
    monkeypatch, fake_definition
):
    providers = _providers_module()
    loaded: list[str] = []
    entries = fake_entry_points(loaded, fake_definition)
    approved = FakeEntryPoint(
        name=entries[0].name,
        value=entries[0].value,
        group=entries[0].group,
        dist=FakeDistribution("Approved.Distribution", "1.2.3"),
        factory=entries[0].factory,
    )
    monkeypatch.setattr(providers, "entry_points", lambda *, group: (approved,))
    registry = PatternRegistry()

    report = providers.load_pattern_providers(
        {
            "approved-distribution",
            "APPROVED_distribution",
            "Approved.Distribution",
        },
        registry,
    )

    assert loaded == ["approved-entry"]
    assert len(report.providers) == 1
    assert report.providers[0].distribution_name == "Approved.Distribution"
    assert report.providers[0].canonical_distribution_name == (
        "approved-distribution"
    )


def test_empty_allow_list_loads_nothing(monkeypatch, fake_definition):
    providers = _providers_module()
    loaded: list[str] = []
    monkeypatch.setattr(
        providers,
        "entry_points",
        lambda *, group: fake_entry_points(loaded, fake_definition),
    )
    registry = PatternRegistry()

    report = providers.load_pattern_providers(set(), registry)

    assert loaded == []
    assert registry.definitions() == ()
    assert report.providers == ()


def test_construction_validation_and_compilation_never_discover_providers(
    monkeypatch, fake_definition, make_spec, numeric_space
):
    from expdoe_dk import Knowledge

    providers = _providers_module()

    def unexpected_discovery(*, group):
        raise AssertionError("provider discovery must be explicit")

    monkeypatch.setattr(providers, "entry_points", unexpected_discovery)
    registry = PatternRegistry()
    registry.register(fake_definition)
    spec = make_spec()

    Knowledge()
    registry.validate(spec, numeric_space)
    registry.compile_many((spec,), numeric_space)

    assert registry.resolve("fake", "1.0") is fake_definition


@pytest.mark.parametrize("allowed", [None, "approved-distribution", {None}, {""}])
def test_invalid_allow_list_is_a_typed_configuration_error(
    monkeypatch, allowed
):
    providers = _providers_module()
    monkeypatch.setattr(providers, "entry_points", lambda *, group: ())

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers(allowed, PatternRegistry())

    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_invalid_registry_is_a_typed_configuration_error(monkeypatch):
    providers = _providers_module()
    monkeypatch.setattr(providers, "entry_points", lambda *, group: ())

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers(set(), object())

    assert caught.value.code is ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize(
    "entries",
    [
        (object(),),
        (
            FakeEntryPoint(
                name="entry",
                value="package.patterns:provide",
                group="expdoe_dk.knowledge_patterns",
                dist=FakeDistribution("", "1.0"),
                factory=lambda: lambda: (),
            ),
        ),
        (
            FakeEntryPoint(
                name="",
                value="package.patterns:provide",
                group="expdoe_dk.knowledge_patterns",
                dist=FakeDistribution("package", "1.0"),
                factory=lambda: lambda: (),
            ),
        ),
        (
            FakeEntryPoint(
                name="entry",
                value="package.patterns:provide",
                group="wrong.group",
                dist=FakeDistribution("package", "1.0"),
                factory=lambda: lambda: (),
            ),
        ),
        (
            FakeEntryPoint(
                name="entry",
                value="package.patterns:provide",
                group="expdoe_dk.knowledge_patterns",
                dist=FakeDistribution("package", ""),
                factory=lambda: lambda: (),
            ),
        ),
    ],
)
def test_malformed_entry_point_metadata_is_a_typed_error(monkeypatch, entries):
    providers = _providers_module()
    monkeypatch.setattr(providers, "entry_points", lambda *, group: entries)

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers({"package"}, PatternRegistry())

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert caught.value.details["stage"] == "metadata"


def test_metadata_enumeration_failure_is_a_typed_error(monkeypatch):
    providers = _providers_module()

    def fail_enumeration(*, group):
        raise RuntimeError("metadata backend failed")

    monkeypatch.setattr(providers, "entry_points", fail_enumeration)

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers({"package"}, PatternRegistry())

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert caught.value.details == {
        "stage": "enumeration",
        "error_type": "RuntimeError",
    }


@pytest.mark.parametrize(
    "factory,stage",
    [
        (
            lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
            "load",
        ),
        (lambda: 42, "provider"),
        (
            lambda: lambda: (_ for _ in ()).throw(
                RuntimeError("factory failed")
            ),
            "factory",
        ),
        (lambda: lambda: None, "result"),
    ],
)
def test_provider_failures_are_structured_and_do_not_mutate_registry(
    monkeypatch, fake_definition, factory, stage
):
    providers = _providers_module()
    entry = FakeEntryPoint(
        name="broken-entry",
        value="broken_package.patterns:provide",
        group="expdoe_dk.knowledge_patterns",
        dist=FakeDistribution("broken-package", "3.0"),
        factory=factory,
    )
    registry = PatternRegistry()
    registry.register(_definition(fake_definition, "existing"))

    monkeypatch.setattr(providers, "entry_points", lambda *, group: (entry,))
    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers({"broken-package"}, registry)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert caught.value.details["stage"] == stage
    assert caught.value.details["distribution"] == "broken-package"
    assert caught.value.details["entry_point"] == "broken-entry"
    assert [(item.pattern, item.version) for item in registry.definitions()] == [
        ("existing", "1.0")
    ]


def test_non_definition_values_are_typed_and_registration_is_atomic(
    monkeypatch, fake_definition
):
    providers = _providers_module()
    entries = (
        FakeEntryPoint(
            name="a-valid",
            value="package.patterns:valid",
            group="expdoe_dk.knowledge_patterns",
            dist=FakeDistribution("package", "1.0"),
            factory=lambda: lambda: (_definition(fake_definition, "valid"),),
        ),
        FakeEntryPoint(
            name="b-invalid",
            value="package.patterns:invalid",
            group="expdoe_dk.knowledge_patterns",
            dist=FakeDistribution("package", "1.0"),
            factory=lambda: lambda: ("not-a-definition",),
        ),
    )
    monkeypatch.setattr(providers, "entry_points", lambda *, group: entries)
    registry = PatternRegistry()

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers({"package"}, registry)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert caught.value.details["stage"] == "definition"
    assert registry.definitions() == ()


def test_invalid_definition_schema_does_not_partially_mutate_registry(
    monkeypatch, fake_definition
):
    providers = _providers_module()
    entry = FakeEntryPoint(
        name="schemas",
        value="package.patterns:schemas",
        group="expdoe_dk.knowledge_patterns",
        dist=FakeDistribution("package", "1.0"),
        factory=lambda: lambda: (
            _definition(fake_definition, "valid"),
            _definition(
                fake_definition,
                "invalid",
                schema={"type": "not-a-json-schema-type"},
            ),
        ),
    )
    monkeypatch.setattr(providers, "entry_points", lambda *, group: (entry,))
    registry = PatternRegistry()

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers({"package"}, registry)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert registry.definitions() == ()


def test_duplicate_definitions_across_providers_are_typed_and_atomic(
    monkeypatch, fake_definition
):
    providers = _providers_module()
    entries = tuple(
        FakeEntryPoint(
            name=name,
            value=f"package.patterns:{name}",
            group="expdoe_dk.knowledge_patterns",
            dist=FakeDistribution("package", "1.0"),
            factory=lambda: lambda: (_definition(fake_definition, "duplicate"),),
        )
        for name in ("a-first", "b-second")
    )
    monkeypatch.setattr(providers, "entry_points", lambda *, group: entries)
    registry = PatternRegistry()

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers({"package"}, registry)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert caught.value.details["stage"] == "duplicate"
    assert caught.value.details["pattern"] == "duplicate"
    assert caught.value.details["version"] == "1.0"
    assert registry.definitions() == ()


def test_duplicate_with_existing_registry_is_typed_and_atomic(
    monkeypatch, fake_definition
):
    providers = _providers_module()
    registry = PatternRegistry()
    registry.register(_definition(fake_definition, "existing"))
    entry = FakeEntryPoint(
        name="duplicate",
        value="package.patterns:duplicate",
        group="expdoe_dk.knowledge_patterns",
        dist=FakeDistribution("package", "1.0"),
        factory=lambda: lambda: (
            _definition(fake_definition, "new"),
            _definition(fake_definition, "existing"),
        ),
    )
    monkeypatch.setattr(providers, "entry_points", lambda *, group: (entry,))

    with pytest.raises(EngineError) as caught:
        providers.load_pattern_providers({"package"}, registry)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
    assert caught.value.details["stage"] == "duplicate"
    assert [(item.pattern, item.version) for item in registry.definitions()] == [
        ("existing", "1.0")
    ]


def test_report_is_deterministic_immutable_detached_and_json_serializable(
    monkeypatch, fake_definition
):
    providers = _providers_module()

    def entry(distribution, entry_name, version, definitions):
        return FakeEntryPoint(
            name=entry_name,
            value=f"{distribution}.patterns:{entry_name}",
            group="expdoe_dk.knowledge_patterns",
            dist=FakeDistribution(distribution, version),
            factory=lambda: lambda: definitions,
        )

    entries = (
        entry(
            "Zed_Distribution",
            "z-entry",
            "2.0",
            (
                _definition(fake_definition, "zeta", version="2.0"),
                _definition(fake_definition, "alpha", version="1.0"),
            ),
        ),
        entry(
            "Alpha.Distribution",
            "b-entry",
            "1.1",
            (_definition(fake_definition, "beta"),),
        ),
        entry(
            "Alpha.Distribution",
            "a-entry",
            "1.1",
            (_definition(fake_definition, "gamma"),),
        ),
    )
    monkeypatch.setattr(providers, "entry_points", lambda *, group: entries)

    report = providers.load_pattern_providers(
        {"zed-distribution", "alpha_distribution"}, PatternRegistry()
    )

    assert [item.entry_point_name for item in report.providers] == [
        "a-entry",
        "b-entry",
        "z-entry",
    ]
    assert [
        (item.pattern, item.version)
        for item in report.providers[2].definitions
    ] == [("alpha", "1.0"), ("zeta", "2.0")]
    payload = report.to_dict()
    assert payload == {
        "providers": [
            {
                "distribution_name": "Alpha.Distribution",
                "canonical_distribution_name": "alpha-distribution",
                "entry_point_name": "a-entry",
                "distribution_version": "1.1",
                "definitions": [{"pattern": "gamma", "version": "1.0"}],
            },
            {
                "distribution_name": "Alpha.Distribution",
                "canonical_distribution_name": "alpha-distribution",
                "entry_point_name": "b-entry",
                "distribution_version": "1.1",
                "definitions": [{"pattern": "beta", "version": "1.0"}],
            },
            {
                "distribution_name": "Zed_Distribution",
                "canonical_distribution_name": "zed-distribution",
                "entry_point_name": "z-entry",
                "distribution_version": "2.0",
                "definitions": [
                    {"pattern": "alpha", "version": "1.0"},
                    {"pattern": "zeta", "version": "2.0"},
                ],
            },
        ]
    }
    json.dumps(payload)
    payload["providers"][0]["definitions"][0]["pattern"] = "mutated"
    assert report.providers[0].definitions[0].pattern == "gamma"
    with pytest.raises(AttributeError):
        report.providers = ()


def test_provider_loading_api_is_publicly_exported():
    import expdoe_dk as ed
    from expdoe_dk.knowledge import registry as registry_api

    assert ed.load_pattern_providers is registry_api.load_pattern_providers
    assert ed.PatternRegistry is registry_api.PatternRegistry
    assert ed.ProviderLoadReport is registry_api.ProviderLoadReport


def test_v05_publication_exports_delivered_domain_and_knowledge_apis():
    import expdoe_dk as ed
    from expdoe_dk import doe, domain
    from expdoe_dk.knowledge.artifacts import (
        OptimizationArtifact,
        OptimizationArtifacts,
    )
    from expdoe_dk.knowledge.guard import (
        CompatibilityResult,
        KnowledgeValidationResult,
    )
    from expdoe_dk.knowledge.registry import (
        KnowledgePatternDefinition,
        ProviderDefinitionRecord,
        ProviderRecord,
    )
    from expdoe_dk.knowledge.specs import (
        Evidence,
        KnowledgePatternSpec,
        KnowledgeScope,
        make_pattern_spec,
    )

    expected = {
        "Parameter": domain.Parameter,
        "Objective": domain.Objective,
        "Space": domain.Space,
        "Constraint": domain.Constraint,
        "ExpressionConstraint": domain.ExpressionConstraint,
        "CategoricalCombinationConstraint": domain.CategoricalCombinationConstraint,
        "OutcomeConstraint": domain.OutcomeConstraint,
        "constraint_from_dict": domain.constraint_from_dict,
        "normalize_objectives": domain.normalize_objectives,
        "ObservationBatch": domain.ObservationBatch,
        "PendingBatch": domain.PendingBatch,
        "DesignBatch": doe.DesignBatch,
        "DesignDiagnostics": doe.DesignDiagnostics,
        "compatible_design_methods": doe.compatible_design_methods,
        "suggest_design": doe.suggest_design,
        "Evidence": Evidence,
        "KnowledgePatternSpec": KnowledgePatternSpec,
        "KnowledgeScope": KnowledgeScope,
        "make_pattern_spec": make_pattern_spec,
        "OptimizationArtifact": OptimizationArtifact,
        "OptimizationArtifacts": OptimizationArtifacts,
        "KnowledgeValidationResult": KnowledgeValidationResult,
        "CompatibilityResult": CompatibilityResult,
        "KnowledgePatternDefinition": KnowledgePatternDefinition,
        "ProviderDefinitionRecord": ProviderDefinitionRecord,
        "ProviderRecord": ProviderRecord,
    }
    for name, item in expected.items():
        assert getattr(ed, name) is item
        assert name in ed.__all__
    assert ed.LinearConstraint is not domain.LinearConstraint
    assert "LinearConstraint" in ed.__all__


def test_project_and_runtime_versions_are_atomically_published_as_v05():
    from importlib.metadata import version

    import expdoe_dk as ed
    from expdoe_dk.domain.space import ENGINE_VERSION, SPACE_SCHEMA_VERSION

    assert version("expdoe-dk") == "0.5.0"
    assert ed.__version__ == "0.5.0"
    assert ENGINE_VERSION == ed.__version__
    assert SPACE_SCHEMA_VERSION == "1.0"
    assert ed.ENGINE_VERSION == ENGINE_VERSION
    assert ed.SPACE_SCHEMA_VERSION == SPACE_SCHEMA_VERSION
