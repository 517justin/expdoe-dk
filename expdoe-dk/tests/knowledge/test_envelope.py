import json
from types import SimpleNamespace

import pandas as pd
import pytest

from expdoe_dk import Campaign, Knowledge, Parameter, Space
from expdoe_dk.errors import EngineError, ErrorCode
from expdoe_dk.knowledge.artifacts import OptimizationArtifacts
from expdoe_dk.knowledge.registry import (
    KnowledgePatternDefinition,
    PatternRegistry,
    load_pattern_providers,
)
from expdoe_dk.knowledge.specs import KnowledgeScope, make_pattern_spec


PROVIDER = {
    "distribution_name": "Acme.Patterns",
    "canonical_distribution_name": "acme-patterns",
    "entry_point_name": "acme-patterns",
    "distribution_version": "2.4.1",
    "definitions": [
        {
            "pattern": "acme_shape",
            "version": "3.0",
            "schema_digest": "a" * 64,
        }
    ],
}

LOADED_PROVIDER = {
    "distribution_name": "Acme.Patterns",
    "canonical_distribution_name": "acme-patterns",
    "entry_point_name": "acme-patterns",
    "distribution_version": "2.4.1",
    "definitions": [
        {
            "pattern": "acme_shape",
            "version": "3.0",
            "schema_digest": (
                "27043154881f68665c7ab61be7a1959f3dbf60965684896a32a68bec66e1dd83"
            ),
        }
    ],
}


def _provider_spec(*, enabled=True):
    return make_pattern_spec(
        pattern="acme_shape",
        version="3.0",
        parameters={"strength": 0.5},
        scope=KnowledgeScope(factors=("x",)),
        confidence=0.7,
        enabled=enabled,
    )


def _knowledge_with_restored_provider(*, enabled=True):
    knowledge = (
        Knowledge()
        .with_safe_region(expression="x >= 0.2", factors=("x",))
        .with_tradeoff(
            "yield", "cost", first_weight=2.0, second_weight=1.0
        )
        .add(_provider_spec(enabled=enabled))
    )
    envelope = knowledge.to_envelope()
    envelope["providers"] = [PROVIDER]
    return Knowledge.from_envelope(envelope)


def _acme_definition(fake_definition, *, schema=None):
    return KnowledgePatternDefinition(
        pattern="acme_shape",
        version="3.0",
        family="provider-test",
        schema=(
            {
                "type": "object",
                "properties": {"strength": {"type": "number"}},
                "required": ["strength"],
                "additionalProperties": False,
            }
            if schema is None
            else schema
        ),
        compiler=fake_definition.compiler,
        validator=fake_definition.validator,
        renderer=fake_definition.renderer,
        compatibility=fake_definition.compatibility,
    )


def _load_acme_provider(
    monkeypatch,
    fake_definition,
    *,
    distribution_name="Acme.Patterns",
    distribution_version="2.4.1",
    schema=None,
):
    from expdoe_dk.knowledge.registry import providers

    definition = _acme_definition(fake_definition, schema=schema)
    entry = SimpleNamespace(
        name="acme-patterns",
        value="acme_patterns:provide",
        group="expdoe_dk.knowledge_patterns",
        dist=SimpleNamespace(
            name=distribution_name,
            version=distribution_version,
        ),
        load=lambda: lambda: (definition,),
    )
    monkeypatch.setattr(providers, "entry_points", lambda *, group: (entry,))
    registry = PatternRegistry()
    report = load_pattern_providers({"acme-patterns"}, registry)
    return registry, report.providers[0]


def _loaded_provider_envelope(monkeypatch, fake_definition):
    registry, _ = _load_acme_provider(monkeypatch, fake_definition)
    return Knowledge(registry=registry).add(_provider_spec()).to_envelope()


def test_versioned_knowledge_envelope_is_bidirectional_and_keeps_audit_specs():
    knowledge = _knowledge_with_restored_provider(enabled=False)
    legacy = knowledge.to_dict()

    envelope = knowledge.to_envelope()
    restored = Knowledge.from_envelope(json.loads(json.dumps(envelope)))

    assert envelope == {
        "schema_version": "1.0",
        "engine_version": "0.5.0",
        "strict": False,
        "specs": [spec.to_dict() for spec in knowledge.specs],
        "providers": [PROVIDER],
    }
    assert [spec.to_dict() for spec in restored.specs] == envelope["specs"]
    assert restored.to_envelope() == envelope
    assert restored.to_dict() == legacy
    assert legacy == {"strict": False, "items": []}


@pytest.mark.parametrize(
    "field,value",
    [("schema_version", "99.0"), ("engine_version", "99.0")],
)
def test_unknown_knowledge_envelope_versions_are_typed(field, value):
    envelope = Knowledge().to_envelope()
    envelope[field] = value

    with pytest.raises(EngineError) as caught:
        Knowledge.from_envelope(envelope)

    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


def test_restored_provider_declaration_requires_explicit_provider_reload():
    knowledge = _knowledge_with_restored_provider(enabled=True)
    space = Space([Parameter("x", bounds=(0.0, 1.0))], objectives="yield")

    with pytest.raises(EngineError) as caught:
        knowledge.compile(space)

    assert caught.value.code is ErrorCode.EXTENSION_NOT_ALLOWED
    assert caught.value.details["missing_patterns"] == ["acme_shape@3.0"]
    assert caught.value.details["required_providers"] == [PROVIDER]
    assert caught.value.hints


def test_manual_same_key_definition_does_not_satisfy_restored_provider_reload(
    monkeypatch, fake_definition
):
    envelope = _loaded_provider_envelope(monkeypatch, fake_definition)
    manual_registry = PatternRegistry()
    manual_registry.register(_acme_definition(fake_definition))
    knowledge = Knowledge.from_envelope(envelope, registry=manual_registry)
    space = Space([Parameter("x", bounds=(0.0, 1.0))], objectives="yield")

    with pytest.raises(EngineError) as caught:
        knowledge.compile(space)

    assert caught.value.code is ErrorCode.EXTENSION_NOT_ALLOWED
    assert caught.value.details["missing_provider_records"] == [LOADED_PROVIDER]


@pytest.mark.parametrize(
    "provider_change",
    [
        {"distribution_version": "2.4.2"},
        {
            "schema": {
                "type": "object",
                "properties": {"strength": {"type": "integer"}},
                "required": ["strength"],
                "additionalProperties": False,
            }
        },
    ],
)
def test_changed_loaded_provider_is_rejected_against_restored_provenance(
    monkeypatch, fake_definition, provider_change
):
    envelope = _loaded_provider_envelope(monkeypatch, fake_definition)
    changed_registry, changed_record = _load_acme_provider(
        monkeypatch, fake_definition, **provider_change
    )
    knowledge = Knowledge.from_envelope(envelope, registry=changed_registry)
    space = Space([Parameter("x", bounds=(0.0, 1.0))], objectives="yield")

    with pytest.raises(EngineError) as caught:
        knowledge.compile(space)

    assert caught.value.code is ErrorCode.EXTENSION_NOT_ALLOWED
    assert caught.value.details["mismatched_provider_records"] == [
        {"expected": LOADED_PROVIDER, "actual": changed_record.to_dict()}
    ]


def test_current_loaded_provider_record_replaces_stale_restored_audit_record(
    monkeypatch, fake_definition
):
    envelope = _loaded_provider_envelope(monkeypatch, fake_definition)
    changed_registry, changed_record = _load_acme_provider(
        monkeypatch,
        fake_definition,
        distribution_version="2.4.2",
    )

    knowledge = Knowledge.from_envelope(envelope, registry=changed_registry)

    assert knowledge.provider_records == (changed_record,)


def test_exact_explicitly_loaded_provider_satisfies_restored_provenance(
    monkeypatch, fake_definition
):
    envelope = _loaded_provider_envelope(monkeypatch, fake_definition)
    registry, expected_record = _load_acme_provider(monkeypatch, fake_definition)
    knowledge = Knowledge.from_envelope(envelope, registry=registry)
    space = Space([Parameter("x", bounds=(0.0, 1.0))], objectives="yield")

    artifacts = knowledge.compile(space)

    assert artifacts == OptimizationArtifacts()
    assert knowledge.provider_records == (expected_record,)


def test_checkpoint_and_result_preserve_versioned_knowledge_and_provider_provenance(
    tmp_path,
):
    knowledge = _knowledge_with_restored_provider(enabled=True)
    space = Space([Parameter("x", bounds=(0.0, 1.0))], objectives="yield")
    campaign = Campaign(space, knowledge=knowledge, seed=17)
    campaign.tell(pd.DataFrame({"x": [0.4]}), [1.5])
    checkpoint = tmp_path / "campaign.json"

    campaign.save_checkpoint(checkpoint)
    payload = json.loads(checkpoint.read_text())
    restored = Campaign.load_checkpoint(checkpoint)
    result_payload = restored.finalize().to_dict()

    assert payload["schema_version"] == "2.0"
    assert payload["engine_version"] == "0.5.0"
    assert payload["knowledge"] == knowledge.to_envelope()
    assert restored.knowledge.to_envelope() == knowledge.to_envelope()
    assert result_payload["knowledge"] == knowledge.to_envelope()
    assert result_payload["knowledge_summary"] == []

    def assert_no_source_keys(value):
        if isinstance(value, dict):
            assert "source_expression" not in value
            assert "expression" not in value
            for child in value.values():
                assert_no_source_keys(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_source_keys(child)

    assert_no_source_keys({"checkpoint": payload, "result": result_payload})


@pytest.mark.parametrize(
    "field,value",
    [("schema_version", "99.0"), ("engine_version", "99.0")],
)
def test_unknown_checkpoint_versions_are_typed(tmp_path, field, value):
    campaign = Campaign(
        Space([Parameter("x", bounds=(0.0, 1.0))]), seed=3
    )
    checkpoint = tmp_path / "campaign.json"
    campaign.save_checkpoint(checkpoint)
    payload = json.loads(checkpoint.read_text())
    payload[field] = value
    checkpoint.write_text(json.dumps(payload))

    with pytest.raises(EngineError) as caught:
        Campaign.load_checkpoint(checkpoint)

    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE
