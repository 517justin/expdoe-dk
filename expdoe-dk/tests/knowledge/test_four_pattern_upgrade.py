"""Upgrade previews require explicit semantics and never edit declarations."""
import pytest

from expdoe_dk import EngineError, ErrorCode, Knowledge, Parameter, Space
from expdoe_dk.knowledge.specs import Evidence, KnowledgePatternSpec, KnowledgeScope, make_pattern_spec


def legacy_spec(pattern, **changes):
    parameters = {"saturation": {"direction": "increasing", "half_response": .5},
                  "threshold": {"threshold": .5, "below_behavior": "increasing", "above_behavior": "flat"},
                  "synergy": {}, "antagonism": {}}[pattern]
    fields = dict(pattern_id="KP-audit", pattern=pattern, version="1.0", parameters=parameters,
                  scope=KnowledgeScope(factors=("x",) if pattern in {"saturation", "threshold"}
                                       else ("x", "z"), conditions={"batch": "A"}),
                  confidence=.7, evidence=(Evidence("interview", "source-123"),), enabled=True)
    fields.update(changes)
    return make_pattern_spec(**fields)


def space():
    return Space([Parameter("x", bounds=(0, 1)), Parameter("z", bounds=(0, 1))], objectives="y")


@pytest.mark.parametrize("pattern,semantics", [("saturation", {"origin": 0}), ("threshold", {}),
    ("synergy", {"reference": "anchored_additive", "response_effect": "higher"}),
    ("antagonism", {"reference": "anchored_additive", "response_effect": "lower"})])
def test_preview_preserves_identity_provenance_and_source(pattern, semantics):
    from expdoe_dk.knowledge.upgrade import preview_v2_upgrade
    original = legacy_spec(pattern)
    before_preview = original.to_dict()
    preview = preview_v2_upgrade(original, space(), objective="y", **semantics)
    assert preview["requires_confirmation"] is True
    assert original.to_dict() == before_preview
    assert preview["original"] == before_preview
    assert preview["candidate"]["version"] == "2.0"
    candidate = KnowledgePatternSpec.from_dict(preview["candidate"])
    assert candidate.pattern_id == "KP-audit"
    assert candidate.evidence == original.evidence
    assert candidate.confidence == .7
    assert candidate.enabled is True
    assert candidate.scope.conditions == original.scope.conditions
    assert candidate.scope.objectives == ("y",)
    assert Knowledge().add(candidate).compile(space()).mean_components
    assert "confirmation" in preview["explanation"].lower()
    preview["original"]["parameters"]["changed"] = True
    preview["candidate"]["parameters"]["changed"] = True
    assert original.to_dict() == before_preview


def test_preview_preserves_disabled_zero_confidence():
    from expdoe_dk.knowledge.upgrade import preview_v2_upgrade
    original = legacy_spec("saturation", confidence=0, enabled=False)
    preview = preview_v2_upgrade(original, space(), objective="y", origin=0)
    restored = Knowledge.from_envelope(Knowledge().add(
        KnowledgePatternSpec.from_dict(preview["candidate"])).to_envelope())
    assert restored.specs[0].enabled is False
    assert restored.specs[0].confidence == 0


@pytest.mark.parametrize("pattern,semantics", [("saturation", {}), ("synergy", {}),
    ("antagonism", {}), ("synergy", {"reference": "anchored_additive"}),
    ("antagonism", {"response_effect": "lower"})])
def test_preview_never_guesses_missing_semantics_from_objective(pattern, semantics):
    from expdoe_dk.knowledge.upgrade import preview_v2_upgrade
    with pytest.raises(ValueError):
        preview_v2_upgrade(legacy_spec(pattern), space(), objective="y", **semantics)


@pytest.mark.parametrize("semantics", [{"objective": ""}, {"objective": "unknown", "origin": 0},
    {"objective": "y", "origin": .1}, {"objective": "y", "origin": float("nan")}])
def test_preview_rejects_invalid_explicit_semantics(semantics):
    from expdoe_dk.knowledge.upgrade import preview_v2_upgrade
    with pytest.raises((ValueError, EngineError)) as caught:
        preview_v2_upgrade(legacy_spec("saturation"), space(), **semantics)
    if isinstance(caught.value, EngineError):
        assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


def test_preview_does_not_downgrade_unknown_or_already_v2_versions():
    from expdoe_dk.knowledge.upgrade import preview_v2_upgrade
    for version in ("2.0", "9.0"):
        with pytest.raises(ValueError):
            preview_v2_upgrade(legacy_spec("threshold", version=version), space(), objective="y")


def test_preview_rejects_semantics_which_do_not_belong_to_pattern():
    from expdoe_dk.knowledge.upgrade import preview_v2_upgrade
    with pytest.raises(ValueError):
        preview_v2_upgrade(legacy_spec("threshold"), space(), objective="y", origin=0)


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_preview_oversized_integer_origin_raises_value_error(value):
    from expdoe_dk.knowledge.upgrade import preview_v2_upgrade
    with pytest.raises(ValueError):
        preview_v2_upgrade(legacy_spec("saturation"), space(), objective="y", origin=value)


@pytest.mark.parametrize("pattern,key", [("saturation", "half_response"), ("threshold", "threshold")])
def test_preview_oversized_integer_source_location_raises_structured_error(pattern, key):
    from expdoe_dk.knowledge.upgrade import preview_v2_upgrade
    parameters = legacy_spec(pattern).to_dict()["parameters"]
    parameters[key] = 10**400
    semantics = {"origin": 0} if pattern == "saturation" else {}
    with pytest.raises(EngineError) as caught:
        preview_v2_upgrade(legacy_spec(pattern, parameters=parameters), space(), objective="y", **semantics)
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID
