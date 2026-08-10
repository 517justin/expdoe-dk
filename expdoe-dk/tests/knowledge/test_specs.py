from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from expdoe_dk.knowledge.artifacts import (
    OptimizationArtifact,
    OptimizationArtifacts,
    merge_artifacts,
)
from expdoe_dk.knowledge.guard import CompatibilityResult, KnowledgeValidationResult
from expdoe_dk.knowledge.specs import (
    Evidence,
    KnowledgePatternSpec,
    KnowledgeScope,
    make_pattern_spec,
    pattern_id_for,
)


def _serialized_spec() -> dict[str, object]:
    return {
        "pattern_id": "KP-monotone-time",
        "pattern": "monotone",
        "version": "1.0",
        "parameters": {"direction": "increasing"},
        "scope": {
            "factors": ["time"],
            "objectives": ["yield"],
            "conditions": None,
            "region": None,
        },
        "confidence": 0.8,
        "evidence": [
            {"kind": "expert_experience", "reference": "ten prior runs"}
        ],
        "enabled": True,
    }


def test_pattern_spec_round_trip_preserves_provenance():
    spec = KnowledgePatternSpec(
        pattern_id="KP-monotone-time",
        pattern="monotone",
        version="1.0",
        parameters={"direction": "increasing"},
        scope=KnowledgeScope(factors=("time",), objectives=("yield",)),
        confidence=0.8,
        evidence=(
            Evidence(kind="expert_experience", reference="ten prior runs"),
        ),
    )

    assert spec.to_dict() == _serialized_spec()
    assert KnowledgePatternSpec.from_dict(spec.to_dict()) == spec


def test_legacy_spec_derives_the_canonical_pattern_id():
    payload = _serialized_spec()
    del payload["pattern_id"]
    payload["evidence"] = []

    restored = KnowledgePatternSpec.from_dict(payload)

    assert restored.pattern_id == "KP-175ef843f010"
    assert restored.pattern_id == KnowledgePatternSpec.from_dict(payload).pattern_id


def test_present_pattern_id_is_preserved_instead_of_rederived():
    payload = _serialized_spec()
    payload["pattern_id"] = "external-id-with-original-spelling"

    assert (
        KnowledgePatternSpec.from_dict(payload).pattern_id
        == "external-id-with-original-spelling"
    )


def test_present_null_pattern_id_is_rejected_instead_of_treated_as_legacy():
    payload = _serialized_spec()
    payload["pattern_id"] = None

    with pytest.raises(TypeError, match="pattern_id"):
        KnowledgePatternSpec.from_dict(payload)


def test_pattern_id_hash_ignores_mapping_insertion_order_but_not_list_order():
    first = {
        "pattern": "interaction",
        "parameters": {"strength": 2, "factors": ["a", "b"]},
    }
    reordered = {
        "parameters": {"factors": ["a", "b"], "strength": 2},
        "pattern": "interaction",
    }
    reversed_factors = {
        "parameters": {"factors": ["b", "a"], "strength": 2},
        "pattern": "interaction",
    }

    assert pattern_id_for(first) == pattern_id_for(reordered)
    assert pattern_id_for(first) != pattern_id_for(reversed_factors)
    assert pattern_id_for(first).startswith("KP-")
    assert len(pattern_id_for(first)) == 15


def test_make_pattern_spec_derives_only_when_pattern_id_is_omitted():
    values = {
        "pattern": "monotone",
        "version": "1.0",
        "parameters": {"direction": "increasing"},
        "scope": KnowledgeScope(factors=("time",), objectives=("yield",)),
        "confidence": 0.8,
    }

    derived = make_pattern_spec(**values)
    explicit = make_pattern_spec(**values, pattern_id="explicit-ID")

    assert derived.pattern_id == "KP-175ef843f010"
    assert explicit.pattern_id == "explicit-ID"
    with pytest.raises(ValueError, match="pattern_id"):
        make_pattern_spec(**values, pattern_id="")


@pytest.mark.parametrize(
    "field,value",
    [
        ("pattern_id", ""),
        ("pattern", ""),
        ("version", ""),
        ("enabled", 1),
        ("confidence", True),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
    ],
)
def test_pattern_spec_rejects_invalid_envelope_fields(field, value):
    payload = _serialized_spec()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        KnowledgePatternSpec.from_dict(payload)


@pytest.mark.parametrize(
    "bad_value",
    [
        {1: "non-string key"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": 1 + 2j},
        {"value": lambda: None},
        {"value": object()},
        {"value": (1, 2)},
    ],
)
def test_pattern_parameters_reject_non_json_values(bad_value):
    payload = _serialized_spec()
    payload["parameters"] = bad_value

    with pytest.raises((TypeError, ValueError)):
        KnowledgePatternSpec.from_dict(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("enabled"),
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload["scope"].pop("region"),
        lambda payload: payload["scope"].update({"unknown": True}),
        lambda payload: payload["evidence"][0].update({"unknown": True}),
    ],
)
def test_serialized_specs_reject_missing_or_unknown_fields(mutation):
    payload = _serialized_spec()
    mutation(payload)

    with pytest.raises(ValueError, match="missing|unknown"):
        KnowledgePatternSpec.from_dict(payload)


def test_spec_and_scope_detach_and_recursively_freeze_json_inputs():
    nested = {"levels": ["A", {"weight": 2}]}
    conditions = {"catalyst": ["A", "B"]}
    region = {"time": {"lower": 1.0, "upper": 9.0}}
    scope = KnowledgeScope(
        factors=("time",),
        objectives=("yield",),
        conditions=conditions,
        region=region,
    )
    spec = KnowledgePatternSpec(
        pattern_id="KP-detached",
        pattern="fake",
        version="1.0",
        parameters=nested,
        scope=scope,
        confidence=0.5,
    )
    nested["levels"].append("caller mutation")
    conditions["catalyst"].append("caller mutation")
    region["time"]["lower"] = -100

    assert spec.to_dict()["parameters"] == {"levels": ["A", {"weight": 2}]}
    assert spec.scope.to_dict()["conditions"] == {"catalyst": ["A", "B"]}
    assert spec.scope.to_dict()["region"] == {
        "time": {"lower": 1.0, "upper": 9.0}
    }
    with pytest.raises(TypeError):
        spec.parameters["new"] = 1
    with pytest.raises(AttributeError):
        spec.parameters["levels"].append("mutation")

    detached = spec.to_dict()
    detached["parameters"]["levels"].append("serialized mutation")
    assert spec.to_dict()["parameters"] == {"levels": ["A", {"weight": 2}]}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"factors": ("time", "time"), "objectives": ("yield",)},
        {"factors": ("time",), "objectives": ("yield", "yield")},
        {"factors": ("",), "objectives": ("yield",)},
        {"factors": ("time",), "objectives": (1,)},
    ],
)
def test_scope_rejects_invalid_or_duplicate_ordered_names(kwargs):
    with pytest.raises((TypeError, ValueError), match="factor|objective|unique"):
        KnowledgeScope(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "", "reference": "citation"},
        {"kind": "literature", "reference": ""},
        {"kind": lambda: None, "reference": "citation"},
    ],
)
def test_evidence_is_non_executable_non_empty_provenance(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Evidence(**kwargs)


def _artifact(kind: str, sequence: int) -> OptimizationArtifact:
    return OptimizationArtifact(
        kind=kind,
        payload={"sequence": sequence, "nested": [sequence]},
        source_pattern_id=f"KP-{sequence}",
        source_pattern="fake",
        source_version="1.0",
    )


def test_artifact_round_trip_is_detached_and_preserves_provenance():
    payload = {"weights": [1, {"name": "x"}]}
    artifact = OptimizationArtifact(
        kind="mean_component",
        payload=payload,
        source_pattern_id="KP-source",
        source_pattern="monotone",
        source_version="1.0",
    )
    payload["weights"].append(99)

    serialized = artifact.to_dict()
    assert serialized == {
        "kind": "mean_component",
        "payload": {"weights": [1, {"name": "x"}]},
        "source_pattern_id": "KP-source",
        "source_pattern": "monotone",
        "source_version": "1.0",
    }
    assert OptimizationArtifact.from_dict(serialized) == artifact
    serialized["payload"]["weights"].append("mutation")
    assert artifact.to_dict()["payload"] == {"weights": [1, {"name": "x"}]}


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", ""),
        ("source_pattern_id", ""),
        ("source_pattern", ""),
        ("source_version", ""),
    ],
)
def test_artifact_requires_non_empty_provenance_fields(field, value):
    values = _artifact("diagnostic", 1).to_dict()
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        OptimizationArtifact.from_dict(values)


def test_artifact_container_has_exact_ordered_categories_and_round_trips():
    artifacts = OptimizationArtifacts(
        input_transforms=(_artifact("input", 1),),
        diagnostics=(_artifact("diagnostic", 2),),
    )

    assert list(artifacts.to_dict()) == [
        "input_transforms",
        "outcome_transforms",
        "mean_components",
        "kernel_components",
        "priors",
        "virtual_observations",
        "parameter_constraints",
        "outcome_constraints",
        "acquisition_preferences",
        "diagnostics",
    ]
    assert OptimizationArtifacts.from_dict(artifacts.to_dict()) == artifacts


def test_merge_artifacts_concatenates_every_category_in_input_order():
    first = OptimizationArtifacts(
        mean_components=(_artifact("mean", 1),),
        diagnostics=(_artifact("diagnostic", 2),),
    )
    second = OptimizationArtifacts(
        mean_components=(_artifact("mean", 3),),
        diagnostics=(_artifact("diagnostic", 4),),
    )

    merged = merge_artifacts([first, second])

    assert [item.payload["sequence"] for item in merged.mean_components] == [1, 3]
    assert [item.payload["sequence"] for item in merged.diagnostics] == [2, 4]
    assert merged.mean_components[0].source_pattern_id == "KP-1"


def test_artifact_container_and_merge_reject_wrong_types():
    with pytest.raises(TypeError, match="OptimizationArtifact"):
        OptimizationArtifacts(diagnostics=(object(),))
    with pytest.raises(TypeError, match="OptimizationArtifacts"):
        merge_artifacts([OptimizationArtifacts(), object()])


@pytest.mark.parametrize(
    "state,valid",
    [
        ("valid", True),
        ("warning", True),
        ("invalid", False),
        ("insufficient_data", True),
    ],
)
def test_validation_result_valid_property_matches_declared_states(state, valid):
    result = KnowledgeValidationResult(
        pattern_id="KP-result",
        state=state,
        summary="summary",
        errors=["error"],
        warnings=["warning"],
        effective_confidence=0.4,
    )

    assert result.valid is valid
    assert result.errors == ("error",)
    assert result.warnings == ("warning",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state": "unknown"},
        {"effective_confidence": True},
        {"effective_confidence": float("nan")},
        {"effective_confidence": -0.1},
        {"errors": [object()]},
        {"warnings": "warning"},
    ],
)
def test_validation_result_rejects_invalid_state_confidence_or_messages(kwargs):
    values = {"pattern_id": "KP-result", "state": "valid", "summary": "summary"}
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        KnowledgeValidationResult(**values)


def test_compatibility_result_requires_a_bool_and_immutable_string_reasons():
    result = CompatibilityResult(compatible=True, reasons=["numeric only"])

    assert result.reasons == ("numeric only",)
    with pytest.raises(FrozenInstanceError):
        result.compatible = False
    with pytest.raises(TypeError, match="compatible"):
        CompatibilityResult(compatible=1)
    with pytest.raises(TypeError, match="reasons"):
        CompatibilityResult(compatible=True, reasons=[object()])
