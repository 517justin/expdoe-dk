import numpy as np
import pandas as pd
import pytest

from expdoe_dk import (
    CapabilityReport, EvaluationResult, Knowledge, Prediction, RecommendationResult,
    inspect_capabilities,
)
from expdoe_dk.knowledge.artifacts import ARTIFACT_CATEGORIES
from expdoe_dk.knowledge.specs import KnowledgeScope, make_pattern_spec


def test_registry_and_consumer_coverage(pattern_cases):
    assert len(pattern_cases) == 23
    for space, knowledge in pattern_cases:
        restored = Knowledge.from_envelope(knowledge.to_envelope())
        report = inspect_capabilities(space, restored)
        assert set(report.applied).isdisjoint(report.blocked)
        assert set(report.applied) | set(report.blocked) == set(
            p["pattern_id"] for p in knowledge.to_envelope()["specs"]
            if p["enabled"]
        )


def test_unknown_consumer_blocks(unsupported_artifact_case):
    space, knowledge = unsupported_artifact_case
    report = inspect_capabilities(space, knowledge)
    assert report.blocked
    assert any(d["reason"] == "unsupported_artifact" for d in report.diagnostics)


def test_golden_cases_match_compiler_kinds_and_schema(pattern_records, pattern_cases):
    assert len({record["pattern"] for record in pattern_records}) == 23
    for record, (space, knowledge) in zip(pattern_records, pattern_cases, strict=True):
        spec = knowledge.specs[0]
        definition = knowledge.registry.resolve(spec.pattern, spec.version)
        knowledge.registry._validate_parameters(spec, definition)
        assert definition.compatibility(spec, space).compatible
        assert knowledge.registry.validate(spec, space).state != "invalid"
        compiled = definition.compiler(spec, space, None)
        kinds = [item.kind for category in ARTIFACT_CATEGORIES
                 for item in getattr(compiled, category)]
        assert kinds == record["expected_artifact_kinds"]


def test_every_emitted_kind_has_explicit_consumer_or_block(pattern_cases):
    for space, knowledge in pattern_cases:
        report = inspect_capabilities(space, knowledge)
        entries = [d for d in report.diagnostics if d.get("artifact_kind")]
        if len(space.objectives) > 1:
            assert any(d.get("reason") == "multi_objective_unsupported"
                       for d in report.diagnostics)
            continue
        assert entries
        assert all(d.get("consumer") or d.get("reason") == "unsupported_artifact"
                   for d in entries)
        assert all(d.get("reason") != "skipped" for d in entries)


def test_active_multiobjective_blocks_but_disabled_storage_restores(pattern_cases):
    multi_space, active = pattern_cases[-1]
    report = inspect_capabilities(multi_space, active)
    assert report.blocked == (active.specs[0].pattern_id,)
    assert any(d["reason"] == "multi_objective_unsupported" for d in report.diagnostics)
    original = active.specs[0]
    disabled = Knowledge().add_spec(make_pattern_spec(
        pattern_id=original.pattern_id, pattern=original.pattern,
        version=original.version, parameters=original.to_dict()["parameters"],
        scope=KnowledgeScope(objectives=original.scope.objectives), enabled=False,
        confidence=original.confidence,
    ))
    restored = Knowledge.from_envelope(disabled.to_envelope())
    assert restored.specs[0].enabled is False
    assert inspect_capabilities(multi_space, restored).blocked == ()


def test_hard_safety_never_skipped_for_insufficient_data(pattern_cases):
    for space, knowledge in pattern_cases:
        if knowledge.specs[0].pattern not in {"safe_region", "forbidden_region"}:
            continue
        report = inspect_capabilities(space, knowledge)
        assert knowledge.specs[0].pattern_id in set(report.applied) | set(report.blocked)
        assert not any(d.get("reason") == "skipped" for d in report.diagnostics)


def test_contracts_copy_mutable_inputs():
    diagnostics = [{"artifact_kind": "gp_prior", "nested": [1]}]
    report = CapabilityReport(("KP-1",), (), tuple(diagnostics))
    diagnostics[0]["nested"].append(2)
    assert report.diagnostics[0]["nested"] == [1]
    mean = np.array([[1.0]])
    variance = np.array([[0.1]])
    prediction = Prediction(mean, variance)
    mean[0, 0] = 99
    assert prediction.mean[0, 0] == 1.0
    frame = pd.DataFrame({"x": [1.0]})
    result = RecommendationResult(frame, prediction, {"candidate_keys": ["x"]})
    frame.loc[0, "x"] = 99
    assert result.frame.loc[0, "x"] == 1.0
    evaluation = EvaluationResult(((0,),), ({"track": "plain"},), ())
    assert evaluation.to_dict()["rows"][0]["track"] == "plain"


@pytest.mark.parametrize("mean,variance", [
    (np.array([1.0]), np.array([[0.1]])),
    (np.array([[1.0]]), np.array([[-0.1]])),
    (np.array([[np.nan]]), np.array([[0.1]])),
])
def test_prediction_rejects_invalid_arrays(mean, variance):
    with pytest.raises((TypeError, ValueError)):
        Prediction(mean, variance)
