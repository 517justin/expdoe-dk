"""Four-pattern v2 contracts at the real public numerical boundaries (no mocks)."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import ndtr

from expdoe_dk import (
    Campaign, EngineError, ErrorCode, ExpressionConstraint, Knowledge, KnowledgeScope,
    ObservationBatch, OutcomeConstraint, Parameter, PendingBatch, Space, candidate_keys,
    evaluate_models, fit_surrogate, inspect_capabilities, make_folds, make_pattern_spec, recommend,
)
from expdoe_dk.bo.gp import build_mean_from_artifacts


PATTERNS = ("saturation", "threshold", "synergy", "antagonism")


@pytest.fixture(autouse=True)
def small_matrix_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _spec(case, **changes):
    fields = dict(pattern_id=case["id"], pattern=case["pattern"], version=case["version"],
        parameters=case["parameters"], confidence=.8,
        scope=KnowledgeScope(factors=tuple(case["factors"]), objectives=("response",)))
    fields.update(changes)
    return make_pattern_spec(**fields)


def _case(cases, pattern):
    case = next(case for case in cases if case["pattern"] == pattern)
    space = Space([Parameter("x", bounds=(0., 10.)), Parameter("z", bounds=(0., 1.))],
                  objectives="response")
    X = pd.DataFrame([(x, z) for x in (0., 1., 2., 3., 4., 6., 7., 8., 9., 10.)
                      for z in (0., 1.)], columns=["x", "z"])
    batch = ObservationBatch(X, pd.DataFrame({"response": case["response_values"]}),
                             ids=tuple(f"measured-{i}" for i in range(len(X))))
    return case, space, batch, Knowledge().add_spec(_spec(case))


def _fold_key(indices):
    return json.dumps(list(indices), separators=(",", ":"))


def test_golden_cases_are_exactly_the_four_v2_patterns(golden_v2_cases):
    assert len(golden_v2_cases) == 4
    assert {case["pattern"] for case in golden_v2_cases} == {"saturation", "threshold", "synergy", "antagonism"}
    assert all(case["version"] == "2.0" for case in golden_v2_cases)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_golden_public_consumer_fit_and_physical_prediction(golden_v2_cases, pattern):
    case, space, batch, knowledge = _case(golden_v2_cases, pattern)
    capability = inspect_capabilities(space, knowledge)
    assert capability.applied == (case["id"],)
    assert capability.blocked == ()
    route, = [d for d in capability.diagnostics if d.get("category") == "mean_components"]
    assert route["artifact_kind"] == case["artifact_kind"]
    assert route["consumer"] == case["consumer"]
    assert route["coverage_status"] == route["empirical_evidence"] == "not_evaluated"
    fitted = fit_surrogate(space, batch, knowledge=knowledge, seed=17)
    audit = fitted.diagnostics["structured_mean"]
    assert audit["pattern_ids"] == [case["id"]]
    assert audit["coverage_status"] == "passed"
    assert audit["uncertainty"] == "conditional_on_MAP"
    assert audit["parameter_identifiability"] == "not_certified"
    prediction = fitted.predict(pd.DataFrame([case["probe"]]))
    assert prediction.mean.item() == pytest.approx(case["expected_response"], abs=.2)
    assert np.isfinite(prediction.mean).all() and np.isfinite(prediction.variance).all()
    assert (prediction.variance >= 0).all()
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_golden_malformed_declaration_is_never_applied(golden_v2_cases, pattern):
    case, space, batch, _ = _case(golden_v2_cases, pattern)
    knowledge = Knowledge().add_spec(_spec(case, parameters=case["malformed_parameters"]))
    capability = inspect_capabilities(space, knowledge)
    assert capability.applied == ()
    assert capability.blocked == (case["id"],)
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, batch, knowledge=knowledge)
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern", PATTERNS)
def test_golden_insufficient_rows_fail_despite_wired_consumer(golden_v2_cases, pattern):
    case, space, batch, knowledge = _case(golden_v2_cases, pattern)
    n = case["insufficient_rows"]
    sparse = ObservationBatch(batch.X.iloc[:n], batch.Y.iloc[:n], ids=batch.ids[:n])
    assert inspect_capabilities(space, knowledge).applied == (case["id"],)
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, sparse, knowledge=knowledge)
    assert caught.value.code == ErrorCode.INSUFFICIENT_DATA
    assert caught.value.details["coverage_status"] == "deficient"
    assert caught.value.details["pattern_ids"] == [case["id"]]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_golden_conflicting_declarations_cannot_reach_fitting(golden_v2_cases, pattern):
    case, space, batch, knowledge = _case(golden_v2_cases, pattern)
    knowledge.add_spec(_spec(case, pattern_id=case["conflict_id"]))
    capability = inspect_capabilities(space, knowledge)
    assert capability.applied == ()
    assert set(capability.blocked) == {case["id"], case["conflict_id"]}
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, batch, knowledge=knowledge)
    assert caught.value.code == ErrorCode.KNOWLEDGE_CONFLICT
    assert set(caught.value.details["pattern_ids"]) == {case["id"], case["conflict_id"]}


@pytest.mark.parametrize("pattern", PATTERNS)
def test_evaluation_audits_only_training_fit_and_perturbation_cannot_leak(golden_v2_cases, pattern):
    case, space, batch, knowledge = _case(golden_v2_cases, pattern)
    result = evaluate_models(space, batch, tracks={"v2": knowledge}, seed=17)
    assert result.folds == make_folds(len(batch.X), seed=17)
    rows = {row["track"]: row for row in result.rows}
    assert set(rows) == {"v2", "ridge", "random_forest"}
    expected_keys = {_fold_key(fold) for fold in result.folds}
    for name, row in rows.items():
        assert set(row["fold_diagnostics"]) == expected_keys
        assert row["observation_ids"] == list(batch.ids)
        residual = batch.Y.response.to_numpy() - row["oof_predictions"]
        assert row["rmse"] == pytest.approx(np.sqrt(np.mean(residual**2)))
        assert row["mae"] == pytest.approx(np.mean(abs(residual)))
        assert row["r2"] == pytest.approx(1 - np.sum(residual**2) / np.sum((batch.Y.response - batch.Y.response.mean())**2))
        for fold in result.folds:
            diagnostic = row["fold_diagnostics"][_fold_key(fold)]
            training_indices = [i for i in range(len(batch.X)) if i not in fold]
            training_y = batch.Y.response.iloc[training_indices]
            validation_ids = [batch.ids[i] for i in fold]
            fit_training_ids = diagnostic["training_ids"]
            assert fit_training_ids == [batch.ids[i] for i in training_indices]
            assert set(fit_training_ids).isdisjoint(validation_ids)
            assert diagnostic["validation_ids"] == validation_ids
            assert diagnostic["validation_indices"] == list(fold)
            assert diagnostic["status"] == "fitted"
            assert diagnostic["response_location"] == pytest.approx(training_y.mean())
            assert diagnostic["response_scale"] == pytest.approx(training_y.std(ddof=0))
            if name == "v2":
                audit = diagnostic["fitted_diagnostics"]["structured_mean"]
                assert audit["pattern_ids"] == [case["id"]]
                assert audit["coverage_status"] == "passed"
                assert audit["uncertainty"] == "conditional_on_MAP"
                assert audit["response_location"] == diagnostic["response_location"]
                assert audit["response_scale"] == diagnostic["response_scale"]
            else:
                assert diagnostic["uncertainty"] == "unavailable"
                assert diagnostic["uncertainty_reason"]
                assert diagnostic["fitted_diagnostics"] == {}
    # Only the held-out outcomes of the first fold change. Its fitted parameters,
    # response transform and predictions must be bit-for-bit unchanged.
    validation = list(result.folds[0])
    changed_y = batch.Y.copy()
    changed_y.iloc[validation, 0] += 1000.
    changed = evaluate_models(space, ObservationBatch(batch.X, changed_y, ids=batch.ids),
                              tracks={"v2": knowledge}, seed=17)
    for row in changed.rows:
        before = rows[row["track"]]
        key = _fold_key(validation)
        assert row["fold_diagnostics"][key] == before["fold_diagnostics"][key]
        np.testing.assert_array_equal(np.array(row["oof_predictions"])[validation],
                                      np.array(before["oof_predictions"])[validation])
    json.dumps(result.to_dict(), allow_nan=False)


def test_failed_fold_keeps_typed_coverage_without_claiming_oof_metrics(golden_v2_cases):
    case, space, batch, knowledge = _case(golden_v2_cases, "saturation")
    # The complete six-row data have two levels on each side, but LOO training
    # has only five successes. This is an actual fold-local coverage failure.
    positions = [0, 2, 4, 12, 14, 16]
    data = ObservationBatch(batch.X.iloc[positions], batch.Y.iloc[positions],
                             ids=tuple(batch.ids[i] for i in positions))
    assert fit_surrogate(space, data, knowledge=knowledge).diagnostics["structured_mean"]["coverage_status"] == "passed"
    with pytest.raises(EngineError) as caught:
        evaluate_models(space, data, tracks={"v2": knowledge}, seed=17)
    error = caught.value
    assert error.code == ErrorCode.INSUFFICIENT_DATA
    assert error.details["coverage_status"] == "deficient"
    assert error.details["track"] == "v2" and error.details["fold"] == 0
    diagnostic = error.details["fold_diagnostics"]["v2"]["[0]"]
    assert diagnostic["status"] == "failed"
    assert diagnostic["reason"]["code"] == "INSUFFICIENT_DATA"
    assert diagnostic["reason"]["details"]["coverage_status"] == "deficient"
    assert diagnostic["reason"]["details"]["pattern_ids"] == [case["id"]]
    assert diagnostic["training_ids"] == list(data.ids[1:])
    assert diagnostic["validation_ids"] == [data.ids[0]]
    assert error.details["oof_complete"] is False
    assert not {"rmse", "mae", "r2", "oof_predictions"} & set(error.details)
    json.dumps(error.details, allow_nan=False)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_declarative_checkpoint_refits_reproducibly(golden_v2_cases, pattern, tmp_path):
    _, space, batch, original_knowledge = _case(golden_v2_cases, pattern)
    campaign = Campaign(space, original_knowledge, seed=17, validate=False)
    campaign.tell_batch(batch)
    pending = PendingBatch(("pending",), pd.DataFrame({"x": [5.], "z": [.5]}))
    campaign.set_pending(pending)
    checkpoint = tmp_path / "v2.json"
    campaign.save_checkpoint(checkpoint)
    restored = Campaign.load_checkpoint(checkpoint)
    restored_knowledge = restored.knowledge
    assert restored_knowledge.to_envelope() == original_knowledge.to_envelope()
    assert restored.to_state() == campaign.to_state()
    original_fit = fit_surrogate(space, batch, knowledge=original_knowledge, seed=17)
    restored_fit = fit_surrogate(restored.space, batch, knowledge=restored_knowledge, seed=17)
    assert restored_fit.diagnostics == original_fit.diagnostics
    probe = pd.DataFrame({"x": [2.5, 5., 7.5], "z": [.25, .5, .75]})
    np.testing.assert_array_equal(restored_fit.predict(probe).mean, original_fit.predict(probe).mean)
    np.testing.assert_array_equal(restored_fit.predict(probe).variance, original_fit.predict(probe).variance)
    # No estimators/optimizer state are persisted; successful reproducibility
    # comes from refitting the declarations and physical rows with the same seed.
    state = json.loads(checkpoint.read_text())
    assert set(state) == {"schema_version", "engine_version", "space", "knowledge", "seed",
        "trial_kind", "observations", "pending", "rng_state", "ask_iteration", "validation"}
    assert set(state["knowledge"]) == {"schema_version", "engine_version", "strict", "specs", "providers"}


def test_v1_checkpoint_envelope_remains_equal_without_automatic_upgrade(tmp_path):
    space = Space([Parameter("x", bounds=(0., 10.)), Parameter("z", bounds=(0., 1.))], objectives="response")
    original_knowledge = (Knowledge().with_saturation("x", direction="increasing", half_response=5.)
        .with_threshold("z", threshold=.5, below_behavior="increasing", above_behavior="flat")
        .with_interaction("x", "z", kind="antagonism"))
    campaign = Campaign(space, original_knowledge, seed=17, validate=False)
    checkpoint = tmp_path / "legacy-declarations.json"
    campaign.save_checkpoint(checkpoint)
    restored_knowledge = Campaign.load_checkpoint(checkpoint).knowledge
    assert restored_knowledge.to_envelope() == original_knowledge.to_envelope()
    for restored_v1_spec in restored_knowledge.specs:
        assert restored_v1_spec.version == "1.0"


@pytest.mark.parametrize("pattern", PATTERNS)
def test_changed_bounds_reject_stale_compiled_coordinates(golden_v2_cases, pattern):
    _, space, _, knowledge = _case(golden_v2_cases, pattern)
    artifacts = knowledge.compile(space).mean_components
    changed = Space([Parameter("x", bounds=(0., 20.)), Parameter("z", bounds=(0., 1.))], objectives="response")
    with pytest.raises(EngineError) as caught:
        build_mean_from_artifacts(changed, artifacts, response_location=4., response_scale=1.)
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID


def _provider_record():
    return {"distribution_name": "never-auto-load", "canonical_distribution_name": "never-auto-load",
        "entry_point_name": "forbidden", "distribution_version": "1.0",
        "definitions": [{"pattern": "external-shape", "version": "2.0", "schema_digest": "a" * 64}]}


@pytest.mark.parametrize("tamper", ["digest", "canonical_name", "executable_field"])
def test_checkpoint_rejects_tampered_provider_records(golden_v2_cases, tamper):
    _, space, batch, knowledge = _case(golden_v2_cases, "saturation")
    campaign = Campaign(space, knowledge, validate=False)
    campaign.tell_batch(batch)
    state = campaign.to_state()
    record = _provider_record()
    if tamper == "digest":
        record["definitions"][0]["schema_digest"] = "not-a-sha256"
    elif tamper == "canonical_name":
        record["canonical_distribution_name"] = "different-distribution"
    else:
        record["module"] = "os.system"
    state["knowledge"]["providers"] = [record]
    with pytest.raises(EngineError) as caught:
        Campaign.from_state(state)
    assert caught.value.code == ErrorCode.CHECKPOINT_INCOMPATIBLE


def test_restore_never_auto_loads_external_consumers(golden_v2_cases):
    _, space, batch, knowledge = _case(golden_v2_cases, "saturation")
    envelope = knowledge.to_envelope()
    envelope["providers"] = [_provider_record()]
    envelope["specs"].append(make_pattern_spec(pattern_id="external", pattern="external-shape",
        version="2.0", parameters={}, scope=KnowledgeScope(), confidence=.8).to_dict())
    restored = Knowledge.from_envelope(deepcopy(envelope))
    assert restored.to_envelope() == envelope
    assert restored.registry.provider_records == ()
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, batch, knowledge=restored)
    assert caught.value.code == ErrorCode.EXTENSION_NOT_ALLOWED
    assert caught.value.details["required_providers"] == [_provider_record()]
    campaign = Campaign(space, knowledge, validate=False)
    campaign.tell_batch(batch)
    state = campaign.to_state()
    state["knowledge"] = envelope
    with pytest.raises(EngineError) as caught:
        Campaign.from_state(state)
    assert caught.value.code == ErrorCode.CHECKPOINT_INCOMPATIBLE


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.slow
def test_mixed_recommendation_preserves_hard_safety_pending_and_seed(golden_v2_cases, pattern, tmp_path):
    case, _, batch, knowledge = _case(golden_v2_cases, pattern)
    # x is logarithmic physical concentration; all other domain kinds remain
    # exact covariates, not declarations about categorical/ordinal effects.
    X = batch.X.assign(x=batch.X.x + 1., count=[0, 1] * 10,
                      dose=[1., 3., 3., 1.] * 5, material=["A", "B", "B", "A"] * 5)
    params = deepcopy(case["parameters"])
    if pattern == "saturation":
        params.update(origin=1., half_response=6.)
    elif pattern == "threshold":
        params["threshold"] = 6.
    knowledge = Knowledge().add_spec(_spec(case, parameters=params))
    space = Space([Parameter("x", bounds=(1., 11.), transform="log"),
        Parameter("z", kind="discrete", values=(0., 1.)),
        Parameter("count", kind="integer", bounds=(0, 1)),
        Parameter("dose", kind="discrete", values=(1., 3.)),
        Parameter("material", kind="categorical", values=("A", "B"))], objectives="response",
        constraints=(ExpressionConstraint("hard-dose", "x <= 10"),),
        outcome_constraints=(OutcomeConstraint("safe-response", "response", "<=", 5.5),))
    data = ObservationBatch(X, batch.Y, ids=batch.ids)
    pending = PendingBatch(("pending",), pd.DataFrame({"x": [5.], "z": [1.], "count": [0],
                                                      "dose": [1.], "material": ["A"]}))
    assert inspect_capabilities(space, knowledge).applied == (case["id"],)
    first = recommend(space, data, knowledge=knowledge, pending=pending, q=2, seed=17)
    pending_and_observed_keys = candidate_keys(X, space) | candidate_keys(pending.X, space)
    recommended_keys = candidate_keys(first.frame, space)
    assert pending_and_observed_keys.isdisjoint(recommended_keys)
    assert len(recommended_keys) == 2
    assert space.feasibility_mask(first.frame).all()
    assert (first.frame.x <= 10.).all()
    probability = ndtr((5.5 - first.predictions.mean[:, 0]) / np.sqrt(first.predictions.variance[:, 0]))
    safety = first.diagnostics["outcome_safety"]
    safety_satisfied_for_every_recommended_candidate = bool((probability >= .95).all())
    assert safety_satisfied_for_every_recommended_candidate
    np.testing.assert_allclose(safety["probabilities"], probability)
    assert safety["guaranteed"] is False and safety["basis"] == "model_based_feasibility"
    assert first.diagnostics["pending_count"] == 1
    campaign = Campaign(space, knowledge, seed=17, validate=False)
    campaign.tell_batch(data)
    campaign.set_pending(pending)
    checkpoint = tmp_path / "mixed-v2.json"
    campaign.save_checkpoint(checkpoint)
    restored = Campaign.load_checkpoint(checkpoint)
    # Explicit iteration zero makes Campaign's seed exactly 17 as above.
    second = restored.ask_with_diagnostics(q=2, iteration=0)
    pd.testing.assert_frame_equal(second.frame, first.frame)
    np.testing.assert_array_equal(second.predictions.mean, first.predictions.mean)
    np.testing.assert_array_equal(second.predictions.variance, first.predictions.variance)
    assert second.diagnostics == first.diagnostics
    json.dumps(first.to_dict(), allow_nan=False)


@pytest.mark.parametrize("transform", ["linear", "log"])
def test_v2_never_weakens_unrepresentable_integer_recommendation_rejection(golden_v2_cases, transform):
    _, _, batch, knowledge = _case(golden_v2_cases, "saturation")
    low, high = (0, 2**54 + 20) if transform == "linear" else (10**14, 2351799813685247)
    space = Space([Parameter("x", bounds=(0., 10.)), Parameter("z", bounds=(0., 1.)),
        Parameter("count", kind="integer", bounds=(low, high), transform=transform)], objectives="response")
    data = ObservationBatch(batch.X.assign(count=[low, high] * 10), batch.Y, ids=batch.ids)
    with pytest.raises(EngineError) as caught:
        recommend(space, data, knowledge=knowledge, seed=17)
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert caught.value.details["parameter"] == "count"
