from functools import cache
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pandas as pd
import pytest

import expdoe_dk as ed


@cache
def example():
    path = Path(__file__).resolve().parents[2] / "examples" / "04_measurement_workflow.py"
    return SimpleNamespace(**runpy.run_path(str(path)))


def workflow():
    space = ed.Space(params=[ed.Parameter("temperature", bounds=(20, 100))],
                     objectives="yield_pct")
    catalog = pd.DataFrame({"temperature": [35, 50, 75]},
                           index=["formula-a", "formula-b", "formula-c"])
    return example().MeasurementWorkflow(space, catalog)


def measurements(*rows):
    return [example().Measurement(*row) for row in rows]


def test_partial_return_aligns_outcomes_by_trial_id_not_dispatch_order():
    run = workflow()
    run.dispatch("trial-1", "formula-a")
    run.dispatch("trial-2", "formula-b")
    run.dispatch("trial-3", "formula-c")

    run.record_returns(measurements(("trial-3", "success", 91), ("trial-1", "success", 78)))

    assert run.campaign.history_df()[["temperature", "y"]].values.tolist() == [[75, 91], [35, 78]]
    assert run.campaign.to_state()["pending"] == {"ids": ["trial-2"], "X": [[50]]}


@pytest.mark.parametrize("invalid", [
    ("unknown", "success", 99),
    ("formula-b", "success", 99),
    ("trial-1", "success", 99),
    ("trial-2", "success"),
    ("trial-2", "success", float("inf")),
    ("trial-2", "unknown", 99),
], ids=["unknown-trial", "candidate-id", "duplicate", "missing", "nonfinite", "status"])
def test_invalid_return_rejects_entire_batch_without_mutating_either_state(invalid):
    run = workflow()
    run.dispatch("trial-1", "formula-a")
    run.dispatch("trial-2", "formula-b")
    before = run.to_state()

    with pytest.raises(ValueError):
        run.record_returns(measurements(("trial-1", "success", 78), invalid))

    assert run.to_state() == before


@pytest.mark.parametrize("trial_id,candidate_id", [
    ("trial-2", "unknown"), ("trial-1", "formula-b"), ("formula-b", "formula-b"),
])
def test_invalid_dispatch_does_not_replace_pending_or_register_a_trial(trial_id, candidate_id):
    run = workflow()
    run.dispatch("trial-1", "formula-a")
    before = run.to_state()

    with pytest.raises(ValueError):
        run.dispatch(trial_id, candidate_id)

    assert run.to_state() == before


def test_failed_and_warning_returns_are_audited_but_not_training_rows():
    run = workflow()
    run.dispatch("trial-1", "formula-a")
    run.dispatch("trial-2", "formula-b")
    run.dispatch("trial-3", "formula-c")

    run.record_returns(measurements(("trial-1", "failed"), ("trial-2", "warning", 98)))
    state = run.campaign.to_state()
    assert state["observations"][0]["ids"] == ["trial-1", "trial-2"]
    assert state["observations"][0]["status"] == ["failed", "warning"]
    assert state["observations"][0]["Y"] == [[None], [98]]
    assert state["pending"]["ids"] == ["trial-3"]
    assert run.campaign.history_df().empty
    with pytest.raises(ed.EngineError) as error:
        run.campaign.ask()
    assert error.value.code == ed.ErrorCode.INSUFFICIENT_DATA

    run.record_returns(measurements(("trial-3", "success", 91)))
    assert run.campaign.history_df()["y"].tolist() == [91]
    assert run.campaign.to_state()["pending"] is None


def test_repeat_measurement_keeps_formula_but_requires_new_trial_id():
    run = workflow()
    run.dispatch("trial-1", "formula-a")
    run.record_returns(measurements(("trial-1", "success", 78)))
    before = run.to_state()

    with pytest.raises(ValueError):
        run.record_returns(measurements(("trial-1", "success", 82)))
    with pytest.raises(ValueError):
        run.dispatch("trial-1", "formula-a")
    assert run.to_state() == before

    run.dispatch("trial-2", "formula-a")
    run.record_returns(measurements(("trial-2", "success", 82)))
    assert run.to_state()["trial_candidates"] == {"trial-1": "formula-a", "trial-2": "formula-a"}
    assert run.campaign.history_df()[["temperature", "y"]].values.tolist() == [[35, 78], [35, 82]]


def test_cancel_removes_only_that_trial_and_keeps_audit_without_an_outcome():
    run = workflow()
    run.dispatch("trial-1", "formula-a")
    run.dispatch("trial-2", "formula-a")

    run.cancel("trial-1", "instrument unavailable")

    state = run.to_state()
    assert state["cancellations"] == {"trial-1": "instrument unavailable"}
    assert state["trial_candidates"]["trial-1"] == "formula-a"
    assert state["campaign"]["pending"] == {"ids": ["trial-2"], "X": [[35]]}
    assert state["campaign"]["observations"] == []
    with pytest.raises(ValueError):
        run.record_returns(measurements(("trial-1", "success", 78)))
    with pytest.raises(ValueError):
        run.cancel("trial-1", "attempt to overwrite the audit")
    with pytest.raises(ValueError):
        run.cancel("unknown", "not dispatched")
    with pytest.raises(ValueError):
        run.cancel("trial-2", "")
    assert run.to_state() == state


def test_save_and_restore_keep_active_and_terminal_trial_ids(tmp_path):
    run = workflow()
    run.dispatch("trial-1", "formula-a")
    run.dispatch("trial-2", "formula-b")
    run.dispatch("trial-3", "formula-c")
    run.dispatch("trial-4", "formula-c")
    run.record_returns(measurements(("trial-1", "success", 78), ("trial-4", "failed")))
    run.cancel("trial-2", "sample lost before measurement")
    path = tmp_path / "workflow.json"
    run.save(path)

    restored = example().MeasurementWorkflow.load(path)

    assert restored.to_state() == run.to_state()
    with pytest.raises(ValueError):
        restored.dispatch("trial-2", "formula-b")
    with pytest.raises(ValueError):
        restored.record_returns(measurements(("trial-4", "success", 92)))
    restored.dispatch("trial-5", "formula-c")
    restored.record_returns(measurements(("trial-5", "success", 88), ("trial-3", "success", 91)))
    assert restored.campaign.history_df()[["temperature", "y"]].values.tolist() == [[35, 78], [75, 88], [75, 91]]
    assert restored.campaign.to_state()["pending"] is None


def test_saved_measurements_rebuild_same_model_for_next_recommendation(tmp_path):
    run = workflow()
    for trial, candidate, value in [("t1", "formula-a", 78.), ("t2", "formula-b", 83.),
                                     ("t3", "formula-c", 91.)]:
        run.dispatch(trial, candidate)
        run.record_returns(measurements((trial, "success", value)))
    run.dispatch("t4", "formula-b")
    path = tmp_path / "workflow.json"
    run.save(path)
    restored = example().MeasurementWorkflow.load(path)
    expected = run.campaign.ask_with_diagnostics()
    actual = restored.campaign.ask_with_diagnostics()
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    assert actual.diagnostics["pending_count"] == 1
    assert actual.predictions.mean.tolist() == expected.predictions.mean.tolist()


def test_mixed_numeric_ordinal_catalog_and_batches_survive_checkpoint(tmp_path):
    large = 2**53 + 1
    space = ed.Space([ed.Parameter("level", kind="ordinal", values=(.5, large))], objectives="y")
    catalog = pd.DataFrame({"level": pd.Series([.5, large], index=["fraction", "large"], dtype=object)})
    run = example().MeasurementWorkflow(space, catalog)
    run.dispatch("trial-large", "large")
    path = tmp_path / "workflow.json"
    run.save(path)
    restored = example().MeasurementWorkflow.load(path)
    assert restored.to_state() == run.to_state()
    restored.dispatch("trial-fraction", "fraction")
    restored.record_returns(measurements(("trial-large", "success", 2.), ("trial-fraction", "success", 1.)))
    restored.save(path)
    final = example().MeasurementWorkflow.load(path)
    assert final.to_state() == restored.to_state()
    assert final.to_state()["campaign"]["observations"][0]["X"] == [[large], [.5]]
    assert type(final.to_state()["catalog"]["X"][1][0]) is int


@pytest.mark.parametrize("corruption", ["pending_formula", "observed_formula", "cancelled_pending",
                                        "cancelled_observed", "orphan_trial", "unknown_candidate"])
def test_restore_rejects_disagreement_between_trial_audit_and_campaign(tmp_path, corruption):
    run = workflow()
    run.dispatch("t1", "formula-a")
    run.dispatch("t2", "formula-b")
    run.record_returns(measurements(("t1", "success", 78.)))
    state = run.to_state()
    if corruption == "pending_formula":
        state["trial_candidates"]["t2"] = "formula-c"
    elif corruption == "observed_formula":
        state["trial_candidates"]["t1"] = "formula-c"
    elif corruption == "cancelled_pending":
        state["cancellations"]["t2"] = "conflict"
    elif corruption == "cancelled_observed":
        state["cancellations"]["t1"] = "conflict"
    elif corruption == "orphan_trial":
        state["trial_candidates"]["orphan"] = "formula-c"
    else:
        state["trial_candidates"]["t2"] = "unknown"
    path = tmp_path / "inconsistent.json"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        example().MeasurementWorkflow.load(path)


def test_dispatch_refuses_hard_constraint_violation_without_mutation():
    from expdoe_dk.domain import LinearConstraint
    space = ed.Space([ed.Parameter("x", bounds=(0., 10.))], objectives="y",
                     constraints=[LinearConstraint("limit", {"x": 1.}, "<=", 4.)])
    run = example().MeasurementWorkflow(space, pd.DataFrame({"x": [2., 6.]}, index=["safe", "bad"]))
    run.dispatch("t1", "safe")
    before = run.to_state()
    with pytest.raises(ValueError, match="constraint"):
        run.dispatch("t2", "bad")
    assert run.to_state() == before
