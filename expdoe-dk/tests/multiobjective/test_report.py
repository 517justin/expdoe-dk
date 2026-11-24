"""Full-vector measured reports, including offline rendering and raw recipes."""
import base64
import csv
import html
import io
import json
import re

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import (
    Campaign, EngineError, ErrorCode, Knowledge, KnowledgeScope, MultiObjectiveConfig,
    Objective, ObservationBatch, OutcomeConstraint, Parameter, Space, make_pattern_spec,
)
from expdoe_dk.bo.knowledge_runtime import compile_runtime
from expdoe_dk.bo.objective_geometry import ObjectiveGeometry, resolve_geometry


def campaign(*, config=None, knowledge=None, constraints=()):
    space = Space([Parameter("x", bounds=(0., 20.))], objectives=[
        Objective("gain", "maximize", unit="g"), Objective("cost", "minimize", unit="USD")],
        outcome_constraints=constraints)
    return Campaign(space, knowledge=knowledge, multiobjective=config, validate=False)


def tell(c, values, *, ids=None, status=None, variance=None):
    n = len(values)
    c.tell_batch(ObservationBatch(pd.DataFrame({"x": np.arange(n, dtype=float)}),
        pd.DataFrame(values, columns=["gain", "cost"]), ids=ids or (),
        status=None if status is None else pd.Series(status),
        Yvar=None if variance is None else pd.DataFrame(variance, columns=["gain", "cost"])))


def test_multiobjective_result_has_no_scalar_winner():
    c = campaign()
    tell(c, [[1., 1.], [2., 2.], [.5, 3.], [1., 1.]], ids=("a", "b", "dominated", "duplicate"))
    result = c.finalize()
    assert result.best_x_physical is None and result.best_y_physical is None
    assert [r["trial_id"] for r in result.pareto_rows] == ["a", "b", "duplicate"]
    assert len(result.history_df) == 4
    assert result.multiobjective_diagnostics["measurement_basis"] == "observed"
    assert "oracle" not in json.dumps(result.to_dict()).lower()
    assert {"pareto_rows", "hypervolume_history", "multiobjective_diagnostics", "stop_signals"} <= result.to_dict().keys()


@pytest.mark.parametrize("values,scale,hv", [([[1., 2.]], [1., 2.], .01),
    ([[1., 2.], [2., 1.]], [1., 1.], 1.21)])
def test_finalize_before_ask_small_data_geometry_does_not_commit(values, scale, hv):
    c = campaign()
    tell(c, values)
    before = c.to_state()
    result = c.finalize()
    geometry = result.multiobjective_diagnostics["geometry"]
    assert geometry["scale"] == scale
    assert geometry["source_row_count"] == len(values)
    assert result.hypervolume_history == pytest.approx([hv])
    assert result.multiobjective_diagnostics["geometry_scope"] == "report_only"
    assert c.to_state() == before and c._geometry is None
    assert result.stop_signals == []
    with pytest.raises(EngineError) as caught:
        resolve_geometry(c.space, c._combined_observations(), compile_runtime(c.space, c.knowledge, None), c.multiobjective)
    assert caught.value.code == ErrorCode.INSUFFICIENT_DATA
    with pytest.raises(EngineError):
        ObjectiveGeometry.from_dict(geometry)


@pytest.mark.parametrize("mode", ["empty", "failed", "warning", "infeasible"])
def test_no_feasible_measurements_are_unresolved_not_a_fake_winner(mode):
    c = campaign(constraints=[OutcomeConstraint("safe", "cost", "<=", 0.)])
    if mode != "empty":
        tell(c, [[np.nan, np.nan]] if mode in ("failed", "warning") else [[1., 2.]],
             status=[mode] if mode in ("failed", "warning") else None)
    result = c.finalize()
    assert result.pareto_rows == [] and result.hypervolume_history == []
    assert result.multiobjective_diagnostics["hypervolume"] == 0.
    assert result.multiobjective_diagnostics["reference_status"] == "unresolved"
    assert result.multiobjective_diagnostics["geometry"] is None
    assert result.stop_signals == []
    assert "No feasible measurements" in result.to_html_string()


def test_measured_feasibility_and_status_never_use_first_objective_only():
    c = campaign(constraints=[OutcomeConstraint("safe", "cost", "<=", 3.)],
                 knowledge=Knowledge().with_safe_region(expression="x <= 2", factors=("x",)))
    tell(c, [[1., 2.], [5., 4.], [999., -999.], [8., 1.], [np.nan, np.nan]],
         ids=("safe", "unsafe", "warning", "bad-x", "failed"),
         status=["success", "success", "warning", "success", "failed"])
    result = c.finalize()
    assert [r["trial_id"] for r in result.pareto_rows] == ["safe"]
    assert result.multiobjective_diagnostics["feasible_measurements"] == 1
    assert len(result.history_df) == 5


def test_target_objectives_and_soft_knowledge_are_distinct_from_report_hv():
    knowledge = Knowledge().add(make_pattern_spec(pattern_id="soft", pattern="target_range", version="1.0",
        scope=KnowledgeScope(objectives=("gain",)), parameters={"lower": 4., "upper": 6.}, confidence=.8))
    c = campaign(knowledge=knowledge, config=MultiObjectiveConfig(preference_weights={"gain": 9., "cost": 1.}))
    plain = campaign()
    for item in (c, plain):
        tell(item, [[2., 8.], [5., 6.], [8., 4.]])
    result = c.finalize()
    assert result.hypervolume_history == plain.finalize().hypervolume_history
    d = result.multiobjective_diagnostics
    assert d["geometry"]["report_reference"] != d["geometry"]["acquisition_reference"]
    assert d["route"]["kind"] == "q_log_nparego"
    assert d["route"]["reason"] == "preference_override"
    assert d["soft_preferences"][0]["pattern_id"] == "soft"
    assert "Soft preferences" in result.to_html_string()
    target = Campaign(Space([Parameter("x", bounds=(0., 3.))], objectives=[
        Objective("gain", "target", target=(4., 6.), unit="g"), Objective("cost", "minimize")]), validate=False)
    tell(target, [[5., 2.], [7., 1.], [8., 3.]])
    r = target.finalize()
    assert len(r.pareto_rows) == 2
    assert r.multiobjective_diagnostics["objectives"][0]["target"] == [4., 6.]
    assert "target" in r.to_html_string()


def test_fixed_geometry_survives_restore_and_reports_later_measurements():
    c = campaign(config=MultiObjectiveConfig(noise_mode="noiseless"))
    tell(c, [[1., 3.], [2., 2.], [3., 1.]])
    c.ask(candidates=pd.DataFrame({"x": [5., 6., 7.]}), candidate_ids=("a", "b", "c"))
    frozen = c._geometry.to_dict()
    tell(c, [[5., -1.]])
    result = c.finalize()
    assert result.multiobjective_diagnostics["geometry"] == frozen
    assert result.hypervolume_history == pytest.approx([1.21, 4.41])
    restored = Campaign.from_state(c.to_state()).finalize()
    assert restored.to_dict() == result.to_dict()


def test_plateau_requires_five_real_updates_after_frozen_baseline():
    c = campaign(config=MultiObjectiveConfig(noise_mode="noiseless"))
    tell(c, [[1., 3.], [2., 2.], [3., 1.]])
    c.ask(candidates=pd.DataFrame({"x": [5., 6.]}), candidate_ids=("a", "b"))
    for _ in range(5):
        tell(c, [[np.nan, np.nan]], status=["failed"])
    assert c.finalize().stop_signals == []
    for _ in range(4):
        tell(c, [[1., 3.]])
    assert c.finalize().stop_signals == []
    tell(c, [[1., 3.]])
    result = c.finalize()
    assert len(result.hypervolume_history) == 6
    assert result.stop_signals[0]["advisory"] is True
    assert result.stop_signals[0]["stop"] is False
    assert c._ask_iteration == 1


def test_collision_safe_html_csv_preserve_raw_catalog_values_and_escape(tmp_path):
    weird = 'value,"<img src=x onerror=alert(1)>'
    category = 'A,"<script>alert(1)</script>'
    large = 2**60 + 3
    space = Space([Parameter("status", kind="categorical", values=[category, "B"]),
                   Parameter("trial_id", kind="integer", bounds=(large - 1024, large + 1024))],
                  objectives=[Objective("status", "maximize", unit='<b>g</b>'),
                              Objective("kind", "minimize"), Objective(weird, "maximize")])
    c = Campaign(space, validate=False)
    c.tell_batch(ObservationBatch(pd.DataFrame({"status": [category], "trial_id": [large]}),
        pd.DataFrame({"status": [1.], "kind": [2.], weird: [3.]}), ids=('<trial,"id>',)))
    result = c.finalize()
    mapping = result.history_columns
    row = result.pareto_rows[0]
    assert row[mapping["parameters"]["trial_id"]] == large
    assert row[mapping["metadata"]["trial_id"]] == '<trial,"id>'
    assert row[mapping["metadata"]["kind"]] == "doe"
    output = result.to_html_string(title='<script>title</script>')
    assert "<script>" not in output and "<img " not in output and "<b>g" not in output
    for value in (category, weird, '<trial,"id>', '<b>g</b>', str(large)):
        assert html.escape(value) in output
    assert "https://" not in output
    csv_text = base64.b64decode(re.search(r'data:text/csv;base64,([^\"]+)', output).group(1)).decode()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert list(rows[0]) == list(result.history_df.columns)
    assert rows[0][mapping["parameters"]["status"]] == category
    assert rows[0][mapping["parameters"]["trial_id"]] == str(large)
    assert rows[0][weird] == "3.0"
    path = result.to_html(tmp_path / "report.html", title='<script>title</script>')
    assert path.read_text() == output
    result.to_json(tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text())["pareto_rows"][0] == row


def test_unsuccessful_missing_values_serialize_as_strict_json_without_state_changes(tmp_path):
    c = campaign()
    tell(c, [[1., 2.], [np.nan, np.inf], [None, None]], status=["success", "failed", "warning"])
    result = c.finalize()
    payload = result.to_dict()
    assert payload["history_records"][1]["gain"] is None
    assert payload["history_records"][1]["cost"] is None
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    result.to_json(tmp_path / "results.json")
    assert "NaN" not in (tmp_path / "results.json").read_text()
    assert np.isinf(c.history_df().cost.iloc[1])


def test_known_noise_physical_reference_and_detached_diagnostics():
    c = campaign(config=MultiObjectiveConfig(reference_point={"cost": 0., "gain": 0.}))
    tell(c, [[1., -2.], [2., -1.]], variance=[[.1, .2], [.3, .4]])
    result = c.finalize()
    assert result.hypervolume_history == [3.]
    assert result.multiobjective_diagnostics["noise_mode"] == "known"
    assert result.multiobjective_diagnostics["measurement_basis"] == "observed"
    assert result.multiobjective_diagnostics["geometry"]["physical_reference"] == {"cost": 0., "gain": 0.}
    assert result.multiobjective_diagnostics["geometry"]["reference_method"] == "physical"
    payload = result.to_dict()
    payload["multiobjective_diagnostics"]["geometry"]["scale"][0] = 99.
    payload["pareto_rows"][0]["gain"] = 99.
    assert result.multiobjective_diagnostics["geometry"]["scale"] == [1., 1.]
    assert result.pareto_rows[0]["gain"] == 1.


def test_invalid_physical_reference_does_not_silently_move_for_report():
    c = campaign(config=MultiObjectiveConfig(reference_point={"cost": -100., "gain": 100.}))
    tell(c, [[1., 2.]])
    with pytest.raises(EngineError) as caught:
        c.finalize()
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert c._geometry is None


def test_pre_resolution_batches_do_not_masquerade_as_five_fixed_updates():
    c = campaign(config=MultiObjectiveConfig(noise_mode="noiseless"))
    for _ in range(6):
        tell(c, [[1., 2.]])
    assert c.finalize().stop_signals == []
    c.ask(candidates=pd.DataFrame({"x": [5., 6.]}), candidate_ids=("a", "b"))
    assert c.finalize().stop_signals == []


def test_fixed_report_still_works_when_later_variance_cannot_be_fitted():
    c = campaign()
    tell(c, [[1., 3.], [2., 2.], [3., 1.]])
    c.ask(candidates=pd.DataFrame({"x": [5., 6.]}), candidate_ids=("a", "b"))
    tell(c, [[5., -1.]], variance=[[.1, .1]])
    r = c.finalize()
    assert r.hypervolume_history == pytest.approx([1.21, 4.41])
    assert r.multiobjective_diagnostics["route"]["kind"] is None
    assert r.multiobjective_diagnostics["route"]["error_code"] == "CONFIG_INVALID"


def test_frozen_report_rejects_changed_objective_semantics():
    c = campaign(config=MultiObjectiveConfig(noise_mode="noiseless"))
    tell(c, [[1., 3.], [2., 2.], [3., 1.]])
    c.ask(candidates=pd.DataFrame({"x": [5., 6.]}), candidate_ids=("a", "b"))
    c.space = Space(c.space.params, objectives=[Objective("gain", "minimize"), Objective("cost", "minimize")])
    with pytest.raises(EngineError) as caught:
        c.finalize()
    assert caught.value.code == ErrorCode.CONFIG_INVALID
