"""Reports separate exact directional declarations from measured outcomes."""
import html
import json

import pandas as pd

import pytest

from expdoe_dk import Campaign, ObservationBatch
from .test_campaign import catalog, declaration, directional, initial, replace_directional, validated_scalar


@pytest.mark.parametrize("multi", [False, True])
def test_v2_report_uses_exact_envelope_not_legacy_items(multi):
    name, pattern_id = 'gain<script>alert("x")</script>', 'rule<img src=x onerror=alert(1)>'
    campaign = initial(multi=multi, name=name, pattern_id=pattern_id)
    result = campaign.finalize()
    assert result.knowledge_summary == []
    assert not any("plain GP baseline" in note for note in result.notes)
    report = result.to_html_string()
    assert html.escape(pattern_id) in report and html.escape(name) in report
    assert pattern_id not in report and name not in report
    assert "monotone" in report and "2.0" in report
    assert "fractional evidence weight" in report and "not a calibrated probability" in report
    assert "partial monotonicity" in report and "64" in report
    assert "No ask-time fit diagnostics" in report
    assert campaign.knowledge.to_dict()["items"] == []


def test_empty_multi_v2_report_has_no_manufactured_model_results():
    space, knowledge = declaration(multi=True)
    result = Campaign(space, knowledge).finalize()
    report = result.to_html_string()
    assert result.history_df.empty and result.pareto_rows == []
    assert "No ask-time fit diagnostics" in report
    assert "2.0" in report and "direction" in report


@pytest.mark.parametrize("multi", [False, True])
def test_report_caches_only_actual_ask_fit_and_never_refits_after_tell(multi, monkeypatch):
    from expdoe_dk.bo import recommendation, surrogates
    campaign = initial(multi=multi)
    ask = campaign.ask_with_diagnostics(candidates=catalog()[0], candidate_ids=catalog()[1])
    expected = directional(ask, multi)

    def forbidden(*args, **kwargs):
        raise AssertionError("report/restore must not fit a model")

    monkeypatch.setattr(recommendation, "fit_surrogate", forbidden)
    monkeypatch.setattr(surrogates, "fit_surrogate", forbidden)
    Y = pd.DataFrame({name: [15.] for name in campaign.space.objectives})
    campaign.tell_batch(ObservationBatch(ask.frame, Y, Yvar=Y * 0 + .04))
    result = campaign.finalize()
    snapshot = result.multiobjective_diagnostics["directional"]
    assert snapshot["basis"] == "last_successful_ask"
    assert snapshot["seed"] == 7 and snapshot["observation_count"] == 5
    assert snapshot["objectives"]["gain"] == expected
    assert len(result.history_df) == 6
    assert result.multiobjective_diagnostics["validation"] == campaign.validation_diagnostics
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["multiobjective_diagnostics"]["directional"] == snapshot
    report = result.to_html_string()
    assert "Last successful ask-time fit" in report
    assert "not a post-tell refit" in report and "physical_endpoints" in report and "weights" in report
    expected["weights"].clear()
    snapshot["objectives"]["gain"]["weights"].clear()
    assert campaign.finalize().multiobjective_diagnostics["directional"]["objectives"]["gain"]["weights"]
    restored = Campaign.from_state(campaign.to_state())
    resumed = restored.finalize()
    assert "directional" not in resumed.multiobjective_diagnostics
    assert "No ask-time fit diagnostics" in resumed.to_html_string()


@pytest.mark.parametrize("multi", [False, True])
def test_inactive_report_has_empty_pairs_and_truthful_noise(multi):
    campaign = initial(multi=multi, confidence=0.)
    campaign.ask(candidates=catalog()[0], candidate_ids=catalog()[1])
    result = campaign.finalize()
    detail = result.multiobjective_diagnostics["directional"]["objectives"]["gain"]
    assert detail["status"] == "inactive" and detail["fits"] == []
    assert detail["pair_diagnostics"][0]["physical_endpoints"] == []
    assert "Zero confidence is inactive" in result.to_html_string()


def test_legacy_scalar_result_keeps_exact_json_keyset():
    from expdoe_dk import Knowledge
    campaign = initial()
    campaign.knowledge = Knowledge()
    assert set(campaign.finalize().to_dict()) == {
        "best_x_physical", "best_y_physical", "n_doe", "n_bo", "objectives", "maximize",
        "knowledge_summary", "knowledge", "notes", "param_units", "param_kinds", "history_records"}


def test_zero_declared_variance_report_does_not_claim_fitted_noiselessness():
    from linear_operator.utils.warnings import NumericalWarning
    template = initial()
    batch = template._observation_batches[0]
    campaign = Campaign(template.space, template.knowledge, seed=2, validate=False)
    campaign.tell_batch(ObservationBatch(batch.X, batch.Y, Yvar=batch.Y * 0))
    with pytest.warns(NumericalWarning, match="Very small noise"):
        campaign.ask(candidates=catalog()[0], candidate_ids=catalog()[1])
    result = campaign.finalize()
    d = result.multiobjective_diagnostics["directional"]["objectives"]["gain"]
    assert d["declared_noise_mode"] == "known_zero"
    assert d["normalized_measured_variance"] == [0.] * 5
    assert all(value > 0 for value in d["fitted_likelihood_noise"])
    assert "zero-noise clamping" in result.to_html_string()


def test_scalar_report_does_not_attach_fit_to_changed_declaration():
    campaign = initial()
    campaign.ask(candidates=catalog()[0], candidate_ids=catalog()[1])
    campaign.knowledge = declaration(confidence=.2)[1]
    result = campaign.finalize()
    assert "directional" not in result.multiobjective_diagnostics
    assert "No ask-time fit diagnostics" in result.to_html_string()


def test_empty_scalar_result_reports_no_measurements_or_model_fit():
    from expdoe_dk.bo.loop import Result
    space, knowledge = declaration()
    result = Result(best_x_physical=None, best_y_physical=None,
        history_df=pd.DataFrame(columns=["x", "y", "kind"]), n_doe=0, n_bo=0,
        objectives=space.objectives, maximize=space.maximize, knowledge=knowledge.to_envelope())
    report = result.to_html_string()
    assert "No measured observations" in report and "No ask-time fit diagnostics" in report
    # Keep the pre-existing scalar Campaign API: it cannot choose a best without data.
    with pytest.raises(RuntimeError, match="no data"):
        Campaign(space, knowledge).finalize()


@pytest.mark.parametrize("change", ["direction", "confidence", "disable", "remove", "reid"])
def test_scalar_report_never_attributes_old_empirical_verdict_to_new_rule(change, monkeypatch):
    import expdoe_dk.bo.loop as loop
    from expdoe_dk.bo import recommendation, surrogates
    campaign = validated_scalar()
    old = campaign.finalize()
    replace_directional(campaign, change)

    def forbidden(*args, **kwargs):
        raise AssertionError("finalize/restore must not refit or implicitly revalidate")

    monkeypatch.setattr(loop, "check_monotone_assumption", forbidden)
    monkeypatch.setattr(recommendation, "fit_surrogate", forbidden)
    monkeypatch.setattr(surrogates, "fit_surrogate", forbidden)
    result = campaign.finalize()
    records = result.multiobjective_diagnostics.get("validation", [])
    if change in {"disable", "remove"}:
        assert records == []
    else:
        assert records[0]["status"] == "not_checked" and records[0]["violation"] is None
        assert "not_checked" in result.to_html_string()
        assert '"violation": false' not in html.unescape(result.to_html_string())
    assert old.multiobjective_diagnostics["validation"][0]["violation"] is False
    assert old.knowledge["specs"][0]["parameters"]["direction"] == "increasing"
    restored = Campaign.from_state(campaign.to_state())
    assert restored.finalize().multiobjective_diagnostics == result.multiobjective_diagnostics
