"""V2 declarations and empirical checks survive strict legacy-compatible codecs."""
from copy import deepcopy
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import Campaign, EngineError, ErrorCode, Knowledge, ObservationBatch
from expdoe_dk.bo.checkpoint import validate_state
from .test_campaign import catalog, declaration, directional, initial


@pytest.mark.parametrize("multi", [False, True])
def test_v2_resume_same_next_batch(multi, tmp_path):
    campaign = initial(multi=multi)
    campaign.ask(q=2, candidates=catalog()[0], candidate_ids=catalog()[1])
    campaign._rng.random(4)
    path = tmp_path / "campaign.json"
    campaign.save_checkpoint(path)
    restored = Campaign.load_checkpoint(path)
    assert restored.to_state() == campaign.to_state()
    actual = restored.ask_with_diagnostics(q=2, candidates=catalog()[0], candidate_ids=catalog()[1])
    expected = campaign.ask_with_diagnostics(q=2, candidates=catalog()[0], candidate_ids=catalog()[1])
    assert actual.candidate_ids == expected.candidate_ids
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    assert directional(actual, multi) == directional(expected, multi)
    if multi:
        assert actual.diagnostics["geometry"] == expected.diagnostics["geometry"]
        assert actual.diagnostics["weights_by_slot"] == expected.diagnostics["weights_by_slot"]
    np.testing.assert_array_equal(restored._rng.random(3), campaign._rng.random(3))


@pytest.mark.parametrize("multi", [False, True])
def test_v2_validation_wire_roundtrip_without_replay(multi, monkeypatch):
    space, knowledge = declaration(multi=multi)
    campaign = Campaign(space, knowledge, validation_min_obs=3, validation_interval=1)
    X = pd.DataFrame({"x": np.linspace(0., 1., 10)})
    Y = pd.DataFrame({name: -X.x for name in space.objectives})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        campaign.tell(X, Y if multi else Y.gain)
    state = campaign.to_state()
    records = state["validation"]["diagnostics"]
    assert records[0]["version"] == "2.0" and records[0]["violation"] is True
    key = "emitted" if multi else "directional_emitted"
    assert state["validation"][key] == [["gain", "direction", "monotone", "x"]]
    if not multi:
        assert state["schema_version"] == "2.1" and state["validation"]["emitted"] == []

    def fail(*args, **kwargs):
        raise AssertionError("restore must not evaluate empirical trends")

    import expdoe_dk.bo.loop as loop
    monkeypatch.setattr(loop, "check_monotone_assumption", fail)
    restored = Campaign.from_state(json.loads(json.dumps(state, allow_nan=False)))
    assert restored.validation_diagnostics == campaign.validation_diagnostics
    assert restored.to_state() == state


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("mutation", ["unknown_version", "bad_epsilon", "scope", "diagnostic_scope", "diagnostic_version", "solver"])
def test_v2_checkpoint_corruption_rejects(multi, mutation):
    state = initial(multi=multi, validate=True).to_state()
    spec = state["knowledge"]["specs"][0]
    if mutation == "unknown_version":
        spec["version"] = "2.9"
    elif mutation == "bad_epsilon":
        spec["parameters"]["epsilon"] = "auto"
    elif mutation == "scope":
        spec["scope"]["objectives"] = ["absent"]
    else:
        diagnostic = state["validation"]["diagnostics"][0]
        if mutation == "diagnostic_scope":
            diagnostic["objective_name"] = "cost" if multi else "absent"
        elif mutation == "diagnostic_version":
            diagnostic["version"] = "1.0"
        else:
            diagnostic["fits"] = []
    for reader in (validate_state, Campaign.from_state):
        with pytest.raises(EngineError) as caught:
            reader(deepcopy(state))
        assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("mutation", ["warning_scope", "warning_parameter", "disabled", "inactive_checked",
    "missing_version", "duplicate_diagnostic", "nonfinite", "legacy_warning"])
def test_v2_diagnostic_identity_cannot_be_relabelled(multi, mutation):
    raw = initial(multi=multi, validate=True).to_state()
    validation = raw["validation"]
    warning_key = "emitted" if multi else "directional_emitted"
    validation[warning_key] = [["gain", "direction", "monotone", "x"]]
    if mutation == "warning_scope":
        validation[warning_key][0][0] = "cost"
    elif mutation == "warning_parameter":
        validation[warning_key][0][3] = "absent"
    elif mutation == "disabled":
        raw["knowledge"]["specs"][0]["enabled"] = False
    elif mutation == "inactive_checked":
        raw["knowledge"]["specs"][0]["confidence"] = 0.
    elif mutation == "missing_version":
        del validation["diagnostics"][0]["version"]
    elif mutation == "duplicate_diagnostic":
        validation["diagnostics"].append(deepcopy(validation["diagnostics"][0]))
    elif mutation == "nonfinite":
        validation["diagnostics"][0]["violation"] = float("nan")
    elif multi:
        validation["diagnostics"][0].pop("version")
        validation["diagnostics"][0].pop("evidence_basis")
    else:
        validation["emitted"] = [["monotone", "x"]]
    with pytest.raises(EngineError) as caught:
        Campaign.from_state(raw)
    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


@pytest.mark.parametrize("missing", [None, pd.NA, float("nan"), float("inf"), -float("inf")])
def test_v2_lossless_multi_markers_and_role_collisions(missing):
    campaign = initial(multi=True, name="gain", factor="gain")
    X = pd.DataFrame({"gain": [.33]})
    Y = pd.DataFrame({"gain": pd.Series([missing], dtype=object), "cost": ["unavailable"]})
    campaign.tell_batch(ObservationBatch(X, Y, Yvar=pd.DataFrame({"gain": [.1], "cost": [.2]}),
                                       status=pd.Series(["failed"], dtype=object), ids=("failed",)))
    restored = Campaign.from_state(campaign.to_state())
    assert restored.history_columns == campaign.history_columns
    assert restored.history_columns["parameters"]["gain"] != restored.history_columns["objectives"]["gain"]
    pd.testing.assert_frame_equal(restored.history_df(), campaign.history_df())
    pd.testing.assert_frame_equal(restored._observation_batches[-1].Y, Y)
    assert restored.to_state() == campaign.to_state()


def test_literal_legacy_archive_and_v1_next_fit_stay_legacy(tmp_path):
    from pathlib import Path
    fixture = Path(__file__).parents[1] / "fixtures" / "v04_checkpoint.json"
    original = fixture.read_bytes()
    legacy = Campaign.load_checkpoint(fixture)
    assert legacy.history_df().y.tolist() == [1.5]
    assert fixture.read_bytes() == original
    campaign = initial()
    campaign.knowledge = Knowledge().with_monotone("x", "increases_objective",
                                                   n_pairs_per_dim=3, epsilon=.05)
    path = tmp_path / "v1.json"
    campaign.save_checkpoint(path)
    wire = json.loads(path.read_text())
    assert set(wire["validation"]) == {"active", "interval", "min_obs", "auto_rescue", "last_n", "emitted"}
    restored = Campaign.load_checkpoint(path)
    before = restored.knowledge.to_envelope()
    actual = restored.ask_with_diagnostics(candidates=catalog()[0], candidate_ids=catalog()[1])
    expected = campaign.ask_with_diagnostics(candidates=catalog()[0], candidate_ids=catalog()[1])
    assert actual.candidate_ids == expected.candidate_ids
    np.testing.assert_array_equal(actual.predictions.mean, expected.predictions.mean)
    assert actual.diagnostics["transforms"]["virtual_rows"] > 0
    assert "directional" not in actual.diagnostics
    assert restored.knowledge.to_envelope() == before
    assert before["specs"][0]["version"] == "1.0"


def test_legacy_only_scalar_rejects_v2_variant_and_four_string_emitted():
    campaign = initial()
    campaign.knowledge = Knowledge()
    state = campaign.to_state()
    for change in ({"directional_emitted": [], "diagnostics": []},
                   {"emitted": [["gain", "direction", "monotone", "x"]]}):
        raw = deepcopy(state)
        raw["validation"].update(change)
        with pytest.raises(EngineError):
            Campaign.from_state(raw)


def test_literal_v1_archive_does_not_resolve_to_new_algorithm():
    # Deliberately literal historical format, not produced by today's writer.
    archive = {"space": {"params": [{"name": "x", "bounds": [0., 1.], "unit": "",
        "kind": "continuous", "step": None, "log_scale": False}], "constraints": [],
        "objectives": ["gain"], "maximize": [True]},
        "knowledge": {"strict": False, "items": [{"kind": "monotone", "param": "x",
            "effect": "increases_objective", "n_pairs_per_dim": 3, "epsilon": .05, "delta_norm": .5}]},
        "seed": 2, "history": [{"x": x, "y": y, "kind": "doe"} for x, y in
            [(.05, 10.), (.25, 12.), (.5, 11.), (.7, 14.), (.95, 13.)]], "trial_kind": ["doe"] * 5}
    before = deepcopy(archive)
    restored = Campaign.from_state(archive)
    result = restored.ask_with_diagnostics(candidates=catalog()[0], candidate_ids=catalog()[1])
    assert archive == before
    assert restored.knowledge.specs[0].version == "1.0"
    assert result.diagnostics["transforms"]["virtual_rows"] == 6
    assert "directional" not in result.diagnostics
