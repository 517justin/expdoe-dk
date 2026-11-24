"""R1: empirical evidence belongs to a declaration, not just its pattern ID."""
import warnings

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import Campaign, EngineError, Knowledge, Objective, ObservationBatch, Parameter, Space
from .test_campaign import catalog, declaration, initial, replace_directional


def validated_multi(*, violating=False):
    space, knowledge = declaration(multi=True)
    campaign = Campaign(space, knowledge, validation_min_obs=3, validation_interval=1)
    x = pd.DataFrame({"x": np.linspace(0., 1., 10)})
    y = pd.DataFrame({"gain": -x.x if violating else x.x, "cost": 1 - x.x})
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        campaign.tell(x, y)
    assert len(notices) == int(violating)
    assert campaign.validation_diagnostics[0]["violation"] is violating
    return campaign, x, y


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("change", ["direction", "confidence", "disable", "remove", "reid", "scope", "space"])
def test_multi_replacement_invalidates_evidence_before_ask(change, restored, tmp_path):
    campaign, _, _ = validated_multi()
    if restored:
        campaign = Campaign.from_state(campaign.to_state())
    if change == "scope":
        envelope = campaign.knowledge.to_envelope()
        envelope["specs"][0]["scope"]["objectives"] = ["cost"]
        campaign.knowledge = Knowledge.from_envelope(envelope)
    elif change == "space":
        campaign.space = Space([Parameter("x", bounds=(-1., 1.))],
                               objectives=[Objective("gain", "minimize"), Objective("cost", "minimize")])
    else:
        replace_directional(campaign, change)
    details = campaign.validation_diagnostics
    if change in {"disable", "remove"}:
        assert details == []
    else:
        detail, = details
        assert detail["status"] == "not_checked" and detail["violation"] is None
        assert detail["reason"] == ("inactive_confidence" if change == "confidence" else "validation_not_run")
        assert detail["objective_name"] == ("cost" if change == "scope" else "gain")
    if change not in {"disable", "remove"}:
        assert campaign.finalize().multiobjective_diagnostics["validation"] == details
    path = tmp_path / "replacement.json"
    campaign.save_checkpoint(path)
    resumed = Campaign.load_checkpoint(path)
    assert resumed.validation_diagnostics == details
    assert resumed.to_state() == campaign.to_state()


@pytest.mark.parametrize("restored", [False, True])
def test_multi_warning_reset_without_intervening_read(restored):
    campaign, x, y = validated_multi(violating=True)
    if restored:
        campaign = Campaign.from_state(campaign.to_state())
    envelope = campaign.knowledge.to_envelope()
    envelope["specs"][0]["confidence"] = .4
    campaign.knowledge = Knowledge.from_envelope(envelope)
    with pytest.warns(UserWarning, match=r"\[direction\]"):
        campaign.tell(x, y)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        campaign.tell(x, y)
    assert not notices


def test_multi_equivalent_restore_retains_verdict_and_suppression():
    campaign, x, y = validated_multi(violating=True)
    state = campaign.to_state()
    campaign = Campaign.from_state(state)
    campaign.knowledge = Knowledge.from_envelope(campaign.knowledge.to_envelope())
    campaign.space = Space.from_dict(campaign.space.to_dict())
    assert campaign.to_state() == state
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        campaign.tell(x, y)
    assert not notices


def test_failed_tell_does_not_restore_stale_declaration_evidence():
    campaign, x, y = validated_multi(violating=True)
    envelope = campaign.knowledge.to_envelope()
    envelope["specs"][0]["confidence"] = .4
    campaign.knowledge = Knowledge.from_envelope(envelope)
    with pytest.raises(EngineError, match="Mixed missing variance"):
        campaign.tell_batch(ObservationBatch(x, y, Yvar=y * 0 + .1))
    detail, = campaign.validation_diagnostics
    assert detail["status"] == "not_checked" and detail["violation"] is None
    assert campaign.to_state()["validation"]["emitted"] == []
    assert len(campaign.history_df()) == 10


def test_multi_warning_reset_is_scoped_to_changed_spec():
    space, knowledge = declaration(multi=True)
    other = declaration(name="cost", pattern_id="other")[1].specs[0]
    knowledge.add(other)
    campaign = Campaign(space, knowledge, validation_min_obs=3, validation_interval=1)
    x = pd.DataFrame({"x": np.linspace(0., 1., 10)})
    y = pd.DataFrame({"gain": -x.x, "cost": -x.x})
    with pytest.warns(UserWarning) as first:
        campaign.tell(x, y)
    assert len(first) == 2
    envelope = campaign.knowledge.to_envelope()
    envelope["specs"][0]["confidence"] = .4
    campaign.knowledge = Knowledge.from_envelope(envelope)
    details = {d["pattern_id"]: d for d in campaign.validation_diagnostics}
    assert details["other"]["violation"] is True
    assert details["direction"]["violation"] is None
    with pytest.warns(UserWarning) as second:
        campaign.tell(x, y)
    assert len(second) == 1 and "[direction]" in str(second[0].message)


@pytest.mark.parametrize("restored", [False, True])
def test_v2_replacement_after_successful_ask_remains_frozen(restored):
    campaign = initial(multi=True, confidence=0.)
    campaign.ask(candidates=catalog()[0], candidate_ids=catalog()[1])
    if restored:
        campaign = Campaign.from_state(campaign.to_state())
    replace_directional(campaign, "direction")
    for operation in [campaign.ask, campaign.to_state, campaign.finalize]:
        with pytest.raises(EngineError, match="frozen"):
            operation()


@pytest.mark.parametrize("change", ["direction", "confidence", "disable", "remove", "reid"])
def test_multi_checkpoint_first_read_clears_stale_warning_keys(change):
    campaign, _, _ = validated_multi(violating=True)
    campaign = Campaign.from_state(campaign.to_state())
    replace_directional(campaign, change)
    state = campaign.to_state()
    assert state["validation"]["emitted"] == []
    assert all(d["status"] == "not_checked" and d["violation"] is None
               for d in state["validation"]["diagnostics"])
    assert Campaign.from_state(state).to_state() == state
