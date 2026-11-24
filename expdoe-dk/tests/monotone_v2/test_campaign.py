"""Campaign v2 diagnostics are measured trends or actual ask-time fits, not both."""
from copy import deepcopy
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import (
    Campaign, EngineError, ErrorCode, Knowledge, KnowledgeScope, MultiObjectiveConfig,
    Objective, ObservationBatch, Parameter, Space, make_pattern_spec,
)


def declaration(*, multi=False, confidence=.8, pattern_id="direction", name="gain", factor="x"):
    space = Space([Parameter(factor, bounds=(0., 1.))],
        objectives=[Objective(name, "maximize")] + ([Objective("cost", "minimize")] if multi else []))
    knowledge = Knowledge().add_spec(make_pattern_spec(pattern="monotone", version="2.0",
        pattern_id=pattern_id, confidence=confidence,
        scope=KnowledgeScope(factors=(factor,), objectives=(name,)),
        parameters={"direction": "increasing", "n_pairs_per_dim": 3, "epsilon": .05, "softness": 1.}))
    return space, knowledge


def initial(*, multi=False, confidence=.8, validate=False, name="gain", factor="x", pattern_id="direction"):
    space, knowledge = declaration(multi=multi, confidence=confidence, name=name,
                                   factor=factor, pattern_id=pattern_id)
    campaign = Campaign(space, knowledge, seed=2, validate=validate,
        validation_min_obs=3, validation_interval=1,
        multiobjective=MultiObjectiveConfig(acquisition="q_log_nparego") if multi else None)
    X = pd.DataFrame({factor: [.05, .25, .5, .7, .95]})
    Y = pd.DataFrame({name: [10., 12., 11., 14., 13.]})
    if multi:
        Y["cost"] = [6., 7., 5., 8., 9.]
    campaign.tell_batch(ObservationBatch(X, Y, Yvar=Y * 0 + .04, ids=("a", "b", "c", "d", "e")))
    return campaign


def catalog(factor="x"):
    return pd.DataFrame({factor: np.linspace(0., 1., 21)}), tuple(f"c{i:03}" for i in range(21))


def directional(result, multi):
    return result.diagnostics["objectives"]["gain"]["directional"] if multi else result.diagnostics["directional"]


@pytest.mark.parametrize("multi", [False, True])
def test_public_ask_exposes_actual_detached_directional_fit(multi, monkeypatch):
    from expdoe_dk.bo import recommendation
    campaign = initial(multi=multi)
    original_fit = recommendation.fit_surrogate
    fits = []

    def record(*args, **kwargs):
        fitted = original_fit(*args, **kwargs)
        fits.append(fitted)
        return fitted

    monkeypatch.setattr(recommendation, "fit_surrogate", record)
    result = campaign.ask_with_diagnostics(q=2, candidates=catalog()[0], candidate_ids=catalog()[1])
    detail = directional(result, multi)
    assert len(fits) == 1
    fitted = fits[0]._fitted[0] if multi else fits[0]
    assert detail == fitted.diagnostics["directional"]
    assert detail["actual_measurement_count"] == 5
    assert detail["weights"] == [.8 / 3] * 3
    pair, = detail["pair_diagnostics"]
    assert (pair["pattern_id"], pair["version"], pair["objective_name"]) == ("direction", "2.0", "gain")
    assert len(pair["physical_endpoints"]) == len(detail["endpoints"]) == 3
    json.dumps(detail, allow_nan=False)
    detail["weights"].clear()
    assert fitted.diagnostics["directional"]["weights"] == [.8 / 3] * 3


@pytest.mark.parametrize("multi", [False, True])
def test_v2_measured_trend_keeps_exact_identity_and_never_rescues(multi):
    space, knowledge = declaration(multi=multi, factor="gain")
    before = knowledge.to_envelope()
    campaign = Campaign(space, knowledge, validation_min_obs=3, validation_interval=1, auto_rescue=True)
    X = pd.DataFrame({"gain": np.linspace(0., 1., 10)})
    Y = pd.DataFrame({"gain": -X.gain})
    if multi:
        Y["cost"] = X.gain
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        campaign.tell(X, Y if multi else Y.gain)
    detail, = campaign.validation_diagnostics
    assert detail == {"objective_name": "gain", "pattern_id": "direction", "pattern": "monotone",
        "version": "2.0", "evidence_basis": "marginal_measured_trend", "status": "checked",
        "violation": True, "reason": None}
    messages = [str(n.message) for n in notices]
    assert any("direction" in m and "2.0" in m and "confounded" in m for m in messages)
    assert all("with_random_augment" not in m and "drop_monotone" not in m for m in messages)
    assert campaign.knowledge.to_envelope() == before
    assert campaign.knowledge.to_dict()["items"] == []
    detail["violation"] = False
    assert campaign.validation_diagnostics[0]["violation"] is True
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        campaign.tell(X, Y if multi else Y.gain)
    assert not again


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("confidence,constant,reason", [
    (0., False, "inactive_confidence"), (.8, True, "constant_factor_or_response")])
def test_v2_inapplicable_trend_is_explicit(multi, confidence, constant, reason):
    space, knowledge = declaration(multi=multi, confidence=confidence)
    campaign = Campaign(space, knowledge, validation_min_obs=3, validation_interval=1)
    X = pd.DataFrame({"x": [.1] * 5 if constant else np.linspace(0., 1., 5)})
    Y = pd.DataFrame({name: np.arange(5.) for name in space.objectives})
    campaign.tell(X, Y if multi else Y.gain)
    d, = campaign.validation_diagnostics
    assert d["status"] == "not_checked" and d["reason"] == reason and d["violation"] is None


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("prior_ask", [False, True])
def test_refit_failure_does_not_commit_campaign_state(multi, prior_ask, monkeypatch):
    from expdoe_dk.bo import directional_posterior
    campaign = initial(multi=multi)
    if prior_ask:
        campaign.ask(candidates=catalog()[0], candidate_ids=catalog()[1])
    before = deepcopy(campaign.to_state())
    report_before = campaign.finalize().multiobjective_diagnostics

    def fail(*args, **kwargs):
        raise EngineError(ErrorCode.MODEL_FIT_FAILED, "injected directional nonconvergence")

    monkeypatch.setattr(directional_posterior, "_fit_directional_laplace", fail)
    with pytest.raises(EngineError, match="nonconvergence"):
        campaign.ask(candidates=catalog()[0], candidate_ids=catalog()[1])
    assert campaign.to_state() == before
    assert campaign.finalize().multiobjective_diagnostics == report_before


@pytest.mark.parametrize("multi", [False, True])
def test_inactive_public_fit_has_no_pairs_or_conditioning_fit(multi):
    campaign = initial(multi=multi, confidence=0.)
    result = campaign.ask_with_diagnostics(candidates=catalog()[0], candidate_ids=catalog()[1])
    d = directional(result, multi)
    assert d["status"] == "inactive" and d["fits"] == []
    assert all(not p["physical_endpoints"] for p in d["pair_diagnostics"])


@pytest.mark.parametrize("multi", [False, True])
def test_inactive_campaign_preserves_plain_recommendation_and_rng(multi):
    import torch
    campaign = initial(multi=multi, confidence=0.)
    plain = Campaign(campaign.space, seed=2, validate=False, multiobjective=campaign.multiobjective)
    plain.tell_batch(campaign._observation_batches[0])
    before = campaign.knowledge.to_envelope()
    torch.manual_seed(101)
    rng = torch.random.get_rng_state().clone()
    actual = campaign.ask_with_diagnostics(q=2, candidates=catalog()[0], candidate_ids=catalog()[1])
    expected = plain.ask_with_diagnostics(q=2, candidates=catalog()[0], candidate_ids=catalog()[1])
    assert actual.candidate_ids == expected.candidate_ids
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    np.testing.assert_array_equal(actual.predictions.mean, expected.predictions.mean)
    np.testing.assert_array_equal(actual.predictions.variance, expected.predictions.variance)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert campaign.knowledge.to_envelope() == before
    assert "directional" not in expected.diagnostics and "objectives" not in expected.diagnostics


@pytest.mark.parametrize("multi", [False, True])
def test_insufficient_and_disabled_validation_are_explicit_without_fitting(multi):
    space, knowledge = declaration(multi=multi)
    for enabled, reason in ((True, "insufficient_observations"), (False, "validation_disabled")):
        campaign = Campaign(space, knowledge, validate=enabled)
        d, = campaign.validation_diagnostics
        assert d["status"] == "not_checked" and d["reason"] == reason
        campaign.tell(pd.DataFrame({"x": [.2]}),
                      pd.DataFrame({name: [1.] for name in space.objectives}) if multi else [1.])
        assert campaign.validation_diagnostics == [d]


def test_mixed_v1_v2_output_checks_retain_separate_versions_and_warnings():
    space, knowledge = declaration(multi=True)
    knowledge.add(make_pattern_spec(pattern="monotone", version="1.0", pattern_id="legacy-cost",
        scope=KnowledgeScope(factors=("x",), objectives=("cost",)), confidence=1.,
        parameters={"direction": "increasing", "n_pairs_per_dim": 3, "epsilon": .05, "delta_norm": .5}))
    campaign = Campaign(space, knowledge, validation_min_obs=3, validation_interval=1)
    before = campaign.knowledge.to_envelope()
    X = pd.DataFrame({"x": np.linspace(0., 1., 10)})
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        campaign.tell(X, pd.DataFrame({"gain": -X.x, "cost": -X.x}))
    assert len(notices) == 2
    by_id = {d["pattern_id"]: d for d in campaign.validation_diagnostics}
    assert by_id["direction"]["version"] == "2.0"
    assert "version" not in by_id["legacy-cost"]
    assert all(d["violation"] for d in by_id.values())
    restored = Campaign.from_state(campaign.to_state())
    assert restored.knowledge.to_envelope() == before
    assert restored.validation_diagnostics == campaign.validation_diagnostics


def test_v2_adapter_ignores_unrelated_category_and_explains_unsupported_factor():
    for factor, expected in (("x", "checked"), ("material", "unsupported")):
        space = Space([Parameter("x", bounds=(0., 1.)),
                       Parameter("material", kind="categorical", values=("A", "B"))], objectives="gain")
        knowledge = declaration(factor=factor)[1]
        campaign = Campaign(space, knowledge, validation_min_obs=3, validation_interval=1)
        X = pd.DataFrame({"x": np.linspace(0., 1., 5), "material": ["A", "B", "A", "B", "A"]})
        campaign.tell(X, X.x)
        d, = campaign.validation_diagnostics
        assert d["status"] == expected
        if expected == "unsupported":
            assert d["reason"] == "unsupported_directional_scope"


@pytest.mark.parametrize("v2", [False, True])
def test_scalar_empirical_validation_does_not_require_uniform_noise_metadata(v2):
    campaign = initial(validate=True)
    if not v2:
        campaign.knowledge = Knowledge()
    # Empirical trends use measured X/Y, not the model's noise-mode combination.
    campaign.tell(pd.DataFrame({"x": [.8]}), [14.])
    assert len(campaign.history_df()) == 6
    if v2:
        assert campaign.validation_diagnostics[0]["status"] == "checked"
    with pytest.raises(EngineError, match="Mixed missing variance"):
        campaign.ask()


@pytest.mark.parametrize("qualifier", ["conditions", "region"])
def test_unsupported_qualified_scope_is_not_checked_as_global_trend(qualifier):
    space, knowledge = declaration()
    envelope = knowledge.to_envelope()
    envelope["specs"][0]["scope"][qualifier] = {"x": [.2, .8]}
    campaign = Campaign(space, Knowledge.from_envelope(envelope),
                        validation_min_obs=3, validation_interval=1)
    campaign.tell(pd.DataFrame({"x": [0., .25, .5, .75, 1.]}), [0., 1., 2., 3., 4.])
    detail, = campaign.validation_diagnostics
    assert detail["status"] == "unsupported"
    assert detail["reason"] == "unsupported_directional_scope"
    assert detail["violation"] is None
