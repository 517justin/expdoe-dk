"""Recommendation must rank the GP, honor safety, and never invent a fallback."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from expdoe_dk import (
    Campaign, EngineError, ErrorCode, ExpressionConstraint, Knowledge, Objective,
    ObservationBatch, OutcomeConstraint, Parameter, PendingBatch, Space,
    candidate_keys, fit_surrogate, recommend,
)


def _line(direction="maximize", *, target=None, outcome_constraints=()):
    space = Space([Parameter("x", bounds=(0, 10), kind="integer")],
        objectives=Objective("y", direction, target=target),
        outcome_constraints=outcome_constraints)
    observations = ObservationBatch(pd.DataFrame({"x": [2, 4, 6, 8]}),
        pd.DataFrame({"y": [120., 140., 160., 180.]}),
        Yvar=pd.DataFrame({"y": [0.001] * 4}))
    return space, observations


def test_exact_mixed_candidates(mixed_case, mixed_observations):
    out = recommend(mixed_case, mixed_observations, q=2, seed=17)
    assert mixed_case.feasibility_mask(out.frame).all()
    assert len(candidate_keys(out.frame, mixed_case)) == 2
    assert not candidate_keys(out.frame, mixed_case) & candidate_keys(mixed_observations.X, mixed_case)
    assert out.predictions.mean.shape == (2, 1)
    assert np.isfinite(out.predictions.variance).all()
    assert out.diagnostics["acquisition_name"] == "qLogNoisyExpectedImprovement"
    assert out.diagnostics["strategy_explanation"]
    assert out.diagnostics["acquisition_value_kind"] == "joint_batch"
    assert np.isfinite(out.diagnostics["joint_acquisition_value"])
    assert "per_point_acquisition" not in out.diagnostics
    json.dumps(out.to_dict(), allow_nan=False)


def test_gp_failure_does_not_return_random(monkeypatch, mixed_case, smooth_batch):
    def fail(*args, **kwargs):
        raise RuntimeError("fit failed")
    monkeypatch.setattr("expdoe_dk.bo.recommendation.fit_surrogate", fail)
    with pytest.raises(EngineError) as error:
        recommend(mixed_case, smooth_batch, seed=17)
    assert error.value.code == ErrorCode.MODEL_FIT_FAILED


def test_acquisition_failure_does_not_return_random(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("acquisition failed")
    monkeypatch.setattr("expdoe_dk.bo.recommendation.qLogNoisyExpectedImprovement", fail)
    with pytest.raises(EngineError) as error:
        recommend(*_line(), seed=17)
    assert error.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE


@pytest.mark.parametrize("direction,target,expected", [
    ("maximize", None, {9, 10}), ("minimize", None, {0, 1}),
    ("target", (149., 151.), {5}), ("target", 150., {5}),
])
def test_physical_objective_ranking(direction, target, expected):
    out = recommend(*_line(direction, target=target), seed=17)
    assert out.frame.x.item() in expected
    assert 80 < out.predictions.mean.item() < 220  # Never standardized/internal units.


def test_pending_excluded_and_conditioned():
    space, observations = _line()
    pending = PendingBatch(("p9", "p10"), pd.DataFrame({"x": [9, 10]}))
    out = recommend(space, observations, pending=pending, q=2, seed=17)
    assert len(out.frame) == 2
    assert len(candidate_keys(out.frame, space)) == 2
    assert not candidate_keys(out.frame, space) & candidate_keys(pending.X, space)
    assert out.diagnostics["pending_count"] == 2
    # Verify numerical conditioning, not merely filtering pending identities.
    from expdoe_dk.bo.recommendation import _make_acquisition
    fitted = fit_surrogate(space, observations, seed=17)
    plain = _make_acquisition(space, fitted, observations.successful().X, None, {}, 17)
    conditioned = _make_acquisition(space, fitted, observations.successful().X, pending.X, {}, 17)
    point = space.physical_to_model(pd.DataFrame({"x": [7]})).unsqueeze(0)
    assert not torch.allclose(plain(point), conditioned(point))


def test_runtime_safety_in_public_and_legacy_campaign():
    space, observations = _line()
    knowledge = Knowledge().with_safe_region(expression="x <= 3", factors=("x",))
    out = recommend(space, observations, knowledge=knowledge, q=2, seed=17)
    assert (out.frame.x <= 3).all()
    campaign = Campaign(space, knowledge=knowledge, seed=17)
    campaign.tell(observations.X, observations.Y.y)
    assert (campaign.ask(q=2).x <= 3).all()
    detailed = campaign.ask_with_diagnostics(q=2)
    pd.testing.assert_frame_equal(campaign.ask(q=2), detailed.frame)
    assert detailed.diagnostics["parameter_constraint_sources"]


def test_outcome_safety_is_physical_model_based_feasibility():
    from scipy.special import ndtr
    space, observations = _line(outcome_constraints=(OutcomeConstraint("safe", "y", "<=", 150.),))
    out = recommend(space, observations, q=2, seed=17)
    assert (out.frame.x <= 5).all()
    safety = out.diagnostics["outcome_safety"]
    assert safety["basis"] == "model_based_feasibility"
    assert safety["guaranteed"] is False
    assert min(safety["probabilities"]) >= safety["minimum_probability"]
    assert safety["minimum_probability"] >= .95
    expected_probability = ndtr((150. - out.predictions.mean[:, 0]) / np.sqrt(out.predictions.variance[:, 0]))
    np.testing.assert_allclose(safety["probabilities"], expected_probability)


def test_safety_blocks_without_sufficient_model_data():
    space, observations = _line(outcome_constraints=(OutcomeConstraint("safe", "y", "<=", 150.),))
    with pytest.raises(EngineError) as error:
        recommend(space, ObservationBatch(observations.X.iloc[:2], observations.Y.iloc[:2]))
    assert error.value.code == ErrorCode.INSUFFICIENT_DATA


def test_safety_blocks_without_observed_feasible_baseline():
    space, observations = _line(outcome_constraints=(OutcomeConstraint("safe", "y", "<=", -1000.),))
    with pytest.raises(EngineError) as error:
        recommend(space, observations)
    assert error.value.code == ErrorCode.INSUFFICIENT_DATA


def test_all_infeasible_is_typed_error():
    space, observations = _line()
    space = space.with_constraints(ExpressionConstraint("impossible", "x < 0"))
    with pytest.raises(EngineError) as error:
        recommend(space, observations)
    assert error.value.code == ErrorCode.NO_FEASIBLE_CANDIDATE


def test_exhaustion_never_returns_partial_batch():
    space = Space([Parameter("x", kind="integer", bounds=(0, 4))], objectives="y")
    observations = ObservationBatch(pd.DataFrame({"x": [0, 1, 2]}), pd.DataFrame({"y": [0., 1., 2.]}))
    pending = PendingBatch(("p3",), pd.DataFrame({"x": [3]}))
    with pytest.raises(EngineError) as error:
        recommend(space, observations, pending=pending, q=2)
    assert error.value.code == ErrorCode.NO_FEASIBLE_CANDIDATE


def test_large_exact_integer_values_remain_distinct():
    base = 2**53 + 10
    space = Space([Parameter("n", kind="integer", bounds=(base, base + 8), step=2)], objectives="y")
    observations = ObservationBatch(pd.DataFrame({"n": [base, base + 2, base + 4]}),
        pd.DataFrame({"y": [0., 1., 2.]}))
    out = recommend(space, observations, q=2)
    assert set(out.frame.n.tolist()) == {base + 6, base + 8}
    assert len(candidate_keys(out.frame, space)) == 2


def test_wide_integer_codec_alias_fails_closed():
    base = 2**54
    space = Space([Parameter("n", kind="integer", bounds=(0, base + 20))], objectives="y")
    observations = ObservationBatch(pd.DataFrame({"n": [base, base + 1, base + 2]}),
        pd.DataFrame({"y": [0., 1., 2.]}))
    with pytest.raises(EngineError) as error:
        recommend(space, observations)
    assert error.value.code == ErrorCode.CONFIG_INVALID
    assert "precision" in str(error.value).lower() or "alias" in str(error.value).lower()


def test_seeded_bounded_pool_scores_large_discrete_space():
    space = Space([Parameter("n", kind="integer", bounds=(0, 100001))], objectives="y")
    observations = ObservationBatch(pd.DataFrame({"n": [0, 40000, 80000]}),
        pd.DataFrame({"y": [0., 4., 8.]}))
    first = recommend(space, observations, seed=17)
    second = recommend(space, observations, seed=17)
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.diagnostics["search_strategy"] == "seeded_bounded_pool"
    assert first.diagnostics["degradation_reason"]
    assert first.frame.n.item() > 80000


def test_log_continuous_deterministic_and_rng_isolated():
    space = Space([Parameter("x", bounds=(1., 100.), transform="log")], objectives="y")
    observations = ObservationBatch(pd.DataFrame({"x": [1., 10., 100.]}),
        pd.DataFrame({"y": [0., 1., 0.]}))
    state = torch.random.get_rng_state().clone()
    first = recommend(space, observations, q=2, seed=17)
    second = recommend(space, observations, q=2, seed=17)
    assert torch.equal(state, torch.random.get_rng_state())
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert space.feasibility_mask(first.frame).all()
    assert len(candidate_keys(first.frame, space)) == 2
    assert first.diagnostics["continuous_subspaces_optimized"] > 0


@pytest.mark.parametrize("q,seed", [(0, 0), (-1, 0), (True, 0), (1, -1), (1, True)])
def test_invalid_options_fail_closed(q, seed):
    with pytest.raises(EngineError) as error:
        recommend(*_line(), q=q, seed=seed)
    assert error.value.code == ErrorCode.CONFIG_INVALID


def test_unknown_artifacts_fail_closed(unsupported_artifact_case, mixed_observations):
    space, knowledge = unsupported_artifact_case
    with pytest.raises(EngineError) as error:
        recommend(space, mixed_observations, knowledge=knowledge)
    assert error.value.code == ErrorCode.KNOWLEDGE_INVALID


def test_acquisition_adapter_is_physical_differentiable_joint_posterior():
    space, observations = _line("maximize")
    fitted = fit_surrogate(space, observations, seed=17)
    adapter = fitted.acquisition_model()
    frame = pd.DataFrame({"x": [3, 5, 7]})
    inputs = space.physical_to_model(frame).requires_grad_(True)
    posterior = adapter.posterior(inputs)
    prediction = fitted.predict(frame)
    np.testing.assert_allclose(posterior.mean.detach().numpy(), prediction.mean)
    np.testing.assert_allclose(posterior.variance.detach().numpy(), prediction.variance)
    gradient, = torch.autograd.grad(posterior.mean.sum(), inputs)
    assert torch.isfinite(gradient).all() and (gradient.abs() > 0).any()
    samples = posterior.rsample(torch.Size([64]))
    assert samples.shape == (64, 3, 1)
    assert samples.mean() > 100  # Same physical sample units used by acquisition.


def test_continuous_optimizer_failure_does_not_use_scored_pool(monkeypatch):
    space = Space([Parameter("x", bounds=(0., 10.))], objectives="y")
    observations = ObservationBatch(pd.DataFrame({"x": [2., 4., 6.]}),
        pd.DataFrame({"y": [120., 140., 160.]}))
    def fail(*args, **kwargs):
        raise RuntimeError("optimizer failed")
    monkeypatch.setattr("expdoe_dk.bo.recommendation.optimize_acqf", fail)
    with pytest.raises(EngineError) as error:
        recommend(space, observations)
    assert error.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE


@pytest.mark.parametrize("direction,target,expected,gradient", [
    ("maximize", None, [0., 5., 10.], [1., 1., 1.]),
    ("minimize", None, [0., -5., -10.], [-1., -1., -1.]),
    ("target", (4., 6.), [-4., 0., -4.], [1., 0., -1.]),
])
def test_objective_utility_is_applied_to_samples(direction, target, expected, gradient):
    from expdoe_dk.bo.recommendation import _utility
    samples = torch.tensor([[0.], [5.], [10.]], dtype=torch.double, requires_grad=True)
    utility = _utility(samples, Objective("y", direction, target=target), {})
    np.testing.assert_allclose(utility.detach(), expected)
    derivative, = torch.autograd.grad(utility.sum(), samples)
    np.testing.assert_allclose(derivative[:, 0], gradient)


def test_soft_outcome_range_adds_physical_shortfall_not_rejection(pattern_cases):
    from expdoe_dk import compile_runtime
    from expdoe_dk.bo.recommendation import _utility
    space, knowledge = next(pair for pair in pattern_cases if pair[1].specs[0].pattern == "target_range")
    runtime = compile_runtime(space, knowledge, None)
    samples = torch.tensor([[0.], [.5], [1.]], dtype=torch.double)
    utility = _utility(samples, space.objective_specs[0], runtime)
    np.testing.assert_allclose(utility, [-.2, .5, .8])


def test_safety_rejects_all_untried_conditions():
    space = Space([Parameter("x", bounds=(0, 4), kind="integer")], objectives="y",
        outcome_constraints=(OutcomeConstraint("safe", "y", "<=", 2.),))
    observations = ObservationBatch(pd.DataFrame({"x": [0, 1, 2]}),
        pd.DataFrame({"y": [0., 1., 2.]}), Yvar=pd.DataFrame({"y": [.0001] * 3}))
    with pytest.raises(EngineError) as error:
        recommend(space, observations)
    assert error.value.code == ErrorCode.NO_FEASIBLE_CANDIDATE


def test_failed_and_warning_rows_are_excluded_from_fit_but_not_recommended():
    space, batch = _line()
    observations = ObservationBatch(pd.concat([batch.X, pd.DataFrame({"x": [9, 10]})], ignore_index=True),
        pd.concat([batch.Y, pd.DataFrame({"y": [np.nan, np.nan]})], ignore_index=True),
        status=pd.Series(["success"] * 4 + ["warning", "failed"]))
    out = recommend(space, observations)
    assert out.frame.x.item() not in {2, 4, 6, 8, 9, 10}
    assert {item["reason"] for item in out.diagnostics["exclusions"]} == {"warning", "failed"}


def test_ordinal_scalar_types_are_not_merged():
    space = Space([Parameter("material", kind="ordinal", values=(True, 1, "1", "C", "A", "B"))],
        objectives="y")
    observations = ObservationBatch(pd.DataFrame({"material": pd.Series(["C", "A", "B"], dtype=object)}),
        pd.DataFrame({"y": [0., 1., 2.]}))
    out = recommend(space, observations, q=3)
    actual = {(type(value), value) for value in out.frame.material.tolist()}
    assert actual == {(bool, True), (int, 1), (str, "1")}
    assert len(candidate_keys(out.frame, space)) == 3


def test_ordinal_null_codec_limitation_is_explicit():
    space = Space([Parameter("grade", kind="ordinal", values=(None, "A", "B", "C"))], objectives="y")
    observations = ObservationBatch(pd.DataFrame({"grade": ["A", "B", "C"]}),
        pd.DataFrame({"y": [0., 1., 2.]}))
    with pytest.raises(EngineError) as error:
        recommend(space, observations)
    assert error.value.code == ErrorCode.CONFIG_INVALID
    assert "null" in str(error.value).lower()


def test_search_pool_respects_existing_row_limit():
    from expdoe_dk.bo.recommendation import _search_pool
    space = Space([Parameter("x", bounds=(0., 1.)),
        Parameter("n", bounds=(0, 25000), kind="integer")])
    pool, _, _, diagnostics = _search_pool(space, q=1, seed=17)
    assert len(pool) <= 100000
    assert diagnostics["finite_combinations"] == 25001
