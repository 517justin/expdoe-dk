"""Real measured-data fits through shared directional and physical adapters."""
import importlib
import json
import random

import numpy as np
import pandas as pd
import pytest
import torch
from botorch import settings
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import min_fixed_noise
from scipy.special import ndtr

from expdoe_dk import (
    EngineError, ErrorCode, Knowledge, KnowledgeScope, MultiObjectiveConfig,
    Objective, ObservationBatch, OutcomeConstraint, Parameter, Space,
    fit_surrogate, make_pattern_spec, recommend,
)
from expdoe_dk.bo.capabilities import inspect_capabilities
from expdoe_dk.bo.gp import build_gp, build_virtual_observation_augmenter
from expdoe_dk.bo.knowledge_runtime import compile_runtime, validate_runtime_artifact
from expdoe_dk.bo.objective_routing import project_objective
from expdoe_dk.bo.surrogates import FittedSurrogate, _PhysicalFeatures
from expdoe_dk.knowledge.artifacts import OptimizationArtifact


def data(*, multi=False, variance=.04, direction="maximize", constant=False,
         factor="x", safety=False):
    objectives = [Objective("gain", direction)]
    if multi:
        objectives.append(Objective("cost", "minimize"))
    space = Space([Parameter(factor, bounds=(0., 1.))], objectives=objectives,
        outcome_constraints=[OutcomeConstraint("safe", "gain", "<=", 14.)] if safety else ())
    X = pd.DataFrame({factor: [.05, .25, .5, .7, .95]})
    Y = pd.DataFrame({"gain": [10.] * 5 if constant else [10., 12., 11., 14., 13.]})
    if multi:
        Y.insert(0, "cost", [6., 7., 5., 8., 9.])
    return space, ObservationBatch(X, Y, Yvar=None if variance is None else Y * 0 + variance,
                                  ids=("a", "b", "c", "d", "e"))


def knowledge(*, factor="x", confidence=.8, objectives=("gain",), effect="increasing"):
    return Knowledge().add_spec(make_pattern_spec(pattern="monotone", version="2.0",
        pattern_id="direction", confidence=confidence,
        scope=KnowledgeScope(factors=(factor,), objectives=objectives),
        parameters={"direction": effect, "n_pairs_per_dim": 3, "epsilon": .05, "softness": 1.}))


def scalar(fitted, index=0):
    return fitted._fitted[index] if hasattr(fitted, "_fitted") else fitted


def rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def assert_rng_equal(before, after):
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("variance,mode", [(0., "known_zero"), (.04, "known"), (None, "inferred")])
def test_real_base_fit_has_only_measurements_and_unchanged_noise(multi, variance, mode):
    space, batch = data(multi=multi, variance=variance)
    original = (batch.X, batch.Y, batch.Yvar, batch.ids, batch.status)
    k = knowledge()
    plain = fit_surrogate(space, batch, seed=13)
    fitted = fit_surrogate(space, batch, knowledge=k, seed=13)
    selected, baseline = scalar(fitted), scalar(plain)
    base = selected._model._source
    assert selected.transforms["virtual_rows"] == 0
    assert base.train_inputs[0].shape == (5, 1)
    torch.testing.assert_close(base.train_targets, baseline._model.train_targets, rtol=0, atol=0)
    for name, parameter in baseline._model.named_parameters():
        torch.testing.assert_close(dict(base.named_parameters())[name], parameter, rtol=0, atol=0)
    torch.testing.assert_close(base.likelihood.noise, baseline._model.likelihood.noise, rtol=0, atol=0)
    if variance is not None:
        expected = torch.full((5,), variance / np.std(batch.Y.gain)**2, dtype=torch.double)
        if variance > 0:
            torch.testing.assert_close(base.likelihood.noise, expected, rtol=0, atol=0)
        assert selected.transforms["normalized_yvar"] == expected.tolist()
    diag = selected.diagnostics["directional"]
    assert diag["objective_name"] == "gain"
    assert diag["actual_measurement_count"] == 5 and diag["declared_noise_mode"] == mode
    assert diag["base_noise_mode"] == ("inferred" if variance is None else "known")
    assert diag["fitted_likelihood_noise"] == base.likelihood.noise.tolist()
    assert diag["status"] == "active" and diag["fits"][0]["converged"]
    detail, = diag["pair_diagnostics"]
    assert detail["pattern_id"] == "direction" and detail["version"] == "2.0"
    assert detail["factor"] == "x" and detail["direction"] == "increasing"
    assert detail["seed"] == 13 and detail["confidence"] == .8
    assert len(detail["physical_endpoints"]) == 3
    assert detail["softness"] == 1. and detail["accepted"] == 3
    json.dumps(diag, allow_nan=False)
    query = pd.DataFrame({"x": [.13, .37, .83]})
    actual, expected = fitted.predict(query), plain.predict(query)
    assert not np.allclose(actual.mean[:, 0], expected.mean[:, 0], rtol=0, atol=1e-8)
    if multi:
        np.testing.assert_array_equal(actual.mean[:, 1], expected.mean[:, 1])
        np.testing.assert_array_equal(actual.variance[:, 1], expected.variance[:, 1])
        assert "directional" not in scalar(fitted, 1).diagnostics
    posterior = fitted.acquisition_model().posterior(space.physical_to_model(query))
    np.testing.assert_allclose(posterior.mean.detach(), actual.mean, atol=1e-12)
    np.testing.assert_allclose(posterior.variance.detach(), actual.variance, atol=1e-12)
    pd.testing.assert_frame_equal(batch.X, original[0])
    pd.testing.assert_frame_equal(batch.Y, original[1])
    if variance is not None:
        pd.testing.assert_frame_equal(batch.Yvar, original[2])
    assert batch.ids == original[3] and batch.status is original[4] is None


@pytest.mark.parametrize("multi", [False, True])
def test_zero_confidence_preserves_posterior_samples_recommendations_and_rng(multi):
    space, batch = data(multi=multi)
    k = knowledge(confidence=0.)
    envelope = k.to_envelope()
    before = rng_state()
    plain = fit_surrogate(space, batch, seed=13)
    fitted = fit_surrogate(space, batch, knowledge=k, seed=13)
    assert_rng_equal(before, rng_state())
    assert type(scalar(fitted)._model) is type(scalar(plain)._model)
    assert scalar(fitted).diagnostics["directional"]["status"] == "inactive"
    query = pd.DataFrame({"x": [.13, .37, .83]})
    np.testing.assert_array_equal(fitted.predict(query).mean, plain.predict(query).mean)
    np.testing.assert_array_equal(fitted.predict(query).variance, plain.predict(query).variance)
    X = space.physical_to_model(query)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        first = plain.acquisition_model().posterior(X).rsample(torch.Size([8]))
        torch.manual_seed(42)
        second = fitted.acquisition_model().posterior(X).rsample(torch.Size([8]))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    a = recommend(space, batch, candidates=query, candidate_ids=("p", "q", "r"), q=2, seed=17)
    b = recommend(space, batch, knowledge=k, candidates=query, candidate_ids=("p", "q", "r"), q=2, seed=17)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert a.candidate_ids == b.candidate_ids
    np.testing.assert_array_equal(a.predictions.mean, b.predictions.mean)
    np.testing.assert_array_equal(a.predictions.variance, b.predictions.variance)
    assert_rng_equal(before, rng_state())
    assert k.to_envelope() == envelope


@pytest.mark.parametrize("multi", [False, True])
def test_capability_advertises_wired_directional_consumer(multi):
    space, batch = data(multi=multi)
    k = knowledge()
    report = inspect_capabilities(space, k)
    assert report.applied == ("direction",) and not report.blocked
    assert any("condition_directional_posterior" in d.get("consumer", "") for d in report.diagnostics)
    compile_runtime(space, k, batch)


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("other", ["duplicate", "opposite", "legacy", "random"])
def test_composition_rejects_before_fit_with_original_identities(multi, other):
    space, batch = data(multi=multi)
    k = knowledge(confidence=0.)
    payload = k.specs[0].to_dict()
    payload["pattern_id"] = "conflict"
    if other == "opposite":
        payload["parameters"]["direction"] = "decreasing"
    elif other == "legacy":
        payload["version"] = "1.0"
        payload["parameters"].pop("softness")
        payload["parameters"]["delta_norm"] = .5
    elif other == "random":
        payload.update(pattern="random_augment", version="1.0", parameters={"n": 2})
        payload["scope"]["factors"] = []
    from expdoe_dk.knowledge.specs import KnowledgePatternSpec
    k.add_spec(KnowledgePatternSpec.from_dict(payload))
    assert set(inspect_capabilities(space, k).blocked) == {"direction", "conflict"}
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, batch, knowledge=k)
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert set(caught.value.details["pattern_ids"]) == {"direction", "conflict"}
    if multi:
        assert caught.value.details["objective_name"] == "gain"


@pytest.mark.parametrize("factor", ["x", "gain"])
def test_global_parameter_constraints_survive_projection_without_response_leakage(factor):
    space, batch = data(multi=True, factor=factor)
    k = knowledge(factor=factor, objectives=("gain", "cost"))
    k.with_safe_region(expression=f"{factor} <= 0.3", factors=(factor,))
    k.add_spec(make_pattern_spec(pattern="target_range", version="1.0", confidence=1.,
        scope=KnowledgeScope(objectives=("cost",)), parameters={"lower": 4., "upper": 10.}))
    fitted = fit_surrogate(space, batch, knowledge=k, seed=9)
    for name, model in zip(space.objectives, fitted._fitted):
        diag = model.diagnostics["directional"]
        assert diag["objective_name"] == name
        assert np.max(diag["endpoints"]) <= .3
        assert diag["pair_diagnostics"][0]["generated_centers"] > 0
        projected, selected, scoped = project_objective(space, batch, k, name)
        scoped.with_safe_region(expression=f"{factor} <= 0.3", factors=(factor,))
        expected = fit_surrogate(projected, selected, knowledge=scoped, seed=9)
        np.testing.assert_array_equal(model.predict(batch.X).mean, expected.predict(batch.X).mean)


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_positive_affine_nonconstant_response_and_physical_direction(direction):
    space, batch = data(direction=direction)
    k = knowledge()
    fitted = fit_surrogate(space, batch, knowledge=k, seed=7)
    scaled = fit_surrogate(space, ObservationBatch(batch.X, batch.Y * 10 + 17,
                           Yvar=batch.Yvar * 100), knowledge=k, seed=7)
    query = pd.DataFrame({"x": [.13, .37, .83]})
    p, s = fitted.predict(query), scaled.predict(query)
    np.testing.assert_allclose(s.mean, p.mean * 10 + 17, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(s.variance, p.variance * 100, rtol=1e-9, atol=1e-9)
    assert fitted.diagnostics["directional"]["signs"] == ([-1.] * 3 if direction == "maximize" else [1.] * 3)
    opposite_space, _ = data(direction="minimize" if direction == "maximize" else "maximize")
    opposite = fit_surrogate(opposite_space, batch, knowledge=k, seed=7).predict(query)
    np.testing.assert_allclose(opposite.mean, p.mean, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(opposite.variance, p.variance, rtol=1e-8, atol=1e-8)


def test_constant_response_uses_existing_unit_scale_fallback():
    space, batch = data(constant=True)
    fitted = fit_surrogate(space, batch, knowledge=knowledge())
    assert fitted.transforms["response_scale"] == 1.
    shifted = fit_surrogate(space, ObservationBatch(batch.X, batch.Y + 17, Yvar=batch.Yvar),
                            knowledge=knowledge())
    np.testing.assert_allclose(shifted.predict(batch.X).mean, fitted.predict(batch.X).mean + 17)
    np.testing.assert_allclose(shifted.predict(batch.X).variance, fitted.predict(batch.X).variance)


@pytest.fixture(scope="module")
def models():
    result = {}
    for mode, variance in [("noiseless", 0.), ("known", .04), ("inferred", None)]:
        space, batch = data(multi=True, variance=variance, safety=True)
        k = knowledge(objectives=("gain", "cost"))
        with min_fixed_noise(double_value=0.):
            fitted = fit_surrogate(space, batch, knowledge=k, seed=11)
        result[mode] = (space, batch, k, fitted)
    return result


def acquisition(case, kind, mode):
    from expdoe_dk.bo.acquisition import build_multiobjective_acquisition
    from expdoe_dk.bo.objective_geometry import resolve_geometry
    space, batch, k, fitted = case
    config = MultiObjectiveConfig(acquisition=kind, noise_mode=mode)
    runtime = compile_runtime(space, k, batch)
    geometry = resolve_geometry(space, batch, runtime, config)
    return build_multiobjective_acquisition(space, fitted, batch, None, runtime, config,
                                          geometry, seed=23, slot=0)[0]


@pytest.mark.parametrize("kind,mode", [("q_log_ehvi", "noiseless"),
    ("q_log_nehvi", "known"), ("q_log_nehvi", "inferred"), ("q_log_nparego", "known")])
def test_real_acquisitions_joint_q_gradients_and_single_physical_safety_factor(models, kind, mode):
    space, batch, k, fitted = models[mode]
    acq = acquisition(models[mode], kind, mode)
    X = torch.tensor([[[.13], [.37]], [[.61], [.83]]], dtype=torch.double, requires_grad=True)
    values = acq(X)
    assert values.shape == (2,) and torch.isfinite(values).all()
    gradient, = torch.autograd.grad(values.sum(), X)
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    for index in np.ndindex(*X.shape):
        plus, minus = X.detach().clone(), X.detach().clone()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        finite_difference = (acq(plus).sum() - acq(minus).sum()) / 2e-5
        torch.testing.assert_close(gradient[index], finite_difference, rtol=3e-3, atol=3e-4)
    constraints = acq._constraints if kind == "q_log_nparego" else acq.constraints
    assert len(constraints) == 1
    samples = torch.tensor([[[13., 999.], [15., -999.]]], dtype=torch.double)
    torch.testing.assert_close(constraints[0](samples), torch.tensor([[-1., 1.]], dtype=torch.double))
    from expdoe_dk.bo.acquisition import joint_feasibility
    query = pd.DataFrame({"x": [.13, .37]})
    p = fitted.predict(query)
    expected = ndtr((14. - p.mean[:, 0]) / np.sqrt(p.variance[:, 0]))
    np.testing.assert_allclose(joint_feasibility(space, fitted, query), expected, rtol=1e-12)


def test_real_joint_samples_match_physical_covariance_and_independent_outputs(models):
    space, _, _, fitted = models["known"]
    X = torch.tensor([[[.13], [.13001], [.83]], [[.21], [.21001], [.73]]],
                     dtype=torch.double, requires_grad=True)
    model = fitted.acquisition_model()
    posterior = model.posterior(X)
    samples = fitted.acquisition_sampler(sample_shape=torch.Size([16384]), seed=19)(posterior)
    assert samples.shape == (16384, 2, 3, 2)
    for output in range(2):
        covariance = torch.cov(samples[:, 0, :, output].detach().T)
        source = model.models[output]
        expected = source._source.posterior(X[:1]).distribution.covariance_matrix[0] * source._scale**2
        diagonal = expected.diag()
        standard_error = ((diagonal[:, None] * diagonal[None, :] + expected.square()) / 16383).sqrt()
        assert ((covariance - expected).abs() / standard_error).max() < 5
    assert torch.corrcoef(samples[:, 0, 0, :].detach().T)[0, 1].abs() < .05
    samples.square().mean().backward()
    assert torch.isfinite(X.grad).all() and X.grad.abs().max() > 0


@pytest.mark.parametrize("kind,mode,module,symbol", [
    ("q_log_ehvi", "noiseless", "botorch.acquisition.multi_objective.logei", "qLogExpectedHypervolumeImprovement"),
    ("q_log_nehvi", "known", "botorch.acquisition.multi_objective.logei", "qLogNoisyExpectedHypervolumeImprovement"),
    ("q_log_nparego", "known", "botorch.acquisition.logei", "qLogNoisyExpectedImprovement"),
])
def test_missing_acquisition_capability_still_fails_typed(models, monkeypatch, kind, mode, module, symbol):
    monkeypatch.delattr(importlib.import_module(module), symbol)
    with pytest.raises(EngineError) as caught:
        acquisition(models[mode], kind, mode)
    assert caught.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE


def legacy_reference(space, batch, k, seed):
    """Independent pre-v2 scalar path: actual legacy augmenter and GP, no new fitter."""
    features = _PhysicalFeatures(space, batch.X)
    location, scale = float(batch.Y.to_numpy().mean()), float(batch.Y.to_numpy().std())
    sign = -1. if space.maximize[0] else 1.
    X = space.physical_to_model(batch.X)
    Y = torch.tensor(sign * (batch.Y.to_numpy() - location) / scale, dtype=torch.double)
    variance = torch.tensor(batch.Yvar.to_numpy() / scale**2, dtype=torch.double)
    runtime = compile_runtime(space, k, batch)
    with torch.random.fork_rng(devices=[]), settings.validate_input_scaling(False):
        torch.manual_seed(seed)
        augmenter = build_virtual_observation_augmenter(space, runtime["virtual_observations"])
        X, Y = augmenter.augment(X, Y, n_pairs_per_dim=3, bo_seed=seed)
        X = torch.cat((X[:5], space.physical_to_model(space.model_to_physical(X[5:]))))
        variance = torch.cat((variance, variance.mean().expand(len(X) - 5, 1)))
        model, _ = build_gp(space, k, X, Y, response_location=location, response_scale=scale,
            train_Yvar=variance, feature_transform=features, feature_groups=features.groups,
            runtime=runtime, normalize_outcome=False)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        model.eval()
    return FittedSurrogate(space, model, location, scale, sign, {}, ())


def test_v1_same_stack_independent_fit_and_recommendation_reference(monkeypatch):
    space, batch = data()
    k = Knowledge().with_monotone("x", "increases_objective", n_pairs_per_dim=3, epsilon=.05)
    envelope = k.to_envelope()
    state = rng_state()
    actual = fit_surrogate(space, batch, knowledge=k, seed=17)
    reference = legacy_reference(space, batch, k, 17)
    query = pd.DataFrame({"x": [.13, .37, .83]})
    np.testing.assert_array_equal(actual.predict(query).mean, reference.predict(query).mean)
    np.testing.assert_array_equal(actual.predict(query).variance, reference.predict(query).variance)
    a = recommend(space, batch, knowledge=k, candidates=query, candidate_ids=("p", "q", "r"), q=2, seed=17)
    monkeypatch.setattr("expdoe_dk.bo.recommendation.fit_surrogate",
                        lambda space, observations, *, knowledge, seed: legacy_reference(space, observations, knowledge, seed))
    b = recommend(space, batch, knowledge=k, candidates=query, candidate_ids=("p", "q", "r"), q=2, seed=17)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    np.testing.assert_array_equal(a.predictions.mean, b.predictions.mean)
    assert a.diagnostics["joint_acquisition_value"] == b.diagnostics["joint_acquisition_value"]
    assert k.to_envelope() == envelope
    assert_rng_equal(state, rng_state())


@pytest.mark.parametrize("multi", [False, True])
def test_exact_zero_likelihood_is_preserved_when_requested_by_existing_setting(multi):
    space, batch = data(multi=multi, variance=0.)
    with min_fixed_noise(double_value=0.):
        fitted = fit_surrogate(space, batch, knowledge=knowledge(), seed=13)
    base = scalar(fitted)._model._source
    assert torch.count_nonzero(base.likelihood.noise) == 0
    prediction = fitted.predict(batch.X)
    np.testing.assert_allclose(prediction.mean[:, 0], batch.Y.gain, atol=1e-10)
    np.testing.assert_array_equal(prediction.variance[:, 0], np.zeros(5))


@pytest.mark.parametrize("change", ["category", "dim", "kind", "version", "source", "confidence"])
def test_runtime_checks_directional_payload_even_when_inactive(change):
    space, _ = data()
    artifact = knowledge(confidence=0.).compile(space).virtual_observations[0].to_dict()
    category = "virtual_observations"
    if change == "category":
        category = "mean_components"
    elif change == "dim":
        artifact["payload"]["dim"] = 1
    elif change == "kind":
        space = Space([Parameter("x", kind="categorical", values=("a", "b"))], objectives=["gain"])
    elif change == "version":
        artifact["source_version"] = "1.0"
    elif change == "source":
        artifact["source_pattern"] = "other"
    else:
        artifact["payload"]["confidence"] = True
    with pytest.raises(EngineError) as caught:
        validate_runtime_artifact(space, category, OptimizationArtifact.from_dict(artifact))
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert "direction" in caught.value.details.get("pattern_ids", [caught.value.details.get("pattern_id")])


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("confidence,status", [(0., "inactive"), (.8, "not_evaluated")])
def test_capability_distinguishes_support_from_actual_conditioning(multi, confidence, status):
    space, _ = data(multi=multi)
    report = inspect_capabilities(space, knowledge(confidence=confidence))
    route = next(d for d in report.diagnostics if d.get("artifact_kind") == "monotone_sign_v2")
    assert route["conditioning_status"] == status


@pytest.mark.parametrize("pattern", ["power_law", "saturation"])
def test_registered_mean_kernel_and_prior_coexist_without_base_refitting(pattern):
    X = pd.DataFrame({"x": np.tile([.05, .2, .4, .6, .8, .95], 2),
                      "category": ["a"] * 6 + ["b"] * 6})
    space = Space([Parameter("x", bounds=(0. if pattern == "saturation" else .01, 1.)),
                   Parameter("category", kind="categorical", values=("a", "b"))], objectives="gain")
    Y = pd.DataFrame({"gain": 4. + 2. * X.x / (.5 + X.x)})
    batch = ObservationBatch(X, Y, Yvar=Y * 0 + .01)
    k = Knowledge().with_gp_prior("weak")
    k.add_spec(make_pattern_spec(pattern=pattern, version="2.0" if pattern == "saturation" else "1.0",
        pattern_id="mean", scope=KnowledgeScope(factors=("x",), objectives=("gain",)), confidence=1.,
        parameters={"direction": "increasing", "origin": 0., "half_response": .5}
        if pattern == "saturation" else {"exponent": 1., "scale": 2.}))
    k.add_spec(make_pattern_spec(pattern="category_similarity", version="1.0", pattern_id="kernel",
        confidence=1., scope=KnowledgeScope(factors=("category",)),
        parameters={"levels": ["a", "b"], "matrix": [[1., .3], [.3, 1.]]}))
    plain = fit_surrogate(space, batch, knowledge=k, seed=13)
    k.add_spec(knowledge().specs[0])
    fitted = fit_surrogate(space, batch, knowledge=k, seed=13)
    for name, parameter in plain._model.named_parameters():
        torch.testing.assert_close(dict(fitted._model._source.named_parameters())[name], parameter, rtol=0, atol=0)
    assert fitted.diagnostics["structured_mean"] == plain.diagnostics["structured_mean"]
    assert len(list(fitted._model._source.named_priors())) == 2
    assert np.isfinite(fitted.predict(X).mean).all()


def test_different_outputs_can_fit_v1_and_v2_without_composition_leakage():
    space, batch = data(multi=True)
    k = knowledge()
    k.add_spec(make_pattern_spec(pattern="monotone", version="1.0", pattern_id="old-cost", confidence=1.,
        scope=KnowledgeScope(factors=("x",), objectives=("cost",)),
        parameters={"direction": "increasing", "n_pairs_per_dim": 3, "epsilon": .05, "delta_norm": .5}))
    fitted = fit_surrogate(space, batch, knowledge=k, seed=13)
    assert scalar(fitted).transforms["virtual_rows"] == 0
    assert scalar(fitted, 1).transforms["virtual_rows"] == 6
    projected, selected, scoped = project_objective(space, batch, k, "cost")
    expected = fit_surrogate(projected, selected, knowledge=scoped, seed=13)
    np.testing.assert_array_equal(scalar(fitted, 1).predict(batch.X).mean, expected.predict(batch.X).mean)


@pytest.mark.parametrize("multi", [False, True])
def test_impossible_global_equality_fails_without_dropping_direction(multi):
    space, batch = data(multi=multi)
    k = knowledge().with_safe_region(expression="x == 0.5", factors=("x",))
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, batch, knowledge=k, seed=13)
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert caught.value.details["reason"] == "no_feasible_directional_pairs"
    assert caught.value.details["pattern_id"] == "direction"
    assert caught.value.details["accepted"] == 0
    if multi:
        assert caught.value.details["objective_name"] == "gain"


@pytest.mark.parametrize("multi", [False, True])
def test_conditioning_failure_is_typed_and_never_falls_back(monkeypatch, multi):
    from expdoe_dk.bo import directional_posterior
    space, batch = data(multi=multi)

    def fail(*args, **kwargs):
        raise EngineError(ErrorCode.MODEL_FIT_FAILED, "injected nonconvergence",
                          details={"reason": "nonconvergence"})

    monkeypatch.setattr(directional_posterior, "_fit_directional_laplace", fail)
    with pytest.raises(EngineError) as caught:
        recommend(space, batch, knowledge=knowledge(), seed=13)
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED
    assert caught.value.details["reason"] == "nonconvergence"
    if multi:
        assert caught.value.details["objective_name"] == "gain"


@pytest.mark.parametrize("kind,mode,variance", [("q_log_ehvi", "noiseless", 0.),
    ("q_log_nehvi", "known", .04), ("q_log_nparego", "inferred", None)])
def test_real_recommendations_keep_catalog_identity_and_original_variance(kind, mode, variance):
    space, batch = data(multi=True, variance=variance, safety=True)
    original = batch.Yvar
    catalog = pd.DataFrame({"x": [.051, .055, .245, .255, .495, .505, .945, .955]})
    ids = tuple(f"candidate-{i}" for i in range(len(catalog)))
    with min_fixed_noise(double_value=0.):
        result = recommend(space, batch, knowledge=knowledge(), candidates=catalog,
            candidate_ids=ids, q=2, seed=17,
            multiobjective=MultiObjectiveConfig(acquisition=kind, noise_mode=mode))
    assert len(set(result.candidate_ids)) == 2
    for identity, value in zip(result.candidate_ids, result.frame.x):
        assert value == catalog.x.iloc[ids.index(identity)]
    assert result.diagnostics["acquisition_kind"] == kind
    assert np.isfinite(result.predictions.mean).all()
    if variance is None:
        assert batch.Yvar is None
    else:
        pd.testing.assert_frame_equal(batch.Yvar, original)


def test_failed_measurements_and_heteroscedastic_variances_remain_audit_only():
    space, batch = data(multi=True)
    variance = pd.DataFrame({"gain": [0., .01, .03, .07, .1], "cost": [.1, .2, .3, .4, .5]})
    original = ObservationBatch(pd.concat([batch.X, pd.DataFrame({"x": [.99]})], ignore_index=True),
        pd.concat([batch.Y, pd.DataFrame({"gain": [None], "cost": [None]})], ignore_index=True),
        Yvar=pd.concat([variance, pd.DataFrame({"gain": [None], "cost": [None]})], ignore_index=True),
        status=pd.Series(["success"] * 5 + ["failed"]), ids=(*batch.ids, "failed"))
    with min_fixed_noise(double_value=0.):
        fitted = fit_surrogate(space, original, knowledge=knowledge(), seed=17)
    assert fitted.exclusions == ({"index": 5, "id": "failed", "reason": "failed"},)
    selected = scalar(fitted)
    assert selected._model._source.train_inputs[0].shape[0] == 5
    torch.testing.assert_close(selected._model._source.likelihood.noise,
        torch.tensor(variance.gain.to_numpy() / np.std(batch.Y.gain)**2), rtol=0, atol=0)
    assert selected.diagnostics["directional"]["actual_measurement_count"] == 5
    assert original.status.tolist() == ["success"] * 5 + ["failed"]
    assert original.Yvar.iloc[-1].isna().all()
