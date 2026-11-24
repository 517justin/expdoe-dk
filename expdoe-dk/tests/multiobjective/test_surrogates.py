"""Named physical GPs and joint posterior sampling, without utility folding."""
import builtins
import importlib

import numpy as np
import pandas as pd
import pytest
import torch

from expdoe_dk import (
    EngineError, ErrorCode, Knowledge, KnowledgeScope, Objective, ObservationBatch,
    Parameter, Space, evaluate_models, fit_surrogate, make_pattern_spec,
)
from expdoe_dk.bo import surrogates
from expdoe_dk.bo.objective_routing import project_objective


@pytest.fixture
def data():
    space = Space([Parameter("x", bounds=(0., 4.))], objectives=(
        Objective("response", "target", target=5.), Objective("cost", "minimize")))
    batch = ObservationBatch(pd.DataFrame({"x": [0., 1., 2., 3., 4.]}),
        pd.DataFrame({"cost": [120., 110., 100., 90., 80.], "response": [3., 4., 5., 6., 7.]}),
        Yvar=pd.DataFrame({"cost": [1., 2., 3., 4., 5.], "response": [.01] * 5}),
        ids=("a", "b", "c", "d", "e"))
    return space, batch


def test_named_outputs_align_y_and_yvar(data):
    space, batch = data
    fitted = fit_surrogate(space, batch, seed=13)
    prediction = fitted.predict(batch.X)
    assert prediction.mean.shape == prediction.variance.shape == (5, 2)
    assert fitted.diagnostics["objective_names"] == ["response", "cost"]
    for i, name in enumerate(space.objectives):
        projected, single, knowledge = project_objective(space, batch, Knowledge(), name)
        expected = fit_surrogate(projected, single, knowledge=knowledge, seed=13).predict(batch.X)
        np.testing.assert_allclose(prediction.mean[:, i], expected.mean[:, 0], atol=1e-9)
        np.testing.assert_allclose(prediction.variance[:, i], expected.variance[:, 0], atol=1e-9)
    adapter = fitted.acquisition_model()
    torch.testing.assert_close(adapter.models[0]._source.likelihood.noise,
                               torch.full((5,), .005, dtype=torch.double))
    torch.testing.assert_close(adapter.models[1]._source.likelihood.noise,
                               torch.tensor([.005, .01, .015, .02, .025], dtype=torch.double))
    assert fitted.transforms["objectives"]["response"]["response_scale"] == pytest.approx(np.sqrt(2))
    assert fitted.transforms["objectives"]["cost"]["response_scale"] == pytest.approx(np.sqrt(200))
    diagnostics = fitted.diagnostics
    diagnostics["objective_names"].clear()
    assert fitted.diagnostics["objective_names"] == ["response", "cost"]
    empty = fitted.predict(batch.X.iloc[:0])
    assert empty.mean.shape == empty.variance.shape == (0, 2)


@pytest.mark.parametrize("dtype", [object, "Float64"])
def test_real_fit_and_projection_retain_missing_unsuccessful_variances(data, dtype):
    space, source = data
    X = pd.concat([source.X, source.X.iloc[:3]], ignore_index=True)
    Y = pd.concat([source.Y, pd.DataFrame({"cost": [None] * 3, "response": [None] * 3})], ignore_index=True)
    variance = pd.DataFrame({"response": pd.Series([.01] * 5 + [None, np.nan, pd.NA], dtype=dtype),
                             "cost": pd.Series([1., 2., 3., 4., 5., pd.NA, None, np.nan], dtype=dtype)})
    batch = ObservationBatch(X, Y, Yvar=variance,
        status=pd.Series(["success"] * 5 + ["failed", "warning", "failed"]),
        ids=(*source.ids, "f", "w", "f2"))
    projected, selected, knowledge = project_objective(space, batch, Knowledge(), "response")
    assert selected.ids == batch.ids
    if dtype is object:
        assert selected.Yvar.response.iloc[5] is None
        assert np.isnan(selected.Yvar.response.iloc[6])
    else:
        assert selected.Yvar.response.iloc[5] is pd.NA
        assert selected.Yvar.response.iloc[6] is pd.NA
    assert selected.Yvar.response.iloc[7] is pd.NA
    pd.testing.assert_frame_equal(selected.successful().Yvar, variance[["response"]].iloc[:5])
    fitted = fit_surrogate(space, batch, seed=13)
    reference = fit_surrogate(space, source, seed=13)
    prediction, expected = fitted.predict(source.X), reference.predict(source.X)
    np.testing.assert_allclose(prediction.mean, expected.mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(prediction.variance, expected.variance, rtol=1e-10, atol=1e-10)
    single = fit_surrogate(projected, selected, knowledge=knowledge, seed=13).predict(source.X)
    np.testing.assert_allclose(single.mean[:, 0], expected.mean[:, 0], rtol=1e-10, atol=1e-10)
    assert [row["id"] for row in fitted.exclusions] == ["f", "w", "f2"]
    pd.testing.assert_frame_equal(batch.Yvar, variance[source.Y.columns])


def test_target_model_preserves_both_sides(data):
    space, batch = data
    fitted = fit_surrogate(space, batch)
    source = fitted.acquisition_model().models[0]._source
    torch.testing.assert_close(-source.train_targets * np.sqrt(2) + 5,
                               torch.tensor([3., 4., 5., 6., 7.], dtype=torch.double))
    physical = fitted.predict(batch.X).mean[:, 0]
    assert physical[0] < 3.2
    assert physical[-1] > 6.8
    np.testing.assert_allclose(physical, [3., 4., 5., 6., 7.], atol=.2)


def test_affine_response_variance_and_constant_outputs(data):
    space, batch = data
    fitted = fit_surrogate(space, batch, seed=7)
    scaled = fit_surrogate(space, ObservationBatch(batch.X, batch.Y * 10 + 17,
                                                    Yvar=batch.Yvar * 100), seed=7)
    p, s = fitted.predict(batch.X), scaled.predict(batch.X)
    np.testing.assert_allclose(s.mean, p.mean * 10 + 17, rtol=1e-6)
    np.testing.assert_allclose(s.variance, p.variance * 100, rtol=1e-6)
    constant = fit_surrogate(space, ObservationBatch(batch.X,
        pd.DataFrame({"cost": [.1] * 5, "response": [5.] * 5})))
    prediction = constant.predict(batch.X)
    np.testing.assert_allclose(prediction.mean, np.tile([5., .1], (5, 1)), atol=1e-8)
    assert np.isfinite(prediction.variance).all()
    assert (prediction.variance >= 0).all()
    assert all(t["response_scale"] == 1. for t in constant.transforms["objectives"].values())


def test_model_list_samples_are_physical_joint_q_and_differentiable(data):
    from botorch.models.model import ModelList
    from botorch.posteriors.posterior_list import PosteriorList
    from botorch.sampling.list_sampler import ListSampler

    space, batch = data
    fitted = fit_surrogate(space, batch, seed=3)
    model = fitted.acquisition_model()
    assert isinstance(model, ModelList)
    X = torch.tensor([[[.13], [.13001], [.81]], [[.21], [.21001], [.73]]],
                     dtype=torch.double, requires_grad=True)
    posterior = model.posterior(X)
    assert isinstance(posterior, PosteriorList)
    sampler = fitted.acquisition_sampler(sample_shape=torch.Size([16384]), seed=19)
    assert isinstance(sampler, ListSampler)
    samples = sampler(posterior)
    assert samples.shape == (16384, 2, 3, 2)
    expected = fitted.predict(space.model_to_physical(X.detach().reshape(-1, 1)))
    np.testing.assert_allclose(posterior.mean.detach().numpy().reshape(-1, 2), expected.mean)
    np.testing.assert_allclose(posterior.variance.detach().numpy().reshape(-1, 2), expected.variance)
    torch.testing.assert_close(samples.mean(0), posterior.mean, atol=.02, rtol=.002)
    torch.testing.assert_close(samples.var(0), posterior.variance, atol=.001, rtol=.03)
    for output in range(2):
        covariance = torch.cov(samples[:, 0, :, output].detach().T)
        expected_covariance = (model.models[output]._source.posterior(X[:1]).distribution
                               .covariance_matrix[0].detach() * model.models[output]._scale**2)
        # Gaussian sample-covariance error scales with marginal variances, not
        # with the (possibly near-zero) true off-diagonal covariance.
        diagonal = expected_covariance.diag()
        standard_error = ((diagonal[:, None] * diagonal[None, :] + expected_covariance.square())
                          / (len(samples) - 1)).sqrt()
        assert ((covariance - expected_covariance).abs() / standard_error).max() < 5
        correlation = torch.corrcoef(samples[:, 0, :2, output].detach().T)[0, 1]
        assert correlation > .999  # Independent marginal samples cannot pass.
    assert torch.corrcoef(samples[:, 0, 0, :].detach().T)[0, 1].abs() < .1
    samples.square().mean().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    assert X.grad.abs().max() > 0
    repeated = fitted.acquisition_sampler(sample_shape=torch.Size([16384]), seed=19)(posterior)
    torch.testing.assert_close(samples, repeated)
    assert not torch.equal(samples, fitted.acquisition_sampler(
        sample_shape=torch.Size([16384]), seed=20)(posterior))
    shaped = fitted.acquisition_sampler(sample_shape=torch.Size([4, 8]), seed=19)(posterior)
    assert shaped.shape == (4, 8, 2, 3, 2)


def test_partial_fit_failure_is_atomic_and_names_objective(data, monkeypatch):
    real_fit = surrogates.fit_gpytorch_mll
    calls = []

    def fail_second(mll, *args, **kwargs):
        calls.append(mll)
        if len(calls) == 2:
            raise RuntimeError("second optimizer failed")
        return real_fit(mll, *args, **kwargs)

    monkeypatch.setattr(surrogates, "fit_gpytorch_mll", fail_second)
    with pytest.raises(EngineError) as caught:
        fit_surrogate(*data)
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED
    assert caught.value.details["objective_name"] == "cost"
    assert caught.value.details["training_ids"] == ["a", "b", "c", "d", "e"]
    assert len(calls) == 2


def test_global_validation_then_scoped_real_fits(data, monkeypatch):
    space, batch = data
    space = Space([Parameter("x", bounds=(1., 5.))], objectives=space.objective_specs)
    batch = ObservationBatch(batch.X + 1, batch.Y, Yvar=batch.Yvar)
    knowledge = Knowledge().add_spec(make_pattern_spec(pattern="power_law", version="1.0",
        pattern_id="response-mean", confidence=1.,
        scope=KnowledgeScope(factors=("x",), objectives=("response",)),
        parameters={"scale": 1., "exponent": 1.}))
    knowledge.add_spec(make_pattern_spec(pattern="tradeoff", version="1.0", confidence=1.,
        scope=KnowledgeScope(objectives=("response", "cost")),
        parameters={"weights": [1., 2.]}))
    original = surrogates.compile_runtime
    seen = []

    def record(space, knowledge, observations):
        runtime = original(space, knowledge, observations)
        seen.append((space.objectives, runtime))
        return runtime

    monkeypatch.setattr(surrogates, "compile_runtime", record)
    fitted = fit_surrogate(space, batch, knowledge=knowledge)
    assert [names for names, _ in seen] == [["response", "cost"], ["response"], ["cost"]]
    assert seen[0][1]["acquisition_preferences"]
    assert [a.source_pattern_id for a in seen[1][1]["mean_components"]] == ["response-mean"]
    assert not seen[2][1]["mean_components"]
    assert all(not runtime["acquisition_preferences"] for _, runtime in seen[1:])
    assert np.isfinite(fitted.predict(batch.X).mean).all()


def test_exclusions_and_minimum_successes_apply_to_all_outputs(data):
    space, batch = data
    Y = batch.Y
    Y.iloc[1] = np.nan
    observations = ObservationBatch(batch.X, Y, status=pd.Series(
        ["success", "failed", "success", "warning", "success"]), ids=batch.ids)
    fitted = fit_surrogate(space, observations)
    assert fitted.exclusions == ({"index": 1, "id": "b", "reason": "failed"},
                                 {"index": 3, "id": "d", "reason": "warning"})
    assert fitted.transforms["objectives"]["cost"]["response_location"] == 100.
    assert fitted.predict(batch.X).mean.shape == (5, 2)
    insufficient = ObservationBatch(batch.X, Y, status=pd.Series(
        ["success", "failed", "success", "warning", "failed"]))
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, insufficient)
    assert caught.value.code == ErrorCode.INSUFFICIENT_DATA


@pytest.mark.parametrize("columns", [["cost"], ["cost", "response", "extra"], ["cost", "unknown"]])
def test_mismatched_response_names_fail_closed(data, columns):
    space, batch = data
    wrong = ObservationBatch(batch.X, pd.DataFrame({name: np.arange(5.) for name in columns}))
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, wrong)
    assert caught.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("module,symbol", [
    ("botorch.models.model", "ModelList"),
    ("botorch.sampling.list_sampler", "ListSampler"),
    ("botorch.sampling.normal", "IIDNormalSampler"),
])
def test_missing_multioutput_apis_are_lazy_typed_capability_errors(data, monkeypatch, module, symbol):
    real_import = builtins.__import__

    def missing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module and symbol in fromlist:
            raise ImportError("simulated old BoTorch")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", missing)
        importlib.reload(surrogates)
        space, batch, knowledge = project_objective(*data, Knowledge(), "response")
        assert surrogates.fit_surrogate(space, batch, knowledge=knowledge).predict(batch.X).mean.shape == (5, 1)
        with pytest.raises(EngineError) as caught:
            surrogates.fit_surrogate(*data)
        assert caught.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
        assert "multioutput" in caught.value.message.lower()
        assert caught.value.hints
    importlib.reload(surrogates)


def test_old_sampler_signature_is_typed_capability_error(data, monkeypatch):
    import botorch.sampling.normal as normal

    class OldSampler:
        def __init__(self, num_samples, seed=None):
            pass

    monkeypatch.setattr(normal, "IIDNormalSampler", OldSampler)
    with pytest.raises(EngineError) as caught:
        fit_surrogate(*data)
    assert caught.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE


def test_evaluation_rejects_multioutput_at_entry(data, monkeypatch):
    import expdoe_dk.bo.evaluation as evaluation

    def forbidden(*args, **kwargs):
        pytest.fail("evaluation entered scalar processing before checking objective count")

    monkeypatch.setattr(evaluation, "_eligible", forbidden)
    with pytest.raises(EngineError) as caught:
        evaluate_models(*data, tracks={"gp": None})
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert "one objective" in caught.value.message
