"""Real v2 MAP fits, with narrow optimizer-boundary fault injection only."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.optim.core import OptimizationStatus
from gpytorch.mlls import ExactMarginalLogLikelihood

from expdoe_dk import (EngineError, ErrorCode, Knowledge, KnowledgeScope, Objective,
    ObservationBatch, Parameter, Space, fit_surrogate, inspect_capabilities, make_pattern_spec)
from expdoe_dk.bo.gp import build_gp, build_mean_from_artifacts
from expdoe_dk.bo.knowledge_runtime import compile_runtime
from expdoe_dk.bo.structured_mean import StructuredMean


@pytest.fixture(autouse=True)
def small_matrix_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def declaration(pattern="saturation", factors=("x",), identity="shape", confidence=1.):
    parameters = {
        "saturation": {"direction": "increasing", "origin": 0., "half_response": .5},
        "threshold": {"threshold": .5, "below_behavior": "increasing", "above_behavior": "flat"},
        "synergy": {"reference": "anchored_additive", "response_effect": "higher"},
        "antagonism": {"reference": "anchored_additive", "response_effect": "higher"},
    }[pattern]
    return make_pattern_spec(pattern_id=identity, pattern=pattern, version="2.0",
        scope=KnowledgeScope(factors=factors, objectives=("response",)),
        parameters=parameters, confidence=confidence)


def case(pattern="saturation", *, mode="minimize", effect=2., confidence=1., known=False):
    objective = Objective("response", mode, target=4.) if mode == "target" else Objective("response", mode)
    space = Space([Parameter(n, bounds=(0., 1.)) for n in ("x", "z")], objectives=objective)
    X = pd.DataFrame([(x, z) for x in (0., .2, .4, .6, .8, 1.) for z in (0., 1.)], columns=space.param_names)
    x, z = X.x.to_numpy(), X.z.to_numpy()
    signal = {"saturation": x / (.5 + x), "threshold": np.minimum(x, .5),
              "synergy": x * z, "antagonism": -x * z}[pattern]
    y = 4. + effect * signal
    data = ObservationBatch(X, pd.DataFrame({"response": y}),
        Yvar=pd.DataFrame({"response": [.01] * len(X)}) if known else None)
    knowledge = Knowledge().add_spec(declaration(pattern,
        ("x", "z") if pattern in {"synergy", "antagonism"} else ("x",), confidence=confidence))
    return space, data, knowledge


def internal_fit(space, data, knowledge, **policy):
    from expdoe_dk.bo.structured_fit import fit_structured_gp
    from expdoe_dk.bo.surrogates import _PhysicalFeatures
    y = data.Y.to_numpy(dtype=float)
    location, scale = float(y.mean()), float(y.std()) or 1.
    sign = -1. if space.maximize[0] else 1.
    features = _PhysicalFeatures(space, data.X)
    return fit_structured_gp(space, knowledge, data, runtime=compile_runtime(space, knowledge, None),
        X=space.physical_to_model(data.X), Y=torch.tensor(sign * (y-location)/scale, dtype=torch.double),
        Yvar=None if data.Yvar is None else torch.tensor(data.Yvar.to_numpy()/scale**2, dtype=torch.double),
        feature_transform=features, feature_groups=features.groups,
        response_location=location, response_scale=scale, seed=13, **policy)


@pytest.mark.parametrize("pattern", ["saturation", "threshold", "synergy", "antagonism"])
def test_all_v2_patterns_are_wired_and_fitted_with_finite_audit(pattern):
    space, data, knowledge = case(pattern)
    report = inspect_capabilities(space, knowledge)
    assert set(report.applied) == {s.pattern_id for s in knowledge.specs if s.enabled}
    assert not report.blocked
    route = next(d for d in report.diagnostics if d.get("category") == "mean_components")
    assert route["coverage_status"] == route["empirical_evidence"] == "not_evaluated"
    fitted = fit_surrogate(space, data, knowledge=knowledge, seed=13)
    audit = fitted.diagnostics["structured_mean"]
    assert isinstance(fitted._model.mean_module, StructuredMean)
    assert audit["uncertainty"] == "conditional_on_MAP"
    assert audit["parameter_identifiability"] == "not_certified"
    assert audit["coverage_status"] == "passed"
    assert audit["seed"] == 13
    assert audit["response_location"] == pytest.approx(float(data.Y.mean().iloc[0]))
    assert audit["response_scale"] == pytest.approx(float(data.Y.to_numpy().std()))
    successful = [s for s in audit["starts"] if s["status"] == "converged"]
    assert audit["selected_start"] == max(successful, key=lambda s: s["objective"])["index"]
    assert audit["selected_objective"] == max(s["objective"] for s in successful)
    assert [(s["initial_magnitude"], s["initial_width"]) for s in audit["starts"]] == [(.05, .02), (.25, .05), (1., .15)]
    assert np.max(np.abs(fitted.predict(data.X).mean[:, 0] - data.Y.response)) < .15
    assert np.all(fitted.predict(data.X).variance >= 0)
    json.dumps(fitted.diagnostics, allow_nan=False)


def test_composed_means_share_one_baseline_and_deduplicate_nuisance():
    space, data, knowledge = case()
    knowledge.add_spec(declaration("threshold", ("z",), "threshold"))
    knowledge.add_spec(declaration("synergy", ("x", "z"), "interaction"))
    # Add independently sampled z levels to identify its threshold columns.
    X = pd.DataFrame([(x, z) for x in (0., .2, .4, .6, .8, 1.) for z in (0., .25, .75, 1.)], columns=["x", "z"])
    data = ObservationBatch(X, pd.DataFrame({"response": 4 + 2*X.x/(.5+X.x) + .8*np.minimum(X.z,.5) + .4*X.x*X.z}))
    fitted = fit_surrogate(space, data, knowledge=knowledge)
    mean = fitted._model.mean_module
    assert list(dict(mean.named_parameters())).count("baseline") == 1
    assert len(mean.components) == 3 and len(mean.nuisance) == 0
    assert set(inspect_capabilities(space, knowledge).applied) == {"shape", "threshold", "interaction"}
    assert len(fitted.diagnostics["structured_mean"]["components"]) == 3
    assert np.max(abs(fitted.predict(X).mean[:, 0] - data.Y.response)) < .15


@pytest.mark.parametrize("effect", [0., 1e-8])
def test_zero_and_weak_effects_do_not_require_nonzero_magnitude(effect):
    space, data, knowledge = case("threshold", effect=effect)
    fitted = fit_surrogate(space, data, knowledge=knowledge)
    audit = fitted.diagnostics["structured_mean"]
    assert audit["coverage_status"] == "passed"
    if effect == 0.:
        assert audit["response_scale"] == 1.
        assert abs(audit["components"][0]["normalized_below_slope"]) < 1e-3
        assert "width_unidentified_near_zero_effect" in audit["components"][0]["flags"]
    np.testing.assert_allclose(fitted.predict(data.X).mean[:, 0], data.Y.response, atol=1e-3)


def test_known_variance_and_gp_prior_and_similarity_coexist():
    space, data, knowledge = case(known=True)
    space = Space(list(space.params) + [Parameter("category", kind="categorical", values=("a", "b"))], objectives="response")
    X = data.X.assign(category=["a", "b"] * 6)
    data = ObservationBatch(X, data.Y, Yvar=data.Yvar)
    knowledge.with_gp_prior("weak").add_spec(make_pattern_spec(pattern_id="similarity",
        pattern="category_similarity", version="1.0", confidence=1.,
        scope=KnowledgeScope(factors=("category",)),
        parameters={"levels": ["a", "b"], "matrix": [[1., .3], [.3, 1.]]}))
    fitted = fit_surrogate(space, data, knowledge=knowledge)
    torch.testing.assert_close(fitted._model.likelihood.noise,
        torch.full((12,), .01 / data.Y.to_numpy().std()**2, dtype=torch.double))
    assert len(list(fitted._model.named_priors())) == 2  # lengthscale + outputscale; no learned noise
    assert not list(fitted._model.mean_module.named_priors())
    assert np.isfinite(fitted.predict(X).variance).all()


def test_objective_is_total_exact_mll_plus_each_prior_once():
    space, data, knowledge = case(known=True)
    knowledge.with_gp_prior("weak")
    fitted = fit_surrogate(space, data, knowledge=knowledge)
    model = fitted._model
    model.train()
    with torch.no_grad():
        output = model(*model.train_inputs)
        # Independent oracle: Gaussian marginal log density and the two Gamma priors.
        expected = model.likelihood(output).log_prob(model.train_targets)
        for _, module, prior, closure, _ in model.named_priors():
            expected = expected + prior.log_prob(closure(module)).sum()
        expected = expected + model.mean_module.log_prior()
        scaled_mll = ExactMarginalLogLikelihood(model.likelihood, model)(output, model.train_targets) * 12
        assert expected.item() == pytest.approx((scaled_mll + model.mean_module.log_prior()).item())
    audit = fitted.diagnostics["structured_mean"]
    assert audit["selected_objective"] == pytest.approx(expected.item(), rel=1e-10)
    assert audit["prior_policy"]["objective"] == "total_exact_mll_with_gp_priors_plus_structured_log_prior"


def test_repeat_seed_and_objective_direction_preserve_raw_physical_predictions():
    predictions, audits = [], []
    for mode in ("minimize", "maximize", "target", "minimize"):
        space, data, knowledge = case(mode=mode)
        fitted = fit_surrogate(space, data, knowledge=knowledge, seed=17)
        predictions.append(fitted.predict(data.X))
        audits.append(fitted.diagnostics)
    assert audits[0] == audits[-1]
    for prediction in predictions[1:]:
        np.testing.assert_allclose(prediction.mean, predictions[0].mean, atol=1e-7)
        np.testing.assert_allclose(prediction.variance, predictions[0].variance, atol=1e-7)


def test_confidence_and_internal_policy_change_prior_without_changing_public_api():
    strong = case(confidence=1., known=True)
    weak = case(confidence=.05, known=True)
    _, first = internal_fit(*strong)
    _, second = internal_fit(*weak)
    _, changed = internal_fit(*strong, magnitude_scale_multiplier=.5, width_prior_center=.1)
    assert first["components"][0]["prior"]["magnitude_scale"] == 1.
    assert second["components"][0]["prior"]["magnitude_scale"] == pytest.approx(.145)
    assert changed["components"][0]["prior"]["magnitude_scale"] == .5
    assert changed["prior_policy"]["width_prior_center"] == .1
    assert first["selected_objective"] != second["selected_objective"]
    assert first["components"][0]["normalized_magnitude"] > second["components"][0]["normalized_magnitude"]


def test_ineligible_width_is_not_optimized_and_models_are_fresh(monkeypatch):
    import expdoe_dk.bo.structured_fit as fitting
    space, _, knowledge = case("threshold")
    knowledge.add_spec(declaration("threshold", ("z",), "second"))
    X = pd.DataFrame([(0.,0.), (.01,.01), (.02,.02), (.98,.95), (.99,.97), (1.,1.)], columns=["x", "z"])
    data = ObservationBatch(X, pd.DataFrame({"response": [4.,4.,4.,4.,4.,4.]}))
    original = fitting.fit_gpytorch_mll_scipy
    seen = []
    def spy(mll, **kwargs):
        seen.append(mll.model)
        return original(mll, **kwargs)
    monkeypatch.setattr(fitting, "fit_gpytorch_mll_scipy", spy)
    fitted = fit_surrogate(space, data, knowledge=knowledge)
    audit = fitted.diagnostics["structured_mean"]
    assert audit["eligible_starts"] == [1, 2]
    assert audit["starts"][0]["status"] == "ineligible"
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert audit["selected_start"] in (1, 2)


@pytest.mark.parametrize("fault", ["nan_result", "nan_parameter", "nonconverged", "exception"])
def test_failed_optimizer_starts_are_excluded_from_selection(monkeypatch, fault):
    import expdoe_dk.bo.structured_fit as fitting
    original = fitting.fit_gpytorch_mll_scipy
    calls = []
    def spy(mll, **kwargs):
        result = original(mll, **kwargs)
        calls.append(mll.model)
        if len(calls) == 1:
            if fault == "nan_result":
                return replace(result, fval=float("nan"))
            if fault == "nan_parameter":
                with torch.no_grad():
                    mll.model.mean_module.baseline.fill_(float("nan"))
            if fault == "nonconverged":
                return replace(result, status=OptimizationStatus.STOPPED, fval=-1e100)
            if fault == "exception":
                raise RuntimeError("injected optimizer failure")
        return result
    monkeypatch.setattr(fitting, "fit_gpytorch_mll_scipy", spy)
    space, data, knowledge = case()
    fitted = fit_surrogate(space, data, knowledge=knowledge)
    audit = fitted.diagnostics["structured_mean"]
    assert audit["selected_start"] in (1, 2)
    assert audit["starts"][0]["status"] == "failed"
    assert audit["starts"][0]["reason"]
    assert len({id(model) for model in calls}) == 3
    json.dumps(fitted.diagnostics, allow_nan=False)


def test_all_optimizer_starts_fail_closed_without_an_unfit_model(monkeypatch):
    import expdoe_dk.bo.structured_fit as fitting
    original = fitting.fit_gpytorch_mll_scipy
    def spy(mll, **kwargs):
        return replace(original(mll, **kwargs), status=OptimizationStatus.FAILURE, message="forced failure")
    monkeypatch.setattr(fitting, "fit_gpytorch_mll_scipy", spy)
    space, data, knowledge = case()
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, data, knowledge=knowledge)
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED
    assert len(caught.value.details["starts"]) == 3
    assert all(start["reason"] for start in caught.value.details["starts"])
    json.dumps(caught.value.details, allow_nan=False)


def test_bad_coverage_fails_before_any_optimizer_and_structural_compile_stays_available(monkeypatch):
    import expdoe_dk.bo.structured_fit as fitting
    def forbidden(*args, **kwargs):
        pytest.fail("coverage failures must precede optimization")
    monkeypatch.setattr(fitting, "fit_gpytorch_mll_scipy", forbidden)
    space, data, knowledge = case()
    sparse = ObservationBatch(data.X.iloc[:6], data.Y.iloc[:6])
    assert compile_runtime(space, knowledge, None)["mean_components"]
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, sparse, knowledge=knowledge)
    assert caught.value.code == ErrorCode.INSUFFICIENT_DATA
    assert caught.value.details["coverage_status"] == "deficient"


def test_all_rank_deficient_starts_fail_before_optimization(monkeypatch):
    import expdoe_dk.bo.structured_fit as fitting
    def forbidden(*args, **kwargs):
        pytest.fail("all deficient basis starts must precede optimization")
    monkeypatch.setattr(fitting, "fit_gpytorch_mll_scipy", forbidden)
    space = Space([Parameter(n, bounds=(0., 1.)) for n in ("x", "z", "t")], objectives="response")
    X = pd.DataFrame([(x, z, x) for x, z in [(0.,0.), (0.,1.), (1.,0.), (1.,1.), (.2,.4), (.8,.6)]], columns=space.param_names)
    data = ObservationBatch(X, pd.DataFrame({"response": np.zeros(6)}))
    knowledge = Knowledge().add_spec(declaration("synergy", ("x", "z"), "xz"))
    knowledge.add_spec(declaration("synergy", ("z", "t"), "zt"))
    with pytest.raises(EngineError) as caught:
        fit_surrogate(space, data, knowledge=knowledge)
    assert caught.value.code == ErrorCode.INSUFFICIENT_DATA
    assert caught.value.details["eligible_starts"] == []
    assert len(caught.value.details["starts"]) == 3
    assert all(item["status"] == "passed" for item in caught.value.details["coverage"])


@pytest.mark.parametrize("transform", ["linear", "log"])
def test_coverage_and_fit_preserve_adjacent_large_integer_physical_levels(transform):
    low = 2**53
    space = Space([Parameter("x", kind="integer", bounds=(low, low+10), transform=transform)], objectives="response")
    data = ObservationBatch(pd.DataFrame({"x": [low+i for i in (0,1,2,6,8,10)]}),
                            pd.DataFrame({"response": [4.] * 6}))
    knowledge = Knowledge().with_saturation("x", direction="increasing", origin=low,
        half_response=low+4, version="2.0", objective="response")
    fitted = fit_surrogate(space, data, knowledge=knowledge)
    audit = fitted.diagnostics["structured_mean"]
    assert audit["coverage_status"] == "passed"
    assert audit["coverage"][0]["left_distinct"] == 3
    assert audit["coverage"][0]["right_distinct"] == 3
    assert audit["eligible_starts"] == [0, 1, 2]
    assert audit["components"][0]["anchors"] == [low]
    np.testing.assert_allclose(fitted.predict(data.X).mean[:, 0], 4., atol=1e-3)


def test_successful_rows_only_supply_coverage_and_response_normalization():
    space, data, knowledge = case()
    X = pd.concat([data.X, pd.DataFrame({"x": [.3], "z": [.3]})], ignore_index=True)
    Y = pd.concat([data.Y, pd.DataFrame({"response": [1e9]})], ignore_index=True)
    data_with_warning = ObservationBatch(X, Y, status=pd.Series(["success"]*12 + ["warning"]))
    fitted = fit_surrogate(space, data_with_warning, knowledge=knowledge)
    audit = fitted.diagnostics["structured_mean"]
    assert audit["successful_rows"] == 12
    assert audit["response_location"] == pytest.approx(data.Y.response.mean())
    assert fitted.exclusions[0]["index"] == 12


def test_direct_build_requires_normalization_and_preserves_invalid_artifact_error():
    space, data, knowledge = case()
    X = space.physical_to_model(data.X)
    Y = torch.tensor(data.Y.to_numpy(), dtype=torch.double)
    with pytest.raises(EngineError) as caught:
        build_gp(space, knowledge, X, Y)
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    artifact, = knowledge.compile(space).mean_components
    payload = artifact.to_dict()["payload"]
    payload["coordinates"][0]["range"] = 10.
    object.__setattr__(artifact, "payload", payload)
    with pytest.raises(EngineError) as caught:
        build_mean_from_artifacts(space, (artifact,), response_location=0., response_scale=1.)
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID


def test_diagnostics_are_defensive_and_legacy_section_is_empty():
    space, data, knowledge = case()
    fitted = fit_surrogate(space, data, knowledge=knowledge)
    changed = fitted.diagnostics
    changed["structured_mean"]["components"][0]["parameters"]["direction"] = "decreasing"
    changed["structured_mean"]["starts"].clear()
    assert fitted.diagnostics["structured_mean"]["components"][0]["parameters"]["direction"] == "increasing"
    assert len(fitted.diagnostics["structured_mean"]["starts"]) == 3
    assert fit_surrogate(space, data).diagnostics["structured_mean"] == {}
