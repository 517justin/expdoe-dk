"""Physical safety and fixed-geometry multiobjective log acquisitions."""
import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import ndtr

from expdoe_dk import (
    EngineError, ErrorCode, Knowledge, MultiObjectiveConfig, Objective,
    ObservationBatch, OutcomeConstraint, Parameter, PendingBatch, Space, fit_surrogate,
)
from expdoe_dk.bo.contracts import Prediction
from expdoe_dk.bo.knowledge_runtime import compile_runtime
from expdoe_dk.bo.objective_geometry import resolve_geometry


def api():
    return importlib.import_module("expdoe_dk.bo.acquisition")


@pytest.fixture(scope="module", autouse=True)
def isolated_optional_extension_cache(tmp_path_factory):
    # Newer BoTorch may JIT an optional accelerator. Do not inherit another
    # process's build lock; numerical Python fallback remains BoTorch's own.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path_factory.mktemp("torch-extensions")))
        yield


def data(m=2, variance=None, constraints=()):
    space = Space([Parameter("x", bounds=(0., 4.))],
                  objectives=[Objective(f"y{i}", "maximize") for i in range(m)],
                  outcome_constraints=constraints)
    X = pd.DataFrame({"x": [0., 1., 2., 3., 4.]})
    Y = pd.DataFrame({f"y{i}": [10. + i, 12. + i, 11. + i, 14. + i, 13. + i]
                      for i in reversed(range(m))})
    batch = ObservationBatch(X, Y, Yvar=None if variance is None else Y * 0 + variance)
    return space, batch


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("mode", ["inferred", "known", "noiseless"])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("preferences", [False, True])
def test_auto_route_matrix(m, mode, pending, preferences):
    decision = api().choose_multiobjective_acquisition(
        m, MultiObjectiveConfig(noise_mode=mode),
        has_pending=pending, has_preferences=preferences)
    expected = ("q_log_nparego" if preferences or m >= 5 else
                "q_log_ehvi" if mode == "noiseless" and not pending else "q_log_nehvi")
    assert decision.kind == expected
    if preferences:
        assert decision.reason == "preference_override"


@pytest.mark.parametrize("mode,variance,expected", [
    ("auto", None, "inferred"), ("auto", 0., "known"), ("auto", .2, "known"),
    ("inferred", None, "inferred"), ("known", .2, "known"),
    ("noiseless", None, "noiseless"), ("noiseless", 0., "noiseless"),
    ("known", None, None), ("inferred", .2, None), ("inferred", 0., None),
    ("noiseless", .2, None),
])
def test_noise_resolution_preserves_original_metadata(mode, variance, expected):
    from expdoe_dk.bo import multiobjective
    _, observations = data(variance=variance)
    original = observations.Yvar
    if expected is None:
        with pytest.raises(EngineError) as caught:
            multiobjective.resolve_noise_mode(observations, MultiObjectiveConfig(noise_mode=mode))
        assert caught.value.code == ErrorCode.CONFIG_INVALID
    else:
        assert multiobjective.resolve_noise_mode(
            observations, MultiObjectiveConfig(noise_mode=mode)) == expected
    if original is None:
        assert observations.Yvar is None
    else:
        pd.testing.assert_frame_equal(observations.Yvar, original)


@pytest.mark.parametrize("kind,mode,pending,preferences", [
    ("q_log_ehvi", "inferred", False, False),
    ("q_log_ehvi", "known", False, False),
    ("q_log_ehvi", "auto", False, False),
    ("q_log_ehvi", "noiseless", True, False),
    ("q_log_ehvi", "noiseless", False, True),
    ("q_log_nehvi", "known", False, True),
])
def test_incompatible_overrides_fail_closed(kind, mode, pending, preferences):
    config = MultiObjectiveConfig(acquisition=kind, noise_mode=mode,
        preference_weights={"y0": .8, "y1": .2} if preferences else None)
    with pytest.raises(EngineError) as caught:
        api().choose_multiobjective_acquisition(2, config,
            has_pending=pending, has_preferences=preferences)
    assert caught.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE


class FixedPrediction:
    """Analytic physical marginals for interval-probability boundary cases."""
    def __init__(self, mean, variance):
        self.prediction = Prediction(np.asarray(mean, float), np.asarray(variance, float))

    def predict(self, X):
        assert len(X) == len(self.prediction.mean)
        return self.prediction


def test_joint_feasibility_intersects_each_output_once():
    constraints = [OutcomeConstraint("lo", "y0", ">=", -1.),
                   OutcomeConstraint("hi", "y0", "<=", 1.),
                   OutcomeConstraint("loose", "y0", "<=", 2.),
                   OutcomeConstraint("other", "y1", "<=", 0.)]
    space, _ = data(constraints=constraints)
    fitted = FixedPrediction([[0., 0.]], [[1., 1.]])
    actual = api().joint_feasibility(space, fitted, pd.DataFrame({"x": [1.]}))
    np.testing.assert_allclose(actual, [(ndtr(1.) - ndtr(-1.)) * .5])
    assert actual[0] != pytest.approx(ndtr(1.) ** 2 * ndtr(2.) * .5)


def test_feasibility_zero_variance_and_positive_tail():
    space, _ = data(constraints=[OutcomeConstraint("lo", "y0", ">=", 9.),
                                OutcomeConstraint("hi", "y0", "<=", 10.)])
    fitted = FixedPrediction([[9., 0.], [10., 0.], [11., 0.], [0., 0.]],
                             [[0., 1.], [0., 1.], [0., 1.], [1., 1.]])
    result = api().joint_feasibility(space, fitted, pd.DataFrame({"x": [0., 1., 2., 3.]}))
    np.testing.assert_allclose(result[:3], [1., 1., 0.])
    assert result[3] == pytest.approx(ndtr(-9.) - ndtr(-10.), rel=1e-12, abs=0.)


@pytest.mark.parametrize("constraints,code", [
    ([OutcomeConstraint("lo", "y0", ">=", 2.), OutcomeConstraint("hi", "y0", "<=", 1.)],
     ErrorCode.CONFIG_INVALID),
    ([OutcomeConstraint("eq", "y0", "==", 1.)], ErrorCode.ACQUISITION_INCOMPATIBLE),
    ([OutcomeConstraint("soft", "y0", "<=", 1., hard=False, penalty="linear", weight=1.)],
     ErrorCode.ACQUISITION_INCOMPATIBLE),
])
def test_unsupported_safety_fails_before_prediction(constraints, code):
    space, batch = data(constraints=constraints)
    with pytest.raises(EngineError) as caught:
        api().joint_feasibility(space, None, batch.X)
    assert caught.value.code == code


@pytest.fixture(scope="module")
def real_models():
    models = {}
    for mode in ("inferred", "known", "noiseless"):
        space, original = data(variance=.04 if mode == "known" else None,
            constraints=[OutcomeConstraint("safety", "y0", "<=", 13.5)])
        model_batch = (ObservationBatch(original.X, original.Y, Yvar=original.Y * 0)
                       if mode == "noiseless" else original)
        models[mode] = (space, original, fit_surrogate(space, model_batch, seed=11))
    return models


def build(real_models, kind, mode, *, pending=None, weights=None, slot=0, seed=23):
    space, batch, fitted = real_models[mode]
    config = MultiObjectiveConfig(acquisition=kind, noise_mode=mode, preference_weights=weights)
    runtime = compile_runtime(space, Knowledge(), batch)
    geometry = resolve_geometry(space, batch, runtime, config)
    acquisition, diagnostics = api().build_multiobjective_acquisition(
        space, fitted, batch, pending, runtime, config, geometry, seed=seed, slot=slot)
    return acquisition, diagnostics, geometry


@pytest.mark.parametrize("kind,mode", [
    ("q_log_ehvi", "noiseless"), ("q_log_nehvi", "inferred"),
    ("q_log_nehvi", "known"), ("q_log_nparego", "known"),
])
def test_real_log_acquisitions_forward_backward_and_greedy_pending(real_models, kind, mode):
    external = None if kind == "q_log_ehvi" else PendingBatch(("pending",), pd.DataFrame({"x": [.7]}))
    acq, diagnostics, geometry = build(real_models, kind, mode, pending=external)
    assert diagnostics["acquisition_kind"] == kind
    assert diagnostics["noise_mode"] == mode
    assert diagnostics["geometry"] == geometry.to_dict()
    assert acq.sampler is not None
    if external is not None:
        # Newer qLogNEI folds pending into its incumbent baseline; older
        # versions keep X_pending. Both must condition on the same coordinates.
        pending_coordinates = acq.X_pending if acq.X_pending is not None else acq.X_baseline[-1:]
        torch.testing.assert_close(pending_coordinates, torch.tensor([[.175]], dtype=torch.double))
    # Internal greedy points do not reroute initially valid EHVI.
    acq.set_X_pending(torch.tensor([[.175], [.41]], dtype=torch.double))
    X = torch.tensor([[[.32]], [[.61]]], dtype=torch.double, requires_grad=True)
    values = acq(X)
    assert values.shape == (2,) and torch.isfinite(values).all()
    values.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all() and X.grad.abs().max() > 0
    constraints = acq._constraints if kind == "q_log_nparego" else acq.constraints
    assert len(constraints) == 1
    physical = torch.tensor([[[12., 999.], [14., -999.]]], dtype=torch.double)
    torch.testing.assert_close(constraints[0](physical), torch.tensor([[-1.5, .5]], dtype=torch.double))


def test_scalarization_fixed_formula_reference_and_seeded_slots(real_models):
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    acq, diagnostic, geometry = build(real_models, "q_log_nparego", "known", weights={"y0": .8, "y1": .2})
    physical = torch.tensor([[[13., 12.], [12., 14.], [30., 31.]]], dtype=torch.double, requires_grad=True)
    z = (physical - torch.tensor(geometry.location)) / torch.tensor(geometry.scale)
    shifted = z - torch.tensor(geometry.acquisition_ideal)
    weights = torch.tensor([.8, .2], dtype=torch.double)
    product = weights * shifted
    expected = product.amin(-1) + .05 * product.sum(-1)
    torch.testing.assert_close(acq.objective(physical), expected)
    reference = get_chebyshev_scalarization(weights, torch.empty(0, 2, dtype=torch.double))
    torch.testing.assert_close(acq.objective(physical), reference(shifted))
    other, _, other_geometry = build(real_models, "q_log_nparego", "known", weights={"y0": .2, "y1": .8})
    assert acq.objective(physical)[0, 0] > acq.objective(physical)[0, 1]
    assert other.objective(physical)[0, 0] < other.objective(physical)[0, 1]
    assert geometry.report_reference == other_geometry.report_reference
    assert diagnostic["scalarization_weights"] == [.8, .2]
    first, first_d, _ = build(real_models, "q_log_nparego", "known", slot=2)
    repeated, repeated_d, _ = build(real_models, "q_log_nparego", "known", slot=2)
    _, next_d, _ = build(real_models, "q_log_nparego", "known", slot=3)
    assert first_d["scalarization_weights"] == repeated_d["scalarization_weights"]
    assert first_d["scalarization_weights"] != next_d["scalarization_weights"]
    torch.testing.assert_close(first(torch.tensor([[[.45]]], dtype=torch.double)),
                               repeated(torch.tensor([[[.45]]], dtype=torch.double)))


@pytest.mark.parametrize("mode,variance", [("known", None), ("noiseless", .1), ("inferred", .1)])
def test_builder_validates_original_noise_before_model_access(mode, variance):
    space, batch = data(variance=variance)
    config = MultiObjectiveConfig(noise_mode=mode)
    runtime = compile_runtime(space, Knowledge(), batch)
    geometry = resolve_geometry(space, batch, runtime, config)
    with pytest.raises(EngineError) as caught:
        api().build_multiobjective_acquisition(space, None, batch, None, runtime, config, geometry, seed=0, slot=0)
    assert caught.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("kind,mode,module,symbol", [
    ("q_log_ehvi", "noiseless", "botorch.acquisition.multi_objective.logei", "qLogExpectedHypervolumeImprovement"),
    ("q_log_nehvi", "known", "botorch.acquisition.multi_objective.logei", "qLogNoisyExpectedHypervolumeImprovement"),
    ("q_log_nparego", "known", "botorch.acquisition.logei", "qLogNoisyExpectedImprovement"),
])
def test_missing_capability_is_lazy_and_typed(real_models, monkeypatch, kind, mode, module, symbol):
    target = importlib.import_module(module)
    monkeypatch.delattr(target, symbol)
    importlib.reload(api())
    with pytest.raises(EngineError) as caught:
        build(real_models, kind, mode)
    assert caught.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
    assert symbol in caught.value.message


@pytest.mark.parametrize("kind,mode", [
    ("q_log_ehvi", "noiseless"), ("q_log_nehvi", "known"), ("q_log_nparego", "known"),
])
def test_high_utility_unsafe_points_are_suppressed(real_models, kind, mode):
    space, batch, fitted = real_models[mode]
    constrained, _, geometry = build(real_models, kind, mode)
    free_space = Space(space.params, objectives=space.objective_specs)
    runtime = compile_runtime(free_space, Knowledge(), batch)
    # The counterfactual has the same physical posterior, but only measured
    # safe incumbents, so an unsafe incumbent cannot hide the candidate's gain.
    safe = batch.Y["y0"] <= 13.5
    safe_batch = ObservationBatch(batch.X.loc[safe], batch.Y.loc[safe],
                                 Yvar=None if batch.Yvar is None else batch.Yvar.loc[safe])
    free, _ = api().build_multiobjective_acquisition(free_space, fitted, safe_batch, None, runtime,
        MultiObjectiveConfig(acquisition=kind, noise_mode=mode), geometry, seed=23, slot=0)
    unsafe = pd.DataFrame({"x": [2.999, 3.001]})
    assert (fitted.predict(unsafe).mean > fitted.predict(pd.DataFrame({"x": [1.]})).mean).all()
    assert (api().joint_feasibility(space, fitted, unsafe) < .05).all()
    X = space.physical_to_model(unsafe).unsqueeze(-2)
    assert (constrained(X) < free(X) - 1.).all()


def test_preference_changes_actual_acquisition_ranking_on_asymmetric_tradeoff():
    space, batch = data(variance=.01)
    batch = ObservationBatch(batch.X, pd.DataFrame({"y0": [10., 12., 14., 16., 18.],
                                                   "y1": [105., 100., 95., 90., 85.]}),
                             Yvar=pd.DataFrame({"y0": [.01] * 5, "y1": [.01] * 5}))
    fitted = fit_surrogate(space, batch, seed=11)
    runtime = compile_runtime(space, Knowledge(), batch)
    rankings, references = [], []
    for weights in ({"y0": .8, "y1": .2}, {"y0": .2, "y1": .8}):
        config = MultiObjectiveConfig(preference_weights=weights)
        geometry = resolve_geometry(space, batch, runtime, config)
        acq, diagnostic = api().build_multiobjective_acquisition(
            space, fitted, batch, None, runtime, config, geometry, seed=23, slot=0)
        assert diagnostic["acquisition_reason"] == "preference_override"
        rankings.append(acq(torch.tensor([[[.2]], [[.8]]], dtype=torch.double)).argmax().item())
        references.append(geometry.report_reference)
    assert rankings == [1, 0]
    assert references[0] == references[1]


def test_later_extreme_data_do_not_refit_scalarization_geometry(real_models):
    space, batch, _ = real_models["known"]
    config = MultiObjectiveConfig(acquisition="q_log_nparego", preference_weights={"y0": .8, "y1": .2})
    runtime = compile_runtime(space, Knowledge(), batch)
    geometry = resolve_geometry(space, batch, runtime, config)
    later = ObservationBatch(pd.concat([batch.X, pd.DataFrame({"x": [.5]})]),
        pd.concat([batch.Y, pd.DataFrame({"y0": [100.], "y1": [1000.]})]),
        Yvar=pd.concat([batch.Yvar, pd.DataFrame({"y0": [.04], "y1": [.04]})]))
    fitted = fit_surrogate(space, later, seed=11)
    snapshot = geometry.to_dict()
    acq, diagnostic = api().build_multiobjective_acquisition(
        space, fitted, later, None, runtime, config, geometry, seed=23, slot=0)
    samples = torch.tensor([[[13., 12.], [100., 1000.]]], dtype=torch.double)
    z = (samples - samples.new_tensor([10., 11.])) / 4.
    product = (z - 1.) * samples.new_tensor([.8, .2])
    torch.testing.assert_close(acq.objective(samples), product.amin(-1) + .05 * product.sum(-1))
    assert geometry.to_dict() == diagnostic["geometry"] == snapshot


def test_builder_merges_constraints_and_keeps_physical_output_indices(real_models):
    space, batch, fitted = real_models["known"]
    space = Space(space.params, objectives=space.objective_specs, outcome_constraints=[
        OutcomeConstraint("lower", "y0", ">=", 11.), OutcomeConstraint("upper", "y0", "<=", 13.5),
        OutcomeConstraint("looser", "y0", "<=", 15.), OutcomeConstraint("other", "y1", "<=", 15.)])
    config = MultiObjectiveConfig()
    runtime = compile_runtime(space, Knowledge(), batch)
    geometry = resolve_geometry(space, batch, runtime, config)
    acq, _ = api().build_multiobjective_acquisition(
        space, fitted, batch, None, runtime, config, geometry, seed=0, slot=0)
    samples = torch.tensor([[[10., 14.], [12., 16.], [14., 0.]]], dtype=torch.double)
    assert len(acq.constraints) == 2
    torch.testing.assert_close(acq.constraints[0](samples), samples.new_tensor([[1., -1., .5]]))
    torch.testing.assert_close(acq.constraints[1](samples), samples.new_tensor([[-1., 1., -15.]]))


def test_many_objective_nparego_forward_backward():
    space, batch = data(m=5, variance=.04)
    fitted = fit_surrogate(space, batch, seed=11)
    config = MultiObjectiveConfig()
    runtime = compile_runtime(space, Knowledge(), batch)
    geometry = resolve_geometry(space, batch, runtime, config)
    acq, diagnostic = api().build_multiobjective_acquisition(
        space, fitted, batch, None, runtime, config, geometry, seed=11, slot=1)
    assert diagnostic["acquisition_kind"] == "q_log_nparego"
    weights = diagnostic["scalarization_weights"]
    assert len(weights) == 5 and min(weights) > 0 and sum(weights) == pytest.approx(1.)
    X = torch.tensor([[[.3], [.6]]], dtype=torch.double, requires_grad=True)
    acq(X).sum().backward()
    assert torch.isfinite(X.grad).all() and X.grad.abs().max() > 0


def test_old_constructor_missing_constraint_capability_is_typed(real_models, monkeypatch):
    import botorch.acquisition.multi_objective.logei as logei

    class OldEHVI:
        def __init__(self, model, ref_point, partitioning):
            pytest.fail("Unsupported constructor must not be entered")

    monkeypatch.setattr(logei, "qLogExpectedHypervolumeImprovement", OldEHVI)
    with pytest.raises(EngineError) as caught:
        build(real_models, "q_log_ehvi", "noiseless")
    assert caught.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
    assert "constraints" in caught.value.message and "sampler" in caught.value.message


@pytest.mark.parametrize("m", [0, 1, True, 2.5])
def test_invalid_objective_count(m):
    with pytest.raises(EngineError) as caught:
        api().choose_multiobjective_acquisition(m, MultiObjectiveConfig(), has_pending=False, has_preferences=False)
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_builder_rejects_preferences_inconsistent_with_frozen_geometry(real_models):
    space, batch, fitted = real_models["known"]
    runtime = compile_runtime(space, Knowledge(), batch)
    config = MultiObjectiveConfig()
    geometry = resolve_geometry(space, batch, runtime, config)
    with pytest.raises(EngineError) as caught:
        api().build_multiobjective_acquisition(space, fitted, batch, None, runtime,
            replace(config, preference_weights={"y0": .8, "y1": .2}), geometry, seed=0, slot=0)
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert "geometry" in caught.value.message.lower()


@pytest.mark.parametrize("mode,pending,expected", [
    ("inferred", False, "q_log_nehvi"), ("known", False, "q_log_nehvi"),
    ("noiseless", False, "q_log_ehvi"), ("noiseless", True, "q_log_nehvi"),
])
def test_builder_auto_uses_original_noise_and_external_pending(real_models, mode, pending, expected):
    external = PendingBatch(("external",), pd.DataFrame({"x": [.7]})) if pending else None
    acq, diagnostic, _ = build(real_models, "auto", mode, pending=external)
    assert diagnostic["acquisition_kind"] == expected
    assert torch.isfinite(acq(torch.tensor([[[.3]]], dtype=torch.double))).all()
    if mode == "noiseless":
        assert real_models[mode][1].Yvar is None


def test_builder_target_utilities_do_not_transform_physical_safety(real_models):
    original_space, batch, fitted = real_models["known"]
    space = Space(original_space.params,
        objectives=[Objective("y0", "target", target=12.), Objective("y1", "minimize")],
        outcome_constraints=[OutcomeConstraint("lower", "y0", ">=", 11.),
                             OutcomeConstraint("upper", "y1", "<=", 13.5)])
    config = MultiObjectiveConfig()
    runtime = compile_runtime(space, Knowledge(), batch)
    geometry = resolve_geometry(space, batch, runtime, config)
    acq, _ = api().build_multiobjective_acquisition(
        space, fitted, batch, None, runtime, config, geometry, seed=0, slot=0)
    samples = torch.tensor([[[14., 10.], [10., 14.]]], dtype=torch.double)
    torch.testing.assert_close(acq.objective(samples), samples.new_tensor([[[0., 1.25], [0., .25]]]))
    torch.testing.assert_close(acq.constraints[0](samples), samples.new_tensor([[-3., 1.]]))
    torch.testing.assert_close(acq.constraints[1](samples), samples.new_tensor([[-3.5, .5]]))


@pytest.mark.parametrize("kind,mode,module,symbol", [
    ("q_log_ehvi", "noiseless", "botorch.acquisition.multi_objective.objective", "GenericMCMultiOutputObjective"),
    ("q_log_ehvi", "noiseless", "botorch.utils.multi_objective.box_decompositions.non_dominated", "NondominatedPartitioning"),
    ("q_log_nparego", "known", "botorch.acquisition.objective", "GenericMCObjective"),
    ("q_log_nparego", "known", "botorch.utils.multi_objective.scalarization", "get_chebyshev_scalarization"),
])
def test_missing_objective_helpers_are_typed(real_models, monkeypatch, kind, mode, module, symbol):
    monkeypatch.delattr(importlib.import_module(module), symbol)
    with pytest.raises(EngineError) as caught:
        build(real_models, kind, mode)
    assert caught.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
    assert symbol in caught.value.message


@pytest.mark.parametrize("source", ["config", "knowledge"])
@pytest.mark.parametrize("kind,mode", [("q_log_ehvi", "noiseless"), ("q_log_nehvi", "known")])
def test_equal_preferences_allow_explicit_hv_but_auto_still_uses_nparego(real_models, source, kind, mode):
    space, batch, fitted = real_models[mode]
    config = MultiObjectiveConfig(acquisition=kind, noise_mode=mode,
        preference_weights={"y0": 1., "y1": 1.} if source == "config" else None)
    snapshot = config.to_dict()
    knowledge = (Knowledge().with_tradeoff("y0", "y1", first_weight=1., second_weight=1.)
                 if source == "knowledge" else Knowledge())
    runtime = compile_runtime(space, knowledge, batch)
    geometry = resolve_geometry(space, batch, runtime, config)
    acq, diagnostic = api().build_multiobjective_acquisition(
        space, fitted, batch, None, runtime, config, geometry, seed=0, slot=0)
    assert diagnostic["acquisition_kind"] == kind
    assert torch.isfinite(acq(torch.tensor([[[.3]]], dtype=torch.double))).all()
    _, auto = api().build_multiobjective_acquisition(
        space, fitted, batch, None, runtime, replace(config, acquisition="auto"), geometry, seed=0, slot=0)
    assert auto["acquisition_kind"] == "q_log_nparego"
    assert auto["acquisition_reason"] == "preference_override"
    assert config.to_dict() == snapshot


def test_ehvi_accepts_object_dtype_successes_with_failed_missing_outcomes():
    space, _ = data()
    batch = ObservationBatch(
        pd.DataFrame({"x": [0., 1., 2., 3.]}),
        pd.DataFrame({"y1": [2., 3., 4., None], "y0": [1., 2., 3., None]}, dtype=object),
        status=pd.Series(["success", "success", "success", "failed"]),
        ids=("a", "b", "c", "failed"))
    original_X, original_Y, original_status = batch.X, batch.Y, batch.status
    assert all(dtype == object for dtype in batch.successful().Y.dtypes)
    config = MultiObjectiveConfig(noise_mode="noiseless")
    runtime = compile_runtime(space, Knowledge(), batch)
    geometry = resolve_geometry(space, batch, runtime, config)
    geometry_snapshot = geometry.to_dict()
    model_batch = ObservationBatch(batch.X, batch.Y,
        Yvar=pd.DataFrame(0., index=batch.Y.index, columns=batch.Y.columns),
        status=batch.status, ids=batch.ids)
    fitted = fit_surrogate(space, model_batch, seed=11)

    acq, diagnostic = api().build_multiobjective_acquisition(
        space, fitted, batch, None, runtime, config, geometry, seed=23, slot=0)
    assert diagnostic["acquisition_kind"] == "q_log_ehvi"
    X = torch.tensor([[[.3]]], dtype=torch.double, requires_grad=True)
    value = acq(X)
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    pd.testing.assert_frame_equal(batch.X, original_X)
    pd.testing.assert_frame_equal(batch.Y, original_Y)
    pd.testing.assert_series_equal(batch.status, original_status)
    assert batch.ids == ("a", "b", "c", "failed") and batch.Yvar is None
    assert geometry.to_dict() == geometry_snapshot
    assert resolve_geometry(space, batch, runtime, config).to_dict() == geometry_snapshot
