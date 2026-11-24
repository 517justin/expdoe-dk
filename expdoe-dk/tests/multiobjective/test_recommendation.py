"""Real multioutput acquisition through the shared physical candidate search."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from expdoe_dk import (
    EngineError, ErrorCode, Knowledge, MultiObjectiveConfig, Objective,
    ObservationBatch, OutcomeConstraint, Parameter, PendingBatch, Space,
    candidate_keys, recommend,
)
from expdoe_dk.domain import LinearConstraint
from expdoe_dk.bo.knowledge_runtime import compile_runtime
from expdoe_dk.bo.objective_geometry import resolve_geometry


def _line(m=2, *, continuous=False, variance=None, constraints=()):
    objectives = [Objective("gain", "maximize"), Objective("cost", "minimize")]
    objectives += [Objective("target", "target", target=5.),
                   Objective("range", "target", target=(3., 5.)),
                   Objective("extra", "maximize")][:m - 2]
    space = Space([Parameter("x", bounds=(0, 10), kind="continuous" if continuous else "integer")],
                  objectives=objectives, outcome_constraints=constraints)
    values = {"gain": [2., 4., 6., 8.], "cost": [3., 5., 7., 9.],
              "target": [2., 4., 6., 8.], "range": [2., 4., 6., 8.],
              "extra": [8., 6., 4., 2.]}
    Y = pd.DataFrame({name: values[name] for name in reversed(space.objectives)})
    batch = ObservationBatch(pd.DataFrame({"x": [2, 4, 6, 8]}), Y,
        Yvar=None if variance is None else pd.DataFrame(variance, index=range(4), columns=Y.columns),
        ids=("t2", "t4", "t6", "t8"))
    return space, batch


def _assert_batch(space, batch, result, q, pending=None):
    keys = candidate_keys(result.frame, space)
    assert len(result.frame) == len(keys) == q
    assert not keys & candidate_keys(batch.X, space)
    if pending is not None:
        assert not keys & candidate_keys(pending.X, space)
    assert space.feasibility_mask(result.frame).all()
    assert result.predictions.mean.shape == (q, space.n_objectives)
    assert np.isfinite(result.predictions.mean).all()
    assert np.isfinite(result.predictions.variance).all()
    assert result.diagnostics["objective_names"] == space.objectives
    assert len(result.diagnostics["conditional_acquisition_values"]) == q
    assert np.isfinite(result.diagnostics["conditional_acquisition_values"]).all()
    assert result.diagnostics["acquisition_value_kind"] == "conditional_per_slot"
    assert "joint_acquisition_value" not in result.diagnostics
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("m,route", [(2, "q_log_nehvi"), (5, "q_log_nparego")])
def test_real_multiobjective_q2_is_seeded_and_preserves_global_rng(m, route):
    space, batch = _line(m)
    torch.manual_seed(777)
    state = torch.random.get_rng_state().clone()
    first = recommend(space, batch, q=2, seed=17)
    assert torch.equal(torch.random.get_rng_state(), state)
    second = recommend(space, batch, q=2, seed=17)
    _assert_batch(space, batch, first, 2)
    assert first.to_dict() == second.to_dict()
    assert first.diagnostics["acquisition_kind"] == route
    assert first.diagnostics["noise_mode"] == "inferred"
    weights = first.diagnostics["weights_by_slot"]
    if m == 5:
        assert weights[0] != weights[1]
        np.testing.assert_allclose(np.sum(weights, axis=1), [1., 1.])
    else:
        assert weights == [None, None]


def _mixed_catalog():
    space = Space([
        Parameter("a", bounds=(0., 1.)), Parameter("b", bounds=(0., 1.)),
        Parameter("n", kind="integer", bounds=(1, 6)),
        Parameter("dose", kind="discrete", values=(.1, .3, .9)),
        Parameter("material", kind="categorical", values=("A", "B")),
        Parameter("grade", kind="ordinal", values=("low", "high")),
    ], objectives=[Objective("gain", "maximize"), Objective("cost", "minimize")],
        constraints=[LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 1.)])
    catalog = pd.DataFrame({"a": [0., .25, 1., .5, .123456789012345, .7, .8],
        "b": [1., .75, 0., .5, .876543210987655, .3, .4],
        "n": [1, 2, 3, 4, 5, 6, 5], "dose": [.1, .3, .9, .1, .3, .9, .3],
        "material": ["A", "B", "A", "B", "A", "B", "A"],
        "grade": ["low", "high", "low", "high", "high", "low", "high"]})
    batch = ObservationBatch(catalog.iloc[:3], pd.DataFrame({"gain": [1., 2., 3.], "cost": [3., 2., 1.]}),
                             ids=("t0", "t1", "t2"))
    pending = PendingBatch(("external",), catalog.iloc[[3]])
    return space, batch, pending, catalog, ("seen0", "seen1", "seen2", "pending", "a", "b", "bad")


def test_multiobjective_catalog_retains_ids_and_physical_values():
    space, batch, pending, catalog, ids = _mixed_catalog()
    original = catalog.copy(deep=True)
    result = recommend(space, batch, pending=pending, candidates=catalog, candidate_ids=ids, q=2, seed=17)
    _assert_batch(space, batch, result, 2, pending)
    assert set(result.candidate_ids) == {"a", "b"}
    expected = catalog.iloc[[ids.index(i) for i in result.candidate_ids]].reset_index(drop=True)
    pd.testing.assert_frame_equal(result.frame, expected, check_exact=True)
    pd.testing.assert_frame_equal(catalog, original, check_exact=True)
    assert result.diagnostics["observation_ids"] == ["t0", "t1", "t2"]
    assert result.diagnostics["pending_ids"] == ["external"]
    assert result.diagnostics["pool_size"] == 7
    assert result.diagnostics["continuous_subspaces_optimized"] == 0
    reverse = recommend(space, batch, pending=pending, candidates=catalog.iloc[::-1],
                        candidate_ids=ids[::-1], q=2, seed=17)
    assert reverse.to_dict() == result.to_dict()


@pytest.mark.parametrize("column", ["gain", "cost"])
def test_catalog_rejects_response_leakage(column):
    space, batch, pending, catalog, ids = _mixed_catalog()
    catalog[column] = 999.
    with pytest.raises(EngineError) as error:
        recommend(space, batch, candidates=catalog, candidate_ids=ids, q=2)
    assert error.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("m,mode,external,route", [
    (2, "noiseless", False, "q_log_ehvi"),
    (2, "noiseless", True, "q_log_nehvi"),
    (5, "inferred", True, "q_log_nparego"),
])
def test_every_slot_conditions_external_and_internal_points(monkeypatch, m, mode, external, route):
    import expdoe_dk.bo.acquisition as builders
    module = importlib.import_module("expdoe_dk.bo.recommendation")
    space, batch = _line(m)
    pending = PendingBatch(("p",), pd.DataFrame({"x": [10]})) if external else None
    acquisitions, scores = [], []
    build, score = builders.build_multiobjective_acquisition, module._score

    def track_build(*args, **kwargs):
        assert args[3] is pending  # Internal selections must never alter route validation.
        result = build(*args, **kwargs)
        acquisitions.append(result[0])
        return result

    def track_score(acquisition, coordinates):
        result = score(acquisition, coordinates)
        effective = acquisition.X_pending
        if effective is None and hasattr(acquisition, "X_baseline"):
            effective = acquisition.X_baseline
        scores.append((None if effective is None else effective.detach().clone(),
                       float(result.max())))
        return result

    monkeypatch.setattr(builders, "build_multiobjective_acquisition", track_build)
    monkeypatch.setattr(module, "_score", track_score)
    result = recommend(space, batch, pending=pending, q=2, seed=17,
                       multiobjective=MultiObjectiveConfig(noise_mode=mode))
    _assert_batch(space, batch, result, 2, pending)
    assert result.diagnostics["acquisition_kind"] == route
    assert len(acquisitions) == 2
    assert acquisitions[0] is not acquisitions[1]
    required = result.frame.iloc[[0]]
    if pending is not None:
        required = pd.concat([pending.X, required], ignore_index=True)
        initial = space.physical_to_model(pending.X)[0]
        assert any(torch.equal(initial, row) for row in scores[0][0])
    for coordinate in space.physical_to_model(required):
        assert any(torch.equal(coordinate, row) for row in scores[1][0])
    np.testing.assert_array_equal(result.diagnostics["conditional_acquisition_values"], [s[1] for s in scores])


@pytest.mark.parametrize("mode,variance,known", [("noiseless", None, True), ("known", .01, True),
                                                ("inferred", None, False)])
def test_noise_fit_uses_model_only_copy_and_original_geometry(monkeypatch, mode, variance, known):
    module = importlib.import_module("expdoe_dk.bo.recommendation")
    space, batch = _line(variance=variance)
    config = MultiObjectiveConfig(noise_mode=mode)
    expected = resolve_geometry(space, batch, compile_runtime(space, Knowledge(), batch), config)
    original_X, original_Y, original_var = batch.X, batch.Y, batch.Yvar
    fit = module.fit_surrogate
    fitted_batches = []

    def track_fit(space, observations, **kwargs):
        fitted_batches.append(observations)
        return fit(space, observations, **kwargs)

    monkeypatch.setattr(module, "fit_surrogate", track_fit)
    result = recommend(space, batch, q=2, seed=17, multiobjective=config)
    assert result.diagnostics["geometry"] == expected.to_dict()
    assert result.diagnostics["noise_mode"] == mode
    for transform in result.diagnostics["transforms"]["objectives"].values():
        assert transform["known_variance"] is known
    fitted_batch = fitted_batches[0]
    if mode == "noiseless":
        assert fitted_batch is not batch
        assert (fitted_batch.Yvar.to_numpy() == 0.).all()
    else:
        assert fitted_batch is batch
    assert fitted_batch.ids == batch.ids
    pd.testing.assert_frame_equal(batch.X, original_X)
    pd.testing.assert_frame_equal(batch.Y, original_Y)
    if original_var is None:
        assert batch.Yvar is None
    else:
        pd.testing.assert_frame_equal(batch.Yvar, original_var)


@pytest.mark.parametrize("mode,variance", [("known", None), ("inferred", 0.), ("noiseless", .01)])
def test_invalid_noise_is_rejected_before_fitting(monkeypatch, mode, variance):
    def no_fit(*args, **kwargs):
        pytest.fail("Invalid noise must fail before fitting")
    monkeypatch.setattr("expdoe_dk.bo.recommendation.fit_surrogate", no_fit)
    with pytest.raises(EngineError) as error:
        recommend(*_line(variance=variance), multiobjective=MultiObjectiveConfig(noise_mode=mode))
    assert error.value.code == ErrorCode.CONFIG_INVALID


def test_explicit_preference_is_fixed_per_slot_and_changes_real_selection():
    space, batch = _line(variance=.001)
    catalog = pd.DataFrame({"x": [0, 1, 9, 10]})
    outputs = []
    for weights in ({"gain": .99, "cost": .01}, {"gain": .01, "cost": .99}):
        result = recommend(space, batch, candidates=catalog, candidate_ids=("lo", "lo1", "hi1", "hi"),
            q=2, seed=17, multiobjective=MultiObjectiveConfig(preference_weights=weights))
        assert result.diagnostics["acquisition_reason"] == "preference_override"
        assert result.diagnostics["weights_by_slot"] == [list(weights.values())] * 2
        outputs.append(result)
    assert outputs[0].frame.x.min() > outputs[1].frame.x.max()
    assert outputs[0].diagnostics["geometry"]["report_reference"] == outputs[1].diagnostics["geometry"]["report_reference"]


@pytest.mark.parametrize("boundary", [False, True])
def test_continuous_linear_search_preserves_equalities_and_boundaries(boundary):
    space = Space([Parameter("a", bounds=(0., 1.)), Parameter("b", bounds=(0., 1.)),
                   Parameter("free", bounds=(0., 1.))], objectives=[Objective("gain", "maximize"),
                   Objective("cost", "target", target=.5)],
                  constraints=[LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 0. if boundary else 1.)])
    X = pd.DataFrame({"a": [0.] * 3 if boundary else [0., .25, 1.],
                      "b": [0.] * 3 if boundary else [1., .75, 0.], "free": [0., .5, 1.]})
    batch = ObservationBatch(X, pd.DataFrame({"gain": [0., 1., .5], "cost": [.2, .8, .4]}))
    result = recommend(space, batch, q=2, seed=17)
    _assert_batch(space, batch, result, 2)
    np.testing.assert_allclose(result.frame.a + result.frame.b, 0. if boundary else 1., atol=1e-9)
    assert result.diagnostics["search_strategy"] == "enumerated_mixed_linear_polytope"
    assert result.diagnostics["continuous_subspaces_optimized"] > 0


def test_discrete_and_categorical_search_honors_forbidden_knowledge():
    space = Space([Parameter("dose", kind="discrete", values=(.1, .3, .9)),
        Parameter("material", kind="categorical", values=("A", "B")),
        Parameter("grade", kind="ordinal", values=("low", "high"))],
        objectives=[Objective("gain", "maximize"), Objective("cost", "minimize")])
    batch = ObservationBatch(pd.DataFrame({"dose": [.1, .3, .9], "material": ["A"] * 3,
                                          "grade": ["low"] * 3}),
                             pd.DataFrame({"gain": [1., 2., 3.], "cost": [3., 2., 1.]}))
    knowledge = Knowledge().with_safe_region(expression="material == 'A' or dose <= 0.3",
                                              factors=("material", "dose"))
    result = recommend(space, batch, knowledge=knowledge, q=2, seed=17)
    _assert_batch(space, batch, result, 2)
    assert not ((result.frame.material == "B") & (result.frame.dose > .3)).any()
    assert result.diagnostics["parameter_constraint_sources"]


def test_joint_physical_safety_threshold_and_exhaustion():
    constraints = [OutcomeConstraint("gain_min", "gain", ">=", 3.),
                   OutcomeConstraint("cost_max", "cost", "<=", 6.),
                   OutcomeConstraint("cost_min", "cost", ">=", 2.)]
    space, batch = _line(variance=.001, constraints=constraints)
    catalog = pd.DataFrame({"x": [0, 3, 5, 9, 10]})
    result = recommend(space, batch, candidates=catalog, candidate_ids=("low", "edge", "safe", "high", "top"),
        multiobjective=MultiObjectiveConfig(min_joint_feasibility=.1), seed=17)
    assert result.frame.x.item() in {3, 5}
    assert result.diagnostics["outcome_safety"]["minimum_probability"] == .1
    assert result.diagnostics["outcome_safety"]["probabilities"][0] >= .1
    assert result.diagnostics["outcome_safety"]["guaranteed"] is False
    with pytest.raises(EngineError) as error:
        recommend(space, batch, candidates=catalog, candidate_ids=("low", "edge", "safe", "high", "top"),
                  multiobjective=MultiObjectiveConfig(min_joint_feasibility=1.), seed=17)
    assert error.value.code == ErrorCode.NO_FEASIBLE_CANDIDATE


@pytest.mark.parametrize("catalog", [False, True])
def test_q_exhaustion_never_returns_partial_batch(catalog):
    space, batch = _line()
    kwargs = dict(candidates=pd.DataFrame({"x": [1]}), candidate_ids=("one",)) if catalog else {}
    with pytest.raises(EngineError) as error:
        recommend(space, batch, q=8, seed=17, **kwargs)
    assert error.value.code == ErrorCode.NO_FEASIBLE_CANDIDATE


def test_precision_collision_stays_config_invalid():
    original, batch = _line()
    space = Space([Parameter("x", bounds=(-1., 1.))], objectives=original.objective_specs)
    batch = ObservationBatch(pd.DataFrame({"x": [-1., -.5, 0., 1.]}), batch.Y)
    with pytest.raises(EngineError) as error:
        recommend(space, batch, candidates=pd.DataFrame({"x": [0., 1e-17]}),
                  candidate_ids=("exact", "near"))
    assert error.value.code == ErrorCode.CONFIG_INVALID
    assert "model-coordinate collision" in str(error.value).lower()


@pytest.mark.parametrize("failure", ["fit", "optimizer", "score"])
def test_numerical_failure_is_typed_not_infeasibility_or_partial_batch(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise RuntimeError("injected numerical failure")
    name = {"fit": "fit_surrogate", "optimizer": "optimize_acqf", "score": "_score"}[failure]
    monkeypatch.setattr("expdoe_dk.bo.recommendation." + name, fail)
    with pytest.raises(EngineError) as error:
        recommend(*_line(continuous=failure == "optimizer"), q=2, seed=17)
    assert error.value.code == (ErrorCode.MODEL_FIT_FAILED if failure == "fit" else ErrorCode.ACQUISITION_INCOMPATIBLE)
    assert "injected numerical failure" in str(error.value)


def test_resolved_entry_reuses_frozen_initial_geometry():
    from expdoe_dk.bo.recommendation import _recommend_resolved
    space, batch = _line()
    config = MultiObjectiveConfig()
    geometry = resolve_geometry(space, batch, compile_runtime(space, Knowledge(), batch), config)
    later = ObservationBatch(pd.concat([batch.X, pd.DataFrame({"x": [9]})], ignore_index=True),
        pd.concat([batch.Y, pd.DataFrame({"cost": [100.], "gain": [100.]})], ignore_index=True))
    result = _recommend_resolved(space, later, multiobjective=config, geometry=geometry, q=2, seed=17)
    assert result.diagnostics["geometry"] == geometry.to_dict()
    assert result.diagnostics["geometry"]["source_row_count"] == 4


def test_single_objective_rejects_multiobjective_config():
    space = Space([Parameter("x", kind="integer", bounds=(0, 10))], objectives="y")
    batch = ObservationBatch(pd.DataFrame({"x": [2, 4, 6]}), pd.DataFrame({"y": [1., 2., 3.]}))
    with pytest.raises(EngineError) as error:
        recommend(space, batch, multiobjective=MultiObjectiveConfig())
    assert error.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("external", [False, True])
def test_incompatible_route_is_not_disguised_as_catalog_exhaustion(external):
    space, batch = _line()
    pending = PendingBatch(("p",), pd.DataFrame({"x": [10]})) if external else None
    config = MultiObjectiveConfig(acquisition="q_log_ehvi", noise_mode="noiseless" if external else "inferred")
    with pytest.raises(EngineError) as error:
        recommend(space, batch, pending=pending, multiobjective=config,
                  candidates=pd.DataFrame({"x": [2]}), candidate_ids=("seen",))
    assert error.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
    assert "qLogEHVI" in str(error.value)


@pytest.mark.parametrize("direction,target,want", [
    ("maximize", None, {9, 10}), ("minimize", None, {0, 1}),
    ("target", 5., {3, 5}), ("target", (4.5, 5.5), {3, 5}),
])
def test_real_min_max_and_target_samples_drive_selection(direction, target, want):
    original, batch = _line(variance=.001)
    space = Space(original.params, objectives=[Objective("gain", direction, target=target),
                                               Objective("cost", "maximize")])
    result = recommend(space, batch, candidates=pd.DataFrame({"x": [0, 1, 3, 5, 9, 10]}),
        candidate_ids=("zero", "one", "three", "five", "nine", "ten"), q=2, seed=17,
        multiobjective=MultiObjectiveConfig(preference_weights={"gain": .999, "cost": .001}))
    assert set(result.frame.x) == want


def test_noiseless_failed_object_rows_preserve_original_data_and_exclusions():
    space, original = _line()
    batch = ObservationBatch(pd.concat([original.X, pd.DataFrame({"x": [9, 10]})], ignore_index=True),
        pd.DataFrame({"cost": pd.Series([3., 5., 7., 9., None, None], dtype=object),
                      "gain": pd.Series([2., 4., 6., 8., None, None], dtype=object)}),
        status=pd.Series(["success"] * 4 + ["failed", "warning"]), ids=original.ids + ("f", "w"))
    config = MultiObjectiveConfig(noise_mode="noiseless")
    geometry = resolve_geometry(space, batch, compile_runtime(space, Knowledge(), batch), config)
    original_Y, original_status = batch.Y, batch.status
    result = recommend(space, batch, multiobjective=config, q=2, seed=17)
    _assert_batch(space, batch, result, 2)
    assert result.diagnostics["geometry"] == geometry.to_dict()
    assert result.diagnostics["observation_ids"] == ["t2", "t4", "t6", "t8", "f", "w"]
    assert {item["reason"] for item in result.diagnostics["exclusions"]} == {"failed", "warning"}
    assert batch.Yvar is None
    pd.testing.assert_frame_equal(batch.Y, original_Y)
    pd.testing.assert_series_equal(batch.status, original_status)


@pytest.mark.parametrize("kind,route", [("auto", "q_log_nparego"), ("q_log_nehvi", "q_log_nehvi")])
def test_equal_explicit_preferences_keep_controller_route_ruling(kind, route):
    result = recommend(*_line(), multiobjective=MultiObjectiveConfig(
        acquisition=kind, preference_weights={"gain": 1., "cost": 1.}), q=2, seed=17)
    assert result.diagnostics["acquisition_kind"] == route
    assert result.diagnostics["weights_by_slot"] == ([[.5, .5]] * 2 if kind == "auto" else [None, None])


def test_no_feasible_measured_baseline_remains_insufficient_data():
    space, batch = _line(constraints=[OutcomeConstraint("unsafe", "cost", "<=", 0.)])
    with pytest.raises(EngineError) as error:
        recommend(space, batch, q=2)
    assert error.value.code == ErrorCode.INSUFFICIENT_DATA
    assert error.value.details["reason"] == "no_feasible_baseline"


def test_resolved_entry_rejects_preferences_that_disagree_with_geometry():
    from expdoe_dk.bo.recommendation import _recommend_resolved
    space, batch = _line()
    geometry = resolve_geometry(space, batch, compile_runtime(space, Knowledge(), batch), MultiObjectiveConfig())
    with pytest.raises(EngineError) as error:
        _recommend_resolved(space, batch, geometry=geometry, multiobjective=MultiObjectiveConfig(
            preference_weights={"gain": .8, "cost": .2}), q=2)
    assert error.value.code == ErrorCode.CONFIG_INVALID
    assert "geometry" in str(error.value)


def test_missing_multiobjective_capability_is_typed_and_scalar_still_works(monkeypatch):
    import botorch.acquisition.multi_objective.logei as logei
    monkeypatch.delattr(logei, "qLogNoisyExpectedHypervolumeImprovement")
    with pytest.raises(EngineError) as error:
        recommend(*_line(), q=2, seed=17)
    assert error.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE
    assert "qLogNoisyExpectedHypervolumeImprovement" in str(error.value)
    test_scalar_fixed_seed_result_diagnostics_and_rng_are_unchanged()


def test_scalar_fixed_seed_result_diagnostics_and_rng_are_unchanged():
    space = Space([Parameter("x", kind="integer", bounds=(0, 10))], objectives="y")
    batch = ObservationBatch(pd.DataFrame({"x": [2, 4, 6, 8]}),
        pd.DataFrame({"y": [120., 140., 160., 180.]}), Yvar=pd.DataFrame({"y": [.001] * 4}))
    torch.manual_seed(777)
    result = recommend(space, batch, q=2, seed=17)
    assert result.frame.x.tolist() == [10, 9]
    assert set(result.to_dict()) == {"frame", "predictions", "diagnostics"}
    assert list(result.diagnostics) == ["search_strategy", "finite_combinations", "degradation_reason",
        "pool_size", "acquisition_name", "acquisition_value_kind", "joint_acquisition_value", "seed",
        "mc_samples", "pending_count", "candidate_keys", "continuous_subspaces_optimized",
        "parameter_constraint_sources", "transforms", "exclusions", "soft_outcome_preference",
        "strategy_explanation", "outcome_safety"]
    assert result.diagnostics["acquisition_value_kind"] == "joint_batch"
    assert result.diagnostics["joint_acquisition_value"] == pytest.approx(2.995382638702547, abs=1e-9)
    np.testing.assert_allclose(result.predictions.mean, [[199.99292819423715], [189.997344460162]], rtol=1e-10)
    np.testing.assert_allclose(result.predictions.variance, [[.008797736882115716], [.003061230017920025]], rtol=1e-7)
    assert torch.rand(3).tolist() == [0.08191531896591187, 0.49113744497299194, 0.40329331159591675]
