"""Campaign owns measured vectors and commits recommendation state on success."""
import importlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from expdoe_dk import (
    Campaign, EngineError, ErrorCode, Knowledge, KnowledgeScope,
    MultiObjectiveConfig, Objective, ObservationBatch, Parameter, PendingBatch,
    Space, make_pattern_spec,
)
from expdoe_dk.bo.objective_geometry import _source_digest
from expdoe_dk.domain import LinearConstraint


def space():
    return Space([Parameter("x", kind="integer", bounds=(0, 12))],
                 objectives=[Objective("gain", "maximize"), Objective("cost", "minimize")])


def oracle(X):
    return pd.DataFrame({"cost": 20. - X.x.to_numpy(), "gain": X.x.to_numpy() ** 2 + 1.})


def initial(campaign=None):
    campaign = Campaign(space(), seed=17, validate=False) if campaign is None else campaign
    X = pd.DataFrame({"x": [0, 2, 4]})
    campaign.tell(X, oracle(X))
    return campaign


def catalog():
    return pd.DataFrame({"x": list(range(13))}), tuple(f"c{i:02}" for i in range(13))


def test_run_never_flattens_multiobjective_response():
    campaign = Campaign(space(), validate=False)
    result = campaign.run(oracle, n_doe=3, n_iter=0, doe_method="auto")
    assert len(result.history_df) == 3
    assert result.n_doe == 3 and result.n_bo == 0
    pd.testing.assert_frame_equal(result.history_df[space().objectives],
                                  oracle(result.history_df)[space().objectives])
    assert result.best_y_physical is None and result.best_x_physical is None
    assert "y" not in result.history_df
    assert campaign._y_physical == campaign._y_internal == campaign._X_phys == []
    assert json.loads(json.dumps(result.to_dict()))["best_y_physical"] is None


def test_real_run_includes_bo_vectors_and_no_fabricated_metrics():
    campaign = Campaign(space(), seed=17, validate=False)
    frame, ids = catalog()
    result = campaign.run(oracle, n_doe=3, n_iter=2, q=2,
                          candidates=frame, candidate_ids=ids)
    assert len(result.history_df) == 7
    assert (result.n_doe, result.n_bo) == (3, 4)
    assert result.history_df.x.is_unique
    pd.testing.assert_frame_equal(result.history_df[space().objectives],
                                  oracle(result.history_df)[space().objectives])
    assert result.best_y_physical is None
    assert "hypervolume" not in result.to_dict()


def test_failed_ask_does_not_advance_iteration(monkeypatch):
    campaign = initial()
    module = importlib.import_module("expdoe_dk.bo.recommendation")
    original = module._recommend_resolved
    seeds = []

    def fail(*args, **kwargs):
        seeds.append(kwargs["seed"])
        original(*args, **kwargs)
        raise RuntimeError("after recommendation")

    frame, ids = catalog()
    monkeypatch.setattr(module, "_recommend_resolved", fail)
    with pytest.raises(RuntimeError, match="after recommendation"):
        campaign.ask(candidates=frame, candidate_ids=ids)
    assert campaign._ask_iteration == 0
    assert campaign._geometry is None
    assert campaign._catalog is None
    monkeypatch.setattr(module, "_recommend_resolved", original)
    result = campaign.ask_with_diagnostics(candidates=frame, candidate_ids=ids)
    assert result.diagnostics["seed"] == seeds[0] == 20
    before = campaign._geometry.to_dict()
    with pytest.raises(EngineError):
        campaign.ask(q=20)
    assert campaign._ask_iteration == 1 and campaign._geometry.to_dict() == before


def test_partial_trial_returns_preserve_other_pending():
    campaign = initial()
    campaign.set_pending(PendingBatch(("a", "repeat", "b"), pd.DataFrame({"x": [6, 6, 8]})))
    X = pd.DataFrame({"x": [6]})
    campaign.tell_batch(ObservationBatch(X, oracle(X), ids=("a",)))
    assert campaign._pending.ids == ("repeat", "b")
    campaign.tell(X, oracle(X))
    assert campaign._pending.ids == ("repeat", "b")
    campaign.tell_batch(ObservationBatch(X, oracle(X), ids=("repeat",)))
    assert campaign._pending.ids == ("b",)
    X = pd.DataFrame({"x": [8]})
    campaign.tell_batch(ObservationBatch(X, oracle(X), ids=("b",)))
    assert campaign._pending is None
    assert len(campaign.history_df()) == 7


def test_reordered_columns_variance_status_and_trial_audit():
    campaign = initial()
    X = pd.DataFrame({"x": [5, 6, 7]})
    Y = pd.DataFrame({"cost": [15., None, "unavailable"], "gain": [26., None, None]})
    batch = ObservationBatch(X, Y, status=pd.Series(["success", "failed", "warning"]), ids=("a", "b", "c"))
    campaign.tell_batch(batch)
    combined = campaign._combined_observations()
    assert combined.Y.columns.tolist() == ["gain", "cost"]
    assert combined.X.x.tolist() == [0, 2, 4, 5, 6, 7]
    assert combined.status.tolist()[-3:] == ["success", "failed", "warning"]
    assert combined.Y.cost.tolist()[-1] == "unavailable"
    history = campaign.history_df()
    assert len(history) == 6
    assert history["status"].tolist()[-3:] == ["success", "failed", "warning"]
    result = campaign.ask_with_diagnostics(q=2)
    assert result.diagnostics["seed"] == 21
    assert not set(result.frame.x) & {0, 2, 4, 5, 6, 7}
    assert campaign._geometry.source_row_count == 6


@pytest.mark.parametrize("bad", [pd.DataFrame({"gain": [1.]}),
    pd.DataFrame({"gain": [1.], "cost": [2.], "extra": [3.]}),
    pd.DataFrame([[1., 2.]], columns=["gain", "gain"]),
    pd.DataFrame({"gain": [1.], "cost": [np.nan]}), [1., 2.]])
def test_invalid_tell_is_atomic(bad):
    campaign = initial()
    campaign.set_pending(PendingBatch(("p",), pd.DataFrame({"x": [6]})))
    before = campaign.history_df()
    with pytest.raises((EngineError, ValueError)):
        campaign.tell(pd.DataFrame({"x": [6]}), bad)
    pd.testing.assert_frame_equal(campaign.history_df(), before)
    assert campaign._pending.ids == ("p",)


def test_duplicate_explicit_trial_id_is_rejected_but_repeated_condition_is_allowed():
    campaign = initial()
    X = pd.DataFrame({"x": [0]})
    campaign.tell_batch(ObservationBatch(X, oracle(X), ids=("repeat",)))
    with pytest.raises(EngineError, match="unique"):
        campaign.tell_batch(ObservationBatch(X, oracle(X), ids=("repeat",)))
    assert len(campaign.history_df()) == 4


def test_catalog_binding_reorder_detachment_and_geometry_freeze():
    campaign = initial()
    frame, ids = catalog()
    first = campaign.ask_with_diagnostics(q=2, candidates=frame, candidate_ids=ids)
    frozen = campaign._geometry.to_dict()
    assert len(first.candidate_ids) == 2
    expected = frame.iloc[[ids.index(i) for i in first.candidate_ids]].reset_index(drop=True)
    pd.testing.assert_frame_equal(first.frame, expected)
    X = first.frame
    campaign.tell(X, oracle(X) * 100.)
    second = campaign.ask_with_diagnostics(candidates=frame.iloc[::-1], candidate_ids=ids[::-1])
    assert second.diagnostics["geometry"] == frozen
    frame.loc[0, "x"] = 1
    assert campaign._catalog.x.iloc[0] == 0
    before = campaign._ask_iteration
    with pytest.raises(EngineError, match="catalog|Catalog"):
        campaign.ask(candidates=pd.DataFrame({"x": list(range(1, 13))}), candidate_ids=ids[1:])
    assert campaign._ask_iteration == before
    assert campaign.ask_with_diagnostics().candidate_ids


def test_generated_ids_never_rewrite_frozen_source_prefix():
    campaign = initial()
    campaign.ask()
    prefix = campaign._combined_observations()
    geometry = campaign._geometry
    X = pd.DataFrame({"x": [6]})
    campaign.tell_batch(ObservationBatch(X, oracle(X), ids=("unidentified:0:0:",)))
    combined = campaign._combined_observations()
    assert combined.ids[:3] == prefix.ids
    sliced = ObservationBatch(combined.X.iloc[:3], combined.Y.iloc[:3],
                              status=combined.status.iloc[:3], ids=combined.ids[:3])
    assert _source_digest(campaign.space, sliced) == geometry.source_digest
    with pytest.raises(EngineError, match="unique"):
        campaign.tell_batch(ObservationBatch(X, oracle(X), ids=(prefix.ids[0],)))


def test_known_variance_is_reordered_without_losing_measurement_metadata():
    campaign = Campaign(space(), validate=False)
    X = pd.DataFrame({"x": [0, 2, 4]})
    Y = oracle(X)
    variance = pd.DataFrame({"cost": [.2] * 3, "gain": [.1] * 3})
    campaign.tell_batch(ObservationBatch(X, Y, Yvar=variance))
    batch = campaign._combined_observations()
    pd.testing.assert_frame_equal(batch.Yvar, variance[["gain", "cost"]])
    assert campaign.ask_with_diagnostics().diagnostics["noise_mode"] == "known"


@pytest.mark.parametrize("at", [0, 1])
def test_oracle_callback_error_never_records_predictions(at):
    campaign = Campaign(space(), validate=False)
    calls = []

    def fail(X):
        calls.append(X.copy())
        if len(calls) == at + 1:
            raise RuntimeError("oracle unavailable")
        return oracle(X)

    frame, ids = catalog()
    with pytest.raises(RuntimeError, match="oracle unavailable"):
        campaign.run(fail, n_doe=3, n_iter=1, candidates=frame, candidate_ids=ids)
    assert len(campaign.history_df()) == (0 if at == 0 else 3)
    assert campaign._ask_iteration == at


def test_multiobjective_checkpoint_cannot_downgrade(tmp_path):
    campaign = initial()
    campaign.save_checkpoint(tmp_path / "multi.json")
    assert campaign.to_state()["schema_version"] == "2.2"
    restored = Campaign.load_checkpoint(tmp_path / "multi.json")
    pd.testing.assert_frame_equal(restored.history_df(), campaign.history_df())


def test_configuration_rejected_for_scalar_and_frozen_for_multiobjective():
    with pytest.raises(EngineError):
        Campaign(Space([Parameter("x", bounds=(0, 1))]), multiobjective=MultiObjectiveConfig())
    campaign = initial(Campaign(space(), multiobjective=MultiObjectiveConfig(), validate=False))
    campaign.ask()
    with pytest.raises(AttributeError):
        campaign.multiobjective = MultiObjectiveConfig(noise_mode="noiseless")
    campaign.knowledge.add(make_pattern_spec(pattern_id="new", pattern="gp_prior", version="1.0",
        scope=KnowledgeScope(objectives=("gain",)), parameters={"lengthscale": "weak"}, confidence=.8))
    with pytest.raises(EngineError, match="new [Cc]ampaign|frozen"):
        campaign.ask()


def test_scalar_tell_pending_rng_and_json_remain_unchanged():
    scalar = Space([Parameter("x", kind="integer", bounds=(0, 12))])
    campaign = Campaign(scalar, seed=17, validate=False)
    campaign.tell(pd.DataFrame({"x": [0, 2, 4]}), [1., 5., 17.])
    campaign.set_pending(PendingBatch(("p",), pd.DataFrame({"x": [6]})))
    campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [6]}), pd.DataFrame({"y": [37.]}), ids=("p",)))
    assert campaign._pending.ids == ("p",)
    torch.manual_seed(777)
    rng = torch.random.get_rng_state().clone()
    first = campaign.ask_with_diagnostics()
    assert first.diagnostics["seed"] == 21
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert campaign.ask_with_diagnostics().diagnostics["seed"] == 22
    assert campaign.ask_with_diagnostics(iteration=9).diagnostics["seed"] == 26
    assert campaign._ask_iteration == 10
    assert set(campaign.finalize().to_dict()) == {
        "best_x_physical", "best_y_physical", "n_doe", "n_bo", "objectives", "maximize",
        "knowledge_summary", "knowledge", "notes", "param_units", "param_kinds", "history_records"}


def test_empirical_validators_project_both_objectives_and_report_unsupported(monkeypatch):
    knowledge = Knowledge()
    for name in ("gain", "cost"):
        knowledge.add(make_pattern_spec(pattern_id=f"mono-{name}", pattern="monotone", version="1.0",
            scope=KnowledgeScope(factors=("x",), objectives=(name,)),
            parameters={"direction": "increasing", "epsilon": "auto", "delta_norm": .5, "n_pairs_per_dim": 3}, confidence=.8))
    knowledge.add(make_pattern_spec(pattern_id="power", pattern="power_law", version="1.0",
        scope=KnowledgeScope(factors=("x",), objectives=("gain",)), parameters={"exponent": 2., "scale": 1.}, confidence=.8))
    module = importlib.import_module("expdoe_dk.bo.loop")
    original = module.check_monotone_assumption
    seen = []

    def track(**kwargs):
        seen.append(kwargs["y_user"].copy())
        return original(**kwargs)

    monkeypatch.setattr(module, "check_monotone_assumption", track)
    positive = Space([Parameter("x", kind="integer", bounds=(1, 12))], objectives=space().objective_specs)
    campaign = Campaign(positive, knowledge, validation_min_obs=3, validation_interval=1)
    X = pd.DataFrame({"x": list(range(1, 11))})
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        campaign.tell(X, oracle(X))
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], oracle(X).gain)
    np.testing.assert_array_equal(seen[1], oracle(X).cost)
    diagnostics = campaign.validation_diagnostics
    assert any(d["pattern_id"] == "power" and d["status"] == "unsupported" for d in diagnostics)
    assert {d["objective_name"] for d in diagnostics if d["status"] == "checked"} == {"gain", "cost"}
    from expdoe_dk.knowledge.validators import MonotoneViolationWarning
    assert any(isinstance(w.message, MonotoneViolationWarning) and "cost:" in str(w.message) for w in notices)


def test_history_name_collisions_preserve_every_parameter_objective_and_metadata():
    names = ["x", "kind", "status", "trial_id", "parameter:x", "metadata:kind"]
    declaration = Space([Parameter("x", bounds=(0., 1.)), Parameter("status", bounds=(0., 1.))],
                        objectives=[Objective(name, "maximize") for name in names])
    campaign = Campaign(declaration, validate=False)
    X = pd.DataFrame({"x": [.1, .2, .3], "status": [.4, .5, .6]})
    Y = pd.DataFrame({name: [i + 10., i + 20., i + 30.] for i, name in enumerate(names)})
    campaign.tell(X, Y)
    result = campaign.finalize()
    columns = result.history_columns
    assert result.history_df.columns.is_unique
    for name in names:
        assert columns["objectives"][name] == name
        assert result.history_df[name].tolist() == Y[name].tolist()
    for name in X:
        assert result.history_df[columns["parameters"][name]].tolist() == X[name].tolist()
    assert result.history_df[columns["metadata"]["kind"]].tolist() == ["doe"] * 3
    assert result.history_df[columns["metadata"]["status"]].tolist() == ["success"] * 3
    assert len(set(result.history_df[columns["metadata"]["trial_id"]])) == 3
    wire = json.loads(json.dumps(result.to_dict()))
    assert wire["history_columns"] == columns
    assert len(wire["history_records"][0]) == len(X.columns) + len(names) + 3


def test_initial_catalog_respects_constraints_pending_and_no_space_doe(monkeypatch):
    declaration = Space(space().params, objectives=space().objective_specs,
        constraints=[LinearConstraint("limit", {"x": 1.}, "<=", 8.)])
    campaign = Campaign(declaration, seed=9, validate=False)
    campaign.set_pending(PendingBatch(("external",), pd.DataFrame({"x": [2]})))
    monkeypatch.setattr(campaign, "suggest_doe", lambda **kw: pytest.fail("catalog must not invoke space DoE"))
    frame = pd.DataFrame({"x": [0, 2, 4, 6, 8, 10, 12]})
    result = campaign.run(oracle, n_doe=3, n_iter=1, candidates=frame,
                          candidate_ids=tuple(f"row{i}" for i in range(len(frame))))
    assert len(result.history_df) == 4
    assert set(result.history_df.x) <= {0, 4, 6, 8}
    assert result.history_df.x.dtype == frame.x.dtype


def test_catalog_exhaustion_before_oracle_and_insufficient_success_before_fit():
    campaign = Campaign(space(), validate=False)
    frame = pd.DataFrame({"x": [0, 2]})
    with pytest.raises(EngineError) as caught:
        campaign.run(lambda X: pytest.fail("oracle must not run"), n_doe=3,
                      candidates=frame, candidate_ids=("a", "b"))
    assert caught.value.code == ErrorCode.NO_FEASIBLE_CANDIDATE
    campaign.tell(frame, oracle(frame))
    with pytest.raises(EngineError) as caught:
        campaign.ask()
    assert caught.value.code == ErrorCode.INSUFFICIENT_DATA
    assert campaign._ask_iteration == 0 and campaign._geometry is None


@pytest.mark.parametrize("response", [lambda X: oracle(X).to_numpy(),
    lambda X: oracle(X)[["gain"]], lambda X: oracle(X).assign(cost=np.nan)])
def test_run_rejects_unnamed_or_incomplete_oracle_values(response):
    campaign = Campaign(space(), validate=False)
    frame, ids = catalog()
    with pytest.raises((EngineError, ValueError)):
        campaign.run(response, n_doe=3, n_iter=0, candidates=frame, candidate_ids=ids)
    assert campaign.history_df().empty
    assert campaign._catalog is None and campaign._geometry is None


def test_empty_pending_and_unsuccessful_return_closure():
    campaign = initial()
    campaign.set_pending(PendingBatch((), pd.DataFrame({"x": pd.Series([], dtype=int)})))
    assert campaign.ask_with_diagnostics().diagnostics["pending_ids"] == []
    X = pd.DataFrame({"x": [6, 8]})
    campaign.set_pending(PendingBatch(("failed", "warning"), X))
    campaign.tell_batch(ObservationBatch(X, pd.DataFrame({"gain": [None, None], "cost": [None, None]}),
        status=pd.Series(["failed", "warning"]), ids=("failed", "warning")))
    assert campaign._pending is None
    assert campaign._successful_count() == 3
    assert len(campaign.finalize().history_df) == 5


def test_explicit_iteration_and_failed_explicit_iteration_are_atomic():
    campaign = initial()
    assert campaign.ask_with_diagnostics(iteration=7).diagnostics["seed"] == 24
    assert campaign._ask_iteration == 8
    geometry = campaign._geometry
    with pytest.raises(EngineError):
        campaign.ask(q=100, iteration=20)
    assert campaign._ask_iteration == 8 and campaign._geometry is geometry
    assert campaign.ask_with_diagnostics().diagnostics["seed"] == 28


def test_shape_validators_use_each_projected_objective(monkeypatch):
    knowledge = Knowledge()
    for name in ("gain", "cost"):
        knowledge.add(make_pattern_spec(pattern_id=f"shape-{name}", pattern="quadratic_peak", version="1.0",
            scope=KnowledgeScope(factors=("x",), objectives=(name,)),
            parameters={"center": 6., "direction": "peak", "frozen": True}, confidence=.8))
    module = importlib.import_module("expdoe_dk.bo.loop")
    original = module.check_shape_prior_fit
    seen = []

    def track(**kw):
        seen.append(kw["y_norm"].copy())
        return original(**kw)

    monkeypatch.setattr(module, "check_shape_prior_fit", track)
    campaign = Campaign(space(), knowledge, validation_min_obs=3, validation_interval=1)
    X = pd.DataFrame({"x": list(range(10))})
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        campaign.tell(X, oracle(X))
    assert len(seen) == 2
    for i, name in enumerate(campaign.space.objectives):
        values = -campaign.space.objective_specs[i].to_utility(oracle(X)[name].to_numpy())
        np.testing.assert_allclose(seen[i], (values - values.mean()) / values.std(ddof=1))
    assert all(d["status"] == "checked" for d in campaign.validation_diagnostics)


def test_multiobjective_epsilon_validation_does_not_cross_objective_scopes():
    knowledge = Knowledge().add(make_pattern_spec(pattern_id="prior", pattern="gp_prior", version="1.0",
        scope=KnowledgeScope(objectives=("cost",)), parameters={"lengthscale": "weak"}, confidence=.8))
    knowledge.add(make_pattern_spec(pattern_id="mono", pattern="monotone", version="1.0",
        scope=KnowledgeScope(factors=("x",), objectives=("gain",)),
        parameters={"direction": "increasing", "epsilon": .001, "delta_norm": .5, "n_pairs_per_dim": 3}, confidence=.8))
    campaign = Campaign(space(), knowledge, auto_rescue=False, validate=False)
    assert campaign.knowledge.specs[1].parameters["epsilon"] == .001


def test_empirical_callback_error_rolls_back_tell_and_pending(monkeypatch):
    campaign = initial(Campaign(space(), validation_min_obs=3, validation_interval=1))
    campaign.set_pending(PendingBatch(("p",), pd.DataFrame({"x": [6]})))
    before = campaign.history_df()
    last_n = campaign._last_validation_n

    def fail():
        raise RuntimeError("validator callback failed")

    monkeypatch.setattr(campaign, "_run_multiobjective_validators", fail)
    X = pd.DataFrame({"x": [6]})
    with pytest.raises(RuntimeError, match="validator callback failed"):
        campaign.tell_batch(ObservationBatch(X, oracle(X), ids=("p",)))
    pd.testing.assert_frame_equal(campaign.history_df(), before)
    assert campaign._pending.ids == ("p",)
    assert campaign._last_validation_n == last_n
    assert campaign._successful_count() == 3


def test_nonlegacy_knowledge_does_not_claim_plain_gp_in_result():
    knowledge = Knowledge().add(make_pattern_spec(pattern_id="shape", pattern="saturation", version="2.0",
        scope=KnowledgeScope(factors=("x",), objectives=("gain",)),
        parameters={"direction": "increasing", "half_response": 4., "origin": 0.}, confidence=.8))
    campaign = initial(Campaign(space(), knowledge, validate=False))
    assert not any("plain GP" in note for note in campaign.finalize().notes)


def test_shared_scope_epsilon_rescue_is_preserved_without_mutating_caller():
    knowledge = Knowledge().add(make_pattern_spec(pattern_id="prior", pattern="gp_prior", version="1.0",
        scope=KnowledgeScope(objectives=("cost",)), parameters={"lengthscale": "weak"}, confidence=.8))
    knowledge.add(make_pattern_spec(pattern_id="mono", pattern="monotone", version="1.0",
        scope=KnowledgeScope(factors=("x",), objectives=("gain", "cost")),
        parameters={"direction": "increasing", "epsilon": .001, "delta_norm": .5, "n_pairs_per_dim": 3}, confidence=.8))
    with pytest.warns(UserWarning, match="Auto-rescued"):
        campaign = Campaign(space(), knowledge, validate=False)
    assert knowledge.specs[1].parameters["epsilon"] == .001
    assert campaign.knowledge.specs[1].parameters["epsilon"] == .3
    assert campaign.knowledge.specs[1].scope.objectives == ("gain", "cost")


@pytest.mark.parametrize("variant", ["categorical", "target", "discrete"])
def test_shape_validation_unsupported_domains_are_explicit(variant):
    params = [Parameter("x", bounds=(0., 12.))]
    objectives = space().objective_specs
    X = pd.DataFrame({"x": np.arange(10, dtype=float)})
    if variant == "categorical":
        params.append(Parameter("material", kind="categorical", values=("A", "B")))
        X["material"] = ["A", "B"] * 5
    elif variant == "target":
        objectives = (Objective("gain", "target", target=6.), objectives[1])
    else:
        params = [Parameter("x", kind="discrete", values=tuple(range(13)))]
    declaration = Space(params, objectives=objectives)
    knowledge = Knowledge().add(make_pattern_spec(pattern_id="shape", pattern="quadratic_peak", version="1.0",
        scope=KnowledgeScope(factors=("x",), objectives=("gain",)),
        parameters={"center": 6., "direction": "peak", "frozen": True}, confidence=.8))
    campaign = Campaign(declaration, knowledge, validation_min_obs=3, validation_interval=1)
    campaign.tell(X, oracle(X))
    assert campaign.validation_diagnostics[0]["status"] == "unsupported"
    assert campaign.validation_diagnostics[0]["reason"] == "shape_validator_requires_bounded_numeric_min_max_space"


def test_initial_catalog_detects_collision_with_existing_physical_observation():
    declaration = Space([Parameter("x", bounds=(-1., 1.))], objectives=space().objective_specs)
    campaign = Campaign(declaration, validate=False)
    X = pd.DataFrame({"x": [0.]})
    campaign.tell(X, oracle(X))
    with pytest.raises(EngineError, match="collision") as caught:
        campaign.run(lambda X: pytest.fail("must reject before oracle"), n_doe=3, n_iter=0,
            candidates=pd.DataFrame({"x": [1e-17, .2, .4, .6]}), candidate_ids=("near", "a", "b", "c"))
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_reused_catalog_detects_new_observation_collision_before_oracle():
    declaration = Space([Parameter("x", bounds=(-1., 1.))], objectives=space().objective_specs)
    campaign = Campaign(declaration, validate=False)
    X = pd.DataFrame({"x": [-1., -.5, 1.]})
    campaign.tell(X, oracle(X))
    campaign.ask(candidates=pd.DataFrame({"x": [1e-17, .2, .4, .6]}), candidate_ids=("near", "a", "b", "c"))
    X = pd.DataFrame({"x": [0.]})
    campaign.tell(X, oracle(X))
    with pytest.raises(EngineError, match="collision"):
        campaign.run(lambda X: pytest.fail("must reject before oracle"), n_doe=3, n_iter=0)


@pytest.mark.parametrize("order", [("matching", "mismatch"), ("mismatch", "matching")])
def test_shape_diagnostics_keep_original_pattern_identity_independent_of_order(order):
    from expdoe_dk.knowledge.validators import ShapePriorMismatchWarning
    declaration = Space([Parameter("x", bounds=(0., 12.))], objectives=space().objective_specs)
    X = pd.DataFrame({"x": np.arange(13, dtype=float)})
    Y = pd.DataFrame({"gain": -(X.x - 6.) ** 2, "cost": X.x})

    def check(names):
        knowledge = Knowledge()
        for name in names:
            knowledge.add(make_pattern_spec(pattern_id=name, pattern="quadratic_peak", version="1.0",
                scope=KnowledgeScope(factors=("x",), objectives=("gain",)), confidence=.8,
                parameters={"center": 6. if name == "matching" else 0., "direction": "peak", "frozen": True}))
        campaign = Campaign(declaration, knowledge, validation_min_obs=3, validation_interval=1)
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            campaign.tell(X, Y)
        assert sum(isinstance(w.message, ShapePriorMismatchWarning) for w in notices) == ("mismatch" in names)
        return {d["pattern_id"]: (d["status"], d["violation"]) for d in campaign.validation_diagnostics}

    assert check(("matching",)) == {"matching": ("checked", False)}
    assert check(("mismatch",)) == {"mismatch": ("checked", True)}
    assert check(order) == {"matching": ("checked", False), "mismatch": ("checked", True)}


@pytest.mark.parametrize("mode", ["known", "noiseless"])
def test_campaign_recommendation_preserves_missing_variance_markers_and_prefix(mode):
    campaign = Campaign(space(), validate=False, multiobjective=MultiObjectiveConfig(noise_mode=mode))
    X = pd.DataFrame({"x": [0, 2, 4, 6, 8, 10]})
    Y = oracle(X)
    Y.iloc[3:] = np.nan
    recorded = 0. if mode == "noiseless" else .1
    variance = pd.DataFrame({"gain": pd.Series([recorded] * 3 + [None, np.nan, pd.NA], dtype=object),
                             "cost": pd.Series([recorded] * 3 + [pd.NA, None, np.nan], dtype=object)})
    batch = ObservationBatch(X, Y, Yvar=variance,
        status=pd.Series(["success"] * 3 + ["failed", "warning", "failed"]))
    campaign.tell_batch(batch)
    result = campaign.ask_with_diagnostics(q=2)
    assert result.diagnostics["noise_mode"] == mode
    assert result.predictions.mean.shape == (2, 2)
    assert not set(result.frame.x) & set(X.x)
    original = campaign._combined_observations()
    assert original.Yvar.gain.iloc[3] is None
    assert np.isnan(original.Yvar.gain.iloc[4])
    assert original.Yvar.gain.iloc[5] is pd.NA
    assert _source_digest(campaign.space, original) == campaign._geometry.source_digest
    campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [12]}),
        pd.DataFrame({"gain": [None], "cost": [None]}), status=pd.Series(["failed"])))
    combined = campaign._combined_observations()
    assert combined.Yvar.iloc[-1].isna().all()
    prefix = ObservationBatch(combined.X.iloc[:6], combined.Y.iloc[:6], Yvar=combined.Yvar.iloc[:6],
        status=combined.status.iloc[:6], ids=combined.ids[:6])
    assert _source_digest(campaign.space, prefix) == campaign._geometry.source_digest
    assert campaign._observation_batches[-1].Yvar is None
    pd.testing.assert_frame_equal(batch.Yvar, variance[Y.columns])


def test_combined_variance_does_not_invent_zeros_for_unmeasured_failed_batch():
    campaign = Campaign(space(), validate=False)
    X = pd.DataFrame({"x": [0, 2, 4]})
    campaign.tell_batch(ObservationBatch(X, oracle(X), Yvar=oracle(X) * 0 + .1))
    campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [6]}),
        pd.DataFrame({"gain": [None], "cost": [None]}), status=pd.Series(["failed"])))
    assert campaign._combined_observations().Yvar.iloc[-1].isna().all()
    assert campaign._observation_batches[-1].Yvar is None


@pytest.mark.parametrize("status", ["failed", "warning"])
@pytest.mark.parametrize("recorded", [0., .25])
def test_combined_rejects_recorded_unsuccessful_variance_when_success_noise_is_missing(status, recorded):
    campaign = initial()
    batch = ObservationBatch(pd.DataFrame({"x": [6]}),
        pd.DataFrame({"gain": [None], "cost": [None]}),
        Yvar=pd.DataFrame({"gain": pd.Series([pd.NA], dtype=object), "cost": [recorded]}),
        status=pd.Series([status]))
    campaign.tell_batch(batch)
    for consume in (campaign._combined_observations, campaign.ask):
        with pytest.raises(EngineError, match="variance") as caught:
            consume()
        assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert campaign._observation_batches[0].Yvar is None
    assert campaign._observation_batches[1].Yvar.cost.iloc[0] == recorded
    assert len(campaign.history_df()) == 4
    assert campaign._ask_iteration == 0 and campaign._geometry is None


def test_combined_accepts_only_missing_unsuccessful_variance_with_unknown_success_noise():
    campaign = initial()
    batch = ObservationBatch(pd.DataFrame({"x": [6]}),
        pd.DataFrame({"gain": [None], "cost": [None]}),
        Yvar=pd.DataFrame({"gain": pd.Series([pd.NA], dtype=object),
                           "cost": pd.Series([None], dtype=object)}), status=pd.Series(["failed"]))
    campaign.tell_batch(batch)
    assert campaign._combined_observations().Yvar is None
    pd.testing.assert_frame_equal(campaign._observation_batches[-1].Yvar, batch.Yvar)


def test_empirical_projection_handles_object_and_nullable_missing_variance():
    from expdoe_dk.knowledge.validators import MonotoneViolationWarning
    knowledge = Knowledge().add(make_pattern_spec(pattern_id="both", pattern="monotone", version="1.0",
        scope=KnowledgeScope(factors=("x",), objectives=("gain", "cost")), confidence=.8,
        parameters={"direction": "increasing", "epsilon": "auto", "delta_norm": .5, "n_pairs_per_dim": 3}))
    campaign = Campaign(space(), knowledge, validation_min_obs=3, validation_interval=1)
    X = pd.DataFrame({"x": list(range(13))})
    Y = oracle(X)
    Y.iloc[10:] = np.nan
    variance = pd.DataFrame({"gain": pd.Series([.1] * 10 + [None, np.nan, pd.NA], dtype=object),
                             "cost": pd.Series([.2] * 10 + [pd.NA] * 3, dtype="Float64")})
    batch = ObservationBatch(X, Y, Yvar=variance,
        status=pd.Series(["success"] * 10 + ["failed", "warning", "failed"]))
    with pytest.warns(MonotoneViolationWarning, match=r"cost: \[both\]"):
        campaign.tell_batch(batch)
    assert {d["objective_name"]: (d["pattern_id"], d["status"], d["violation"])
            for d in campaign.validation_diagnostics} == {
                "gain": ("both", "checked", False), "cost": ("both", "checked", True)}
    pd.testing.assert_frame_equal(campaign._observation_batches[0].Yvar, variance[space().objectives])
    pd.testing.assert_frame_equal(batch.Yvar, variance[Y.columns])
