"""Schema 2.2 persists measured vectors and the exact next recommendation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import (
    Campaign, EngineError, ErrorCode, Knowledge, KnowledgeScope, MultiObjectiveConfig, Objective,
    ObservationBatch, Parameter, PendingBatch, Space, make_pattern_spec,
)
from expdoe_dk.bo.checkpoint import validate_state


def space():
    return Space([Parameter("x", kind="integer", bounds=(0, 20)),
                  Parameter("material", kind="categorical", values=["A", "B"])],
                 objectives=[Objective("gain", "maximize"), Objective("cost", "minimize")])


def catalog():
    return pd.DataFrame({"x": np.arange(21, dtype=np.int32),
                         "material": ["A", "B", "A"] * 7}), tuple(f"c{i:02}" for i in range(21))


def oracle(X):
    return pd.DataFrame({"gain": X.x.to_numpy() ** 2 + 1., "cost": 30. - X.x.to_numpy()})


def initial(*, config=None):
    campaign = Campaign(space(), seed=17, validate=False, multiobjective=config)
    frame, _ = catalog()
    X = frame.iloc[[0, 2, 4]].reset_index(drop=True)
    campaign.tell_batch(ObservationBatch(X, oracle(X)))
    return campaign


@pytest.mark.parametrize("preferences", [None, {"gain": .8, "cost": .2}])
def test_resume_next_catalog_batch_and_weights(tmp_path, preferences):
    campaign = initial(config=MultiObjectiveConfig(
        acquisition="q_log_nparego", preference_weights=preferences,
        reference_point={"gain": 0., "cost": 40.}))
    frame, ids = catalog()
    first = campaign.ask_with_diagnostics(q=2, candidates=frame, candidate_ids=ids)
    frozen = campaign._geometry.to_dict()
    campaign.tell(first.frame, oracle(first.frame) * 100.)
    campaign.set_pending(PendingBatch(("repeat-trial",), frame.iloc[[0]]))
    campaign._rng.random(4)
    path = tmp_path / "checkpoint.json"
    campaign.save_checkpoint(path)
    restored = Campaign.load_checkpoint(path)
    assert restored.to_state() == campaign.to_state()
    pd.testing.assert_frame_equal(restored._catalog, campaign._catalog)
    pd.testing.assert_frame_equal(restored.history_df(), campaign.history_df())
    assert restored.history_columns == campaign.history_columns
    actual = restored.ask_with_diagnostics(q=2, candidates=frame.iloc[::-1], candidate_ids=ids[::-1])
    expected = campaign.ask_with_diagnostics(q=2)
    assert actual.candidate_ids == expected.candidate_ids
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    assert actual.diagnostics["weights_by_slot"] == expected.diagnostics["weights_by_slot"]
    assert actual.diagnostics["geometry"] == expected.diagnostics["geometry"] == frozen
    np.testing.assert_array_equal(restored._rng.random(5), campaign._rng.random(5))
    assert restored._observation_batches[0].ids == ()
    assert restored._observation_ids == campaign._observation_ids
    assert restored._resolved_declaration == campaign._resolved_declaration


@pytest.mark.parametrize("missing", [None, pd.NA, float("nan"), float("inf"), -float("inf")])
def test_unsuccessful_markers_known_variance_and_initial_prefix(missing):
    campaign = Campaign(space(), seed=17, validate=False)
    X = catalog()[0].iloc[:5]
    Y = oracle(X).astype(object)
    Y.loc[3:, :] = missing
    variance = pd.DataFrame({"cost": pd.Series([.2, .2, .2, None, pd.NA], dtype=object),
                             "gain": pd.Series([.1, .1, .1, np.nan, .3], dtype=object)})
    campaign.tell_batch(ObservationBatch(X, Y, Yvar=variance,
        status=pd.Series(["success"] * 3 + ["failed", "warning"])))
    campaign.ask(candidates=catalog()[0], candidate_ids=catalog()[1])
    resumed = Campaign.from_state(campaign.to_state())
    expected = campaign.ask_with_diagnostics(q=2)
    actual = resumed.ask_with_diagnostics(q=2)
    assert actual.candidate_ids == expected.candidate_ids
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    assert actual.diagnostics["geometry"] == expected.diagnostics["geometry"]
    assert actual.diagnostics["weights_by_slot"] == expected.diagnostics["weights_by_slot"]
    # A later unsupported mixed-noise view must not prevent archival persistence.
    later = catalog()[0].iloc[[6]]
    campaign.tell(later, oracle(later))
    state = campaign.to_state()
    restored = Campaign.from_state(json.loads(json.dumps(state, allow_nan=False)))
    assert restored.to_state() == state
    assert restored._geometry.to_dict() == campaign._geometry.to_dict()
    for actual, expected in zip(restored._observation_batches, campaign._observation_batches):
        pd.testing.assert_frame_equal(actual.Y, expected.Y)
        if expected.Yvar is not None:
            pd.testing.assert_frame_equal(actual.Yvar, expected.Yvar)
    with pytest.raises(EngineError, match="Mixed missing variance"):
        restored.ask()


def test_empty_and_unresolved_geometry_roundtrip():
    for campaign in (Campaign(space(), validate=False), initial()):
        state = campaign.to_state()
        assert state["schema_version"] == "2.2"
        assert state["geometry"] is None and state["catalog"] is None
        assert Campaign.from_state(state).to_state() == state


@pytest.fixture
def resolved_state():
    campaign = initial()
    campaign.ask(candidates=catalog()[0], candidate_ids=catalog()[1])
    return campaign.to_state()


@pytest.mark.parametrize("path,value", [
    (("schema_version",), "2.3"), (("schema_version",), "2.1"),
    (("engine_version",), "0.6.0"), (("space", "engine_version"), "0.5.99"),
    (("knowledge", "engine_version"), "0.5.99"),
    (("geometry", "location"), [0.]),
    (("geometry", "report_reference"), [float("inf"), 0.]),
    (("geometry", "source_digest"), "0" * 64),
    (("geometry", "source_row_count"), 100),
    (("geometry", "acquisition_ideal"), [999., 999.]),
    (("geometry", "objective_names"), ["cost", "gain"]),
    (("observations", 0, "Y", "columns"), ["gain"]),
    (("observations", 0, "Y", "data"), [[1., 2.]]),
    (("observation_ids", 0), ["duplicate"] * 3),
    (("batch_kinds",), ["bo"]), (("trial_kind",), ["bo"] * 3),
    (("catalog", "ids"), ["duplicate"] * 21),
    (("catalog", "X", "dtypes", 0), "evil.module:load"),
    (("multiobjective", "reference_point"), {"gain": 0.}),
    (("validation", "emitted"), [["gain", "monotone"]]),
    (("rng_state", "state", "inc"), 2), (("ask_iteration",), 0),
])
def test_rejects_corrupt_schema_and_semantics(resolved_state, path, value):
    raw = deepcopy(resolved_state)
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    for load in (validate_state, Campaign.from_state):
        with pytest.raises(EngineError) as caught:
            load(raw)
        assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


def test_rejects_mixed_schema_fields_and_missing_config(resolved_state):
    for key in ("history", "pickle", "callable"):
        raw = deepcopy(resolved_state)
        raw[key] = []
        with pytest.raises(EngineError):
            validate_state(raw)
    del resolved_state["multiobjective"]
    with pytest.raises(EngineError):
        validate_state(resolved_state)


def test_same_condition_distinct_trial_legal_but_overlap_illegal():
    campaign = initial()
    X = catalog()[0].iloc[[0]]
    campaign.set_pending(PendingBatch(("new-trial",), X))
    assert Campaign.from_state(campaign.to_state())._pending.ids == ("new-trial",)
    campaign.set_pending(PendingBatch((campaign._observation_ids[0][0],), X))
    with pytest.raises(EngineError) as caught:
        campaign.to_state()
    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


def test_original_object_numeric_types_and_large_integers_roundtrip():
    campaign = initial()
    X = catalog()[0].iloc[[8, 9]].reset_index(drop=True).astype(object)
    X["x"] = pd.Series([np.int32(8), 9.], dtype=object)
    Y = pd.DataFrame({"gain": pd.Series([2**60 + 1, np.int64(2**60 + 3)], dtype=object),
                      "cost": pd.Series([np.float32(.25), 1], dtype=object)})
    campaign.tell_batch(ObservationBatch(X, Y, ids=("big", "numpy")))
    restored = Campaign.from_state(campaign.to_state())
    for name in ("X", "Y"):
        before = getattr(campaign._observation_batches[-1], name)
        after = getattr(restored._observation_batches[-1], name)
        pd.testing.assert_frame_equal(after, before)
        for col in before:
            assert [type(v) for v in after[col]] == [type(v) for v in before[col]]


def test_multiobjective_run_can_checkpoint(tmp_path):
    path = tmp_path / "run.json"
    campaign = initial()
    campaign.run(oracle, n_doe=3, n_iter=1, checkpoint=path,
                 candidates=catalog()[0], candidate_ids=catalog()[1])
    assert Campaign.load_checkpoint(path).to_state() == campaign.to_state()


def test_literal_schema22_fixture():
    path = Path(__file__).parents[1] / "fixtures" / "multiobjective_checkpoint_2_2.json"
    raw = json.loads(path.read_text())
    assert raw["engine_version"] == raw["space"]["engine_version"] == raw["knowledge"]["engine_version"] == "0.5.1"
    assert Campaign.load_checkpoint(path).to_state() == raw


@pytest.mark.parametrize("dtype", ["Float64", "Float32", "object"])
def test_nullable_variance_and_markers_are_preserved(dtype):
    campaign = Campaign(space(), validate=False)
    X = catalog()[0].iloc[:4]
    variance = pd.DataFrame({name: pd.Series([.25, .5, .25, pd.NA], dtype=dtype)
                             for name in space().objectives})
    campaign.tell_batch(ObservationBatch(X, oracle(X), Yvar=variance,
        status=pd.Series(["success"] * 3 + ["failed"])))
    campaign.ask()
    restored = Campaign.from_state(campaign.to_state())
    pd.testing.assert_frame_equal(restored._observation_batches[0].Yvar, variance)
    assert restored._observation_batches[0].Yvar.iloc[3, 0] is pd.NA


def test_later_recorded_variance_does_not_invalidate_unknown_noise_prefix():
    campaign = initial()
    campaign.ask()
    X = catalog()[0].iloc[[8]]
    campaign.tell_batch(ObservationBatch(X, oracle(X),
        Yvar=pd.DataFrame({"gain": [.1], "cost": [.2]}), status=pd.Series(["failed"])))
    restored = Campaign.from_state(campaign.to_state())
    assert restored.to_state() == campaign.to_state()
    with pytest.raises(EngineError, match="Recorded unsuccessful variance"):
        restored.ask()


def test_effective_ids_allocated_against_past_pending_are_never_regenerated():
    campaign = Campaign(space(), validate=False)
    frame, _ = catalog()
    campaign.set_pending(PendingBatch(("unidentified:0:0",), frame.iloc[[8]]))
    campaign.tell(frame.iloc[:3], oracle(frame.iloc[:3]))
    campaign.set_pending(None)
    assert campaign._observation_ids[0][0] == "unidentified:0:0:"
    campaign.ask()
    restored = Campaign.from_state(campaign.to_state())
    assert restored._observation_ids == campaign._observation_ids
    restored.tell_batch(ObservationBatch(frame.iloc[[8]], oracle(frame.iloc[[8]]), ids=("unidentified:0:0",)))
    assert restored._observation_ids[0][0] == "unidentified:0:0:"


def test_empirical_warning_identity_and_diagnostics_roundtrip(monkeypatch):
    knowledge = Knowledge().add(make_pattern_spec(
        pattern="monotone", version="1.0", pattern_id="gain-monotone",
        parameters={"direction": "decreasing", "epsilon": "auto", "delta_norm": .5, "n_pairs_per_dim": 3},
        scope=KnowledgeScope(factors=("x",), objectives=("gain",)), confidence=.8))
    numeric_space = Space([Parameter("x", kind="integer", bounds=(0, 20))], objectives=space().objective_specs)
    campaign = Campaign(numeric_space, knowledge, validate=True, validation_min_obs=3, auto_rescue=False)
    X = catalog()[0].iloc[:10][["x"]]
    with pytest.warns(UserWarning, match="gain"):
        campaign.tell(X, oracle(X))
    state = campaign.to_state()
    assert state["validation"]["emitted"]
    assert all(len(key) == 4 for key in state["validation"]["emitted"])
    assert state["validation"]["diagnostics"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Restore must not replay empirical validators")

    monkeypatch.setattr(Campaign, "_maybe_run_validators", forbidden)
    restored = Campaign.from_state(state)
    assert restored._validation_emitted == campaign._validation_emitted
    assert restored.validation_diagnostics == campaign.validation_diagnostics
    assert restored._last_validation_n == campaign._last_validation_n
    assert restored.to_state() == state


@pytest.mark.parametrize("field", ["X", "Y", "Yvar"])
def test_rejects_lossy_column_dtype_conversion(field):
    raw = json.loads((Path(__file__).parents[1] / "fixtures" / "multiobjective_checkpoint_2_2.json").read_text())
    if field == "Y":
        raw["observations"][0][field]["data"][0][0] = .1
    raw["observations"][0][field]["dtypes"][0] = "float32"
    with pytest.raises(EngineError) as caught:
        validate_state(raw)
    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


def test_unsupported_catalog_dtype_rejected_before_replacement(tmp_path):
    campaign = initial()
    frame, ids = catalog()
    frame["material"] = frame.material.astype("category")
    campaign.ask(candidates=frame, candidate_ids=ids)
    target = tmp_path / "existing.json"
    target.write_text("original bytes")
    with pytest.raises(EngineError, match="dtype") as caught:
        campaign.save_checkpoint(target)
    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE
    assert target.read_text() == "original bytes"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_replace_failure_preserves_target_and_cleans_temporary(tmp_path, monkeypatch):
    from expdoe_dk.bo import checkpoint
    target = tmp_path / "existing.json"
    initial().save_checkpoint(target)
    original = target.read_bytes()

    def fail(*args):
        raise OSError("replace unavailable")

    monkeypatch.setattr(checkpoint.os, "replace", fail)
    with pytest.raises(OSError, match="replace unavailable"):
        initial().save_checkpoint(target)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("section", ["space", "knowledge"])
def test_schema22_rejects_mismatched_historical_engine_envelopes(section):
    raw = initial().to_state()
    raw[section]["engine_version"] = "0.5.0"
    with pytest.raises(EngineError) as caught:
        Campaign.from_state(raw)
    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE


def test_object_catalog_restores_exact_scalar_types_and_next_returned_rows():
    campaign = initial()
    frame, ids = catalog()
    frame["x"] = pd.Series([np.int32(x) if x % 2 else float(x) for x in range(21)], dtype=object)
    campaign.ask(candidates=frame, candidate_ids=ids)
    restored = Campaign.from_state(campaign.to_state())
    assert [type(v) for v in restored._catalog.x] == [type(v) for v in frame.x]
    actual = restored.ask_with_diagnostics(q=2, candidates=frame, candidate_ids=ids)
    expected = campaign.ask_with_diagnostics(q=2)
    assert actual.candidate_ids == expected.candidate_ids
    for i, identifier in enumerate(actual.candidate_ids):
        original = frame.iloc[ids.index(identifier)]
        for column in frame:
            assert type(actual.frame.iloc[i][column]) is type(original[column])
            assert actual.frame.iloc[i][column] == original[column]


def test_geometry_prefix_spans_original_batches_and_history_role_collisions():
    collision_space = Space([Parameter("gain", bounds=(0., 1.)), Parameter("status", bounds=(0., 1.))],
                            objectives=space().objective_specs)
    campaign = Campaign(collision_space, validate=False)
    for x in (0., .5, 1.):
        campaign.tell(pd.DataFrame({"gain": [x], "status": [x]}),
                      pd.DataFrame({"gain": [x + 1.], "cost": [10. - x]}))
    campaign.ask()
    campaign.tell(pd.DataFrame({"gain": [.25], "status": [.25]}),
                  pd.DataFrame({"gain": [100.], "cost": [-100.]}))
    restored = Campaign.from_state(campaign.to_state())
    assert restored._geometry.source_row_count == 3
    assert restored.history_columns == campaign.history_columns
    pd.testing.assert_frame_equal(restored.history_df(), campaign.history_df())


@pytest.mark.parametrize("mutation", [
    "success_missing", "variance_negative", "variance_inf", "explicit_ids", "effective_count",
    "pending_count", "pending_overlap", "catalog_count", "catalog_collision", "dtype_count",
    "dtype_rounding", "cell_callable", "geometry_unknown", "unresolved_catalog", "scalar_space",
    "validation_count", "config_bool", "numpy_overflow",
])
def test_additional_semantic_corruptions(resolved_state, mutation):
    raw = deepcopy(resolved_state)
    batch = raw["observations"][0]
    if mutation == "success_missing":
        batch["Y"]["data"][0][0] = {"missing": "NA"}
    elif mutation in ("variance_negative", "variance_inf"):
        batch["Yvar"] = deepcopy(batch["Y"])
        batch["Yvar"]["data"][0][0] = -1. if mutation == "variance_negative" else {"missing": "+inf"}
    elif mutation == "explicit_ids":
        batch["ids"] = ["a", "b", "c"]
    elif mutation == "effective_count":
        raw["observation_ids"][0].pop()
    elif mutation.startswith("pending"):
        raw["pending"] = {"X": deepcopy(batch["X"]), "ids": ["p0", "p1", "p2"]}
        if mutation == "pending_count":
            raw["pending"]["ids"].pop()
        else:
            raw["pending"]["ids"][0] = raw["observation_ids"][0][0]
    elif mutation == "catalog_count":
        raw["catalog"]["ids"].pop()
    elif mutation == "catalog_collision":
        raw["catalog"]["X"]["data"][1] = raw["catalog"]["X"]["data"][0]
    elif mutation == "dtype_count":
        batch["X"]["dtypes"].pop()
    elif mutation == "dtype_rounding":
        batch["X"]["data"][0][0] = .1
    elif mutation == "cell_callable":
        batch["Y"]["data"][0][0] = {"callable": "os.system"}
    elif mutation == "geometry_unknown":
        raw["geometry"]["unknown"] = None
    elif mutation == "unresolved_catalog":
        raw["geometry"], raw["ask_iteration"] = None, 0
    elif mutation == "scalar_space":
        raw["space"]["objectives"].pop()
    elif mutation == "validation_count":
        raw["validation"]["last_n"] = 100
    elif mutation == "config_bool":
        raw["multiobjective"]["min_joint_feasibility"] = True
    elif mutation == "numpy_overflow":
        batch["Y"]["dtypes"][0] = "object"
        batch["Y"]["data"][0][0] = {"scalar": "numpy", "dtype": "int8", "value": 10000}
    with pytest.raises(EngineError) as caught:
        validate_state(raw)
    assert caught.value.code is ErrorCode.CHECKPOINT_INCOMPATIBLE
