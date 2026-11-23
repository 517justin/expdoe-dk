"""Fixed-catalog recommendation must never invent or misidentify a formula."""
import json

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import (
    EngineError, ErrorCode, Knowledge, Objective,
    ObservationBatch, OutcomeConstraint, Parameter, PendingBatch, Space, recommend,
)
from expdoe_dk.domain import LinearConstraint


def line(direction="maximize", outcome_constraints=()):
    space = Space([Parameter("x", bounds=(0, 10))],
                  objectives=Objective("y", direction), outcome_constraints=outcome_constraints)
    observations = ObservationBatch(pd.DataFrame({"x": [2., 4., 6., 8.]}),
        pd.DataFrame({"y": [120., 140., 160., 180.]}),
        Yvar=pd.DataFrame({"y": [0.001] * 4}), ids=("t2", "t4", "t6", "t8"))
    return space, observations


@pytest.mark.parametrize("direction,want", [("maximize", "high"), ("minimize", "low")])
def test_catalog_uses_objective_direction_and_original_values(direction, want):
    catalog = pd.DataFrame({"x": [0.123456789012345, 9.87654321098765]})
    original = catalog.copy(deep=True)
    result = recommend(*line(direction), candidates=catalog, candidate_ids=("low", "high"), seed=17)
    assert result.candidate_ids == (want,)
    expected = catalog.iloc[[0 if want == "low" else 1]].reset_index(drop=True)
    pd.testing.assert_frame_equal(result.frame, expected, check_exact=True)
    pd.testing.assert_frame_equal(catalog, original)
    assert result.predictions.mean.item() > 80  # Physical, not standardized units.
    assert result.diagnostics["search_strategy"] == "provided_catalog"
    assert result.diagnostics["continuous_subspaces_optimized"] == 0
    assert result.to_dict()["candidate_ids"] == [want]
    json.dumps(result.to_dict(), allow_nan=False)


def test_catalog_excludes_observed_failed_and_pending_conditions_not_trial_ids():
    space, observations = line()
    observations = ObservationBatch(
        pd.concat([observations.X, pd.DataFrame({"x": [9.]})], ignore_index=True),
        pd.concat([observations.Y, pd.DataFrame({"y": [np.nan]})], ignore_index=True),
        status=pd.Series(["success"] * 4 + ["failed"]))
    catalog = pd.DataFrame({"x": [2., 9., 10., 1., 3.]})
    result = recommend(space, observations, candidates=catalog,
        candidate_ids=("observed", "failed", "pending", "a", "b"),
        pending=PendingBatch(("trial-pending",), pd.DataFrame({"x": [10.]})), q=2, seed=17)
    assert set(result.candidate_ids) == {"a", "b"}
    assert set(result.frame.x) == {1., 3.}


def test_catalog_reordering_preserves_id_alignment_and_predictions():
    catalog = pd.DataFrame({"x": [1., 3., 7., 9.]}, index=[30, 30, 10, 20])
    ids = ("d", "c", "b", "a")
    first = recommend(*line(), candidates=catalog, candidate_ids=ids, q=2, seed=17)
    order = [2, 0, 3, 1]
    second = recommend(*line(), candidates=catalog.iloc[order],
                       candidate_ids=tuple(ids[i] for i in order), q=2, seed=17)
    assert first.candidate_ids == second.candidate_ids
    pd.testing.assert_frame_equal(first.frame, second.frame, check_exact=True)
    np.testing.assert_allclose(first.predictions.mean, second.predictions.mean)
    for identifier, value in zip(first.candidate_ids, first.frame.x):
        assert value == dict(zip(ids, catalog.x))[identifier]


@pytest.mark.parametrize("catalog,ids", [
    (pd.DataFrame({"x": [1.]}), None),
    (None, ("a",)),
    (pd.DataFrame({"x": [1., 3.]}), ("a",)),
    (pd.DataFrame({"x": [1., 3.]}), ("a", "a")),
    (pd.DataFrame({"x": [1.]}), ("",)),
    (pd.DataFrame({"x": [1.]}), (1,)),
    (pd.DataFrame({"x": [1.]}), "a"),
    (pd.DataFrame({"x": [1., 1.]}), ("a", "b")),
    (pd.DataFrame({"x": [1.], "y": [999.]}), ("a",)),
    (pd.DataFrame({"wrong": [1.]}), ("a",)),
    (pd.DataFrame([[1., 2.]], columns=["x", "x"]), ("a",)),
    (pd.DataFrame({"x": [np.nan]}), ("a",)),
    (pd.DataFrame({"x": [11.]}), ("a",)),
    ([1.], ("a",)),
])
def test_catalog_invalid_inputs_are_typed(catalog, ids):
    with pytest.raises(EngineError) as error:
        recommend(*line(), candidates=catalog, candidate_ids=ids)
    assert error.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("values,ids,q", [([], (), 1), ([2., 1.], ("seen", "one"), 2)])
def test_catalog_exhaustion_never_returns_partial_batch(values, ids, q):
    with pytest.raises(EngineError) as error:
        recommend(*line(), candidates=pd.DataFrame({"x": values}), candidate_ids=ids, q=q)
    assert error.value.code == ErrorCode.NO_FEASIBLE_CANDIDATE


def test_catalog_respects_knowledge_and_outcome_safety():
    space, observations = line(outcome_constraints=(OutcomeConstraint("safe", "y", "<=", 150.),))
    knowledge = Knowledge().with_safe_region(expression="x >= 1", factors=("x",))
    result = recommend(space, observations, knowledge=knowledge,
        candidates=pd.DataFrame({"x": [0., 1., 3., 9., 10.]}),
        candidate_ids=("below", "a", "b", "unsafe1", "unsafe2"), q=2, seed=17)
    assert set(result.candidate_ids) == {"a", "b"}
    assert result.diagnostics["parameter_constraint_sources"]
    assert min(result.diagnostics["outcome_safety"]["probabilities"]) >= .95


def test_catalog_mixed_factors_preserves_rows_and_hard_equality():
    space = Space([
        Parameter("a", bounds=(0., 1.)), Parameter("b", bounds=(0., 1.)),
        Parameter("material", kind="categorical", values=("A", "B")),
        Parameter("grade", kind="ordinal", values=("low", "high")),
    ], objectives="y", constraints=[LinearConstraint("sum", {"a": 1., "b": 1.}, "==", 1.)])
    X = pd.DataFrame({"a": [0., .5, 1.], "b": [1., .5, 0.],
                      "material": ["A", "B", "A"], "grade": ["low", "high", "low"]})
    observations = ObservationBatch(X, pd.DataFrame({"y": [1., 2., 3.]}))
    catalog = pd.DataFrame({"grade": ["high", "low", "high"], "material": ["A", "B", "A"],
                            "b": [.7, .3, .8], "a": [.3, .7, .8]})
    out = recommend(space, observations, candidates=catalog, candidate_ids=("a", "b", "bad"), q=2)
    assert set(out.candidate_ids) == {"a", "b"}
    for i, identifier in enumerate(out.candidate_ids):
        expected = catalog.iloc[[0 if identifier == "a" else 1]][space.param_names].reset_index(drop=True)
        pd.testing.assert_frame_equal(out.frame.iloc[[i]].reset_index(drop=True), expected, check_exact=True)
    assert space.feasibility_mask(out.frame).all()


def test_catalog_does_not_remove_multiobjective_guard():
    space, observations = line()
    space = Space(space.params, objectives=(Objective("y", "maximize"), Objective("cost", "minimize")))
    with pytest.raises(EngineError) as error:
        recommend(space, observations, candidates=pd.DataFrame({"x": [1.]}), candidate_ids=("a",))
    assert error.value.code == ErrorCode.ACQUISITION_INCOMPATIBLE


def test_catalog_exact_large_integers_survive_mixed_columns_and_serialization():
    base = 2**53 + 10
    space = Space([Parameter("count", kind="integer", bounds=(base, base + 8), step=2),
                   Parameter("fraction", bounds=(0., 1.))], objectives="y")
    observations = ObservationBatch(
        pd.DataFrame({"count": [base, base + 2, base + 4], "fraction": [.1, .5, .9]}),
        pd.DataFrame({"y": [1., 2., 3.]}))
    catalog = pd.DataFrame({"count": [base + 6, base + 8], "fraction": [.123, .987]})
    result = recommend(space, observations, candidates=catalog, candidate_ids=("a", "b"), q=2)
    by_id = dict(zip(result.candidate_ids, result.to_dict()["frame"]))
    assert by_id["a"] == {"count": base + 6, "fraction": .123}
    assert by_id["b"] == {"count": base + 8, "fraction": .987}
    assert type(by_id["a"]["count"]) is int
    returned_frame = result.frame
    returned_frame.loc[:, "count"] = base
    assert set(result.frame["count"]) == {base + 6, base + 8}


def test_catalog_result_ids_are_validated_and_detached():
    from expdoe_dk import Prediction, RecommendationResult
    frame = pd.DataFrame({"x": [1., 2.]})
    predictions = Prediction(np.ones((2, 1)), np.ones((2, 1)))
    ids = ["a", "b"]
    result = RecommendationResult(frame, predictions, {}, candidate_ids=ids)
    ids[0] = "changed"
    assert result.candidate_ids == ("a", "b")
    for invalid_ids in [("a",), ("a", "a"), ("", "b"), (1, "b"), "ab"]:
        with pytest.raises(ValueError):
            RecommendationResult(frame, predictions, {}, candidate_ids=invalid_ids)


def test_catalog_scores_all_294_supplied_rows_without_expanding_them():
    # A local synthetic catalog, not the textbook's experimental dataset.
    catalog = pd.DataFrame({"x": np.linspace(.01, 9.99, 294)})
    ids = tuple(f"formula-{i:03}" for i in range(294))
    result = recommend(*line(), candidates=catalog, candidate_ids=ids, q=2, seed=17)
    assert result.diagnostics["pool_size"] == 294
    assert len(result.candidate_ids) == len(set(result.candidate_ids)) == 2
    for identifier, value in zip(result.candidate_ids, result.frame.x):
        assert value == catalog.x.iloc[ids.index(identifier)]


def test_returned_mixed_numeric_ordinal_values_remain_feasible_after_copy():
    large = 2**53 + 1
    space = Space([Parameter("level", kind="ordinal", values=(.5, large, 1, 2, 3))], objectives="y")
    observed = ObservationBatch(pd.DataFrame({"level": pd.Series([1, 2, 3], dtype=object)}),
                                pd.DataFrame({"y": [1., 2., 3.]}))
    catalog = pd.DataFrame({"level": pd.Series([.5, large], dtype=object)})
    result = recommend(space, observed, candidates=catalog, candidate_ids=("fraction", "large"), q=2)
    assert space.feasibility_mask(result.frame).all()
    for identifier, row in zip(result.candidate_ids, result.to_dict()["frame"]):
        expected = .5 if identifier == "fraction" else large
        assert row["level"] == expected
        assert type(row["level"]) is type(expected)


@pytest.mark.parametrize("source", ["catalog", "observations", "pending"])
def test_model_precision_collision_is_not_reported_as_duplicate_or_exhaustion(source):
    space = Space([Parameter("x", bounds=(-1., 1.))], objectives="y")
    observations = ObservationBatch(pd.DataFrame({"x": [-1., 0., 1.]}),
                                    pd.DataFrame({"y": [1., 2., 3.]}))
    pending = None
    values, ids = [1e-17], ["tiny"]
    if source == "catalog":
        values, ids = [0., 1e-17], ["zero", "tiny"]
    elif source == "pending":
        observations = ObservationBatch(pd.DataFrame({"x": [-1., -.5, 1.]}), observations.Y)
        pending = PendingBatch(("trial-zero",), pd.DataFrame({"x": [0.]}))
    with pytest.raises(EngineError) as error:
        recommend(space, observations, pending=pending,
                  candidates=pd.DataFrame({"x": values}), candidate_ids=ids)
    assert error.value.code == ErrorCode.CONFIG_INVALID
    assert "model-coordinate collision" in str(error.value).lower()
