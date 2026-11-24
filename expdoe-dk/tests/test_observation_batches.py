import warnings

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import ObservationBatch, PendingBatch


def test_object_numeric_conditions_keep_exact_scalars_in_both_batch_boundaries():
    values = [.5, 2**53 + 1]
    X = pd.DataFrame({"level": pd.Series(values, dtype=object)})
    observed = ObservationBatch(X, pd.DataFrame({"y": [1., 2.]}))
    pending = PendingBatch(("a", "b"), X)
    for frame in (observed.X, observed.successful().X, pending.X):
        assert frame.level.tolist() == values
        assert type(frame.level.iloc[1]) is int


def test_successful_filters_failed_rows_without_imputation():
    batch = ObservationBatch(
        X=pd.DataFrame({"x": [0.1, 0.2]}),
        Y=pd.DataFrame({"yield": [10.0, float("nan")]}),
        status=pd.Series(["success", "failed"]),
        ids=("o1", "o2"),
    )

    successful = batch.successful()

    assert successful.ids == ("o1",)
    assert successful.Y["yield"].tolist() == [10.0]


def test_pending_ids_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        PendingBatch(ids=("p1", "p1"), X=pd.DataFrame({"x": [0.1, 0.2]}))


def test_missing_status_means_all_rows_are_successful():
    batch = ObservationBatch(
        X=pd.DataFrame({"x": [0.1, 0.2]}, index=[5, 7]),
        Y=pd.DataFrame({"yield": [10.0, 12.0]}, index=[11, 13]),
    )

    successful = batch.successful()

    assert successful.X.index.tolist() == [0, 1]
    assert successful.Y["yield"].tolist() == [10.0, 12.0]
    assert successful.status.tolist() == ["success", "success"]


def test_successful_rows_require_finite_objectives():
    with pytest.raises(ValueError, match="successful.*finite"):
        ObservationBatch(
            X=pd.DataFrame({"x": [0.1]}),
            Y=pd.DataFrame({"yield": [float("nan")]}),
            status=pd.Series(["success"]),
        )


@pytest.mark.parametrize("value", ["1.25", True, 1 + 2j])
def test_successful_rows_reject_coercible_nonnumeric_boolean_or_complex_values(value):
    """Catches successful outcomes being silently coerced to lossy floats."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="successful.*finite.*real.*non-boolean"):
            ObservationBatch(
                X=pd.DataFrame({"x": [0.1]}),
                Y=pd.DataFrame({"yield": [value]}),
                status=pd.Series(["success"]),
            )


def test_successful_rows_accept_finite_real_numeric_scalar_types():
    """Catches strict outcome validation excluding genuine NumPy real scalars."""
    batch = ObservationBatch(
        X=pd.DataFrame({"x": [0.1, 0.2]}),
        Y=pd.DataFrame(
            {"yield": pd.Series([np.int64(4), np.float32(5.5)], dtype=object)}
        ),
    )

    assert batch.Y["yield"].tolist() == [np.int64(4), np.float32(5.5)]


def test_failed_and_warning_rows_may_retain_missing_objectives():
    batch = ObservationBatch(
        X=pd.DataFrame({"x": [0.1, 0.2]}),
        Y=pd.DataFrame({"yield": [float("nan"), float("nan")]}),
        status=pd.Series(["failed", "warning"]),
    )

    successful = batch.successful()

    assert successful.X.empty
    assert successful.Y.empty
    assert successful.status.empty


def test_yvar_requires_matching_finite_non_negative_numeric_values():
    with pytest.raises(ValueError, match="Yvar.*finite.*non-negative"):
        ObservationBatch(
            X=pd.DataFrame({"x": [0.1]}),
            Y=pd.DataFrame({"yield": [1.0]}),
            Yvar=pd.DataFrame({"yield": [-0.1]}),
        )

    with pytest.raises(ValueError, match="Yvar.*columns"):
        ObservationBatch(
            X=pd.DataFrame({"x": [0.1]}),
            Y=pd.DataFrame({"yield": [1.0]}),
            Yvar=pd.DataFrame({"other": [0.1]}),
        )


def test_batches_detach_inputs_and_return_defensive_frame_copies():
    X = pd.DataFrame({"x": [0.1]})
    Y = pd.DataFrame({"yield": [1.0]})
    batch = ObservationBatch(X=X, Y=Y, ids=("o1",))
    pending = PendingBatch(ids=("p1",), X=X)

    X.loc[0, "x"] = 99.0
    Y.loc[0, "yield"] = 99.0
    retrieved_X = batch.X
    retrieved_Y = batch.Y
    pending_X = pending.X
    retrieved_X.loc[0, "x"] = 88.0
    retrieved_Y.loc[0, "yield"] = 88.0
    pending_X.loc[0, "x"] = 77.0

    assert batch.X.loc[0, "x"] == 0.1
    assert batch.Y.loc[0, "yield"] == 1.0
    assert pending.X.loc[0, "x"] == 0.1


def test_batches_deeply_detach_mutable_object_cells():
    x_payload = {"levels": ["original-x"]}
    y_payload = ["original-y"]
    pending_payload = {"levels": ["original-pending"]}
    X = pd.DataFrame({"payload": [x_payload]})
    Y = pd.DataFrame({"payload": [y_payload]})
    pending_X = pd.DataFrame({"payload": [pending_payload]})
    batch = ObservationBatch(X=X, Y=Y, status=pd.Series(["failed"]))
    pending = PendingBatch(ids=("p1",), X=pending_X)

    x_payload["levels"].append("caller-change")
    y_payload.append("caller-change")
    pending_payload["levels"].append("caller-change")
    batch_X = batch.X
    batch_Y = batch.Y
    returned_pending_X = pending.X
    batch_X.loc[0, "payload"]["levels"].append("accessor-change")
    batch_Y.loc[0, "payload"].append("accessor-change")
    returned_pending_X.loc[0, "payload"]["levels"].append("accessor-change")

    assert batch.X.loc[0, "payload"] == {"levels": ["original-x"]}
    assert batch.Y.loc[0, "payload"] == ["original-y"]
    assert pending.X.loc[0, "payload"] == {"levels": ["original-pending"]}


def test_successful_batch_deeply_detaches_mutable_condition_cells():
    batch = ObservationBatch(
        X=pd.DataFrame({"payload": [["original"]]}),
        Y=pd.DataFrame({"yield": [1.0]}),
    )
    successful = batch.successful()
    successful_X = successful.X

    successful_X.loc[0, "payload"].append("accessor-change")

    assert successful.X.loc[0, "payload"] == ["original"]


def test_yvar_rejects_complex_values_without_complex_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="Yvar.*numeric"):
            ObservationBatch(
                X=pd.DataFrame({"x": [0.1]}),
                Y=pd.DataFrame({"yield": [1.0]}),
                Yvar=pd.DataFrame({"yield": [1 + 100j]}),
            )


def test_multioutput_yvar_aligns_independent_column_order_without_mutating_input():
    X = pd.DataFrame({"x": [1., 2.]})
    Y = pd.DataFrame({"a": [3., 4.], "b": [5., 6.]})
    variance = pd.DataFrame({"b": [.5, .6], "a": [.3, .4]}, index=[8, 9])
    original = variance.copy(deep=True)
    batch = ObservationBatch(X, Y, Yvar=variance)
    pd.testing.assert_frame_equal(batch.Yvar, original[["a", "b"]].reset_index(drop=True))
    pd.testing.assert_frame_equal(variance, original)
    variance.iloc[0, 0] = 100.
    assert batch.Yvar.b.tolist() == [.5, .6]


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
@pytest.mark.parametrize("status", ["failed", "warning"])
def test_multioutput_missing_unsuccessful_variance_preserves_original_marker(missing, status):
    variance = pd.DataFrame({"a": pd.Series([.1, missing], dtype=object),
                             "b": pd.Series([.2, 0.], dtype=object)})
    batch = ObservationBatch(pd.DataFrame({"x": [1., 2.]}),
        pd.DataFrame({"a": [3., None], "b": [4., None]}), Yvar=variance,
        status=pd.Series(["success", status]), ids=("measured", "not-successful"))
    actual = batch.Yvar.a.iloc[1]
    if missing is None or missing is pd.NA:
        assert actual is missing
    else:
        assert isinstance(actual, float) and np.isnan(actual)
    pd.testing.assert_frame_equal(batch.Yvar, variance)
    assert batch.successful().Yvar.values.tolist() == [[.1, .2]]


@pytest.mark.parametrize("status", ["success", "failed", "warning"])
@pytest.mark.parametrize("value", [-.1, np.inf, -np.inf, True, np.bool_(False), "0.1", 1 + 0j, [0.]])
def test_multioutput_rejects_every_invalid_recorded_variance(value, status):
    variance = pd.DataFrame({"a": pd.Series([value], dtype=object), "b": [.1]})
    with pytest.raises(ValueError, match="Yvar"):
        ObservationBatch(pd.DataFrame({"x": [1.]}), pd.DataFrame({"a": [3.], "b": [4.]}),
                         Yvar=variance, status=pd.Series([status]))


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
@pytest.mark.parametrize("status", [None, pd.Series(["success"])])
def test_multioutput_successful_variance_cannot_be_missing(missing, status):
    variance = pd.DataFrame({"a": pd.Series([missing], dtype=object), "b": [.1]})
    with pytest.raises(ValueError, match="Yvar"):
        ObservationBatch(pd.DataFrame({"x": [1.]}), pd.DataFrame({"a": [3.], "b": [4.]}),
                         Yvar=variance, status=status)


@pytest.mark.parametrize("columns", [["a"], ["a", "extra"], ["a", "b", "extra"], ["a", "a"]])
def test_multioutput_variance_names_must_be_exact_and_unique(columns):
    with pytest.raises(ValueError, match="columns"):
        ObservationBatch(pd.DataFrame({"x": [1.]}), pd.DataFrame({"a": [3.], "b": [4.]}),
                         Yvar=pd.DataFrame([[.1] * len(columns)], columns=columns))


@pytest.mark.parametrize("status", ["success", "failed", "warning"])
@pytest.mark.parametrize("variance", [pd.Series([.1], dtype=object), pd.Series([None], dtype=object),
                                     pd.Series([np.nan]), pd.Series([pd.NA], dtype="Float64")])
def test_scalar_variance_validation_does_not_adopt_multioutput_relaxations(status, variance):
    with pytest.raises((ValueError, TypeError)):
        ObservationBatch(pd.DataFrame({"x": [1.]}), pd.DataFrame({"a": [3.]}),
                         Yvar=pd.DataFrame({"a": variance}), status=pd.Series([status]))


def test_validated_subset_projects_and_reorders_without_reinterpreting_scalar_input():
    batch = ObservationBatch(pd.DataFrame({"x": [1., 2., 3.]}),
        pd.DataFrame({"a": [3., None, None], "b": [4., None, None]}),
        Yvar=pd.DataFrame({"a": pd.Series([.1, None, pd.NA], dtype=object),
                           "b": pd.Series([.2, np.nan, None], dtype=object)}),
        status=pd.Series(["success", "failed", "warning"]), ids=("s", "f", "w"))
    selected = batch._subset(rows=[2, 0, 1], objectives=["a"])
    assert selected.ids == ("w", "s", "f")
    assert selected.status.tolist() == ["warning", "success", "failed"]
    assert selected.Yvar.a.iloc[0] is pd.NA and selected.Yvar.a.iloc[2] is None
    assert selected.successful().ids == ("s",)
    assert selected.successful().Yvar.a.tolist() == [.1]
    assert selected._Yvar is not batch._Yvar
    detached_variance = selected.Yvar
    detached_variance.iloc[0, 0] = 99.
    assert batch.Yvar.a.iloc[2] is pd.NA
    with pytest.raises(ValueError, match="Yvar"):
        ObservationBatch(selected.X, selected.Y, Yvar=selected.Yvar, status=selected.status, ids=selected.ids)
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        selected._ids = ("new",)


@pytest.mark.parametrize("selection", [
    {"rows": [0, 0]}, {"rows": [3]}, {"rows": [-1]}, {"rows": [True]},
    {"objectives": ["a", "a"]}, {"objectives": ["unknown"]},
])
def test_validated_subset_rejects_non_subset_selection(selection):
    batch = ObservationBatch(pd.DataFrame({"x": [1., 2.]}),
                             pd.DataFrame({"a": [3., 4.], "b": [5., 6.]}))
    with pytest.raises((ValueError, IndexError, KeyError)):
        batch._subset(**selection)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"X": pd.Series([1]), "Y": pd.DataFrame({"y": [1.0]})}, "X must"),
        ({"X": pd.DataFrame({"x": [1]}), "Y": pd.Series([1.0])}, "Y must"),
        (
            {"X": pd.DataFrame([[1, 2]], columns=["x", "x"]), "Y": pd.DataFrame({"y": [1.0]})},
            "duplicate",
        ),
        (
            {"X": pd.DataFrame({"x": [1]}), "Y": pd.DataFrame({"y": [1.0]}), "status": pd.Series(["unknown"])},
            "status",
        ),
        (
            {"X": pd.DataFrame({"x": [1]}), "Y": pd.DataFrame({"y": [1.0]}), "ids": ("",)},
            "non-empty",
        ),
    ],
)
def test_observation_batch_rejects_malformed_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ObservationBatch(**kwargs)


def test_row_counts_and_pending_ids_must_align_with_x():
    with pytest.raises(ValueError, match="row count"):
        ObservationBatch(X=pd.DataFrame({"x": [1]}), Y=pd.DataFrame({"y": [1.0, 2.0]}))

    with pytest.raises(ValueError, match="count"):
        PendingBatch(ids=("p1", "p2"), X=pd.DataFrame({"x": [1]}))
