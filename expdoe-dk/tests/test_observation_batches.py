import warnings

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import ObservationBatch, PendingBatch


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
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        with pytest.raises(ValueError, match="Yvar.*numeric"):
            ObservationBatch(
                X=pd.DataFrame({"x": [0.1]}),
                Y=pd.DataFrame({"yield": [1.0]}),
                Yvar=pd.DataFrame({"yield": [1 + 100j]}),
            )


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
