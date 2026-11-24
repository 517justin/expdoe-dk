"""R2: exact physical data must survive batching before any math boundary."""
import csv
import io
import math

import pandas as pd
import pytest

from expdoe_dk import Campaign, ObservationBatch, Parameter, Space, candidate_keys

B = 2**60


def mixed_campaign():
    space = Space([Parameter("x", kind="integer", bounds=(B, B + 1024))],
                  objectives=["gain", "cost"])
    campaign = Campaign(space, validate=False)
    for dtype, value in [("int64", B + 1), ("uint64", B + 3)]:
        x = pd.DataFrame({"x": pd.Series([value], dtype=dtype)})
        # Deliberately reverse names; alignment must remain by name.
        y = pd.DataFrame({"cost": [1.], "gain": pd.Series([value], dtype=dtype)})
        campaign.tell_batch(ObservationBatch(x, y, Yvar=y[["gain", "cost"]]))
    return campaign


@pytest.mark.parametrize("boundary", ["X", "Y", "Yvar", "history", "csv", "pareto", "keys"])
@pytest.mark.parametrize("restored", [False, True])
def test_mixed_integer_batches_keep_exact_values(boundary, restored, tmp_path):
    campaign = mixed_campaign()
    if restored:
        path = tmp_path / "mixed.json"
        campaign.save_checkpoint(path)
        campaign = Campaign.load_checkpoint(path)
    combined = campaign._combined_observations()
    assert combined.ids == ("unidentified:0:0", "unidentified:1:0")
    assert combined.status.tolist() == ["success", "success"]
    assert list(combined.Y.columns) == list(combined.Yvar.columns) == ["gain", "cost"]
    if boundary in {"X", "Y", "Yvar"}:
        frame = getattr(combined, boundary)
        name = "x" if boundary == "X" else "gain"
        assert [int(value) for value in frame[name]] == [B + 1, B + 3]
        assert frame[name].dtype == object
        if boundary != "X":
            assert str(frame.cost.dtype) == "float64"
    elif boundary == "history":
        history = campaign.history_df()
        assert [int(value) for value in history.x] == [B + 1, B + 3]
        assert [int(value) for value in history.gain] == [B + 1, B + 3]
    elif boundary == "csv":
        rows = list(csv.DictReader(io.StringIO(campaign.history_df().to_csv(index=False))))
        assert [row["x"] for row in rows] == [str(B + 1), str(B + 3)]
        assert [row["gain"] for row in rows] == [str(B + 1), str(B + 3)]
    elif boundary == "pareto":
        rows = campaign.finalize().pareto_rows
        assert rows
        assert all(row["x"] in {B + 1, B + 3} and row["gain"] == row["x"] for row in rows)
    else:
        keys = candidate_keys(combined.X, campaign.space)
        assert len(set(keys)) == 2
        assert keys == set().union(*(candidate_keys(b.X, campaign.space) for b in campaign._observation_batches))


@pytest.mark.parametrize("marker", [None, float("nan"), pd.NA])
@pytest.mark.parametrize("restored", [False, True])
def test_cross_batch_unsuccessful_markers_remain_distinct(marker, restored):
    campaign = mixed_campaign()
    failed = pd.DataFrame({name: pd.Series([marker], dtype=object) for name in ["gain", "cost"]})
    campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [B + 5]}), failed,
        Yvar=failed, status=pd.Series(["failed"]), ids=["failed-trial"]))
    if restored:
        campaign = Campaign.from_state(campaign.to_state())
    combined = campaign._combined_observations()
    assert combined.ids[-1] == "failed-trial" and combined.status.iloc[-1] == "failed"
    for frame in [combined.Y, combined.Yvar, campaign.history_df()]:
        for name in ["gain", "cost"]:
            value = frame[name].iloc[-1]
            if marker is None or marker is pd.NA:
                assert value is marker
            else:
                assert isinstance(value, float) and math.isnan(value)
        assert [int(value) for value in frame.gain.iloc[:2]] == [B + 1, B + 3]


@pytest.mark.parametrize("dtypes,values,expected", [
    (("int64", "int64"), (B + 1, B + 3), "int64"),
    (("uint64", "uint64"), (B + 1, B + 3), "uint64"),
    (("float64", "float64"), (1.25, 2.5), "float64"),
    (("int64", "uint64"), (1, 3), "float64"),
])
def test_lossless_common_dtypes_are_not_forced_to_object(dtypes, values, expected):
    campaign = Campaign(Space([Parameter("x", bounds=(0., float(2 * B)))],
                              objectives=["gain", "cost"]), validate=False)
    for dtype, value in zip(dtypes, values):
        frame = pd.DataFrame({"x": pd.Series([value], dtype=dtype)})
        y = pd.DataFrame({name: pd.Series([value], dtype=dtype) for name in ["gain", "cost"]})
        campaign.tell_batch(ObservationBatch(frame, y, Yvar=y))
    batch = campaign._combined_observations()
    for frame, name in [(batch.X, "x"), (batch.Y, "gain"), (batch.Yvar, "gain"), (campaign.history_df(), "x")]:
        assert str(frame[name].dtype) == expected


@pytest.mark.parametrize("dtypes", [("Int64", "UInt64"), ("int64", "uint64")])
def test_restart_append_preserves_exact_physical_prefix(dtypes):
    campaign = Campaign(Space([Parameter("x", kind="integer", bounds=(B, B + 1024))],
                              objectives=["gain", "cost"]), validate=False)
    for index, dtype in enumerate([*dtypes, dtypes[0]]):
        value = B + 2 * index + 1
        x = pd.DataFrame({"x": pd.Series([value], dtype=dtype)})
        y = pd.DataFrame({"gain": pd.Series([value], dtype=dtype), "cost": [1.]})
        campaign.tell_batch(ObservationBatch(x, y, Yvar=y, ids=[f"trial-{index}"]))
        campaign = Campaign.from_state(campaign.to_state())
    batch = campaign._combined_observations()
    assert batch.ids == ("trial-0", "trial-1", "trial-2")
    for frame, name in [(batch.X, "x"), (batch.Y, "gain"), (batch.Yvar, "gain"), (campaign.history_df(), "x")]:
        assert [int(value) for value in frame[name]] == [B + 1, B + 3, B + 5]
    assert len(candidate_keys(batch.X, campaign.space)) == 3


def test_failed_boolean_outcome_does_not_turn_into_integer():
    campaign = Campaign(Space([Parameter("x", bounds=(0., 1.))], objectives=["gain", "cost"]), validate=False)
    for value, status in [(1, "success"), (True, "failed")]:
        campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [.2]}),
            pd.DataFrame({"gain": [value], "cost": [value]}), status=pd.Series([status])))
    for current in [campaign, Campaign.from_state(campaign.to_state())]:
        for frame in [current._combined_observations().Y, current.history_df()]:
            assert type(frame.gain.iloc[0]) is int
            assert frame.gain.iloc[1] is True


def test_absent_failed_variance_remains_unknown_none():
    campaign = mixed_campaign()
    campaign.tell_batch(ObservationBatch(pd.DataFrame({"x": [B + 5]}),
        pd.DataFrame({"gain": [None], "cost": [None]}), status=pd.Series(["failed"])))
    for current in [campaign, Campaign.from_state(campaign.to_state())]:
        variance = current._combined_observations().Yvar
        assert variance.gain.iloc[-1] is None and variance.cost.iloc[-1] is None
        assert [int(value) for value in variance.gain.iloc[:2]] == [B + 1, B + 3]
