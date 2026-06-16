"""
Regression test for the D-bug class of errors.

Verifies that `flip_for_minimize` translates physical-space intuition
into the correct GP-frame direction across all four combinations of
(effect × maximize).
"""
import pytest

from expdoe_dk.knowledge._frame import (
    flip_for_minimize,
    negate_for_minimize,
    internal_to_physical,
)


@pytest.mark.parametrize(
    "effect, maximize, expected",
    [
        ("increases_objective", True,  "decreasing"),   # The D-bug case.
        ("decreases_objective", True,  "increasing"),
        ("increases_objective", False, "increasing"),
        ("decreases_objective", False, "decreasing"),
    ],
)
def test_flip_for_minimize_truth_table(effect, maximize, expected):
    assert flip_for_minimize(effect, maximize=maximize) == expected


def test_flip_rejects_unknown_effect():
    with pytest.raises(ValueError, match="Unknown effect"):
        flip_for_minimize("increases", maximize=True)


def test_negate_round_trip():
    # maximize=True: internal = -user; back-and-forth must be identity.
    user = 0.85
    internal = negate_for_minimize(user, maximize=True)
    assert internal == pytest.approx(-0.85)
    back = internal_to_physical(internal, maximize=True)
    assert back == pytest.approx(0.85)

    # maximize=False: identity in both directions.
    assert negate_for_minimize(0.5, maximize=False) == 0.5
    assert internal_to_physical(0.5, maximize=False) == 0.5


def test_negate_handles_lists():
    user_list = [1.0, 2.0, 3.0]
    out = negate_for_minimize(user_list, maximize=True)
    assert out == [-1.0, -2.0, -3.0]
