"""
Physical ↔ Internal frame translator.

This is THE single chokepoint that prevents the D-bug class of errors
(Exp-10 in AGENT_KNOWLEDGE.md):

> The user writes `{"T": "increases_objective"}` thinking in physical space
> (raising T raises yield). BO internally minimizes Y = −yield, so in the
> GP's normalized Y_norm space, T actually DECREASES Y_norm. A naive
> implementation that takes the user's string and feeds it straight to
> `MonotonicGPWithDerivatives` produces virtual points pulling the GP in
> the wrong direction → 13× worse in 6D.

This module exposes one function:

    flip_for_minimize(effect: PhysicalEffect, maximize: bool) -> InternalDirection

Everything else in `knowledge/` and `bo/` MUST go through this function
when translating a Knowledge spec into the GP-frame virtual points.

A regression test in `tests/test_frame_translation.py` reproduces the 6D
D_correct scenario through the API and asserts the right outcome.
"""
from __future__ import annotations

from typing import Literal

PhysicalEffect = Literal["increases_objective", "decreases_objective"]
InternalDirection = Literal["increasing", "decreasing"]


def flip_for_minimize(
    effect: PhysicalEffect,
    maximize: bool,
) -> InternalDirection:
    """
    Convert a user-facing PhysicalEffect into the direction used inside
    the GP frame (where the internal scalar is always minimized).

    Truth table:
        maximize=True, effect=increases_objective → "decreasing"
            (high objective in physical = low Y_internal)
        maximize=True, effect=decreases_objective → "increasing"
        maximize=False, effect=increases_objective → "increasing"
        maximize=False, effect=decreases_objective → "decreasing"

    The "maximize=False" case mirrors what happens when the user already
    thinks in "lower is better" (e.g. impurity, cost).

    Parameters
    ----------
    effect : "increases_objective" | "decreases_objective"
        How the parameter affects the user's objective (in physical units).
    maximize : bool
        The user's declared direction for this objective.

    Returns
    -------
    "increasing" | "decreasing" — the direction the GP virtual points must
    use in Y_norm space so the GP fit goes the right way.
    """
    if effect not in ("increases_objective", "decreases_objective"):
        raise ValueError(
            f"Unknown effect: {effect!r}. Use 'increases_objective' or "
            f"'decreases_objective'."
        )

    # Internally we minimize Y_internal = -Y_user when maximize=True,
    # Y_internal = Y_user when maximize=False.
    # So "increases_objective" maps to "decreasing Y_internal" iff maximize.
    if maximize:
        return "decreasing" if effect == "increases_objective" else "increasing"
    else:
        return "increasing" if effect == "increases_objective" else "decreasing"


def negate_for_minimize(y: float | list[float], maximize: bool) -> float | list[float]:
    """
    Convert user-facing objective values to internally-minimized values.

    If maximize=True, returns -y so the BO minimizer treats high y as good.
    If maximize=False, returns y unchanged.
    """
    if isinstance(y, (list, tuple)):
        return [-v if maximize else v for v in y]
    return -y if maximize else y


def physical_to_internal_best(
    user_best: float, maximize: bool
) -> float:
    """Same as negate_for_minimize for a scalar; kept named for clarity in calls."""
    return -user_best if maximize else user_best


def internal_to_physical(
    internal_y: float | list[float], maximize: bool
) -> float | list[float]:
    """Inverse of negate_for_minimize: convert internal min-frame back to user-frame."""
    if isinstance(internal_y, (list, tuple)):
        return [-v if maximize else v for v in internal_y]
    return -internal_y if maximize else internal_y
