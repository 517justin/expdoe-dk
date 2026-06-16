"""
v0.2 — Empirical validator tests.

Covers:
  - check_monotone_assumption with positive-truth data → no violation
  - check_monotone_assumption with reversed-truth data → violation
  - below min_observations → returns None
  - check_shape_prior_fit detects unrelated frozen shape
  - Knowledge.drop_monotone removes the right items
  - Campaign emits MonotoneViolationWarning after K BO trials, then once
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import (
    Campaign,
    Knowledge,
    LinearConstraint,
    Parameter,
    Space,
)
from expdoe_dk.knowledge import (
    MonotoneViolationWarning,
    ShapePriorMismatchWarning,
    check_monotone_assumption,
    check_shape_prior_fit,
)


# --------------------------------------------------------------------- #
# Pure-function tests for the validators
# --------------------------------------------------------------------- #
def _make_xy(n: int, slope: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    y = slope * x + rng.normal(0.0, 0.05, size=n)
    X = np.stack([x, rng.uniform(size=n)], axis=1)
    return X, y


def test_monotone_check_passes_when_sign_matches():
    X, y = _make_xy(n=30, slope=+1.0)
    res = check_monotone_assumption(
        X_user_units=X,
        y_user=y,
        param_index=0,
        param_name="x",
        declared_effect="increases_objective",
        min_observations=10,
    )
    assert res is not None
    assert res.sign_matches
    assert not res.violation


def test_monotone_check_violation_when_sign_reversed():
    X, y = _make_xy(n=30, slope=-1.0)  # y decreases with x
    res = check_monotone_assumption(
        X_user_units=X,
        y_user=y,
        param_index=0,
        param_name="x",
        declared_effect="increases_objective",  # user thought it rises
        min_observations=10,
    )
    assert res is not None
    assert not res.sign_matches
    assert res.violation
    assert res.observed_spearman < 0
    # And the rendered warning mentions the param + a remediation hint.
    w = res.to_warning()
    assert isinstance(w, MonotoneViolationWarning)
    msg = str(w)
    assert "x" in msg
    assert "drop_monotone" in msg


def test_monotone_check_below_min_obs_returns_none():
    X, y = _make_xy(n=5, slope=+1.0)
    assert check_monotone_assumption(
        X, y, param_index=0, param_name="x",
        declared_effect="increases_objective",
        min_observations=10,
    ) is None


def test_monotone_check_noise_below_threshold_no_violation():
    # Tiny correlation: even if signs disagree, it's noise — no warn.
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 2))
    y = rng.normal(size=30) * 0.01 - X[:, 0] * 0.001  # nearly flat
    res = check_monotone_assumption(
        X, y, param_index=0, param_name="x",
        declared_effect="increases_objective",
        min_observations=10,
    )
    assert res is not None
    assert not res.violation


def test_shape_check_high_correlation_no_violation():
    # Mean function predictions strongly tracking the data.
    n = 30
    rng = np.random.default_rng(0)
    m = np.linspace(0, 1, n)
    y_norm = m + rng.normal(0.0, 0.05, size=n)
    res = check_shape_prior_fit(
        mean_predictions=m, y_norm=y_norm,
        kind="arrhenius", param="T",
        min_observations=10,
    )
    assert res is not None
    assert abs(res.correlation) > 0.8
    assert not res.violation


def test_shape_check_low_correlation_violation():
    n = 30
    rng = np.random.default_rng(0)
    m = np.linspace(0, 1, n)
    y_norm = rng.normal(size=n)  # totally unrelated
    res = check_shape_prior_fit(
        mean_predictions=m, y_norm=y_norm,
        kind="quadratic_peak", param="conc_A",
        min_observations=10,
    )
    assert res is not None
    assert res.violation
    w = res.to_warning()
    assert isinstance(w, ShapePriorMismatchWarning)
    assert "conc_A" in str(w)


# --------------------------------------------------------------------- #
# Knowledge.drop_monotone behavior
# --------------------------------------------------------------------- #
def test_knowledge_drop_monotone_removes_single_param():
    k = (Knowledge()
         .with_monotone("T", effect="increases_objective")
         .with_monotone("time", effect="increases_objective")
         .with_quadratic_peak("pH", center=7.0))
    k.drop_monotone("T")
    monos = k.items_of("monotone")
    assert {m.param for m in monos} == {"time"}
    assert k.has_kind("quadratic_peak")  # untouched


def test_knowledge_drop_monotone_removes_all_when_no_arg():
    k = (Knowledge()
         .with_monotone("T", effect="increases_objective")
         .with_monotone("time", effect="increases_objective"))
    k.drop_monotone()
    assert not k.items_of("monotone")


# --------------------------------------------------------------------- #
# Campaign integration: warning is emitted after K observations
# --------------------------------------------------------------------- #
def _toy_space():
    return Space(
        params=[
            Parameter("x0", bounds=(0.0, 1.0)),
            Parameter("x1", bounds=(0.0, 1.0)),
        ],
        objectives="y",
        maximize=True,
    )


def test_campaign_emits_monotone_violation_after_min_obs():
    space = _toy_space()
    # User says "x0 increases yield" but truth is the opposite.
    knowledge = Knowledge().with_monotone(
        "x0", effect="increases_objective"
    )
    campaign = Campaign(
        space, knowledge, seed=0,
        validate=True, validation_interval=1, validation_min_obs=10,
    )

    rng = np.random.default_rng(0)
    n = 15
    X = pd.DataFrame(rng.uniform(size=(n, 2)), columns=["x0", "x1"])
    # y decreases with x0 → reverses user's "increases_objective"
    y = -X["x0"].to_numpy() + rng.normal(0.0, 0.02, size=n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        campaign.tell(X, y)
    monotone_warnings = [
        w for w in caught
        if issubclass(w.category, MonotoneViolationWarning)
    ]
    assert len(monotone_warnings) == 1
    assert "x0" in str(monotone_warnings[0].message)


def test_campaign_does_not_warn_when_assumption_holds():
    space = _toy_space()
    knowledge = Knowledge().with_monotone(
        "x0", effect="increases_objective"
    )
    campaign = Campaign(
        space, knowledge, seed=0,
        validate=True, validation_interval=1, validation_min_obs=10,
    )

    rng = np.random.default_rng(0)
    n = 15
    X = pd.DataFrame(rng.uniform(size=(n, 2)), columns=["x0", "x1"])
    y = X["x0"].to_numpy() + rng.normal(0.0, 0.02, size=n)  # matches

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        campaign.tell(X, y)
    monotone_warnings = [
        w for w in caught
        if issubclass(w.category, MonotoneViolationWarning)
    ]
    assert monotone_warnings == []


def test_campaign_validation_can_be_disabled():
    space = _toy_space()
    knowledge = Knowledge().with_monotone("x0", effect="increases_objective")
    campaign = Campaign(
        space, knowledge, seed=0,
        validate=False,
        validation_interval=1, validation_min_obs=10,
    )

    rng = np.random.default_rng(0)
    n = 15
    X = pd.DataFrame(rng.uniform(size=(n, 2)), columns=["x0", "x1"])
    y = -X["x0"].to_numpy() + rng.normal(0.0, 0.02, size=n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        campaign.tell(X, y)
    monotone_warnings = [
        w for w in caught
        if issubclass(w.category, MonotoneViolationWarning)
    ]
    assert monotone_warnings == []


def test_campaign_warns_only_once_per_param():
    """Repeated tells under continued violation shouldn't re-spam the user."""
    space = _toy_space()
    knowledge = Knowledge().with_monotone("x0", effect="increases_objective")
    campaign = Campaign(
        space, knowledge, seed=0,
        validate=True, validation_interval=1, validation_min_obs=10,
    )

    rng = np.random.default_rng(0)
    X1 = pd.DataFrame(rng.uniform(size=(15, 2)), columns=["x0", "x1"])
    y1 = -X1["x0"].to_numpy()
    X2 = pd.DataFrame(rng.uniform(size=(5, 2)), columns=["x0", "x1"])
    y2 = -X2["x0"].to_numpy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        campaign.tell(X1, y1)
        campaign.tell(X2, y2)
    monotone_warnings = [
        w for w in caught
        if issubclass(w.category, MonotoneViolationWarning)
    ]
    assert len(monotone_warnings) == 1


def test_campaign_below_min_obs_does_not_warn():
    space = _toy_space()
    knowledge = Knowledge().with_monotone("x0", effect="increases_objective")
    campaign = Campaign(
        space, knowledge, seed=0,
        validate=True, validation_interval=1, validation_min_obs=10,
    )
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(size=(6, 2)), columns=["x0", "x1"])
    y = -X["x0"].to_numpy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        campaign.tell(X, y)
    monotone_warnings = [
        w for w in caught
        if issubclass(w.category, MonotoneViolationWarning)
    ]
    assert monotone_warnings == []
