"""
v0.3 — Epsilon auto-rescue tests.

When ``Knowledge.validate(auto_rescue=True)`` (or ``Campaign(auto_rescue=True)``,
default) sees the Exp-14 conflict (small ε + informative GP prior), it should:

  - replace the offending ``with_monotone`` item with one whose ``epsilon``
    is bumped to the safe value (``≥ 0.3 × prior_lengthscale_mode``)
  - emit ``EpsilonAutoRescueNotice`` exactly once per rescued item
  - NOT raise ``EpsilonConflictError``
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import Campaign, Knowledge, Parameter, Space
from expdoe_dk.knowledge import (
    EpsilonAutoRescueNotice,
    EpsilonConflictError,
    GP_PRIOR_PRESETS,
)


def _ls_mode(preset: str) -> float:
    a, b = GP_PRIOR_PRESETS[preset]["ls"]
    return (a - 1.0) / b


# ----------------------------------------------------------------------- #
# Knowledge.validate(auto_rescue=True)
# ----------------------------------------------------------------------- #
def test_validate_auto_rescue_bumps_epsilon_and_warns():
    k = (
        Knowledge()
        .with_monotone("T", effect="increases_objective", epsilon=0.02)
        .with_gp_prior(lengthscale="strong")
    )
    min_eps = 0.3 * _ls_mode("strong")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        k.validate(auto_rescue=True)
    notices = [w for w in caught if issubclass(w.category, EpsilonAutoRescueNotice)]
    assert len(notices) == 1, notices

    mono = k.items_of("monotone")[0]
    assert mono.epsilon != "auto"
    assert float(mono.epsilon) >= min_eps - 1e-9


def test_validate_auto_rescue_does_not_raise():
    k = (
        Knowledge()
        .with_monotone("T", effect="increases_objective", epsilon=0.01)
        .with_gp_prior(lengthscale="medium")
    )
    # No exception expected.
    k.validate(auto_rescue=True)


def test_validate_default_still_raises():
    """Default auto_rescue=False preserves v0.1 strict behaviour."""
    k = (
        Knowledge()
        .with_monotone("T", effect="increases_objective", epsilon=0.02)
        .with_gp_prior(lengthscale="medium")
    )
    with pytest.raises(EpsilonConflictError):
        k.validate()  # auto_rescue=False
    with pytest.raises(EpsilonConflictError):
        k.validate(auto_rescue=False)


def test_validate_no_conflict_passes_silently():
    """auto_rescue=True should be a no-op when there's no conflict."""
    safe_eps = max(0.3 * _ls_mode("medium"), 0.05) + 0.05
    k = (
        Knowledge()
        .with_monotone("T", effect="increases_objective", epsilon=safe_eps)
        .with_gp_prior(lengthscale="medium")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        k.validate(auto_rescue=True)
    notices = [w for w in caught if issubclass(w.category, EpsilonAutoRescueNotice)]
    assert notices == []


def test_validate_auto_rescue_is_idempotent():
    """Calling validate(auto_rescue=True) twice does not double-warn."""
    k = (
        Knowledge()
        .with_monotone("T", effect="increases_objective", epsilon=0.02)
        .with_gp_prior(lengthscale="strong")
    )
    with warnings.catch_warnings(record=True) as caught_first:
        warnings.simplefilter("always")
        k.validate(auto_rescue=True)
    # After the first rescue, eps is now safe; second call should be silent.
    with warnings.catch_warnings(record=True) as caught_second:
        warnings.simplefilter("always")
        k.validate(auto_rescue=True)
    first = [w for w in caught_first if issubclass(w.category, EpsilonAutoRescueNotice)]
    second = [w for w in caught_second if issubclass(w.category, EpsilonAutoRescueNotice)]
    assert len(first) == 1
    assert len(second) == 0


def test_validate_auto_rescue_handles_multiple_monotones():
    k = (
        Knowledge()
        .with_monotone("T", effect="increases_objective", epsilon=0.02)
        .with_monotone("time", effect="increases_objective", epsilon=0.01)
        .with_gp_prior(lengthscale="strong")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        k.validate(auto_rescue=True)
    notices = [w for w in caught if issubclass(w.category, EpsilonAutoRescueNotice)]
    assert len(notices) == 2

    min_eps = 0.3 * _ls_mode("strong")
    for mono in k.items_of("monotone"):
        assert float(mono.epsilon) >= min_eps - 1e-9


# ----------------------------------------------------------------------- #
# Campaign integration: default auto_rescue=True
# ----------------------------------------------------------------------- #
def _toy_space():
    return Space(
        params=[
            Parameter("x0", bounds=(0.0, 1.0)),
            Parameter("x1", bounds=(0.0, 1.0)),
        ],
        objectives="y",
        maximize=True,
    )


def test_campaign_default_auto_rescues_eps_conflict():
    """A configuration that would have raised in v0.1 now silently rescues."""
    space = _toy_space()
    knowledge = (
        Knowledge()
        .with_monotone("x0", effect="increases_objective", epsilon=0.02)
        .with_gp_prior(lengthscale="strong")
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        campaign = Campaign(space, knowledge, seed=0)
    notices = [w for w in caught if issubclass(w.category, EpsilonAutoRescueNotice)]
    assert len(notices) == 1
    # Campaign object built successfully.
    assert campaign.knowledge.has_kind("monotone")
    min_eps = 0.3 * _ls_mode("strong")
    mono = campaign.knowledge.items_of("monotone")[0]
    assert float(mono.epsilon) >= min_eps - 1e-9


def test_campaign_auto_rescue_off_raises():
    space = _toy_space()
    knowledge = (
        Knowledge()
        .with_monotone("x0", effect="increases_objective", epsilon=0.02)
        .with_gp_prior(lengthscale="strong")
    )
    with pytest.raises(EpsilonConflictError):
        Campaign(space, knowledge, seed=0, auto_rescue=False)


def test_campaign_auto_rescue_does_not_affect_safe_configs():
    """Safe configs build silently regardless of auto_rescue."""
    space = _toy_space()
    knowledge = (
        Knowledge()
        .with_monotone("x0", effect="increases_objective", epsilon="auto")
        .with_gp_prior(lengthscale="medium")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Campaign(space, knowledge, seed=0)
        Campaign(space, knowledge, seed=0, auto_rescue=False)
    notices = [w for w in caught if issubclass(w.category, EpsilonAutoRescueNotice)]
    assert notices == []
