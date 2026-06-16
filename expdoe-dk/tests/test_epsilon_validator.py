"""
Exp-14 ε-vs-lengthscale conflict rule:
    ε ≥ 0.3 × prior_lengthscale_mode

Verify that:
  - Combining `with_monotone(epsilon=0.02)` with a strong GP prior raises.
  - `epsilon="auto"` resolves to a safe value.
  - `epsilon` set above the threshold passes.
"""
import pytest

from expdoe_dk import Knowledge
from expdoe_dk.knowledge import EpsilonConflictError, GP_PRIOR_PRESETS
from expdoe_dk.knowledge.monotone import epsilon_from_prior


def _ls_mode(preset_name: str) -> float:
    a, b = GP_PRIOR_PRESETS[preset_name]["ls"]
    return (a - 1.0) / b


def test_strong_prior_with_tiny_epsilon_raises():
    k = (Knowledge()
         .with_monotone("T", effect="increases_objective", epsilon=0.02)
         .with_gp_prior(lengthscale="strong"))
    with pytest.raises(EpsilonConflictError):
        k.validate()


def test_medium_prior_with_tiny_epsilon_raises():
    k = (Knowledge()
         .with_monotone("T", effect="increases_objective", epsilon=0.02)
         .with_gp_prior(lengthscale="medium"))
    with pytest.raises(EpsilonConflictError):
        k.validate()


def test_auto_epsilon_passes_with_strong_prior():
    k = (Knowledge()
         .with_monotone("T", effect="increases_objective", epsilon="auto")
         .with_gp_prior(lengthscale="strong"))
    k.validate()  # must not raise
    m = k.items_of("monotone")[0]
    resolved = k.resolve_epsilon(m)
    assert resolved >= 0.3 * _ls_mode("strong")


def test_auto_epsilon_passes_with_medium_prior():
    k = (Knowledge()
         .with_monotone("T", effect="increases_objective", epsilon="auto")
         .with_gp_prior(lengthscale="medium"))
    k.validate()
    m = k.items_of("monotone")[0]
    resolved = k.resolve_epsilon(m)
    assert resolved >= 0.3 * _ls_mode("medium")


def test_no_gp_prior_uses_minimum_epsilon():
    """Without a GP prior, ε defaults to the minimum of 0.05."""
    assert epsilon_from_prior(None) == 0.05


def test_explicit_safe_epsilon_passes():
    safe = max(0.3 * _ls_mode("medium"), 0.05) + 0.01
    k = (Knowledge()
         .with_monotone("T", effect="increases_objective", epsilon=safe)
         .with_gp_prior(lengthscale="medium"))
    k.validate()


def test_monotone_alone_no_validation_error():
    """Monotone without GP prior must always validate."""
    k = Knowledge().with_monotone("T", effect="increases_objective", epsilon=0.02)
    k.validate()
