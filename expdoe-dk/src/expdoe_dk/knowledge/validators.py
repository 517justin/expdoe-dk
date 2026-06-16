"""
Empirical validators that run on the campaign's accumulated data.

Each validator returns either ``None`` (silent — assumption holds) or a
``Warning`` instance ready to be emitted via ``warnings.warn(...)``. The
Campaign decides when/how often to invoke them (every K BO iters), and
whether to surface them as warnings, log lines, or both.

Two validators ship in v0.2:

  - ``check_monotone_assumption``  → Spearman rank correlation between a
    parameter and the user-frame objective; flags sign reversals.

  - ``check_shape_prior_fit``      → Pearson correlation between a frozen
    mean function's prediction and the normalised objective; flags shape
    priors that are unrelated to the data.

References:
  - AGENT_KNOWLEDGE.md Exp-10 (D-bug discovered via Spearman in 6D)
  - AGENT_KNOWLEDGE.md Exp-7 (learnable mean MLE absorption motivates frozen)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

# --------------------------------------------------------------------- #
# Warning classes (importable from `expdoe_dk.knowledge`)
# --------------------------------------------------------------------- #


class MonotoneViolationWarning(UserWarning):
    """Observed Spearman correlation contradicts the declared monotone effect."""


class ShapePriorMismatchWarning(UserWarning):
    """Frozen mean function shape correlates poorly with the observed data."""


# --------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class MonotoneCheckResult:
    param: str
    declared_effect: str
    observed_spearman: float
    p_value: float
    n_observations: int
    expected_sign: int  # +1 / -1
    sign_matches: bool
    violation: bool  # True if we should warn

    def to_warning(self) -> MonotoneViolationWarning:
        direction = ("increases" if self.expected_sign > 0 else "decreases")
        observed_dir = (
            "increases" if self.observed_spearman > 0 else "decreases"
        )
        msg = (
            f"with_monotone(param={self.param!r}, "
            f"effect={self.declared_effect!r}): expected the objective to "
            f"{direction} with {self.param}, but observed Spearman "
            f"correlation = {self.observed_spearman:+.3f} "
            f"(p={self.p_value:.3g}, n={self.n_observations}) — the "
            f"objective actually {observed_dir} with {self.param}.\n"
            f"  → consider Knowledge.drop_monotone({self.param!r}) and "
            f"falling back to with_random_augment() (Cat ②).\n"
            f"  → or flip effect to the opposite direction if your physical "
            f"intuition needs revisiting."
        )
        return MonotoneViolationWarning(msg)


@dataclass(frozen=True)
class ShapeCheckResult:
    kind: str  # "arrhenius" | "quadratic_peak"
    param: str
    correlation: float
    n_observations: int
    violation: bool

    def to_warning(self) -> ShapePriorMismatchWarning:
        msg = (
            f"with_{self.kind}(param={self.param!r}): the frozen mean "
            f"function's predicted shape correlates only "
            f"{self.correlation:+.3f} with observed Y (n={self.n_observations}). "
            f"The prior shape may be unrelated to the actual response.\n"
            f"  → consider removing this knowledge item and using "
            f"with_random_augment() instead (Cat ② is the safe baseline)."
        )
        return ShapePriorMismatchWarning(msg)


# --------------------------------------------------------------------- #
# Spearman correlation (no scipy dependency for this small helper)
# --------------------------------------------------------------------- #


def _rank(a: np.ndarray) -> np.ndarray:
    """Average-rank (handles ties); returns float ranks 1..n."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (correlation, two-sided p-value) using the t-distribution approx."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rx = _rank(x)
    ry = _rank(y)
    rxc = rx - rx.mean()
    ryc = ry - ry.mean()
    denom = math.sqrt(float((rxc ** 2).sum() * (ryc ** 2).sum()))
    if denom == 0:
        return 0.0, 1.0
    rho = float((rxc * ryc).sum() / denom)
    # Avoid division by zero in t-stat for perfect correlation.
    if abs(rho) >= 1.0 - 1e-12:
        return rho, 0.0
    t_stat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    # Two-sided p-value via normal-approx (good for n >= 10).
    # Use the survival function of the standard normal as a simple proxy.
    # For better accuracy at small n, scipy.stats.t would be used, but we
    # avoid the dependency here.
    z = abs(t_stat)
    # erfc-based normal SF
    p = math.erfc(z / math.sqrt(2.0))
    return rho, p


# --------------------------------------------------------------------- #
# Public validators
# --------------------------------------------------------------------- #


def check_monotone_assumption(
    X_user_units: np.ndarray,
    y_user: np.ndarray,
    param_index: int,
    param_name: str,
    declared_effect: Literal["increases_objective", "decreases_objective"],
    *,
    min_observations: int = 10,
    p_threshold: float = 0.05,
    rho_threshold: float = 0.15,
) -> MonotoneCheckResult | None:
    """
    Empirically check whether a declared monotone effect agrees with the
    data observed so far.

    Returns ``None`` if there are too few observations to call. Otherwise a
    ``MonotoneCheckResult`` with ``violation`` set when the sign disagrees
    AND ``|rho|`` exceeds the noise threshold AND p < p_threshold.
    """
    if len(X_user_units) < min_observations:
        return None
    x_col = np.asarray(X_user_units[:, param_index], dtype=np.float64)
    y_arr = np.asarray(y_user, dtype=np.float64).flatten()
    if len(x_col) != len(y_arr):
        raise ValueError(
            f"X_user_units rows ({len(x_col)}) must match y length ({len(y_arr)})."
        )

    rho, p = _spearman_with_p(x_col, y_arr)
    expected_sign = +1 if declared_effect == "increases_objective" else -1
    sign_matches = (
        (rho > 0 and expected_sign > 0) or (rho < 0 and expected_sign < 0)
    )
    # Violation = strong, statistically significant correlation that points
    # the opposite way to the user's declaration.
    violation = (
        not sign_matches
        and abs(rho) >= rho_threshold
        and p <= p_threshold
    )
    return MonotoneCheckResult(
        param=param_name,
        declared_effect=declared_effect,
        observed_spearman=rho,
        p_value=p,
        n_observations=len(y_arr),
        expected_sign=expected_sign,
        sign_matches=sign_matches,
        violation=violation,
    )


def check_shape_prior_fit(
    mean_predictions: np.ndarray,
    y_norm: np.ndarray,
    *,
    kind: Literal["arrhenius", "quadratic_peak"],
    param: str,
    min_observations: int = 10,
    correlation_threshold: float = 0.30,
) -> ShapeCheckResult | None:
    """
    Check that a frozen mean function's predicted shape resembles the data.

    ``mean_predictions`` is m(X_unit) evaluated at the observed (unit-space)
    inputs; ``y_norm`` is the centred-and-scaled objective in the GP's
    internal frame. Pearson correlation between them ≈ 1 when the prior
    captures the structure, ≈ 0 when it doesn't.

    A weak correlation does not prove the prior is wrong — the prior may
    still help GP fit — but it suggests it isn't adding signal.
    """
    if len(mean_predictions) < min_observations:
        return None
    m = np.asarray(mean_predictions, dtype=np.float64).flatten()
    y = np.asarray(y_norm, dtype=np.float64).flatten()
    if len(m) != len(y):
        raise ValueError(
            f"mean_predictions ({len(m)}) != y_norm ({len(y)}) length."
        )
    if m.std() < 1e-9 or y.std() < 1e-9:
        return None
    corr = float(np.corrcoef(m, y)[0, 1])
    violation = abs(corr) < correlation_threshold
    return ShapeCheckResult(
        kind=kind,
        param=param,
        correlation=corr,
        n_observations=len(y),
        violation=violation,
    )


__all__ = [
    "MonotoneViolationWarning",
    "ShapePriorMismatchWarning",
    "MonotoneCheckResult",
    "ShapeCheckResult",
    "check_monotone_assumption",
    "check_shape_prior_fit",
]
