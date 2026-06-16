"""
Monotonic GP via virtual derivative observations.

Implements the MonotonicGPV2 design from DOEGP Plan 2 (Exp-10):
- Virtual point pairs at (x − ε, y_center − δ/2) and (x + ε, y_center + δ/2)
  along each declared monotone dimension push the GP fit toward that
  direction in [0,1] unit input space, Y_norm output space.

Two safety rules from prior experiments are enforced here:

(P3) Per-iteration seed varies with iteration and bo_seed, so virtual
     points re-randomize each BO step (otherwise GP overfits to fixed
     virtual locations).

(P4) `delta_norm` is in Y_norm std-dev units (default 0.5 → ±0.25 σ),
     NOT hardcoded raw-Y units. This matches the GP's actual fit scale.

Exp-14 rule (lengthscale conflict avoidance):
     When combined with an informative GP lengthscale prior (Gamma(a,b)),
     ε must be ≥ 0.3 × prior_mode = 0.3 × (a-1)/b. Otherwise the GP MAP
     gets stuck between "real data wants ls≈0.4" and "virtual points want
     ls≤2ε". The Knowledge composition layer auto-tunes ε when "auto" is
     requested; this module just consumes the value.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

InternalDirection = Literal["increasing", "decreasing"]


def epsilon_from_prior(
    lengthscale_prior_mode: float | None,
    safety_margin: float = 0.3,
    minimum: float = 0.05,
) -> float:
    """
    Compute a safe ε relative to the GP lengthscale prior mode.

    For Gamma(a, b) with a > 1, the mode is (a-1)/b. With no informative
    prior pass `None` and we default to a conservative 0.05.
    """
    if lengthscale_prior_mode is None:
        return minimum
    return max(safety_margin * lengthscale_prior_mode, minimum)


class MonotonicAugmenter:
    """
    Adds derivative-encoding virtual point pairs to the GP training data.

    Operates in normalized space:
        X in [0,1]^d (unit hypercube)
        Y in Y_norm (caller-normalized to zero-mean unit-variance)

    Parameters
    ----------
    monotone_dims_internal : dict[int, "increasing" | "decreasing"]
        Maps input dim index to the direction in GP frame. Use
        `_frame.flip_for_minimize` to translate from user's physical
        intent first.
    epsilon : float
        Half-width of the virtual point pair in unit space; ≥ 0.05
        recommended (see Exp-14).
    delta_norm : float
        Total Y_norm separation between the lower and upper virtual
        partner of a pair. 0.5 means ±0.25 standard deviations.
    """

    def __init__(
        self,
        monotone_dims_internal: dict[int, InternalDirection],
        epsilon: float = 0.10,
        delta_norm: float = 0.5,
    ) -> None:
        if not monotone_dims_internal:
            raise ValueError("monotone_dims_internal cannot be empty.")
        for dim, direction in monotone_dims_internal.items():
            if direction not in ("increasing", "decreasing"):
                raise ValueError(
                    f"Direction for dim {dim} must be 'increasing' or "
                    f"'decreasing', got {direction!r}."
                )
        if epsilon <= 0 or epsilon >= 0.5:
            raise ValueError(f"epsilon must be in (0, 0.5), got {epsilon}.")
        self.dims = monotone_dims_internal
        self.epsilon = float(epsilon)
        self.delta_norm = float(delta_norm)

    def augment(
        self,
        train_X_unit: Tensor,
        train_Y_norm: Tensor,
        n_pairs_per_dim: int = 5,
        iteration: int = 0,
        bo_seed: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """
        Append virtual points to (train_X, train_Y).

        Returns NEW tensors; originals are unchanged.
        """
        d = train_X_unit.shape[1]
        y_center = float(train_Y_norm.mean())
        half_delta = self.delta_norm / 2.0
        eps = self.epsilon

        extra_X: list[Tensor] = []
        extra_Y: list[Tensor] = []

        for dim, direction in self.dims.items():
            if not (0 <= dim < d):
                raise IndexError(
                    f"monotone dim {dim} out of range for d={d}."
                )
            # P3: seed varies with dim, iteration, bo_seed.
            rng_seed = 42 + dim * 1000 + iteration * 100 + bo_seed
            rng = torch.Generator().manual_seed(rng_seed)
            pts = torch.rand(n_pairs_per_dim, d, dtype=torch.float64, generator=rng)
            pts[:, dim] = pts[:, dim].clamp(eps, 1.0 - eps)

            pts_minus = pts.clone()
            pts_minus[:, dim] -= eps
            pts_plus = pts.clone()
            pts_plus[:, dim] += eps

            sign = 1.0 if direction == "increasing" else -1.0
            for i in range(n_pairs_per_dim):
                extra_X.append(pts_minus[i])
                extra_Y.append(
                    torch.tensor([y_center - sign * half_delta], dtype=torch.float64)
                )
                extra_X.append(pts_plus[i])
                extra_Y.append(
                    torch.tensor([y_center + sign * half_delta], dtype=torch.float64)
                )

        if not extra_X:
            return train_X_unit, train_Y_norm

        X_extra = torch.stack(extra_X)
        Y_extra = torch.stack(extra_Y)
        return (
            torch.cat([train_X_unit, X_extra], dim=0),
            torch.cat([train_Y_norm, Y_extra], dim=0),
        )

    def n_rows_per_iter(self, n_pairs_per_dim: int) -> int:
        """How many virtual rows this augmenter adds at each BO iter."""
        return 2 * n_pairs_per_dim * len(self.dims)
