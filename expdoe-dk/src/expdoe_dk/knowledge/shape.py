"""
Frozen mean functions: shape priors for GP without MLE absorption.

These were validated in the DOEGP Plan 2 experiments (Exp-9, Exp-10).
Lessons baked in:
- Learnable mean parameters get absorbed by the MLE during GP fitting
  (Exp-7 finding): the GP attributes structure to the kernel instead and
  the mean function adds no value. Frozen variants preserve the prior.
- Operate in [0,1] unit input space (so callers must normalize X first).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ArrheniusMeanFrozen(nn.Module):
    """
    Arrhenius-type mean: m(x) = amplitude * exp(-Ea / x_T)

    The activation energy Ea is FROZEN (a buffer, not a Parameter); only
    the amplitude is learned. This prevents MLE absorption.

    Parameters
    ----------
    temp_dim_index : int
        Which input dimension is temperature (in [0,1] unit space).
        Default 0.
    activation_energy : float
        Frozen Ea. Default 1.0 — gives a smooth rising shape over [0,1].
    amplitude_init : float
        Initial value of the learned amplitude. Sign indicates whether the
        objective rises (positive) or falls (negative) with temperature in
        the GP frame; the frame translator handles physical-space sign.
    """

    def __init__(
        self,
        temp_dim_index: int = 0,
        activation_energy: float = 1.0,
        amplitude_init: float = -1.0,
    ) -> None:
        super().__init__()
        self.temp_dim_index = temp_dim_index
        self.register_buffer(
            "activation_energy",
            torch.tensor(activation_energy, dtype=torch.double),
        )
        self.amplitude = nn.Parameter(
            torch.tensor(amplitude_init, dtype=torch.double)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = x[..., self.temp_dim_index].clamp(min=0.01)
        return self.amplitude * torch.exp(-self.activation_energy / temp)


class QuadraticMeanFrozen(nn.Module):
    """
    Quadratic mean centered at a known optimum, summed over all input dims.

        m(x) = sum_i ( sign_i * |c_i| * (x_i - center_i)^2 ) + offset

    `curvature_signs[i] = -1` → x_i has a peak at center_i (bell shape);
    `curvature_signs[i] = +1` → x_i has a valley at center_i;
    `curvature_signs[i] =  0` → x_i contributes nothing.

    Sign list and centers are FROZEN. Only the per-dim magnitudes and a
    scalar offset are learned, which keeps the "where is the optimum"
    structure injected by the user.
    """

    def __init__(
        self,
        input_dim: int,
        curvature_signs: list[float],
        centers: list[float],
    ) -> None:
        if len(curvature_signs) != input_dim:
            raise ValueError(
                f"curvature_signs length {len(curvature_signs)} != input_dim "
                f"{input_dim}."
            )
        if len(centers) != input_dim:
            raise ValueError(
                f"centers length {len(centers)} != input_dim {input_dim}."
            )
        super().__init__()
        self.register_buffer(
            "curvature_signs",
            torch.tensor(curvature_signs, dtype=torch.double),
        )
        self.register_buffer(
            "centers",
            torch.tensor(centers, dtype=torch.double),
        )
        self.curvature_magnitudes = nn.Parameter(
            torch.ones(input_dim, dtype=torch.double)
        )
        self.offset = nn.Parameter(torch.tensor(0.0, dtype=torch.double))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x - self.centers
        return (
            self.curvature_signs * self.curvature_magnitudes.abs() * diff ** 2
        ).sum(dim=-1) + self.offset


class CombinedMean(nn.Module):
    """Sum of arbitrary mean-function modules."""

    def __init__(self, *mean_modules: nn.Module) -> None:
        if not mean_modules:
            raise ValueError("CombinedMean: pass at least one mean module.")
        super().__init__()
        self.means = nn.ModuleList(mean_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.means[0](x)
        for m in self.means[1:]:
            out = out + m(x)
        return out
