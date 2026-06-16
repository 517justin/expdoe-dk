"""
GP builder. Reads a Knowledge spec, returns a configured SingleTaskGP +
its MonotonicAugmenter (None if no monotone item).

All inputs/outputs work in unit + Y_norm space; the caller (Campaign) is
responsible for physical ↔ unit conversion.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior

from ..space import Space
from ..knowledge import Knowledge, GP_PRIOR_PRESETS
from ..knowledge._frame import flip_for_minimize
from ..knowledge.monotone import MonotonicAugmenter
from ..knowledge.shape import (
    ArrheniusMeanFrozen,
    QuadraticMeanFrozen,
    CombinedMean,
)


def _resolve_dim_index(space: Space, param_name: str) -> int:
    return space.param_names.index(param_name)


def _build_mean_function(
    space: Space, knowledge: Knowledge
) -> nn.Module | None:
    """Combine all shape priors into a CombinedMean (or None)."""
    mean_modules: list[nn.Module] = []

    for it in knowledge.items_of("arrhenius"):
        temp_idx = _resolve_dim_index(space, it.param)
        mean_modules.append(
            ArrheniusMeanFrozen(
                temp_dim_index=temp_idx,
                activation_energy=it.activation_energy,
                amplitude_init=it.amplitude_init,
            )
        )

    quad_items = knowledge.items_of("quadratic_peak")
    if quad_items:
        for it in quad_items:
            dim = _resolve_dim_index(space, it.param)
            # center in physical units → convert to [0,1]
            param = space.param_by_name(it.param)
            lo, hi = param.bounds
            center_unit = (it.center - lo) / (hi - lo)
            curvature_signs = [0.0] * space.n_dims
            # peak in physical → minimum in internal frame (we minimize -y when maximize)
            # The sign here is for the GP's Y_internal frame: a peak in user
            # objective = a minimum in internal Y when maximize=True.
            # CombinedMean is added directly to GP mean; positive curvature
            # = upward parabola = valley = "GP fits low value at center".
            # For maximize=True + user "peak": internal minimum at center
            # → curvature_sign = +1 (upward parabola).
            if it.direction == "peak":
                sign_internal = +1.0 if space.maximize[0] else -1.0
            else:  # valley in user frame
                sign_internal = -1.0 if space.maximize[0] else +1.0
            curvature_signs[dim] = sign_internal
            centers = [0.5] * space.n_dims
            centers[dim] = float(center_unit)
            mean_modules.append(
                QuadraticMeanFrozen(
                    input_dim=space.n_dims,
                    curvature_signs=curvature_signs,
                    centers=centers,
                )
            )

    if not mean_modules:
        return None
    if len(mean_modules) == 1:
        return mean_modules[0]
    return CombinedMean(*mean_modules)


def _build_augmenter(
    space: Space, knowledge: Knowledge
) -> MonotonicAugmenter | None:
    mono_items = knowledge.items_of("monotone")
    if not mono_items:
        return None
    internal_dims: dict[int, str] = {}
    epsilons: list[float] = []
    delta_norms: list[float] = []
    for it in mono_items:
        dim = _resolve_dim_index(space, it.param)
        # Translate physical-effect → GP-frame direction.
        direction = flip_for_minimize(it.effect, space.maximize[0])
        internal_dims[dim] = direction
        epsilons.append(knowledge.resolve_epsilon(it))
        delta_norms.append(it.delta_norm)
    # Use the smallest ε across declared monotone params (most conservative).
    # All monotone dims share one Augmenter for simplicity in v0.1.
    return MonotonicAugmenter(
        monotone_dims_internal=internal_dims,  # type: ignore[arg-type]
        epsilon=min(epsilons),
        delta_norm=min(delta_norms),
    )


def _build_likelihood_and_kernel(
    knowledge: Knowledge,
    d: int,
) -> tuple[GaussianLikelihood | None, Any | None]:
    """Return (likelihood, covar_module) per GP-prior preset; None if no preset."""
    gp_prior_items = knowledge.items_of("gp_prior")
    if not gp_prior_items:
        return None, None
    preset = GP_PRIOR_PRESETS[gp_prior_items[-1].lengthscale]
    covar = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=d,
            lengthscale_prior=GammaPrior(*preset["ls"]),
        ),
        outputscale_prior=GammaPrior(*preset["os"]),
    )
    lik = GaussianLikelihood(noise_prior=GammaPrior(*preset["noise"]))
    return lik, covar


def build_gp(
    space: Space,
    knowledge: Knowledge,
    train_X_unit: torch.Tensor,
    train_Y_norm: torch.Tensor,
) -> tuple[SingleTaskGP, MonotonicAugmenter | None]:
    """
    Build a SingleTaskGP given the space + knowledge + already-augmented
    training tensors.

    Returns
    -------
    (model, augmenter)
        The augmenter is returned so the Campaign can call it again on the
        next iteration before refit. None if no monotone item.
    """
    knowledge.validate()
    mean_function = _build_mean_function(space, knowledge)
    lik, covar = _build_likelihood_and_kernel(knowledge, train_X_unit.shape[1])

    kwargs: dict[str, Any] = {}
    if mean_function is not None:
        kwargs["mean_module"] = mean_function
    if covar is not None:
        kwargs["covar_module"] = covar
    if lik is not None:
        kwargs["likelihood"] = lik

    model = SingleTaskGP(train_X_unit, train_Y_norm, **kwargs)
    augmenter = _build_augmenter(space, knowledge)
    return model, augmenter
