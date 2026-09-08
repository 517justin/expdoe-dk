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
from ..knowledge import Knowledge
from ..knowledge.artifacts import OptimizationArtifact
from ..knowledge.monotone import MonotonicAugmenter
from ..knowledge.shape import (
    ArrheniusMeanFrozen,
    QuadraticMeanFrozen,
    CombinedMean,
)


def build_mean_from_artifacts(
    space: Space, artifacts: tuple[OptimizationArtifact, ...]
) -> nn.Module | None:
    """Combine all shape priors into a CombinedMean (or None)."""
    mean_modules: list[nn.Module] = []
    for artifact in artifacts:
        payload = artifact.payload
        if artifact.kind == "arrhenius_mean":
            mean_modules.append(
                ArrheniusMeanFrozen(
                    temp_dim_index=payload["temp_dim_index"],
                    activation_energy=payload["activation_energy"],
                    amplitude_init=payload["amplitude_init"],
                )
            )
        elif artifact.kind == "quadratic_mean":
            mean_modules.append(
                QuadraticMeanFrozen(
                    input_dim=payload["input_dim"],
                    curvature_signs=list(payload["curvature_signs"]),
                    centers=list(payload["centers"]),
                )
            )

    if not mean_modules:
        return None
    if len(mean_modules) == 1:
        return mean_modules[0]
    return CombinedMean(*mean_modules)


def build_virtual_observation_augmenter(
    space: Space, artifacts: tuple[OptimizationArtifact, ...]
) -> MonotonicAugmenter | None:
    monotone_artifacts = [item for item in artifacts if item.kind == "monotone"]
    if not monotone_artifacts:
        return None
    internal_dims: dict[int, str] = {}
    epsilons: list[float] = []
    delta_norms: list[float] = []
    for artifact in monotone_artifacts:
        payload = artifact.payload
        internal_dims[payload["dim"]] = payload["direction"]
        epsilons.append(payload["epsilon"])
        delta_norms.append(payload["delta_norm"])
    # Use the smallest ε across declared monotone params (most conservative).
    # All monotone dims share one Augmenter for simplicity in v0.1.
    return MonotonicAugmenter(
        monotone_dims_internal=internal_dims,  # type: ignore[arg-type]
        epsilon=min(epsilons),
        delta_norm=min(delta_norms),
    )


def build_prior_modules(
    artifacts: tuple[OptimizationArtifact, ...],
    d: int,
) -> tuple[GaussianLikelihood | None, Any | None]:
    """Return (likelihood, covar_module) per GP-prior preset; None if no preset."""
    prior_artifacts = [item for item in artifacts if item.kind == "gp_prior"]
    if not prior_artifacts:
        return None, None
    preset = prior_artifacts[-1].payload
    covar = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=d,
            lengthscale_prior=GammaPrior(*preset["lengthscale"]),
        ),
        outputscale_prior=GammaPrior(*preset["outputscale"]),
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
    artifacts = knowledge.compile(space=space, observations=None)
    mean_function = build_mean_from_artifacts(space, artifacts.mean_components)
    lik, covar = build_prior_modules(artifacts.priors, train_X_unit.shape[1])

    kwargs: dict[str, Any] = {}
    if mean_function is not None:
        kwargs["mean_module"] = mean_function
    if covar is not None:
        kwargs["covar_module"] = covar
    if lik is not None:
        kwargs["likelihood"] = lik

    model = SingleTaskGP(train_X_unit, train_Y_norm, **kwargs)
    augmenter = build_virtual_observation_augmenter(
        space, artifacts.virtual_observations
    )
    return model, augmenter
