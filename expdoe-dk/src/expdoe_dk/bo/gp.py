"""
GP builder. Reads a Knowledge spec, returns a configured SingleTaskGP +
its MonotonicAugmenter (None if no monotone item).

All inputs/outputs work in unit + Y_norm space; the caller (Campaign) is
responsible for physical ↔ unit conversion.
"""
from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior

from ..space import Space
from ..knowledge import Knowledge
from ..knowledge.artifacts import OptimizationArtifact
from ..knowledge._frame import flip_for_minimize
from ..errors import EngineError, ErrorCode
from ..knowledge.monotone import MonotonicAugmenter
from ..knowledge.shape import (
    ArrheniusMeanFrozen,
    QuadraticMeanFrozen,
    CombinedMean,
)


class _FeatureKernel(Kernel):
    """Keep knowledge in Space coordinates while the GP uses shared features."""

    def __init__(self, kernel, transform):
        super().__init__()
        self.kernel = kernel
        self.transform = transform

    def forward(self, x1, x2, diag=False, **params):
        return self.kernel(self.transform(x1), self.transform(x2), diag=diag, **params)


class _PhysicalFormulaMean(nn.Module):
    """Physical formulas, decoded before evaluation and normalized afterward.

    Periodic phase is in radians: sin(2*pi*x/period + phase). Quadratic width
    scales displacement: ((x-center)/width)**2. The caller supplies the physical
    response location/scale of its training normalization, before utility sign.
    """

    def __init__(self, space, artifact, response_location, response_scale):
        super().__init__()
        self.artifact = artifact
        self.dimension = artifact.payload["dimension"]
        parameter = space.param_by_name(artifact.payload["factor"])
        limits = parameter.bounds or (min(parameter.numeric_levels), max(parameter.numeric_levels))
        self.low, self.high = map(float, limits)
        self.log = parameter.transform == "log" or parameter.log_scale
        self.sign = -1. if flip_for_minimize("increases_objective", space.maximize[0]) == "decreasing" else 1.
        self.location, self.scale = response_location, response_scale

    def forward(self, X):
        unit = X[..., self.dimension]
        if self.log:
            x = torch.exp(math.log(self.low) + unit * math.log(self.high / self.low))
        else:
            x = self.low + unit * (self.high - self.low)
        p = self.artifact.payload["parameters"]
        kind = self.artifact.kind
        if kind == "power_law_descriptor":
            value = p["scale"] * x.pow(p["exponent"])
        elif kind == "exponential_descriptor":
            value = p["amplitude"] * torch.exp(p["rate"] * x)
        elif kind == "quadratic_valley_descriptor":
            value = ((x - p["center"]) / p["width"]).square()
        else:
            value = torch.sin(2 * math.pi * x / p["period"] + p["phase"])
        result = self.sign * (value - self.location) / self.scale
        if not torch.isfinite(result).all():
            raise EngineError(ErrorCode.KNOWLEDGE_INVALID, "Knowledge mean produced non-finite values",
                              details={"pattern_id": self.artifact.source_pattern_id, "kind": kind})
        return result


class _CategorySimilarityKernel(Kernel):
    def __init__(self, space, artifact):
        super().__init__(active_dims=(artifact.payload["dimension"],))
        parameter = space.param_by_name(artifact.payload["factor"])
        p = artifact.payload["parameters"]
        indices = parameter.encode(p["levels"])
        order = [indices.index(float(i)) for i in range(len(indices))]
        matrix = torch.tensor(p["matrix"], dtype=torch.double)
        self.register_buffer("similarities", matrix[order][:, order])

    def forward(self, x1, x2, diag=False, **params):
        first, second = x1[..., 0], x2[..., 0]
        for values in (first, second):
            if (not torch.isfinite(values).all() or (values != values.round()).any()
                    or (values < 0).any() or (values >= len(self.similarities)).any()):
                raise EngineError(ErrorCode.CONFIG_INVALID, "Similarity kernel requires declared categorical indices")
        if diag:
            return self.similarities[first.long(), second.long()]
        return self.similarities[first.long().unsqueeze(-1), second.long().unsqueeze(-2)]


def build_kernel_from_artifacts(space: Space, artifacts: tuple[OptimizationArtifact, ...]) -> Kernel | None:
    """Product of PSD categorical similarity components, in model coordinates."""
    from .knowledge_runtime import validate_runtime_artifact

    kernels = []
    for artifact in artifacts:
        if artifact.kind != "category_similarity_descriptor":
            raise EngineError(ErrorCode.KNOWLEDGE_INVALID, "Unsupported kernel artifact",
                              details={"pattern_id": artifact.source_pattern_id, "kind": artifact.kind})
        validate_runtime_artifact(space, "kernel_components", artifact)
        kernels.append(_CategorySimilarityKernel(space, artifact))
    if not kernels:
        return None
    result = kernels[0]
    for kernel in kernels[1:]:
        result = result * kernel
    return result


def build_mean_from_artifacts(
    space: Space, artifacts: tuple[OptimizationArtifact, ...], *,
    response_location: float = 0., response_scale: float = 1.,
) -> nn.Module | None:
    """Build one structured mean, or combine supported legacy shape priors."""
    if not math.isfinite(response_location) or not math.isfinite(response_scale) or response_scale <= 0:
        raise EngineError(ErrorCode.CONFIG_INVALID, "Response normalization must have finite location and positive scale")
    from .structured_coverage import STRUCTURED_KINDS, check_structured_composition
    if any(artifact.kind in STRUCTURED_KINDS for artifact in artifacts):
        from ..knowledge.artifacts import OptimizationArtifacts
        from .structured_mean import StructuredMean
        check_structured_composition(space, OptimizationArtifacts(mean_components=artifacts))
        return StructuredMean(space, artifacts, response_location=response_location,
                              response_scale=response_scale)
    mean_modules: list[nn.Module] = []
    physical_location_applied = False
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
        elif artifact.kind in {"power_law_descriptor", "exponential_descriptor",
                               "quadratic_valley_descriptor", "periodic_descriptor"}:
            location = 0. if physical_location_applied else response_location
            mean_modules.append(_PhysicalFormulaMean(space, artifact, location, response_scale))
            physical_location_applied = True
        else:
            raise EngineError(ErrorCode.KNOWLEDGE_INVALID, "Unsupported mean artifact",
                              details={"pattern_id": artifact.source_pattern_id, "kind": artifact.kind})

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
    *, response_location: float | None = None, response_scale: float | None = None,
    train_Yvar: torch.Tensor | None = None,
    feature_transform: nn.Module | None = None,
    feature_groups: tuple[tuple[int, ...], ...] | None = None,
    runtime: dict | None = None,
    normalize_outcome: bool = True,
) -> tuple[SingleTaskGP, MonotonicAugmenter | None]:
    """
    Build a SingleTaskGP given the space + knowledge + already-augmented
    training tensors.

    Physical formula means require ``response_location`` and ``response_scale``
    computed from physical outcomes in the training subset only. The GP frame
    remains internally minimized. Omitted statistics raise CONFIG_INVALID;
    legacy means do not require them.

    Returns
    -------
    (model, augmenter)
        The augmenter is returned so the Campaign can call it again on the
        next iteration before refit. None if no monotone item.
    """
    knowledge.validate()
    from .knowledge_runtime import compile_runtime
    from .structured_coverage import STRUCTURED_KINDS
    artifacts = compile_runtime(space, knowledge, None) if runtime is None else runtime
    physical_means = [artifact for artifact in artifacts["mean_components"]
                      if artifact.kind.endswith("_descriptor") or artifact.kind in STRUCTURED_KINDS]
    if physical_means and (response_location is None or response_scale is None):
        raise EngineError(ErrorCode.CONFIG_INVALID,
            "Physical mean formulas require the physical response_location and response_scale used to normalize training outcomes",
            details={"pattern_ids": [artifact.source_pattern_id for artifact in physical_means],
                     "required": ["response_location", "response_scale"],
                     "source": "physical training outcomes only"})
    mean_function = build_mean_from_artifacts(space, artifacts["mean_components"],
        response_location=0. if response_location is None else response_location,
        response_scale=1. if response_scale is None else response_scale)
    dimensions = (sum(map(len, feature_groups)) if feature_groups is not None
                  else train_X_unit.shape[1])
    lik, covar = build_prior_modules(artifacts["priors"], dimensions)
    if train_Yvar is not None and lik is not None:
        # A learned-noise prior must not replace supplied measurement variance.
        lik = None
    similarity = build_kernel_from_artifacts(space, artifacts["kernel_components"])
    if similarity is not None:
        covered = {a.payload["dimension"] for a in artifacts["kernel_components"]}
        remaining = ([j for i, group in enumerate(feature_groups) if i not in covered for j in group]
                     if feature_groups is not None else
                     [i for i in range(space.n_dims) if i not in covered])
        if covar is not None:
            if remaining:
                # Rebuild the ARD prior kernel with the actual continuous dims.
                _, covar = build_prior_modules(artifacts["priors"], len(remaining))
                covar.base_kernel.active_dims = torch.tensor(remaining)
                base = covar.base_kernel
                if feature_transform is not None:
                    base = _FeatureKernel(base, feature_transform)
                covar.base_kernel = base * similarity
            else:
                covar.base_kernel = similarity
        elif remaining:
            base = MaternKernel(nu=2.5, ard_num_dims=len(remaining), active_dims=remaining)
            if feature_transform is not None:
                base = _FeatureKernel(base, feature_transform)
            covar = base * similarity
        else:
            covar = similarity
    elif feature_transform is not None:
        if covar is None:
            covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dimensions))
        covar = _FeatureKernel(covar, feature_transform)

    kwargs: dict[str, Any] = {}
    if mean_function is not None:
        kwargs["mean_module"] = mean_function
    if covar is not None:
        kwargs["covar_module"] = covar
    if lik is not None:
        kwargs["likelihood"] = lik
    if train_Yvar is not None:
        kwargs["train_Yvar"] = train_Yvar
    if not normalize_outcome:
        # Public fitting already normalized using real training outcomes only.
        kwargs["outcome_transform"] = None

    model = SingleTaskGP(train_X_unit, train_Y_norm, **kwargs)
    augmenter = build_virtual_observation_augmenter(
        space, artifacts["virtual_observations"]
    )
    return model, augmenter
