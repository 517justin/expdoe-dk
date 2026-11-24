"""Physical-response surrogate fitting, with training-only feature statistics."""
from __future__ import annotations

from copy import deepcopy
import inspect
import math

import numpy as np
import pandas as pd
import torch
from botorch import settings
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from botorch.posteriors.transformed import TransformedPosterior
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..domain import ObservationBatch, Space
from ..errors import EngineError, ErrorCode
from ..knowledge import Knowledge
from ..knowledge.artifacts import OptimizationArtifacts
from .contracts import Prediction
from .gp import build_gp, build_virtual_observation_augmenter
from .knowledge_runtime import compile_runtime
from .objective_routing import project_objective


def _model_inputs(space, frame):
    try:
        return space.physical_to_model(frame)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise EngineError(ErrorCode.CONFIG_INVALID, f"Invalid model inputs: {error}") from error


def _eligible(space, observations, *, minimum):
    if not isinstance(space, Space) or not isinstance(observations, ObservationBatch):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Expected Space and ObservationBatch")
    responses = observations.Y
    if not responses.columns.is_unique or set(responses.columns) != set(space.objectives):
        raise EngineError(ErrorCode.CONFIG_INVALID, "Observation response columns must match Space objectives")
    responses = responses[space.objectives]
    status = observations.status
    status = ["success"] * len(observations.X) if status is None else status.tolist()
    indices = [i for i, value in enumerate(status) if value == "success"]
    exclusions = tuple({"index": i, "id": observations.ids[i] if observations.ids else str(i),
                        "reason": value} for i, value in enumerate(status) if value != "success")
    if len(indices) < minimum:
        raise EngineError(ErrorCode.INSUFFICIENT_DATA, f"At least {minimum} successful observations are required",
                          details={"successes": len(indices), "exclusions": list(exclusions)})
    variance = observations.Yvar
    if variance is not None:
        try:
            if not variance.columns.is_unique or set(variance.columns) != set(space.objectives):
                raise ValueError("Variance columns must match Space objectives")
            variance = variance[space.objectives]
            values = variance.iloc[indices].to_numpy(dtype=float)
            valid = (values.shape == (len(indices), space.n_objectives)
                     and np.isfinite(values).all() and (values >= 0).all())
        except (ValueError, TypeError, IndexError):
            valid = False
        if not valid:
            raise EngineError(ErrorCode.CONFIG_INVALID,
                "Known variance must be finite and non-negative for every successful row; mixed missing variance is unsupported")
    selected = observations._subset(rows=indices, objectives=space.objectives)
    variance = selected.Yvar
    if variance is not None and any(dtype == object for dtype in variance.dtypes):
        # Only the validated, successful fitting copy needs numeric dtype;
        # original/projected audit batches retain their missing-value markers.
        variance = variance.astype(float)
    batch = ObservationBatch(selected.X, selected.Y, Yvar=variance,
        ids=tuple(observations.ids[i] if observations.ids else str(i) for i in indices))
    _model_inputs(space, batch.X)
    return batch, exclusions


class _PhysicalFeatures(torch.nn.Module):
    """One-hot nominal, declared ordinal rank, physical numeric z-scores.

    GP inputs remain Space coordinates so declarative means and similarity
    kernels retain their semantics. This differentiable kernel transform decodes
    those inputs into the same features consumed by the reference models.
    Integral numeric factors are centered as exact integer offsets before float
    conversion. Their standardized features are physical z-scores without ever
    reconstructing large absolute integer values in floating point.
    """

    def __init__(self, space, training):
        super().__init__()
        self.params = tuple(space.params)
        self.groups = []
        self.names = []
        self.integral_limits = []
        self.numeric_origins = []
        self.physical_locations = []
        numeric_location, numeric_scale = [], []
        offset = 0
        for parameter in self.params:
            integral_limits = None
            if parameter.kind == "integer":
                integral_limits = tuple(int(value) for value in parameter.bounds)
            elif parameter.kind == "discrete":
                levels = parameter.numeric_levels
                if all(value == int(value) for value in levels):
                    integral_limits = (int(min(levels)), int(max(levels)))
            self.integral_limits.append(integral_limits)
            if parameter.kind == "categorical":
                width = len(parameter.values)
                self.names.extend(f"{parameter.name}={level}" for level in parameter.values)
            else:
                width = 1
                self.names.append(parameter.name)
                origin = 0 if integral_limits is None else integral_limits[0]
                if integral_limits is not None:
                    values = np.asarray([int(value) - origin for value in training[parameter.name].tolist()],
                                        dtype=float)
                else:
                    values = (np.array(parameter.encode(training[parameter.name].tolist()))
                              if parameter.kind == "ordinal" else training[parameter.name].to_numpy(dtype=float))
                constant = bool(np.all(values == values[0]))
                location = float(values[0] if constant else values.mean())
                numeric_location.append(location)
                numeric_scale.append(1. if constant else (float(values.std()) or 1.))
                self.numeric_origins.append(origin)
                # Preserve an exactly representable integer center in diagnostics;
                # origin + offset remains the authoritative decomposition otherwise.
                self.physical_locations.append(origin + (int(location) if integral_limits is not None
                                                          and location.is_integer() else location))
            self.groups.append(tuple(range(offset, offset + width)))
            offset += width
        self.groups = tuple(self.groups)
        self.register_buffer("location", torch.tensor(numeric_location, dtype=torch.double))
        self.register_buffer("scale", torch.tensor(numeric_scale, dtype=torch.double))

    def forward(self, X):
        columns = []
        numeric = 0
        for dim, parameter in enumerate(self.params):
            values = X[..., dim:dim + 1]
            if parameter.kind == "categorical":
                levels = torch.arange(len(parameter.values), dtype=X.dtype, device=X.device)
                columns.append((values == levels).to(X.dtype))
            else:
                if parameter.kind != "ordinal":
                    integral_limits = self.integral_limits[dim]
                    if integral_limits is not None:
                        lo, hi = integral_limits
                        span = hi - lo  # Exact subtraction, before float conversion.
                        if parameter.transform == "log" or parameter.log_scale:
                            values = float(lo) * torch.expm1(values * math.log1p(span / lo))
                        else:
                            values = values * float(span)
                    else:
                        lo, hi = parameter.bounds or (min(parameter.numeric_levels), max(parameter.numeric_levels))
                        if parameter.transform == "log" or parameter.log_scale:
                            values = torch.exp(math.log(lo) + values * math.log(hi / lo))
                        else:
                            values = lo + values * (hi - lo)
                columns.append((values - self.location[numeric]) / self.scale[numeric])
                numeric += 1
        return torch.cat(columns, dim=-1)

    def describe(self):
        return {"input_encoding": "physical_numeric_zscore_nominal_one_hot_ordinal_rank_zscore",
                "features": list(self.names), "numeric_location": list(self.physical_locations),
                "numeric_origin": list(self.numeric_origins),
                "numeric_location_offset": self.location.tolist(),
                "numeric_scale": self.scale.tolist()}


class _PhysicalAcquisitionModel(Model):
    """Posterior-only adapter; no fitting or internal response-frame access."""

    def __init__(self, model, location, scale, sign):
        super().__init__()
        self._source = model
        self._location, self._scale, self._sign = location, scale, sign
        self._directional_guard = None
        if getattr(model, "_directional_affine", False):
            from .directional_source import PhysicalGuard
            self._directional_guard = PhysicalGuard(self, "_source")

    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return self._source.batch_shape

    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None):
        if output_indices not in (None, [0]) or observation_noise is not False:
            raise EngineError(ErrorCode.CONFIG_INVALID,
                "Acquisition adapter supports only the single latent response")
        if self._directional_guard is not None:
            self._directional_guard.check()
        posterior = self._source.posterior(X)
        if self._directional_guard is not None:
            from .directional_source import checked_affine, checked_square
            guard = self._directional_guard
            guard.check()
            multiplier = self._sign * self._scale
            squared = checked_square(self._scale, guard)
            physical = TransformedPosterior(posterior,
                sample_transform=lambda samples: checked_affine(samples, multiplier, self._location, guard=guard),
                mean_transform=lambda mean, variance: checked_affine(mean, multiplier, self._location, guard=guard),
                variance_transform=lambda mean, variance: checked_affine(variance, squared, guard=guard, variance=True))
            return physical if posterior_transform is None else posterior_transform(physical)
        multiplier = self._sign * self._scale
        physical = TransformedPosterior(posterior,
            sample_transform=lambda samples: samples * multiplier + self._location,
            mean_transform=lambda mean, variance: mean * multiplier + self._location,
            variance_transform=lambda mean, variance: variance * self._scale**2)
        return physical if posterior_transform is None else posterior_transform(physical)


class FittedSurrogate:
    """Fitted production GP; predict returns physical means and latent variances.

    ``predict``, ``acquisition_model``, ``transforms``, ``diagnostics`` and ``exclusions`` are public. Reference
    estimators are evaluation-only and cannot construct this public contract.
    """

    def __init__(self, space, model, location, scale, sign, transforms, exclusions):
        self._space = space
        self._model = model
        self._location, self._scale, self._sign = location, scale, sign
        self._transforms, self._exclusions = deepcopy(transforms), deepcopy(exclusions)
        self._diagnostics = {"response_location": location, "response_scale": scale,
                             "structured_mean": {}}
        self._directional_guard = None
        if getattr(model, "_directional_affine", False):
            from .directional_source import PhysicalGuard
            self._directional_guard = PhysicalGuard(self, "_model")

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)

    @property
    def transforms(self):
        return deepcopy(self._transforms)

    @property
    def exclusions(self):
        return deepcopy(self._exclusions)

    def acquisition_model(self) -> Model:
        """Return a posterior-only BoTorch adapter for acquisition evaluation.

        Inputs are differentiable ``(..., q, d)`` Space model coordinates, as
        produced by ``Space.physical_to_model`` (NOT physical values or expanded
        kernel features). Joint latent posterior means, variances and samples
        are in physical response units, exactly like ``predict``. Use MC
        acquisitions with ``cache_root=False`` for this transformed posterior.
        The raw fitted GP and its internal response frame are not public.
        """
        return _PhysicalAcquisitionModel(self._model, self._location, self._scale, self._sign)

    def predict(self, X: pd.DataFrame) -> Prediction:
        inputs = _model_inputs(self._space, X)
        if len(inputs) == 0:
            return Prediction(np.empty((0, 1)), np.empty((0, 1)))
        try:
            with torch.no_grad():
                if self._directional_guard is not None:
                    self._directional_guard.check()
                posterior = self._model.posterior(inputs)
                if self._directional_guard is None:
                    mean = posterior.mean.numpy() * self._sign * self._scale + self._location
                    variance = posterior.variance.numpy() * self._scale ** 2
                else:
                    from .directional_source import checked_affine, checked_square
                    guard = self._directional_guard
                    signed = checked_affine(posterior.mean, self._sign, guard=guard)
                    mean = checked_affine(signed, self._scale, self._location, guard=guard).numpy()
                    variance = checked_affine(posterior.variance, checked_square(self._scale, guard),
                                              guard=guard, variance=True).numpy()
            return Prediction(mean, variance)
        except EngineError:
            raise
        except Exception as error:
            raise EngineError(ErrorCode.MODEL_FIT_FAILED, f"Surrogate prediction failed: {error}") from error


def _multioutput_types():
    # New BoTorch APIs must never become import requirements for scalar fits.
    try:
        from botorch.models.model import ModelList
        from botorch.sampling.list_sampler import ListSampler
        from botorch.sampling.normal import IIDNormalSampler

        inspect.signature(ModelList).bind(object(), object())
        inspect.signature(ListSampler).bind(object(), object())
        inspect.signature(IIDNormalSampler).bind(sample_shape=torch.Size([1]), seed=0)
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
            "Installed BoTorch lacks the ModelList/ListSampler multioutput sampling APIs",
            details={"capability": "physical_multioutput_sampling"},
            hints=("Use a BoTorch version supporting ModelList, ListSampler and "
                   "IIDNormalSampler(sample_shape, seed).",)) from error
    return ModelList, ListSampler, IIDNormalSampler


class _MultiOutputFittedSurrogate(FittedSurrogate):
    """Independent physical GPs ordered by Space objective names."""

    def __init__(self, space, fitted, exclusions):
        self._space, self._fitted = space, tuple(fitted)
        self._exclusions = deepcopy(exclusions)
        self._transforms = {"objective_names": list(space.objectives), "objectives": {
            name: model.transforms for name, model in zip(space.objectives, self._fitted)}}
        self._diagnostics = {"objective_names": list(space.objectives),
            "independent_outputs": True, "objectives": {
                name: model.diagnostics for name, model in zip(space.objectives, self._fitted)}}

    def predict(self, X: pd.DataFrame) -> Prediction:
        predictions = [model.predict(X) for model in self._fitted]
        return Prediction(np.concatenate([p.mean for p in predictions], axis=-1),
                          np.concatenate([p.variance for p in predictions], axis=-1))

    def acquisition_model(self) -> Model:
        """Physical ModelList posterior in Space coordinates; use cache_root=False.

        Pair with ``acquisition_sampler`` for joint-q PosteriorList samples.
        Outputs are independent; within-output joint covariance is preserved.
        Only latent responses are supported, as in the single-output adapter.
        """
        ModelList, _, _ = _multioutput_types()
        return ModelList(*(model.acquisition_model() for model in self._fitted))

    def acquisition_sampler(self, *, sample_shape: torch.Size, seed: int = 0):
        """Seeded ListSampler yielding sample_shape x batch_shape x q x m."""
        # Separate low-dimensional Sobol streams can couple independent outputs.
        # IID base samples avoid this while each GP still samples its full q joint.
        _, ListSampler, IIDNormalSampler = _multioutput_types()
        return ListSampler(*(IIDNormalSampler(sample_shape=sample_shape, seed=seed + i)
                             for i in range(len(self._fitted))))


def fit_surrogate(space: Space, observations: ObservationBatch, *,
                  knowledge: Knowledge | None = None, seed: int = 0) -> FittedSurrogate:
    """Fit a production GP to >=3 successes, never to objective utilities."""
    return _fit_surrogate(space, observations, knowledge=knowledge, seed=seed, minimum=3)


def _fit_surrogate(space, observations, *, knowledge=None, seed=0, minimum=2,
                   parameter_constraints=()):
    # Two-row fits are necessary only inside three-row leave-one-out evaluation.
    batch, exclusions = _eligible(space, observations, minimum=minimum)
    if knowledge is not None and not isinstance(knowledge, Knowledge):
        raise EngineError(ErrorCode.CONFIG_INVALID, "knowledge must be Knowledge or None")
    if type(seed) is not int or seed < 0:
        raise EngineError(ErrorCode.CONFIG_INVALID, "seed must be a non-negative integer")
    knowledge = knowledge if knowledge is not None else Knowledge()
    runtime = compile_runtime(space, knowledge, batch)
    parameter_constraints = (*parameter_constraints, *runtime["parameter_constraints"])
    if space.n_objectives > 1:
        _multioutput_types()
        fitted = []
        for name in space.objectives:
            projected, selected, scoped = project_objective(space, batch, knowledge, name)
            try:
                fitted.append(_fit_surrogate(projected, selected, knowledge=scoped,
                                             seed=seed, minimum=minimum,
                                             parameter_constraints=parameter_constraints))
            except EngineError as error:
                raise EngineError(error.code, f"Objective {name!r} fitting failed: {error}",
                    details={**error.details, "objective_name": name,
                             "objective_names": list(space.objectives)}, hints=error.hints) from error
        return _MultiOutputFittedSurrogate(space, fitted, exclusions)
    features = _PhysicalFeatures(space, batch.X)
    X = _model_inputs(space, batch.X)
    y = batch.Y.to_numpy(dtype=float)
    constant = bool(np.all(y == y[0, 0]))
    location = float(y[0, 0] if constant else y.mean())
    scale = 1. if constant else (float(y.std()) or 1.)
    sign = -1. if space.maximize[0] else 1.
    Y = torch.tensor(sign * (y - location) / scale, dtype=torch.double)
    variance = None if batch.Yvar is None else torch.tensor(batch.Yvar.to_numpy(dtype=float) / scale**2,
                                                          dtype=torch.double)
    transforms = {**features.describe(), "response_location": location, "response_scale": scale,
                  "response_transform": "training_zscore_internal_sign", "internal_sign": sign,
                  "known_variance": variance is not None,
                  "normalized_yvar": None if variance is None else variance[:, 0].tolist(),
                  "virtual_rows": 0}
    structured_diagnostics = {}
    directional = tuple(a for a in runtime["virtual_observations"] if a.kind == "monotone_sign_v2")
    directional_diagnostics = None
    if directional:
        # Directional factors never enter the legacy function-value augmenter,
        # including the builder's returned augmenter and structured base fits.
        runtime = {**runtime, "virtual_observations": tuple(
            a for a in runtime["virtual_observations"] if a.kind != "monotone_sign_v2")}
    try:
        with torch.random.fork_rng(devices=[]), settings.validate_input_scaling(False):
            torch.manual_seed(seed)
            augmenter = build_virtual_observation_augmenter(space, runtime["virtual_observations"])
            if augmenter is not None:
                count = max(a.payload["n_pairs_per_dim"] for a in runtime["virtual_observations"]
                            if a.kind == "monotone")
                X, Y = augmenter.augment(X, Y, n_pairs_per_dim=count, bo_seed=seed)
            for artifact in runtime["virtual_observations"]:
                if artifact.kind == "random_augment":
                    extra = torch.rand(artifact.payload["n"], space.n_dims, dtype=torch.double)
                    X = torch.cat((X, extra))
                    Y = torch.cat((Y, torch.zeros(len(extra), 1, dtype=torch.double)))
            extra_count = len(X) - len(batch.X)
            if extra_count:
                # Virtual conditions must still use declared categorical/ordinal levels.
                for dim, parameter in enumerate(space.params):
                    if parameter.kind in {"categorical", "ordinal"}:
                        X[len(batch.X):, dim] = (X[len(batch.X):, dim] * len(parameter.values)).floor().clamp(
                            max=len(parameter.values) - 1)
                X = torch.cat((X[:len(batch.X)], _model_inputs(space,
                    space.model_to_physical(X[len(batch.X):]))))
                if variance is not None:
                    variance = torch.cat((variance, variance.mean().expand(extra_count, 1)))
                transforms.update(virtual_rows=extra_count,
                    virtual_variance="mean_training_known_variance" if variance is not None else "learned_noise")
            from .structured_coverage import STRUCTURED_KINDS
            if any(a.kind in STRUCTURED_KINDS for a in runtime["mean_components"]):
                from .structured_fit import fit_structured_gp
                model, structured_diagnostics = fit_structured_gp(space, knowledge, batch,
                    runtime=runtime, X=X, Y=Y, Yvar=variance, feature_transform=features,
                    feature_groups=features.groups, response_location=location,
                    response_scale=scale, seed=seed)
            else:
                model, _ = build_gp(space, knowledge, X, Y, response_location=location, response_scale=scale,
                    train_Yvar=variance, feature_transform=features, feature_groups=features.groups,
                    runtime=runtime, normalize_outcome=False)
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            model.eval()
            if directional:
                from .directional_pairs import build_directional_pairs
                from .directional_posterior import condition_directional_posterior
                pairs = build_directional_pairs(space, batch,
                    OptimizationArtifacts(virtual_observations=directional), seed=seed,
                    parameter_constraints=parameter_constraints)
                likelihood_noise = model.likelihood.noise.detach().tolist()
                model, directional_diagnostics = condition_directional_posterior(model, pairs)
                directional_diagnostics.update(objective_name=space.objectives[0],
                    actual_measurement_count=len(batch.X),
                    base_noise_mode="inferred" if variance is None else "known",
                    declared_noise_mode=("inferred" if variance is None else
                                         "known_zero" if not torch.count_nonzero(variance) else "known"),
                    normalized_measured_variance=transforms["normalized_yvar"],
                    fitted_likelihood_noise=likelihood_noise)
                by_id = {a.source_pattern_id: a for a in directional}
                for detail in directional_diagnostics["pair_diagnostics"]:
                    artifact = by_id[detail["pattern_id"]]
                    payload = artifact.payload
                    direction = payload["direction"]
                    if sign == -1.:
                        direction = "decreasing" if direction == "increasing" else "increasing"
                    endpoints = [space.model_to_physical(pair).to_dict(orient="records")
                                 for pair, identity in zip(pairs.endpoints, pairs.pattern_ids)
                                 if identity == artifact.source_pattern_id]
                    detail.update(pattern=artifact.source_pattern, version=artifact.source_version,
                        objective_name=space.objectives[0], factor=space.param_names[payload["dim"]],
                        seed=seed, direction=direction, internal_direction=payload["direction"],
                        confidence=payload["confidence"], softness=payload["softness"],
                        softness_units="standardized_response_per_model_coordinate",
                        physical_endpoints=endpoints)
        fitted = FittedSurrogate(space, model, location, scale, sign, transforms, exclusions)
        fitted._diagnostics["structured_mean"] = deepcopy(structured_diagnostics)
        if directional_diagnostics is not None:
            fitted._diagnostics["directional"] = directional_diagnostics
        fitted.predict(batch.X)  # Reject non-finite fits before exposing a fitted object.
        return fitted
    except EngineError:
        raise
    except Exception as error:
        raise EngineError(ErrorCode.MODEL_FIT_FAILED, f"GP fitting failed: {error}",
                          details={"model": "gp", "training_ids": list(batch.ids)}) from error


__all__ = ["FittedSurrogate", "fit_surrogate"]
