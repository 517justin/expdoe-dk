"""Tempered-probit Laplace conditioning of an already fitted scalar GP.

No measured responses, noise, hyperparameters, or response transforms are changed.
Fitting is CPU double precision and deterministic. Query evaluation preserves
autograd, model batches, and full joint covariance in the base response frame.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from linear_operator.operators import RootLinearOperator
from scipy import linalg, optimize
from torch import Tensor

from ..errors import EngineError, ErrorCode
from .directional_pairs import DirectionalPairs


_EPS = torch.finfo(torch.double).eps
_RANK_MULTIPLIER = 64
_CHECK_MULTIPLIER = 256
_GTOL = 1e-8
_MAXITER = 200


def _failure(reason, **details):
    return EngineError(ErrorCode.MODEL_FIT_FAILED,
                       "Soft directional posterior failed numerical validation",
                       details={"reason": reason, **details})


def _finite(*values):
    if any(not torch.isfinite(value).all() for value in values):
        raise _failure("nonfinite")


def _joint_latent_moments(model, X):
    """Validate GP construction roundoff before using posterior-scale PSD checks.

    A GP subtracts a training correction from its prior covariance. Prior-scale
    roundoff can therefore exceed a symmetry budget based on the small residual.
    The construction allowance validates skew and certified deterministic entries
    only; it never selects an eigenvalue or positive-noise observation to drop.
    """
    posterior = model.posterior(X)
    mean, covariance = posterior.mean[..., 0], posterior.distribution.covariance_matrix
    _finite(mean, covariance)
    if not isinstance(model, SingleTaskGP):
        return mean, covariance
    with torch.no_grad():
        # posterior() has put the model in eval mode and transformed train_inputs.
        transformed = model.transform_inputs(X)
        training = model.train_inputs[0]
        zero_entries = None
        if isinstance(model.likelihood, FixedNoiseGaussianLikelihood):
            noise = model.likelihood.noise
            if (noise == 0).any():
                matches = (transformed.unsqueeze(-2) == training.unsqueeze(-3)).all(dim=-1)
                known = (matches & (noise == 0).unsqueeze(-2)).any(dim=-1)
                if known.any():
                    zero_entries = known.unsqueeze(-1) | known.unsqueeze(-2)
        skew = (covariance - covariance.mT).abs()
        has_skew = bool(skew.any())
        if not has_skew and zero_entries is None:
            return mean, covariance
        prior = GPyTorchPosterior(model.forward(transformed))
        if hasattr(model, "outcome_transform"):
            prior = model.outcome_transform.untransform_posterior(prior, X=transformed)
        prior_variance = prior.distribution.covariance_matrix.diagonal(dim1=-2, dim2=-1)
        _finite(prior_variance)
        if (prior_variance < 0).any():
            raise _failure("negative_prior_variance")
        sd = prior_variance.sqrt()
        budget = (_RANK_MULTIPLIER * (training.shape[-2] + X.shape[-2]) * _EPS
                  * sd.unsqueeze(-1) * sd.unsqueeze(-2))
        _finite(budget)
        if (skew > budget).any():
            raise _failure("asymmetric_covariance", maximum_residual=skew.max().item(),
                           maximum_budget=budget.max().item())
        if zero_entries is not None:
            residual = torch.where(zero_entries, covariance.detach().abs(), 0.)
            if (residual > budget).any():
                raise _failure("deterministic_covariance_residual",
                               maximum_residual=residual.max().item(),
                               maximum_budget=budget.max().item())
    # This projects only validated roundoff onto the same symmetric target;
    # gradients follow the symmetric covariance, not a detached value repair.
    if has_skew:
        covariance = (covariance + covariance.mT) / 2
    if zero_entries is None:
        return mean, covariance
    # A known value is deterministic, but its derivative covariance with an
    # unobserved query need not vanish. Correct values without masking gradients.
    return mean, covariance - torch.where(zero_entries, covariance, 0.).detach()


def _psd(matrix):
    """Validate PSD without noise; retain eigenvalues above relative roundoff."""
    _finite(matrix)
    n = matrix.shape[-1]
    symmetry_tol = _RANK_MULTIPLIER * n * _EPS * matrix.abs().amax()
    if (matrix - matrix.mT).abs().amax() > symmetry_tol:
        raise _failure("asymmetric_covariance")
    symmetric = (matrix + matrix.mT) / 2
    values, vectors = torch.linalg.eigh(symmetric)
    scale = values.abs().amax()
    tolerance = _RANK_MULTIPLIER * n * _EPS * scale
    if values.min() < -tolerance:
        raise _failure("negative_covariance", minimum_eigenvalue=values.min().item(),
                       tolerance=tolerance.item())
    return symmetric, values, vectors, values > tolerance, tolerance


@dataclass(frozen=True)
class _LaplaceState:
    whitening: Tensor
    inverse_whitening: Tensor
    nullspace: Tensor
    mode: Tensor
    hessian_cholesky: Tensor
    covariance_scale: Tensor
    _diagnostics: dict = field(repr=False)

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)


def _fit_directional_laplace(mean, covariance, signs, scales, weights):
    """Fit D=m+B u, p(u) proportional to N(u;0,I) prod Phi(s D/a)^w.

    The returned frozen state owns detached tensors. ``whitening`` is B,
    ``inverse_whitening`` is U Lambda^-1/2, and H=L L^T uses the stored
    ``hessian_cholesky``. No arbitrary variance floor or score clipping is used.
    """
    try:
        return _fit(mean, covariance, signs, scales, weights)
    except EngineError:
        raise
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        raise _failure("laplace_numerics", error=str(exc)) from exc


def _fit(mean, covariance, signs, scales, weights):
    inputs = (mean, covariance, signs, scales, weights)
    if any(not isinstance(v, Tensor) or v.dtype != torch.double or v.device.type != "cpu"
           for v in inputs):
        raise _failure("expected_cpu_double")
    mean, covariance, signs, scales, weights = (v.detach().clone() for v in inputs)
    n = mean.numel()
    if (mean.shape != (n,) or n == 0 or covariance.shape != (n, n)
            or any(v.shape != (n,) for v in (signs, scales, weights))):
        raise _failure("invalid_shapes")
    _finite(mean, covariance, signs, scales, weights)
    if (not ((signs == 1) | (signs == -1)).all() or (scales <= 0).any()
            or (weights < 0).any() or (weights > 1).any()):
        raise _failure("invalid_factor")
    _, eigenvalues, eigenvectors, keep, tolerance = _psd(covariance)
    roots = eigenvalues[keep].sqrt()
    B = eigenvectors[:, keep] * roots
    W = eigenvectors[:, keep] / roots
    rank = len(roots)
    identity = torch.eye(rank, dtype=torch.double)

    def evaluate(u):
        with torch.enable_grad():
            d = (mean + B @ u).detach().requires_grad_(True)
            nll = -weights * torch.special.log_ndtr(signs*d/scales)
            score = torch.autograd.grad(nll.sum(), d, create_graph=True)[0]
            curvature = torch.autograd.grad(score.sum(), d)[0]
        _finite(nll, score, curvature)
        if (curvature < 0).any():
            raise _failure("negative_likelihood_curvature")
        value = .5*u.dot(u) + nll.sum().detach()
        gradient = u + B.T @ score.detach()
        hessian = identity + B.T @ (curvature.detach()[:, None] * B)
        _finite(value, gradient, hessian)
        return value, gradient, hessian, nll.detach(), curvature.detach()

    last_x, last_evaluation = None, None

    def scipy_evaluate(x):
        nonlocal last_x, last_evaluation
        if last_x is None or not np.array_equal(x, last_x):
            last_x = x.copy()
            last_evaluation = evaluate(torch.from_numpy(x.copy()))
        return last_evaluation

    mode = torch.zeros(rank, dtype=torch.double)
    iterations, solver_message = 0, "deterministic zero-rank likelihood"
    if rank:
        result = optimize.minimize(
            lambda x: scipy_evaluate(x)[0].item(), np.zeros(rank),
            jac=lambda x: scipy_evaluate(x)[1].numpy(),
            hess=lambda x: scipy_evaluate(x)[2].numpy(), method="trust-exact",
            options={"gtol": _GTOL, "maxiter": _MAXITER})
        if not result.success:
            raise _failure("nonconvergence", solver="trust-exact", status=int(result.status),
                           iterations=int(result.nit), message=str(result.message))
        mode = torch.from_numpy(result.x.copy())
        iterations, solver_message = int(result.nit), str(result.message)
    value, gradient, hessian, nll, curvature = evaluate(mode)
    gradient_norm = gradient.abs().max().item() if rank else 0.
    if gradient_norm > 1e-8:
        raise _failure("nonconvergence", gradient_inf_norm=gradient_norm)
    factor = torch.linalg.cholesky(hessian)
    diagnostics = {
        "approximation": "tempered_probit_laplace", "rank": rank, "pairs": n,
        "rank_tolerance": tolerance.item(), "minimum_eigenvalue": eigenvalues.min().item(),
        "maximum_eigenvalue": eigenvalues.max().item(),
        "retained_condition_number": (eigenvalues[keep].max()/eigenvalues[keep].min()).item() if rank else None,
        "solver": "trust-exact" if rank else "none", "converged": True,
        "iterations": iterations, "solver_message": solver_message,
        "gtol": _GTOL, "maxiter": _MAXITER, "gradient_inf_norm": gradient_norm,
        "negative_log_posterior": value.item(), "negative_log_factors": nll.tolist(),
        "mode_scores": (signs*(mean+B@mode)/scales).tolist(),
        "likelihood_curvature": curvature.tolist(),
        "hessian_condition_number": torch.linalg.cond(hessian).item() if rank else None,
    }
    return _LaplaceState(B, W, eigenvectors[:, ~keep], mode, factor,
                         eigenvalues.abs().max(), diagnostics)


class _FitBuffers(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        for name in ("inverse_whitening", "nullspace", "mode", "hessian_cholesky", "covariance_scale"):
            self.register_buffer(name, getattr(state, name).detach().clone())


def _covariance_root(covariance):
    """Differentiable no-jitter root, including exact repeated queries."""
    with torch.no_grad():
        _, values, vectors, keep, _ = _psd(covariance.detach())
        rank = int(keep.sum())
    q = covariance.shape[-1]
    if rank == q:
        return torch.linalg.cholesky(covariance)
    if not rank:
        return covariance * 0
    # QR only selects coordinates; differentiable reconstruction uses the full
    # covariance, avoiding unstable eigenvector derivatives at repeated queries.
    _, _, pivots = linalg.qr(vectors[:, keep].detach().cpu().numpy().T, pivoting=True)
    selected = torch.as_tensor(pivots[:rank].copy(), device=covariance.device, dtype=torch.long)
    columns = covariance[:, selected]
    factor = torch.linalg.cholesky(columns[selected, :])
    root = torch.linalg.solve_triangular(factor, columns.T, upper=False).T
    residual = (covariance-root@root.T).abs().max()
    tolerance = _CHECK_MULTIPLIER * q * _EPS * values.abs().max()
    if residual > tolerance:
        raise _failure("root_reconstruction", residual=residual.item(), tolerance=tolerance.item())
    return torch.cat([root, root.new_zeros(q, q-rank)], dim=-1)


class _DirectionalPosterior(GPyTorchPosterior):
    @property
    def variance(self):
        # GPyTorch's default marginal variance floor invents variance at rank zero.
        return self.distribution.lazy_covariance_matrix.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)


class _DirectionalModel(Model):
    def __init__(self, source, endpoints, states):
        super().__init__()
        self._source = source
        self.register_buffer("_endpoints", endpoints.detach().clone().flatten(0, 1))
        self._fits = torch.nn.ModuleList(_FitBuffers(state) for state in states)

    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return self._source.batch_shape

    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None):
        if output_indices not in (None, [0]) or observation_noise is not False:
            raise EngineError(ErrorCode.CONFIG_INVALID, "Directional adapter supports one latent response")
        try:
            posterior = self._posterior(X)
        except EngineError:
            raise
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            raise _failure("query_numerics", error=str(exc)) from exc
        return posterior if posterior_transform is None else posterior_transform(posterior)

    def _posterior(self, X):
        if X.dtype != torch.double:
            raise _failure("expected_double_query")
        endpoints = self._endpoints.expand(X.shape[:-2] + self._endpoints.shape)
        mean, covariance = _joint_latent_moments(self._source, torch.cat([X, endpoints], dim=-2))
        q, n = X.shape[-2], len(self._endpoints)//2
        cross = covariance[..., :q, q+1::2] - covariance[..., :q, q::2]
        batch_shape = mean.shape[:-1]
        base_indices = torch.arange(len(self._fits)).reshape(self.batch_shape).expand(batch_shape).flatten().tolist()
        means, roots = [], []
        for index, C, c, m in zip(base_indices, covariance.reshape(-1,q+2*n,q+2*n),
                                   cross.reshape(-1,q,n), mean.reshape(-1,q+2*n)):
            state = self._fits[index]
            Cxx = C[:q,:q]
            _psd(Cxx.detach())
            if state.nullspace.shape[-1]:
                residual = (c @ state.nullspace).abs().max()
                tolerance = (_CHECK_MULTIPLIER * max(n,q) * _EPS
                             * (Cxx.diag().max().clamp_min(0)*state.covariance_scale).sqrt())
                if residual > tolerance:
                    raise _failure("nullspace_cross_covariance", residual=residual.item(), tolerance=tolerance.item())
            A = c @ state.inverse_whitening
            conditional = Cxx - A @ A.T
            # Schur residuals are subtraction of large terms: validate against
            # the parent query scale, not the potentially zero residual scale.
            tolerance = _RANK_MULTIPLIER * q * _EPS * torch.linalg.eigvalsh(Cxx.detach()).abs().max()
            if torch.linalg.eigvalsh(((conditional+conditional.T)/2).detach()).min() < -tolerance:
                raise _failure("inconsistent_retained_covariance")
            corrected = conditional + A @ torch.cholesky_solve(A.T, state.hessian_cholesky)
            corrected = (corrected+corrected.T)/2
            means.append(m[:q] + A @ state.mode)
            roots.append(_covariance_root(corrected))
        result_mean = torch.stack(means).reshape(batch_shape+(q,))
        result_root = torch.stack(roots).reshape(batch_shape+(q,q))
        return _DirectionalPosterior(MultivariateNormal(result_mean, RootLinearOperator(result_root)))


def condition_directional_posterior(model: Model, pairs: DirectionalPairs) -> tuple[Model, dict]:
    """Snapshot active pairs and condition a fixed scalar latent posterior.

    Empty evidence returns the identical model without evaluation, optimization,
    or RNG use. Diagnostics are detached and include per-base-batch fit details.
    The caller must keep the fitted source model fixed for this adapter's lifetime.
    """
    diagnostics = {"status": "inactive", "approximation": "tempered_probit_laplace",
                   "pair_diagnostics": pairs.diagnostics, "fits": []}
    if not len(pairs.endpoints):
        return model, diagnostics
    if model.num_outputs != 1:
        raise EngineError(ErrorCode.CONFIG_INVALID, "Directional conditioning requires one response")
    endpoints = pairs.endpoints.detach().clone()
    signs, widths, softness, weights = (v.detach().clone() for v in
                                        (pairs.signs, pairs.widths, pairs.softness, pairs.weights))
    try:
        _finite(endpoints, signs, widths, softness, weights)
        n = len(endpoints)
        if (endpoints.ndim != 3 or endpoints.shape[1] != 2 or endpoints.shape[2] == 0
                or any(v.shape != (n,) for v in (signs, widths, softness, weights))
                or len(pairs.pattern_ids) != n or (widths <= 0).any() or (softness <= 0).any()
                or not ((signs == 1) | (signs == -1)).all()
                or (weights < 0).any() or (weights > 1).any()):
            raise _failure("invalid_pairs")
        active = weights > 0
        if not active.any():
            return model, diagnostics
        endpoints, signs, widths, softness, weights = (
            v[active] for v in (endpoints, signs, widths, softness, weights))
        pattern_ids = [pid for pid, keep in zip(pairs.pattern_ids, active.tolist()) if keep]
        with torch.no_grad():
            mean, C = _joint_latent_moments(model, endpoints.flatten(0,1))
            n = len(endpoints)
            dm = mean[...,1::2] - mean[...,::2]
            K = C[...,1::2,1::2] - C[...,1::2,::2] - C[...,::2,1::2] + C[...,::2,::2]
            states = []
            for index in product(*(range(size) for size in model.batch_shape)):
                _psd(C[index])
                states.append(_fit_directional_laplace(dm[index], K[index], signs, widths*softness, weights))
        diagnostics.update(status="active", fits=[state.diagnostics for state in states],
                           pattern_ids=pattern_ids, endpoints=endpoints.tolist(),
                           signs=signs.tolist(), widths=widths.tolist(), softness=softness.tolist(),
                           weights=weights.tolist())
        return _DirectionalModel(model, endpoints, states), diagnostics
    except EngineError:
        raise
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        raise _failure("conditioning_numerics", error=str(exc)) from exc


__all__ = ["condition_directional_posterior"]
