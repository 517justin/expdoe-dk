"""Independent mathematical references for soft directional conditioning."""
import importlib
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DenseLinearOperator
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import log_ndtr

from expdoe_dk.bo.directional_pairs import DirectionalPairs
from expdoe_dk.errors import EngineError, ErrorCode


def engine():
    return importlib.import_module("expdoe_dk.bo.directional_posterior")


def tensor(value):
    return torch.tensor(value, dtype=torch.double)


def fit(mean=-.3, variance=.4, sign=1., scale=1.2, weight=.7):
    return engine()._fit_directional_laplace(
        tensor([mean]), tensor([[variance]]), tensor([sign]), tensor([scale]), tensor([weight]))


def scalar_reference(mean, variance, sign, scale, weight):
    def mills(t):
        return np.exp(-.5*t*t - .5*np.log(2*np.pi) - log_ndtr(t))
    mode = brentq(lambda d: (d-mean)/variance-weight*sign/scale*mills(sign*d/scale),
                  mean-100*np.sqrt(variance), mean+100*np.sqrt(variance), xtol=1e-13)
    t = sign*mode/scale
    hessian = 1/variance + weight/scale**2*mills(t)*(t+mills(t))
    return mode, hessian


class GaussianField(Model):
    """Analytic joint RBF Gaussian field; no GP fit or optimizer oracle."""
    def __init__(self, offset=0., batch=False, constant=False):
        super().__init__()
        self.offset, self.batched, self.constant = offset, batch, constant

    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return torch.Size([2]) if self.batched else torch.Size()

    def posterior(self, X, **kwargs):
        x = X[..., 0]
        if self.batched:
            x = x.expand(torch.broadcast_shapes(x.shape[:-1], self.batch_shape) + x.shape[-1:])
        mean = -.35*x + self.offset
        distance = x.unsqueeze(-1)-x.unsqueeze(-2)
        covariance = torch.ones_like(distance) if self.constant else torch.exp(-distance.square()/.32)
        if self.batched:
            covariance = covariance * tensor([1., .6]).view(2, 1, 1)
        return GPyTorchPosterior(MultivariateNormal(mean, DenseLinearOperator(covariance)))


def pairs(repeat=False, empty=False, sign=1., weight=.6):
    endpoints = tensor([[[.1], [.4]], [[.55], [.9]]])
    if repeat:
        endpoints[1] = endpoints[0]
    if empty:
        endpoints = endpoints[:0]
    n = len(endpoints)
    return DirectionalPairs(endpoints, tensor([sign]*n),
                            (endpoints[:, 1, 0]-endpoints[:, 0, 0]),
                            tensor([1.2]*n), tensor([weight]*n), ("p",)*n, [{"status": "inactive" if empty else "active"}])


@pytest.mark.parametrize("mean,variance,sign,scale,weight", [
    (-.3,.4,1.,1.2,.7), (1.,2.,-1.,.4,1.), (-10.,.02,1.,.1,.5),
    (10.,.02,1.,.1,.5), (0.,1e-20,1.,1.,.5)])
def test_one_difference_laplace_matches_independent_mode_and_hessian(mean, variance, sign, scale, weight):
    state = fit(mean, variance, sign, scale, weight)
    mode, hessian = scalar_reference(mean, variance, sign, scale, weight)
    actual_mode = tensor([mean]) + state.whitening @ state.mode
    actual_variance = state.whitening @ torch.cholesky_solve(state.whitening.T, state.hessian_cholesky)
    assert actual_mode.item() == pytest.approx(mode, abs=2e-8, rel=2e-8)
    assert actual_variance.item() == pytest.approx(1/hessian, abs=2e-8, rel=2e-8)
    assert state.diagnostics["rank"] == 1
    assert state.diagnostics["converged"]
    with pytest.raises(FrozenInstanceError):
        state.mode = tensor([0.])


@pytest.mark.parametrize("mean", [-.3, 0., .3])
@pytest.mark.parametrize("weight", [.25, 1.])
def test_laplace_moments_against_quadrature_in_declared_mild_regime(mean, weight):
    state = fit(mean=mean, weight=weight)
    def density(d):
        return np.exp(-.5*(d-mean)**2/.4 + weight*log_ndtr(d/1.2))
    z = quad(density, -np.inf, np.inf, epsabs=1e-11)[0]
    exact_mean = quad(lambda d: d*density(d), -np.inf, np.inf, epsabs=1e-11)[0]/z
    exact_variance = quad(lambda d: (d-exact_mean)**2*density(d), -np.inf, np.inf, epsabs=1e-11)[0]/z
    approx_mean = mean + (state.whitening @ state.mode).item()
    approx_variance = (state.whitening @ torch.cholesky_solve(state.whitening.T, state.hessian_cholesky)).item()
    assert abs(approx_mean-exact_mean) < .035
    assert abs(approx_variance-exact_variance) < .025


def test_joint_posterior_matches_block_formula():
    base, evidence = GaussianField(), pairs()
    conditioned, diagnostics = engine().condition_directional_posterior(base, evidence)
    X = tensor([[.2], [.65], [.8]])
    joint = base.posterior(torch.cat([X, evidence.endpoints.flatten(0, 1)]))
    m, C = joint.mean[:, 0], joint.distribution.covariance_matrix
    L = tensor([[-1., 1., 0., 0.], [0., 0., -1., 1.]])
    dm, K, cross = L @ m[3:], L @ C[3:, 3:] @ L.T, C[:3, 3:] @ L.T
    # Independent optimization in unwhitened D coordinates using analytic score.
    from scipy.optimize import root
    invK = np.linalg.inv(K.numpy())  # Independent reference only, never production.
    scales, weights = (evidence.widths*evidence.softness).numpy(), evidence.weights.numpy()
    def curvature(d):
        t = d/scales
        r = np.exp(-t*t/2-np.log(2*np.pi)/2-log_ndtr(t))
        return weights*r*(t+r)/scales**2
    def score(d):
        t = d/scales
        r = np.exp(-t*t/2-np.log(2*np.pi)/2-log_ndtr(t))
        return invK @ (d-dm.numpy()) - weights*r/scales
    solution = root(score, dm.numpy(), jac=lambda d: invK+np.diag(curvature(d)), tol=1e-11)
    assert np.linalg.norm(score(solution.x)) < 1e-8
    V = tensor(np.linalg.inv(invK+np.diag(curvature(solution.x))))
    gain = torch.linalg.solve(K, cross.T).T
    expected_mean = m[:3] + gain @ (tensor(solution.x)-dm)
    expected_cov = C[:3,:3] - gain @ cross.T + gain @ V @ gain.T
    posterior = conditioned.posterior(X)
    torch.testing.assert_close(posterior.mean[:,0], expected_mean, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(posterior.distribution.covariance_matrix, expected_cov, atol=2e-8, rtol=2e-8)
    assert diagnostics["status"] == "active"


def test_repeated_pairs_preserve_total_fractional_evidence():
    repeated = pairs(repeat=True, weight=.3)
    single = DirectionalPairs(repeated.endpoints[:1], repeated.signs[:1], repeated.widths[:1],
                              repeated.softness[:1], tensor([.6]), ("p",), [])
    a, diagnostic = engine().condition_directional_posterior(GaussianField(), repeated)
    b, _ = engine().condition_directional_posterior(GaussianField(), single)
    X = tensor([[.2],[.7]])
    torch.testing.assert_close(a.posterior(X).mean, b.posterior(X).mean)
    torch.testing.assert_close(a.posterior(X).distribution.covariance_matrix, b.posterior(X).distribution.covariance_matrix)
    assert diagnostic["fits"][0]["rank"] == 1


def test_zero_rank_has_no_invented_uncertainty_or_optimizer(monkeypatch):
    module = engine()
    monkeypatch.setattr(module.optimize, "minimize", lambda *a, **k: pytest.fail("zero rank optimized"))
    state = module._fit_directional_laplace(tensor([-100.]), tensor([[0.]]), tensor([1.]), tensor([1.]), tensor([1.]))
    assert state.mode.numel() == 0
    assert state.diagnostics["rank"] == 0
    base = GaussianField(constant=True)
    adapted, _ = module.condition_directional_posterior(base, pairs())
    X = tensor([[.2], [.6]])
    torch.testing.assert_close(adapted.posterior(X).mean, base.posterior(X).mean)
    torch.testing.assert_close(adapted.posterior(X).distribution.covariance_matrix, torch.ones(2,2,dtype=torch.double))


@pytest.mark.parametrize("bad", ["negative", "nan", "asymmetric", "scale", "sign", "weight"])
def test_invalid_fit_inputs_are_typed_failures(bad):
    mean, cov, signs, scales, weights = tensor([0.,0.]), torch.eye(2,dtype=torch.double), tensor([1.,1.]), tensor([1.,1.]), tensor([.5,.5])
    if bad == "negative": cov[0,0] = -.01
    if bad == "nan": mean[0] = float("nan")
    if bad == "asymmetric": cov[0,1] = .1
    if bad == "scale": scales[0] = 0.
    if bad == "sign": signs[0] = 0.
    if bad == "weight": weights[0] = -1.
    with pytest.raises(EngineError) as caught:
        engine()._fit_directional_laplace(mean,cov,signs,scales,weights)
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED


def test_nonconvergence_is_not_plain_gp_fallback(monkeypatch):
    module = engine()
    monkeypatch.setattr(module.optimize, "minimize", lambda *a, **k: SimpleNamespace(
        success=False, x=np.zeros(1), message="iteration limit", nit=200, status=1))
    with pytest.raises(EngineError) as caught:
        fit()
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED
    assert caught.value.details["reason"] == "nonconvergence"


def test_no_active_pairs_returns_original_without_rng_or_optimizer(monkeypatch):
    module, base = engine(), GaussianField()
    monkeypatch.setattr(module.optimize, "minimize", lambda *a, **k: pytest.fail("inactive optimized"))
    rng = torch.random.get_rng_state().clone()
    adapted, diagnostic = module.condition_directional_posterior(base, pairs(empty=True))
    assert adapted is base
    assert diagnostic["status"] == "inactive"
    assert torch.equal(rng, torch.random.get_rng_state())


def test_direction_weight_softness_and_offset_mechanisms():
    positive, negative = fit(mean=0.), fit(mean=0.,sign=-1.)
    assert positive.mode.item() > 0
    torch.testing.assert_close(positive.mode, -negative.mode)
    assert fit(mean=0.,weight=.1).mode.item() < positive.mode.item()
    assert fit(mean=0.,scale=3.).mode.item() < positive.mode.item()
    X = tensor([[.2],[.7]])
    a, _ = engine().condition_directional_posterior(GaussianField(), pairs())
    b, _ = engine().condition_directional_posterior(GaussianField(offset=13.), pairs())
    torch.testing.assert_close(a.posterior(X).mean+13, b.posterior(X).mean)
    torch.testing.assert_close(a.posterior(X).distribution.covariance_matrix,b.posterior(X).distribution.covariance_matrix)


def test_pair_tensors_and_diagnostics_are_snapshots():
    evidence = pairs()
    adapted, diagnostic = engine().condition_directional_posterior(GaussianField(), evidence)
    X = tensor([[.2],[.7]])
    before = adapted.posterior(X)
    evidence.endpoints.fill_(.99)
    evidence.weights.zero_()
    evidence.signs.neg_()
    evidence.widths.fill_(10.)
    diagnostic.clear()
    after = adapted.posterior(X)
    torch.testing.assert_close(before.mean, after.mean)
    torch.testing.assert_close(before.distribution.covariance_matrix, after.distribution.covariance_matrix)


@pytest.mark.parametrize("batched", [False, True])
def test_joint_t_batches_gradients_and_samples(batched):
    adapted, _ = engine().condition_directional_posterior(GaussianField(batch=batched), pairs())
    X = tensor([[[.2],[.6],[.8]], [[.25],[.65],[.85]]]).requires_grad_()
    posterior = adapted.posterior(X)
    assert adapted.batch_shape == (torch.Size([2]) if batched else torch.Size())
    assert adapted.num_outputs == 1
    assert posterior.mean.shape == (2,3,1)
    def objective(x):
        p = adapted.posterior(x)
        return p.mean.square().sum() + p.distribution.covariance_matrix.square().sum()
    gradient = torch.autograd.grad(objective(X), X)[0]
    for i in range(X.numel()):
        a, b = X.detach().clone(), X.detach().clone()
        a.flatten()[i] += 1e-5
        b.flatten()[i] -= 1e-5
        numerical = (objective(a)-objective(b))/2e-5
        torch.testing.assert_close(gradient.flatten()[i], numerical, rtol=2e-4, atol=2e-6)
    from botorch.sampling.normal import IIDNormalSampler
    samples = IIDNormalSampler(sample_shape=torch.Size([16384]), seed=19)(posterior)
    assert samples.shape == (16384,2,3,1)
    assert torch.isfinite(torch.autograd.grad(samples.square().mean(), X)[0]).all()
    mean_error = (samples.mean(0)-posterior.mean).abs()
    assert (mean_error < 5*(posterior.variance/len(samples)).sqrt()).all()
    for i in range(2):
        C = posterior.distribution.covariance_matrix[i]
        error = torch.cov(samples[:,i,:,0].detach().T)-C
        se = ((C.diag()[:,None]*C.diag()[None,:]+C.square())/(len(samples)-1)).sqrt()
        assert (error.abs() < 5*se).all()


def test_repeated_queries_sample_identically_without_jitter():
    adapted, _ = engine().condition_directional_posterior(GaussianField(), pairs())
    X = tensor([[.2],[.2],[.7]]).requires_grad_()
    posterior = adapted.posterior(X)
    from botorch.sampling.normal import IIDNormalSampler
    samples = IIDNormalSampler(sample_shape=torch.Size([64]), seed=2)(posterior)
    torch.testing.assert_close(samples[:,0], samples[:,1], atol=1e-12, rtol=1e-12)
    assert torch.isfinite(torch.autograd.grad(samples.square().mean(),X)[0]).all()


def test_nullspace_inconsistent_query_cross_covariance_is_typed():
    class Inconsistent(GaussianField):
        def posterior(self, X, **kwargs):
            p = super().posterior(X)
            if len(X) > 4:
                C = p.distribution.covariance_matrix.clone()
                C[0,-1] += .01
                C[-1,0] += .01
                return GPyTorchPosterior(MultivariateNormal(p.mean[:,0], DenseLinearOperator(C)))
            return p
    adapted, _ = engine().condition_directional_posterior(Inconsistent(), pairs(repeat=True))
    with pytest.raises(EngineError) as caught:
        adapted.posterior(tensor([[.2],[.7]]))
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED
    assert caught.value.details["reason"] == "nullspace_cross_covariance"


def test_explicit_zero_weights_are_inactive_identity(monkeypatch):
    module, base = engine(), GaussianField()
    monkeypatch.setattr(module.optimize, "minimize", lambda *a, **k: pytest.fail("inactive optimized"))
    adapted, diagnostics = module.condition_directional_posterior(base, pairs(weight=0.))
    assert adapted is base
    assert diagnostics["status"] == "inactive"


def test_pair_vector_shapes_cannot_silently_broadcast():
    p = pairs()
    malformed = DirectionalPairs(p.endpoints,p.signs,p.widths[:1],p.softness,p.weights,p.pattern_ids,[])
    with pytest.raises(EngineError) as caught:
        engine().condition_directional_posterior(GaussianField(), malformed)
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED


def test_tiny_positive_covariance_is_retained_not_floored():
    state = fit(mean=0.,variance=1e-20)
    assert state.diagnostics["rank"] == 1
    torch.testing.assert_close(state.whitening @ state.whitening.T, tensor([[1e-20]]), rtol=1e-12,atol=0.)
    variance = state.whitening @ torch.cholesky_solve(state.whitening.T,state.hessian_cholesky)
    assert variance.item() == pytest.approx(1e-20,rel=1e-12,abs=0.)


def test_extreme_scores_remain_unclipped_and_finite():
    for mean in [-100.,-10.,0.,10.,100.]:
        state = fit(mean=mean, variance=0., scale=1.,weight=1.)
        d = state.diagnostics
        assert d["negative_log_factors"][0] == pytest.approx(-log_ndtr(mean),rel=1e-12,abs=1e-14)
        assert np.isfinite(d["likelihood_curvature"]).all()
        assert min(d["likelihood_curvature"]) >= 0


@pytest.mark.parametrize("noise", [0., .03, None])
def test_real_gp_keeps_parameters_measurements_noise_and_rng(noise):
    import random
    from botorch.models import SingleTaskGP
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
    from botorch.exceptions.warnings import InputDataWarning
    from gpytorch import settings
    X, Y = tensor([[0.],[.3],[.7],[1.]]), tensor([[.5],[.2],[-.3],[-.1]])
    variance = None if noise is None else torch.full_like(Y,noise)
    with settings.min_fixed_noise(double_value=0.):
        likelihood = None if variance is None else FixedNoiseGaussianLikelihood(noise=variance[:,0])
        with pytest.warns(InputDataWarning, match="not standardized"):
            base = SingleTaskGP(X,Y,likelihood=likelihood,outcome_transform=None)
    original_noise = base.likelihood.noise.detach().clone()
    parameters = {name:value.detach().clone() for name,value in base.named_parameters()}
    before = (torch.random.get_rng_state().clone(),np.random.get_state(),random.getstate())
    adapted, _ = engine().condition_directional_posterior(base,pairs())
    after = (torch.random.get_rng_state(),np.random.get_state(),random.getstate())
    assert torch.equal(before[0],after[0])
    np.testing.assert_equal(before[1],after[1])
    assert before[2] == after[2]
    for name,value in base.named_parameters():
        assert torch.equal(value,parameters[name])
    assert torch.equal(base.train_inputs[0],X)
    assert torch.equal(base.train_targets,Y[:,0])
    assert torch.equal(base.likelihood.noise,original_noise)
    query = tensor([[[.1],[.5]],[[.2],[.8]]]).requires_grad_()
    result = adapted.posterior(query)
    assert torch.isfinite(result.mean).all()
    assert torch.isfinite(torch.autograd.grad(result.mean.sum()+result.variance.sum(),query)[0]).all()


@pytest.mark.parametrize("kind", ["ehvi","nehvi","nparego"])
def test_all_acquisition_families_have_joint_q_gradients(kind):
    from botorch.models.model import ModelList
    from botorch.sampling.list_sampler import ListSampler
    from botorch.sampling.normal import IIDNormalSampler
    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
    from expdoe_dk.bo.surrogates import _PhysicalAcquisitionModel
    a,_ = engine().condition_directional_posterior(GaussianField(),pairs())
    b,_ = engine().condition_directional_posterior(GaussianField(),pairs(sign=-1.))
    model = ModelList(_PhysicalAcquisitionModel(a,0.,1.,1.),_PhysicalAcquisitionModel(b,0.,1.,-1.))
    sampler = ListSampler(*(IIDNormalSampler(sample_shape=torch.Size([32]),seed=s) for s in [13,19]))
    baseline = tensor([[0.],[.45],[1.]])
    if kind == "ehvi":
        partition = NondominatedPartitioning(ref_point=tensor([-3.,-3.]),Y=tensor([[.1,.2],[.2,.1]]))
        acquisition = qLogExpectedHypervolumeImprovement(model,ref_point=[-3.,-3.],partitioning=partition,sampler=sampler)
    elif kind == "nehvi":
        acquisition = qLogNoisyExpectedHypervolumeImprovement(model,ref_point=[-3.,-3.],X_baseline=baseline,
            sampler=sampler,prune_baseline=False,cache_root=False)
    else:
        scalarize = get_chebyshev_scalarization(weights=tensor([.4,.6]),Y=torch.empty(0,2,dtype=torch.double))
        acquisition = qLogNoisyExpectedImprovement(model,X_baseline=baseline,sampler=sampler,
            objective=GenericMCObjective(lambda samples,X=None: scalarize(samples)),prune_baseline=False,cache_root=False)
    X = tensor([[[.2],[.7]],[[.3],[.8]]]).requires_grad_()
    value = acquisition(X)
    gradient = torch.autograd.grad(value.sum(),X)[0]
    assert value.shape == (2,)
    assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
    for i in range(X.numel()):
        hi,lo = X.detach().clone(),X.detach().clone()
        hi.flatten()[i] += 1e-5
        lo.flatten()[i] -= 1e-5
        numerical = (acquisition(hi).sum()-acquisition(lo).sum())/2e-5
        torch.testing.assert_close(gradient.flatten()[i],numerical,rtol=2e-4,atol=2e-6)


@pytest.mark.parametrize("reason", ["nan","negative","retained"])
def test_query_invalid_covariance_does_not_fallback(reason):
    class InvalidQuery(GaussianField):
        def posterior(self,X,**kwargs):
            p = super().posterior(X)
            C = p.distribution.covariance_matrix.clone()
            if len(X)>4:
                if reason == "nan": C[0,0] = float("nan")
                if reason == "negative": C[0,0] = -1.
                if reason == "retained":
                    C[0,-1] += 10.
                    C[-1,0] += 10.
            return GPyTorchPosterior(MultivariateNormal(p.mean[:,0],DenseLinearOperator(C)))
    model,_ = engine().condition_directional_posterior(InvalidQuery(),pairs())
    with pytest.raises(EngineError) as caught:
        model.posterior(tensor([[.2],[.7]]))
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED


def test_zero_query_variance_has_exact_zero_samples_and_variance():
    class Deterministic(GaussianField):
        def posterior(self,X,**kwargs):
            p = super().posterior(X)
            return GPyTorchPosterior(MultivariateNormal(p.mean[...,0],DenseLinearOperator(p.distribution.covariance_matrix*0)))
    model,_ = engine().condition_directional_posterior(Deterministic(),pairs())
    p = model.posterior(tensor([[.2],[.7]]))
    assert torch.equal(p.variance,torch.zeros_like(p.variance))
    assert torch.equal(p.rsample(torch.Size([3])),p.mean.expand(3,2,1))


def test_posterior_transform_and_unsupported_observation_noise():
    from botorch.acquisition.objective import ScalarizedPosteriorTransform
    model,_ = engine().condition_directional_posterior(GaussianField(),pairs())
    X = tensor([[.2],[.7]])
    p = model.posterior(X)
    transformed = model.posterior(X,posterior_transform=ScalarizedPosteriorTransform(tensor([2.]),offset=3.))
    torch.testing.assert_close(transformed.mean,2*p.mean+3.)
    torch.testing.assert_close(transformed.variance,4*p.variance)
    with pytest.raises(EngineError) as caught:
        model.posterior(X,observation_noise=True)
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_extra_t_batch_axis_broadcasts_original_model_batch():
    model,_ = engine().condition_directional_posterior(GaussianField(batch=True),pairs())
    X = tensor([[[[.2],[.7]]],[[[.3],[.8]]],[[[.1],[.6]]]])
    assert model.posterior(X).mean.shape == (3,2,2,1)
    for i in range(3):
        torch.testing.assert_close(model.posterior(X).mean[i],model.posterior(X[i]).mean)


def test_maximum_pair_budget_has_independent_stationarity_and_curvature():
    n = 64
    t = torch.linspace(0,1,n,dtype=torch.double)
    # This diagonal is a declared reference prior covariance, not added fit noise.
    C = torch.exp(-(t[:,None]-t[None,:]).square()/.05)+torch.eye(n,dtype=torch.double)*.01
    state = engine()._fit_directional_laplace(-t,C,torch.ones_like(t),.2+t,torch.ones_like(t)/n)
    assert state.diagnostics["rank"] == n
    mode_d = -t+state.whitening@state.mode
    z = (mode_d/(.2+t)).numpy()
    mills = tensor(np.exp(-z*z/2-np.log(2*np.pi)/2-log_ndtr(z)))
    score = torch.linalg.solve(C,mode_d+t)-mills/(n*(.2+t))
    torch.testing.assert_close(score,torch.zeros_like(score),rtol=0.,atol=1e-8)
    curvature = mills*(tensor(z)+mills)/(n*(.2+t).square())
    H = torch.eye(n,dtype=torch.double)+state.whitening.T@(curvature[:,None]*state.whitening)
    torch.testing.assert_close(state.hessian_cholesky@state.hessian_cholesky.T,H,rtol=2e-8,atol=2e-8)


def test_discarded_tiny_eigenvalue_cannot_hide_material_query_coupling():
    class NearlySingular(Model):
        num_outputs = 1
        batch_shape = torch.Size()

        def posterior(self,X,**kwargs):
            x = X[...,0]
            features = torch.stack([(x == .4).double(),
                (x == .9).double()*1e-8+(x == 2.).double(),(x == 2.).double()],dim=-1)
            return GPyTorchPosterior(MultivariateNormal(torch.zeros_like(x),DenseLinearOperator(features@features.mT)))
    model,diagnostics = engine().condition_directional_posterior(NearlySingular(),pairs())
    assert diagnostics["fits"][0]["rank"] == 1
    with pytest.raises(EngineError) as caught:
        model.posterior(tensor([[2.]]))
    assert caught.value.details["reason"] == "nullspace_cross_covariance"


def test_solver_success_flag_is_not_sufficient_without_stationarity(monkeypatch):
    module = engine()
    monkeypatch.setattr(module.optimize,"minimize",lambda *a,**k: SimpleNamespace(
        success=True,x=np.zeros(1),nit=0,message="incorrect success",status=0))
    with pytest.raises(EngineError) as caught:
        fit()
    assert caught.value.details["reason"] == "nonconvergence"


def test_fitted_state_owns_inputs_and_diagnostic_snapshots():
    mean, covariance = tensor([-.3]),tensor([[.4]])
    state = engine()._fit_directional_laplace(mean,covariance,tensor([1.]),tensor([1.2]),tensor([.7]))
    whitening, mode = state.whitening.clone(),state.mode.clone()
    mean.fill_(100.)
    covariance.zero_()
    d = state.diagnostics
    d["mode_scores"].clear()
    assert torch.equal(state.whitening,whitening)
    assert torch.equal(state.mode,mode)
    assert len(state.diagnostics["mode_scores"]) == 1


def test_nonfinite_score_fails_explicitly_without_clipping():
    with pytest.raises(EngineError) as caught:
        fit(mean=-1.,scale=1e-320)
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED
    assert caught.value.details["reason"] == "nonfinite"


def test_noiseless_training_endpoint_factors_are_constant():
    from botorch.models import SingleTaskGP
    from botorch.exceptions.warnings import InputDataWarning
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
    from gpytorch import settings
    X, Y = tensor([[0.],[.3],[.7],[1.]]), tensor([[.5],[.2],[-.3],[-.1]])
    with settings.min_fixed_noise(double_value=0.):
        likelihood = FixedNoiseGaussianLikelihood(noise=torch.zeros(4,dtype=torch.double))
        with pytest.warns(InputDataWarning, match="not standardized"):
            base = SingleTaskGP(X,Y,likelihood=likelihood,outcome_transform=None)
    evidence = DirectionalPairs(X.reshape(2,2,1),tensor([1.,1.]),tensor([.3,.3]),
        tensor([1.2,1.2]),tensor([.5,.5]),("p","p"),[])
    covariance = base.posterior(X).distribution.covariance_matrix.detach()
    print("noiseless endpoint eigenvalues:",torch.linalg.eigvalsh(covariance).tolist())
    adapted,diagnostics = engine().condition_directional_posterior(base,evidence)
    assert diagnostics["fits"][0]["rank"] == 0
    assert diagnostics["fits"][0]["solver"] == "none"
    query = tensor([[[.15],[.55],[.85]],[[0.],[.3],[.7]]]).requires_grad_()
    actual,expected = adapted.posterior(query),base.posterior(query)
    torch.testing.assert_close(actual.mean,expected.mean,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,atol=1e-12,rtol=1e-12)
    assert torch.equal(actual.variance[1],torch.zeros_like(actual.variance[1]))
    assert torch.isfinite(torch.autograd.grad(actual.mean.sum()+actual.variance.sum(),query)[0]).all()


def training_field(*, noise=0., amplitude=None, batched=False, transformed=False):
    from botorch.models import SingleTaskGP
    from botorch.exceptions.warnings import InputDataWarning
    from botorch.models.transforms.input import Normalize
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
    from gpytorch.kernels import ScaleKernel
    from gpytorch import settings
    X, Y = tensor([[0.],[.3],[.7],[1.]]), tensor([[.5],[.2],[-.3],[-.1]])
    if batched:
        X,Y = X.expand(2,4,1),Y.expand(2,4,1)
    variance = torch.full(Y.shape[:-1],noise,dtype=torch.double)
    with settings.min_fixed_noise(double_value=0.):
        likelihood = FixedNoiseGaussianLikelihood(noise=variance)
        with pytest.warns(InputDataWarning,match="not standardized"):
            model = SingleTaskGP(X,Y,likelihood=likelihood,outcome_transform=None,
                input_transform=Normalize(d=1,bounds=tensor([[-1.],[2.]])) if transformed else None)
    if amplitude is not None:
        model.covar_module = ScaleKernel(model.covar_module,batch_shape=model.batch_shape)
        model.covar_module.outputscale = amplitude
    endpoints = tensor([[[0.],[.3]],[[.7],[1.]]])
    evidence = DirectionalPairs(endpoints,tensor([1.,1.]),tensor([.3,.3]),
        tensor([1.2,1.2]),tensor([.5,.5]),("p","p"),[])
    return model,evidence


@pytest.mark.parametrize("batched,transformed", [(False,False),(False,True),(True,True)])
def test_noiseless_endpoint_representation_preserves_query_cross_gradients(batched,transformed):
    base,evidence = training_field(batched=batched,transformed=transformed)
    model,diagnostics = engine().condition_directional_posterior(base,evidence)
    assert all(fit["rank"] == 0 for fit in diagnostics["fits"])
    X = tensor([[[[0.],[.55]]],[[[.3],[.85]]]]).requires_grad_()
    actual,expected = model.posterior(X),base.posterior(X)
    a = actual.mean.sum()+actual.distribution.covariance_matrix.sum()
    b = expected.mean.sum()+expected.distribution.covariance_matrix.sum()
    torch.testing.assert_close(torch.autograd.grad(a,X)[0],torch.autograd.grad(b,X)[0],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(actual.mean,expected.mean,atol=1e-12,rtol=1e-12)
    assert (actual.variance[...,0,:] == 0).all()


def test_noiseless_mixed_pairs_keep_uncertain_direction_and_exact_nullspace():
    base,p = training_field()
    p.endpoints[1] = tensor([[.55],[.9]])
    p.widths[1] = .35
    both,diagnostics = engine().condition_directional_posterior(base,p)
    only = DirectionalPairs(p.endpoints[1:],p.signs[1:],p.widths[1:],p.softness[1:],p.weights[1:],("p",),[])
    reference,_ = engine().condition_directional_posterior(base,only)
    assert diagnostics["fits"][0]["rank"] == 1
    X = tensor([[0.],[.2],[.3],[.8]])
    actual,expected = both.posterior(X),reference.posterior(X)
    torch.testing.assert_close(actual.mean,expected.mean,atol=2e-8,rtol=2e-8)
    torch.testing.assert_close(actual.distribution.covariance_matrix,expected.distribution.covariance_matrix,atol=2e-8,rtol=2e-8)
    assert (actual.variance[[0,2]] == 0).all()


@pytest.mark.parametrize("amplitude", [1e-20,1.,1e20])
def test_noiseless_correction_scales_with_prior_not_universal_floor(amplitude):
    base,evidence = training_field(amplitude=amplitude)
    model,diagnostics = engine().condition_directional_posterior(base,evidence)
    assert diagnostics["fits"][0]["rank"] == 0
    X = tensor([[.3],[.3+1e-6]])
    actual = model.posterior(X)
    raw_joint = base.posterior(torch.cat([X,evidence.endpoints.flatten(0,1)]))
    assert actual.variance[0].item() == 0.
    assert actual.variance[1].item() > 0.
    torch.testing.assert_close(actual.variance[1,0],raw_joint.distribution.covariance_matrix[1,1],rtol=1e-12,atol=0.)


def test_genuinely_tiny_positive_training_covariance_is_not_declared_deterministic():
    base,evidence = training_field(amplitude=1e-20,noise=1e-20)
    model,diagnostics = engine().condition_directional_posterior(base,evidence)
    assert diagnostics["fits"][0]["rank"] == 2
    X = evidence.endpoints.flatten(0,1)
    actual,expected = model.posterior(X),base.posterior(X)
    assert (actual.variance > 0).all()
    torch.testing.assert_close(actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,rtol=1e-12,atol=0.)


@pytest.mark.parametrize("where", ["endpoints","query"])
def test_known_zero_coordinates_do_not_hide_material_covariance_errors(monkeypatch,where):
    base,evidence = training_field()
    original = base.posterior
    def corrupt(X,**kwargs):
        p = original(X,**kwargs)
        C = p.distribution.covariance_matrix.clone()
        if where == "endpoints" or X.shape[-2] > 4:
            C[...,0,-1] += .01
            C[...,-1,0] += .01
        return GPyTorchPosterior(MultivariateNormal(p.mean[...,0],DenseLinearOperator(C)))
    monkeypatch.setattr(base,"posterior",corrupt)
    with pytest.raises(EngineError) as caught:
        model,_ = engine().condition_directional_posterior(base,evidence)
        model.posterior(tensor([[.2],[.8]]))
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED
    assert caught.value.details["reason"] == "deterministic_covariance_residual"


def test_noiseless_representation_is_scoped_to_each_model_batch():
    base,evidence = training_field(batched=True)
    base.likelihood.noise = tensor([[0.]*4,[.03]*4])
    model,diagnostics = engine().condition_directional_posterior(base,evidence)
    assert [fit["rank"] for fit in diagnostics["fits"]] == [0,2]
    actual = model.posterior(tensor([[[0.],[.3]],[[0.],[.3]]]))
    assert (actual.variance[0] == 0).all()
    assert (actual.variance[1] > 0).all()


@pytest.mark.parametrize("response_scale", [1e-4,1e4])
def test_noiseless_residue_budget_uses_returned_response_frame(monkeypatch,response_scale):
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
    from gpytorch import settings
    X,Y = tensor([[0.],[.3],[.7],[1.]]),tensor([[.5],[.2],[-.3],[-.1]])*response_scale
    with settings.min_fixed_noise(double_value=0.):
        likelihood = FixedNoiseGaussianLikelihood(noise=torch.zeros(4,dtype=torch.double))
        base = SingleTaskGP(X,Y,likelihood=likelihood,outcome_transform=Standardize(m=1))
    evidence = DirectionalPairs(X.reshape(2,2,1),tensor([1.,1.]),tensor([.3,.3]),
        tensor([1.2,1.2]),tensor([.5,.5]),("p","p"),[])
    model,diagnostics = engine().condition_directional_posterior(base,evidence)
    assert diagnostics["fits"][0]["rank"] == 0
    assert (model.posterior(X).variance == 0).all()
    original = base.posterior
    def corrupt(X,**kwargs):
        p = original(X,**kwargs)
        C = p.distribution.covariance_matrix.clone()
        C[...,0,0] += 1e-7*Y.std().square()
        return GPyTorchPosterior(MultivariateNormal(p.mean[...,0],DenseLinearOperator(C)))
    monkeypatch.setattr(base,"posterior",corrupt)
    with pytest.raises(EngineError) as caught:
        engine().condition_directional_posterior(base,evidence)
    assert caught.value.details["reason"] == "deterministic_covariance_residual"


@pytest.mark.parametrize("amplitude", [1e-20, 1., 1e20])
@pytest.mark.parametrize("batched,transformed", [(False, False), (False, True), (True, True)])
def test_gp_source_skew_uses_construction_scale_not_cancelled_residual(
        monkeypatch, amplitude, batched, transformed):
    base, _ = training_field(noise=1e-8*amplitude, amplitude=amplitude,
                             batched=batched, transformed=transformed)
    X = tensor([[[[0.], [.3], [.7], [1.]]], [[[0.], [.3], [.7], [1.]]]]).requires_grad_()
    original = base.posterior
    reference = original(X)
    C = reference.distribution.covariance_matrix
    expected = (C+C.mT)/2

    def roundoff(X, **kwargs):
        p = original(X, **kwargs)
        covariance = p.distribution.covariance_matrix.clone()
        # Antisymmetric, differentiable construction error; well below the
        # prior-scale budget, but above a residual-scale symmetry threshold.
        delta = amplitude*1e-14*(1 + X[..., 0, 0])
        covariance[..., 0, 1] += delta
        covariance[..., 1, 0] -= delta
        return GPyTorchPosterior(MultivariateNormal(p.mean[..., 0], DenseLinearOperator(covariance)))

    monkeypatch.setattr(base, "posterior", roundoff)
    mean, actual = engine()._joint_latent_moments(base, X)
    torch.testing.assert_close(actual, actual.mT, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=amplitude*1e-28)
    torch.testing.assert_close(mean, reference.mean[..., 0], rtol=0, atol=0)
    torch.testing.assert_close(actual.diagonal(dim1=-2, dim2=-1),
                               C.diagonal(dim1=-2, dim2=-1), rtol=0, atol=0)
    assert (actual.diagonal(dim1=-2, dim2=-1) > 0).all()
    for matrix in actual.detach().reshape(-1, 4, 4):
        assert engine()._psd(matrix)[3].sum() == 4
    weights = torch.arange(16, dtype=torch.double).reshape(4, 4)
    actual_gradient, = torch.autograd.grad((actual*weights).sum(), X)
    expected_gradient, = torch.autograd.grad((expected*weights).sum(), X)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-9, atol=amplitude*1e-12)
    assert torch.isfinite(actual_gradient).all()


@pytest.mark.parametrize("where", ["endpoints", "query"])
@pytest.mark.parametrize("bad", ["asymmetric", "negative", "nonfinite"])
@pytest.mark.parametrize("amplitude", [1e-20, 1.])
def test_gp_construction_budget_does_not_hide_material_invalid_covariance(
        monkeypatch, where, bad, amplitude):
    base, evidence = training_field(noise=.03*amplitude, amplitude=amplitude)
    if where == "query":
        model, _ = engine().condition_directional_posterior(base, evidence)
    original = base.posterior

    def corrupt(X, **kwargs):
        p = original(X, **kwargs)
        C = p.distribution.covariance_matrix.clone()
        if bad == "asymmetric":
            C[..., 0, 1] += .01*amplitude
        elif bad == "negative":
            C[..., 0, 0] = -amplitude
        else:
            C[..., 0, 1] = float("nan")
        return GPyTorchPosterior(MultivariateNormal(p.mean[..., 0], DenseLinearOperator(C)))

    monkeypatch.setattr(base, "posterior", corrupt)
    with pytest.raises(EngineError) as caught:
        if where == "endpoints":
            engine().condition_directional_posterior(base, evidence)
        else:
            model.posterior(tensor([[.2], [.8]]))
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED
    assert caught.value.details["reason"] == {
        "asymmetric": "asymmetric_covariance", "negative": "negative_covariance",
        "nonfinite": "nonfinite"}[bad]


@pytest.mark.parametrize("response_scale", [1e-4, 1e4])
def test_source_skew_budget_is_in_returned_outcome_frame(monkeypatch, response_scale):
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch import settings
    X, Y = tensor([[0.], [.3], [.7], [1.]]), tensor([[.5], [.2], [-.3], [-.1]])*response_scale
    with settings.min_fixed_noise(double_value=0.):
        base = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, 1e-8*response_scale**2),
                            outcome_transform=Standardize(m=1))
    original = base.posterior
    reference = original(X).distribution.covariance_matrix

    def roundoff(X, **kwargs):
        p = original(X, **kwargs)
        C = p.distribution.covariance_matrix.clone()
        C[..., 0, 1] += 1e-14*Y.var()
        C[..., 1, 0] -= 1e-14*Y.var()
        return GPyTorchPosterior(MultivariateNormal(p.mean[..., 0], DenseLinearOperator(C)))

    monkeypatch.setattr(base, "posterior", roundoff)
    _, actual = engine()._joint_latent_moments(base, X)
    torch.testing.assert_close(actual, actual.mT, rtol=0, atol=0)
    torch.testing.assert_close(actual, (reference+reference.mT)/2, rtol=1e-12, atol=response_scale**2*1e-28)

    def corrupt(X, **kwargs):
        p = original(X, **kwargs)
        C = p.distribution.covariance_matrix.clone()
        C[..., 0, 1] += 1e-7*Y.var()
        return GPyTorchPosterior(MultivariateNormal(p.mean[..., 0], DenseLinearOperator(C)))

    monkeypatch.setattr(base, "posterior", corrupt)
    with pytest.raises(EngineError) as caught:
        engine()._joint_latent_moments(base, X)
    assert caught.value.details["reason"] == "asymmetric_covariance"


@pytest.mark.parametrize("amplitude", [1e-20, 1., 1e20])
def test_source_skew_allowance_is_not_an_eigenvalue_tolerance(monkeypatch, amplitude):
    base, _ = training_field(noise=1e-8*amplitude, amplitude=amplitude)
    original = base.posterior

    def corrupt(X, **kwargs):
        p = original(X, **kwargs)
        C = p.distribution.covariance_matrix.clone()
        C[..., 0, :] = 0.
        C[..., :, 0] = 0.
        # This eigenvalue is below the prior construction allowance but is
        # materially negative relative to the cancelled posterior covariance.
        C[..., 0, 0] = -1e-16*amplitude
        C[..., 0, 1] = 1e-14*amplitude
        C[..., 1, 0] = -1e-14*amplitude
        return GPyTorchPosterior(MultivariateNormal(p.mean[..., 0], DenseLinearOperator(C)))

    monkeypatch.setattr(base, "posterior", corrupt)
    _, C = engine()._joint_latent_moments(base, base.train_inputs[0])
    with pytest.raises(EngineError) as caught:
        engine()._psd(C)
    assert caught.value.details["reason"] == "negative_covariance"


@pytest.mark.parametrize("noise", [0., .03])
def test_real_gp_repeated_query_samples_and_gradients_survive_source_correction(noise):
    from botorch.sampling.normal import IIDNormalSampler
    base, evidence = training_field(noise=noise)
    model, _ = engine().condition_directional_posterior(base, evidence)
    X = tensor([[[.2], [.2], [.3]], [[.8], [.8], [.7]]]).requires_grad_()
    posterior = model.posterior(X)
    samples = IIDNormalSampler(sample_shape=torch.Size([128]), seed=73)(posterior)
    torch.testing.assert_close(samples[..., 0, :], samples[..., 1, :], rtol=0, atol=1e-12)
    gradient, = torch.autograd.grad(samples.square().sum(), X)
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    if noise == 0:
        assert (posterior.variance[..., 2, :] == 0).all()
