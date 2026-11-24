"""Required local compositions must actually select the repaired source."""
import numpy as np
import pandas as pd
import pytest
import torch
from botorch import settings

from expdoe_dk import Knowledge, KnowledgeScope, Parameter, Space, make_pattern_spec
from expdoe_dk.bo.gp import build_gp
from expdoe_dk.bo.surrogates import _PhysicalFeatures
from expdoe_dk.bo.directional_source_state import eligible
from .test_source_adapter import adapter
from .test_posterior import tensor


def composed_model(pattern="power_law"):
    X = pd.DataFrame({"x": np.tile([.05, .2, .4, .6, .8, .95], 2),
                      "category": ["a"] * 6 + ["b"] * 6})
    space = Space([Parameter("x", bounds=(.01 if pattern == "power_law" else 0., 1.)),
                   Parameter("category", kind="categorical", values=("a", "b"))], objectives="gain")
    knowledge = Knowledge().with_gp_prior("weak")
    knowledge.add_spec(make_pattern_spec(pattern=pattern, version="1.0" if pattern == "power_law" else "2.0",
        pattern_id="mean", confidence=1., scope=KnowledgeScope(factors=("x",), objectives=("gain",)),
        parameters={"scale": 2., "exponent": 1.} if pattern == "power_law" else
        {"direction": "increasing", "origin": 0., "half_response": .5}))
    knowledge.add_spec(make_pattern_spec(pattern="category_similarity", version="1.0", pattern_id="kernel",
        confidence=1., scope=KnowledgeScope(factors=("category",)),
        parameters={"levels": ["a", "b"], "matrix": [[1., .3], [.3, 1.]]}))
    features = _PhysicalFeatures(space, X)
    training = space.physical_to_model(X)
    Y = torch.sin(training[:, :1]*3)
    with settings.validate_input_scaling(False):
        model, _ = build_gp(space, knowledge, training, Y, response_location=0., response_scale=1.,
            train_Yvar=torch.full_like(Y, .03), feature_transform=features, feature_groups=features.groups,
            normalize_outcome=False)
    assert eligible(model), "mandatory builder composition was routed away from MP"
    return model


@pytest.mark.parametrize("pattern", ["power_law", "saturation"])
def test_registered_mean_similarity_and_prior_use_actual_mp_adapter(pattern):
    model = composed_model(pattern)
    source = adapter(model)
    continuous = torch.tensor([.25, .65], dtype=torch.double, requires_grad=True)
    def values(x):
        query = torch.stack((x, x.new_tensor([0., 1.])), -1)
        return source.joint_moments(query)
    assert torch.autograd.gradcheck(values, (continuous,), eps=1e-5, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(values, (continuous,), eps=1e-5, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("transform", ["normalize", "standardize", "log_chain", "reverse", "query_only"])
@pytest.mark.parametrize("batched", [False, True])
def test_actual_input_transform_frames_and_independent_batches(transform, batched):
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize, InputStandardize, Log10, ChainedInputTransform
    from gpytorch.kernels import RBFKernel, ScaleKernel
    batch = torch.Size([2]) if batched else torch.Size()
    X = tensor([[.1, .3], [.25, .8], [.6, .2], [.9, .7]])
    Y = tensor([[.5], [.2], [-.3], [-.1]])
    if batched:
        X, Y = X.expand(2, 4, 2), Y.expand(2, 4, 1).clone()
        Y[1] *= .7
    if transform == "standardize":
        tf = InputStandardize(d=2, batch_shape=batch)
    elif transform == "log_chain":
        tf = ChainedInputTransform(log=Log10(indices=[0]), norm=Normalize(d=2, batch_shape=batch))
    else:
        tf = Normalize(d=2, bounds=tensor([[0., 0.], [2., 3.]]),
                       reverse=transform == "reverse", transform_on_train=transform != "query_only")
    kernel = ScaleKernel(RBFKernel(ard_num_dims=2, batch_shape=batch), batch_shape=batch)
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, .03), covar_module=kernel,
                             input_transform=tf, outcome_transform=None)
    query = tensor([[[[.2, .4], [.7, .6]]], [[[.3, .5], [.8, .4]]]]).requires_grad_()
    reference = model.posterior(query)
    rm, rc = reference.mean[..., 0], reference.distribution.covariance_matrix
    rg, = torch.autograd.grad(rm.sum()+rc.sum(), query)
    source = adapter(model)
    m, C = source.joint_moments(query)
    torch.testing.assert_close(m, rm, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(C, rc, atol=1e-11, rtol=1e-11)
    g, = torch.autograd.grad(m.sum()+C.sum(), query, create_graph=True)
    torch.testing.assert_close(g, rg, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(torch.autograd.grad(g.square().sum(), query)[0]).all()
    assert m.shape == (2, 2 if batched else 1, 2)


@pytest.mark.parametrize("mean_kind", ["zero", "arrhenius", "quadratic", "combined"])
def test_registered_mean_modules_keep_their_torch_chain_rules(mean_kind):
    from botorch.models import SingleTaskGP
    from gpytorch.means import ZeroMean
    from expdoe_dk.knowledge.shape import ArrheniusMeanFrozen, QuadraticMeanFrozen, CombinedMean
    means = {"zero": ZeroMean(), "arrhenius": ArrheniusMeanFrozen(),
             "quadratic": QuadraticMeanFrozen(1, [1.], [.4]),
             "combined": CombinedMean(ArrheniusMeanFrozen(), QuadraticMeanFrozen(1, [-1.], [.4]))}
    X, Y = tensor([[.1], [.3], [.7], [.95]]), tensor([[.5], [.2], [-.3], [-.1]])
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, .03), mean_module=means[mean_kind], outcome_transform=None)
    source = adapter(model)
    query = tensor([[.2], [.6]]).requires_grad_()
    assert torch.autograd.gradcheck(source.joint_moments, (query,), eps=1e-5, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(source.joint_moments, (query,), eps=1e-5, atol=1e-7, rtol=1e-5)


def test_fixed_plus_learned_noise_uses_total_without_changing_original_flags():
    from botorch.models import SingleTaskGP
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
    X, Y = tensor([[.1], [.3], [.7], [.95]]), tensor([[.5], [.2], [-.3], [-.1]])
    likelihood = FixedNoiseGaussianLikelihood(noise=tensor([.01, .02, .03, .04]), learn_additional_noise=True).double()
    likelihood.second_noise_covar.noise = .07
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=None)
    flags = [p.requires_grad for p in model.parameters()]
    expected_noise = model.likelihood.noise.detach().clone()
    source = adapter(model)
    assert source.blocks[0].noise == expected_noise.tolist()
    assert flags == [p.requires_grad for p in model.parameters()]
    assert torch.equal(model.likelihood.noise, expected_noise)


def test_mixed_physical_feature_frames_retain_real_chain_rules():
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from expdoe_dk.bo.gp import _FeatureKernel
    origin = 2**60
    space = Space([Parameter("log", bounds=(1., 100.), transform="log"),
                   Parameter("integer", kind="integer", bounds=(origin, origin+1000)),
                   Parameter("nominal", kind="categorical", values=("a", "b")),
                   Parameter("ordered", kind="ordinal", values=("low", "mid", "high"))], objectives="gain")
    frame = pd.DataFrame({"log": [2., 5., 20., 80.], "integer": [origin+x for x in (10, 250, 600, 900)],
                          "nominal": ["a", "b", "a", "b"], "ordered": ["low", "mid", "high", "low"]})
    features = _PhysicalFeatures(space, frame)
    X, Y = space.physical_to_model(frame), tensor([[.1], [.4], [-.3], [.2]])
    kernel = _FeatureKernel(ScaleKernel(RBFKernel(ard_num_dims=5)), features)
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, .03), covar_module=kernel, outcome_transform=None)
    assert eligible(model)
    query = tensor([[.3, .4, 0., .5], [.6, .7, 1., 1.]])
    reference = model.posterior(query)
    source = adapter(model)
    m, C = source.joint_moments(query)
    torch.testing.assert_close(m, reference.mean[..., 0], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(C, reference.distribution.covariance_matrix, atol=1e-11, rtol=1e-11)
    numeric = query[:, :2].clone().requires_grad_()
    def values(z):
        return source.joint_moments(torch.cat((z, query[:, 2:]), -1))
    assert torch.autograd.gradcheck(values, (numeric,), eps=1e-5, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(values, (numeric,), eps=1e-5, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("nested", [False, True])
def test_scale_active_dimension_order_matches_native_kernel_calls(nested):
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import RBFKernel, ScaleKernel
    X, Y = tensor([[.1, .3], [.25, .8], [.6, .2], [.9, .7]]), tensor([[.5], [.2], [-.3], [-.1]])
    kernel = ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[1, 0]), active_dims=[1, 0])
    kernel.base_kernel.lengthscale = tensor([.4, 1.3])
    if nested:
        kernel = ScaleKernel(kernel, active_dims=[1, 0])
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, .03), covar_module=kernel, outcome_transform=None)
    query = tensor([[.2, .4], [.7, .6]]).requires_grad_()
    reference = model.posterior(query)
    rm, rc = reference.mean[..., 0], reference.distribution.covariance_matrix
    rg, = torch.autograd.grad(rm.sum()+rc.sum(), query)
    source = adapter(model)
    m, C = source.joint_moments(query)
    torch.testing.assert_close(m, rm, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(C, rc, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(torch.autograd.grad(m.sum()+C.sum(), query)[0], rg, atol=1e-10, rtol=1e-10)


def test_scalar_outcome_chain_and_learned_noise_use_actual_adapter():
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize, ChainedOutcomeTransform
    X, Y = tensor([[.1], [.3], [.7], [.95]]), tensor([[5.], [2.], [-3.], [-1.]])
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(X, Y, outcome_transform=ChainedOutcomeTransform(
            first=Standardize(m=1), second=Standardize(m=1)))
    query = tensor([[.2], [.6]]).requires_grad_()
    reference = model.posterior(query)
    source = adapter(model)
    m, C = source.joint_moments(query)
    torch.testing.assert_close(m, reference.mean[..., 0], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(C, reference.distribution.covariance_matrix, atol=1e-11, rtol=1e-11)
    assert torch.autograd.gradcheck(source.joint_moments, (query,), eps=1e-5, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(source.joint_moments, (query,), eps=1e-5, atol=1e-7, rtol=1e-5)


def test_category_only_source_retains_zero_continuous_derivatives():
    from botorch.models import SingleTaskGP
    from expdoe_dk.bo.gp import _CategorySimilarityKernel
    kernel = next(module for module in composed_model().modules() if type(module) is _CategorySimilarityKernel)
    X, Y = tensor([[.1, 0.], [.3, 1.], [.7, 0.], [.95, 1.]]), tensor([[.5], [.2], [-.3], [-.1]])
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, .03), covar_module=kernel, outcome_transform=None)
    query = tensor([[.2, 0.], [.6, 1.]]).requires_grad_()
    reference = model.posterior(query)
    source = adapter(model)
    m, C = source.joint_moments(query)
    torch.testing.assert_close(m, reference.mean[..., 0], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(C, reference.distribution.covariance_matrix, atol=1e-11, rtol=1e-11)
    g, = torch.autograd.grad(m.sum()+C.sum(), query, create_graph=True)
    assert torch.equal(g, torch.zeros_like(query))
    gg, = torch.autograd.grad(g.sum(), query)
    assert torch.equal(gg, torch.zeros_like(query))


def test_multiple_category_factors_and_nested_scale_keep_joint_covariance():
    from copy import deepcopy
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import ScaleKernel
    from expdoe_dk.bo.gp import _CategorySimilarityKernel
    original = composed_model()
    category = deepcopy(next(module for module in original.modules() if type(module) is _CategorySimilarityKernel))
    category.active_dims = torch.tensor([2], dtype=torch.long)
    kernel = ScaleKernel(original.covar_module * category)
    X = tensor([[.1, 0., 0.], [.3, 1., 0.], [.7, 0., 1.], [.95, 1., 1.]])
    Y = tensor([[.5], [.2], [-.3], [-.1]])
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, .03), covar_module=kernel, outcome_transform=None)
    query = tensor([[.2, 0., 1.], [.6, 1., 0.]])
    reference = model.posterior(query)
    source = adapter(model)
    m, C = source.joint_moments(query)
    torch.testing.assert_close(m, reference.mean[..., 0], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(C, reference.distribution.covariance_matrix, atol=1e-11, rtol=1e-11)
    numeric = query[:, :1].clone().requires_grad_()
    def values(z):
        return source.joint_moments(torch.cat((z, query[:, 1:]), -1))
    assert torch.autograd.gradcheck(values, (numeric,), eps=1e-5, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(values, (numeric,), eps=1e-5, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("pattern", ["threshold", "synergy", "antagonism"])
def test_remaining_structured_mean_formulas_keep_owned_query_derivatives(pattern):
    from botorch.models import SingleTaskGP
    from tests.phase_one.test_structured_mean import case
    _, _, mean = case(pattern)
    X, Y = tensor([[.1, .3], [.25, .8], [.6, .2], [.9, .7]]), tensor([[.5], [.2], [-.3], [-.1]])
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, .03), mean_module=mean, outcome_transform=None)
    source = adapter(model)
    query = tensor([[.3, .6]]).requires_grad_()
    assert torch.autograd.gradcheck(source.joint_moments, (query,), eps=1e-5, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(source.joint_moments, (query,), eps=1e-5, atol=1e-7, rtol=1e-5)
