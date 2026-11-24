"""Actual affine arithmetic must guard moments, samples and derivative products."""
import pytest
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

from expdoe_dk.bo import directional_source as source_module
from expdoe_dk.errors import EngineError
from .test_posterior import tensor
from .test_source_adapter import adapter
from .test_posterior import training_field, pairs, engine
from expdoe_dk.bo.surrogates import _PhysicalAcquisitionModel
from expdoe_dk.bo.surrogates import FittedSurrogate
from expdoe_dk import Space, Parameter


def affine(*args, **kwargs):
    assert hasattr(source_module, "checked_affine"), "affine checks not implemented"
    return source_module.checked_affine(*args, **kwargs)


def test_affine_forward_rejects_nonfinite_intermediate():
    with pytest.raises(EngineError) as caught:
        affine(tensor([2.]), 1e308, -1e308)
    assert caught.value.details["reason"] == "source_output_range"


def test_affine_backward_and_double_backward_guard_scale_products():
    x = tensor([1e-300]).requires_grad_()
    y = affine(x, 1e300)
    with pytest.raises(EngineError) as caught:
        torch.autograd.grad(y*1e100, x)
    assert caught.value.details["reason"] == "source_output_range"
    y = affine(x, 1e300)
    weight = tensor([1e-300]).requires_grad_()
    grad, = torch.autograd.grad(y*weight, x, create_graph=True)
    with pytest.raises(EngineError) as caught:
        torch.autograd.grad(grad*1e100, weight)
    assert caught.value.details["reason"] == "source_output_range"


def test_affine_samples_preserve_actual_operation_and_rng():
    samples = tensor([[-.3, .2], [.7, -.4]])
    rng = torch.random.get_rng_state().clone()
    actual = affine(samples, -3., 1.)
    assert torch.equal(actual, samples*(-3.)+1.)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert affine(tensor([.5]), 2., -1.).item() == 0.


def test_actual_standardize_source_and_chain_rule():
    X, Y = tensor([[0.], [.3], [.7], [1.]]), tensor([[.5], [.2], [-.3], [-.1]])
    model = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, .03), outcome_transform=Standardize(m=1))
    query = tensor([[.2], [.65]]).requires_grad_()
    raw = model.posterior(query)
    expected_mean, expected_covariance = raw.mean[..., 0].detach(), raw.distribution.covariance_matrix.detach()
    source = adapter(model)
    mean, covariance = source.joint_moments(query)
    torch.testing.assert_close(mean, expected_mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(covariance, expected_covariance, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(source.joint_moments, (query,), eps=1e-5, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(source.joint_moments, (query,), eps=1e-5, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("scale", [1e200, 1e-200])
@pytest.mark.parametrize("view", ["acquisition", "predict"])
def test_physical_adapter_rejects_scale_squared_range_before_sampling(scale, view):
    base, _ = training_field(noise=.03)
    model, _ = engine().condition_directional_posterior(base, pairs())
    physical = _PhysicalAcquisitionModel(model, 0., scale, 1.)
    with pytest.raises(EngineError) as caught:
        if view == "acquisition":
            physical.posterior(tensor([[.2], [.65]])).variance
        else:
            fitted = FittedSurrogate(Space([Parameter("x", bounds=(0., 1.))], objectives="y"),
                                      model, 0., scale, 1., {}, ())
            fitted.predict(pd.DataFrame({"x": [.2, .65]}))
    assert caught.value.details["reason"] == "source_output_range"


def test_physical_sample_backward_checks_source_and_response_state():
    base, _ = training_field(noise=.03)
    model, _ = engine().condition_directional_posterior(base, pairs())
    physical = _PhysicalAcquisitionModel(model, .2, 3., -1.)
    X = tensor([[.2], [.65]]).requires_grad_()
    posterior = physical.posterior(X)
    samples = posterior.rsample_from_base_samples(torch.Size([2]), tensor([[.2, -.3], [.5, .7]]))
    physical._location = .3
    with pytest.raises(EngineError) as caught:
        torch.autograd.grad(samples.sum(), X)
    assert caught.value.details["reason"] == "source_state_changed"


def test_actual_affine_signed_subnormals_and_cancelled_underflow_use_local_budget():
    assert affine(tensor([1e-300]), 1e-10).item() == 1e-310
    assert affine(tensor([1e-300]), 1e-30, 1.).item() == 1.
    assert affine(tensor([1e-300]), 1e-30, 1e-300).item() == 1e-300
    with pytest.raises(EngineError) as caught:
        affine(tensor([1e-300]), 1e-30)
    assert caught.value.details["reason"] == "source_precision_exhausted"
    with pytest.raises(EngineError) as caught:
        affine(tensor([1e-300]), 1e-10, variance=True)
    assert caught.value.details["reason"] == "source_output_range"


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_actual_outcome_variance_range_fails_in_its_own_frame(scale):
    X, Y = tensor([[0.], [.3], [.7], [1.]]), tensor([[.5], [.2], [-.3], [-.1]])
    model = SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, .03), outcome_transform=Standardize(m=1))
    model.outcome_transform.stdvs.fill_(scale)
    model.outcome_transform._stdvs_sq.fill_(scale*scale)
    prepared = adapter(model)
    with pytest.raises(EngineError) as caught:
        prepared.joint_moments(tensor([[.2], [.65]]))
    assert caught.value.details["reason"] == "source_output_range"
