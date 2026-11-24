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


def test_affine_variance_accepts_exact_maximum_and_certified_zero():
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    assert affine(tensor([maximum, 0.]), 1., variance=True).tolist() == [maximum, 0.]
    assert affine(tensor([0.]), maximum, variance=True).item() == 0.


@pytest.mark.parametrize("stage", ["product", "intermediate", "result"])
@pytest.mark.parametrize("route", ["auditor", "torch_adapter"])
def test_affine_variance_rejects_mp_above_maximum_despite_finite_cast(stage, route):
    import mpmath
    from expdoe_dk.bo.directional_source_arithmetic import audit_affine
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    value, scale, offset = (maximum, 1., 2.**969) if stage == "result" else (
        3.5953862697246315e307, 5., -1e292 if stage == "intermediate" else 0.)
    product, actual = value*scale, value*scale+offset
    c = mpmath.mp.clone()
    c.dps = 100
    mp_product = c.mpf(value)*c.mpf(scale)
    mp_result = mp_product+c.mpf(offset)
    assert product == maximum and actual <= maximum
    assert (mp_result if stage == "result" else mp_product) > c.mpf(maximum)
    if stage == "intermediate":
        assert mp_result < c.mpf(maximum)
    def evaluate(variance):
        if route == "auditor":
            return audit_affine([value], [scale], [offset], [product], [actual], [0.], [variance])
        return affine(tensor([value]), scale, offset, variance=True if variance else None)
    # The same finite signed maps remain valid under the local absolute budget.
    evaluate(False)
    with pytest.raises(EngineError) as caught:
        evaluate(True)
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
