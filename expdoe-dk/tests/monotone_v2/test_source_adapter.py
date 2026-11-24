"""Actual Torch query/mean chain rules and guards across derivative passes."""
import importlib
import importlib.util

import pytest
import torch

from expdoe_dk.errors import EngineError
from .test_posterior import training_field, tensor


def adapter(model):
    name = "expdoe_dk.bo.directional_source"
    assert importlib.util.find_spec(name) is not None, "source adapter is not implemented"
    return importlib.import_module(name)._FrozenExactGPSource(model)


def test_actual_adapter_preserves_first_and_second_query_derivatives():
    model, _ = training_field(noise=.03, transformed=True)
    source = adapter(model)
    X = tensor([[.2], [.65]]).requires_grad_()
    def values(x):
        m, C = source.joint_moments(x)
        return m, C*10
    assert torch.autograd.gradcheck(values, (X,), eps=1e-5, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(values, (X,), eps=1e-5, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("when", ["forward", "backward", "double_backward"])
@pytest.mark.parametrize("operand", [False, True])
def test_actual_adapter_guards_mutations_across_passes(when, operand):
    model, _ = training_field(noise=.03)
    source = adapter(model)
    X = tensor([[.2], [.65]]).requires_grad_()
    if when != "forward":
        m, C = source.joint_moments(X)
        loss = m.sum()+C.sum()
        if when == "double_backward":
            gradient, = torch.autograd.grad(loss, X, create_graph=True)
    target = X if operand and when != "forward" else model.train_inputs[0]
    target.detach().numpy()[0, 0] += .01
    with pytest.raises(EngineError) as caught:
        if when == "forward":
            source.joint_moments(X)
        elif when == "backward":
            torch.autograd.grad(loss, X)
        else:
            torch.autograd.grad(gradient.sum(), X)
    expected = "autograd_operand_changed" if operand and when != "forward" else "source_state_changed"
    assert caught.value.details["reason"] == expected


@pytest.mark.parametrize("when", ["forward", "backward", "double_backward"])
@pytest.mark.parametrize("mutation", ["targets", "copied_noise", "copied_training", "kernel", "endpoints", "uninspectable"])
def test_extracted_numeric_operands_are_part_of_the_frozen_generation(when, mutation):
    model, _ = training_field(noise=.03)
    source = adapter(model)
    X = tensor([[.2], [.65]]).requires_grad_()
    if when != "forward":
        m, C = source.joint_moments(X)
        loss = m.sum()+C.sum()
        if when == "double_backward":
            gradient, = torch.autograd.grad(loss, X, create_graph=True)
    if mutation == "targets":
        source.blocks[0].targets[0] += .01
    elif mutation == "copied_noise":
        source.noise.numpy()[0] = 0.
    elif mutation == "copied_training":
        source.training.numpy()[0, 0] = .2
    elif mutation == "kernel":
        source.blocks[0].kernel["lengthscale"][0] *= 2
    elif mutation == "endpoints":
        source.blocks[0]._endpoints.append([.3])
    else:
        source.blocks[0].kernel["unexpected"] = object()
    with pytest.raises(EngineError) as caught:
        if when == "forward":
            source.joint_moments(X)
        elif when == "backward":
            torch.autograd.grad(loss, X)
        else:
            torch.autograd.grad(gradient.sum(), X)
    assert caught.value.details["reason"] == "source_state_changed"


@pytest.mark.parametrize("cotangent", ["mean", "covariance"])
def test_actual_double_backward_rejects_cotangent_storage_mutation(cotangent):
    model, _ = training_field(noise=.03)
    source = adapter(model)
    X = tensor([[.2], [.65]]).requires_grad_()
    m, C = source.joint_moments(X)
    h, G = torch.ones_like(m, requires_grad=True), torch.ones_like(C, requires_grad=True)
    gradient, = torch.autograd.grad((m, C), X, (h, G), create_graph=True)
    operand = h if cotangent == "mean" else G
    operand.detach().numpy().flat[0] += .1
    with pytest.raises(EngineError) as caught:
        torch.autograd.grad(gradient.sum(), X)
    assert caught.value.details["reason"] == "autograd_operand_changed"


@pytest.mark.parametrize("when", ["forward", "backward", "double_backward"])
@pytest.mark.parametrize("mutation", ["category", "feature_scalar", "feature_tensor", "parameter_payload",
                                     "artifact", "structured_payload", "child_order", "prior", "method",
                                     "global_hook"])
def test_composed_semantic_manifest_across_all_derivative_passes(when, mutation):
    from .test_source_grammar import composed_model
    from expdoe_dk.bo.gp import _FeatureKernel, _CategorySimilarityKernel
    model = composed_model("saturation" if mutation == "structured_payload" else "power_law")
    source = adapter(model)
    X = tensor([[.2, 0.], [.65, 1.]]).requires_grad_()
    if when != "forward":
        m, C = source.joint_moments(X)
        loss = m.sum()+C.sum()
        if when == "double_backward":
            gradient, = torch.autograd.grad(loss, X, create_graph=True)
    feature = next(module.transform for module in model.modules() if type(module) is _FeatureKernel)
    category = next(module for module in model.modules() if type(module) is _CategorySimilarityKernel)
    handle = None
    if mutation == "category":
        category.similarities.numpy()[0, 1] += .01
    elif mutation == "feature_scalar":
        feature.numeric_origins[0] += 1
    elif mutation == "feature_tensor":
        feature.scale.numpy()[0] *= 2
    elif mutation == "parameter_payload":
        object.__setattr__(feature.params[0], "bounds", (.02, 1.))
    elif mutation == "artifact":
        artifact = model.mean_module.artifact
        object.__setattr__(artifact, "payload", dict(artifact.payload) | {"dimension": 1})
    elif mutation == "structured_payload":
        model.mean_module.components[0].payload["parameters"]["half_response"] = .7
    elif mutation == "child_order":
        from gpytorch.kernels import ProductKernel
        product = next(module for module in model.modules() if type(module) is ProductKernel)
        product.kernels = torch.nn.ModuleList(reversed(product.kernels))
    elif mutation == "prior":
        prior = next(module for module in model.modules() if hasattr(module, "concentration"))
        prior.concentration.data.add_(1.)
    elif mutation == "method":
        feature.forward = lambda x: x
    else:
        handle = torch.nn.modules.module.register_module_forward_hook(lambda module, args, output: output)
    try:
        with pytest.raises(EngineError) as caught:
            if when == "forward":
                source.joint_moments(X)
            elif when == "backward":
                torch.autograd.grad(loss, X)
            else:
                torch.autograd.grad(gradient.sum(), X)
        assert caught.value.details["reason"] == "source_state_changed"
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize("when", ["forward", "backward", "double_backward"])
@pytest.mark.parametrize("mutation", ["target", "noise", "constraint", "input_coefficient", "input_flag",
                                     "output_coefficient", "output_flag", "fit", "endpoint"])
def test_prepared_transform_and_directional_fit_guards_across_passes(when, mutation):
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch import settings
    from .test_posterior import engine, pairs
    train, targets = tensor([[.1], [.3], [.7], [.95]]), tensor([[.5], [.2], [-.3], [-.1]])
    with settings.validate_input_scaling(False):
        base = SingleTaskGP(train, targets, train_Yvar=torch.full_like(targets, .03),
                            input_transform=Normalize(d=1), outcome_transform=Standardize(m=1))
    model, _ = engine().condition_directional_posterior(base, pairs())
    assert model._moment_source is not None
    X = tensor([[.2], [.65]]).requires_grad_()
    if when != "forward":
        posterior = model.posterior(X)
        loss = posterior.mean.sum()+posterior.distribution.covariance_matrix.sum()
        if when == "double_backward":
            gradient, = torch.autograd.grad(loss, X, create_graph=True)
    if mutation == "target":
        base.train_targets = base.train_targets.clone()
    elif mutation == "noise":
        base.likelihood.noise_covar.noise.numpy()[0] += .01
    elif mutation == "constraint":
        base.covar_module.raw_lengthscale_constraint.lower_bound.data.add_(.01)
    elif mutation == "input_coefficient":
        base.input_transform._coefficient.numpy()[0, 0] += .01
    elif mutation == "input_flag":
        base.input_transform.transform_on_fantasize = False
    elif mutation == "output_coefficient":
        base.outcome_transform.stdvs.numpy()[0, 0] += .01
    elif mutation == "output_flag":
        base.outcome_transform._outputs = [0]
    elif mutation == "fit":
        next(model._fits[0].buffers()).numpy().flat[0] += .01
    else:
        model._endpoints.numpy()[0, 0] += .01
    with pytest.raises(EngineError) as caught:
        if when == "forward":
            model.posterior(X)
        elif when == "backward":
            torch.autograd.grad(loss, X)
        else:
            torch.autograd.grad(gradient.sum(), X)
    assert caught.value.details["reason"] == "source_state_changed"


@pytest.mark.parametrize("operation", ["deepcopy", "copy", "load"])
def test_old_directional_generation_cannot_be_reused_by_copy_or_load(operation):
    from copy import copy, deepcopy
    from .test_posterior import engine, pairs
    base, _ = training_field(noise=.03)
    model, _ = engine().condition_directional_posterior(base, pairs())
    X = tensor([[.2], [.65]])
    model.posterior(X)
    with pytest.raises(EngineError) as caught:
        if operation == "deepcopy":
            model = deepcopy(model)
        elif operation == "copy":
            model = copy(model)
        else:
            model.load_state_dict(model.state_dict())
        model.posterior(X)
    assert caught.value.details["reason"] == "source_state_changed"


@pytest.mark.parametrize("empty", [False, True])
def test_inactive_real_gp_keeps_identity_rng_and_does_not_import_mp(monkeypatch, empty):
    import builtins
    from .test_posterior import engine, pairs
    from expdoe_dk.bo.surrogates import _PhysicalAcquisitionModel
    base, _ = training_field(noise=.03)
    rng = torch.random.get_rng_state().clone()
    original_import = builtins.__import__
    def no_mp(name, *args, **kwargs):
        assert not name.startswith("mpmath"), "inactive route imported MP"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_mp)
    model, diagnostic = engine().condition_directional_posterior(base, pairs(empty=empty, weight=0.))
    assert model is base
    assert diagnostic["status"] == "inactive"
    assert _PhysicalAcquisitionModel(model, 0., 1., 1.)._directional_guard is None
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("component", ["features", "mean"])
def test_nonfinite_torch_input_boundary_fails_directly_before_mp(component):
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Log10
    from expdoe_dk.knowledge.shape import QuadraticMeanFrozen
    from botorch import settings
    train, targets = tensor([[.1], [.3], [.7], [.95]]), tensor([[.5], [.2], [-.3], [-.1]])
    kwargs = ({"input_transform": Log10(indices=[0])} if component == "features" else
              {"mean_module": QuadraticMeanFrozen(1, [1.], [.4])})
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(train, targets, train_Yvar=torch.full_like(targets, .03),
                             outcome_transform=None, **kwargs)
    prepared = adapter(model)
    with pytest.raises(EngineError) as caught:
        prepared.joint_moments(tensor([[-.1 if component == "features" else 1e200]]))
    assert caught.value.details["reason"] == "invalid_source"
    assert not prepared.blocks[0]._states
