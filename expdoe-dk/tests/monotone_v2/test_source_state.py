"""Owned fixed-model state, including mutations invisible to tensor versions."""
import importlib
import importlib.util

import pytest
import torch

from expdoe_dk.errors import EngineError, ErrorCode
from .test_posterior import training_field


def state_module():
    name = "expdoe_dk.bo.directional_source_state"
    assert importlib.util.find_spec(name) is not None, "owned source state is not implemented"
    return importlib.import_module(name)


@pytest.mark.parametrize("mutation", ["raw_numpy", "training_data", "target_replace", "noise_numpy",
                                     "flag", "constraint", "method", "child_hook", "unknown"])
def test_owned_source_detects_semantic_changes(mutation):
    module = state_module()
    model, _ = training_field(noise=.03, transformed=True)
    state = module.OwnedState(model)
    state.check()
    if mutation == "raw_numpy":
        model.covar_module.raw_lengthscale.detach().numpy()[...] += .01
    elif mutation == "training_data":
        model.train_inputs[0].data.add_(.01)
    elif mutation == "target_replace":
        model.train_targets = model.train_targets.clone()
    elif mutation == "noise_numpy":
        model.likelihood.noise_covar.noise.detach().numpy()[0] += .01
    elif mutation == "flag":
        model.input_transform.transform_on_eval = False
    elif mutation == "constraint":
        model.covar_module.raw_lengthscale_constraint.lower_bound.data.add_(.01)
    elif mutation == "method":
        model.mean_module.forward = lambda x: x[..., 0]
    elif mutation == "child_hook":
        model.mean_module.register_forward_hook(lambda m, args, result: result)
    else:
        model.mean_module.unregistered_state = [1.]
    with pytest.raises(EngineError) as caught:
        state.check()
    assert caught.value.code == ErrorCode.MODEL_FIT_FAILED
    assert caught.value.details["reason"] == "source_state_changed"


def test_owned_source_query_modules_are_detached_without_changing_original():
    module = state_module()
    model, _ = training_field(noise=.03, transformed=True)
    flags = [p.requires_grad for p in model.parameters()]
    state = module.OwnedState(model)
    assert flags == [p.requires_grad for p in model.parameters()]
    assert all(not p.requires_grad for p in state.model.parameters())
    for name, tensor in model.state_dict().items():
        copied = state.model.state_dict()[name]
        assert torch.equal(tensor, copied)
        if tensor.numel():
            assert tensor.data_ptr() != copied.data_ptr()
    query = torch.tensor([[.2], [.7]], dtype=torch.double, requires_grad=True)
    transformed = state.model.transform_inputs(query)
    mean = state.model.mean_module(transformed)
    loss = transformed.square().sum() + mean.sum()
    grad, = torch.autograd.grad(loss, query, create_graph=True)
    assert torch.isfinite(torch.autograd.grad(grad.sum(), query)[0]).all()
    state.check()


def test_modified_provider_is_ineligible_before_preparation():
    module = state_module()
    model, _ = training_field(noise=.03)
    model.posterior = lambda *args, **kwargs: None
    assert not module.eligible(model)


def test_operand_snapshot_detects_external_storage_write():
    module = state_module()
    value = torch.tensor([.2, .7], dtype=torch.double, requires_grad=True)
    snapshot = module.OperandSnapshot(value)
    value.detach().numpy()[0] = .3
    with pytest.raises(EngineError) as caught:
        snapshot.check()
    assert caught.value.details["reason"] == "autograd_operand_changed"


def test_owned_query_module_mutation_cannot_change_a_prepared_generation():
    module = state_module()
    model, _ = training_field(noise=.03, transformed=True)
    state = module.OwnedState(model)
    state.model.mean_module.raw_constant.data.add_(1.)
    with pytest.raises(EngineError) as caught:
        state.check()
    assert caught.value.details["reason"] == "source_state_changed"


def test_audited_function_at_wrong_instance_slot_is_still_modified_provider():
    module = state_module()
    model, _ = training_field(noise=.03)
    model.mean_module.forward = type(model.mean_module).forward
    assert not module.eligible(model)


def test_audited_global_math_function_change_is_not_ignored(monkeypatch):
    module = state_module()
    model, _ = training_field(noise=.03)
    state = module.OwnedState(model)
    monkeypatch.setattr(torch.nn.functional, "softplus", lambda value, **kwargs: value)
    with pytest.raises(EngineError) as caught:
        state.check()
    assert caught.value.details["reason"] == "source_state_changed"


def test_derived_total_and_constrained_values_have_exact_value_snapshots():
    module = state_module()
    model, _ = training_field(noise=.03, amplitude=1.7)
    state = module.OwnedState(model)
    assert hasattr(state, "_derived_record"), "derived values must be snapshotted explicitly"
    values = dict(state._derived_record)
    for key, value in (("likelihood.noise", model.likelihood.noise),
                       ("covar_module.outputscale", model.covar_module.outputscale),
                       ("covar_module.base_kernel.lengthscale", model.covar_module.base_kernel.lengthscale)):
        assert values[key] == (tuple(value.shape), value.dtype, value.device,
                              value.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes())
    # Re-evaluation creates fresh constrained tensors; identity is not their
    # semantic comparison. Ordinary stored tensors still retain identity guards.
    state.check()
    state.check()


@pytest.mark.parametrize("pattern", ["power_law", "saturation"])
def test_all_audited_copies_have_disjoint_mutable_storage_and_payloads(pattern):
    from collections.abc import Mapping
    from .test_source_grammar import composed_model
    model = composed_model(pattern)
    state = state_module().OwnedState(model)
    def inventory(root):
        seen, objects, storage = set(), set(), set()
        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                objects.add(id(value))
                if value.numel():
                    storage.add(value.untyped_storage().data_ptr())
                assert value.device.type == "cpu"
            elif isinstance(value, Mapping):
                objects.add(id(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple, set)):
                if not isinstance(value, tuple):
                    objects.add(id(value))
                for child in value:
                    visit(child)
            elif hasattr(value, "__dict__") and not callable(value):
                objects.add(id(value))
                for child in vars(value).values():
                    visit(child)
            elif isinstance(value, torch.nn.Module):
                objects.add(id(value))
                for name, child in vars(value).items():
                    if name not in ("prediction_strategy", "_memoize_cache"):
                        visit(child)
        visit(root)
        return objects, storage
    original_objects, original_storage = inventory(model)
    owned_objects, owned_storage = inventory(state.model)
    assert not original_objects & owned_objects
    assert not original_storage & owned_storage
