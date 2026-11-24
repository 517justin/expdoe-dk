"""Frozen Task 6 failures against an independent forward-only MP conditional.

No optimizer, monkeypatch of the source, or saved rounded kernel is used as an
oracle. The JSON operands are projections of the historical failure/checkpoint
pair, not new campaign fits. The original diagnostic files remain untouched.
"""
import json
from pathlib import Path

import mpmath
import numpy as np
import pandas as pd
import pytest
import torch
from botorch import settings

from expdoe_dk import Knowledge, Objective, Parameter, Space
from expdoe_dk.bo import directional_posterior as dp
from expdoe_dk.bo.directional_pairs import DirectionalPairs
from expdoe_dk.bo.gp import build_gp
from expdoe_dk.bo.surrogates import _PhysicalFeatures


def tensor(value):
    return torch.tensor(value, dtype=torch.double)


@pytest.fixture(params=["current", "floor"])
def frozen_fit(request):
    data = json.loads((Path(__file__).parent / "fixtures" /
                       f"source-{request.param}.json").read_text())
    space = Space([Parameter("x", bounds=(0., 1.))],
                  objectives=[Objective("gain", "maximize")])
    frame = pd.DataFrame(data["training"], columns=["x"])
    features = _PhysicalFeatures(space, frame)
    y = np.asarray(data["physical_gain"])
    targets = tensor(-(y - y.mean()) / y.std()).unsqueeze(-1)
    with settings.validate_input_scaling(False):
        model, _ = build_gp(
            space, Knowledge(), tensor(data["training"]), targets,
            train_Yvar=tensor(data["noise"]).unsqueeze(-1),
            feature_transform=features, feature_groups=features.groups,
            normalize_outcome=False)
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        assert set(parameters) == set(data["parameters"])
        for name, value in data["parameters"].items():
            parameters[name].copy_(tensor(value))
    model.eval()
    assert torch.equal(model.likelihood.noise, tensor(data["noise"]))
    return data, model


def oracle(data, *, digits=100, feature_queries=None):
    """Explicit inverse conditional, independent of the proposed Cholesky source.

    Feature/response normalization is the existing binary64 input boundary.
    The saved positive raw parameters exceed the softplus identity threshold;
    this fixture-specific fact is not a production parameter extraction rule.
    """
    c = mpmath.mp.clone()
    c.dps = digits
    training = np.asarray(data["training"])[:, 0]
    queries = np.asarray(data["query"])[:, 0]
    t = [c.mpf(float(v)) for v in (training - training.mean()) / training.std()]
    z = [c.mpf(float(v)) for v in (queries - training.mean()) / training.std()]
    if feature_queries is not None:
        z = [c.mpf(v) for v in feature_queries]
    physical_y = np.asarray(data["physical_gain"])
    y = c.matrix([c.mpf(float(v)) for v in
                  -(physical_y - physical_y.mean()) / physical_y.std()])
    parameters = data["parameters"]
    ell = c.mpf(parameters["covar_module.kernel.base_kernel.raw_lengthscale"][0][0])
    amplitude = c.mpf(parameters["covar_module.kernel.raw_outputscale"])
    assert ell > 100 and amplitude > 100
    constant = c.mpf(parameters["mean_module.raw_constant"])

    def kernel(a, b):
        distance = c.sqrt(5) * abs(a - b) / ell
        return amplitude * (1 + distance + distance**2 / 3) * c.exp(-distance)

    A = c.matrix([[kernel(a, b) for b in t] for a in t])
    for i, noise in enumerate(data["noise"]):
        A[i, i] += c.mpf(noise)
    B = c.matrix([[kernel(a, b) for b in t] for a in z])
    D = c.matrix([[kernel(a, b) for b in z] for a in z])
    inverse = A**-1
    terms = B * inverse * (y - c.ones(len(t), 1) * constant)
    mean = c.ones(len(z), 1) * constant + terms
    covariance = D - B * inverse * B.T
    alpha = inverse * (y - c.ones(len(t), 1) * constant)
    mean_scales = [abs(constant) + c.fsum(abs(B[i, j] * alpha[j])
                                         for j in range(len(t))) for i in range(len(z))]
    return c, mean, covariance, mean_scales


def assert_local_accuracy(data, mean, covariance):
    c, expected_mean, expected_covariance, mean_scales = oracle(data)
    tau = 32 * c.mpf(2)**-53
    for i in range(len(data["query"])):
        assert abs(c.mpf(mean[i].item()) - expected_mean[i]) <= tau * mean_scales[i], (
            "mean", i, mean[i].item(), str(expected_mean[i]))
        assert covariance[i, i] > 0
        for j in range(len(data["query"])):
            budget = tau * c.sqrt(expected_covariance[i, i] * expected_covariance[j, j])
            assert abs(c.mpf(covariance[i, j].item()) - expected_covariance[i, j]) <= budget, (
                "covariance", i, j, covariance[i, j].item(), str(expected_covariance[i, j]))


def test_frozen_source_full_joint_matches_mp_conditional(frozen_fit):
    # Returning the old float64 Schur residual must fail these local budgets.
    data, model = frozen_fit
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    training, targets, noise = (model.train_inputs[0].clone(), model.train_targets.clone(),
                                model.likelihood.noise.clone())
    flags = {name: value.requires_grad for name, value in model.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    global_precision = mpmath.mp.prec
    mean, covariance = dp._joint_latent_moments(model, tensor(data["query"]))
    assert mpmath.mp.prec == global_precision
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(model.train_inputs[0], training)
    assert torch.equal(model.train_targets, targets)
    assert torch.equal(model.likelihood.noise, noise)
    assert flags == {name: value.requires_grad for name, value in model.named_parameters()}
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    assert_local_accuracy(data, mean, covariance)
    n = len(data["training"])
    assert torch.equal(covariance[n], covariance[-1])
    assert torch.equal(covariance[:, n], covariance[:, -1])
    assert mean[n] == mean[-1]
    assert torch.linalg.eigvalsh(covariance[:n, :n]).min() > 0


def test_frozen_source_passes_unchanged_directional_psd_and_rank_policy(frozen_fit):
    # The real production composition must accept these fits, not only an MP probe.
    data, model = frozen_fit
    n = len(data["training"])
    endpoints = tensor(data["query"][n:]).reshape(3, 2, 1)
    pairs = DirectionalPairs(
        endpoints, tensor([-1.] * 3), endpoints[:, 1, 0] - endpoints[:, 0, 0],
        tensor([1.] * 3), tensor([.8 / 3] * 3), ("frozen",) * 3, [])
    conditioned, diagnostics = dp.condition_directional_posterior(model, pairs)
    posterior = conditioned.posterior(tensor(data["training"]))
    assert diagnostics["fits"][0]["rank"] == 3
    assert torch.linalg.eigvalsh(posterior.distribution.covariance_matrix).min() > 0


def test_frozen_oracle_precision_and_nonzero_training_variance(frozen_fit):
    data, model = frozen_fit
    c, mean, covariance, _ = oracle(data)
    _, finer_mean, finer_covariance, _ = oracle(data, digits=130)
    n = len(data["training"])
    assert c.eigsy(covariance[:n, :n], eigvals_only=True)[0] > 0
    for i in range(covariance.rows):
        assert abs(mean[i] - finer_mean[i]) < c.mpf("1e-80")
        for j in range(covariance.cols):
            assert abs(covariance[i, j] - finer_covariance[i, j]) < c.mpf("1e-80")
    # Verify the reconstruction against constrained APIs, not raw assumptions.
    kernel = model.covar_module.kernel
    assert kernel.outputscale.item() == data["parameters"]["covar_module.kernel.raw_outputscale"]
    assert kernel.base_kernel.lengthscale.item() == data["parameters"]["covar_module.kernel.base_kernel.raw_lengthscale"][0][0]


def test_frozen_covariance_first_and_second_derivatives_against_independent_mp(frozen_fit):
    from .test_source_adapter import adapter
    data, model = frozen_fit
    source = adapter(model)
    query = tensor(data["query"][:2]).requires_grad_()
    _, covariance = source.joint_moments(query)
    weights, direction = tensor([[.3, -.2], [.5, .7]]), tensor([[.2], [-.3]])
    gradient, = torch.autograd.grad((covariance*weights).sum(), query, create_graph=True)
    hv, = torch.autograd.grad((gradient*direction).sum(), query)
    c = mpmath.mp.clone()
    c.dps = 100
    feature_module = model.covar_module.transform
    z = [c.mpf(v) for v in feature_module(query).detach().flatten().tolist()]
    derivative_scale = c.mpf(1./feature_module.scale.item())
    v = [c.mpf(v)*derivative_scale for v in direction.flatten().tolist()]
    step = c.mpf("1e-20")
    def loss(points):
        _, _, C, _ = oracle(data, feature_queries=points)
        return c.fsum(c.mpf(weights[i,j].item())*C[i,j] for i in range(2) for j in range(2))
    for i in range(2):
        plus, minus = z[:], z[:]
        plus[i] += step
        minus[i] -= step
        expected = (loss(plus)-loss(minus))/(2*step)*derivative_scale
        assert abs(c.mpf(gradient[i,0].item())-expected) < c.mpf("1e-19")
    second = (loss([x+step*d for x, d in zip(z, v)])-2*loss(z)+
              loss([x-step*d for x, d in zip(z, v)]))/step**2
    actual = c.fsum(c.mpf(h)*c.mpf(d) for h, d in zip(hv.flatten().tolist(), direction.flatten().tolist()))
    assert abs(actual-second) < c.mpf("1e-19")
