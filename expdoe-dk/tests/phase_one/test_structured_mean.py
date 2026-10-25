"""Physical v2 formulas and their single-counted MAP priors, without fitting."""
import itertools
import json
import math

import pandas as pd
import pytest
import torch
from gpytorch.means import Mean

from expdoe_dk import EngineError, ErrorCode, Knowledge, KnowledgeScope, Parameter, Space, make_pattern_spec
from expdoe_dk.bo.structured_mean import StructuredMean
from expdoe_dk.knowledge.artifacts import OptimizationArtifact


def case(pattern="saturation", *, factors=None, bounds=((0., 1.), (0., 1.)),
         log=False, maximize=False, location=0., scale=1., confidence=1., parameters=None,
         kinds=("continuous", "continuous")):
    """Compile real F1 artifacts and assign known parameters by their names."""
    space = Space([Parameter(name, bounds=limit, kind=kind, transform="log" if log else "linear")
                   for name, limit, kind in zip(("x", "y"), bounds, kinds)],
                  objectives="response", maximize=maximize)
    defaults = {
        "saturation": {"direction": "increasing", "origin": bounds[0][0],
                       "half_response": sum(bounds[0]) / 2},
        "threshold": {"threshold": sum(bounds[0]) / 2,
                      "below_behavior": "increasing", "above_behavior": "flat"},
        "synergy": {"reference": "anchored_additive", "response_effect": "higher"},
        "antagonism": {"reference": "anchored_additive", "response_effect": "higher"},
    }
    spec = make_pattern_spec(pattern_id="shape", pattern=pattern, version="2.0",
        scope=KnowledgeScope(factors=factors or (("x", "y") if pattern in {"synergy", "antagonism"} else ("x",)),
                             objectives=("response",)),
        confidence=confidence, parameters=defaults[pattern] | (parameters or {}))
    artifacts = Knowledge().add_spec(spec).compile(space).mean_components
    mean = StructuredMean(space, artifacts, response_location=location, response_scale=scale).double()
    assign(mean, baseline=location, **{
        name: .05 if name.endswith("raw_width") else .4
        for name, _ in mean.named_parameters() if "raw_" in name})
    return space, artifacts, mean


def assign(mean, **values):
    with torch.no_grad():
        for name, value in values.items():
            parameter = dict(mean.named_parameters())[name]
            target = parameter.new_tensor(value)
            constraint = mean.constraint_for_parameter_name(name)
            parameter.copy_(constraint.inverse_transform(target) if constraint else target)


def inputs(space, rows):
    return space.physical_to_model(pd.DataFrame(rows, columns=space.param_names))


@pytest.mark.parametrize("direction,d", [("increasing", 1), ("decreasing", -1)])
def test_saturation_anchor_half_limit_and_analytic_derivatives(direction, d):
    space, _, mean = case(parameters={"direction": direction})
    amplitude = d * .4
    saturation_at_anchor, saturation_at_half_response = mean.physical(inputs(space, [[0., 0.], [.5, 0.]])).tolist()
    assert saturation_at_half_response == pytest.approx(amplitude / 2)
    assert saturation_at_anchor == pytest.approx(0)
    # Evaluate the mathematical limit directly, outside Space prediction validation.
    assert mean.physical(torch.tensor([[1e12, 0.]], dtype=torch.double)).item() == pytest.approx(amplitude)
    X = torch.tensor([[.25, 0.]], dtype=torch.double, requires_grad=True)
    first = torch.autograd.grad(mean.physical(X).sum(), X, create_graph=True)[0]
    second = torch.autograd.grad(first[0, 0], X)[0][0, 0]
    assert first[0, 0].item() == pytest.approx(d * .4 * .5 / .75**2)
    assert second.item() == pytest.approx(-2 * d * .4 * .5 / .75**3)


BEHAVIORS = {"increasing": 1, "decreasing": -1, "flat": 0}


@pytest.mark.parametrize("log", [False, True], ids=["linear", "log"])
@pytest.mark.parametrize("direction,sign", [("increasing", 1.), ("decreasing", -1.)])
def test_saturation_huge_finite_domain_preserves_values_and_gradients(log, direction, sign):
    upper, lower = 1.7e308, 1e307 if log else 0.
    span = upper - lower
    _, _, mean = case(bounds=((lower, upper), (1., 2.)), log=log,
        parameters={"origin": lower, "half_response": lower + .6 * span,
                    "direction": direction})
    assign(mean, **{"components.0.raw_amplitude": 1.})
    fractions = torch.tensor([.2, .6, 1.], dtype=torch.double)
    model_x = (torch.log1p(fractions * (span / lower)) / math.log1p(span / lower)
               if log else fractions)
    X = torch.stack([model_x, torch.zeros_like(model_x)], dim=-1).requires_grad_()
    values = mean.physical(X)
    torch.testing.assert_close(values, sign * torch.tensor([.25, .5, .625], dtype=torch.double),
                               rtol=1e-12, atol=0.)
    first = torch.autograd.grad(values.sum(), X)[0][:, 0]
    expected = sign * torch.tensor([.9375, 5. / 12., .234375], dtype=torch.double)
    if log:
        expected *= (lower / span + fractions) * math.log1p(span / lower)
    assert torch.isfinite(first).all() and (sign * first > 0).all()
    torch.testing.assert_close(first, expected, rtol=1e-12, atol=0.)


def test_saturation_rejects_nonfinite_decoded_displacement_instead_of_false_limit():
    _, _, mean = case(bounds=((0., 1.7e308), (1., 2.)),
                      parameters={"half_response": 1.02e308})
    with pytest.raises(EngineError) as caught:
        mean.physical(torch.tensor([[2., 0.]], dtype=torch.double))
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("below,above", list(itertools.permutations(BEHAVIORS, 2)))
def test_threshold_all_six_pairs_anchor_midpoint_curvature_continuity_and_tails(below, above):
    _, _, mean = case("threshold", parameters={"below_behavior": below, "above_behavior": above})
    p, q, w = BEHAVIORS[below] * .4, BEHAVIORS[above] * .4, .05
    threshold_at_anchor = mean.physical(torch.zeros(1, 2, dtype=torch.double)).item()
    assert threshold_at_anchor == pytest.approx(0)
    X = torch.tensor([[.5, 0.]], dtype=torch.double, requires_grad=True)
    first = torch.autograd.grad(mean.physical(X).sum(), X, create_graph=True)[0]
    threshold_slope_at_h = first[0, 0].item()
    assert threshold_slope_at_h == pytest.approx((p + q) / 2)
    second = torch.autograd.grad(first[0, 0], X)[0][0, 0]
    assert second.item() == pytest.approx((q - p) / (4 * w))
    near = mean.physical(torch.tensor([[.5 - 1e-9, 0.], [.5 + 1e-9, 0.]], dtype=torch.double))
    assert abs((near[1] - near[0]).item()) < 1e-8
    extreme = torch.tensor([[-1e6, 0.], [1e6, 0.]], dtype=torch.double, requires_grad=True)
    values = mean.physical(extreme)
    assert torch.isfinite(values).all()
    slopes = torch.autograd.grad(values.sum(), extreme)[0][:, 0]
    torch.testing.assert_close(slopes, torch.tensor([p, q], dtype=torch.double))
    flat = "raw_below" if below == "flat" else "raw_above" if above == "flat" else None
    if flat:
        assert not any(name.endswith(flat) for name, _ in mean.named_parameters())


@pytest.mark.parametrize("kind,s", [("synergy", 1), ("antagonism", -1)])
@pytest.mark.parametrize("effect,d", [("higher", 1), ("lower", -1)])
def test_signed_interaction_corner_mixed_derivative_and_anchors(kind, s, effect, d):
    bounds = ((2., 6.), (10., 20.))
    space, _, mean = case(kind, bounds=bounds, parameters={"response_effect": effect})
    amplitude, R1, R2 = .4, 4., 10.
    interaction_at_upper_corner = mean.physical(inputs(space, [[6., 20.]])).item()
    assert interaction_at_upper_corner == pytest.approx(d * s * amplitude)
    X = torch.tensor([[.3, .4]], dtype=torch.double, requires_grad=True)
    first = torch.autograd.grad(mean.physical(X).sum(), X, create_graph=True)[0]
    mixed_physical_derivative = torch.autograd.grad(first[0, 0], X)[0][0, 1].item() / (R1 * R2)
    assert mixed_physical_derivative == pytest.approx(d * s * amplitude / (R1 * R2))
    for row in ([2., 20.], [6., 10.]):
        interaction_at_either_anchor = mean.physical(inputs(space, [row])).item()
        assert interaction_at_either_anchor == pytest.approx(0)


@pytest.mark.parametrize("pattern", ["saturation", "threshold", "synergy", "antagonism"])
def test_log_coordinates_match_physical_coordinates_and_have_autodiff(pattern):
    linear, _, first = case(pattern, bounds=((1., 9.), (2., 10.)))
    logged, _, second = case(pattern, bounds=((1., 9.), (2., 10.)), log=True)
    rows = [[1., 2.], [3., 5.], [9., 10.]]
    torch.testing.assert_close(first.physical(inputs(linear, rows)), second.physical(inputs(logged, rows)))
    X = inputs(logged, rows[1:2]).requires_grad_()
    assert torch.autograd.gradcheck(second.physical, (X,))
    loss = second(X).sum() + second.log_prior()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in second.parameters())


@pytest.mark.parametrize("maximize,sign", [(False, 1), (True, -1)])
def test_baseline_normalization_batch_dtype_and_device(maximize, sign):
    _, _, mean = case(location=17., scale=8., maximize=maximize)
    baseline_parameters = [name for name, _ in mean.named_parameters() if "baseline" in name]
    assert len(baseline_parameters) == 1
    assert mean.baseline.item() == 17.
    X = torch.tensor([[[0., 0.], [.5, 0.]], [[.5, .1], [1., 1.]]], dtype=torch.double)
    expected = torch.tensor([[17., 18.6], [18.6, 17. + 3.2 / 1.5]], dtype=torch.double)
    torch.testing.assert_close(mean.physical(X), expected)
    torch.testing.assert_close(mean(X), sign * (expected - 17.) / 8.)
    assert mean(X).shape == X.shape[:-1]
    assert mean(X).dtype == torch.double
    assert all(v.dtype == torch.double and v.device == X.device for v in mean.parameters())
    mean.float().initialize(magnitude=.2, width=.1)
    assert mean(X.float()).dtype == torch.float
    assert all(v.dtype == torch.float for v in mean.buffers())


@pytest.mark.parametrize("pattern", ["saturation", "threshold", "synergy", "antagonism"])
def test_factor_and_response_unit_changes_preserve_normalized_predictions(pattern):
    space, _, mean = case(pattern)
    changed, _, converted = case(pattern, bounds=((100., 1100.), (-5., 5.)), location=50., scale=100.)
    torch.testing.assert_close(mean(inputs(space, [[.2, 0.], [.8, .5]])),
                               converted(inputs(changed, [[300., -5.], [900., 0.]])))
    if pattern == "saturation":
        assert converted.physical(inputs(changed, [[600., 0.]])).item() == pytest.approx(70.)


def test_multiple_univariate_shapes_add_before_one_normalization():
    space, saturation, _ = case()
    spec = make_pattern_spec(pattern_id="y-threshold", pattern="threshold", version="2.0",
        scope=KnowledgeScope(factors=("y",), objectives=("response",)), confidence=.8,
        parameters={"threshold": .5, "below_behavior": "increasing", "above_behavior": "flat"})
    threshold = Knowledge().add_spec(spec).compile(space).mean_components
    mean = StructuredMean(space, saturation + threshold, response_location=7., response_scale=3.)
    mean.initialize(magnitude=.4, width=.05)
    # Saturation at half response is .2*S; symmetric flat-tail threshold at u=1 is .2*S.
    X = torch.tensor([[.5, 1.]], dtype=torch.double)
    assert mean.physical(X).item() == pytest.approx(8.2)
    assert mean.forward(X).item() == pytest.approx(.4)
    assert mean.baseline.item() == 7.
    mean.initialize(magnitude=.2, width=.1)
    assert mean.baseline.item() == 7.


def test_multiple_shapes_single_baseline_and_shared_nuisance_slope():
    space, sat, _ = case()
    _, interaction, _ = case("synergy")
    _, other_interaction, _ = case("antagonism", factors=("y", "x"))
    # Real independently compiled declarations, with distinct provenance IDs.
    second = OptimizationArtifact.from_dict(other_interaction[0].to_dict() | {"source_pattern_id": "second"})
    mean = StructuredMean(space, sat + interaction + (second,), response_location=10., response_scale=2.).double()
    mean.initialize(magnitude=.3, width=.05)
    main_effect_for_shared_factor = [name for name, _ in mean.named_parameters() if name.startswith("nuisance.")]
    assert len(main_effect_for_shared_factor) == 1
    assert mean.nuisance["1"].item() == 0.
    assign(mean, **{"nuisance.1": .7})
    # Opposite interaction terms cancel; x has a shape and gets no free main effect.
    assert mean.physical(torch.ones(1, 2, dtype=torch.double)).item() == pytest.approx(10 + .6 / 1.5 + 1.4)
    assert len([name for name, _ in mean.named_parameters() if name == "baseline"]) == 1


def test_swapping_interaction_factors_does_not_change_function_or_prior():
    _, first, mean = case("synergy")
    _, second, swapped = case("synergy", factors=("y", "x"))
    assert first[0].payload == second[0].payload
    X = torch.tensor([[.2, .7], [.8, .1]], dtype=torch.double)
    torch.testing.assert_close(mean(X), swapped(X))
    torch.testing.assert_close(mean.log_prior(), swapped.log_prior())


@pytest.mark.parametrize("below,above", list(itertools.permutations(BEHAVIORS, 2)))
def test_prior_is_explicit_single_sum_with_constants_and_no_baseline_penalty(below, above):
    _, _, mean = case("threshold", scale=10., confidence=.6,
                     parameters={"below_behavior": below, "above_behavior": above})
    dtype = torch.double
    value = torch.tensor(.4, dtype=dtype)
    count = int(below != "flat") + int(above != "flat")
    for multiplier, center in [(1., .05), (2., .1)]:
        half = torch.distributions.HalfNormal(torch.tensor((.1 + .9 * .6) * multiplier, dtype=dtype))
        width = torch.distributions.Normal(torch.tensor(math.log(center), dtype=dtype), .5)
        expected = count * half.log_prob(value) + width.log_prob(torch.tensor(math.log(.05), dtype=dtype))
        torch.testing.assert_close(mean.log_prior(magnitude_scale_multiplier=multiplier, width_prior_center=center), expected)
    before = mean.log_prior().detach().clone()
    assign(mean, baseline=1e6)
    torch.testing.assert_close(mean.log_prior(), before)


def test_prior_counts_each_interaction_magnitude_and_shared_free_beta_once():
    space, one, _ = case("synergy", confidence=.6)
    second = OptimizationArtifact.from_dict(one[0].to_dict() | {"source_pattern_id": "second"})
    mean = StructuredMean(space, one + (second,), response_location=3., response_scale=10.).double()
    mean.initialize(magnitude=.4, width=.05)
    assign(mean, **{"nuisance.0": -.2, "nuisance.1": .3})
    half = torch.distributions.HalfNormal(torch.tensor(.64, dtype=torch.double))
    normal = torch.distributions.Normal(torch.tensor(0., dtype=torch.double), 1.)
    expected = 2 * half.log_prob(torch.tensor(.4, dtype=torch.double)) + normal.log_prob(torch.tensor(-.2, dtype=torch.double)) + normal.log_prob(torch.tensor(.3, dtype=torch.double))
    torch.testing.assert_close(mean.log_prior(), expected)
    _, _, high = case("synergy", confidence=1.)
    _, _, low = case("synergy", confidence=.1)
    high_confidence_log_prior, low_confidence_log_prior = high.log_prior().item(), low.log_prior().item()
    assert high_confidence_log_prior != low_confidence_log_prior


def test_initialize_constraints_approach_zero_and_width_description():
    _, _, mean = case("threshold", bounds=((2., 6.), (0., 1.)), scale=10.)
    mean.initialize(magnitude=1e-5, width=.1)
    description = mean.describe()
    json.dumps(description, allow_nan=False)
    assert description["parameter_identifiability"] == "not_certified"
    assert description["uncertainty"] == "conditional_on_MAP"
    item = description["components"][0]
    assert item["normalized_below_slope"] == pytest.approx(1e-5)
    assert item["physical_below_slope"] == pytest.approx(2.5e-5)
    assert item["normalized_above_slope"] == 0.
    assert item["width"] == pytest.approx(.1)
    assert item["physical_width"] == pytest.approx(.4)
    assert item["anchors"] == [2.]
    assert item["confidence"] == 1.
    assert item["prior"]["magnitude_scale"] == 1.
    assert item["source_pattern_id"] == "shape"
    assert "width_unidentified_near_zero_effect" in item["flags"]
    for raw in (-1000., 1000.):
        with torch.no_grad():
            dict(mean.named_parameters())["components.0.raw_width"].fill_(raw)
        assert .01 <= mean.describe()["components"][0]["width"] <= .25
    assert isinstance(mean, Mean)


@pytest.mark.parametrize("field,value", [
    ("confidence", float("nan")), ("confidence", 0.), ("confidence", 2.),
    ("objectives", ["other"]), ("dimensions", [1]), ("dimensions", [False]),
    ("factors", ["y"]), ("coordinates", []), ("unknown", 1),
])
def test_invalid_metadata_is_typed_knowledge_invalid(field, value):
    space, artifacts, _ = case()
    artifact = artifacts[0]
    # Simulate corrupted in-memory input; JSON construction itself forbids NaN.
    object.__setattr__(artifact, "payload", artifact.to_dict()["payload"] | {field: value})
    with pytest.raises(EngineError) as caught:
        StructuredMean(space, (artifact,), response_location=0., response_scale=1.)
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("field,value", [("lower", -.1), ("upper", 2.), ("range", 2.),
    ("transform", "log"), ("factor", "y"), ("range", float("inf"))])
def test_coordinate_tampering_is_not_clipped(field, value):
    space, artifacts, _ = case()
    artifact = artifacts[0]
    payload = artifact.to_dict()["payload"]
    payload["coordinates"][0][field] = value
    object.__setattr__(artifact, "payload", payload)
    with pytest.raises(EngineError) as caught:
        StructuredMean(space, (artifact,), response_location=0., response_scale=1.)
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("pattern,parameters", [
    ("saturation", {"origin": .1}), ("saturation", {"half_response": 1.}),
    ("saturation", {"direction": "flat"}), ("threshold", {"threshold": float("nan")}),
    ("threshold", {"below_behavior": "flat", "above_behavior": "flat"}),
    ("synergy", {"reference": "unanchored"}), ("synergy", {"response_effect": "unknown"}),
])
def test_malformed_parameters_are_typed(pattern, parameters):
    space, artifacts, _ = case(pattern)
    artifact = artifacts[0]
    payload = artifact.to_dict()["payload"]
    payload["parameters"].update(parameters)
    object.__setattr__(artifact, "payload", payload)
    with pytest.raises(EngineError) as caught:
        StructuredMean(space, (artifact,), response_location=0., response_scale=1.)
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("location,scale", [(float("nan"), 1.), (0., 0.), (0., -1.), (0., float("inf"))])
def test_bad_response_normalization_is_typed(location, scale):
    space, artifacts, _ = case()
    with pytest.raises(EngineError) as caught:
        StructuredMean(space, artifacts, response_location=location, response_scale=scale)
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID


def test_empty_mean_has_only_baseline_and_nonfinite_evaluation_is_typed():
    space, _, _ = case()
    mean = StructuredMean(space, (), response_location=3., response_scale=2.)
    assert list(dict(mean.named_parameters())) == ["baseline"]
    torch.testing.assert_close(mean(torch.zeros(2, 2, dtype=torch.double)), torch.zeros(2, dtype=torch.double))
    assert mean.log_prior().item() == 0.
    with pytest.raises(EngineError) as caught:
        mean(torch.tensor([[float("nan"), 0.]], dtype=torch.double))
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID


@pytest.mark.parametrize("log", [False, True], ids=["linear", "log"])
@pytest.mark.parametrize("pattern", ["saturation", "threshold", "synergy", "antagonism", "nuisance"])
def test_large_integer_anchor_relative_values_and_gradients(pattern, log):
    low = 2**53
    declaration = "synergy" if pattern == "nuisance" else pattern
    parameters = ({"half_response": low + 1} if pattern == "saturation" else
                  {"threshold": low + 1} if pattern == "threshold" else {})
    space, artifacts, mean = case(declaration, bounds=((low, low + 4), (1., 2.)),
        kinds=("integer", "continuous"), log=log, parameters=parameters)
    assert artifacts[0].payload["coordinates"][0]["lower"] == low
    if pattern == "nuisance":
        # Isolate the shared free main effect without changing the public API.
        assign(mean, **{"components.0.raw_amplitude": 1e-100, "nuisance.0": .7})
    X = inputs(space, [[low, 2.], [low + 1, 2.], [low + 2, 2.], [low + 4, 2.]]).requires_grad_()
    values = mean.physical(X)
    assert values[0].item() == pytest.approx(0., abs=1e-15)
    assert len(set(values.detach().tolist())) == 4
    if pattern == "threshold":
        # h=.25, p=.4, q=0: the midpoint value includes the anchored smoothing tail.
        expected_half = .1 - .02 * (math.log(2.) - math.log1p(math.exp(-5.)))
        assert values[1].item() == pytest.approx(expected_half)
        assert values[2].item() == pytest.approx(.1)
        expected_first, expected_second = .2, -2.
    else:
        expected, expected_first, expected_second = {
            "saturation": ([0., .2, .4 * 2 / 3, .32], .4, -1.6),
            "synergy": ([0., .1, .2, .4], .4, 0.),
            "antagonism": ([0., -.1, -.2, -.4], -.4, 0.),
            "nuisance": ([0., .175, .35, .7], .7, 0.),
        }[pattern]
        torch.testing.assert_close(values, torch.tensor(expected, dtype=torch.double), rtol=1e-12, atol=1e-15)
    first = torch.autograd.grad(values.sum(), X, create_graph=True)[0]
    second = torch.autograd.grad(first[1, 0], X)[0][1, 0]
    # log1p(4/low) makes the log-coordinate chain derivative 4 to float64 precision.
    assert first[1, 0].item() == pytest.approx(expected_first, rel=1e-12)
    assert second.item() == pytest.approx(expected_second, rel=1e-12, abs=1e-12)
    assert torch.autograd.gradcheck(mean.physical, (X,), eps=1e-6, atol=1e-8)


def test_wide_continuous_log_domain_has_zero_anchor_finite_values_and_gradients():
    space, artifacts, mean = case("synergy", bounds=((1e-200, 1e200), (1., 2.)), log=True)
    assert artifacts[0].payload["coordinates"][0]["range"] == 1e200
    X = inputs(space, [[1e-200, 2.], [1., 2.], [1e200, 2.]]).requires_grad_()
    values = mean.physical(X)
    assert values[0].item() == 0.
    torch.testing.assert_close(values, torch.tensor([0., 4e-201, .4], dtype=torch.double), rtol=1e-12, atol=0.)
    log_span = 400 * math.log(10.)
    first = torch.autograd.grad(values.sum(), X, create_graph=True)[0]
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first[:, 0], torch.tensor([0., 4e-201 * log_span, .4 * log_span],
        dtype=torch.double), rtol=1e-12, atol=0.)
    second = torch.autograd.grad(first[:, 0].sum(), X)[0]
    assert torch.isfinite(second).all()
    torch.testing.assert_close(second[:, 0], torch.tensor([0., 4e-201 * log_span**2, .4 * log_span**2],
        dtype=torch.double), rtol=1e-12, atol=0.)
    assert torch.autograd.gradcheck(mean.physical, (X,), eps=1e-7, atol=1e-8)
    mean.physical(X).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mean.parameters())
