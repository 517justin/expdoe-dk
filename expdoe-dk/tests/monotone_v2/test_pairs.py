from dataclasses import FrozenInstanceError
import importlib
import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from expdoe_dk.domain import LinearConstraint, ObservationBatch, Parameter, Space
from expdoe_dk.errors import EngineError, ErrorCode
from expdoe_dk.knowledge import Knowledge
from expdoe_dk.knowledge.artifacts import OptimizationArtifact, OptimizationArtifacts
from expdoe_dk.bo.knowledge_runtime import compile_runtime


def _build(*args, **kwargs):
    module = importlib.import_module("expdoe_dk.bo.directional_pairs")
    return module.build_directional_pairs(*args, **kwargs)


def _space(*names, constraints=()):
    return Space([Parameter(name, bounds=(0., 1.)) for name in names or ("x",)],
                 objectives="y", constraints=constraints)


def _observations(space, rows=()):
    return ObservationBatch(pd.DataFrame(rows, columns=space.param_names),
                            pd.DataFrame({"y": np.arange(len(rows), dtype=float)}))


def _artifact(pattern_id="KP-pairs", **payload):
    return OptimizationArtifact("monotone_sign_v2", {
        "dim": 0, "direction": "increasing", "n_pairs_per_dim": 5,
        "epsilon": .05, "softness": 1., "confidence": 1., **payload,
    }, pattern_id, "monotone", "2.0")


def _artifacts(*items):
    return OptimizationArtifacts(virtual_observations=items or (_artifact(),))


def test_seeded_pairs_respect_physical_domain():
    space = Space([Parameter("x", bounds=(10., 20.)), Parameter("z", bounds=(-2., 2.))],
                  objectives="y")
    observations = ObservationBatch(pd.DataFrame({"x": [10., 15., 15.], "z": [-2., 0., 0.]}),
        pd.DataFrame({"y": [10., 4., 99.]}), Yvar=pd.DataFrame({"y": [0., .1, .2]}),
        ids=("first", "second", "duplicate"))
    before = (observations.X, observations.Y, observations.Yvar)
    rng = (torch.random.get_rng_state().clone(), np.random.get_state(), random.getstate())
    first = _build(space, observations, _artifacts(), seed=17)
    second = _build(space, observations, _artifacts(), seed=17)
    assert torch.equal(first.endpoints, second.endpoints)
    torch.testing.assert_close(first.endpoints[:2], torch.tensor([
        [[0., 0.], [.05, 0.]], [[.45, .5], [.55, .5]]], dtype=torch.double))
    assert first.endpoints.shape == (5, 2, 2)
    assert first.endpoints.dtype == torch.double
    assert not first.endpoints.requires_grad
    assert len(set(map(tuple, first.endpoints.flatten(1).tolist()))) == 5
    assert torch.equal(first.endpoints[:, 0, 1], first.endpoints[:, 1, 1])
    assert space.feasibility_mask(space.model_to_physical(first.endpoints.reshape(-1, 2))).all()
    assert torch.equal(first.widths, first.endpoints[:, 1, 0] - first.endpoints[:, 0, 0])
    torch.testing.assert_close(first.weights, torch.full((5,), .2, dtype=torch.double))
    assert torch.equal(torch.random.get_rng_state(), rng[0])
    after_numpy = np.random.get_state()
    assert after_numpy[0] == rng[1][0] and after_numpy[2:] == rng[1][2:]
    np.testing.assert_array_equal(after_numpy[1], rng[1][1])
    assert random.getstate() == rng[2]
    for actual, expected in zip((observations.X, observations.Y, observations.Yvar), before):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert observations.ids == ("first", "second", "duplicate")
    changed = ObservationBatch(observations.X, pd.DataFrame({"y": [-1e9, 0., 1e9]}))
    assert torch.equal(first.endpoints, _build(space, changed, _artifacts(), seed=17).endpoints)
    assert not torch.equal(first.endpoints[2:], _build(space, observations, _artifacts(), seed=18).endpoints[2:])


@pytest.mark.parametrize("parameter,center,epsilon,physical,width", [
    (Parameter("x", kind="integer", bounds=(0, 10), step=2), 4, .11, (2, 6), .4),
    (Parameter("x", kind="discrete", values=[1., 10., 100.], transform="log"),
     10., .3, (1., 10.), .5),
    (Parameter("x", bounds=(1., 100.), transform="log"), 10., .1,
     (100. ** .4, 100. ** .6), .2),
])
def test_width_is_actual_post_snap_model_difference(parameter, center, epsilon, physical, width):
    space = Space([parameter], objectives="y")
    pairs = _build(space, _observations(space, [[center]]),
                   _artifacts(_artifact(epsilon=epsilon, n_pairs_per_dim=1)), seed=17)
    assert space.model_to_physical(pairs.endpoints.reshape(-1, 1)).x.tolist() == pytest.approx(physical)
    assert pairs.widths.tolist() == pytest.approx([width])


@pytest.mark.parametrize("kind,levels", [
    ("categorical", ["copper", "steel", "glass"]),
    ("ordinal", ["low", "mid", "high"]),
    ("categorical", ["1", "1.0", "01"]),
])
def test_context_coordinates_are_exactly_equal_after_roundtrip(kind, levels):
    space = Space([Parameter("x", bounds=(0., 1.)), Parameter("context", kind=kind, values=levels)],
                  objectives="y")
    pairs = _build(space, _observations(space, [[.5, levels[-1]]]), _artifacts(), seed=7)
    assert torch.equal(pairs.endpoints[:, 0, 1], pairs.endpoints[:, 1, 1])
    assert pairs.endpoints[0, :, 1].tolist() == [2., 2.]
    physical = space.model_to_physical(pairs.endpoints.reshape(-1, 2))
    for i in range(0, len(physical), 2):
        left, right = physical.context.iloc[i:i + 2]
        assert type(left) is type(right) and left == right


@pytest.mark.parametrize("constraint", [
    LinearConstraint("equality", {"x": 1., "z": 1.}, "==", 1.),
    LinearConstraint("fixed", {"x": 1.}, "==", .5),
])
def test_no_free_direction_fails_after_bounded_search(constraint):
    space = _space("x", "z", constraints=(constraint,))
    with pytest.raises(EngineError) as error:
        _build(space, _observations(space, [[.5, .5]]), _artifacts(), seed=3)
    assert error.value.code is ErrorCode.CONFIG_INVALID
    assert error.value.details["reason"] == "no_feasible_directional_pairs"
    assert error.value.details["pattern_id"] == "KP-pairs"
    assert error.value.details["requested"] == 5
    assert error.value.details["accepted"] == 0
    assert error.value.details["generated_centers"] == 4096


def test_duplicate_and_collapsed_pairs_cannot_fill_requested_count():
    space = Space([Parameter("x", kind="integer", bounds=(0, 1))], objectives="y")
    with pytest.raises(EngineError) as error:
        _build(space, _observations(space, [[0], [0], [1]]),
               _artifacts(_artifact(n_pairs_per_dim=2)), seed=4)
    assert error.value.details["reason"] == "no_feasible_directional_pairs"
    assert error.value.details["requested"] == 2
    assert error.value.details["accepted"] == 1
    assert error.value.details["generated_centers"] == 4096


@pytest.mark.parametrize("name", ["x", "trial_id", "y"])
def test_global_knowledge_constraints_use_parameter_identity(name):
    space = _space(name)
    knowledge = Knowledge().with_safe_region(expression=f"{name} <= 0.25", factors=(name,))
    constraints = compile_runtime(space, knowledge, None)["parameter_constraints"]
    pairs = _build(space, _observations(space, [[.9], [.15]]), _artifacts(), seed=17,
                   parameter_constraints=constraints)
    physical = space.model_to_physical(pairs.endpoints.reshape(-1, 1))
    assert physical[name].le(.25).all()
    torch.testing.assert_close(pairs.endpoints[0, :, 0], torch.tensor([.1, .2], dtype=torch.double))


def test_soft_parameter_constraint_does_not_reject_locations():
    constraint = LinearConstraint("soft", {"x": 1.}, "<=", .1, hard=False,
                                  penalty="linear", weight=1.)
    space = _space(constraints=(constraint,))
    pairs = _build(space, _observations(space, [[.8]]),
                   _artifacts(_artifact(n_pairs_per_dim=1)), seed=1)
    assert pairs.endpoints[0, :, 0].tolist() == pytest.approx([.75, .85])


def test_inactive_pattern_does_not_construct_sobol_or_change_active_pairs(monkeypatch):
    space = _space("x", "z")
    obs = _observations(space)
    active = _artifact("KP-z", dim=1)
    inactive = _artifact("KP-a", confidence=0.)
    expected = _build(space, obs, _artifacts(active), seed=11)
    actual = _build(space, obs, _artifacts(inactive, active), seed=11)
    assert torch.equal(actual.endpoints, expected.endpoints)
    assert actual.diagnostics[0]["status"] == "inactive"

    def unexpected(*args, **kwargs):
        pytest.fail("Zero confidence must not initialize or consume Sobol RNG")

    monkeypatch.setattr(torch.quasirandom, "SobolEngine", unexpected)
    pairs = _build(space, obs, _artifacts(inactive), seed=11)
    assert pairs.endpoints.shape == (0, 2, 2)
    assert pairs.pattern_ids == ()
    for name in ("signs", "widths", "softness", "weights"):
        assert getattr(pairs, name).shape == (0,)
    assert pairs.diagnostics[0]["accepted"] == 0
    assert pairs.diagnostics[0]["generated_centers"] == 0


def test_observed_centers_satisfy_request_without_sobol(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Observed centers must be tried before Sobol")

    monkeypatch.setattr(torch.quasirandom, "SobolEngine", unexpected)
    space = _space()
    pairs = _build(space, _observations(space, [[.8], [.2], [.5]]),
                   _artifacts(_artifact(n_pairs_per_dim=2)), seed=1)
    assert pairs.endpoints[:, :, 0].tolist() == pytest.approx(np.array([[.75, .85], [.15, .25]]))


def test_pattern_order_signs_weights_and_detached_diagnostics():
    space = _space("x", "z")
    a = _artifact("KP-a", dim=1, direction="decreasing", n_pairs_per_dim=2,
                  confidence=.6, softness=2.)
    z = _artifact("KP-z", n_pairs_per_dim=1)
    artifacts = _artifacts(z, a)
    pairs = _build(space, _observations(space), artifacts, seed=3)
    reordered = _build(space, _observations(space), _artifacts(a, z), seed=3)
    assert torch.equal(pairs.endpoints, reordered.endpoints)
    assert pairs.pattern_ids == ("KP-a", "KP-a", "KP-z")
    assert pairs.signs.tolist() == [-1., -1., 1.]
    assert pairs.weights.tolist() == pytest.approx([.3, .3, 1.])
    assert pairs.softness.tolist() == [2., 2., 1.]
    diagnostics = pairs.diagnostics
    json.dumps(diagnostics, allow_nan=False)
    diagnostics[0]["accepted"] = -1
    diagnostics.clear()
    assert pairs.diagnostics[0]["accepted"] == 2
    assert pairs.diagnostics[0]["location_domain"] == "declared_parameter_domain"
    assert pairs.diagnostics[0]["catalog_membership_required"] is False
    assert pairs.diagnostics[0]["outcome_safety_guaranteed"] is False
    assert artifacts.virtual_observations[1].payload["confidence"] == .6
    with pytest.raises(FrozenInstanceError):
        pairs.widths = torch.zeros(3)


@pytest.mark.parametrize("payload", [
    {"dim": True}, {"dim": 1}, {"dim": -1}, {"n_pairs_per_dim": True},
    {"n_pairs_per_dim": 1.0}, {"n_pairs_per_dim": 33}, {"epsilon": True},
    {"epsilon": "auto"}, {"epsilon": .5}, {"softness": 0},
    {"confidence": True}, {"confidence": -1}, {"direction": "unknown"},
])
def test_malformed_payloads_fail_explicitly(payload):
    space = _space()
    with pytest.raises(EngineError) as error:
        _build(space, _observations(space), _artifacts(_artifact(**payload)), seed=1)
    assert error.value.code is ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("kind", ["categorical", "ordinal"])
def test_nonnumeric_scoped_dimension_rejects_even_when_inactive(kind):
    space = Space([Parameter("x", kind=kind, values=["a", "b"])], objectives="y")
    with pytest.raises(EngineError) as error:
        _build(space, _observations(space), _artifacts(_artifact(confidence=0.)), seed=1)
    assert error.value.code is ErrorCode.CONFIG_INVALID


def test_total_pair_budget_allows_64_but_rejects_65():
    space = _space("x", "z", "w")
    a = _artifact("KP-a", n_pairs_per_dim=32)
    b = _artifact("KP-b", dim=1, n_pairs_per_dim=32)
    pairs = _build(space, _observations(space), _artifacts(a, b), seed=1)
    assert pairs.endpoints.shape == (64, 2, 3)
    with pytest.raises(EngineError) as error:
        _build(space, _observations(space), _artifacts(a, b, _artifact("KP-c", dim=2, n_pairs_per_dim=1)), seed=1)
    assert error.value.code is ErrorCode.CONFIG_INVALID


def test_compiled_direction_is_not_flipped_again():
    space = _space()
    knowledge = Knowledge().with_monotone_v2("x", "increases_objective")
    pairs = _build(space, _observations(space), knowledge.compile(space), seed=1)
    assert pairs.signs.tolist() == [-1.] * 5


def test_legacy_only_is_empty_and_does_not_fabricate_rows():
    space = _space()
    observations = _observations(space, [[.2]])
    artifacts = Knowledge().with_monotone("x", "increases_objective").compile(space)
    pairs = _build(space, observations, artifacts, seed=1)
    assert pairs.endpoints.shape == (0, 2, 1)
    assert pairs.diagnostics == []
    assert not hasattr(pairs, "Y") and not hasattr(pairs, "Yvar")


def test_sobol_centers_cover_full_declared_model_bounds():
    space = _space("x", "z")
    pairs = _build(space, _observations(space), _artifacts(), seed=13)
    centers = torch.quasirandom.SobolEngine(2, scramble=True, seed=13).draw(5, dtype=torch.double)
    torch.testing.assert_close(pairs.endpoints[:, 0, 0], (centers[:, 0] - .05).clamp(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(pairs.endpoints[:, 1, 0], (centers[:, 0] + .05).clamp(0, 1), rtol=0, atol=0)
    assert torch.equal(pairs.endpoints[:, 0, 1], centers[:, 1])


def test_both_space_and_global_constraints_are_enforced():
    space = _space(constraints=(LinearConstraint("floor", {"x": 1.}, ">=", .6),))
    knowledge = Knowledge().with_safe_region(expression="x <= 0.8", factors=("x",))
    constraints = compile_runtime(space, knowledge, None)["parameter_constraints"]
    pairs = _build(space, _observations(space, [[.3], [.9], [.7]]), _artifacts(),
                   seed=17, parameter_constraints=constraints)
    assert (pairs.endpoints >= .6).all() and (pairs.endpoints <= .8).all()
    assert pairs.endpoints[0, :, 0].tolist() == pytest.approx([.65, .75])


@pytest.mark.parametrize("seed", [True, 1.0, -1, "1", None])
def test_malformed_seed_is_explicit_config_error(seed):
    space = _space()
    with pytest.raises(EngineError) as error:
        _build(space, _observations(space), _artifacts(), seed=seed)
    assert error.value.code is ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("argument", ["space", "observations", "artifacts"])
def test_malformed_boundary_types_reject(argument):
    space = _space()
    kwargs = dict(space=space, observations=_observations(space), artifacts=_artifacts(), seed=1)
    kwargs[argument] = object()
    with pytest.raises(TypeError):
        _build(**kwargs)


def test_pair_container_detaches_constructor_inputs():
    module = importlib.import_module("expdoe_dk.bo.directional_pairs")
    endpoints = torch.tensor([[[.1], [.2]]], dtype=torch.double, requires_grad=True)
    vector = torch.tensor([1.], dtype=torch.double, requires_grad=True)
    diagnostic = [{"nested": {"accepted": 1}}]
    pairs = module.DirectionalPairs(endpoints, vector, vector, vector, vector, ["KP-test"], diagnostic)
    with torch.no_grad():
        endpoints.fill_(0)
        vector.fill_(0)
    diagnostic[0]["nested"]["accepted"] = 0
    assert pairs.endpoints.flatten().tolist() == [.1, .2]
    for name in ("signs", "widths", "softness", "weights"):
        assert getattr(pairs, name).tolist() == [1.]
        assert not getattr(pairs, name).requires_grad
    snapshot = pairs.diagnostics
    snapshot[0]["nested"]["accepted"] = -1
    assert pairs.diagnostics == [{"nested": {"accepted": 1}}]


def test_unprojected_space_and_unsuccessful_batch_reject():
    space = Space([Parameter("x", bounds=(0., 1.))], objectives=["y", "z"])
    with pytest.raises(EngineError) as error:
        _build(space, _observations(space), _artifacts(), seed=1)
    assert error.value.code is ErrorCode.CONFIG_INVALID
    obs = ObservationBatch(pd.DataFrame({"x": [.5]}), pd.DataFrame({"y": [None]}),
                           status=pd.Series(["failed"]))
    with pytest.raises(EngineError) as error:
        _build(_space(), obs, _artifacts(), seed=1)
    assert error.value.code is ErrorCode.CONFIG_INVALID
