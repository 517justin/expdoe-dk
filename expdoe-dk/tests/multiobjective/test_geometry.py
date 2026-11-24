"""Fixed observed geometry, physical sample utilities, and explicit preferences."""
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pandas as pd
import pytest
import torch

from expdoe_dk import (
    EngineError, ErrorCode, Knowledge, KnowledgeScope, MultiObjectiveConfig,
    Objective, ObservationBatch, OutcomeConstraint, Parameter, Space, make_pattern_spec,
)
from expdoe_dk.bo.knowledge_runtime import compile_runtime


def api():
    from expdoe_dk.bo.objective_geometry import ObjectiveGeometry, objective_samples, resolve_geometry
    return ObjectiveGeometry, objective_samples, resolve_geometry


def space_for(*objectives, **kwargs):
    return Space([Parameter("x", bounds=(0., 10.))],
                 objectives=objectives or (Objective("a", "maximize"), Objective("b", "minimize")),
                 **kwargs)


def batch(data=None, *, status=None, yvar=None, x=None):
    y = pd.DataFrame({"b": [8., 6., 4.], "a": [2., 5., 8.]} if data is None else data)
    return ObservationBatch(pd.DataFrame({"x": list(range(len(y))) if x is None else x}), y,
                            Yvar=yvar, status=None if status is None else pd.Series(status))


def resolve(space=None, observations=None, knowledge=None, config=None):
    space = space_for() if space is None else space
    observations = batch() if observations is None else observations
    runtime = compile_runtime(space, Knowledge() if knowledge is None else knowledge, observations)
    geometry = api()[2](space, observations, runtime, MultiObjectiveConfig() if config is None else config)
    return geometry, runtime


def range_knowledge(*, duplicate=False):
    knowledge = Knowledge()
    for identity in (["range", "same-range"] if duplicate else ["range"]):
        knowledge.add(make_pattern_spec(pattern_id=identity, pattern="target_range", version="1.0",
            scope=KnowledgeScope(objectives=("a",)), parameters={"lower": 4., "upper": 6.}, confidence=.8))
    return knowledge


def test_mixed_directions_and_target_utilities():
    space = space_for(Objective("a", "maximize"), Objective("b", "minimize"), Objective("c", "target", target=5.))
    geometry, runtime = resolve(space, batch({"c": [5., 6., 7.], "b": [2., 3., 4.], "a": [1., 2., 3.]}))
    physical = torch.tensor([3., 4., 7.], dtype=torch.double)
    normalized = api()[1](physical, space, geometry, runtime, with_preferences=False)
    torch.testing.assert_close(normalized * torch.tensor(geometry.scale) + torch.tensor(geometry.location),
                               torch.tensor([3., -4., -2.], dtype=torch.double))


@pytest.mark.parametrize("target", [5., (3., 6.)])
def test_tensor_utility_matches_public_objectives_without_mean_substitution(target):
    space = space_for(Objective("a", "target", target=target), Objective("b", "minimize"))
    geometry, runtime = resolve(space, batch())
    samples = torch.tensor([[[[-2., 8.], [4., 3.]], [[9., 2.], [5., 7.]]]], dtype=torch.double)
    actual = api()[1](samples, space, geometry, runtime, with_preferences=False)
    expected = np.stack([o.to_utility(samples[..., i].numpy()) for i, o in enumerate(space.objective_specs)], axis=-1)
    np.testing.assert_allclose(actual.numpy() * geometry.scale + geometry.location, expected)
    assert actual.shape == samples.shape
    if target == 5.:
        draws = torch.tensor([[3., 0.], [7., 0.]], dtype=torch.double)
        result = api()[1](draws, space, geometry, runtime, with_preferences=False)
        assert (result[:, 0] * geometry.scale[0] + geometry.location[0]).mean() == -2.


def test_fixed_geometry_ignores_future_outcomes():
    geometry, runtime = resolve()
    before = geometry.to_dict()
    actual = api()[1](torch.tensor([[20., -4.]], dtype=torch.double), space_for(), geometry, runtime, with_preferences=True)
    torch.testing.assert_close(actual, torch.tensor([[3., 3.]], dtype=torch.double))
    assert geometry.to_dict() == before
    assert geometry.location == (2., -8.)
    assert geometry.scale == (6., 4.)
    assert geometry.report_reference == pytest.approx((-.1, -.1))
    assert geometry.acquisition_ideal == (1., 1.)
    with pytest.raises(FrozenInstanceError):
        geometry.scale = (1., 1.)


def test_constant_columns_have_finite_documented_fallback():
    geometry, runtime = resolve(observations=batch({"a": [0., 0., 0.], "b": [4., 4., 4.]}))
    assert geometry.scale == (1., 4.)
    assert geometry.report_reference == pytest.approx((-.1, -.1))
    assert torch.isfinite(api()[1](torch.tensor([[1., 9.]]), space_for(), geometry, runtime, with_preferences=True)).all()


def test_exact_named_mapping_and_physical_reference():
    config = MultiObjectiveConfig(reference_point={"b": 10., "a": 0.})
    geometry, _ = resolve(config=config)
    assert geometry.report_reference == pytest.approx((-1 / 3, -.5))
    assert dict(geometry.physical_reference) == {"a": 0., "b": 10.}
    assert geometry.reference_method == "physical"
    reordered, _ = resolve(observations=batch({"a": [2., 5., 8.], "b": [8., 6., 4.]}), config=config)
    assert reordered == geometry


@pytest.mark.parametrize("columns", [{"a": [1., 2., 3.]}, {"a": [1., 2., 3.], "b": [1., 2., 3.], "extra": [0., 0., 0.]}])
def test_observation_names_must_match_exactly(columns):
    with pytest.raises(EngineError) as caught:
        resolve(observations=batch(columns))
    assert caught.value.code == ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("reference", [{"a": 8., "b": 4.}, {"a": 100., "b": 100.}])
def test_explicit_reference_requires_one_strictly_dominating_feasible_row(reference):
    with pytest.raises(EngineError) as caught:
        resolve(config=MultiObjectiveConfig(reference_point=reference))
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_no_single_row_dominates_reference_even_when_column_maxima_do():
    with pytest.raises(EngineError) as caught:
        resolve(observations=batch({"a": [1., 5., 9.], "b": [1., 5., 9.]}),
                config=MultiObjectiveConfig(reference_point={"a": 4., "b": 4.}))
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_feasible_baseline_respects_physical_safety_and_knowledge_constraints():
    space = space_for(outcome_constraints=[OutcomeConstraint("safe-b", "b", "<=", 6.)])
    knowledge = Knowledge().with_safe_region(expression="x <= 1", factors=("x",))
    geometry, _ = resolve(space, knowledge=knowledge)
    assert geometry.location == (2., -8.)
    assert geometry.report_reference == pytest.approx((.4, .4))
    with pytest.raises(EngineError) as caught:
        resolve(space, knowledge=Knowledge().with_safe_region(expression="x >= 9", factors=("x",)))
    assert caught.value.code == ErrorCode.INSUFFICIENT_DATA
    assert caught.value.details["reason"] == "no_feasible_baseline"


def test_success_only_geometry_but_digest_records_initial_unsuccessful_rows():
    observations = batch({"a": [2., 5., 8., np.nan], "b": [8., 6., 4., np.nan]},
                         status=["success", "success", "success", "failed"])
    geometry, _ = resolve(observations=observations)
    assert geometry.source_row_count == 4
    assert geometry.scale == (6., 4.)
    assert geometry.source_digest != resolve()[0].source_digest
    changed = batch({"a": [2., 5., 8., 99.], "b": [8., 6., 4., np.nan]},
                    status=["success", "success", "success", "failed"])
    assert resolve(observations=changed)[0].source_digest != geometry.source_digest
    with pytest.raises(EngineError) as caught:
        resolve(observations=batch(status=["success", "warning", "failed"]))
    assert caught.value.code == ErrorCode.INSUFFICIENT_DATA


def test_digest_covers_space_semantics_and_source_data_is_independent():
    source = pd.DataFrame({"b": [8., 6., 4.], "a": [2., 5., 8.]})
    observations = batch(source)
    geometry, _ = resolve(observations=observations)
    source.iloc[0, 0] = 100.
    detached = observations.Y
    detached.iloc[0, 0] = 200.
    assert resolve(observations=observations)[0] == geometry
    other = space_for(Objective("a", "maximize", unit="g"), Objective("b", "minimize"))
    assert resolve(other)[0].source_digest != geometry.source_digest
    assert resolve(observations=batch(x=[0., 1., 3.]))[0].source_digest != geometry.source_digest


def test_target_range_is_scoped_soft_unweighted_and_has_separate_reference():
    geometry, runtime = resolve(knowledge=range_knowledge(), config=MultiObjectiveConfig(preference_weights={"a": 9., "b": 1.}))
    samples = torch.tensor([[2., 8.], [5., 6.], [8., 4.]], dtype=torch.double, requires_grad=True)
    report = api()[1](samples, space_for(), geometry, runtime, with_preferences=False)
    acquisition = api()[1](samples, space_for(), geometry, runtime, with_preferences=True)
    torch.testing.assert_close(acquisition - report, torch.tensor([[-1/3, 0.], [0., 0.], [-1/3, 0.]], dtype=torch.double))
    assert geometry.report_reference == pytest.approx((-.1, -.1))
    assert geometry.acquisition_reference == pytest.approx((-1/3 - .1, -.1))
    assert geometry.acquisition_ideal == pytest.approx((2/3, 1.))
    assert torch.autograd.gradcheck(lambda s: api()[1](s, space_for(), geometry, runtime, with_preferences=True), (samples,))
    duplicate, dup_runtime = resolve(knowledge=range_knowledge(duplicate=True), config=MultiObjectiveConfig(preference_weights={"a": 9., "b": 1.}))
    assert duplicate.acquisition_reference == geometry.acquisition_reference
    torch.testing.assert_close(api()[1](samples, space_for(), duplicate, dup_runtime, with_preferences=True), acquisition)


def test_physical_reference_gets_scoped_target_penalty():
    geometry, _ = resolve(knowledge=range_knowledge(), config=MultiObjectiveConfig(reference_point={"a": 0., "b": 10.}))
    assert geometry.report_reference == pytest.approx((-1/3, -.5))
    assert geometry.acquisition_reference == pytest.approx((-1., -.5))


def test_priorities_and_equivalent_sources_merge_provenance_in_precedence_order():
    space = space_for(Objective("a", "maximize", priority=0), Objective("b", "minimize", priority=1))
    knowledge = Knowledge().with_tradeoff("b", "a", first_weight=1., second_weight=2.)
    knowledge.add(make_pattern_spec(pattern_id="same", pattern="objective_priority", version="1.0",
        scope=KnowledgeScope(objectives=("a", "b")), parameters={"weights": [2., 1.]}, confidence=.8))
    geometry, runtime = resolve(space, knowledge=knowledge, config=MultiObjectiveConfig(preference_weights={"b": 3., "a": 6.}))
    assert geometry.preference_weights == pytest.approx((2/3, 1/3))
    assert [p["source"] for p in geometry.preference_provenance] == ["config", "knowledge", "knowledge", "objective_priority"]
    assert [p["source_pattern_id"] for p in geometry.preference_provenance if p["source"] == "knowledge"] == [p.source_pattern_id for p in runtime["acquisition_preferences"]]
    assert resolve(space)[0].preference_weights == pytest.approx((2/3, 1/3))
    assert resolve()[0].preference_weights is None
    assert resolve(config=MultiObjectiveConfig(preference_weights={"a": 1., "b": 1.}))[0].preference_weights == (.5, .5)


@pytest.mark.parametrize("source", ["config_knowledge", "config_priority", "knowledge_priority"])
def test_conflicting_preference_sources_never_silently_override(source):
    space = space_for(Objective("a", "maximize"), Objective("b", "minimize", priority=1 if "priority" in source else 0))
    knowledge = Knowledge().with_tradeoff("a", "b", first_weight=1., second_weight=1.) if "knowledge" in source else Knowledge()
    weights = {"a": 1., "b": 2.} if "config" in source else None
    with pytest.raises(EngineError) as caught:
        resolve(space, knowledge=knowledge, config=MultiObjectiveConfig(preference_weights=weights))
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert "sources" in caught.value.details


def test_nonconstant_unit_scaling_preserves_geometry_with_scaled_reference_noise():
    original = batch(yvar=pd.DataFrame({"b": [.4, .6, .8], "a": [.1, .2, .3]}))
    config = MultiObjectiveConfig(reference_point={"a": 0., "b": 10.})
    first, runtime = resolve(observations=original, config=config)
    scaled = batch(original.Y * pd.Series({"a": 1000., "b": .01}),
                   yvar=original.Yvar * pd.Series({"a": 1e6, "b": .0001}))
    second, _ = resolve(observations=scaled, config=MultiObjectiveConfig(reference_point={"a": 0., "b": .1}))
    assert first.report_reference == pytest.approx(second.report_reference)
    samples = torch.tensor([[3., 7.], [9., 2.]], dtype=torch.double)
    torch.testing.assert_close(api()[1](samples, space_for(), first, runtime, with_preferences=False),
        api()[1](samples * torch.tensor([1000., .01], dtype=torch.double), space_for(), second, runtime, with_preferences=False))
    pd.testing.assert_frame_equal(original.Yvar, pd.DataFrame({"b": [.4, .6, .8], "a": [.1, .2, .3]}))


def test_strict_json_roundtrip_is_detached_including_provenance():
    geometry, _ = resolve(config=MultiObjectiveConfig(reference_point={"a": 0., "b": 10.}, preference_weights={"a": 2., "b": 1.}))
    payload = geometry.to_dict()
    restored = api()[0].from_dict(json.loads(json.dumps(payload, allow_nan=False)))
    assert restored == geometry
    payload["scale"][0] = 999.
    payload["physical_reference"]["a"] = 99.
    payload["preference_provenance"][0]["source"] = "changed"
    assert restored == geometry
    with pytest.raises(TypeError):
        geometry.preference_provenance[0]["source"] = "changed"


@pytest.mark.parametrize("field,value", [
    ("scale", [0., 1.]), ("scale", [1.]), ("location", [True, 0.]),
    ("report_reference", [float("nan"), 0.]), ("acquisition_reference", [0., float("inf")]),
    ("acquisition_ideal", [0., False]), ("objective_names", ["a", "a"]),
    ("preference_weights", [1., 0.]), ("preference_weights", [1., 1.]),
    ("preference_weights", [1e308, 1e308]),
    ("source_digest", "invalid"), ("source_row_count", True), ("source_row_count", 2),
    ("reference_method", "guessed"), ("physical_reference", {"a": 0., "wrong": 9.}),
    ("preference_provenance", [{"source": "unknown"}]),
])
def test_geometry_rejects_corrupt_checkpoint_fields(field, value):
    payload = resolve()[0].to_dict()
    payload[field] = value
    with pytest.raises(EngineError) as caught:
        api()[0].from_dict(payload)
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_geometry_rejects_missing_and_unknown_fields():
    payload = resolve()[0].to_dict()
    for bad in ({**payload, "extra": 1}, {k: v for k, v in payload.items() if k != "scale"}):
        with pytest.raises(EngineError):
            api()[0].from_dict(bad)


def test_samples_require_geometry_order_and_correct_finite_output_axis():
    geometry, runtime = resolve()
    wrong = space_for(Objective("b", "minimize"), Objective("a", "maximize"))
    for samples, space in [(torch.ones(3, 2), wrong), (torch.ones(3, 1), space_for()),
                           (torch.tensor([[float("nan"), 1.]]), space_for())]:
        with pytest.raises(EngineError):
            api()[1](samples, space, geometry, runtime, with_preferences=False)


@pytest.mark.parametrize("target", [5., (3., 6.)])
def test_target_sample_gradients_and_float32_preserved(target):
    space = space_for(Objective("a", "target", target=target), Objective("b", "minimize"))
    geometry, runtime = resolve(space)
    samples = torch.tensor([[1., 8.], [4., 6.], [9., 4.]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda s: api()[1](s, space, geometry, runtime, with_preferences=False), (samples,))
    value = api()[1](samples.float(), space, geometry, runtime, with_preferences=False)
    assert value.dtype == torch.float32
    assert value.device == samples.device


def test_digest_is_name_ordered_including_variance_and_tracks_noise_and_ids():
    data = batch(yvar=pd.DataFrame({"b": [.4, .6, .8], "a": [.1, .2, .3]}))
    original, _ = resolve(observations=data)
    reordered = ObservationBatch(data.X, data.Y[["a", "b"]], data.Yvar[["a", "b"]])
    assert resolve(observations=reordered)[0].source_digest == original.source_digest
    with_ids = ObservationBatch(data.X, data.Y, data.Yvar, ids=("one", "two", "three"))
    assert resolve(observations=with_ids)[0].source_digest != original.source_digest
    noisy = ObservationBatch(data.X, data.Y, data.Yvar * 2.)
    assert resolve(observations=noisy)[0].source_digest != original.source_digest
    reversed_space = space_for(Objective("b", "minimize"), Objective("a", "maximize"))
    assert resolve(reversed_space, data)[0].source_digest == original.source_digest


@pytest.mark.parametrize("field,values", [
    ("reference_point", {"a": float("inf"), "b": 10.}),
    ("reference_point", {"a": True, "b": 10.}),
    ("reference_point", {"wrong": 0., "b": 10.}),
    ("preference_weights", {"a": 0., "b": 1.}),
    ("preference_weights", {"a": 1.}),
])
def test_invalid_config_cannot_resolve_geometry(field, values):
    with pytest.raises(EngineError) as caught:
        resolve(config=MultiObjectiveConfig(**{field: values}))
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_equal_nonzero_priorities_are_not_explicit_preferences():
    space = space_for(Objective("a", "maximize", priority=3), Objective("b", "minimize", priority=3))
    geometry, _ = resolve(space)
    assert geometry.preference_weights is None
    assert geometry.preference_provenance == ()


@pytest.mark.parametrize("constraints", [
    [OutcomeConstraint("equal", "b", "==", 5.)],
    [OutcomeConstraint("soft", "b", "<=", 5., hard=False, penalty="linear", weight=1.)],
    [OutcomeConstraint("lower", "b", ">=", 8.), OutcomeConstraint("upper", "b", "<=", 4.)],
])
def test_unsupported_or_empty_outcome_safety_is_not_silently_ignored(constraints):
    with pytest.raises(EngineError) as caught:
        resolve(space_for(outcome_constraints=constraints))
    assert caught.value.code in (ErrorCode.CONFIG_INVALID, ErrorCode.ACQUISITION_INCOMPATIBLE)
