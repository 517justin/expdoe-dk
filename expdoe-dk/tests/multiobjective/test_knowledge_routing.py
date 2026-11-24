"""Objective routing must not leak physical assumptions or weaken validation."""
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from expdoe_dk import (
    EngineError, ErrorCode, Knowledge, KnowledgeScope, Objective, ObservationBatch,
    OutcomeConstraint, Parameter, Space, inspect_capabilities, make_pattern_spec,
)
from expdoe_dk.bo.knowledge_runtime import compile_runtime, validate_runtime_artifact
from expdoe_dk.knowledge.artifacts import OptimizationArtifact
from expdoe_dk.knowledge.registry import PatternRegistry, load_pattern_providers


@pytest.fixture
def space():
    return Space([Parameter("x", bounds=(1., 10.), transform="log", unit="K"),
                  Parameter("material", kind="categorical", values=("A", "B"))],
                 objectives=(Objective("yield", "maximize"), Objective("cost", "minimize")))


@pytest.fixture
def observations():
    return ObservationBatch(
        pd.DataFrame({"x": [1., 3., 7.], "material": ["A", "B", "A"]}),
        pd.DataFrame({"cost": [8., 6., 4.], "yield": [2., 5., 8.]}),
        Yvar=pd.DataFrame({"cost": [.8, .6, .4], "yield": [.2, .5, .8]}),
        status=pd.Series(["success", "warning", "success"]), ids=("a", "b", "c"))


def spec(pattern="power_law", *, identity="yield-shape", objectives=("yield",),
         factors=("x",), parameters=None, version="1.0", **scope):
    return make_pattern_spec(pattern_id=identity, pattern=pattern, version=version,
        scope=KnowledgeScope(factors=factors, objectives=objectives, **scope),
        parameters={"exponent": 2., "scale": 3.} if parameters is None else parameters,
        confidence=.8)


def project(space, observations, knowledge, name):
    from expdoe_dk.bo.objective_routing import project_objective
    return project_objective(space, observations, knowledge, name)


def test_scoped_mean_does_not_leak_to_other_response(space, observations):
    knowledge = Knowledge().add(spec())
    assert inspect_capabilities(space, knowledge).applied == ("yield-shape",)
    for name, expected in (("yield", ["yield-shape"]), ("cost", [])):
        projected, batch, selected = project(space, observations, knowledge, name)
        runtime = compile_runtime(projected, selected, batch)
        assert [a.source_pattern_id for a in runtime["mean_components"]] == expected


def test_v2_projection_retains_provenance(space, observations):
    original = spec("saturation", version="2.0", identity="saturation-source",
        parameters={"direction": "increasing", "half_response": 4., "origin": 1.})
    knowledge = Knowledge().add(original)
    assert inspect_capabilities(space, knowledge).blocked == ()
    projected, batch, selected = project(space, observations, knowledge, "yield")
    assert selected.specs[0] == original
    artifact = compile_runtime(projected, selected, batch)["mean_components"][0]
    assert (artifact.source_pattern_id, artifact.source_version) == ("saturation-source", "2.0")
    assert artifact.payload["objectives"] == ("yield",)
    assert not compile_runtime(space, knowledge, observations)["mean_components"]


def test_projection_preserves_named_data_and_encoding(space, observations):
    reordered = Space(space.params, objectives=tuple(reversed(space.objective_specs)))
    for source in (space, reordered):
        projected, batch, selected = project(source, observations, Knowledge().add(spec()), "yield")
        assert projected.objectives == ["yield"]
        assert projected.params == source.params
        assert projected.objective_specs == (source.objective_specs[source.objectives.index("yield")],)
        pd.testing.assert_frame_equal(batch.X, observations.X)
        pd.testing.assert_frame_equal(batch.Y, pd.DataFrame({"yield": [2., 5., 8.]}))
        pd.testing.assert_frame_equal(batch.Yvar, pd.DataFrame({"yield": [.2, .5, .8]}))
        pd.testing.assert_series_equal(batch.status, observations.status)
        assert batch.ids == ("a", "b", "c")
        assert selected.specs[0].pattern_id == "yield-shape"


@pytest.mark.parametrize("pattern,parameters,factors,category", [
    ("power_law", {"exponent": 2., "scale": 3.}, ("x",), "mean_components"),
    ("arrhenius", {"frozen": True, "activation_energy": 2., "amplitude_init": -1.}, ("x",), "mean_components"),
    ("gp_prior", {"lengthscale": "weak"}, (), "priors"),
    ("monotone", {"direction": "increasing", "epsilon": "auto", "delta_norm": .5, "n_pairs_per_dim": 3}, ("x",), "virtual_observations"),
    ("random_augment", {"n": 5}, (), "virtual_observations"),
])
def test_response_patterns_require_scope_and_project_each_named_output(space, observations, pattern, parameters, factors, category):
    unscoped = Knowledge().add(spec(pattern, objectives=(), parameters=parameters, factors=factors))
    report = inspect_capabilities(space, unscoped)
    assert report.blocked == ("yield-shape",)
    assert any(d.get("reason") == "missing_objective_scope" for d in report.diagnostics)
    with pytest.raises(EngineError) as caught:
        compile_runtime(space, unscoped, observations)
    assert "yield-shape" in str(caught.value.details)
    assert "objective" in str(caught.value.details)
    scoped = Knowledge().add(spec(pattern, objectives=("cost", "yield"), parameters=parameters, factors=factors))
    for name in ("yield", "cost"):
        projected, batch, selected = project(space, observations, scoped, name)
        assert selected.specs[0].scope.objectives == (name,)
        assert compile_runtime(projected, selected, batch)[category][0].source_pattern_id == "yield-shape"


def test_global_safety_and_preferences_do_not_enter_response_models(space, observations):
    knowledge = Knowledge().with_safe_region(expression="x <= 8", factors=("x",))
    knowledge.add(spec("target_range", identity="yield-range", factors=(), parameters={"lower": 2., "upper": 6.}))
    knowledge.with_tradeoff("yield", "cost", first_weight=3., second_weight=1.)
    runtime = compile_runtime(space, knowledge, observations)
    assert len(runtime["parameter_constraints"]) == 1
    assert not runtime["parameter_constraints"][0].satisfied({"x": 9.})
    assert runtime["outcome_constraints"][0].margin({"yield": 8.}) == -2.
    assert runtime["acquisition_preferences"][0]({"yield": 4., "cost": 8.}) == 5.
    for name in space.objectives:
        projected, batch, selected = project(space, observations, knowledge, name)
        assert selected.specs == ()
        assert projected.outcome_constraints == []


def test_shared_similarity_enters_each_model(space, observations):
    knowledge = Knowledge().add(spec("category_similarity", identity="similarity", objectives=(),
        factors=("material",), parameters={"levels": ["A", "B"], "matrix": [[1., .3], [.3, 1.]]}))
    assert inspect_capabilities(space, knowledge).applied == ("similarity",)
    for name in space.objectives:
        projected, batch, selected = project(space, observations, knowledge, name)
        assert compile_runtime(projected, selected, batch)["kernel_components"][0].source_pattern_id == "similarity"


@pytest.mark.parametrize("kwargs,reason", [
    ({"objectives": ("missing",)}, "unknown_objective_scope"),
    ({"conditions": {"material": "A"}}, "unsupported_conditional_scope"),
    ({"region": {"x": [1., 3.]}}, "unsupported_conditional_scope"),
])
def test_scope_errors_fail_closed_with_source_and_objective(space, observations, kwargs, reason):
    knowledge = Knowledge().add(spec(**kwargs))
    report = inspect_capabilities(space, knowledge)
    assert report.blocked == ("yield-shape",)
    assert report.diagnostics[0]["reason"] == reason
    assert "objective_name" in report.diagnostics[0]
    assert "scope" in report.diagnostics[0]
    with pytest.raises(EngineError) as caught:
        project(space, observations, knowledge, "cost")
    assert "yield-shape" in str(caught.value.details)


@pytest.mark.parametrize("pattern,parameters", [
    ("saturation", {"direction": "increasing", "half_response": 4.}),
    ("optimum_range", {"lower": 2., "upper": 5.}),
])
def test_unsupported_descriptors_stay_blocked(space, observations, pattern, parameters):
    knowledge = Knowledge().add(spec(pattern, parameters=parameters))
    assert inspect_capabilities(space, knowledge).blocked == ("yield-shape",)
    with pytest.raises(EngineError) as caught:
        project(space, observations, knowledge, "yield")
    assert "yield-shape" in str(caught.value.details)


def test_v2_unrelated_validation_is_not_disabled(space, observations):
    knowledge = Knowledge().add(spec("saturation", version="2.0",
        parameters={"direction": "increasing", "half_response": 4., "origin": 0.}))
    with pytest.raises(EngineError) as caught:
        compile_runtime(space, knowledge, observations)
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID
    assert "yield-shape" in str(caught.value.details)


@pytest.mark.parametrize("objectives,weights", [(("yield",), [1.]), (("yield", "cost"), [1., 0.])])
def test_preference_requires_all_outputs_and_positive_weights(space, objectives, weights):
    knowledge = Knowledge().add(spec("objective_priority", identity="priority", factors=(),
        objectives=objectives, parameters={"weights": weights}))
    with pytest.raises(EngineError) as caught:
        compile_runtime(space, knowledge, None)
    assert "priority" in str(caught.value.details)
    assert inspect_capabilities(space, knowledge).blocked == ("priority",)


@pytest.mark.parametrize("payload", [
    {}, {"objectives": ["yield", "cost"], "weights": [.5, .5], "extra": 1},
    {"objectives": ["yield", "yield"], "weights": [.5, .5]},
    {"objectives": ["yield", "cost"], "weights": [True, 0.]},
    {"objectives": ["yield", "cost"], "weights": [.5, .4]},
    {"objectives": ["yield"], "weights": [1.]},
])
def test_tradeoff_payload_is_strict(space, payload):
    artifact = OptimizationArtifact("tradeoff_preference", payload, "tradeoff-source", "provider", "1.0")
    with pytest.raises(EngineError) as caught:
        validate_runtime_artifact(space, "acquisition_preferences", artifact)
    assert caught.value.details["pattern_id"] == "tradeoff-source"


@pytest.mark.parametrize("kind", ["priority", "target_range", "prior", "structured", "safety"])
@pytest.mark.parametrize("bad_scope", [None, (), ("unknown",)])
def test_whole_declaration_conflicts_keep_original_ids(space, kind, bad_scope):
    if kind == "priority":
        specs = [spec("objective_priority", identity=f"source-{i}", objectives=("yield", "cost"),
                      factors=(), parameters={"weights": weights}) for i, weights in enumerate(([1., 2.], [2., 1.]))]
    elif kind == "target_range":
        specs = [spec("target_range", identity=f"source-{i}", factors=(),
                      parameters={"lower": lower, "upper": lower + 1}) for i, lower in enumerate((1., 5.))]
    elif kind == "prior":
        specs = [spec("gp_prior", identity=f"source-{i}", factors=(),
                      parameters={"lengthscale": strength}) for i, strength in enumerate(("weak", "strong"))]
    elif kind == "structured":
        specs = [spec("saturation", identity=f"source-{i}", version="2.0",
                      parameters={"direction": direction, "half_response": 4., "origin": 1.})
                 for i, direction in enumerate(("increasing", "decreasing"))]
    else:
        original = Knowledge().with_safe_region(expression="x <= 2", factors=("x",))
        original.with_safe_region(expression="x >= 8", factors=("x",))
        specs = [replace(s, pattern_id=f"source-{i}", parameters=s.to_dict()["parameters"])
                 for i, s in enumerate(original.specs)]
    knowledge = Knowledge()
    for declaration in specs:
        knowledge.add(declaration)
    with pytest.raises(EngineError) as caught:
        compile_runtime(space, knowledge, None)
    assert caught.value.code == ErrorCode.KNOWLEDGE_CONFLICT
    assert all(identity in str(caught.value.details) for identity in ("source-0", "source-1"))
    expected_blocked = {"source-0", "source-1"}
    if bad_scope is not None:
        knowledge.add(spec(identity="bad-scope", objectives=bad_scope))
        expected_blocked.add("bad-scope")
    knowledge.add(spec("gp_prior", identity="unrelated-valid", factors=(),
                       objectives=("cost",), parameters={"lengthscale": "weak"}))
    report = inspect_capabilities(space, knowledge)
    assert set(report.blocked) == expected_blocked
    assert report.applied == ("unrelated-valid",)
    conflicts = [d for d in report.diagnostics if d.get("code") == ErrorCode.KNOWLEDGE_CONFLICT.value]
    assert conflicts
    assert all(identity in str(conflicts) for identity in ("source-0", "source-1"))
    if bad_scope is not None:
        reason = "unknown_objective_scope" if bad_scope else "missing_objective_scope"
        assert any(d.get("pattern_id") == "bad-scope" and d.get("reason") == reason
                   for d in report.diagnostics)


def test_same_normalized_preferences_retain_both_sources(space):
    knowledge = Knowledge().add(spec("objective_priority", identity="priority", factors=(),
        objectives=("yield", "cost"), parameters={"weights": [3., 1.]}))
    knowledge.add(spec("tradeoff", identity="tradeoff", factors=(),
        objectives=("cost", "yield"), parameters={"weights": [2., 6.]}))
    runtime = compile_runtime(space, knowledge, None)
    assert [a.source_pattern_id for a in runtime["acquisition_preferences"]] == ["priority", "tradeoff"]
    assert [a({"yield": 4., "cost": 8.}) for a in runtime["acquisition_preferences"]] == [5., 5.]


def test_single_objective_projection_is_identity_and_runtime_unchanged(observations):
    single = Space([Parameter("x", bounds=(1., 10.))], objectives="yield")
    knowledge = Knowledge().with_gp_prior("weak")
    batch = ObservationBatch(observations.X[["x"]], observations.Y[["yield"]])
    projected, selected_batch, selected = project(single, batch, knowledge, "yield")
    assert projected is single and selected_batch is batch and selected is knowledge
    assert compile_runtime(single, knowledge, batch) == compile_runtime(projected, selected, selected_batch)


@pytest.mark.parametrize("pattern,parameters", [("gp_prior", {"lengthscale": "weak"}),
                                               ("random_augment", {"n": 5})])
def test_whole_model_patterns_reject_factor_subsets(space, observations, pattern, parameters):
    knowledge = Knowledge().add(spec(pattern, parameters=parameters, factors=("x",)))
    with pytest.raises(EngineError) as caught:
        project(space, observations, knowledge, "yield")
    assert caught.value.details["pattern_id"] == "yield-shape"
    assert caught.value.details["reason"] == "unsupported_factor_scope"


def test_safety_is_compiled_once_against_full_space(space, observations):
    class RecordingRegistry(PatternRegistry):
        def __init__(self):
            super().__init__()
            self.safety_calls = []

        def compile_many(self, specs, space, observations=None):
            specs = tuple(specs)
            if any(s.pattern == "safe_region" for s in specs):
                self.safety_calls.append(tuple(space.objectives))
            return super().compile_many(specs, space, observations)

    registry = RecordingRegistry()
    registry.register_many(Knowledge().registry.definitions())
    knowledge = Knowledge(registry=registry).with_safe_region(expression="x <= 8", factors=("x",))
    knowledge.add(spec(objectives=("yield", "cost")))
    runtime = compile_runtime(space, knowledge, observations)
    assert len(runtime["parameter_constraints"]) == 1
    for name in space.objectives:
        projected, batch, selected = project(space, observations, knowledge, name)
        compile_runtime(projected, selected, batch)
    assert registry.safety_calls == [("yield", "cost")]


@pytest.fixture
def provider_knowledge(monkeypatch):
    from expdoe_dk.knowledge.registry import providers

    definition = Knowledge().registry.resolve("power_law", "1.0")
    entry = SimpleNamespace(name="shapes", value="test_shapes:patterns",
        group="expdoe_dk.knowledge_patterns",
        dist=SimpleNamespace(name="test-shapes", version="1.0"),
        load=lambda: lambda: (definition,))
    monkeypatch.setattr(providers, "entry_points", lambda *, group: (entry,))
    registry = PatternRegistry()
    load_pattern_providers({"test-shapes"}, registry)
    return Knowledge(registry=registry).add(spec())


def test_provider_records_survive_projection_and_roundtrip(space, observations, provider_knowledge):
    knowledge = Knowledge.from_envelope(provider_knowledge.to_envelope(), registry=provider_knowledge.registry)
    projected, batch, selected = project(space, observations, knowledge, "yield")
    assert selected.registry is knowledge.registry
    assert selected.to_envelope()["providers"] == knowledge.to_envelope()["providers"]
    restored = Knowledge.from_envelope(selected.to_envelope(), registry=knowledge.registry)
    assert compile_runtime(projected, restored, batch)["mean_components"][0].source_pattern_id == "yield-shape"


def test_projection_cannot_erase_unloaded_provider_requirement(space, observations, provider_knowledge):
    # Builtins alone have the same pattern name but not the recorded provider.
    restored = Knowledge.from_envelope(provider_knowledge.to_envelope())
    with pytest.raises(EngineError) as caught:
        project(space, observations, restored, "cost")
    assert caught.value.code == ErrorCode.EXTENSION_NOT_ALLOWED
    assert "yield-shape" in str(caught.value.details)
    assert "test-shapes" in str(caught.value.details)


def test_space_constraints_preserved_but_outcome_safety_remains_global(space, observations):
    from expdoe_dk import ExpressionConstraint

    constraint = ExpressionConstraint("input-safe", "x <= 8")
    safety = OutcomeConstraint("cost-limit", "cost", "<=", 9.)
    original = Space(space.params, constraints=(constraint,), objectives=space.objective_specs,
                     outcome_constraints=(safety,))
    projected, _, _ = project(original, observations, Knowledge(), "yield")
    assert projected.constraints == [constraint]
    assert projected.outcome_constraints == []
    assert original.outcome_constraints == [safety]


def test_distinct_response_priors_do_not_conflict(space, observations):
    knowledge = Knowledge().add(spec("gp_prior", factors=(), parameters={"lengthscale": "weak"}))
    knowledge.add(spec("gp_prior", identity="cost-prior", factors=(), objectives=("cost",),
                       parameters={"lengthscale": "strong"}))
    assert inspect_capabilities(space, knowledge).blocked == ()
    for name, expected in (("yield", (1.5, .5)), ("cost", (6., 15.))):
        projected, batch, selected = project(space, observations, knowledge, name)
        assert compile_runtime(projected, selected, batch)["priors"][0].payload["lengthscale"] == expected


def test_disabled_unscoped_unsupported_knowledge_does_not_block(space, observations):
    original = spec("saturation", objectives=(), parameters={"direction": "increasing", "half_response": 4.})
    disabled = replace(original, enabled=False, parameters=original.to_dict()["parameters"])
    knowledge = Knowledge().add(disabled)
    assert inspect_capabilities(space, knowledge).blocked == ()
    assert not any(compile_runtime(space, knowledge, observations).values())
    assert project(space, observations, knowledge, "yield")[2].specs == ()


@pytest.mark.parametrize("columns", [["yield"], ["yield", "cost", "extra"]])
def test_projection_rejects_incomplete_or_extra_response_columns(space, observations, columns):
    frame = observations.Y.assign(extra=1.)[columns]
    batch = ObservationBatch(observations.X, frame)
    with pytest.raises(EngineError) as caught:
        project(space, batch, Knowledge(), "yield")
    assert caught.value.code == ErrorCode.CONFIG_INVALID


def test_projection_rejects_unknown_objective(space, observations):
    with pytest.raises(EngineError) as caught:
        project(space, observations, Knowledge(), "absent")
    assert caught.value.code == ErrorCode.CONFIG_INVALID
    assert caught.value.details["objective_name"] == "absent"


def test_projected_mean_consumer_retains_physical_values_and_direction(space, observations):
    from expdoe_dk.bo.gp import build_mean_from_artifacts

    knowledge = Knowledge().add(spec(objectives=("yield", "cost")))
    for name, expected in (("yield", -13.), ("cost", 13.)):
        projected, batch, selected = project(space, observations, knowledge, name)
        runtime = compile_runtime(projected, selected, batch)
        mean = build_mean_from_artifacts(projected, runtime["mean_components"],
                                        response_location=1., response_scale=2.)
        # Physical x=3: (3 * 3**2 - 1) / 2 = 13, independent of log input coding.
        assert mean(projected.physical_to_model(batch.X.iloc[[1]])).item() == pytest.approx(expected)


def test_structured_and_legacy_composition_is_checked_per_response(space):
    structured = spec("saturation", identity="structured", version="2.0",
        parameters={"direction": "increasing", "half_response": 4., "origin": 1.})
    knowledge = Knowledge().add(structured).add(spec(identity="legacy", objectives=("cost",)))
    assert inspect_capabilities(space, knowledge).blocked == ()
    compile_runtime(space, knowledge, None)
    conflicting = Knowledge().add(structured).add(spec(identity="legacy"))
    with pytest.raises(EngineError) as caught:
        compile_runtime(space, conflicting, None)
    assert caught.value.details["reason"] == "unsupported_composition"
    assert set(caught.value.details["pattern_ids"]) == {"structured", "legacy"}
    assert caught.value.details["objective_name"] == "yield"


def test_narrow_tradeoff_is_blocked_with_three_objectives(space):
    three = Space(space.params, objectives=(*space.objective_specs, Objective("purity", "maximize")))
    knowledge = Knowledge().with_tradeoff("yield", "cost", first_weight=1., second_weight=1.)
    with pytest.raises(EngineError) as caught:
        compile_runtime(three, knowledge, None)
    assert knowledge.specs[0].pattern_id in str(caught.value.details)
    assert inspect_capabilities(three, knowledge).blocked == (knowledge.specs[0].pattern_id,)


@pytest.mark.parametrize("category,kind,payload", [
    ("priors", "gp_prior", {}),
    ("mean_components", "unknown_mean", {}),
    ("input_transforms", "unknown_transform", {}),
    ("mean_components", "gp_prior", {"lengthscale": [2., 3.], "outputscale": [2., 3.], "noise": [2., 3.]}),
])
def test_provider_artifacts_remain_fail_closed(space, category, kind, payload):
    from expdoe_dk.knowledge.artifacts import OptimizationArtifacts
    from expdoe_dk.knowledge.guard import CompatibilityResult, KnowledgeValidationResult
    from expdoe_dk.knowledge.registry import KnowledgePatternDefinition

    registry = PatternRegistry()
    registry.register(KnowledgePatternDefinition(pattern="provider", version="1.0", family="test",
        schema={"type": "object"}, compatibility=lambda s, sp: CompatibilityResult(True),
        validator=lambda s, sp, observations=None: KnowledgeValidationResult(s.pattern_id, "valid", "test"),
        renderer=lambda s, sp: "test",
        compiler=lambda s, sp, observations=None: OptimizationArtifacts(**{category: (
            OptimizationArtifact(kind, payload, s.pattern_id, s.pattern, s.version),)})))
    knowledge = Knowledge(registry=registry).add(spec("provider", identity="provider-source", parameters={}))
    assert inspect_capabilities(space, knowledge).blocked == ("provider-source",)
    with pytest.raises(EngineError) as caught:
        compile_runtime(space, knowledge, None)
    assert "provider-source" in str(caught.value.details)
    assert caught.value.details["objective_name"] == "yield"
