"""Runtime consumers must either preserve an enabled source or reject it."""
import json

import pandas as pd
import pytest
import torch

from expdoe_dk import (
    EngineError, ErrorCode, Knowledge, KnowledgeScope, Parameter, Space,
    candidate_keys, inspect_capabilities, make_pattern_spec, suggest_design,
)
from expdoe_dk.bo.knowledge_runtime import compile_runtime
from expdoe_dk.bo.gp import build_gp, build_kernel_from_artifacts, build_mean_from_artifacts
from expdoe_dk.knowledge.artifacts import OptimizationArtifact, OptimizationArtifacts
from expdoe_dk.knowledge.registry import KnowledgePatternDefinition, PatternRegistry
from expdoe_dk.knowledge.guard import CompatibilityResult, KnowledgeValidationResult


def _provider_case(space, category, kind, payload, state="valid"):
    registry = PatternRegistry()
    registry.register(KnowledgePatternDefinition(pattern="test_runtime", version="1.0", family="test",
        schema={"type": "object"}, compatibility=lambda spec, space: CompatibilityResult(True),
        validator=lambda spec, space, observations=None: KnowledgeValidationResult(spec.pattern_id, state, "test"),
        renderer=lambda spec, space: "test",
        compiler=lambda spec, space, observations=None: OptimizationArtifacts(**{category: (
            OptimizationArtifact(kind, payload, spec.pattern_id, spec.pattern, spec.version),)})))
    return Knowledge(registry).add_spec(make_pattern_spec(pattern_id="provider-source",
        pattern="test_runtime", version="1.0", parameters={}, scope=KnowledgeScope(), confidence=1.))


def test_no_enabled_artifact_is_dropped(pattern_cases):
    for space, knowledge in pattern_cases:
        report = inspect_capabilities(space, knowledge)
        if report.blocked:
            with pytest.raises(EngineError):
                compile_runtime(space, knowledge, None)
        else:
            runtime = compile_runtime(space, knowledge, None)
            assert sum(len(v) for v in runtime.values()) > 0
            assert all(a.source_pattern_id == knowledge.specs[0].pattern_id
                       for items in runtime.values() for a in items)


def test_design_diagnostics_and_safety(mixed_case):
    space = mixed_case
    batch = suggest_design(space, 8, seed=17, return_diagnostics=True)
    d = batch.diagnostics.to_dict()
    assert isinstance(candidate_keys(batch.frame, space), set)
    assert len(candidate_keys(batch.frame, space)) == 8
    assert {"feasibility", "constraint_margins", "coverage", "duplicates",
            "degradation_reason", "marginal_balance"} <= d.keys()
    assert d["duplicates"] == 0
    assert d["feasibility"] == [True] * 8
    assert d["degradation_reason"] is None
    assert d["candidate_keys"] == [next(iter(candidate_keys(batch.frame.iloc[[i]], space)))
                                    for i in range(8)]
    assert d["coverage"]["x"]["definition"] == "physical_range_fraction"
    assert d["coverage"]["material"]["definition"] == "occupied_declared_levels_fraction"
    json.dumps(d, allow_nan=False)


@pytest.mark.parametrize("pattern", ["arrhenius", "monotone", "quadratic_peak",
    "random_augment", "gp_prior", "safe_region", "forbidden_region", "target_range",
    "power_law", "exponential", "quadratic_valley", "periodic"])
def test_supported_patterns_have_applied_ids(pattern_cases, pattern):
    space, knowledge = next(pair for pair in pattern_cases if pair[1].specs[0].pattern == pattern)
    assert inspect_capabilities(space, knowledge).applied == (f"KP-{pattern}",)
    assert compile_runtime(space, knowledge, None)


def test_hard_knowledge_constraints_drive_initial_design():
    space = Space([Parameter("x", bounds=(0.0, 1.0))], objectives="y")
    knowledge = Knowledge().with_safe_region(expression="x <= 0.25", factors=("x",))
    runtime = compile_runtime(space, knowledge, None)
    batch = suggest_design(space, 8, method="sobol", seed=17,
        parameter_constraints=runtime["parameter_constraints"], return_diagnostics=True)
    assert (batch.frame.x <= 0.25).all()
    d = batch.diagnostics.to_dict()
    source = knowledge.specs[0].pattern_id
    assert source in d["knowledge_sources"]
    margins = d["constraint_margins"][f"knowledge-{source}"]
    assert margins == pytest.approx((0.25 - batch.frame.x).tolist())
    assert d["feasibility"] == [True] * 8


def test_conflicting_safety_is_not_silently_dropped():
    space = Space([Parameter("x", bounds=(0.0, 1.0))], objectives="y")
    knowledge = Knowledge().with_safe_region(expression="x <= 0.25", factors=("x",))
    knowledge.with_safe_region(expression="x >= 0.75", factors=("x",))
    with pytest.raises(EngineError) as error:
        compile_runtime(space, knowledge, None)
    assert error.value.code == ErrorCode.KNOWLEDGE_CONFLICT


@pytest.mark.parametrize("fixture", ["unsupported_artifact_case", "wrong_category_artifact_case"])
def test_unknown_artifacts_fail_closed(request, fixture):
    space, knowledge = request.getfixturevalue(fixture)
    with pytest.raises(EngineError) as error:
        compile_runtime(space, knowledge, None)
    assert error.value.code == ErrorCode.KNOWLEDGE_INVALID
    assert "KP-unknown" in str(error.value.details)


def _similarity(matrix, levels=("A", "B", "C")):
    space = Space([Parameter("material", kind="categorical", values=("A", "B", "C"))], objectives="y")
    knowledge = Knowledge().add_spec(make_pattern_spec(pattern_id="similarity",
        pattern="category_similarity", version="1.0", confidence=1.0,
        scope=KnowledgeScope(factors=("material",)),
        parameters={"levels": list(levels), "matrix": matrix}))
    return space, knowledge


def test_non_psd_similarity_rejected_explicitly():
    space, knowledge = _similarity([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    assert inspect_capabilities(space, knowledge).blocked == ("similarity",)
    with pytest.raises(EngineError) as error:
        compile_runtime(space, knowledge, None)
    assert error.value.code == ErrorCode.KNOWLEDGE_INVALID
    assert "positive semidefinite" in str(error.value.details)


def test_partial_similarity_has_no_fabricated_missing_covariances():
    space, knowledge = _similarity([[1, 0.5], [0.5, 1]], levels=("A", "B"))
    with pytest.raises(EngineError):
        compile_runtime(space, knowledge, None)


def test_similarity_kernel_consumes_matrix_with_level_order_and_gp():
    space, knowledge = _similarity([[1, 0.4, 0.2], [0.4, 1, 0.3], [0.2, 0.3, 1]],
                                   levels=("C", "A", "B"))
    runtime = compile_runtime(space, knowledge, None)
    assert inspect_capabilities(space, knowledge).applied == ("similarity",)
    kernel = build_kernel_from_artifacts(space, runtime["kernel_components"])
    X = torch.tensor([[0.], [1.], [2.]], dtype=torch.double)
    expected = torch.tensor([[1, .3, .4], [.3, 1, .2], [.4, .2, 1]], dtype=torch.double)
    torch.testing.assert_close(kernel(X).to_dense(), expected)
    torch.testing.assert_close(kernel(X, diag=True), torch.ones(3, dtype=torch.double))
    model, _ = build_gp(space, knowledge, X, torch.tensor([[-1.], [0.], [1.]], dtype=torch.double))
    torch.testing.assert_close(model.covar_module(X).to_dense(), expected)


def test_conflicting_priors_rejected_instead_of_last_wins(mixed_case):
    knowledge = Knowledge().with_gp_prior("weak").with_gp_prior("strong")
    with pytest.raises(EngineError) as error:
        compile_runtime(mixed_case, knowledge, None)
    assert error.value.code == ErrorCode.KNOWLEDGE_CONFLICT


def test_monotone_auto_epsilon_resolves_and_opposite_directions_conflict(mixed_case):
    knowledge = Knowledge().with_monotone("x", "increases_objective")
    runtime = compile_runtime(mixed_case, knowledge, None)
    assert isinstance(runtime["virtual_observations"][0].payload["epsilon"], float)
    knowledge.with_monotone("x", "decreases_objective")
    with pytest.raises(EngineError) as error:
        compile_runtime(mixed_case, knowledge, None)
    assert error.value.code == ErrorCode.KNOWLEDGE_CONFLICT


def test_outcome_range_and_preference_use_physical_outcome_and_utility(pattern_cases):
    space, knowledge = next(pair for pair in pattern_cases if pair[1].specs[0].pattern == "target_range")
    consumer = compile_runtime(space, knowledge, None)["outcome_constraints"][0]
    assert consumer.hard is False
    assert consumer.satisfied({"y": 0.5})
    assert not consumer.satisfied({"y": 1.0})
    assert consumer.margin({"y": 1.0}) == pytest.approx(-0.2)
    priority = Knowledge().add_spec(make_pattern_spec(pattern="objective_priority", version="1.0",
        scope=KnowledgeScope(objectives=("y",)), confidence=1.0, parameters={"weights": [3.0]}))
    preference = compile_runtime(space, priority, None)["acquisition_preferences"][0]
    assert preference({"y": 2.5}) == pytest.approx(2.5)


def test_gp_rejects_unsupported_instead_of_ignoring_it(mixed_case):
    knowledge = Knowledge().with_interaction("x", "z", kind="synergy")
    X = mixed_case.physical_to_model(pd.DataFrame({"x": [2., 3.], "z": [2., 3.],
        "count": [1, 2], "grade": ["low", "high"], "material": ["A", "B"]}))
    with pytest.raises(EngineError):
        build_gp(mixed_case, knowledge, X, torch.tensor([[0.], [1.]], dtype=torch.double))


@pytest.mark.parametrize("pattern,parameters,expected", [
    ("power_law", {"exponent": 2., "scale": 3.}, [12., 48.]),
    ("exponential", {"rate": 0., "amplitude": 3.}, [3., 3.]),
    ("quadratic_valley", {"center": 2., "width": 2.}, [0., 1.]),
    ("periodic", {"period": 8., "phase": 0.}, [1., 0.]),
])
@pytest.mark.parametrize("maximize,sign", [(False, 1.), (True, -1.)])
def test_numerical_means_decode_log_input_before_response_normalization(pattern, parameters, expected, maximize, sign):
    space = Space([Parameter("x", bounds=(1., 10.), transform="log")], objectives="y", maximize=maximize)
    knowledge = Knowledge().add_spec(make_pattern_spec(pattern=pattern, version="1.0",
        scope=KnowledgeScope(factors=("x",)), confidence=1., parameters=parameters))
    runtime = compile_runtime(space, knowledge, None)
    mean = build_mean_from_artifacts(space, runtime["mean_components"], response_location=1., response_scale=2.)
    X = space.physical_to_model(pd.DataFrame({"x": [2., 4.]}))
    torch.testing.assert_close(mean(X), sign * (torch.tensor(expected, dtype=torch.double) - 1.) / 2.)


@pytest.mark.parametrize("category,kind", [("input_transforms", "unknown_transform"),
    ("outcome_transforms", "unknown_transform"), ("mean_components", "power_law_descriptor"),
    ("priors", "gp_prior"), ("virtual_observations", "monotone")])
def test_malformed_or_unknown_provider_artifacts_fail_closed(mixed_case, category, kind):
    knowledge = _provider_case(mixed_case, category, kind, {})
    assert inspect_capabilities(mixed_case, knowledge).blocked == ("provider-source",)
    with pytest.raises(EngineError) as error:
        compile_runtime(mixed_case, knowledge, None)
    assert error.value.code == ErrorCode.KNOWLEDGE_INVALID
    assert "provider-source" in str(error.value.details)


def test_insufficient_data_does_not_skip_hard_safety():
    space = Space([Parameter("x", bounds=(0., 1.))], objectives="y")
    original = Knowledge().with_safe_region(expression="x <= 0.25", factors=("x",))
    artifact = original.compile(space).parameter_constraints[0]
    knowledge = _provider_case(space, "parameter_constraints", artifact.kind,
                               artifact.to_dict()["payload"], state="insufficient_data")
    runtime = compile_runtime(space, knowledge, None)
    assert len(runtime["parameter_constraints"]) == 1
    assert not runtime["parameter_constraints"][0].satisfied({"x": .8})
    assert runtime["diagnostics"][0].payload["state"] == "insufficient_data"


def test_skipped_safety_diagnostic_is_rejected(mixed_case):
    knowledge = _provider_case(mixed_case, "diagnostics", "knowledge_validation", {"state": "skipped"})
    with pytest.raises(EngineError):
        compile_runtime(mixed_case, knowledge, None)


def test_combined_physical_means_normalize_location_once():
    space = Space([Parameter("x", bounds=(1., 10.))], objectives="y")
    knowledge = Knowledge()
    for pattern, parameters in (("power_law", {"exponent": 1., "scale": 1.}),
                               ("exponential", {"rate": 0., "amplitude": 2.})):
        knowledge.add_spec(make_pattern_spec(pattern=pattern, version="1.0", parameters=parameters,
            confidence=1., scope=KnowledgeScope(factors=("x",))))
    artifacts = compile_runtime(space, knowledge, None)["mean_components"]
    mean = build_mean_from_artifacts(space, artifacts, response_location=1., response_scale=2.)
    X = space.physical_to_model(pd.DataFrame({"x": [3.]}))
    assert mean(X).item() == pytest.approx(-2.)  # internal minimization: -(3 + 2 - 1) / 2


def test_gp_does_not_guess_physical_response_normalization(pattern_cases):
    space, knowledge = next(pair for pair in pattern_cases if pair[1].specs[0].pattern == "power_law")
    X = torch.zeros(3, space.n_dims, dtype=torch.double)
    Y = torch.tensor([[-1.], [0.], [1.]], dtype=torch.double)
    with pytest.raises(EngineError) as error:
        build_gp(space, knowledge, X, Y)
    assert error.value.code == ErrorCode.CONFIG_INVALID
    model, _ = build_gp(space, knowledge, X, Y, response_location=10., response_scale=2.)
    assert model.mean_module(X)[0].item() == pytest.approx(4.5)


def test_finite_coverage_counts_only_admissible_levels():
    space = Space([Parameter("x", bounds=(0, 2), kind="integer")], objectives="y")
    knowledge = Knowledge().with_safe_region(expression="x >= 1", factors=("x",))
    runtime = compile_runtime(space, knowledge, None)
    batch = suggest_design(space, 2, method="random_uniform",
        parameter_constraints=runtime["parameter_constraints"], return_diagnostics=True)
    coverage = batch.diagnostics.to_dict()["coverage"]["x"]
    assert coverage["definition"] == "occupied_admissible_levels_fraction"
    assert coverage["value"] == 1.
    assert coverage["admissible"] == 2


def test_large_integer_domain_diagnostics_do_not_expand_unoccupied_levels():
    space = Space([Parameter("x", bounds=(0., 1.)),
                   Parameter("count", bounds=(0, 100001), kind="integer")], objectives="y")
    batch = suggest_design(space, 2, method="random_uniform", return_diagnostics=True)
    d = batch.diagnostics.to_dict()
    assert d["coverage"]["count"]["declared"] == 100002
    assert len(d["marginal_balance"]["count"]["counts"]) <= 2
    assert d["marginal_balance"]["count"]["unoccupied_levels"] >= 100000
