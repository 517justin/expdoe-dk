"""Structural guards and empirical minimums, never evidence of an effect."""
import json

import pandas as pd
import pytest

from expdoe_dk import (EngineError, ErrorCode, Knowledge, KnowledgeScope,
    ObservationBatch, Parameter, Space, inspect_capabilities, make_pattern_spec)
from expdoe_dk.bo.knowledge_runtime import compile_runtime, validate_runtime_artifact
from expdoe_dk.knowledge.artifacts import OptimizationArtifact, OptimizationArtifacts


@pytest.fixture
def space():
    return Space([Parameter(n, bounds=(0., 1.)) for n in ("x", "z", "t")],
                 objectives="response")


def declaration(pattern="saturation", factors=("x",), *, identity="shape", enabled=True,
                response="response", **parameters):
    defaults = {
        "saturation": {"direction": "increasing", "origin": 0., "half_response": .5},
        "threshold": {"threshold": .5, "below_behavior": "increasing", "above_behavior": "flat"},
        "synergy": {"reference": "anchored_additive", "response_effect": "higher"},
        "antagonism": {"reference": "anchored_additive", "response_effect": "higher"},
    }
    return make_pattern_spec(pattern_id=identity, pattern=pattern, version="2.0",
        scope=KnowledgeScope(factors=factors, objectives=(response,)),
        parameters=defaults[pattern] | parameters, confidence=1., enabled=enabled)


def knowledge(*specs):
    result = Knowledge()
    for spec in specs:
        result.add_spec(spec)
    return result


def batch(space, rows=None, status=None):
    if rows is None:
        rows = [[x, z, t] for x, z, t in
                [(0., 0., 0.), (.2, 1., .3), (.4, .3, 1.),
                 (.6, .8, .2), (.8, .1, .7), (1., .6, .8)]]
    return ObservationBatch(pd.DataFrame(rows, columns=space.param_names),
        pd.DataFrame({"response": [0.] * len(rows)}),
        status=None if status is None else pd.Series(status))


def coverage(space, observations, *specs, **kwargs):
    from expdoe_dk.bo.structured_coverage import check_structured_coverage
    return check_structured_coverage(space, observations,
        knowledge(*(specs or (declaration(),))).compile(space).mean_components, **kwargs)


def composition(space, declarations):
    from expdoe_dk.bo.structured_coverage import check_structured_composition
    check_structured_composition(space, declarations.compile(space))


def test_six_successes_pass_all_starts_without_nonzero_outcome_requirement(space):
    coverage_six_or_more = coverage(space, batch(space))
    assert coverage_six_or_more["eligible_starts"] == [0, 1, 2]
    assert coverage_six_or_more["coverage_status"] == "passed"
    assert coverage_six_or_more["parameter_identifiability"] == "not_certified"
    assert coverage_six_or_more["empirical_evidence"] == "not_evaluated"
    json.dumps(coverage_six_or_more, allow_nan=False)


def test_no_observations_do_not_imply_coverage_or_empirical_support(space):
    no_data_diagnostic = coverage(space, None)
    assert no_data_diagnostic["coverage_status"] == "not_evaluated"
    assert no_data_diagnostic["eligible_starts"] == []
    assert no_data_diagnostic["empirical_evidence"] == "not_evaluated"


@pytest.mark.parametrize("statuses", [None, ["success"] * 5 + ["warning"],
    ["success"] * 5 + ["failed"]])
def test_five_successes_are_insufficient_even_with_other_rows(space, statuses):
    rows = batch(space).X.values.tolist()
    with pytest.raises(EngineError) as caught:
        coverage(space, batch(space, rows[:5] if statuses is None else rows, statuses))
    insufficient_rows_error = caught.value
    assert insufficient_rows_error.code == ErrorCode.INSUFFICIENT_DATA
    assert insufficient_rows_error.details["successful_rows"] == 5
    assert insufficient_rows_error.details["pattern_ids"] == ["shape"]
    assert insufficient_rows_error.details["remediation"]


@pytest.mark.parametrize("pattern", ["saturation", "threshold"])
@pytest.mark.parametrize("values,left,right", [([.5, .5, .6, .7, .8, 1.], 0, 4),
    ([.2, .2, .2, .6, .8, 1.], 1, 3), ([0., .2, .5, .5, .8, .8], 2, 1)])
def test_strict_sides_and_distinct_values(space, pattern, values, left, right):
    with pytest.raises(EngineError) as caught:
        coverage(space, batch(space, [[x, 0., 0.] for x in values]), declaration(pattern))
    missing_left_values_error = caught.value
    assert missing_left_values_error.code == ErrorCode.INSUFFICIENT_DATA
    diagnostic, = missing_left_values_error.details["coverage"]
    assert diagnostic["left_distinct"] == left
    assert diagnostic["right_distinct"] == right
    assert diagnostic["pattern_id"] == "shape"


@pytest.mark.parametrize("rows", [ [[x, x, 0.] for x in (0., 0., 0., 1., 1., 1.)],
    [[x, 0., 0.] for x in (0., .2, .4, .6, .8, 1.)]])
def test_interaction_levels_and_diagonal_rank(space, rows):
    with pytest.raises(EngineError) as caught:
        coverage(space, batch(space, rows), declaration("synergy", ("x", "z")))
    diagonal_only_interaction_error = caught.value
    assert diagonal_only_interaction_error.code == ErrorCode.INSUFFICIENT_DATA
    assert diagonal_only_interaction_error.details["coverage"][0]["status"] == "deficient"
    json.dumps(caught.value.details, allow_nan=False)


@pytest.mark.parametrize("delta,passes", [(1e-8, True), (1e-12, False)])
def test_float64_relative_svd_cutoff(space, delta, passes):
    # Two almost-identical columns: smallest singular value is O(delta).
    rows = [[x, x + perturb * delta, 0.] for x, perturb in
            [(.1, 0), (.2, 1), (.3, -1), (.6, 1), (.7, -1), (.8, 0)]]
    spec = declaration("synergy", ("x", "z"))
    if passes:
        result = coverage(space, batch(space, rows), spec)
        assert result["eligible_starts"] == [0, 1, 2]
        assert result["svd_relative_cutoff"] == 1e-10
        assert result["starts"][0]["rank"] == 4
    else:
        with pytest.raises(EngineError) as caught:
            coverage(space, batch(space, rows), spec)
        assert caught.value.code == ErrorCode.INSUFFICIENT_DATA


def test_complete_basis_can_fail_after_each_interaction_passes(space):
    rows = [[x, z, x] for x, z in [(0, 0), (0, 1), (1, 0), (1, 1), (.2, .4), (.8, .6)]]
    specs = [declaration("synergy", pair, identity=str(pair))
             for pair in [("x", "z"), ("z", "t")]]
    for spec in specs:
        assert coverage(space, batch(space, rows), spec)["eligible_starts"] == [0, 1, 2]
    with pytest.raises(EngineError) as caught:
        coverage(space, batch(space, rows), *specs)
    assert caught.value.code == ErrorCode.INSUFFICIENT_DATA
    assert all(c["status"] == "passed" for c in caught.value.details["coverage"])
    assert len(caught.value.details["starts"]) == 3
    assert caught.value.details["deficient_columns"]


def test_only_full_rank_width_starts_are_eligible(space):
    # Both shapes are numerically flat at high x for the narrow start;
    # broader starts distinguish the independently varied plateau coordinates.
    rows = [[x, z, 0.] for x, z in
            [(0., 0.), (.01, .01), (.02, .02), (.98, .95), (.99, .97), (1., 1.)]]
    specs = [declaration("threshold", (factor,), identity=factor) for factor in ("x", "z")]
    result = coverage(space, batch(space, rows), *specs)
    assert result["eligible_starts"] == [1, 2]
    assert [s["rank"] for s in result["starts"]] == [2, 3, 3]
    assert len(result["starts"][0]["columns"]) == 3  # Flat branch omitted.


@pytest.mark.parametrize("other,effect", [("synergy", "higher"), ("antagonism", "higher"),
    ("synergy", "lower"), ("antagonism", "lower")])
def test_unordered_duplicate_or_opposing_pair_conflicts(space, other, effect):
    declarations = knowledge(declaration("synergy", ("x", "z"), identity="first"),
        declaration(other, ("z", "x"), identity="second", response_effect=effect))
    with pytest.raises(EngineError) as caught:
        composition(space, declarations)
    duplicate_pair_error = caught.value
    assert duplicate_pair_error.code == ErrorCode.KNOWLEDGE_CONFLICT
    assert duplicate_pair_error.details["pattern_ids"] == ["first", "second"]
    assert duplicate_pair_error.details["factors"] == ["x", "z"]
    with pytest.raises(EngineError) as runtime:
        compile_runtime(space, declarations, None)
    assert runtime.value.code == ErrorCode.KNOWLEDGE_CONFLICT
    report = inspect_capabilities(space, declarations)
    assert any(d.get("code") == "KNOWLEDGE_CONFLICT" for d in report.diagnostics)


def test_duplicate_univariate_shapes_conflict(space):
    with pytest.raises(EngineError) as caught:
        composition(space, knowledge(declaration(identity="one"), declaration("threshold", identity="two")))
    assert caught.value.code == ErrorCode.KNOWLEDGE_CONFLICT


@pytest.mark.parametrize("legacy", ["mean", "saturation", "threshold", "monotone", "random"])
def test_v2_legacy_mean_and_virtual_compositions_are_invalid(space, legacy):
    declarations = knowledge(declaration())
    if legacy == "mean":
        declarations.add_spec(make_pattern_spec(pattern="periodic", version="1.0",
            scope=KnowledgeScope(factors=("x",)), parameters={"period": 1., "phase": 0.}, confidence=1.))
    elif legacy == "saturation":
        declarations.with_saturation("x", direction="increasing", half_response=.5)
    elif legacy == "threshold":
        declarations.with_threshold("x", threshold=.5, below_behavior="increasing", above_behavior="flat")
    elif legacy == "monotone":
        declarations.with_monotone("x", "increases_objective")
    else:
        declarations.with_random_augment(2)
    with pytest.raises(EngineError) as caught:
        composition(space, declarations)
    v2_plus_legacy_error = caught.value
    assert v2_plus_legacy_error.code == ErrorCode.KNOWLEDGE_INVALID
    assert v2_plus_legacy_error.details["reason"] == "unsupported_composition"
    assert v2_plus_legacy_error.details["remediation"]
    with pytest.raises(EngineError) as runtime:
        compile_runtime(space, declarations, None)
    assert runtime.value.details["reason"] == "unsupported_composition"


def test_distinct_shapes_shared_interaction_factors_and_covariates_are_valid(space):
    specs = [declaration(), declaration("threshold", ("z",), identity="z"),
        declaration("synergy", ("x", "z"), identity="xz"),
        declaration("antagonism", ("x", "t"), identity="xt")]
    composition(space, knowledge(*specs))
    mixed = Space(list(space.params) + [Parameter("category", kind="categorical", values=("a", "b")),
        Parameter("grade", kind="ordinal", values=("low", "high"))], objectives="response")
    rows = [row + ["a", "low"] for row in batch(space).X.values.tolist()]
    assert coverage(mixed, batch(mixed, rows), *specs)["eligible_starts"] == [0, 1, 2]


def test_disabled_declarations_neither_cover_nor_conflict(space):
    declarations = knowledge(declaration(enabled=False), declaration("threshold", identity="enabled"))
    composition(space, declarations)
    result = coverage(space, None, *declarations.specs)
    assert result["pattern_ids"] == ["enabled"]
    assert coverage(space, batch(space, []), declaration(enabled=False))["coverage_status"] == "not_applicable"


def test_unknown_response_in_structured_artifact_is_invalid(space):
    from expdoe_dk.bo.structured_coverage import check_structured_composition
    original, = knowledge(declaration()).compile(space).mean_components
    payload = original.to_dict()["payload"]
    payload["objectives"] = ["unknown"]
    bad = OptimizationArtifact(original.kind, payload, "shape", "saturation", "2.0")
    with pytest.raises(EngineError) as caught:
        check_structured_composition(space, OptimizationArtifacts(mean_components=(bad,)))
    assert caught.value.code == ErrorCode.KNOWLEDGE_INVALID
    with pytest.raises(EngineError):
        validate_runtime_artifact(space, "mean_components", bad)


def test_valid_v2_is_wired_without_discovery_claiming_coverage(space):
    declarations = knowledge(declaration())
    report = inspect_capabilities(space, declarations)
    assert report.applied == ("shape",)
    assert report.blocked == ()
    structured = next(d for d in report.diagnostics if d.get("artifact_kind") == "saturation_mean_v2")
    assert structured["coverage_status"] == "not_evaluated"
    assert structured["empirical_evidence"] == "not_evaluated"
    assert compile_runtime(space, declarations, None)["mean_components"][0].kind == "saturation_mean_v2"


def test_hard_safety_route_is_independent_of_empirical_coverage(space):
    declarations = Knowledge().with_safe_region(expression="x <= .8", factors=("x",))
    declarations.add_spec(make_pattern_spec(pattern="target_range", version="1.0",
        scope=KnowledgeScope(objectives=("response",)),
        parameters={"lower": -1., "upper": 1.}, confidence=1.))
    runtime = compile_runtime(space, declarations, batch(space, []))
    assert not runtime["parameter_constraints"][0].satisfied({"x": .9})
    assert runtime["outcome_constraints"][0].satisfied({"response": 0.})
