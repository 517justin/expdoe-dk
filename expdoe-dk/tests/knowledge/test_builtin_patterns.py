import pytest
from jsonschema import Draft202012Validator

from expdoe_dk import Knowledge, Objective, Parameter, Space
from expdoe_dk.knowledge.patterns import builtin_pattern_definitions
from expdoe_dk.knowledge.registry import PatternRegistry
from expdoe_dk.knowledge.specs import KnowledgeScope, make_pattern_spec


EXPECTED = {
    "monotone",
    "saturation",
    "threshold",
    "quadratic_peak",
    "quadratic_valley",
    "optimum_range",
    "power_law",
    "exponential",
    "periodic",
    "arrhenius",
    "synergy",
    "antagonism",
    "conditional_effect",
    "ratio_optimum",
    "ordinal_categories",
    "category_similarity",
    "safe_region",
    "forbidden_region",
    "target_range",
    "objective_priority",
    "tradeoff",
    "gp_prior",
    "random_augment",
}


def _builtin_registry() -> PatternRegistry:
    registry = PatternRegistry()
    for definition in builtin_pattern_definitions():
        registry.register(definition)
    return registry


def test_builtin_registry_contains_complete_taxonomy():
    assert {
        definition.pattern for definition in _builtin_registry().definitions()
    } == EXPECTED


def test_builtin_registry_uses_approved_taxonomy_families():
    assert {
        definition.pattern: definition.family
        for definition in _builtin_registry().definitions()
    } == {
        "monotone": "shape",
        "saturation": "shape",
        "threshold": "shape",
        "quadratic_peak": "shape",
        "quadratic_valley": "shape",
        "optimum_range": "shape",
        "power_law": "shape",
        "exponential": "shape",
        "periodic": "shape",
        "arrhenius": "physics",
        "synergy": "interaction",
        "antagonism": "interaction",
        "conditional_effect": "interaction",
        "ratio_optimum": "interaction",
        "ordinal_categories": "categorical",
        "category_similarity": "categorical",
        "safe_region": "feasibility",
        "forbidden_region": "feasibility",
        "target_range": "multiobjective",
        "objective_priority": "multiobjective",
        "tradeoff": "multiobjective",
        "gp_prior": "prior",
        "random_augment": "experimental",
    }


NEW_PATTERN_CASES = (
    ("saturation", {"direction": "increasing", "half_response": 0.5}, ("x",), ("y",), "mean_components"),
    ("threshold", {"threshold": 0.4, "below_behavior": "flat", "above_behavior": "increasing"}, ("x",), ("y",), "mean_components"),
    ("quadratic_valley", {"center": 0.5, "width": 0.2}, ("x",), ("y",), "mean_components"),
    ("optimum_range", {"lower": 0.2, "upper": 0.8}, ("x",), ("y",), "virtual_observations"),
    ("power_law", {"exponent": 2.0, "scale": 1.0}, ("x",), ("y",), "mean_components"),
    ("exponential", {"rate": 1.0, "amplitude": 1.0}, ("x",), ("y",), "mean_components"),
    ("periodic", {"period": 0.5, "phase": 0.0}, ("x",), ("y",), "mean_components"),
    ("synergy", {}, ("x", "z"), ("y",), "kernel_components"),
    ("antagonism", {}, ("x", "z"), ("y",), "kernel_components"),
    ("conditional_effect", {"operator": ">=", "value": 0.4}, ("x", "z"), ("y",), "mean_components"),
    ("ratio_optimum", {"ratio": 1.0, "tolerance": 0.2}, ("x", "z"), ("y",), "mean_components"),
    ("ordinal_categories", {"levels": ["low", "medium", "high"]}, ("grade",), ("y",), "kernel_components"),
    ("category_similarity", {"levels": ["A", "B"], "matrix": [[1.0, 0.5], [0.5, 1.0]]}, ("material",), ("y",), "kernel_components"),
    ("safe_region", {"expression": "x >= 0.2"}, ("x",), ("y",), "parameter_constraints"),
    ("forbidden_region", {"expression": "x > 0.8"}, ("x",), ("y",), "parameter_constraints"),
    ("target_range", {"lower": 0.2, "upper": 0.8}, (), ("y",), "outcome_constraints"),
    ("objective_priority", {"weights": [1.0, 2.0]}, (), ("y", "cost"), "acquisition_preferences"),
    ("tradeoff", {"weights": [2.0, 1.0]}, (), ("y", "cost"), "acquisition_preferences"),
)

EXPECTED_KINDS = {
    "saturation": "saturation_descriptor",
    "threshold": "threshold_descriptor",
    "quadratic_valley": "quadratic_valley_descriptor",
    "optimum_range": "optimum_range_descriptor",
    "power_law": "power_law_descriptor",
    "exponential": "exponential_descriptor",
    "periodic": "periodic_descriptor",
    "synergy": "synergy_descriptor",
    "antagonism": "antagonism_descriptor",
    "conditional_effect": "conditional_effect_descriptor",
    "ratio_optimum": "ratio_optimum_descriptor",
    "ordinal_categories": "ordinal_categories_descriptor",
    "category_similarity": "category_similarity_descriptor",
    "safe_region": "hard_parameter_constraint",
    "forbidden_region": "hard_parameter_constraint",
    "target_range": "target_range_constraint",
    "objective_priority": "objective_priority_preference",
    "tradeoff": "tradeoff_preference",
}

EXPECTED_PAYLOADS = {
    "saturation": {"factor": "x", "dimension": 0, "parameters": {"direction": "increasing", "half_response": 0.5}, "confidence": 0.8},
    "threshold": {"factor": "x", "dimension": 0, "parameters": {"threshold": 0.4, "below_behavior": "flat", "above_behavior": "increasing"}, "confidence": 0.8},
    "quadratic_valley": {"factor": "x", "dimension": 0, "parameters": {"center": 0.5, "width": 0.2}, "confidence": 0.8},
    "optimum_range": {"factor": "x", "dimension": 0, "parameters": {"lower": 0.2, "upper": 0.8}, "confidence": 0.8},
    "power_law": {"factor": "x", "dimension": 0, "parameters": {"exponent": 2.0, "scale": 1.0}, "confidence": 0.8},
    "exponential": {"factor": "x", "dimension": 0, "parameters": {"rate": 1.0, "amplitude": 1.0}, "confidence": 0.8},
    "periodic": {"factor": "x", "dimension": 0, "parameters": {"period": 0.5, "phase": 0.0}, "confidence": 0.8},
    "synergy": {"factors": ["x", "z"], "dimensions": [0, 1], "parameters": {}, "confidence": 0.8},
    "antagonism": {"factors": ["x", "z"], "dimensions": [0, 1], "parameters": {}, "confidence": 0.8},
    "conditional_effect": {"factors": ["x", "z"], "dimensions": [0, 1], "parameters": {"operator": ">=", "value": 0.4}, "confidence": 0.8},
    "ratio_optimum": {"factors": ["x", "z"], "dimensions": [0, 1], "parameters": {"ratio": 1.0, "tolerance": 0.2}, "confidence": 0.8},
    "ordinal_categories": {"factor": "grade", "dimension": 2, "parameters": {"levels": ["low", "medium", "high"]}, "confidence": 0.8},
    "category_similarity": {"factor": "material", "dimension": 3, "parameters": {"levels": ["A", "B"], "matrix": [[1.0, 0.5], [0.5, 1.0]]}, "confidence": 0.8},
    "safe_region": {
        "constraint": {
            "kind": "expression", "name": "knowledge-KP-test-safe_region", "hard": True,
            "ast": {"type": "compare", "op": ">=", "left": {"type": "name", "name": "x"}, "right": {"type": "constant", "value": 0.2}},
        },
        "source_expression": "x >= 0.2", "region_semantics": "allowed",
    },
    "forbidden_region": {
        "constraint": {
            "kind": "expression", "name": "knowledge-KP-test-forbidden_region", "hard": True,
            "ast": {
                "type": "compare", "op": "==",
                "left": {"type": "compare", "op": ">", "left": {"type": "name", "name": "x"}, "right": {"type": "constant", "value": 0.8}},
                "right": {"type": "constant", "value": False},
            },
        },
        "source_expression": "x > 0.8", "region_semantics": "forbidden",
    },
    "target_range": {"objective": "y", "lower": 0.2, "upper": 0.8, "hard": False, "confidence": 0.8},
    "objective_priority": {"objectives": ["y", "cost"], "weights": [0.3333333333333333, 0.6666666666666666], "confidence": 0.8},
    "tradeoff": {"objectives": ["y", "cost"], "weights": [0.6666666666666666, 0.3333333333333333], "confidence": 0.8},
}


@pytest.fixture
def taxonomy_space() -> Space:
    return Space(
        [
            Parameter("x", bounds=(0.1, 1.0)),
            Parameter("z", bounds=(0.1, 2.0)),
            Parameter("grade", kind="ordinal", values=("low", "medium", "high")),
            Parameter("material", kind="categorical", values=("A", "B", "C")),
        ],
        objectives=(Objective("y", "maximize"), Objective("cost", "minimize")),
    )


def _spec(pattern, parameters, factors, objectives):
    return make_pattern_spec(
        pattern_id=f"KP-test-{pattern}",
        pattern=pattern,
        version="1.0",
        parameters=parameters,
        scope=KnowledgeScope(factors=factors, objectives=objectives),
        confidence=0.8,
    )


@pytest.mark.parametrize(
    "pattern,parameters,factors,objectives,category", NEW_PATTERN_CASES
)
def test_each_new_pattern_has_strict_schema_and_compiles_typed_artifact(
    taxonomy_space, pattern, parameters, factors, objectives, category
):
    registry = _builtin_registry()
    definition = registry.resolve(pattern, "1.0")
    schema_validator = Draft202012Validator(definition.schema)
    spec = _spec(pattern, parameters, factors, objectives)

    assert list(schema_validator.iter_errors(parameters)) == []
    assert list(schema_validator.iter_errors({**parameters, "unexpected": True}))

    result = registry.validate(spec, taxonomy_space)
    artifacts = registry.compile_many((spec,), taxonomy_space)
    artifact = getattr(artifacts, category)[0]

    assert result.state != "invalid"
    assert artifact.source_pattern_id == spec.pattern_id
    assert artifact.source_pattern == pattern
    assert artifact.source_version == "1.0"
    assert artifact.kind == EXPECTED_KINDS[pattern]
    assert artifact.to_dict()["payload"] == EXPECTED_PAYLOADS[pattern]
    assert registry.render(spec, taxonomy_space)
    assert definition.compatibility(spec, taxonomy_space).compatible


def test_new_helpers_create_only_versioned_specs_and_preserve_legacy_payload():
    knowledge = (
        Knowledge()
        .with_saturation("x", direction="increasing", half_response=0.5, confidence=0.7)
        .with_threshold(
            "x",
            threshold=0.4,
            below_behavior="flat",
            above_behavior="increasing",
            confidence=0.6,
        )
        .with_interaction("x", "z", kind="synergy", confidence=0.5)
        .with_safe_region(expression="x >= 0.1", factors=("x",), confidence=0.9)
        .with_tradeoff(
            "y",
            "cost",
            first_weight=2.0,
            second_weight=1.0,
            confidence=0.8,
        )
    )

    assert [spec.pattern for spec in knowledge.specs] == [
        "saturation",
        "threshold",
        "synergy",
        "safe_region",
        "tradeoff",
    ]
    assert all(spec.version == "1.0" for spec in knowledge.specs)
    assert knowledge.to_dict() == {"strict": False, "items": []}
    assert Knowledge().with_saturation(
        "x", direction="increasing", half_response=0.5, confidence=0.7
    ).specs[0].pattern_id == knowledge.specs[0].pattern_id


@pytest.mark.parametrize(
    "call,message",
    [
        (lambda k: k.with_saturation("x", direction="sideways", half_response=0.5), "direction"),
        (lambda k: k.with_threshold("x", threshold=0.5, below_behavior="bad", above_behavior="flat"), "below_behavior"),
        (lambda k: k.with_interaction("x", "x", kind="synergy"), "distinct"),
        (lambda k: k.with_interaction("x", "z", kind="neutral"), "kind"),
        (lambda k: k.with_safe_region(expression="", factors=("x",)), "expression"),
        (lambda k: k.with_safe_region(expression="__import__('os')", factors=("x",)), "not allowed"),
        (lambda k: k.with_tradeoff("y", "cost", first_weight=-1.0, second_weight=1.0), "weight"),
    ],
)
def test_new_helpers_validate_typed_arguments_before_construction(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call(Knowledge())
