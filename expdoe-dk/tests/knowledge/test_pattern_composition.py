import pytest
import pandas as pd

from expdoe_dk import ObservationBatch, Parameter, Space
from expdoe_dk.errors import EngineError, ErrorCode
from expdoe_dk.knowledge.patterns import builtin_pattern_definitions
from expdoe_dk.knowledge.registry import PatternRegistry


def _builtin_registry() -> PatternRegistry:
    registry = PatternRegistry()
    for definition in builtin_pattern_definitions():
        registry.register(definition)
    return registry


def test_safe_and_forbidden_artifacts_intersect(numeric_space, make_spec):
    safe = make_spec("safe_region", parameters={"expression": "x >= 0.2"})
    forbidden = make_spec(
        "forbidden_region", parameters={"expression": "x > 0.8"}
    )

    artifacts = _builtin_registry().compile_many((safe, forbidden), numeric_space)

    assert len(artifacts.parameter_constraints) == 2


def test_forbidding_the_constant_true_region_is_provably_empty(
    numeric_space, make_spec
):
    spec = make_spec("forbidden_region", parameters={"expression": "True"})

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"
    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((spec,), numeric_space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT


def test_safety_scope_rejects_expression_references_outside_declared_factors(
    make_spec,
):
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0)), Parameter("z", bounds=(0.0, 1.0))]
    )
    spec = make_spec(
        "safe_region",
        factors=("x",),
        parameters={"expression": "z >= 0.2"},
    )

    result = _builtin_registry().validate(spec, space)

    assert result.state == "invalid"


def test_single_out_of_bounds_safe_region_is_invalid(numeric_space, make_spec):
    spec = make_spec("safe_region", parameters={"expression": "x > 2"})

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"


def test_constant_numeric_equality_matches_constraint_semantics(
    numeric_space, make_spec
):
    spec = make_spec("safe_region", parameters={"expression": "1 == 1.0"})

    result = _builtin_registry().validate(spec, numeric_space)
    artifacts = _builtin_registry().compile_many((spec,), numeric_space)

    assert result.state == "valid"
    assert len(artifacts.parameter_constraints) == 1


def test_constant_false_ordering_is_provably_empty(numeric_space, make_spec):
    spec = make_spec("safe_region", parameters={"expression": "2 < 1"})

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"
    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((spec,), numeric_space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT


def test_non_boolean_constant_safety_root_is_invalid(numeric_space, make_spec):
    spec = make_spec("safe_region", parameters={"expression": "0"})

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"
    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((spec,), numeric_space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


def test_provably_empty_hard_safety_intersection_is_a_typed_conflict(
    numeric_space, make_spec
):
    below = make_spec(
        "safe_region",
        pattern_id="KP-below",
        parameters={"expression": "x <= 0.2"},
    )
    above = make_spec(
        "safe_region",
        pattern_id="KP-above",
        parameters={"expression": "x >= 0.8"},
    )

    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((above, below), numeric_space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT
    assert caught.value.details == {
        "factor": "x",
        "source_pattern_ids": ["KP-above", "KP-below"],
        "lower": 0.8,
        "upper": 0.2,
    }


def test_strict_boundary_and_equal_point_are_provably_conflicting(
    numeric_space, make_spec
):
    strict = make_spec(
        "safe_region",
        pattern_id="KP-a-strict",
        parameters={"expression": "x > 0.5"},
    )
    point = make_spec(
        "safe_region",
        pattern_id="KP-z-point",
        parameters={"expression": "x == 0.5"},
    )

    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((strict, point), numeric_space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT


@pytest.mark.parametrize(
    "parameter,lower_expression,upper_expression",
    [
        (Parameter("x", kind="integer", bounds=(0, 1)), "x > 0", "x < 1"),
        (
            Parameter("x", kind="discrete", values=(0.1, 0.9)),
            "x > 0.2",
            "x < 0.8",
        ),
    ],
)
def test_finite_numeric_domains_detect_empty_open_intersections(
    make_spec, parameter, lower_expression, upper_expression
):
    space = Space([parameter])
    lower = make_spec(
        "safe_region", pattern_id="KP-lower", parameters={"expression": lower_expression}
    )
    upper = make_spec(
        "safe_region", pattern_id="KP-upper", parameters={"expression": upper_expression}
    )

    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((lower, upper), space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT


@pytest.mark.parametrize(
    "parameter",
    [
        Parameter("x", kind="integer", bounds=(0, 2)),
        Parameter("x", kind="discrete", values=(0.1, 0.5, 0.9)),
    ],
)
def test_finite_numeric_domains_preserve_nonempty_open_intersections(
    make_spec, parameter
):
    space = Space([parameter])
    lower = make_spec(
        "safe_region", pattern_id="KP-lower", parameters={"expression": "x > 0.2"}
    )
    upper = make_spec(
        "safe_region", pattern_id="KP-upper", parameters={"expression": "x < 1.8"}
    )

    artifacts = _builtin_registry().compile_many((lower, upper), space)

    assert len(artifacts.parameter_constraints) == 2


def test_arrhenius_rejects_non_temperature_scope(numeric_space, make_spec):
    spec = make_spec(
        "arrhenius",
        factors=("x",),
        parameters={
            "frozen": True,
            "activation_energy": 1.0,
            "amplitude_init": -1.0,
        },
    )

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"


def test_arrhenius_validation_uses_only_its_temperature_factor(make_spec):
    mixed_space = Space(
        [
            Parameter("temperature", bounds=(20.0, 80.0), unit="C"),
            Parameter("material", kind="categorical", values=("A", "B")),
        ]
    )
    spec = make_spec(
        "arrhenius",
        factors=("temperature",),
        parameters={
            "frozen": True,
            "activation_energy": 1.0,
            "amplitude_init": -1.0,
        },
    )

    result = _builtin_registry().validate(spec, mixed_space)

    assert result.state == "insufficient_data"


def test_interaction_requires_two_distinct_numeric_factors(numeric_space, make_spec):
    spec = make_spec("synergy", factors=("x",), parameters={})

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"


def test_shape_validation_supports_numeric_discrete_levels(make_spec):
    discrete_space = Space(
        [Parameter("dose", kind="discrete", values=(0.1, 0.5, 1.0))]
    )
    spec = make_spec(
        "saturation",
        factors=("dose",),
        parameters={"direction": "increasing", "half_response": 0.5},
    )

    result = _builtin_registry().validate(spec, discrete_space)

    assert result.state == "insufficient_data"


def test_legacy_monotone_requires_a_numeric_factor(make_spec):
    categorical_space = Space(
        [Parameter("material", kind="categorical", values=("A", "B"))]
    )
    spec = make_spec(
        "monotone",
        factors=("material",),
        parameters={
            "direction": "increasing",
            "n_pairs_per_dim": 5,
            "epsilon": "auto",
            "delta_norm": 0.5,
        },
    )

    result = _builtin_registry().validate(spec, categorical_space)

    assert result.state == "invalid"


def test_legacy_quadratic_peak_center_must_be_in_factor_bounds(
    numeric_space, make_spec
):
    spec = make_spec(
        "quadratic_peak",
        factors=("x",),
        parameters={"center": 2.0, "direction": "peak", "frozen": True},
    )

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"


def test_ratio_optimum_rejects_a_categorical_denominator_without_crashing(
    make_spec,
):
    mixed_space = Space(
        [
            Parameter("amount", bounds=(0.1, 1.0)),
            Parameter("material", kind="categorical", values=("A", "B")),
        ]
    )
    spec = make_spec(
        "ratio_optimum",
        factors=("amount", "material"),
        parameters={"ratio": 1.0, "tolerance": 0.2},
    )

    result = _builtin_registry().validate(spec, mixed_space)

    assert result.state == "invalid"


def test_ratio_optimum_rejects_an_unreachable_ratio(make_spec):
    space = Space(
        [Parameter("numerator", bounds=(0.1, 1.0)), Parameter("denominator", bounds=(0.1, 1.0))]
    )
    spec = make_spec(
        "ratio_optimum",
        factors=("numerator", "denominator"),
        parameters={"ratio": 20.0, "tolerance": 0.1},
    )

    result = _builtin_registry().validate(spec, space)

    assert result.state == "invalid"


def test_quadratic_valley_width_must_fit_factor_range(numeric_space, make_spec):
    spec = make_spec(
        "quadratic_valley",
        factors=("x",),
        parameters={"center": 0.5, "width": 2.0},
    )

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"


def test_periodic_empirical_validation_requires_one_covered_period(make_spec):
    space = Space([Parameter("x", bounds=(0.0, 1.0))])
    spec = make_spec(
        "periodic",
        factors=("x",),
        parameters={"period": 0.5, "phase": 0.0},
    )
    observations = ObservationBatch(
        X=pd.DataFrame({"x": [0.1, 0.2]}),
        Y=pd.DataFrame({"y": [0.0, 1.0]}),
    )

    result = _builtin_registry().validate(spec, space, observations)

    assert result.state == "insufficient_data"


def test_interaction_empirical_validation_requires_joint_coverage(make_spec):
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0)), Parameter("z", bounds=(0.0, 1.0))]
    )
    spec = make_spec("synergy", factors=("x", "z"), parameters={})
    observations = ObservationBatch(
        X=pd.DataFrame({"x": [0.0, 1.0], "z": [0.5, 0.5]}),
        Y=pd.DataFrame({"y": [0.0, 1.0]}),
    )

    result = _builtin_registry().validate(spec, space, observations)

    assert result.state == "insufficient_data"
