import json

import pytest
import pandas as pd

from expdoe_dk import Knowledge, ObservationBatch, Parameter, Space
from expdoe_dk.domain import ExpressionConstraint
from expdoe_dk.errors import EngineError, ErrorCode
from expdoe_dk.knowledge.patterns import feasibility
from expdoe_dk.knowledge.patterns import builtin_pattern_definitions
from expdoe_dk.knowledge.registry import PatternRegistry


def _builtin_registry() -> PatternRegistry:
    registry = PatternRegistry()
    for definition in builtin_pattern_definitions():
        registry.register(definition)
    return registry


def _safety_parameters(expression: str) -> dict:
    constraint = ExpressionConstraint("source-boundary", expression)
    return {
        "constraint": {
            "kind": "expression",
            "hard": True,
            "ast": constraint.ast,
        }
    }


def test_safe_and_forbidden_artifacts_intersect(numeric_space, make_spec):
    safe = make_spec("safe_region", parameters=_safety_parameters("x >= 0.2"))
    forbidden = make_spec(
        "forbidden_region", parameters=_safety_parameters("x > 0.8")
    )

    artifacts = _builtin_registry().compile_many((safe, forbidden), numeric_space)

    assert len(artifacts.parameter_constraints) == 2


@pytest.mark.parametrize(
    "expression",
    (
        "x >= 0.2 and x <= 0.8",
        "abs(x - 0.5) <= 0.25",
    ),
    ids=("boolean", "call"),
)
def test_frozen_canonical_safety_ast_compiles_directly(make_spec, expression):
    """Catches frozen tuple children reaching the strict AST rehydration boundary."""
    space = Space([Parameter("x", bounds=(0.0, 1.0))])
    spec = make_spec("safe_region", parameters=_safety_parameters(expression))

    artifacts = _builtin_registry().compile_many((spec,), space)

    assert len(artifacts.parameter_constraints) == 1


@pytest.mark.parametrize(
    "expression",
    (
        "x >= 0.2 and x <= 0.8",
        "abs(x - 0.5) <= 0.25",
    ),
    ids=("boolean", "call"),
)
def test_envelope_roundtrip_safety_ast_compiles(make_spec, expression):
    """Catches JSON-restored AST arrays being frozen before compilation."""
    space = Space([Parameter("x", bounds=(0.0, 1.0))])
    knowledge = Knowledge().add(
        make_spec("safe_region", parameters=_safety_parameters(expression))
    )
    restored = Knowledge.from_envelope(
        json.loads(json.dumps(knowledge.to_envelope()))
    )

    artifacts = restored.compile(space)

    assert len(artifacts.parameter_constraints) == 1


def test_strict_affine_tiny_positive_margin_is_certified(make_spec):
    """Catches valid strict regions being erased by an absolute margin cutoff."""
    space = Space([Parameter("x", bounds=(0.0, 1.0))])
    spec = make_spec(
        "safe_region",
        parameters=_safety_parameters("x < 1e-10"),
    )

    result = _builtin_registry().validate(spec, space)
    artifacts = _builtin_registry().compile_many((spec,), space)

    assert result.state == "valid"
    assert len(artifacts.parameter_constraints) == 1


def test_integral_discrete_safety_preserves_values_above_float_precision(make_spec):
    """Catches scalar float shortcuts running before exact finite enumeration."""
    exact_boundary = 2**53
    space = Space(
        [
            Parameter(
                "x",
                kind="discrete",
                values=(exact_boundary, exact_boundary + 1),
            )
        ]
    )
    spec = make_spec(
        "safe_region",
        parameters=_safety_parameters(f"x > {exact_boundary}"),
    )

    result = _builtin_registry().validate(spec, space)
    artifacts = _builtin_registry().compile_many((spec,), space)

    assert result.state == "valid"
    assert len(artifacts.parameter_constraints) == 1


def test_large_integral_finite_domain_is_inconclusive_before_float_shortcuts(
    make_spec,
):
    """Catches resource-limited exact domains falling through to float proofs."""
    exact_boundary = 2**53
    upper = exact_boundary + 100_000
    space = Space(
        [Parameter("x", kind="integer", bounds=(exact_boundary, upper))]
    )
    spec = make_spec(
        "safe_region",
        parameters=_safety_parameters(f"x > {upper - 1}"),
    )

    result = _builtin_registry().validate(spec, space)
    artifacts = _builtin_registry().compile_many((spec,), space)
    diagnostic = next(
        item for item in artifacts.diagnostics if item.kind == "safety_feasibility"
    )

    assert result.state == "insufficient_data"
    assert diagnostic.payload["reason"] == "finite_domain_resource_limit"
    assert diagnostic.payload["enumeration_size"] == 100_001


def test_large_integral_mixed_safety_is_explicitly_inconclusive_not_conflicting(
    make_spec,
):
    """An unrelated continuous dimension must not force a lossy conflict proof."""
    exact_boundary = 2**53
    space = Space(
        [
            Parameter(
                "x",
                kind="discrete",
                values=(exact_boundary, exact_boundary + 1),
            ),
            Parameter("y", bounds=(0.0, 1.0)),
        ]
    )
    spec = make_spec(
        "safe_region",
        factors=("x",),
        parameters=_safety_parameters(f"x > {exact_boundary}"),
    )

    result = _builtin_registry().validate(spec, space)
    artifacts = _builtin_registry().compile_many((spec,), space)

    assert result.state == "insufficient_data"
    assert len(artifacts.parameter_constraints) == 1
    diagnostic = next(
        item for item in artifacts.diagnostics if item.kind == "safety_feasibility"
    )
    assert diagnostic.payload["reason"] == "finite_domain_numeric_precision"


def test_mixed_domain_large_integral_levels_avoid_float_certification(make_spec):
    """Catches mixed finite/continuous spaces re-entering lossy scalar proofs."""
    exact_boundary = 2**53
    space = Space(
        [
            Parameter(
                "x",
                kind="discrete",
                values=(exact_boundary, exact_boundary + 1),
            ),
            Parameter("y", bounds=(0.0, 1.0)),
        ]
    )
    spec = make_spec(
        "safe_region",
        factors=("x",),
        parameters=_safety_parameters(f"x > {exact_boundary}"),
    )

    result = _builtin_registry().validate(spec, space)
    artifacts = _builtin_registry().compile_many((spec,), space)
    diagnostic = next(
        item for item in artifacts.diagnostics if item.kind == "safety_feasibility"
    )

    assert result.state == "insufficient_data"
    assert diagnostic.payload["reason"] == "finite_domain_numeric_precision"


def test_safety_validation_does_not_invoke_the_compiler(
    monkeypatch, make_spec
):
    """Catches safety certification bypassing validator-before-compiler ordering."""
    space = Space([Parameter("x", bounds=(0.0, 1.0))])
    spec = make_spec(
        "safe_region",
        parameters=_safety_parameters("x >= 0.2"),
    )

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("safety validator invoked the compiler")

    monkeypatch.setattr(feasibility, "_compile", unexpected_compile)

    result = _builtin_registry().validate(spec, space)

    assert result.state == "valid"


def test_forbidding_the_constant_true_region_is_provably_empty(
    numeric_space, make_spec
):
    spec = make_spec("forbidden_region", parameters=_safety_parameters("True"))

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
        parameters=_safety_parameters("z >= 0.2"),
    )

    result = _builtin_registry().validate(spec, space)

    assert result.state == "invalid"


def test_single_out_of_bounds_safe_region_is_invalid(numeric_space, make_spec):
    spec = make_spec("safe_region", parameters=_safety_parameters("x > 2"))

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"


def test_constant_numeric_equality_matches_constraint_semantics(
    numeric_space, make_spec
):
    spec = make_spec("safe_region", parameters=_safety_parameters("1 == 1.0"))

    result = _builtin_registry().validate(spec, numeric_space)
    artifacts = _builtin_registry().compile_many((spec,), numeric_space)

    assert result.state == "valid"
    assert len(artifacts.parameter_constraints) == 1


def test_constant_false_ordering_is_provably_empty(numeric_space, make_spec):
    spec = make_spec("safe_region", parameters=_safety_parameters("2 < 1"))

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"
    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((spec,), numeric_space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT


def test_non_boolean_constant_safety_root_is_invalid(numeric_space, make_spec):
    spec = make_spec("safe_region", parameters=_safety_parameters("0"))

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
        parameters=_safety_parameters("x <= 0.2"),
    )
    above = make_spec(
        "safe_region",
        pattern_id="KP-above",
        parameters=_safety_parameters("x >= 0.8"),
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
        parameters=_safety_parameters("x > 0.5"),
    )
    point = make_spec(
        "safe_region",
        pattern_id="KP-z-point",
        parameters=_safety_parameters("x == 0.5"),
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
        "safe_region", pattern_id="KP-lower", parameters=_safety_parameters(lower_expression)
    )
    upper = make_spec(
        "safe_region", pattern_id="KP-upper", parameters=_safety_parameters(upper_expression)
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
        "safe_region", pattern_id="KP-lower", parameters=_safety_parameters("x > 0.2")
    )
    upper = make_spec(
        "safe_region", pattern_id="KP-upper", parameters=_safety_parameters("x < 1.8")
    )

    artifacts = _builtin_registry().compile_many((lower, upper), space)

    assert len(artifacts.parameter_constraints) == 2


def test_self_inequality_is_a_typed_empty_safety_conflict(make_spec):
    """Catches tautologically false comparisons bypassing interval analysis."""
    space = Space([Parameter("x", bounds=(0.0, 1.0))])
    spec = make_spec(
        "safe_region",
        pattern_id="KP-self-inequality",
        parameters=_safety_parameters("x != x"),
    )

    result = _builtin_registry().validate(spec, space)

    assert result.state == "invalid"
    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((spec,), space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT
    assert caught.value.details["reason"] == "tautological_false"


def test_continuous_multivariate_affine_contradiction_is_certified(make_spec):
    """Catches mutually empty affine half-spaces being silently accepted."""
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0)), Parameter("y", bounds=(0.0, 1.0))]
    )
    low = make_spec(
        "safe_region",
        pattern_id="KP-affine-low",
        factors=("x", "y"),
        parameters=_safety_parameters("x + y <= 0.25"),
    )
    high = make_spec(
        "safe_region",
        pattern_id="KP-affine-high",
        factors=("x", "y"),
        parameters=_safety_parameters("x + y >= 1.75"),
    )

    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((high, low), space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT
    assert caught.value.details == {
        "source_pattern_ids": ["KP-affine-high", "KP-affine-low"],
        "reason": "continuous_affine_infeasible",
    }


def test_finite_domain_safety_is_exactly_enumerated(make_spec):
    """Catches continuous relaxations incorrectly certifying a finite domain."""
    space = Space(
        [
            Parameter("x", kind="integer", bounds=(0, 1)),
            Parameter("y", kind="integer", bounds=(0, 1)),
        ]
    )
    impossible = make_spec(
        "safe_region",
        pattern_id="KP-finite-impossible",
        factors=("x", "y"),
        parameters=_safety_parameters("x + y == 0.5"),
    )
    possible = make_spec(
        "safe_region",
        pattern_id="KP-finite-possible",
        factors=("x", "y"),
        parameters=_safety_parameters("x + y == 1"),
    )

    with pytest.raises(EngineError) as caught:
        _builtin_registry().compile_many((impossible,), space)
    assert caught.value.code is ErrorCode.KNOWLEDGE_CONFLICT
    assert caught.value.details["reason"] == "exact_finite_enumeration"

    artifacts = _builtin_registry().compile_many((possible,), space)
    assert len(artifacts.parameter_constraints) == 1


def test_tautological_safety_is_certified_without_inconclusive_diagnostic(make_spec):
    space = Space([Parameter("x", bounds=(0.0, 1.0))])
    spec = make_spec(
        "safe_region",
        pattern_id="KP-tautology",
        parameters=_safety_parameters("x == x"),
    )

    result = _builtin_registry().validate(spec, space)
    artifacts = _builtin_registry().compile_many((spec,), space)

    assert result.state == "valid"
    assert not [item for item in artifacts.diagnostics if item.kind == "safety_feasibility"]


def test_unsupported_continuous_safety_emits_provenance_diagnostic(make_spec):
    """Unsupported forms remain explicit instead of being labelled feasible."""
    space = Space([Parameter("x", bounds=(0.0, 1.0))])
    spec = make_spec(
        "safe_region",
        pattern_id="KP-nonlinear",
        parameters=_safety_parameters("x * x <= 0.25"),
    )

    result = _builtin_registry().validate(spec, space)
    artifacts = _builtin_registry().compile_many((spec,), space)
    diagnostic = next(
        item for item in artifacts.diagnostics if item.kind == "safety_feasibility"
    )

    assert result.state == "insufficient_data"
    assert diagnostic.source_pattern_id == spec.pattern_id
    assert diagnostic.to_dict()["payload"] == {
        "status": "inconclusive",
        "reason": "unsupported_continuous_ast",
        "source_pattern_ids": [spec.pattern_id],
    }


def test_finite_enumeration_resource_limit_is_explicit(make_spec):
    space = Space([Parameter("x", kind="integer", bounds=(0, 100_000))])
    spec = make_spec(
        "safe_region",
        pattern_id="KP-finite-limit",
        parameters=_safety_parameters("x * x >= 0"),
    )

    result = _builtin_registry().validate(spec, space)
    artifacts = _builtin_registry().compile_many((spec,), space)
    diagnostic = next(
        item for item in artifacts.diagnostics if item.kind == "safety_feasibility"
    )

    assert result.state == "insufficient_data"
    assert diagnostic.payload["reason"] == "finite_domain_resource_limit"
    assert diagnostic.payload["enumeration_size"] == 100_001
    assert diagnostic.payload["enumeration_limit"] == 100_000


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


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_conditional_effect_accepts_conditioning_domain_boundaries(
    make_spec, threshold
):
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0)), Parameter("z", bounds=(0.0, 1.0))]
    )
    spec = make_spec(
        "conditional_effect",
        factors=("x", "z"),
        parameters={"operator": ">=", "value": threshold},
    )
    definition = _builtin_registry().resolve("conditional_effect", "1.0")

    result = _builtin_registry().validate(spec, space)
    compatibility = definition.compatibility(spec, space)

    assert result.state == "insufficient_data"
    assert compatibility.compatible is True


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_conditional_effect_rejects_threshold_outside_conditioning_domain(
    make_spec, threshold
):
    space = Space(
        [Parameter("x", bounds=(0.0, 1.0)), Parameter("z", bounds=(0.0, 1.0))]
    )
    spec = make_spec(
        "conditional_effect",
        factors=("x", "z"),
        parameters={"operator": ">=", "value": threshold},
    )
    definition = _builtin_registry().resolve("conditional_effect", "1.0")

    result = _builtin_registry().validate(spec, space)
    compatibility = definition.compatibility(spec, space)

    assert result.state == "invalid"
    assert result.errors == (
        "conditional_effect value must lie within the conditioning factor domain",
    )
    assert compatibility.compatible is False


def test_conditional_effect_rejects_categorical_conditioning_factor(make_spec):
    space = Space(
        [
            Parameter("x", bounds=(0.0, 1.0)),
            Parameter("material", kind="categorical", values=("A", "B")),
        ]
    )
    spec = make_spec(
        "conditional_effect",
        factors=("x", "material"),
        parameters={"operator": "==", "value": 0.5},
    )

    result = _builtin_registry().validate(spec, space)

    assert result.state == "invalid"
    assert result.errors == (
        "Interaction factors must be numeric; got ['material']",
    )


def test_quadratic_valley_width_must_fit_factor_range(numeric_space, make_spec):
    spec = make_spec(
        "quadratic_valley",
        factors=("x",),
        parameters={"center": 0.5, "width": 2.0},
    )

    result = _builtin_registry().validate(spec, numeric_space)

    assert result.state == "invalid"


@pytest.mark.parametrize(
    "pattern,parameters,space",
    [
        (
            "saturation",
            {"direction": "increasing", "half_response": 2.0},
            Space([Parameter("x", bounds=(0.0, 1.0))]),
        ),
        (
            "threshold",
            {
                "threshold": -0.1,
                "below_behavior": "flat",
                "above_behavior": "increasing",
            },
            Space([Parameter("x", bounds=(0.0, 1.0))]),
        ),
        (
            "quadratic_valley",
            {"center": 0.5, "width": 2.0},
            Space([Parameter("x", bounds=(0.0, 1.0))]),
        ),
        (
            "optimum_range",
            {"lower": 0.8, "upper": 0.2},
            Space([Parameter("x", bounds=(0.0, 1.0))]),
        ),
        (
            "power_law",
            {"exponent": 2.0, "scale": 1.0},
            Space([Parameter("x", bounds=(-1.0, 1.0))]),
        ),
        (
            "periodic",
            {"period": 0.0, "phase": 0.0},
            Space([Parameter("x", bounds=(0.0, 1.0))]),
        ),
    ],
)
def test_shape_compatibility_rejects_every_static_validation_defect(
    make_spec, pattern, parameters, space
):
    spec = make_spec(pattern, parameters=parameters)
    registry = _builtin_registry()
    definition = registry.resolve(pattern, "1.0")

    compatibility = definition.compatibility(spec, space)
    try:
        validation = registry.validate(spec, space)
    except EngineError as error:
        assert error.code is ErrorCode.KNOWLEDGE_INVALID
    else:
        assert validation.state == "invalid"

    assert compatibility.compatible is False
    assert compatibility.reasons


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


@pytest.mark.parametrize(
    "observations",
    [
        ObservationBatch(
            X=pd.DataFrame({"x": pd.Series(dtype=float)}),
            Y=pd.DataFrame({"y": pd.Series(dtype=float)}),
        ),
        ObservationBatch(
            X=pd.DataFrame({"x": [0.0, 0.4, 0.8, 1.0]}),
            Y=pd.DataFrame({"y": [0.0, 0.4, 0.8, 1.0]}),
            status=pd.Series(["failed"] * 4),
        ),
        ObservationBatch(
            X=pd.DataFrame({"dose": [0.0, 0.4, 0.8, 1.0]}),
            Y=pd.DataFrame({"outcome": [0.0, 0.4, 0.8, 1.0]}),
        ),
        ObservationBatch(
            X=pd.DataFrame({"x": [0.0, 1.0]}),
            Y=pd.DataFrame({"y": [0.0, 1.0]}),
        ),
        ObservationBatch(
            X=pd.DataFrame({"x": [0.0, 0.4, float("inf"), 1.0]}),
            Y=pd.DataFrame({"y": [0.0, 0.4, 0.8, 1.0]}),
        ),
    ],
    ids=(
        "empty",
        "all-failed",
        "irrelevant-columns",
        "inadequate-coverage",
        "non-finite-factor",
    ),
)
def test_nonperiodic_shape_requires_usable_empirical_evidence(
    numeric_space, make_spec, observations
):
    spec = make_spec(
        "saturation",
        parameters={"direction": "increasing", "half_response": 0.5},
    )

    result = _builtin_registry().validate(spec, numeric_space, observations)

    assert result.state == "insufficient_data"
    assert "empirical" in result.summary


@pytest.mark.parametrize(
    "observations",
    [
        ObservationBatch(
            X=pd.DataFrame({"temperature": pd.Series(dtype=float)}),
            Y=pd.DataFrame({"y": pd.Series(dtype=float)}),
        ),
        ObservationBatch(
            X=pd.DataFrame({"temperature": [300.0, 320.0, 340.0, 360.0]}),
            Y=pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}),
            status=pd.Series(["failed"] * 4),
        ),
        ObservationBatch(
            X=pd.DataFrame({"pressure": [1.0, 2.0, 3.0, 4.0]}),
            Y=pd.DataFrame({"yield": [1.0, 2.0, 3.0, 4.0]}),
        ),
        ObservationBatch(
            X=pd.DataFrame({"temperature": [300.0, 360.0]}),
            Y=pd.DataFrame({"y": [1.0, 4.0]}),
        ),
        ObservationBatch(
            X=pd.DataFrame(
                {"temperature": [300.0, 320.0, float("inf"), 360.0]}
            ),
            Y=pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}),
        ),
    ],
    ids=(
        "empty",
        "all-failed",
        "irrelevant-columns",
        "inadequate-coverage",
        "non-finite-factor",
    ),
)
def test_arrhenius_requires_usable_empirical_evidence(make_spec, observations):
    space = Space(
        [Parameter("temperature", bounds=(280.0, 400.0), unit="K")],
        objectives="y",
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

    result = _builtin_registry().validate(spec, space, observations)

    assert result.state == "insufficient_data"
    assert "empirical" in result.summary


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


def test_ordinal_levels_accept_all_json_scalar_types_with_typed_identity(make_spec):
    space = Space(
        [Parameter("grade", kind="ordinal", values=("low", 1, True, None))]
    )
    spec = make_spec(
        "ordinal_categories",
        factors=("grade",),
        parameters={"levels": ["low", 1, True, None]},
    )

    result = _builtin_registry().validate(spec, space)
    artifact = _builtin_registry().compile_many((spec,), space).kernel_components[0]

    assert result.state == "valid"
    assert artifact.to_dict()["payload"]["parameters"] == {
        "levels": ["low", 1, True, None]
    }


def test_ordinal_boolean_and_number_levels_are_distinct(make_spec):
    space = Space([Parameter("grade", kind="ordinal", values=(True, 1))])
    spec = make_spec(
        "ordinal_categories",
        factors=("grade",),
        parameters={"levels": [True, 1]},
    )

    result = _builtin_registry().validate(spec, space)

    assert result.state == "valid"


def test_ordinal_duplicate_level_is_schema_invalid(make_spec):
    space = Space([Parameter("grade", kind="ordinal", values=(True, 1))])
    spec = make_spec(
        "ordinal_categories",
        factors=("grade",),
        parameters={"levels": [True, True]},
    )

    with pytest.raises(EngineError) as caught:
        _builtin_registry().validate(spec, space)

    assert caught.value.code is ErrorCode.KNOWLEDGE_INVALID


def test_ordinal_membership_does_not_alias_false_to_zero(make_spec):
    space = Space([Parameter("grade", kind="ordinal", values=(True, 1))])
    spec = make_spec(
        "ordinal_categories",
        factors=("grade",),
        parameters={"levels": [False, 1]},
    )

    result = _builtin_registry().validate(spec, space)

    assert result.state == "invalid"
    assert result.errors == ("All referenced levels must exist in the factor",)


@pytest.mark.parametrize(
    "pattern,parameters,factors,category",
    [
        (
            "saturation",
            {"direction": "increasing", "half_response": 0.5},
            ("x",),
            "mean_components",
        ),
        ("synergy", {}, ("x", "z"), "kernel_components"),
        (
            "ordinal_categories",
            {"levels": ["low", "high"]},
            ("grade",),
            "kernel_components",
        ),
    ],
)
@pytest.mark.parametrize("objectives", [(), ("y", "cost")], ids=("default", "multiple"))
def test_descriptor_patterns_validate_and_preserve_objective_scope(
    make_spec, pattern, parameters, factors, category, objectives
):
    space = Space(
        [
            Parameter("x", bounds=(0.0, 1.0)),
            Parameter("z", bounds=(0.0, 1.0)),
            Parameter("grade", kind="ordinal", values=("low", "high")),
        ],
        objectives=("y", "cost"),
    )
    spec = make_spec(
        pattern,
        parameters=parameters,
        factors=factors,
        objectives=objectives,
    )

    result = _builtin_registry().validate(spec, space)
    artifact = getattr(_builtin_registry().compile_many((spec,), space), category)[0]

    assert result.state != "invalid"
    assert artifact.to_dict()["payload"]["objectives"] == list(objectives)


@pytest.mark.parametrize(
    "pattern,parameters,factors",
    [
        (
            "saturation",
            {"direction": "increasing", "half_response": 0.5},
            ("x",),
        ),
        ("synergy", {}, ("x", "z")),
        (
            "ordinal_categories",
            {"levels": ["low", "high"]},
            ("grade",),
        ),
    ],
)
def test_descriptor_patterns_reject_unknown_objective_scope(
    make_spec, pattern, parameters, factors
):
    space = Space(
        [
            Parameter("x", bounds=(0.0, 1.0)),
            Parameter("z", bounds=(0.0, 1.0)),
            Parameter("grade", kind="ordinal", values=("low", "high")),
        ],
        objectives=("y", "cost"),
    )
    spec = make_spec(
        pattern,
        parameters=parameters,
        factors=factors,
        objectives=("missing",),
    )
    definition = _builtin_registry().resolve(pattern, "1.0")

    result = _builtin_registry().validate(spec, space)
    compatibility = definition.compatibility(spec, space)

    assert result.state == "invalid"
    assert result.errors == ("Unknown objectives ['missing']",)
    assert compatibility.compatible is False
