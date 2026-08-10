import math

import pytest

from expdoe_dk.domain import (
    CategoricalCombinationConstraint,
    Constraint,
    ExpressionConstraint,
    LinearConstraint,
    OutcomeConstraint,
    constraint_from_dict,
)
from expdoe_dk.errors import EngineError, ErrorCode


def assert_config_invalid(action, match=None):
    """Assert the stable configuration error contract without hiding behavior."""
    with pytest.raises(EngineError, match=match) as caught:
        action()
    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_expression_constraint_evaluates_allowlisted_math():
    """Catches canonical evaluation reversing or dropping a comparison."""
    constraint = ExpressionConstraint("ratio", "a / b <= 2", hard=True)

    assert constraint.satisfied({"a": 4.0, "b": 2.0})
    assert not constraint.satisfied({"a": 5.0, "b": 2.0})


def test_expression_constraint_evaluates_all_allowlisted_operations():
    """Catches an advertised safe operator or math call being omitted."""
    constraint = ExpressionConstraint(
        "math",
        "(-x + +y ** 2 >= abs(z)) and "
        "(min(x, y) < max(y, z) or log(exp(one)) == one)",
        allowed_names={"x", "y", "z", "one"},
    )

    assert constraint.satisfied({"x": 1.0, "y": 2.0, "z": 2.0, "one": 1.0})
    assert not constraint.satisfied(
        {"x": 10.0, "y": 2.0, "z": 2.0, "one": 1.0}
    )


def test_expression_constraint_supports_two_argument_log():
    """Catches safe ``math.log(value, base)`` calls being rejected by arity checks."""
    constraint = ExpressionConstraint("base-log", "log(eight, two) == 3")

    assert constraint.satisfied({"eight": 8.0, "two": 2.0})


def test_expression_constraint_serializes_canonical_ast_not_source_text():
    """Catches persisted expression source surviving instead of the safe AST."""
    constraint = ExpressionConstraint("ratio", "a / b <= 2", hard=True)

    payload = constraint.to_dict()

    assert "expression" not in payload
    assert "expression" not in vars(constraint)
    assert payload["ast"] == {
        "type": "compare",
        "op": "<=",
        "left": {
            "type": "binary",
            "op": "/",
            "left": {"type": "name", "name": "a"},
            "right": {"type": "name", "name": "b"},
        },
        "right": {"type": "constant", "value": 2},
    }
    assert constraint_from_dict(payload).satisfied({"a": 4.0, "b": 2.0})


def test_expression_ast_is_not_mutated_through_serialized_payload():
    """Catches callers being able to rewrite an already-validated AST in place."""
    constraint = ExpressionConstraint("positive", "x > 0")
    payload = constraint.to_dict()

    payload["ast"]["op"] = "<"

    assert constraint.satisfied({"x": 1.0})
    assert constraint.to_dict()["ast"]["op"] == ">"


@pytest.mark.parametrize("expression", ["__import__('os')", "x.__class__", "x[0]"])
def test_expression_constraint_rejects_executable_syntax(expression):
    """Catches executable Python syntax crossing the text parser boundary."""
    with pytest.raises(EngineError, match="not allowed") as caught:
        ExpressionConstraint("unsafe", expression)

    assert caught.value.code is ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize(
    "expression",
    [
        "round(x) > 0",
        "abs(x, y) > 0",
        "log() > 0",
        "exp(x, y) > 0",
        "min() > 0",
        "max(x) > 0",
    ],
)
def test_expression_constraint_rejects_unknown_calls_and_invalid_arity(expression):
    """Catches call allow-list or arity checks being relaxed."""
    assert_config_invalid(
        lambda: ExpressionConstraint("invalid-call", expression),
        match="not allowed|argument",
    )


def test_expression_constraint_rejects_factor_outside_declared_names():
    """Catches misspelled or undeclared factors entering a validated expression."""
    assert_config_invalid(
        lambda: ExpressionConstraint("names", "x + typo <= 1", allowed_names={"x"}),
        match="Name 'typo' is not allowed",
    )


def test_expression_constraint_rejects_non_finite_text_constant():
    """Catches an overflowing numeric literal becoming persisted infinity."""
    assert_config_invalid(
        lambda: ExpressionConstraint("finite", "x < 1e309"),
        match="finite",
    )


def test_expression_constraint_normalizes_chained_comparisons():
    """Catches chained Python comparisons losing an adjacent comparison."""
    constraint = ExpressionConstraint("range", "0 <= x < 1")

    assert constraint.satisfied({"x": 0.5})
    assert not constraint.satisfied({"x": 1.0})
    assert constraint.to_dict()["ast"]["type"] == "boolean"


def test_forbidden_category_combination():
    """Catches a forbidden partial assignment being treated as allowed."""
    constraint = CategoricalCombinationConstraint(
        "unstable_pair", forbidden=({"binder": "A", "solvent": "water"},)
    )

    assert not constraint.satisfied({"binder": "A", "solvent": "water"})
    assert constraint.satisfied({"binder": "B", "solvent": "water"})
    assert constraint.satisfied({"binder": "A"})


def test_allowed_category_combination_requires_one_complete_partial_match():
    """Catches allowed patterns being combined across separate assignments."""
    constraint = CategoricalCombinationConstraint(
        "supported_pairs",
        allowed=(
            {"binder": "A", "solvent": "water"},
            {"binder": "B", "solvent": "ethanol"},
        ),
    )

    assert constraint.satisfied({"binder": "B", "solvent": "ethanol", "temp": 25})
    assert not constraint.satisfied({"binder": "A", "solvent": "ethanol"})


def test_categorical_constraint_requires_exactly_one_pattern_mode():
    """Catches ambiguous allowed-plus-forbidden categorical semantics."""
    assert_config_invalid(lambda: CategoricalCombinationConstraint("empty"))
    assert_config_invalid(
        lambda: CategoricalCombinationConstraint(
            "ambiguous", allowed=({"a": "A"},), forbidden=({"a": "B"},)
        )
    )


def test_linear_constraint_satisfies_weighted_bound_and_round_trips():
    """Catches coefficient application or comparison direction errors."""
    constraint = LinearConstraint(
        "budget", coefficients={"a": 2.0, "b": 1.0}, operator="<=", bound=5.0
    )

    assert constraint.satisfied({"a": 2.0, "b": 1.0})
    assert not constraint.satisfied({"a": 2.1, "b": 1.0})
    assert constraint_from_dict(constraint.to_dict()).satisfied({"a": 2.0, "b": 1.0})


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [("<=", 2.0, True), (">=", 2.0, True), ("==", 2.0, True), ("==", 2.1, False)],
)
def test_linear_constraint_supports_declared_comparisons(operator, value, expected):
    """Catches a supported linear comparison mapping to the wrong predicate."""
    constraint = LinearConstraint(
        "linear", coefficients={"x": 1.0}, operator=operator, bound=2.0
    )

    assert constraint.satisfied({"x": value}) is expected


def test_outcome_constraint_checks_named_objective_and_round_trips():
    """Catches an outcome bound reading the constraint label instead of objective."""
    constraint = OutcomeConstraint(
        "minimum_yield", objective="yield", operator=">=", bound=80.0
    )

    assert constraint.satisfied({"yield": 80.0})
    assert not constraint.satisfied({"yield": 79.9})
    assert constraint_from_dict(constraint.to_dict()).satisfied({"yield": 85.0})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LinearConstraint(
            "bad", coefficients={"x": math.inf}, operator="<=", bound=1.0
        ),
        lambda: LinearConstraint(
            "bad", coefficients={"x": 1.0}, operator="!=", bound=1.0
        ),
        lambda: LinearConstraint(
            "bad", coefficients={"x": 1.0}, operator="<=", bound=10**400
        ),
        lambda: OutcomeConstraint("bad", objective="y", operator="<=", bound=math.nan),
        lambda: CategoricalCombinationConstraint(
            "bad", forbidden=({"grade": math.inf},)
        ),
    ],
)
def test_constraints_reject_invalid_scalar_configuration(factory):
    """Catches non-finite or unsupported scalar configuration being serialized."""
    assert_config_invalid(factory)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LinearConstraint(
            "soft", {"x": 1.0}, "<=", 1.0, hard=False, penalty=None, weight=1.0
        ),
        lambda: LinearConstraint(
            "soft", {"x": 1.0}, "<=", 1.0, hard=False, penalty="quadratic"
        ),
        lambda: LinearConstraint(
            "soft",
            {"x": 1.0},
            "<=",
            1.0,
            hard=False,
            penalty="quadratic",
            weight=0.0,
        ),
        lambda: LinearConstraint(
            "hard",
            {"x": 1.0},
            "<=",
            1.0,
            hard=True,
            penalty="quadratic",
            weight=1.0,
        ),
    ],
)
def test_hard_and_soft_constraints_enforce_penalty_contract(factory):
    """Catches soft constraints lacking explicit finite penalty configuration."""
    assert_config_invalid(factory, match="penalty|weight")


def test_soft_constraint_serializes_penalty_and_positive_weight():
    """Catches soft-constraint penalty metadata being dropped on round-trip."""
    constraint = LinearConstraint(
        "soft",
        {"x": 1.0},
        "<=",
        1.0,
        hard=False,
        penalty="quadratic",
        weight=2.5,
    )

    payload = constraint.to_dict()

    assert payload["penalty"] == "quadratic"
    assert payload["weight"] == 2.5
    assert constraint_from_dict(payload).to_dict() == payload


def test_constraint_protocol_covers_all_produced_constraint_types():
    """Catches a produced constraint omitting the shared consumer interface."""
    constraints = (
        LinearConstraint("linear", {"x": 1.0}, "<=", 1.0),
        ExpressionConstraint("expression", "x <= 1"),
        CategoricalCombinationConstraint("category", forbidden=({"c": "bad"},)),
        OutcomeConstraint("outcome", "y", ">=", 0.0),
    )

    assert all(isinstance(constraint, Constraint) for constraint in constraints)


def test_constraint_from_dict_accepts_only_structured_expression_ast():
    """Catches persisted expression text being reparsed as executable input."""
    assert_config_invalid(
        lambda: constraint_from_dict(
            {"kind": "expression", "name": "unsafe", "expression": "x > 0", "hard": True}
        ),
        match="Unknown keys|requires keys",
    )
    assert_config_invalid(
        lambda: constraint_from_dict(
            {"kind": "expression", "name": "unsafe", "ast": "x > 0", "hard": True}
        ),
        match="AST node",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ast: ast.update({"extra": 1}),
        lambda ast: ast.update({"type": "attribute"}),
        lambda ast: ast.update({"op": "%"}),
        lambda ast: ast["left"].update({"extra": 1}),
        lambda ast: ast["left"].update({"name": "undeclared"}),
    ],
)
def test_constraint_from_dict_rejects_noncanonical_ast_shapes_and_names(mutate):
    """Catches persisted AST validation trusting malformed or undeclared nodes."""
    payload = ExpressionConstraint("safe", "x <= 2").to_dict()
    mutate(payload["ast"])

    assert_config_invalid(
        lambda: constraint_from_dict(payload, allowed_names={"x"})
    )


@pytest.mark.parametrize(
    "ast",
    [
        {"type": "constant", "value": math.nan},
        {"type": "constant", "value": math.inf},
        {"type": "constant", "value": [1]},
        {
            "type": "call",
            "function": "round",
            "args": [{"type": "constant", "value": 1}],
        },
        {
            "type": "call",
            "function": "abs",
            "args": [
                {"type": "constant", "value": 1},
                {"type": "constant", "value": 2},
            ],
        },
    ],
)
def test_constraint_from_dict_rejects_non_json_constants_functions_and_arity(ast):
    """Catches malformed persisted values or calls bypassing text validation."""
    assert_config_invalid(
        lambda: constraint_from_dict(
            {"kind": "expression", "name": "invalid", "ast": ast, "hard": True}
        )
    )


def test_constraint_from_dict_rejects_ast_deeper_than_32_nodes():
    """Catches unbounded persisted AST recursion at the documented depth limit."""
    ast = {"type": "name", "name": "x"}
    for _ in range(32):
        ast = {"type": "unary", "op": "+", "operand": ast}

    assert_config_invalid(
        lambda: constraint_from_dict(
            {"kind": "expression", "name": "deep", "ast": ast, "hard": True}
        ),
        match="depth",
    )


def test_constraint_from_dict_rejects_unknown_constraint_keys_and_kind():
    """Catches loose persisted schemas silently ignoring configuration."""
    payload = LinearConstraint("linear", {"x": 1.0}, "<=", 1.0).to_dict()
    payload["typo"] = True

    assert_config_invalid(lambda: constraint_from_dict(payload), match="Unknown keys")
    assert_config_invalid(
        lambda: constraint_from_dict({"kind": "python", "name": "unsafe", "hard": True}),
        match="Unknown constraint kind",
    )


def test_allowed_names_validation_uses_typed_configuration_errors():
    """Catches malformed allow-lists leaking built-in hashing errors."""
    assert_config_invalid(
        lambda: ExpressionConstraint("names", "x <= 1", allowed_names=[["x"]])
    )


def test_constraint_from_dict_applies_allowed_names_to_all_factor_references():
    """Catches persisted factor references escaping a caller's declared space."""
    payloads = (
        LinearConstraint("linear", {"typo": 1.0}, "<=", 1.0).to_dict(),
        CategoricalCombinationConstraint(
            "category", forbidden=({"typo": "bad"},)
        ).to_dict(),
        OutcomeConstraint("outcome", "typo", ">=", 0.0).to_dict(),
    )

    for payload in payloads:
        assert_config_invalid(
            lambda payload=payload: constraint_from_dict(payload, allowed_names={"x", "y"})
        )
