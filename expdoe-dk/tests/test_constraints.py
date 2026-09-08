import json
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


def balanced_binary_source(leaves):
    """Build a shallow source tree with ``2 * leaves - 1`` canonical nodes."""
    nodes = ["x"] * leaves
    while len(nodes) > 1:
        nodes = [
            f"({nodes[index]} + {nodes[index + 1]})"
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0]


def balanced_canonical_tree(leaves):
    """Build the canonical counterpart of :func:`balanced_binary_source`."""
    nodes = [{"type": "name", "name": "x"} for _ in range(leaves)]
    while len(nodes) > 1:
        nodes = [
            {
                "type": "binary",
                "op": "+",
                "left": nodes[index],
                "right": nodes[index + 1],
            }
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0]


def expression_payload(ast):
    return {"kind": "expression", "name": "bounded", "ast": ast, "hard": True}


def test_expression_constraint_evaluates_allowlisted_math():
    """Catches canonical evaluation reversing or dropping a comparison."""
    constraint = ExpressionConstraint("ratio", "a / b <= 2", hard=True)

    assert constraint.satisfied({"a": 4.0, "b": 2.0})
    assert not constraint.satisfied({"a": 5.0, "b": 2.0})


@pytest.mark.parametrize(
    ("expression", "values"),
    [
        ("x * x == x * x + 1", {"x": 1e308}),
        ("x > 0", {"x": math.inf}),
        ("flag + 1 == 2", {"flag": True}),
        ("x == x", {"x": [1.0]}),
    ],
)
def test_expression_runtime_rejects_unsafe_context_and_numeric_values(
    expression, values
):
    """Catches unsafe context values or non-finite arithmetic satisfying a predicate."""
    constraint = ExpressionConstraint("runtime-boundary", expression)

    assert_config_invalid(lambda: constraint.satisfied(values))


def test_expression_runtime_never_invokes_custom_comparison_methods():
    """Catches arbitrary context objects executing code through comparison dunders."""

    class CustomValue:
        comparison_called = False

        def __lt__(self, other):
            self.comparison_called = True
            raise AssertionError("custom comparison executed")

    value = CustomValue()
    constraint = ExpressionConstraint("custom", "x < 1")

    assert_config_invalid(lambda: constraint.satisfied({"x": value}))
    assert not value.comparison_called


def test_expression_equality_distinguishes_boolean_and_numeric_scalars():
    """Catches Python's ``True == 1`` coercion changing declarative equality."""
    equal = ExpressionConstraint("typed-equal", "flag == one")
    unequal = ExpressionConstraint("typed-unequal", "flag != one")

    assert not equal.satisfied({"flag": True, "one": 1})
    assert unequal.satisfied({"flag": True, "one": 1})
    assert equal.satisfied({"flag": True, "one": True})
    assert equal.satisfied({"flag": 1, "one": 1.0})


@pytest.mark.parametrize(
    "expression",
    ["1 / zero > 0", "log(negative) > 0", "exp(huge) > 0"],
)
def test_expression_runtime_domain_failures_use_typed_configuration_error(expression):
    """Catches arithmetic/domain failures leaking built-in exceptions."""
    constraint = ExpressionConstraint("domain", expression)

    assert_config_invalid(
        lambda: constraint.satisfied({"zero": 0.0, "negative": -1.0, "huge": 1e308})
    )


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


def test_constraints_copy_caller_owned_configuration_mappings():
    """Catches post-construction caller mutation changing validated constraints."""
    coefficients = {"x": 1.0}
    assignment = {"grade": "A"}
    linear = LinearConstraint("linear-copy", coefficients, "<=", 1.0)
    categorical = CategoricalCombinationConstraint(
        "category-copy", forbidden=(assignment,)
    )
    expression_payload_input = ExpressionConstraint(
        "expression-copy", "x <= 1"
    ).to_dict()
    expression = constraint_from_dict(expression_payload_input)

    coefficients["x"] = 99.0
    assignment["grade"] = "B"
    expression_payload_input["ast"]["op"] = ">"

    assert linear.to_dict()["coefficients"] == {"x": 1.0}
    assert categorical.to_dict()["forbidden"] == [{"grade": "A"}]
    assert expression.satisfied({"x": 0.0})


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


def test_expression_source_length_accepts_4096_and_rejects_4097_characters():
    """Catches construction-time source bypassing its deterministic size budget."""
    base = "x > 0"
    ExpressionConstraint("at-source-limit", base + " " * (4096 - len(base)))

    assert_config_invalid(
        lambda: ExpressionConstraint(
            "over-source-limit", base + " " * (4097 - len(base))
        ),
        match="length",
    )


def test_expression_source_accepts_256_and_rejects_257_canonical_nodes():
    """Catches a shallow source expression bypassing the total-node budget."""
    tree = balanced_binary_source(128)
    ExpressionConstraint("at-node-limit", f"+({tree})")

    assert_config_invalid(
        lambda: ExpressionConstraint("over-node-limit", f"++({tree})"),
        match="node",
    )


def test_persisted_ast_accepts_256_and_rejects_257_canonical_nodes():
    """Catches persisted shallow trees bypassing the same total-node budget."""
    tree = balanced_canonical_tree(128)
    at_limit = {"type": "unary", "op": "+", "operand": tree}
    constraint_from_dict(expression_payload(at_limit))
    over_limit = {"type": "unary", "op": "+", "operand": at_limit}

    assert_config_invalid(
        lambda: constraint_from_dict(expression_payload(over_limit)),
        match="node",
    )


@pytest.mark.parametrize(
    "expression",
    [
        "min(" + ",".join(["x"] * 65) + ")",
        " or ".join(["x > 0"] * 65),
    ],
)
def test_expression_source_rejects_more_than_64_children(expression):
    """Catches source call/boolean fan-out bypassing the child budget."""
    assert_config_invalid(
        lambda: ExpressionConstraint("too-many-children", expression),
        match="children|argument",
    )


def test_expression_source_accepts_64_call_and_boolean_children():
    """Catches source child-budget validation rejecting its inclusive boundary."""
    ExpressionConstraint("call-child-limit", "min(" + ",".join(["x"] * 64) + ")")
    ExpressionConstraint("boolean-child-limit", " or ".join(["x > 0"] * 64))


@pytest.mark.parametrize(
    "ast",
    [
        {
            "type": "call",
            "function": "min",
            "args": [{"type": "name", "name": "x"} for _ in range(65)],
        },
        {
            "type": "boolean",
            "op": "or",
            "values": [
                {
                    "type": "compare",
                    "op": ">",
                    "left": {"type": "name", "name": "x"},
                    "right": {"type": "constant", "value": 0},
                }
                for _ in range(65)
            ],
        },
    ],
)
def test_persisted_ast_rejects_more_than_64_children(ast):
    """Catches persisted call/boolean fan-out bypassing the child budget."""
    assert_config_invalid(
        lambda: constraint_from_dict(expression_payload(ast)),
        match="children|argument",
    )


def test_persisted_ast_accepts_64_call_and_boolean_children():
    """Catches persisted child-budget validation rejecting its inclusive boundary."""
    call = {
        "type": "call",
        "function": "min",
        "args": [{"type": "name", "name": "x"} for _ in range(64)],
    }
    boolean = {
        "type": "boolean",
        "op": "or",
        "values": [
            {
                "type": "compare",
                "op": ">",
                "left": {"type": "name", "name": "x"},
                "right": {"type": "constant", "value": 0},
            }
            for _ in range(64)
        ],
    }

    constraint_from_dict(expression_payload(call))
    constraint_from_dict(expression_payload(boolean))


def test_expression_integer_constants_accept_256_and_reject_257_bits():
    """Catches source and persisted integers bypassing the constant-size budget."""
    at_limit = 1 << 255
    over_limit = 1 << 256
    ExpressionConstraint("source-int-limit", str(at_limit))
    constraint_from_dict(
        expression_payload({"type": "constant", "value": at_limit})
    )

    assert_config_invalid(
        lambda: ExpressionConstraint("source-int-over", str(over_limit)),
        match="integer|bits",
    )
    assert_config_invalid(
        lambda: constraint_from_dict(
            expression_payload({"type": "constant", "value": over_limit})
        ),
        match="integer|bits",
    )


def test_expression_power_accepts_exponent_64_and_rejects_magnitude_65():
    """Catches costly source or persisted powers bypassing exponent limits."""
    ExpressionConstraint("source-power-limit", "x ** 64 > 0")
    persisted_at_limit = {
        "type": "binary",
        "op": "**",
        "left": {"type": "name", "name": "x"},
        "right": {"type": "constant", "value": -64},
    }
    constraint_from_dict(expression_payload(persisted_at_limit))

    assert_config_invalid(
        lambda: ExpressionConstraint("source-power-over", "x ** 65 > 0"),
        match="exponent",
    )
    persisted_over_limit = dict(persisted_at_limit)
    persisted_over_limit["right"] = {"type": "constant", "value": -65}
    assert_config_invalid(
        lambda: constraint_from_dict(expression_payload(persisted_over_limit)),
        match="exponent",
    )


def test_expression_runtime_rejects_variable_power_exponent_over_64():
    """Catches a context-supplied exponent bypassing the static power budget."""
    constraint = ExpressionConstraint("runtime-power", "base ** exponent > 0")

    assert constraint.satisfied({"base": 2, "exponent": 64})
    assert_config_invalid(
        lambda: constraint.satisfied({"base": 2, "exponent": 65}),
        match="exponent",
    )


def test_expression_source_accepts_depth_32_and_rejects_depth_33():
    """Catches source depth enforcing a different boundary from persisted ASTs."""
    ExpressionConstraint("source-depth-limit", "+" * 31 + "x")

    assert_config_invalid(
        lambda: ExpressionConstraint("source-depth-over", "+" * 32 + "x"),
        match="depth",
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


def test_allowed_category_distinguishes_boolean_from_number():
    """Catches allowed patterns treating JSON boolean ``True`` as numeric ``1``."""
    constraint = CategoricalCombinationConstraint(
        "numeric-level", allowed=({"level": 1},)
    )

    assert constraint.satisfied({"level": 1.0})
    assert not constraint.satisfied({"level": True})


def test_forbidden_category_distinguishes_boolean_from_number():
    """Catches forbidden boolean patterns matching a numeric category value."""
    constraint = CategoricalCombinationConstraint(
        "boolean-level", forbidden=({"level": True},)
    )

    assert not constraint.satisfied({"level": True})
    assert constraint.satisfied({"level": 1})


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
        lambda: LinearConstraint("bad-op", {"x": 1.0}, [], 1.0),
        lambda: OutcomeConstraint("bad-op", "y", [], 1.0),
    ],
)
def test_direct_constraints_reject_non_string_operators_with_typed_error(factory):
    """Catches list-valued operators leaking unhashable-type exceptions."""
    assert_config_invalid(factory)


@pytest.mark.parametrize(
    "ast",
    [
        {
            "type": "unary",
            "op": [],
            "operand": {"type": "constant", "value": 1},
        },
        {
            "type": "binary",
            "op": [],
            "left": {"type": "constant", "value": 1},
            "right": {"type": "constant", "value": 2},
        },
        {
            "type": "compare",
            "op": [],
            "left": {"type": "constant", "value": 1},
            "right": {"type": "constant", "value": 2},
        },
        {
            "type": "boolean",
            "op": [],
            "values": [
                {"type": "constant", "value": True},
                {"type": "constant", "value": False},
            ],
        },
    ],
)
def test_persisted_ast_rejects_non_string_operators_with_typed_error(ast):
    """Catches malformed canonical operators leaking built-in exceptions."""
    assert_config_invalid(lambda: constraint_from_dict(expression_payload(ast)))


@pytest.mark.parametrize("location", ["constraint", "ast"])
def test_persisted_payload_rejects_mixed_type_mapping_keys(location):
    """Catches mixed mapping keys breaking deterministic unknown-key validation."""
    payload = LinearConstraint("linear", {"x": 1.0}, "<=", 1.0).to_dict()
    if location == "constraint":
        payload[1] = "invalid"
        payload["typo"] = "invalid"
    else:
        payload = ExpressionConstraint("expression", "x <= 1").to_dict()
        payload["ast"][1] = "invalid"
        payload["ast"]["typo"] = "invalid"

    assert_config_invalid(lambda: constraint_from_dict(payload), match="keys")


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


@pytest.mark.parametrize("hard", [True, False])
def test_all_constraint_types_round_trip_through_json_for_hard_and_soft(hard):
    """Catches a produced constraint emitting non-JSON or lossy soft metadata."""
    common = {} if hard else {"hard": False, "penalty": "quadratic", "weight": 2.5}
    constraints = (
        LinearConstraint("linear-json", {"x": 1.0}, "<=", 1.0, **common),
        ExpressionConstraint("expression-json", "x <= 1", **common),
        CategoricalCombinationConstraint(
            "category-json", forbidden=({"grade": "bad"},), **common
        ),
        OutcomeConstraint("outcome-json", "yield", ">=", 0.0, **common),
    )

    for constraint in constraints:
        payload = json.loads(json.dumps(constraint.to_dict()))
        assert constraint_from_dict(payload).to_dict() == payload
        assert ("penalty" in payload, "weight" in payload) == (not hard, not hard)


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
    at_limit = {"type": "name", "name": "x"}
    for _ in range(31):
        at_limit = {"type": "unary", "op": "+", "operand": at_limit}
    constraint_from_dict(expression_payload(at_limit))
    over_limit = {"type": "unary", "op": "+", "operand": at_limit}

    assert_config_invalid(
        lambda: constraint_from_dict(expression_payload(over_limit)),
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
        ExpressionConstraint("expression", "typo <= 1").to_dict(),
        CategoricalCombinationConstraint(
            "category", forbidden=({"typo": "bad"},)
        ).to_dict(),
    )

    for payload in payloads:
        assert_config_invalid(
            lambda payload=payload: constraint_from_dict(payload, allowed_names={"x", "y"})
        )


def test_factor_allowlist_does_not_restrict_outcome_objective_names():
    """Catches factor and objective namespaces being conflated during loading."""
    payload = OutcomeConstraint("outcome", "yield", ">=", 0.0).to_dict()

    loaded = constraint_from_dict(payload, allowed_names={"temperature"})

    assert loaded.objective == "yield"


def test_outcome_objective_uses_separate_optional_allowlist():
    """Catches outcome references bypassing their dedicated objective namespace."""
    payload = OutcomeConstraint("outcome", "yield", ">=", 0.0).to_dict()

    assert constraint_from_dict(
        payload, allowed_objective_names={"yield"}
    ).objective == "yield"
    assert_config_invalid(
        lambda: constraint_from_dict(
            payload, allowed_objective_names={"cost"}
        ),
        match="yield|allowed",
    )
