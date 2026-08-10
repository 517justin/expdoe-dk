"""Serializable experiment constraints with a safe expression interpreter."""
from __future__ import annotations

import ast as python_ast
import math
import operator
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import ClassVar, Protocol, runtime_checkable

from expdoe_dk.errors import EngineError, ErrorCode

JSONScalar = str | int | float | bool | None
_MAX_AST_DEPTH = 32
_COMPARISON_OPERATORS = {"<=", ">=", "=="}
_AST_COMPARISONS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
_AST_UNARY = {"+": operator.pos, "-": operator.neg}
_AST_BINARY = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
}
_ALLOWED_CALLS = {
    "abs": abs,
    "min": min,
    "max": max,
    "log": math.log,
    "exp": math.exp,
}
_CALL_ARITY = {
    "abs": (1, 1),
    "min": (2, None),
    "max": (2, None),
    "log": (1, 2),
    "exp": (1, 1),
}
_PYTHON_UNARY = {
    python_ast.UAdd: "+",
    python_ast.USub: "-",
}
_PYTHON_BINARY = {
    python_ast.Add: "+",
    python_ast.Sub: "-",
    python_ast.Mult: "*",
    python_ast.Div: "/",
    python_ast.Pow: "**",
}
_PYTHON_COMPARISONS = {
    python_ast.Lt: "<",
    python_ast.LtE: "<=",
    python_ast.Gt: ">",
    python_ast.GtE: ">=",
    python_ast.Eq: "==",
    python_ast.NotEq: "!=",
}


def _invalid(message: str, *, details: dict | None = None) -> EngineError:
    return EngineError(ErrorCode.CONFIG_INVALID, message, details=details)


def _is_json_scalar(value: object) -> bool:
    if not isinstance(value, (str, int, float, bool, type(None))):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _validate_json_scalar(value: object, context: str) -> JSONScalar:
    if not _is_json_scalar(value):
        raise _invalid(f"{context} must be a finite JSON scalar")
    return value  # type: ignore[return-value]


def _validate_name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(f"{context} must be a non-empty string")
    return value


def _validate_finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _invalid(f"{context} must be a finite number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise _invalid(f"{context} must be a finite number") from error
    if not math.isfinite(numeric):
        raise _invalid(f"{context} must be a finite number")
    return numeric


def _normalize_allowed_names(allowed_names: Collection[str] | None) -> frozenset[str] | None:
    if allowed_names is None:
        return None
    if isinstance(allowed_names, (str, bytes, bytearray)) or not isinstance(
        allowed_names, Collection
    ):
        raise _invalid("allowed_names must be a collection of factor names")
    try:
        names = frozenset(allowed_names)
    except TypeError as error:
        raise _invalid("allowed_names must contain non-empty strings") from error
    if any(not isinstance(name, str) or not name for name in names):
        raise _invalid("allowed_names must contain non-empty strings")
    return names


def _validate_factor_name(name: object, allowed_names: frozenset[str] | None) -> str:
    factor_name = _validate_name(name, "Factor name")
    if allowed_names is not None and factor_name not in allowed_names:
        raise _invalid(f"Name {factor_name!r} is not allowed")
    return factor_name


def _validate_common(
    name: object,
    hard: object,
    penalty: object,
    weight: object,
) -> tuple[str, bool, str | None, float | None]:
    constraint_name = _validate_name(name, "Constraint name")
    if not isinstance(hard, bool):
        raise _invalid("Constraint hard must be a boolean")
    if hard:
        if penalty is not None or weight is not None:
            raise _invalid("Hard constraints reject penalty and weight fields")
        return constraint_name, hard, None, None
    if not isinstance(penalty, str) or not penalty:
        raise _invalid("Soft constraints require a non-empty penalty form")
    normalized_weight = _validate_finite_number(weight, "Soft constraint weight")
    if normalized_weight <= 0:
        raise _invalid("Soft constraint weight must be finite and positive")
    return constraint_name, hard, penalty, normalized_weight


def _base_payload(
    kind: str,
    name: str,
    hard: bool,
    penalty: str | None,
    weight: float | None,
) -> dict[str, object]:
    payload: dict[str, object] = {"kind": kind, "name": name, "hard": hard}
    if not hard:
        payload["penalty"] = penalty
        payload["weight"] = weight
    return payload


def _compare(operator_name: str, left: object, right: object, tolerance: float = 1e-9) -> bool:
    if operator_name == "==" and isinstance(left, Real) and isinstance(right, Real):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return bool(_AST_COMPARISONS[operator_name](left, right))


def _row_number(row: Mapping[str, object], name: str, context: str) -> float:
    if name not in row:
        raise _invalid(f"{context} requires value {name!r}")
    return _validate_finite_number(row[name], f"{context} value {name!r}")


@runtime_checkable
class Constraint(Protocol):
    """Interface consumed by space validation and candidate post-processing."""

    name: str
    hard: bool
    penalty: str | None
    weight: float | None

    def satisfied(self, values: Mapping[str, object]) -> bool:
        """Return whether one physical-unit row satisfies the constraint."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible declarative payload."""


@dataclass(frozen=True)
class LinearConstraint:
    """A weighted factor sum compared with one finite bound."""

    name: str
    coefficients: Mapping[str, float]
    operator: str
    bound: float
    hard: bool = True
    penalty: str | None = None
    weight: float | None = None

    kind: ClassVar[str] = "linear"

    def __post_init__(self) -> None:
        name, hard, penalty, weight = _validate_common(
            self.name, self.hard, self.penalty, self.weight
        )
        if not isinstance(self.coefficients, Mapping) or not self.coefficients:
            raise _invalid("LinearConstraint coefficients must be a non-empty mapping")
        coefficients: dict[str, float] = {}
        for factor, coefficient in self.coefficients.items():
            factor_name = _validate_name(factor, "LinearConstraint factor name")
            coefficients[factor_name] = _validate_finite_number(
                coefficient, f"LinearConstraint coefficient {factor_name!r}"
            )
        if self.operator not in _COMPARISON_OPERATORS:
            raise _invalid(
                "LinearConstraint operator must be one of '<=', '>=', or '=='"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "coefficients",
            MappingProxyType(dict(sorted(coefficients.items()))),
        )
        object.__setattr__(
            self,
            "bound",
            _validate_finite_number(self.bound, "LinearConstraint bound"),
        )
        object.__setattr__(self, "hard", hard)
        object.__setattr__(self, "penalty", penalty)
        object.__setattr__(self, "weight", weight)

    def evaluate(self, values: Mapping[str, object]) -> float:
        return sum(
            coefficient * _row_number(values, factor, f"LinearConstraint {self.name!r}")
            for factor, coefficient in self.coefficients.items()
        )

    def satisfied(self, values: Mapping[str, object]) -> bool:
        return _compare(self.operator, self.evaluate(values), self.bound)

    def to_dict(self) -> dict[str, object]:
        payload = _base_payload(self.kind, self.name, self.hard, self.penalty, self.weight)
        payload.update(
            {
                "coefficients": dict(self.coefficients),
                "operator": self.operator,
                "bound": self.bound,
            }
        )
        return payload


@dataclass(frozen=True)
class _CanonicalNode:
    node_type: str
    fields: tuple[tuple[str, object], ...] = field(default_factory=tuple)

    def get(self, key: str) -> object:
        for field_name, value in self.fields:
            if field_name == key:
                return value
        raise KeyError(key)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"type": self.node_type}
        for key, value in self.fields:
            if isinstance(value, _CanonicalNode):
                payload[key] = value.to_dict()
            elif isinstance(value, tuple):
                payload[key] = [item.to_dict() for item in value]
            else:
                payload[key] = value
        return payload


def _check_depth(depth: int) -> None:
    if depth > _MAX_AST_DEPTH:
        raise _invalid(f"Expression AST exceeds maximum depth {_MAX_AST_DEPTH}")


def _check_call_arity(function: str, count: int) -> None:
    minimum, maximum = _CALL_ARITY[function]
    if count < minimum or (maximum is not None and count > maximum):
        if minimum == maximum:
            expected = f"exactly {minimum}"
        elif maximum is None:
            expected = f"at least {minimum}"
        else:
            expected = f"between {minimum} and {maximum}"
        raise _invalid(f"Call {function!r} requires {expected} argument(s)")


def _python_node(
    node: python_ast.AST,
    allowed_names: frozenset[str] | None,
    depth: int,
) -> _CanonicalNode:
    _check_depth(depth)
    if isinstance(node, python_ast.Constant):
        value = _validate_json_scalar(node.value, "Expression constant")
        return _CanonicalNode("constant", (("value", value),))
    if isinstance(node, python_ast.Name):
        name = _validate_factor_name(node.id, allowed_names)
        return _CanonicalNode("name", (("name", name),))
    if isinstance(node, python_ast.UnaryOp):
        operator_name = _PYTHON_UNARY.get(type(node.op))
        if operator_name is None:
            raise _invalid(f"Expression operator {type(node.op).__name__} is not allowed")
        operand = _python_node(node.operand, allowed_names, depth + 1)
        return _CanonicalNode("unary", (("op", operator_name), ("operand", operand)))
    if isinstance(node, python_ast.BinOp):
        operator_name = _PYTHON_BINARY.get(type(node.op))
        if operator_name is None:
            raise _invalid(f"Expression operator {type(node.op).__name__} is not allowed")
        left = _python_node(node.left, allowed_names, depth + 1)
        right = _python_node(node.right, allowed_names, depth + 1)
        return _CanonicalNode(
            "binary", (("op", operator_name), ("left", left), ("right", right))
        )
    if isinstance(node, python_ast.Compare):
        comparison_nodes: list[_CanonicalNode] = []
        operands = (node.left, *node.comparators)
        child_depth = depth if len(node.ops) == 1 else depth + 1
        for index, comparison in enumerate(node.ops):
            operator_name = _PYTHON_COMPARISONS.get(type(comparison))
            if operator_name is None:
                raise _invalid(
                    f"Expression operator {type(comparison).__name__} is not allowed"
                )
            left = _python_node(operands[index], allowed_names, child_depth + 1)
            right = _python_node(operands[index + 1], allowed_names, child_depth + 1)
            comparison_nodes.append(
                _CanonicalNode(
                    "compare",
                    (("op", operator_name), ("left", left), ("right", right)),
                )
            )
        if len(comparison_nodes) == 1:
            return comparison_nodes[0]
        return _CanonicalNode(
            "boolean", (("op", "and"), ("values", tuple(comparison_nodes)))
        )
    if isinstance(node, python_ast.BoolOp):
        if isinstance(node.op, python_ast.And):
            operator_name = "and"
        elif isinstance(node.op, python_ast.Or):
            operator_name = "or"
        else:  # pragma: no cover - all current Python BoolOp variants are covered.
            raise _invalid(f"Expression operator {type(node.op).__name__} is not allowed")
        values = tuple(
            _python_node(value, allowed_names, depth + 1) for value in node.values
        )
        return _CanonicalNode("boolean", (("op", operator_name), ("values", values)))
    if isinstance(node, python_ast.Call):
        if not isinstance(node.func, python_ast.Name):
            raise _invalid("Only direct allow-listed calls are allowed")
        function = node.func.id
        if function not in _ALLOWED_CALLS:
            raise _invalid(f"Call {function!r} is not allowed")
        if node.keywords:
            raise _invalid("Call keyword arguments are not allowed")
        _check_call_arity(function, len(node.args))
        args = tuple(_python_node(arg, allowed_names, depth + 1) for arg in node.args)
        return _CanonicalNode("call", (("function", function), ("args", args)))
    raise _invalid(f"Expression node {type(node).__name__} is not allowed")


def _parse_expression(
    expression: object,
    allowed_names: Collection[str] | None,
) -> _CanonicalNode:
    if not isinstance(expression, str) or not expression.strip():
        raise _invalid("Expression input must be a non-empty string")
    names = _normalize_allowed_names(allowed_names)
    try:
        tree = python_ast.parse(expression, mode="eval")
    except (SyntaxError, TypeError, ValueError) as error:
        raise _invalid("Expression syntax is not allowed") from error
    return _python_node(tree.body, names, 1)


def _exact_node_keys(node: Mapping[str, object], expected: set[str], node_type: str) -> None:
    actual = set(node)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise _invalid(f"Expression {node_type!r} node requires keys {sorted(expected)!r}")
    if unknown:
        raise _invalid(f"Unknown keys for expression {node_type!r} node: {sorted(unknown)!r}")


def _canonical_node(
    raw_node: object,
    allowed_names: frozenset[str] | None,
    depth: int,
) -> _CanonicalNode:
    _check_depth(depth)
    if not isinstance(raw_node, Mapping):
        raise _invalid("Expression AST node must be an object")
    node_type = raw_node.get("type")
    if not isinstance(node_type, str):
        raise _invalid("Expression AST node type must be a string")
    if node_type == "constant":
        _exact_node_keys(raw_node, {"type", "value"}, node_type)
        value = _validate_json_scalar(raw_node["value"], "Expression constant")
        return _CanonicalNode(node_type, (("value", value),))
    if node_type == "name":
        _exact_node_keys(raw_node, {"type", "name"}, node_type)
        name = _validate_factor_name(raw_node["name"], allowed_names)
        return _CanonicalNode(node_type, (("name", name),))
    if node_type == "unary":
        _exact_node_keys(raw_node, {"type", "op", "operand"}, node_type)
        operator_name = raw_node["op"]
        if operator_name not in _AST_UNARY:
            raise _invalid(f"Expression unary operator {operator_name!r} is not allowed")
        operand = _canonical_node(raw_node["operand"], allowed_names, depth + 1)
        return _CanonicalNode(node_type, (("op", operator_name), ("operand", operand)))
    if node_type == "binary":
        _exact_node_keys(raw_node, {"type", "op", "left", "right"}, node_type)
        operator_name = raw_node["op"]
        if operator_name not in _AST_BINARY:
            raise _invalid(f"Expression binary operator {operator_name!r} is not allowed")
        left = _canonical_node(raw_node["left"], allowed_names, depth + 1)
        right = _canonical_node(raw_node["right"], allowed_names, depth + 1)
        return _CanonicalNode(
            node_type, (("op", operator_name), ("left", left), ("right", right))
        )
    if node_type == "compare":
        _exact_node_keys(raw_node, {"type", "op", "left", "right"}, node_type)
        operator_name = raw_node["op"]
        if operator_name not in _AST_COMPARISONS:
            raise _invalid(f"Expression comparison {operator_name!r} is not allowed")
        left = _canonical_node(raw_node["left"], allowed_names, depth + 1)
        right = _canonical_node(raw_node["right"], allowed_names, depth + 1)
        return _CanonicalNode(
            node_type, (("op", operator_name), ("left", left), ("right", right))
        )
    if node_type == "boolean":
        _exact_node_keys(raw_node, {"type", "op", "values"}, node_type)
        operator_name = raw_node["op"]
        if operator_name not in {"and", "or"}:
            raise _invalid(f"Expression boolean operator {operator_name!r} is not allowed")
        raw_values = raw_node["values"]
        if not isinstance(raw_values, list) or len(raw_values) < 2:
            raise _invalid("Expression boolean node requires at least 2 values")
        values = tuple(
            _canonical_node(value, allowed_names, depth + 1) for value in raw_values
        )
        return _CanonicalNode(node_type, (("op", operator_name), ("values", values)))
    if node_type == "call":
        _exact_node_keys(raw_node, {"type", "function", "args"}, node_type)
        function = raw_node["function"]
        if not isinstance(function, str) or function not in _ALLOWED_CALLS:
            raise _invalid(f"Call {function!r} is not allowed")
        raw_args = raw_node["args"]
        if not isinstance(raw_args, list):
            raise _invalid("Expression call args must be an array")
        _check_call_arity(function, len(raw_args))
        args = tuple(_canonical_node(arg, allowed_names, depth + 1) for arg in raw_args)
        return _CanonicalNode(node_type, (("function", function), ("args", args)))
    raise _invalid(f"Expression node type {node_type!r} is not allowed")


def _evaluate_node(node: _CanonicalNode, values: Mapping[str, object]) -> object:
    if node.node_type == "constant":
        return node.get("value")
    if node.node_type == "name":
        name = node.get("name")
        assert isinstance(name, str)
        if name not in values:
            raise _invalid(f"Expression requires factor {name!r}")
        return values[name]
    if node.node_type == "unary":
        operator_name = node.get("op")
        operand = node.get("operand")
        assert isinstance(operator_name, str) and isinstance(operand, _CanonicalNode)
        return _AST_UNARY[operator_name](_evaluate_node(operand, values))
    if node.node_type == "binary":
        operator_name = node.get("op")
        left = node.get("left")
        right = node.get("right")
        assert isinstance(operator_name, str)
        assert isinstance(left, _CanonicalNode) and isinstance(right, _CanonicalNode)
        return _AST_BINARY[operator_name](
            _evaluate_node(left, values), _evaluate_node(right, values)
        )
    if node.node_type == "compare":
        operator_name = node.get("op")
        left = node.get("left")
        right = node.get("right")
        assert isinstance(operator_name, str)
        assert isinstance(left, _CanonicalNode) and isinstance(right, _CanonicalNode)
        return _AST_COMPARISONS[operator_name](
            _evaluate_node(left, values), _evaluate_node(right, values)
        )
    if node.node_type == "boolean":
        operator_name = node.get("op")
        nodes = node.get("values")
        assert isinstance(operator_name, str) and isinstance(nodes, tuple)
        if operator_name == "and":
            return all(bool(_evaluate_node(value, values)) for value in nodes)
        return any(bool(_evaluate_node(value, values)) for value in nodes)
    function = node.get("function")
    args = node.get("args")
    assert isinstance(function, str) and isinstance(args, tuple)
    return _ALLOWED_CALLS[function](*(_evaluate_node(arg, values) for arg in args))


@dataclass(frozen=True, init=False)
class ExpressionConstraint:
    """A predicate parsed once into an immutable allow-listed expression AST."""

    name: str
    _ast: _CanonicalNode = field(repr=False)
    hard: bool
    penalty: str | None
    weight: float | None

    kind: ClassVar[str] = "expression"

    def __init__(
        self,
        name: str,
        expression: str,
        hard: bool = True,
        penalty: str | None = None,
        weight: float | None = None,
        *,
        allowed_names: Collection[str] | None = None,
    ) -> None:
        normalized = _validate_common(name, hard, penalty, weight)
        canonical_ast = _parse_expression(expression, allowed_names)
        object.__setattr__(self, "name", normalized[0])
        object.__setattr__(self, "_ast", canonical_ast)
        object.__setattr__(self, "hard", normalized[1])
        object.__setattr__(self, "penalty", normalized[2])
        object.__setattr__(self, "weight", normalized[3])

    @classmethod
    def _from_ast(
        cls,
        name: str,
        raw_ast: object,
        hard: bool,
        penalty: str | None,
        weight: object,
        allowed_names: Collection[str] | None,
    ) -> "ExpressionConstraint":
        normalized = _validate_common(name, hard, penalty, weight)
        canonical_ast = _canonical_node(raw_ast, _normalize_allowed_names(allowed_names), 1)
        instance = object.__new__(cls)
        object.__setattr__(instance, "name", normalized[0])
        object.__setattr__(instance, "_ast", canonical_ast)
        object.__setattr__(instance, "hard", normalized[1])
        object.__setattr__(instance, "penalty", normalized[2])
        object.__setattr__(instance, "weight", normalized[3])
        return instance

    @property
    def ast(self) -> dict[str, object]:
        """Return a detached JSON-compatible view of the canonical AST."""
        return self._ast.to_dict()

    def satisfied(self, values: Mapping[str, object]) -> bool:
        try:
            result = _evaluate_node(self._ast, values)
        except EngineError:
            raise
        except Exception as error:
            raise _invalid(
                f"ExpressionConstraint {self.name!r} could not be evaluated",
                details={"error": type(error).__name__},
            ) from error
        if self._ast.node_type not in {"compare", "boolean"} and not isinstance(
            result, bool
        ):
            raise _invalid(f"ExpressionConstraint {self.name!r} must evaluate to a boolean")
        return bool(result)

    def to_dict(self) -> dict[str, object]:
        payload = _base_payload(self.kind, self.name, self.hard, self.penalty, self.weight)
        payload["ast"] = self._ast.to_dict()
        return payload


def _normalize_patterns(
    patterns: object,
    context: str,
    allowed_names: frozenset[str] | None = None,
) -> tuple[Mapping[str, JSONScalar], ...] | None:
    if patterns is None:
        return None
    if isinstance(patterns, (str, bytes, bytearray)) or not isinstance(patterns, Sequence):
        raise _invalid(f"{context} must be an ordered sequence of assignments")
    if not patterns:
        raise _invalid(f"{context} must contain at least one assignment")
    normalized: list[Mapping[str, JSONScalar]] = []
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, Mapping) or not pattern:
            raise _invalid(f"{context}[{index}] must be a non-empty assignment object")
        assignment: dict[str, JSONScalar] = {}
        for factor, value in pattern.items():
            factor_name = _validate_factor_name(factor, allowed_names)
            assignment[factor_name] = _validate_json_scalar(
                value, f"{context}[{index}] value for {factor_name!r}"
            )
        normalized.append(MappingProxyType(dict(sorted(assignment.items()))))
    return tuple(normalized)


@dataclass(frozen=True)
class CategoricalCombinationConstraint:
    """Allow or forbid partial categorical assignments."""

    name: str
    allowed: Sequence[Mapping[str, JSONScalar]] | None = None
    forbidden: Sequence[Mapping[str, JSONScalar]] | None = None
    hard: bool = True
    penalty: str | None = None
    weight: float | None = None

    kind: ClassVar[str] = "categorical_combination"

    def __post_init__(self) -> None:
        name, hard, penalty, weight = _validate_common(
            self.name, self.hard, self.penalty, self.weight
        )
        allowed = _normalize_patterns(self.allowed, "allowed")
        forbidden = _normalize_patterns(self.forbidden, "forbidden")
        if (allowed is None) == (forbidden is None):
            raise _invalid(
                "CategoricalCombinationConstraint requires exactly one of allowed or forbidden"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "allowed", allowed)
        object.__setattr__(self, "forbidden", forbidden)
        object.__setattr__(self, "hard", hard)
        object.__setattr__(self, "penalty", penalty)
        object.__setattr__(self, "weight", weight)

    @staticmethod
    def _matches(pattern: Mapping[str, JSONScalar], values: Mapping[str, object]) -> bool:
        return all(
            factor in values and values[factor] == value
            for factor, value in pattern.items()
        )

    def satisfied(self, values: Mapping[str, object]) -> bool:
        if self.allowed is not None:
            return any(self._matches(pattern, values) for pattern in self.allowed)
        assert self.forbidden is not None
        return not any(self._matches(pattern, values) for pattern in self.forbidden)

    def to_dict(self) -> dict[str, object]:
        payload = _base_payload(self.kind, self.name, self.hard, self.penalty, self.weight)
        if self.allowed is not None:
            payload["allowed"] = [dict(pattern) for pattern in self.allowed]
        else:
            assert self.forbidden is not None
            payload["forbidden"] = [dict(pattern) for pattern in self.forbidden]
        return payload


@dataclass(frozen=True)
class OutcomeConstraint:
    """A finite bound on one observed or modeled objective."""

    name: str
    objective: str
    operator: str
    bound: float
    hard: bool = True
    penalty: str | None = None
    weight: float | None = None

    kind: ClassVar[str] = "outcome"

    def __post_init__(self) -> None:
        name, hard, penalty, weight = _validate_common(
            self.name, self.hard, self.penalty, self.weight
        )
        objective = _validate_name(self.objective, "OutcomeConstraint objective")
        if self.operator not in _COMPARISON_OPERATORS:
            raise _invalid(
                "OutcomeConstraint operator must be one of '<=', '>=', or '=='"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(
            self,
            "bound",
            _validate_finite_number(self.bound, "OutcomeConstraint bound"),
        )
        object.__setattr__(self, "hard", hard)
        object.__setattr__(self, "penalty", penalty)
        object.__setattr__(self, "weight", weight)

    def satisfied(self, values: Mapping[str, object]) -> bool:
        actual = _row_number(values, self.objective, f"OutcomeConstraint {self.name!r}")
        return _compare(self.operator, actual, self.bound)

    def to_dict(self) -> dict[str, object]:
        payload = _base_payload(self.kind, self.name, self.hard, self.penalty, self.weight)
        payload.update(
            {"objective": self.objective, "operator": self.operator, "bound": self.bound}
        )
        return payload


def _exact_constraint_keys(
    payload: Mapping[str, object], expected: set[str], kind: str
) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise _invalid(f"Constraint kind {kind!r} requires keys {sorted(expected)!r}")
    if unknown:
        raise _invalid(f"Unknown keys for constraint kind {kind!r}: {sorted(unknown)!r}")


def _factory_common(payload: Mapping[str, object], kind: str) -> tuple[set[str], bool]:
    hard = payload.get("hard")
    if not isinstance(hard, bool):
        raise _invalid(f"Constraint kind {kind!r} requires boolean hard")
    common = {"kind", "name", "hard"}
    if not hard:
        common.update({"penalty", "weight"})
    return common, hard


def constraint_from_dict(
    payload: Mapping[str, object],
    *,
    allowed_names: Collection[str] | None = None,
) -> Constraint:
    """Build a constraint from a strict declarative payload.

    Expression payloads must contain the canonical structured AST. Persisted
    source text is never parsed by this boundary.
    """
    if not isinstance(payload, Mapping):
        raise _invalid("Constraint payload must be an object")
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise _invalid("Constraint payload requires a string kind")
    if kind not in {
        LinearConstraint.kind,
        ExpressionConstraint.kind,
        CategoricalCombinationConstraint.kind,
        OutcomeConstraint.kind,
    }:
        raise _invalid(f"Unknown constraint kind {kind!r}")
    names = _normalize_allowed_names(allowed_names)
    common_keys, hard = _factory_common(payload, kind)
    penalty = payload.get("penalty")
    weight = payload.get("weight")
    if kind == LinearConstraint.kind:
        _exact_constraint_keys(
            payload, common_keys | {"coefficients", "operator", "bound"}, kind
        )
        coefficients = payload["coefficients"]
        if not isinstance(coefficients, Mapping):
            raise _invalid("LinearConstraint coefficients must be an object")
        for factor in coefficients:
            _validate_factor_name(factor, names)
        return LinearConstraint(
            payload["name"],
            coefficients,
            payload["operator"],
            payload["bound"],
            hard=hard,
            penalty=penalty,  # type: ignore[arg-type]
            weight=weight,  # type: ignore[arg-type]
        )
    if kind == ExpressionConstraint.kind:
        _exact_constraint_keys(payload, common_keys | {"ast"}, kind)
        return ExpressionConstraint._from_ast(
            payload["name"],  # type: ignore[arg-type]
            payload["ast"],
            hard,
            penalty,  # type: ignore[arg-type]
            weight,
            names,
        )
    if kind == CategoricalCombinationConstraint.kind:
        has_allowed = "allowed" in payload
        has_forbidden = "forbidden" in payload
        if has_allowed == has_forbidden:
            raise _invalid(
                "CategoricalCombinationConstraint payload requires exactly one of "
                "allowed or forbidden"
            )
        pattern_key = "allowed" if has_allowed else "forbidden"
        _exact_constraint_keys(payload, common_keys | {pattern_key}, kind)
        patterns = _normalize_patterns(payload[pattern_key], pattern_key, names)
        return CategoricalCombinationConstraint(
            payload["name"],  # type: ignore[arg-type]
            allowed=patterns if has_allowed else None,
            forbidden=patterns if has_forbidden else None,
            hard=hard,
            penalty=penalty,  # type: ignore[arg-type]
            weight=weight,  # type: ignore[arg-type]
        )
    _exact_constraint_keys(
        payload, common_keys | {"objective", "operator", "bound"}, kind
    )
    objective = _validate_factor_name(payload["objective"], names)
    return OutcomeConstraint(
        payload["name"],  # type: ignore[arg-type]
        objective,
        payload["operator"],  # type: ignore[arg-type]
        payload["bound"],  # type: ignore[arg-type]
        hard=hard,
        penalty=penalty,  # type: ignore[arg-type]
        weight=weight,  # type: ignore[arg-type]
    )


__all__ = [
    "CategoricalCombinationConstraint",
    "Constraint",
    "ExpressionConstraint",
    "LinearConstraint",
    "OutcomeConstraint",
    "constraint_from_dict",
]
