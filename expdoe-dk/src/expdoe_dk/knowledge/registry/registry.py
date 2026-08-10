"""Insertion-independent exact knowledge pattern registry."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from expdoe_dk.errors import EngineError, ErrorCode

from ..artifacts import ARTIFACT_CATEGORIES, OptimizationArtifacts, merge_artifacts
from ..guard import KnowledgeValidationResult
from ..specs import KnowledgePatternSpec
from .definition import KnowledgePatternDefinition

if TYPE_CHECKING:
    from expdoe_dk.domain import ObservationBatch, Space


class PatternRegistry:
    """Own exact ``(pattern, version)`` definitions and dispatch to them."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], KnowledgePatternDefinition] = {}

    def register(self, definition: KnowledgePatternDefinition) -> None:
        if not isinstance(definition, KnowledgePatternDefinition):
            raise TypeError("definition must be a KnowledgePatternDefinition")
        key = (definition.pattern, definition.version)
        if key in self._definitions:
            raise ValueError(f"Pattern {key} already registered")
        self._definitions[key] = definition

    def resolve(self, pattern: str, version: str) -> KnowledgePatternDefinition:
        try:
            return self._definitions[(pattern, version)]
        except KeyError as error:
            raise EngineError(
                ErrorCode.KNOWLEDGE_INVALID,
                f"Unknown pattern {pattern}@{version}",
            ) from error

    def definitions(self) -> tuple[KnowledgePatternDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    @staticmethod
    def _require_spec(spec: object) -> KnowledgePatternSpec:
        if not isinstance(spec, KnowledgePatternSpec):
            raise TypeError("spec must be a KnowledgePatternSpec")
        return spec

    def validate(
        self,
        spec: KnowledgePatternSpec,
        space: "Space",
        observations: "ObservationBatch | None" = None,
    ) -> KnowledgeValidationResult:
        spec = self._require_spec(spec)
        result = self.resolve(spec.pattern, spec.version).validator(
            spec, space, observations
        )
        if not isinstance(result, KnowledgeValidationResult):
            raise TypeError("pattern validator must return KnowledgeValidationResult")
        if result.pattern_id != spec.pattern_id:
            raise EngineError(
                ErrorCode.KNOWLEDGE_INVALID,
                "Knowledge validation provenance mismatch: "
                f"expected pattern_id {spec.pattern_id!r}, "
                f"got {result.pattern_id!r}",
                details={
                    "expected": {"pattern_id": spec.pattern_id},
                    "actual": {"pattern_id": result.pattern_id},
                },
            )
        return result

    def validate_many(
        self,
        specs: Iterable[KnowledgePatternSpec],
        space: "Space",
        observations: "ObservationBatch | None" = None,
    ) -> tuple[KnowledgeValidationResult, ...]:
        return tuple(self.validate(spec, space, observations) for spec in specs)

    def compile_many(
        self,
        specs: Iterable[KnowledgePatternSpec],
        space: "Space",
        observations: "ObservationBatch | None" = None,
    ) -> OptimizationArtifacts:
        compiled_sets: list[OptimizationArtifacts] = []
        for spec in specs:
            checked = self._require_spec(spec)
            artifacts = self.resolve(checked.pattern, checked.version).compiler(
                checked, space, observations
            )
            if not isinstance(artifacts, OptimizationArtifacts):
                raise TypeError("pattern compiler must return OptimizationArtifacts")
            self._validate_artifact_provenance(checked, artifacts)
            compiled_sets.append(artifacts)
        merged = merge_artifacts(compiled_sets)
        self._reject_proven_empty_safety_intersection(merged, space)
        return merged

    @classmethod
    def _reject_proven_empty_safety_intersection(
        cls, artifacts: OptimizationArtifacts, space: "Space"
    ) -> None:
        """Reject only interval contradictions proved by canonical hard constraints."""
        intervals: dict[str, dict[str, object]] = {}
        for artifact in sorted(
            artifacts.parameter_constraints,
            key=lambda item: (item.source_pattern_id, item.kind),
        ):
            if artifact.source_pattern not in {"safe_region", "forbidden_region"}:
                continue
            constraint = artifact.payload.get("constraint")
            if not isinstance(constraint, Mapping) or constraint.get("hard") is not True:
                continue
            ast = constraint.get("ast")
            if isinstance(ast, Mapping) and ast.get("type") not in {
                "compare",
                "boolean",
                "constant",
            }:
                raise EngineError(
                    ErrorCode.KNOWLEDGE_INVALID,
                    "Hard safety expression must have a boolean root",
                    details={
                        "source_pattern_ids": [artifact.source_pattern_id],
                        "constraint": constraint.get("name"),
                        "root_type": ast.get("type"),
                    },
                )
            is_constant, constant_value = cls._constant_value(ast)
            if is_constant and type(constant_value) is not bool:
                raise EngineError(
                    ErrorCode.KNOWLEDGE_INVALID,
                    "Hard safety expression must evaluate to a boolean",
                    details={
                        "source_pattern_ids": [artifact.source_pattern_id],
                        "constraint": constraint.get("name"),
                        "root_type": ast.get("type") if isinstance(ast, Mapping) else None,
                    },
                )
            if is_constant and constant_value is False:
                raise EngineError(
                    ErrorCode.KNOWLEDGE_CONFLICT,
                    "Hard safety region is provably empty",
                    details={
                        "source_pattern_ids": [artifact.source_pattern_id],
                        "constraint": constraint.get("name"),
                        "reason": "constant_false",
                    },
                )
            comparisons = cls._interval_comparisons(ast)
            if comparisons is None:
                continue
            for factor, operator, value in comparisons:
                if factor not in space.param_names:
                    continue
                parameter = space.param_by_name(factor)
                if parameter.kind not in {"continuous", "integer", "discrete"}:
                    continue
                if parameter.bounds is not None:
                    physical_lower, physical_upper = map(float, parameter.bounds)
                else:
                    levels = tuple(float(item) for item in parameter.numeric_levels)
                    physical_lower, physical_upper = min(levels), max(levels)
                interval = intervals.setdefault(
                    factor,
                    {
                        "lower": physical_lower,
                        "upper": physical_upper,
                        "lower_closed": True,
                        "upper_closed": True,
                        "sources": set(),
                    },
                )
                sources = interval["sources"]
                assert isinstance(sources, set)
                sources.add(artifact.source_pattern_id)
                if operator in {">", ">="}:
                    current = float(interval["lower"])
                    if value > current:
                        interval["lower"] = value
                        interval["lower_closed"] = operator == ">="
                    elif value == current and operator == ">":
                        interval["lower_closed"] = False
                elif operator in {"<", "<="}:
                    current = float(interval["upper"])
                    if value < current:
                        interval["upper"] = value
                        interval["upper_closed"] = operator == "<="
                    elif value == current and operator == "<":
                        interval["upper_closed"] = False
                elif operator == "==":
                    current_lower = float(interval["lower"])
                    current_upper = float(interval["upper"])
                    interval["lower"] = max(current_lower, value)
                    interval["upper"] = min(current_upper, value)
                    if value > current_lower:
                        interval["lower_closed"] = True
                    if value < current_upper:
                        interval["upper_closed"] = True
                lower = float(interval["lower"])
                upper = float(interval["upper"])
                if not cls._domain_has_interval_value(
                    parameter,
                    lower,
                    upper,
                    bool(interval["lower_closed"]),
                    bool(interval["upper_closed"]),
                ):
                    source_ids = sorted(sources)
                    raise EngineError(
                        ErrorCode.KNOWLEDGE_CONFLICT,
                        f"Hard safety regions have an empty intersection for {factor!r}",
                        details={
                            "factor": factor,
                            "source_pattern_ids": source_ids,
                            "lower": lower,
                            "upper": upper,
                        },
                    )

    @staticmethod
    def _domain_has_interval_value(
        parameter,
        lower: float,
        upper: float,
        lower_closed: bool,
        upper_closed: bool,
    ) -> bool:
        if lower > upper:
            return False
        if parameter.kind == "continuous":
            return lower < upper or (lower_closed and upper_closed)
        if parameter.kind == "discrete":
            return any(
                (value > lower or (lower_closed and value == lower))
                and (value < upper or (upper_closed and value == upper))
                for value in (float(item) for item in parameter.numeric_levels)
            )
        grid_low = int(parameter.bounds[0])
        step = int(parameter.step or 1)
        index = max(0, math.ceil((lower - grid_low) / step))
        candidate = grid_low + index * step
        if candidate == lower and not lower_closed:
            candidate += step
        return candidate < upper or (upper_closed and candidate == upper)

    @classmethod
    def _constant_value(cls, node: object) -> tuple[bool, object]:
        if not isinstance(node, Mapping):
            return False, None
        if node.get("type") == "constant":
            return True, node.get("value")
        if node.get("type") == "unary":
            known, value = cls._constant_value(node.get("operand"))
            if not known or type(value) not in (int, float):
                return False, None
            return True, value if node.get("op") == "+" else -value
        if node.get("type") == "binary":
            left_known, left_value = cls._constant_value(node.get("left"))
            right_known, right_value = cls._constant_value(node.get("right"))
            if (
                not left_known
                or not right_known
                or type(left_value) not in (int, float)
                or type(right_value) not in (int, float)
            ):
                return False, None
            operations = {
                "+": lambda: left_value + right_value,
                "-": lambda: left_value - right_value,
                "*": lambda: left_value * right_value,
                "/": lambda: left_value / right_value,
                "**": lambda: left_value**right_value,
            }
            try:
                return True, operations[node.get("op")]()
            except (KeyError, ArithmeticError, OverflowError):
                return False, None
        if node.get("type") == "compare":
            left = node.get("left")
            right = node.get("right")
            left_known, left_value = cls._constant_value(left)
            right_known, right_value = cls._constant_value(right)
            if not left_known or not right_known:
                return False, None
            operator = node.get("op")
            left_numeric = type(left_value) in (int, float)
            right_numeric = type(right_value) in (int, float)
            if operator in {"==", "!="}:
                if left_numeric or right_numeric:
                    equal = left_numeric and right_numeric and left_value == right_value
                else:
                    equal = type(left_value) is type(right_value) and left_value == right_value
                return True, equal if operator == "==" else not equal
            if not left_numeric or not right_numeric:
                return False, None
            comparisons = {
                "<": left_value < right_value,
                "<=": left_value <= right_value,
                ">": left_value > right_value,
                ">=": left_value >= right_value,
            }
            return (True, comparisons[operator]) if operator in comparisons else (False, None)
        if node.get("type") == "boolean" and node.get("op") in {"and", "or"}:
            values = node.get("values")
            if not isinstance(values, (list, tuple)):
                return False, None
            resolved = tuple(cls._constant_value(value) for value in values)
            if any(not known or type(value) is not bool for known, value in resolved):
                return False, None
            booleans = tuple(value for _, value in resolved)
            return True, all(booleans) if node.get("op") == "and" else any(booleans)
        return False, None

    @classmethod
    def _interval_comparisons(
        cls, node: object
    ) -> list[tuple[str, str, float]] | None:
        """Extract conjunctions of scalar interval comparisons, else be inconclusive."""
        if not isinstance(node, Mapping):
            return None
        if node.get("type") == "boolean" and node.get("op") == "and":
            values = node.get("values")
            if not isinstance(values, (list, tuple)):
                return None
            combined: list[tuple[str, str, float]] = []
            for value in values:
                extracted = cls._interval_comparisons(value)
                if extracted is None:
                    return None
                combined.extend(extracted)
            return combined
        if node.get("type") != "compare":
            return None
        operator = node.get("op")
        left = node.get("left")
        right = node.get("right")
        if (
            operator == "=="
            and isinstance(right, Mapping)
            and right.get("type") == "constant"
            and right.get("value") is False
        ):
            extracted = cls._interval_comparisons(left)
            if extracted is None or len(extracted) != 1:
                return None
            factor, inner_operator, value = extracted[0]
            inverse = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!="}
            inverted = inverse.get(inner_operator)
            return None if inverted in {None, "!="} else [(factor, inverted, value)]
        if operator not in {"<", "<=", "==", ">=", ">"}:
            return None
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return None
        if left.get("type") == "name" and right.get("type") == "constant":
            factor = left.get("name")
            raw_value = right.get("value")
        elif left.get("type") == "constant" and right.get("type") == "name":
            factor = right.get("name")
            raw_value = left.get("value")
            operator = {"<": ">", "<=": ">=", "==": "==", ">=": "<=", ">": "<"}[operator]
        else:
            return None
        if type(factor) is not str or type(raw_value) not in (int, float):
            return None
        return [(factor, operator, float(raw_value))]

    @staticmethod
    def _validate_artifact_provenance(
        spec: KnowledgePatternSpec, artifacts: OptimizationArtifacts
    ) -> None:
        expected = {
            "source_pattern_id": spec.pattern_id,
            "source_pattern": spec.pattern,
            "source_version": spec.version,
        }
        for category in ARTIFACT_CATEGORIES:
            for index, artifact in enumerate(getattr(artifacts, category)):
                actual = {
                    "source_pattern_id": artifact.source_pattern_id,
                    "source_pattern": artifact.source_pattern,
                    "source_version": artifact.source_version,
                }
                if actual != expected:
                    raise EngineError(
                        ErrorCode.KNOWLEDGE_INVALID,
                        "Knowledge artifact provenance mismatch in "
                        f"{category}[{index}] for {spec.pattern}@{spec.version}",
                        details={
                            "category": category,
                            "artifact_index": index,
                            "expected": expected,
                            "actual": actual,
                        },
                    )

    def render(self, spec: KnowledgePatternSpec, space: "Space") -> str:
        spec = self._require_spec(spec)
        rendered = self.resolve(spec.pattern, spec.version).renderer(spec, space)
        if type(rendered) is not str:
            raise TypeError("pattern renderer must return a string")
        return rendered


__all__ = ["PatternRegistry"]
