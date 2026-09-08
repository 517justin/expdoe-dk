"""Insertion-independent exact knowledge pattern registry."""
from __future__ import annotations

import math
from itertools import product
from collections.abc import Iterable, Mapping
from threading import RLock
from typing import TYPE_CHECKING

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from scipy.optimize import linprog

from expdoe_dk.domain import ExpressionConstraint, constraint_from_dict
from expdoe_dk.errors import EngineError, ErrorCode

from ..artifacts import (
    ARTIFACT_CATEGORIES,
    OptimizationArtifact,
    OptimizationArtifacts,
    merge_artifacts,
)
from ..guard import CompatibilityResult, KnowledgeValidationResult
from ..specs import KnowledgePatternSpec
from .definition import KnowledgePatternDefinition

if TYPE_CHECKING:
    from expdoe_dk.domain import ObservationBatch, Space


_SAFETY_ENUMERATION_LIMIT = 100_000

_SafetyConstraint = tuple[str, ExpressionConstraint]


class PatternRegistry:
    """Own exact ``(pattern, version)`` definitions and dispatch to them."""

    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], KnowledgePatternDefinition] = {}
        self._provider_records: dict[tuple[str, str], object] = {}
        self._lock = RLock()

    def register(self, definition: KnowledgePatternDefinition) -> None:
        self.register_many((definition,))

    def register_many(
        self, definitions: Iterable[KnowledgePatternDefinition]
    ) -> None:
        """Atomically register a batch after validating every definition."""
        if isinstance(definitions, (str, bytes)) or not isinstance(
            definitions, Iterable
        ):
            raise TypeError("definitions must be an iterable of definitions")
        incoming = tuple(definitions)
        for definition in incoming:
            self._validate_definition(definition)

        with self._lock:
            original = self._definitions
            replacement = dict(original)
            for definition in incoming:
                key = (definition.pattern, definition.version)
                if key in replacement:
                    raise ValueError(f"Pattern {key} already registered")
                replacement[key] = definition
            try:
                self._definitions = replacement
            except BaseException:
                try:
                    object.__setattr__(self, "_definitions", original)
                except BaseException:
                    pass
                raise

    @staticmethod
    def _validate_definition(definition: KnowledgePatternDefinition) -> None:
        if not isinstance(definition, KnowledgePatternDefinition):
            raise TypeError("definition must be a KnowledgePatternDefinition")
        try:
            Draft202012Validator.check_schema(
                PatternRegistry._plain_json(definition.schema)
            )
        except SchemaError as error:
            raise EngineError(
                ErrorCode.KNOWLEDGE_INVALID,
                f"Invalid JSON Schema for {definition.pattern}@{definition.version}",
                details={
                    "pattern": definition.pattern,
                    "version": definition.version,
                    "schema_path": list(error.schema_path),
                    "message": error.message,
                },
            ) from error

    def resolve(self, pattern: str, version: str) -> KnowledgePatternDefinition:
        with self._lock:
            try:
                return self._definitions[(pattern, version)]
            except KeyError as error:
                raise EngineError(
                    ErrorCode.KNOWLEDGE_INVALID,
                    f"Unknown pattern {pattern}@{version}",
                ) from error

    def definitions(self) -> tuple[KnowledgePatternDefinition, ...]:
        with self._lock:
            return tuple(
                self._definitions[key] for key in sorted(self._definitions)
            )

    @property
    def provider_records(self) -> tuple[object, ...]:
        """Return detached immutable provenance for explicitly loaded providers."""
        with self._lock:
            return tuple(
                self._provider_records[key] for key in sorted(self._provider_records)
            )

    def _publish_provider_transaction(
        self,
        definitions: Iterable[KnowledgePatternDefinition],
        records: Iterable[object],
    ) -> None:
        """Publish provider definitions and provenance as one rollback-safe state."""
        if isinstance(definitions, (str, bytes)) or not isinstance(
            definitions, Iterable
        ):
            raise TypeError("definitions must be an iterable of definitions")
        if isinstance(records, (str, bytes)) or not isinstance(records, Iterable):
            raise TypeError("records must be an iterable of provider records")
        incoming_definitions = tuple(definitions)
        incoming_records = tuple(records)

        with self._lock:
            from .providers import ProviderRecord

            for definition in incoming_definitions:
                self._validate_definition(definition)
            if not all(isinstance(record, ProviderRecord) for record in incoming_records):
                raise TypeError("records must contain ProviderRecord values")

            original_definitions = self._definitions
            original_records = self._provider_records
            replacement_definitions = dict(original_definitions)
            replacement_records = dict(original_records)

            incoming_definition_keys: list[tuple[str, str]] = []
            for definition in incoming_definitions:
                key = (definition.pattern, definition.version)
                if key in replacement_definitions:
                    raise ValueError(f"Pattern {key} already registered")
                replacement_definitions[key] = definition
                incoming_definition_keys.append(key)

            recorded_definition_keys: list[tuple[str, str]] = []
            for record in incoming_records:
                key = (
                    record.canonical_distribution_name,
                    record.entry_point_name,
                )
                if key in replacement_records:
                    raise ValueError(f"Provider {key} already registered")
                replacement_records[key] = record
                recorded_definition_keys.extend(
                    (definition.pattern, definition.version)
                    for definition in record.definitions
                )

            if sorted(recorded_definition_keys) != sorted(incoming_definition_keys):
                raise ValueError(
                    "Provider records must describe exactly the published definitions"
                )

            try:
                self._definitions = replacement_definitions
                self._provider_records = replacement_records
            except BaseException:
                object.__setattr__(self, "_definitions", original_definitions)
                object.__setattr__(self, "_provider_records", original_records)
                raise

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
        definition = self.resolve(spec.pattern, spec.version)
        self._validate_parameters(spec, definition)
        result = definition.validator(
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
        return tuple(
            self.validate(spec, space, observations)
            for spec in specs
            if self._require_spec(spec).enabled
        )

    def compile_many(
        self,
        specs: Iterable[KnowledgePatternSpec],
        space: "Space",
        observations: "ObservationBatch | None" = None,
    ) -> OptimizationArtifacts:
        compiled_sets: list[OptimizationArtifacts] = []
        for spec in specs:
            checked = self._require_spec(spec)
            if not checked.enabled:
                continue
            definition = self.resolve(checked.pattern, checked.version)
            self._validate_parameters(checked, definition)
            compatibility = definition.compatibility(checked, space)
            if not isinstance(compatibility, CompatibilityResult):
                raise TypeError(
                    "pattern compatibility must return CompatibilityResult"
                )
            if not compatibility.compatible:
                raise EngineError(
                    ErrorCode.KNOWLEDGE_INVALID,
                    f"Knowledge pattern {checked.pattern}@{checked.version} is incompatible",
                    details={
                        "pattern_id": checked.pattern_id,
                        "pattern": checked.pattern,
                        "version": checked.version,
                        "reasons": list(compatibility.reasons),
                    },
                )
            validation = self.validate(checked, space, observations)
            if validation.state == "invalid":
                raise EngineError(
                    validation.error_code or ErrorCode.KNOWLEDGE_INVALID,
                    f"Knowledge pattern {checked.pattern}@{checked.version} is invalid",
                    details=(
                        dict(validation.error_details)
                        if validation.error_details is not None
                        else {
                            "pattern_id": checked.pattern_id,
                            "pattern": checked.pattern,
                            "version": checked.version,
                            "summary": validation.summary,
                            "errors": list(validation.errors),
                        }
                    ),
                )
            artifacts = definition.compiler(
                checked, space, observations
            )
            if not isinstance(artifacts, OptimizationArtifacts):
                raise TypeError("pattern compiler must return OptimizationArtifacts")
            self._validate_artifact_provenance(checked, artifacts)
            compiled_sets.append(artifacts)
            if validation.state != "valid":
                compiled_sets.append(
                    OptimizationArtifacts(
                        diagnostics=(
                            OptimizationArtifact(
                                kind="knowledge_validation",
                                payload={
                                    "state": validation.state,
                                    "summary": validation.summary,
                                    "errors": list(validation.errors),
                                    "warnings": list(validation.warnings),
                                    "effective_confidence": validation.effective_confidence,
                                },
                                source_pattern_id=checked.pattern_id,
                                source_pattern=checked.pattern,
                                source_version=checked.version,
                            ),
                        )
                    )
                )
        merged = merge_artifacts(compiled_sets)
        safety_diagnostics = self._safety_feasibility_diagnostics(merged, space)
        if not safety_diagnostics:
            return merged
        return merge_artifacts(
            (
                merged,
                OptimizationArtifacts(diagnostics=safety_diagnostics),
            )
        )

    @staticmethod
    def _validate_parameters(
        spec: KnowledgePatternSpec, definition: KnowledgePatternDefinition
    ) -> None:
        schema = PatternRegistry._plain_json(definition.schema)
        parameters = PatternRegistry._plain_json(spec.parameters)
        error = next(
            Draft202012Validator(schema).iter_errors(parameters),
            None,
        )
        if error is None:
            return
        raise EngineError(
            ErrorCode.KNOWLEDGE_INVALID,
            f"Invalid parameters for {spec.pattern}@{spec.version}: {error.message}",
            details={
                "pattern_id": spec.pattern_id,
                "pattern": spec.pattern,
                "version": spec.version,
                "parameter_path": list(error.path),
                "schema_path": list(error.schema_path),
                "validator": error.validator,
                "message": error.message,
            },
        )

    @staticmethod
    def _plain_json(value):
        if isinstance(value, Mapping):
            return {
                key: PatternRegistry._plain_json(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return [PatternRegistry._plain_json(item) for item in value]
        return value

    @classmethod
    def _reject_proven_empty_safety_intersection(
        cls, artifacts: OptimizationArtifacts, space: "Space"
    ) -> None:
        """Reject certified conflicts while retaining a compatibility entry point."""
        cls._safety_feasibility_diagnostics(artifacts, space)

    @classmethod
    def _safety_feasibility_diagnostics(
        cls, artifacts: OptimizationArtifacts, space: "Space"
    ) -> tuple[OptimizationArtifact, ...]:
        """Certify supported safety sets or return explicit inconclusive diagnostics."""
        safety_artifacts = cls._safety_artifacts(artifacts)
        if not safety_artifacts:
            return ()
        safety_constraints = cls._constraints_from_safety_artifacts(
            safety_artifacts, space
        )
        status, details = cls._safety_constraint_certificate(
            safety_constraints, space
        )
        if status == "infeasible":
            raise EngineError(
                ErrorCode.KNOWLEDGE_CONFLICT,
                "Hard safety regions have a certified empty intersection",
                details=details,
            )
        if status == "feasible":
            return ()
        source_pattern_ids = sorted(
            {artifact.source_pattern_id for artifact in safety_artifacts}
        )
        payload = {
            "status": "inconclusive",
            **details,
            "source_pattern_ids": source_pattern_ids,
        }
        return tuple(
            OptimizationArtifact(
                kind="safety_feasibility",
                payload=payload,
                source_pattern_id=artifact.source_pattern_id,
                source_pattern=artifact.source_pattern,
                source_version=artifact.source_version,
            )
            for artifact in safety_artifacts
        )

    @staticmethod
    def _safety_artifacts(
        artifacts: OptimizationArtifacts,
    ) -> tuple[OptimizationArtifact, ...]:
        return tuple(
            artifact
            for artifact in sorted(
                artifacts.parameter_constraints,
                key=lambda item: (item.source_pattern_id, item.kind),
            )
            if artifact.source_pattern in {"safe_region", "forbidden_region"}
            and isinstance(artifact.payload.get("constraint"), Mapping)
            and artifact.payload["constraint"].get("hard") is True
        )

    @classmethod
    def _constraints_from_safety_artifacts(
        cls,
        safety_artifacts: tuple[OptimizationArtifact, ...],
        space: "Space",
    ) -> tuple[_SafetyConstraint, ...]:
        parsed: list[_SafetyConstraint] = []
        for artifact in safety_artifacts:
            constraint = constraint_from_dict(
                cls._plain_json(artifact.payload["constraint"]),
                allowed_names=space.param_names,
            )
            if not isinstance(constraint, ExpressionConstraint):
                raise EngineError(
                    ErrorCode.KNOWLEDGE_INVALID,
                    "Hard safety artifact must contain an expression constraint",
                    details={"source_pattern_ids": [artifact.source_pattern_id]},
                )
            parsed.append((artifact.source_pattern_id, constraint))
        return tuple(parsed)

    @classmethod
    def _safety_constraint_certificate(
        cls,
        safety_constraints: tuple[_SafetyConstraint, ...],
        space: "Space",
    ) -> tuple[str, dict[str, object]]:
        """Certify parsed constraints without invoking a pattern compiler."""
        ordered = tuple(sorted(safety_constraints, key=lambda item: item[0]))
        cls._reject_invalid_safety_roots(ordered)
        all_finite = all(
            parameter.cardinality is not None for parameter in space.params
        )
        if all_finite:
            return cls._extended_safety_certificate(ordered, space)
        if cls._mixed_domain_requires_exact_finite_arithmetic(space):
            return "inconclusive", {
                "reason": "finite_domain_numeric_precision"
            }

        cls._reject_scalar_safety_intersection(ordered, space)
        return cls._extended_safety_certificate(ordered, space)

    @staticmethod
    def _mixed_domain_requires_exact_finite_arithmetic(space: "Space") -> bool:
        """Detect finite integral levels that cannot safely enter float proofs."""
        for parameter in space.params:
            if parameter.cardinality is None:
                continue
            if parameter.kind == "integer":
                assert parameter.bounds is not None
                if any(abs(int(value)) > 2**53 for value in parameter.bounds):
                    return True
                continue
            if parameter.kind == "discrete" and any(
                type(value) is int and float(value) != value
                for value in parameter.numeric_levels
            ):
                return True
        return False

    @classmethod
    def _reject_invalid_safety_roots(
        cls,
        safety_constraints: tuple[_SafetyConstraint, ...],
    ) -> None:
        """Reject non-predicate roots and exact constant contradictions."""
        for source_pattern_id, constraint in safety_constraints:
            ast = constraint.ast
            if ast.get("type") not in {"compare", "boolean", "constant"}:
                raise EngineError(
                    ErrorCode.KNOWLEDGE_INVALID,
                    "Hard safety expression must have a boolean root",
                    details={
                        "source_pattern_ids": [source_pattern_id],
                        "constraint": constraint.name,
                        "root_type": ast.get("type"),
                    },
                )
            is_constant, constant_value = cls._constant_value(ast)
            if is_constant and type(constant_value) is not bool:
                raise EngineError(
                    ErrorCode.KNOWLEDGE_INVALID,
                    "Hard safety expression must evaluate to a boolean",
                    details={
                        "source_pattern_ids": [source_pattern_id],
                        "constraint": constraint.name,
                        "root_type": ast.get("type"),
                    },
                )
            if is_constant and constant_value is False:
                raise EngineError(
                    ErrorCode.KNOWLEDGE_CONFLICT,
                    "Hard safety region is provably empty",
                    details={
                        "source_pattern_ids": [source_pattern_id],
                        "constraint": constraint.name,
                        "reason": "constant_false",
                    },
                )

    @classmethod
    def _reject_scalar_safety_intersection(
        cls,
        safety_constraints: tuple[_SafetyConstraint, ...],
        space: "Space",
    ) -> None:
        """Reject interval contradictions proved by canonical hard constraints."""
        intervals: dict[str, dict[str, object]] = {}
        for source_pattern_id, constraint in safety_constraints:
            ast = constraint.ast
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
                sources.add(source_pattern_id)
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

    @classmethod
    def _extended_safety_certificate(
        cls,
        safety_constraints: tuple[_SafetyConstraint, ...],
        space: "Space",
    ) -> tuple[str, dict[str, object]]:
        """Return feasible/infeasible/inconclusive without sampling."""
        source_pattern_ids = sorted(
            {source_pattern_id for source_pattern_id, _ in safety_constraints}
        )
        constraints = tuple(
            constraint for _, constraint in safety_constraints
        )
        asts = tuple(
            constraint.ast for constraint in constraints
        )

        for ast in asts:
            predicate_status, _ = cls._affine_predicates(ast)
            if predicate_status == "infeasible":
                return "infeasible", {
                    "source_pattern_ids": source_pattern_ids,
                    "reason": "tautological_false",
                }

        finite_parameters = tuple(
            parameter for parameter in space.params if parameter.cardinality is not None
        )
        enumeration_size = math.prod(
            int(parameter.cardinality) for parameter in finite_parameters
        )
        if enumeration_size > _SAFETY_ENUMERATION_LIMIT:
            return "inconclusive", {
                "reason": "finite_domain_resource_limit",
                "enumeration_size": enumeration_size,
                "enumeration_limit": _SAFETY_ENUMERATION_LIMIT,
            }

        finite_levels = tuple(
            cls._finite_parameter_levels(parameter)
            for parameter in finite_parameters
        )
        finite_assignments = product(*finite_levels) if finite_levels else ((),)
        all_finite = len(finite_parameters) == len(space.params)
        if all_finite:
            for values in finite_assignments:
                row = {
                    parameter.name: value
                    for parameter, value in zip(finite_parameters, values)
                }
                try:
                    if all(constraint.satisfied(row) for constraint in constraints):
                        return "feasible", {"reason": "exact_finite_enumeration"}
                except (ArithmeticError, EngineError, OverflowError, TypeError, ValueError):
                    return "inconclusive", {
                        "reason": "finite_domain_evaluation_inconclusive"
                    }
            return "infeasible", {
                "source_pattern_ids": source_pattern_ids,
                "reason": "exact_finite_enumeration",
            }

        predicates: list[tuple[dict[str, float], float, str]] = []
        for ast in asts:
            predicate_status, extracted = cls._affine_predicates(ast)
            if predicate_status == "unsupported":
                return "inconclusive", {"reason": "unsupported_continuous_ast"}
            if predicate_status == "infeasible":
                return "infeasible", {
                    "source_pattern_ids": source_pattern_ids,
                    "reason": "tautological_false",
                }
            predicates.extend(extracted)

        continuous_parameters = tuple(
            parameter for parameter in space.params if parameter.kind == "continuous"
        )
        saw_inconclusive = False
        for values in finite_assignments:
            fixed = {
                parameter.name: value
                for parameter, value in zip(finite_parameters, values)
            }
            status = cls._solve_affine_feasibility(
                predicates,
                continuous_parameters,
                fixed,
            )
            if status == "feasible":
                return "feasible", {"reason": "continuous_affine_feasible"}
            if status == "inconclusive":
                saw_inconclusive = True
        if saw_inconclusive:
            return "inconclusive", {"reason": "affine_solver_inconclusive"}
        return "infeasible", {
            "source_pattern_ids": source_pattern_ids,
            "reason": "continuous_affine_infeasible",
        }

    @staticmethod
    def _finite_parameter_levels(parameter) -> tuple[object, ...]:
        if parameter.kind in {"categorical", "ordinal"}:
            return tuple(parameter.values or ())
        return tuple(parameter.numeric_levels)

    @classmethod
    def _affine_predicates(
        cls, node: object
    ) -> tuple[str, list[tuple[dict[str, float], float, str]]]:
        """Extract an affine conjunction, recognizing structural tautologies."""
        if not isinstance(node, Mapping):
            return "unsupported", []
        known, value = cls._constant_value(node)
        if known:
            if type(value) is not bool:
                return "unsupported", []
            return ("feasible", []) if value else ("infeasible", [])
        if node.get("type") == "boolean" and node.get("op") == "and":
            values = node.get("values")
            if not isinstance(values, (list, tuple)):
                return "unsupported", []
            combined: list[tuple[dict[str, float], float, str]] = []
            for item in values:
                status, predicates = cls._affine_predicates(item)
                if status != "feasible":
                    return status, []
                combined.extend(predicates)
            return "feasible", combined
        if node.get("type") != "compare":
            return "unsupported", []
        operator = node.get("op")
        left = node.get("left")
        right = node.get("right")
        if cls._plain_json(left) == cls._plain_json(right):
            if operator in {"==", "<=", ">="}:
                return "feasible", []
            if operator in {"!=", "<", ">"}:
                return "infeasible", []
            return "unsupported", []
        if (
            operator == "=="
            and isinstance(right, Mapping)
            and right.get("type") == "constant"
            and right.get("value") is False
            and isinstance(left, Mapping)
            and left.get("type") == "compare"
        ):
            inverse = {
                "<": ">=",
                "<=": ">",
                ">": "<=",
                ">=": "<",
                "==": "!=",
                "!=": "==",
            }.get(left.get("op"))
            if inverse is None:
                return "unsupported", []
            inverted = dict(left)
            inverted["op"] = inverse
            return cls._affine_predicates(inverted)
        if operator not in {"<", "<=", "==", ">=", ">"}:
            return "unsupported", []
        left_affine = cls._affine_expression(left)
        right_affine = cls._affine_expression(right)
        if left_affine is None or right_affine is None:
            return "unsupported", []
        coefficients = dict(left_affine[0])
        for name, coefficient in right_affine[0].items():
            coefficients[name] = coefficients.get(name, 0.0) - coefficient
            if coefficients[name] == 0.0:
                del coefficients[name]
        return "feasible", [
            (coefficients, left_affine[1] - right_affine[1], operator)
        ]

    @classmethod
    def _affine_expression(
        cls, node: object
    ) -> tuple[dict[str, float], float] | None:
        if not isinstance(node, Mapping):
            return None
        node_type = node.get("type")
        if node_type == "constant":
            value = node.get("value")
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                return None
            return {}, float(value)
        if node_type == "name":
            name = node.get("name")
            return ({name: 1.0}, 0.0) if type(name) is str else None
        if node_type == "unary" and node.get("op") in {"+", "-"}:
            operand = cls._affine_expression(node.get("operand"))
            if operand is None:
                return None
            scale = 1.0 if node.get("op") == "+" else -1.0
            return (
                {name: scale * value for name, value in operand[0].items()},
                scale * operand[1],
            )
        if node_type != "binary":
            return None
        left = cls._affine_expression(node.get("left"))
        right = cls._affine_expression(node.get("right"))
        if left is None or right is None:
            return None
        operator = node.get("op")
        if operator in {"+", "-"}:
            scale = 1.0 if operator == "+" else -1.0
            coefficients = dict(left[0])
            for name, value in right[0].items():
                coefficients[name] = coefficients.get(name, 0.0) + scale * value
                if coefficients[name] == 0.0:
                    del coefficients[name]
            return coefficients, left[1] + scale * right[1]
        if operator == "*":
            if not left[0]:
                return (
                    {name: left[1] * value for name, value in right[0].items()},
                    left[1] * right[1],
                )
            if not right[0]:
                return (
                    {name: right[1] * value for name, value in left[0].items()},
                    right[1] * left[1],
                )
            return None
        if operator == "/" and not right[0] and right[1] != 0.0:
            return (
                {name: value / right[1] for name, value in left[0].items()},
                left[1] / right[1],
            )
        return None

    @staticmethod
    def _solve_affine_feasibility(
        predicates: list[tuple[dict[str, float], float, str]],
        continuous_parameters: tuple,
        fixed: Mapping[str, object],
    ) -> str:
        names = tuple(parameter.name for parameter in continuous_parameters)
        strict = any(operator in {"<", ">"} for _, _, operator in predicates)
        n_variables = len(names) + (1 if strict else 0)
        if not names:
            for coefficients, constant, operator in predicates:
                value = constant + sum(
                    coefficient * float(fixed[name])
                    for name, coefficient in coefficients.items()
                )
                satisfied = {
                    "<": value < 0.0,
                    "<=": value <= 0.0,
                    "==": value == 0.0,
                    ">=": value >= 0.0,
                    ">": value > 0.0,
                }[operator]
                if not satisfied:
                    return "infeasible"
            return "feasible"

        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        a_eq: list[list[float]] = []
        b_eq: list[float] = []
        for coefficients, raw_constant, operator in predicates:
            try:
                constant = raw_constant + sum(
                    coefficient * float(fixed[name])
                    for name, coefficient in coefficients.items()
                    if name in fixed
                )
            except (OverflowError, TypeError, ValueError):
                return "inconclusive"
            unknown = set(coefficients) - set(names) - set(fixed)
            if unknown:
                return "inconclusive"
            row = [coefficients.get(name, 0.0) for name in names]
            if strict:
                row.append(1.0 if operator in {"<", ">"} else 0.0)
            if operator in {"<", "<="}:
                a_ub.append(row)
                b_ub.append(-constant)
            elif operator in {">", ">="}:
                a_ub.append([-value for value in row])
                if strict and operator == ">":
                    a_ub[-1][-1] = 1.0
                b_ub.append(constant)
            else:
                a_eq.append(row)
                b_eq.append(-constant)
        objective = [0.0] * n_variables
        if strict:
            objective[-1] = -1.0
        bounds = [
            (float(parameter.bounds[0]), float(parameter.bounds[1]))
            for parameter in continuous_parameters
        ]
        if strict:
            bounds.append((None, None))
        try:
            result = linprog(
                objective,
                A_ub=a_ub or None,
                b_ub=b_ub or None,
                A_eq=a_eq or None,
                b_eq=b_eq or None,
                bounds=bounds,
                method="highs",
            )
        except Exception:
            return "inconclusive"
        if result.status == 2:
            return "infeasible"
        if result.status != 0 or result.x is None:
            return "inconclusive"
        if strict and result.x[-1] <= 0.0:
            return "infeasible"
        return "feasible"

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
