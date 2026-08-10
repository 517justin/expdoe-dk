"""Ordered experiment spaces spanning physical and numeric model frames."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from expdoe_dk.errors import EngineError, ErrorCode

from .constraints import (
    CategoricalCombinationConstraint,
    Constraint,
    LinearConstraint as DomainLinearConstraint,
    OutcomeConstraint,
    constraint_from_dict,
)
from .objective import Objective, normalize_objectives
from .parameter import Parameter, _json_scalars_equal

SPACE_SCHEMA_VERSION = "1.0"
ENGINE_VERSION = "0.5.0"


def _config_invalid(message: str) -> EngineError:
    return EngineError(ErrorCode.CONFIG_INVALID, message)


def _checkpoint_incompatible(message: str) -> EngineError:
    return EngineError(ErrorCode.CHECKPOINT_INCOMPATIBLE, message)


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise _config_invalid(f"{context} must be an object with string keys")
    return value


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], context: str
) -> None:
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing:
        raise _config_invalid(f"{context} is missing keys {sorted(missing)!r}")
    if unknown:
        raise _config_invalid(f"{context} has unknown keys {sorted(unknown)!r}")


def _require_sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _config_invalid(f"{context} must be an ordered sequence")
    return value


def _require_list(value: object, context: str) -> list[object]:
    if type(value) is not list:
        raise _config_invalid(f"{context} must be a JSON array")
    return value


def _require_string(value: object, context: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, (str, np.str_)) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise _config_invalid(f"{context} must be a {qualifier}string")
    return str(value)


def _normalize_number(value: object, context: str) -> int | float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise _config_invalid(f"{context} must be a finite non-boolean number")
    if isinstance(value, Integral):
        return int(value)
    numeric = float(value)
    if not math.isfinite(numeric):
        raise _config_invalid(f"{context} must be a finite non-boolean number")
    return numeric


def _normalize_json_scalar(value: object, context: str) -> object:
    if value is None:
        return value
    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Real):
        return _normalize_number(value, context)
    raise _config_invalid(f"{context} must be a finite JSON scalar")


def _json_compatible(value: object, context: str) -> object:
    """Return a detached structure accepted by ``json.dumps(allow_nan=False)``."""
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, (str, np.str_)):
                raise _config_invalid(f"{context} keys must be strings")
            normalized[str(key)] = _json_compatible(item, f"{context}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    return _normalize_json_scalar(value, context)


def _duplicate_names(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return sorted(duplicates)


def _is_legacy_linear_constraint(constraint: object) -> bool:
    return all(
        hasattr(constraint, member)
        for member in ("coeffs", "lower", "upper", "satisfied")
    )


def _parameter_payload(parameter: Parameter) -> dict[str, object]:
    transform = (
        "log" if parameter.transform == "log" or parameter.log_scale else "linear"
    )
    return {
        "name": parameter.name,
        "kind": parameter.kind,
        "bounds": list(parameter.bounds) if parameter.bounds is not None else None,
        "values": list(parameter.values) if parameter.values is not None else None,
        "step": parameter.step,
        "unit": parameter.unit,
        "transform": transform,
    }


def _parameter_from_payload(payload: object, index: int) -> Parameter:
    item = _require_mapping(payload, f"Space params[{index}]")
    expected = {"name", "kind", "bounds", "values", "step", "unit", "transform"}
    _require_exact_keys(item, expected, f"Space params[{index}]")
    name = _require_string(
        item["name"], f"Space params[{index}].name", nonempty=True
    )
    unit = _require_string(item["unit"], f"Space params[{index}].unit")
    kind = _require_string(item["kind"], f"Space params[{index}].kind")
    transform = _require_string(
        item["transform"], f"Space params[{index}].transform"
    )
    bounds = item["bounds"]
    if bounds is not None:
        pair = _require_list(bounds, f"Space params[{index}].bounds")
        if len(pair) != 2:
            raise _config_invalid(f"Space params[{index}].bounds must contain two values")
        bounds = tuple(
            _normalize_number(value, f"Space params[{index}].bounds[{bound_index}]")
            for bound_index, value in enumerate(pair)
        )
    values = item["values"]
    if values is not None:
        raw_values = _require_list(values, f"Space params[{index}].values")
        values = [
            _normalize_json_scalar(
                value, f"Space params[{index}].values[{value_index}]"
            )
            for value_index, value in enumerate(raw_values)
        ]
    step = item["step"]
    if step is not None:
        step = _normalize_number(step, f"Space params[{index}].step")
    try:
        return Parameter(
            name,
            bounds=bounds,  # type: ignore[arg-type]
            unit=unit,
            kind=kind,  # type: ignore[arg-type]
            step=step,
            values=values,  # type: ignore[arg-type]
            transform=transform,  # type: ignore[arg-type]
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise _config_invalid(f"Space params[{index}] is invalid: {error}") from error


def _objective_payload(objective: Objective) -> dict[str, object]:
    target = (
        list(objective.target)
        if isinstance(objective.target, tuple)
        else objective.target
    )
    return {
        "name": objective.name,
        "direction": objective.direction,
        "target": target,
        "unit": objective.unit,
        "priority": objective.priority,
    }


def _objective_from_payload(payload: object, index: int) -> Objective:
    item = _require_mapping(payload, f"Space objectives[{index}]")
    expected = {"name", "direction", "target", "unit", "priority"}
    _require_exact_keys(item, expected, f"Space objectives[{index}]")
    name = _require_string(
        item["name"], f"Space objectives[{index}].name", nonempty=True
    )
    direction = _require_string(
        item["direction"], f"Space objectives[{index}].direction"
    )
    unit = _require_string(item["unit"], f"Space objectives[{index}].unit")
    priority_value = item["priority"]
    if (
        isinstance(priority_value, (bool, np.bool_))
        or not isinstance(priority_value, Integral)
        or int(priority_value) < 0
    ):
        raise _config_invalid(
            f"Space objectives[{index}].priority must be a non-negative integer"
        )
    priority = int(priority_value)
    target = item["target"]
    if type(target) is list:
        if len(target) != 2:
            raise _config_invalid(
                f"Space objectives[{index}].target range must contain two values"
            )
        target = tuple(
            _normalize_number(
                value, f"Space objectives[{index}].target[{target_index}]"
            )
            for target_index, value in enumerate(target)
        )
    elif target is not None:
        target = _normalize_number(target, f"Space objectives[{index}].target")
    try:
        return Objective(
            name,
            direction,  # type: ignore[arg-type]
            target=target,  # type: ignore[arg-type]
            unit=unit,
            priority=priority,
        )
    except EngineError:
        raise
    except (TypeError, ValueError) as error:
        raise _config_invalid(
            f"Space objectives[{index}] is invalid: {error}"
        ) from error


class Space:
    """An ordered set of parameters, objectives, and declarative constraints."""

    def __init__(
        self,
        params: Sequence[Parameter],
        constraints: Sequence[Constraint | object] | None = None,
        objectives: str | Objective | Sequence[str | Objective] = "y",
        maximize: bool | Sequence[bool] = True,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
    ) -> None:
        if isinstance(params, (str, bytes, bytearray)) or not isinstance(
            params, Sequence
        ):
            raise _config_invalid("Space params must be an ordered sequence")
        if not params:
            raise ValueError("Space: at least one parameter is required.")
        if not all(isinstance(parameter, Parameter) for parameter in params):
            raise _config_invalid("Space params must contain Parameter instances")

        parameter_names = [parameter.name for parameter in params]
        duplicates = _duplicate_names(parameter_names)
        if duplicates:
            raise ValueError(
                f"Space: parameter names must be unique; duplicate names {duplicates}."
            )

        objective_specs = tuple(
            Objective(
                str(objective.name)
                if isinstance(objective.name, np.str_)
                else objective.name,
                objective.direction,
                target=objective.target,
                unit=(
                    str(objective.unit)
                    if isinstance(objective.unit, np.str_)
                    else objective.unit
                ),
                priority=objective.priority,
            )
            if isinstance(objective.name, np.str_)
            or isinstance(objective.unit, np.str_)
            else objective
            for objective in normalize_objectives(objectives, maximize)
        )
        if not objective_specs:
            raise _config_invalid("Space requires at least one objective")
        objective_names = [objective.name for objective in objective_specs]
        duplicates = _duplicate_names(objective_names)
        if duplicates:
            raise ValueError(
                f"Space: objective names must be unique; duplicate names {duplicates}."
            )

        self.params = list(params)
        self.constraints = list(constraints or ())
        self.objective_specs = objective_specs
        self.outcome_constraints = list(outcome_constraints or ())

        constraint_names = [
            constraint.name
            for constraint in (*self.constraints, *self.outcome_constraints)
            if isinstance(getattr(constraint, "name", None), str)
            and constraint.name
        ]
        duplicates = _duplicate_names(constraint_names)
        if duplicates:
            raise ValueError(
                f"Space: constraint names must be unique; duplicate names {duplicates}."
            )

        self._validate_parameter_constraints(set(parameter_names), set(objective_names))
        self._validate_outcome_constraints(set(parameter_names), set(objective_names))

    @property
    def objectives(self) -> list[str]:
        return [objective.name for objective in self.objective_specs]

    @property
    def maximize(self) -> list[bool]:
        return [objective.direction != "minimize" for objective in self.objective_specs]

    @property
    def n_dims(self) -> int:
        return len(self.params)

    @property
    def n_objectives(self) -> int:
        return len(self.objective_specs)

    @property
    def param_names(self) -> list[str]:
        return [parameter.name for parameter in self.params]

    def param_by_name(self, name: str) -> Parameter:
        for parameter in self.params:
            if parameter.name == name:
                return parameter
        raise KeyError(f"Unknown parameter: {name}. Available: {self.param_names}.")

    @property
    def lower(self) -> Tensor:
        return torch.tensor(
            [self._physical_bounds(parameter)[0] for parameter in self.params],
            dtype=torch.float64,
        )

    @property
    def upper(self) -> Tensor:
        return torch.tensor(
            [self._physical_bounds(parameter)[1] for parameter in self.params],
            dtype=torch.float64,
        )

    @property
    def bounds_tensor(self) -> Tensor:
        return torch.stack([self.lower, self.upper], dim=0)

    @property
    def model_bounds_tensor(self) -> Tensor:
        return torch.tensor(
            [parameter.model_bounds for parameter in self.params], dtype=torch.float64
        ).transpose(0, 1)

    def physical_to_model(self, frame: pd.DataFrame) -> Tensor:
        """Encode a physical DataFrame into ordered numeric model coordinates."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("physical_to_model requires a pandas DataFrame")
        missing = [name for name in self.param_names if name not in frame.columns]
        if missing:
            raise ValueError(f"Physical frame is missing parameter columns {missing}.")
        columns = [
            parameter.encode(frame[parameter.name].tolist())
            for parameter in self.params
        ]
        if not columns:
            return torch.empty((len(frame), 0), dtype=torch.float64)
        return torch.tensor(np.column_stack(columns), dtype=torch.float64)

    def model_to_physical(self, tensor: Tensor | np.ndarray) -> pd.DataFrame:
        """Decode ordered numeric model coordinates into a physical DataFrame."""
        model = torch.as_tensor(tensor, dtype=torch.float64)
        if model.ndim == 1:
            model = model.unsqueeze(0)
        if model.ndim != 2 or model.shape[1] != self.n_dims:
            raise ValueError(
                f"Model coordinates must have shape (n, {self.n_dims}), got {tuple(model.shape)}."
            )
        values = model.detach().cpu().numpy()
        data = {}
        for index, parameter in enumerate(self.params):
            decoded = parameter.decode(values[:, index].tolist())
            data[parameter.name] = (
                pd.Series(decoded, dtype=object)
                if parameter.kind in {"categorical", "ordinal"}
                else decoded
            )
        return pd.DataFrame(data, columns=self.param_names)

    def physical_to_unit(self, X_phys: Tensor) -> Tensor:
        """Legacy tensor alias for numerical physical-to-model conversion."""
        physical = torch.as_tensor(X_phys)
        one_row = physical.ndim == 1
        if one_row:
            physical = physical.unsqueeze(0)
        frame = pd.DataFrame(
            physical.detach().cpu().numpy(), columns=self.param_names
        )
        model = self.physical_to_model(frame)
        return model.squeeze(0) if one_row else model

    def unit_to_physical(self, X_unit: Tensor) -> Tensor:
        """Legacy tensor alias for numerical model-to-physical conversion."""
        model = torch.as_tensor(X_unit, dtype=torch.float64)
        one_row = model.ndim == 1
        frame = self.model_to_physical(model)
        if any(
            parameter.kind in {"categorical", "ordinal"} for parameter in self.params
        ):
            raise TypeError(
                "unit_to_physical tensor output is unavailable for string-valued "
                "spaces; use model_to_physical"
            )
        physical = torch.tensor(frame.to_numpy(dtype=float), dtype=torch.float64)
        return physical.squeeze(0) if one_row else physical

    def feasibility_mask(self, X_phys: Tensor | pd.DataFrame) -> Tensor:
        """Return parameter-domain and hard-constraint physical feasibility."""
        if isinstance(X_phys, pd.DataFrame):
            self._validate_required_columns(X_phys)
            rows = X_phys[self.param_names].to_dict(orient="records")
        else:
            frame = self.to_dataframe(X_phys)
            rows = frame.to_dict(orient="records")
        mask = [self._row_within_parameter_domains(row) for row in rows]
        hard_constraints = [
            constraint
            for constraint in self.constraints
            if getattr(constraint, "hard", True)
        ]
        for index, row in enumerate(rows):
            if mask[index] and hard_constraints:
                mask[index] = all(
                    constraint.satisfied(row) for constraint in hard_constraints
                )
        return torch.tensor(mask, dtype=torch.bool)

    def _validate_required_columns(self, frame: pd.DataFrame) -> None:
        missing = [name for name in self.param_names if name not in frame.columns]
        duplicate = [
            name for name in self.param_names if list(frame.columns).count(name) > 1
        ]
        if missing or duplicate:
            details = []
            if missing:
                details.append(f"missing required columns {missing}")
            if duplicate:
                details.append(f"duplicate required columns {duplicate}")
            raise _config_invalid("Physical frame has " + " and ".join(details))

    def _row_within_parameter_domains(self, row: Mapping[str, object]) -> bool:
        for parameter in self.params:
            value = row[parameter.name]
            if parameter.kind in {"continuous", "integer", "discrete"} and (
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            ):
                return False
            try:
                parameter.encode([value])
            except (TypeError, ValueError, OverflowError):
                return False
        return True

    def with_constraints(self, *additional: Constraint | object) -> "Space":
        """Return a validated copy with additional parameter constraints."""
        return Space(
            params=self.params,
            constraints=(*self.constraints, *additional),
            objectives=self.objective_specs,
            outcome_constraints=self.outcome_constraints,
        )

    def to_dataframe(self, X_phys: Tensor) -> pd.DataFrame:
        values = torch.as_tensor(X_phys, dtype=torch.float64)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.n_dims:
            raise ValueError(
                f"Physical coordinates must have shape (n, {self.n_dims}), "
                f"got {tuple(values.shape)}."
            )
        if any(
            parameter.kind in {"categorical", "ordinal"} for parameter in self.params
        ):
            raise TypeError(
                "to_dataframe tensor input is unavailable for string-valued spaces; "
                "use model_to_physical"
            )
        return pd.DataFrame(
            values.detach().cpu().numpy(), columns=self.param_names
        )

    def to_tensor(self, frame: pd.DataFrame) -> Tensor:
        try:
            values = frame[self.param_names].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "to_tensor is unavailable for string-valued spaces; use physical_to_model"
            ) from error
        return torch.tensor(values, dtype=torch.float64)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": SPACE_SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "params": [_parameter_payload(parameter) for parameter in self.params],
            "constraints": [self._constraint_payload(item) for item in self.constraints],
            "objectives": [
                _objective_payload(objective) for objective in self.objective_specs
            ],
            "outcome_constraints": [
                constraint.to_dict() for constraint in self.outcome_constraints
            ],
        }
        normalized = _json_compatible(payload, "Space payload")
        assert isinstance(normalized, dict)
        return normalized

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Space":
        state = _require_mapping(payload, "Space payload")
        if "schema_version" not in state:
            return cls._from_v04_dict(state)

        expected = {
            "schema_version",
            "engine_version",
            "params",
            "constraints",
            "objectives",
            "outcome_constraints",
        }
        _require_exact_keys(state, expected, "Space payload")
        if state["schema_version"] != SPACE_SCHEMA_VERSION:
            raise _checkpoint_incompatible(
                f"Unsupported Space schema version {state['schema_version']!r}"
            )
        if state["engine_version"] != ENGINE_VERSION:
            raise _checkpoint_incompatible(
                f"Unsupported Space engine version {state['engine_version']!r}"
            )

        raw_params = _require_list(state["params"], "Space params")
        params = [
            _parameter_from_payload(item, index)
            for index, item in enumerate(raw_params)
        ]
        raw_objectives = _require_list(state["objectives"], "Space objectives")
        objectives = [
            _objective_from_payload(item, index)
            for index, item in enumerate(raw_objectives)
        ]
        parameter_names = {parameter.name for parameter in params}
        objective_names = {objective.name for objective in objectives}

        constraints: list[Constraint | object] = []
        raw_constraints = _require_list(state["constraints"], "Space constraints")
        for index, raw in enumerate(raw_constraints):
            item = _require_mapping(raw, f"Space constraints[{index}]")
            normalized_item = _json_compatible(
                item, f"Space constraints[{index}]"
            )
            assert isinstance(normalized_item, dict)
            item = normalized_item
            if item.get("kind") == "legacy_linear":
                constraints.append(cls._legacy_constraint_from_payload(item, index))
            else:
                constraints.append(
                    constraint_from_dict(item, allowed_names=parameter_names)
                )

        outcome_constraints: list[OutcomeConstraint] = []
        raw_outcomes = _require_list(
            state["outcome_constraints"], "Space outcome_constraints"
        )
        for index, raw in enumerate(raw_outcomes):
            item = _require_mapping(raw, f"Space outcome_constraints[{index}]")
            normalized_item = _json_compatible(
                item, f"Space outcome_constraints[{index}]"
            )
            assert isinstance(normalized_item, dict)
            item = normalized_item
            constraint = constraint_from_dict(
                item,
                allowed_names=parameter_names,
                allowed_objective_names=objective_names,
            )
            if not isinstance(constraint, OutcomeConstraint):
                raise _config_invalid(
                    f"Space outcome_constraints[{index}] must be an OutcomeConstraint"
                )
            outcome_constraints.append(constraint)
        try:
            return cls(
                params=params,
                constraints=constraints,
                objectives=objectives,
                outcome_constraints=outcome_constraints,
            )
        except EngineError:
            raise
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            raise _config_invalid(f"Current Space payload is invalid: {error}") from error

    @classmethod
    def _from_v04_dict(cls, payload: Mapping[str, object]) -> "Space":
        raw_params = _require_sequence(payload.get("params"), "v0.4 Space params")
        params: list[Parameter] = []
        for index, raw in enumerate(raw_params):
            item = _require_mapping(raw, f"v0.4 Space params[{index}]")
            bounds = item.get("bounds")
            pair = _require_sequence(bounds, f"v0.4 Space params[{index}].bounds")
            params.append(
                Parameter(
                    item.get("name"),  # type: ignore[arg-type]
                    bounds=tuple(pair),  # type: ignore[arg-type]
                    unit=item.get("unit", ""),  # type: ignore[arg-type]
                    kind=item.get("kind", "continuous"),  # type: ignore[arg-type]
                    step=item.get("step"),  # type: ignore[arg-type]
                    log_scale=item.get("log_scale", False),  # type: ignore[arg-type]
                )
            )
        constraints = []
        raw_constraints = _require_sequence(
            payload.get("constraints", []), "v0.4 Space constraints"
        )
        for index, raw in enumerate(raw_constraints):
            item = _require_mapping(raw, f"v0.4 Space constraints[{index}]")
            constraints.append(cls._v04_constraint_from_payload(item, index))
        return cls(
            params=params,
            constraints=constraints,
            objectives=payload.get("objectives", "y"),  # type: ignore[arg-type]
            maximize=payload.get("maximize", True),  # type: ignore[arg-type]
        )

    @staticmethod
    def _v04_constraint_from_payload(
        payload: Mapping[str, object], index: int
    ) -> object:
        from expdoe_dk.space import LinearConstraint

        if "coeffs" not in payload:
            raise _config_invalid(
                f"v0.4 Space constraints[{index}] requires a coeffs mapping"
            )
        return LinearConstraint(
            coeffs=dict(_require_mapping(payload["coeffs"], "constraint coeffs")),
            lower=(
                -math.inf if payload.get("lower") is None else payload.get("lower")
            ),
            upper=(
                math.inf if payload.get("upper") is None else payload.get("upper")
            ),
            name=payload.get("name", ""),
        )

    @staticmethod
    def _legacy_constraint_from_payload(
        payload: Mapping[str, object], index: int
    ) -> object:
        from expdoe_dk.space import LinearConstraint

        expected = {"kind", "name", "coeffs", "lower", "upper"}
        _require_exact_keys(payload, expected, f"Space constraints[{index}]")
        _require_string(payload["kind"], f"Space constraints[{index}].kind")
        name = _require_string(
            payload["name"], f"Space constraints[{index}].name"
        )
        raw_coefficients = _require_mapping(
            payload["coeffs"], f"Space constraints[{index}].coeffs"
        )
        if not raw_coefficients:
            raise _config_invalid(
                f"Space constraints[{index}].coeffs must be non-empty"
            )
        coefficients = {
            _require_string(
                factor,
                f"Space constraints[{index}].coeffs factor",
                nonempty=True,
            ): _normalize_number(
                coefficient,
                f"Space constraints[{index}].coeffs[{factor!r}]",
            )
            for factor, coefficient in raw_coefficients.items()
        }
        lower = payload["lower"]
        upper = payload["upper"]
        normalized_lower = (
            -math.inf
            if lower is None
            else _normalize_number(lower, f"Space constraints[{index}].lower")
        )
        normalized_upper = (
            math.inf
            if upper is None
            else _normalize_number(upper, f"Space constraints[{index}].upper")
        )
        if normalized_lower > normalized_upper:
            raise _config_invalid(
                f"Space constraints[{index}] lower must not exceed upper"
            )
        return LinearConstraint(
            coeffs=coefficients,
            lower=normalized_lower,
            upper=normalized_upper,
            name=name,
        )

    @staticmethod
    def _constraint_payload(constraint: object) -> dict[str, object]:
        if _is_legacy_linear_constraint(constraint):
            lower = constraint.lower  # type: ignore[attr-defined]
            upper = constraint.upper  # type: ignore[attr-defined]
            return {
                "kind": "legacy_linear",
                "name": constraint.name,  # type: ignore[attr-defined]
                "coeffs": dict(constraint.coeffs),  # type: ignore[attr-defined]
                "lower": lower if math.isfinite(lower) else None,
                "upper": upper if math.isfinite(upper) else None,
            }
        return constraint.to_dict()  # type: ignore[no-any-return, attr-defined]

    def _validate_parameter_constraints(
        self, parameter_names: set[str], objective_names: set[str]
    ) -> None:
        for index, constraint in enumerate(self.constraints):
            if isinstance(constraint, OutcomeConstraint):
                raise _config_invalid(
                    "OutcomeConstraint belongs in Space outcome_constraints"
                )
            if _is_legacy_linear_constraint(constraint):
                unknown = set(constraint.coeffs) - parameter_names  # type: ignore[attr-defined]
                if unknown:
                    raise ValueError(
                        f"LinearConstraint references unknown params {sorted(unknown)}."
                    )
                self._validate_linear_factor_kinds(constraint.coeffs)  # type: ignore[attr-defined]
                continue
            if not isinstance(constraint, Constraint):
                raise _config_invalid(
                    f"Space constraints[{index}] does not implement Constraint"
                )
            validated = constraint_from_dict(
                constraint.to_dict(),
                allowed_names=parameter_names,
                allowed_objective_names=objective_names,
            )
            if isinstance(validated, CategoricalCombinationConstraint):
                self._validate_categorical_assignments(validated)
            elif isinstance(validated, DomainLinearConstraint):
                self._validate_linear_factor_kinds(validated.coefficients)

    def _validate_linear_factor_kinds(
        self, coefficients: Mapping[str, object]
    ) -> None:
        nonnumeric = [
            factor
            for factor in coefficients
            if self.param_by_name(factor).kind
            not in {"continuous", "integer", "discrete"}
        ]
        if nonnumeric:
            raise _config_invalid(
                "LinearConstraint coefficients require numeric physical factors; "
                f"got {sorted(nonnumeric)}"
            )

    def _validate_outcome_constraints(
        self, parameter_names: set[str], objective_names: set[str]
    ) -> None:
        for index, constraint in enumerate(self.outcome_constraints):
            if not isinstance(constraint, OutcomeConstraint):
                raise _config_invalid(
                    f"Space outcome_constraints[{index}] must be an OutcomeConstraint"
                )
            constraint_from_dict(
                constraint.to_dict(),
                allowed_names=parameter_names,
                allowed_objective_names=objective_names,
            )

    def _validate_categorical_assignments(
        self, constraint: CategoricalCombinationConstraint
    ) -> None:
        patterns = constraint.allowed or constraint.forbidden or ()
        for assignment in patterns:
            for factor, value in assignment.items():
                parameter = self.param_by_name(factor)
                if parameter.kind not in {"categorical", "ordinal"}:
                    raise _config_invalid(
                        f"Categorical constraint {constraint.name!r} references "
                        f"non-categorical parameter {factor!r}"
                    )
                if not any(
                    _json_scalars_equal(value, level)
                    for level in parameter.values or ()
                ):
                    raise _config_invalid(
                        f"Categorical constraint {constraint.name!r} value {value!r} "
                        f"is not declared for parameter {factor!r}"
                    )

    @staticmethod
    def _physical_bounds(parameter: Parameter) -> tuple[float, float]:
        if parameter.bounds is not None:
            return float(parameter.bounds[0]), float(parameter.bounds[1])
        if parameter.kind == "discrete" and parameter.values is not None:
            numeric = [float(value) for value in parameter.values]
            return min(numeric), max(numeric)
        raise TypeError(
            f"Physical tensor bounds are unavailable for {parameter.kind} parameter "
            f"{parameter.name!r}."
        )

    def __repr__(self) -> str:
        return (
            f"Space(n_dims={self.n_dims}, objectives={self.objectives}, "
            f"maximize={self.maximize})"
        )


__all__ = ["ENGINE_VERSION", "SPACE_SCHEMA_VERSION", "Space"]
