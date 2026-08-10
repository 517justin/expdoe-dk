"""Tagged experiment parameters and physical/model frame transforms."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Sequence

import numpy as np

ParameterKind = Literal["continuous", "integer", "discrete", "categorical", "ordinal"]
ParameterTransform = Literal["linear", "log"]

_NUMERIC_KINDS = {"continuous", "integer", "discrete"}
_VALUE_KINDS = {"categorical", "ordinal"}
_KINDS = _NUMERIC_KINDS | _VALUE_KINDS
_TRANSFORMS = {"linear", "log"}


@dataclass(frozen=True)
class Parameter:
    """A serializable experimental factor in physical units."""

    name: str
    bounds: tuple[float, float] | None = None
    unit: str = ""
    kind: ParameterKind = "continuous"
    step: float | int | None = None
    values: tuple[object, ...] | list[object] | None = None
    transform: ParameterTransform = "linear"
    log_scale: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"Parameter {self.name}: unknown kind {self.kind!r}.")
        if self.transform not in _TRANSFORMS:
            raise ValueError(
                f"Parameter {self.name}: unknown transform {self.transform!r}."
            )
        if not isinstance(self.log_scale, bool):
            raise ValueError(f"Parameter {self.name}: log_scale must be a boolean.")

        if self.values is not None:
            try:
                normalized_values = tuple(self.values)
            except TypeError as error:
                raise ValueError(
                    f"Parameter {self.name}: values must be a sequence."
                ) from error
            object.__setattr__(self, "values", normalized_values)

        if self.kind == "continuous":
            self._validate_continuous()
        elif self.kind == "integer":
            self._validate_integer()
        elif self.kind == "discrete":
            self._validate_discrete()
        elif self.kind == "categorical":
            self._validate_categorical()
        else:
            self._validate_ordinal()

        if self._uses_log_transform:
            if self.kind not in _NUMERIC_KINDS:
                raise ValueError(
                    f"Parameter {self.name}: log transforms require numeric values."
                )
            if any(value <= 0 for value in self._physical_limits_and_levels()):
                raise ValueError(
                    f"Parameter {self.name}: log transforms require positive physical values."
                )

    @property
    def model_bounds(self) -> tuple[float, float]:
        """Bounds used by optimizers in the parameter's model frame."""
        if self.kind in _VALUE_KINDS:
            return 0.0, float(len(self.values or ()) - 1)
        return 0.0, 1.0

    @property
    def cardinality(self) -> int | None:
        """Number of physical levels, or ``None`` for continuous factors."""
        if self.kind == "continuous":
            return None
        if self.kind in {"integer", "discrete"}:
            return len(self.numeric_levels)
        return len(self.values or ())

    @property
    def numeric_levels(self) -> tuple[int | float, ...]:
        """Ordered physical levels for integer and discrete factors."""
        if self.kind not in {"integer", "discrete"}:
            raise AttributeError(f"Parameter {self.name} is not integer or discrete.")
        if self.values is not None:
            return tuple(sorted(float(value) for value in self.values))

        low, high = self._validated_bounds()
        step = 1 if self.kind == "integer" and self.step is None else self.step
        assert step is not None  # Validated for bounded discrete parameters.
        count = int(math.floor((high - low) / float(step) + 1e-12)) + 1
        if self.kind == "integer":
            return tuple(int(low) + index * int(step) for index in range(count))
        return tuple(float(low) + index * float(step) for index in range(count))

    def encode(self, values: Sequence[object]) -> list[float]:
        """Map physical values into the parameter's model frame."""
        if self.kind in _VALUE_KINDS:
            levels = list(self.values or ())
            try:
                return [float(levels.index(value)) for value in values]
            except ValueError as error:
                raise ValueError(
                    f"Parameter {self.name}: value is not a declared level."
                ) from error

        numeric = self._validated_physical_values(values)
        low, high = self._physical_limits()
        if self._uses_log_transform:
            encoded = np.log(numeric / low) / math.log(high / low)
        else:
            encoded = (numeric - low) / (high - low)
        return encoded.tolist()

    def decode(self, encoded: Sequence[float]) -> list[object]:
        """Map model values to physical values, snapping finite factors."""
        model = np.asarray(encoded, dtype=np.float64)
        if not np.all(np.isfinite(model)):
            raise ValueError(f"Parameter {self.name}: model values must be finite.")
        model_low, model_high = self.model_bounds
        if np.any(model < model_low) or np.any(model > model_high):
            raise ValueError(
                f"Parameter {self.name}: model values must be within {self.model_bounds}."
            )

        if self.kind in _VALUE_KINDS:
            levels = list(self.values or ())
            return [levels[int(round(value))] for value in model.tolist()]

        low, high = self._physical_limits()
        if self._uses_log_transform:
            numeric = low * np.power(high / low, model)
        else:
            numeric = low + model * (high - low)
        physical = np.asarray(self.snap(numeric))
        self._validated_physical_values(physical.tolist())
        return physical.tolist()

    def to_model(self, values: Sequence[object]) -> list[float]:
        """Alias for :meth:`encode` using the design terminology."""
        return self.encode(values)

    def to_physical(self, encoded: Sequence[float]) -> list[object]:
        """Alias for :meth:`decode` using the design terminology."""
        return self.decode(encoded)

    def snap(self, values: np.ndarray) -> np.ndarray:
        """Snap numeric values to the nearest declared finite level."""
        numeric = np.asarray(values, dtype=np.float64)
        if self.kind == "continuous":
            return numeric
        if self.kind not in {"integer", "discrete"}:
            raise TypeError(f"Parameter {self.name}: {self.kind} values are not numeric.")
        levels = np.asarray(self.numeric_levels, dtype=np.float64)
        indices = np.abs(numeric[..., None] - levels).argmin(axis=-1)
        return levels[indices]

    def _validate_continuous(self) -> None:
        self._validated_bounds()
        if self.step is not None:
            raise ValueError(
                f"Parameter {self.name}: step must be None for continuous parameters."
            )
        if self.values is not None:
            raise ValueError(
                f"Parameter {self.name}: continuous parameters do not accept values."
            )

    def _validate_integer(self) -> None:
        low, high = self._validated_bounds()
        if not (low.is_integer() and high.is_integer()):
            raise ValueError(f"Parameter {self.name}: integer bounds must be integral.")
        if self.values is not None:
            raise ValueError(
                f"Parameter {self.name}: integer parameters do not accept values."
            )
        if self.step is not None:
            step = self._validated_step()
            if not step.is_integer():
                raise ValueError(f"Parameter {self.name}: integer step must be integral.")

    def _validate_discrete(self) -> None:
        if self.values is not None:
            if self.bounds is not None or self.step is not None:
                raise ValueError(
                    f"Parameter {self.name}: discrete parameters use either values "
                    "or bounds and step."
                )
            self._validate_value_count()
            numeric = self._validate_numeric_levels(self.values)
            self._validate_unique(numeric)
            return

        self._validated_bounds()
        if self.step is None:
            raise ValueError(
                f"Parameter {self.name}: bounded discrete parameters require step > 0."
            )
        self._validated_step()

    def _validate_categorical(self) -> None:
        self._validate_value_only_configuration()
        if not all(isinstance(value, str) for value in self.values or ()):
            raise ValueError(
                f"Parameter {self.name}: categorical values must all be strings."
            )
        self._validate_unique(self.values or ())

    def _validate_ordinal(self) -> None:
        self._validate_value_only_configuration()
        scalar_types = (str, int, float, bool, type(None))
        if not all(isinstance(value, scalar_types) for value in self.values or ()):
            raise ValueError(
                f"Parameter {self.name}: ordinal values must be scalar values."
            )
        if any(
            isinstance(value, Real) and not math.isfinite(float(value))
            for value in self.values or ()
        ):
            raise ValueError(f"Parameter {self.name}: ordinal values must be finite.")
        self._validate_unique(self.values or ())

    def _validate_value_only_configuration(self) -> None:
        if self.bounds is not None or self.step is not None:
            raise ValueError(
                f"Parameter {self.name}: {self.kind} parameters use values, not bounds or step."
            )
        self._validate_value_count()

    def _validate_value_count(self) -> None:
        if self.values is None or len(self.values) < 2:
            raise ValueError(
                f"Parameter {self.name}: {self.kind} parameters require at least two values."
            )

    def _validated_bounds(self) -> tuple[float, float]:
        if self.bounds is None:
            raise ValueError(f"Parameter {self.name}: {self.kind} requires bounds.")
        if len(self.bounds) != 2:
            raise ValueError(f"Parameter {self.name}: bounds must be a (low, high) pair.")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in self.bounds):
            raise ValueError(f"Parameter {self.name}: bounds must be numeric.")
        low, high = (float(self.bounds[0]), float(self.bounds[1]))
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(f"Parameter {self.name}: bounds must be finite.")
        if low >= high:
            raise ValueError(
                f"Parameter {self.name}: bounds={self.bounds} invalid (low must be < high)."
            )
        return low, high

    def _validated_step(self) -> float:
        if isinstance(self.step, bool) or not isinstance(self.step, Real):
            raise ValueError(f"Parameter {self.name}: step must be numeric.")
        step = float(self.step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError(f"Parameter {self.name}: step must be finite and > 0.")
        low, high = self._validated_bounds()
        if step > high - low:
            raise ValueError(
                f"Parameter {self.name}: step={self.step} is larger than the parameter range."
            )
        return step

    def _validate_numeric_levels(self, values: Sequence[object]) -> tuple[float, ...]:
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError(
                f"Parameter {self.name}: discrete values must all be numeric."
            )
        numeric = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"Parameter {self.name}: discrete values must be finite.")
        return numeric

    def _validate_unique(self, values: Sequence[object]) -> None:
        if len(set(values)) != len(values):
            raise ValueError(f"Parameter {self.name}: values must be unique.")

    @property
    def _uses_log_transform(self) -> bool:
        return self.transform == "log" or self.log_scale

    def _physical_limits_and_levels(self) -> tuple[float, ...]:
        if self.values is not None:
            return tuple(float(value) for value in self.values)
        return self._validated_bounds()

    def _physical_limits(self) -> tuple[float, float]:
        if self.values is not None:
            levels = self.numeric_levels
            return float(levels[0]), float(levels[-1])
        return self._validated_bounds()

    def _validated_physical_values(self, values: Sequence[object]) -> np.ndarray:
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Parameter {self.name}: physical values must be numeric.") from error
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"Parameter {self.name}: physical values must be finite.")
        low, high = self._physical_limits()
        if np.any(numeric < low) or np.any(numeric > high):
            raise ValueError(
                f"Parameter {self.name}: physical values must be within {(low, high)}."
            )
        if self.kind in {"integer", "discrete"}:
            levels = np.asarray(self.numeric_levels, dtype=np.float64)
            matches = np.isclose(numeric[..., None], levels, rtol=1e-9, atol=1e-12)
            if not np.all(matches.any(axis=-1)):
                raise ValueError(
                    f"Parameter {self.name}: physical values must be declared levels."
                )
        return numeric


__all__ = ["Parameter", "ParameterKind", "ParameterTransform"]
