"""Validated immutable boundaries for observed and pending experiment rows."""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


_OBSERVATION_STATUSES = frozenset({"success", "warning", "failed"})


def _clone_frame(value: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return an index-reset frame detached through nested object cells."""
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame.")
    if not value.columns.is_unique:
        raise ValueError(f"{name} must not have duplicate columns.")
    cloned = value.copy(deep=True).reset_index(drop=True)
    for column in cloned.columns:
        if cloned[column].dtype == object:
            cloned[column] = pd.Series(
                [deepcopy(value) for value in cloned[column]], index=cloned.index, dtype=object,
            )
    return cloned


def _copy_series(value: pd.Series, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise ValueError(f"{name} must be a pandas Series.")
    return value.copy(deep=True).reset_index(drop=True)


def _normalize_ids(
    ids: object, count: int, batch_name: str, *, optional: bool = False
) -> tuple[str, ...]:
    if isinstance(ids, (str, bytes, bytearray)) or not isinstance(ids, Sequence):
        raise ValueError(f"{batch_name} ids must be a sequence of strings.")
    normalized = tuple(ids)
    if optional and not normalized:
        return ()
    if len(normalized) != count:
        raise ValueError(f"{batch_name} ids count must equal the X row count.")
    if any(not isinstance(identifier, str) or not identifier for identifier in normalized):
        raise ValueError(f"{batch_name} ids must be non-empty strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{batch_name} ids must be unique.")
    return normalized


def _validate_yvar(Yvar: pd.DataFrame, Y: pd.DataFrame) -> None:
    if len(Yvar) != len(Y):
        raise ValueError("Yvar row count must equal Y row count.")
    if not Yvar.columns.equals(Y.columns):
        raise ValueError("Yvar columns must exactly match Y columns.")
    if any(
        not is_numeric_dtype(column)
        or is_bool_dtype(column)
        or is_complex_dtype(column)
        for _, column in Yvar.items()
    ):
        raise ValueError("Yvar must be numeric, finite, and non-negative.")
    values = Yvar.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Yvar must be numeric, finite, and non-negative.")


def _align_multioutput_yvar(Yvar: pd.DataFrame, Y: pd.DataFrame, status: pd.Series) -> pd.DataFrame:
    if len(Yvar) != len(Y):
        raise ValueError("Yvar row count must equal Y row count.")
    if set(Yvar.columns) != set(Y.columns):
        raise ValueError("Yvar columns must exactly match Y columns.")
    aligned = Yvar.loc[:, Y.columns]
    for row_status, row in zip(status, aligned.itertuples(index=False, name=None), strict=True):
        for value in row:
            missing = (value is None or value is pd.NA or
                       isinstance(value, (float, np.floating)) and np.isnan(value))
            if missing and row_status != "success":
                continue
            valid = not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
            if valid:
                try:
                    valid = math.isfinite(float(value)) and value >= 0
                except (OverflowError, TypeError, ValueError):
                    valid = False
            if not valid:
                raise ValueError("Yvar must be finite real non-boolean and non-negative; "
                                 "only unsuccessful rows may have missing variances.")
    return aligned


def _validate_successful_objectives(Y: pd.DataFrame, status: pd.Series) -> None:
    successful = status.eq("success").to_numpy()
    if not successful.any():
        return
    values = Y.iloc[successful].to_numpy(dtype=object).reshape(-1).tolist()
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(
                "successful rows must have finite real non-boolean objective values."
            )
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(
                "successful rows must have finite real non-boolean objective values."
            )


@dataclass(frozen=True, init=False)
class ObservationBatch:
    """Aligned observed conditions and outcomes at the model-data boundary."""

    _X: pd.DataFrame = field(repr=False)
    _Y: pd.DataFrame = field(repr=False)
    _Yvar: pd.DataFrame | None = field(repr=False)
    _status: pd.Series | None = field(repr=False)
    _ids: tuple[str, ...]

    def __init__(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        Yvar: pd.DataFrame | None = None,
        status: pd.Series | None = None,
        ids: Sequence[str] = (),
    ) -> None:
        copied_X = _clone_frame(X, "X")
        copied_Y = _clone_frame(Y, "Y")
        if len(copied_X) != len(copied_Y):
            raise ValueError("X and Y row counts must be equal.")

        copied_Yvar = None
        if Yvar is not None:
            copied_Yvar = _clone_frame(Yvar, "Yvar")
            if len(copied_Y.columns) <= 1:
                _validate_yvar(copied_Yvar, copied_Y)

        copied_status = None
        effective_status = pd.Series(["success"] * len(copied_X), dtype="object")
        if status is not None:
            copied_status = _copy_series(status, "status")
            if len(copied_status) != len(copied_X):
                raise ValueError("status count must equal the X row count.")
            if any(
                not isinstance(item, str) or item not in _OBSERVATION_STATUSES
                for item in copied_status
            ):
                raise ValueError(
                    "status values must be one of: success, warning, failed."
                )
            effective_status = copied_status

        if copied_Yvar is not None and len(copied_Y.columns) > 1:
            copied_Yvar = _align_multioutput_yvar(copied_Yvar, copied_Y, effective_status)

        normalized_ids = _normalize_ids(
            ids, len(copied_X), "ObservationBatch", optional=True
        )
        _validate_successful_objectives(copied_Y, effective_status)

        object.__setattr__(self, "_X", copied_X)
        object.__setattr__(self, "_Y", copied_Y)
        object.__setattr__(self, "_Yvar", copied_Yvar)
        object.__setattr__(self, "_status", copied_status)
        object.__setattr__(self, "_ids", normalized_ids)

    @property
    def X(self) -> pd.DataFrame:
        """A detached, index-reset copy of observed physical conditions."""
        return _clone_frame(self._X, "X")

    @property
    def Y(self) -> pd.DataFrame:
        """A detached, index-reset copy of observed outcomes."""
        return _clone_frame(self._Y, "Y")

    @property
    def Yvar(self) -> pd.DataFrame | None:
        """A detached copy of known outcome variances, when supplied."""
        return None if self._Yvar is None else _clone_frame(self._Yvar, "Yvar")

    @property
    def status(self) -> pd.Series | None:
        """A detached copy of row statuses, or ``None`` when omitted."""
        return None if self._status is None else self._status.copy(deep=True)

    @property
    def ids(self) -> tuple[str, ...]:
        """Stable optional observation identifiers."""
        return self._ids

    def _subset(self, *, rows: Sequence[int] | None = None,
                objectives: Sequence[str] | None = None) -> "ObservationBatch":
        """Detach only existing validated rows/outputs, retaining their metadata.

        Projection must not reinterpret a multioutput variance column through
        the stricter public scalar-input contract. No new data is admitted here.
        """
        positions = list(range(len(self._X))) if rows is None else list(rows)
        if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                or not 0 <= i < len(self._X) for i in positions)
                or len(set(positions)) != len(positions)):
            raise ValueError("Subset rows must be unique existing row positions.")
        names = list(self._Y.columns) if objectives is None else list(objectives)
        if len(set(names)) != len(names) or not set(names) <= set(self._Y.columns):
            raise ValueError("Subset objectives must be unique existing columns.")
        subset = object.__new__(ObservationBatch)
        object.__setattr__(subset, "_X", _clone_frame(self._X.iloc[positions], "X"))
        object.__setattr__(subset, "_Y", _clone_frame(self._Y.iloc[positions][names], "Y"))
        object.__setattr__(subset, "_Yvar", None if self._Yvar is None else
                           _clone_frame(self._Yvar.iloc[positions][names], "Yvar"))
        object.__setattr__(subset, "_status", None if self._status is None else
                           _copy_series(self._status.iloc[positions], "status"))
        object.__setattr__(subset, "_ids", tuple(self._ids[i] for i in positions) if self._ids else ())
        return subset

    def successful(self) -> "ObservationBatch":
        """Return a detached batch containing only successful observations."""
        status = (
            pd.Series(["success"] * len(self._X), dtype="object")
            if self._status is None
            else self._status.copy(deep=True)
        )
        mask = status.eq("success").to_numpy()
        subset = self._subset(rows=np.flatnonzero(mask).tolist())
        object.__setattr__(subset, "_status", status.iloc[mask].reset_index(drop=True))
        return subset


@dataclass(frozen=True, init=False)
class PendingBatch:
    """Unique candidate identifiers paired with pending physical conditions."""

    _ids: tuple[str, ...]
    _X: pd.DataFrame = field(repr=False)

    def __init__(self, ids: Sequence[str], X: pd.DataFrame) -> None:
        copied_X = _clone_frame(X, "X")
        normalized_ids = _normalize_ids(ids, len(copied_X), "PendingBatch")
        object.__setattr__(self, "_ids", normalized_ids)
        object.__setattr__(self, "_X", copied_X)

    @property
    def ids(self) -> tuple[str, ...]:
        """Stable identifiers for each pending row."""
        return self._ids

    @property
    def X(self) -> pd.DataFrame:
        """A detached, index-reset copy of pending physical conditions."""
        return _clone_frame(self._X, "X")
