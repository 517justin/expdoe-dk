"""Stable, detached result contracts for the Phase 1 numerical boundary."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd


def _json_object(value: dict[str, Any]) -> dict[str, Any]:
    detached = deepcopy(value)
    json.dumps(detached, allow_nan=False)
    return detached


def _json_rows(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    return tuple(_json_object(row) for row in rows)


def _clone_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cloned = frame.copy(deep=True).reset_index(drop=True)
    for column in cloned.columns:
        if cloned[column].dtype == object:
            cloned[column] = cloned[column].map(deepcopy)
    return cloned


@dataclass(frozen=True, init=False)
class CapabilityReport:
    _applied: tuple[str, ...]
    _blocked: tuple[str, ...]
    _diagnostics: tuple[dict, ...]

    def __init__(
        self, applied: tuple[str, ...], blocked: tuple[str, ...],
        diagnostics: tuple[dict, ...],
    ) -> None:
        for name, source in (("applied", applied), ("blocked", blocked)):
            values = tuple(source)
            if any(type(value) is not str or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, f"_{name}", values)
        if set(self.applied) & set(self.blocked):
            raise ValueError("applied and blocked must be disjoint")
        object.__setattr__(self, "_diagnostics", _json_rows(tuple(diagnostics)))

    @property
    def applied(self) -> tuple[str, ...]:
        return self._applied

    @property
    def blocked(self) -> tuple[str, ...]:
        return self._blocked

    @property
    def diagnostics(self) -> tuple[dict, ...]:
        return _json_rows(self._diagnostics)

    def to_dict(self) -> dict:
        return _json_object({
            "applied": list(self._applied), "blocked": list(self._blocked),
            "diagnostics": list(self._diagnostics),
        })


@dataclass(frozen=True, init=False)
class Prediction:
    _mean: np.ndarray
    _variance: np.ndarray

    def __init__(self, mean: np.ndarray, variance: np.ndarray) -> None:
        mean = np.array(mean, dtype=float, copy=True)
        variance = np.array(variance, dtype=float, copy=True)
        self._validate_arrays(mean, variance)
        mean.setflags(write=False)
        variance.setflags(write=False)
        object.__setattr__(self, "_mean", mean)
        object.__setattr__(self, "_variance", variance)

    @staticmethod
    def _validate_arrays(mean: np.ndarray, variance: np.ndarray) -> None:
        if mean.ndim != 2 or mean.shape[1] != 1 or variance.shape != mean.shape:
            raise ValueError("Prediction arrays must have matching (n, 1) shape")
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError("Prediction arrays must be finite")
        if (variance < 0).any():
            raise ValueError("Prediction variance must be non-negative")

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def variance(self) -> np.ndarray:
        return self._variance.copy()

    def to_dict(self) -> dict:
        self._validate_arrays(self._mean, self._variance)
        return {"mean": self._mean.tolist(), "variance": self._variance.tolist()}


@dataclass(frozen=True, init=False)
class EvaluationResult:
    _folds: tuple[tuple[int, ...], ...]
    _rows: tuple[dict, ...]
    _exclusions: tuple[dict, ...]

    def __init__(
        self, folds: tuple[tuple[int, ...], ...], rows: tuple[dict, ...],
        exclusions: tuple[dict, ...],
    ) -> None:
        folds = tuple(tuple(fold) for fold in folds)
        if any(any(type(index) is not int or index < 0 for index in fold) for fold in folds):
            raise ValueError("Fold indices must be non-negative integers")
        object.__setattr__(self, "_folds", folds)
        object.__setattr__(self, "_rows", _json_rows(tuple(rows)))
        object.__setattr__(self, "_exclusions", _json_rows(tuple(exclusions)))

    @property
    def folds(self) -> tuple[tuple[int, ...], ...]:
        return self._folds

    @property
    def rows(self) -> tuple[dict, ...]:
        return _json_rows(self._rows)

    @property
    def exclusions(self) -> tuple[dict, ...]:
        return _json_rows(self._exclusions)

    def to_dict(self) -> dict:
        return _json_object({
            "folds": [list(fold) for fold in self._folds],
            "rows": list(self._rows), "exclusions": list(self._exclusions),
        })


@dataclass(frozen=True, init=False)
class RecommendationResult:
    _frame: pd.DataFrame
    _predictions: Prediction
    _diagnostics: dict

    def __init__(
        self, frame: pd.DataFrame, predictions: Prediction, diagnostics: dict,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("RecommendationResult.frame must be a DataFrame")
        if not isinstance(predictions, Prediction):
            raise TypeError("RecommendationResult.predictions must be a Prediction")
        if len(frame) != predictions.mean.shape[0]:
            raise ValueError("Recommendation rows and prediction rows differ")
        object.__setattr__(self, "_frame", _clone_frame(frame))
        object.__setattr__(self, "_predictions", predictions)
        object.__setattr__(self, "_diagnostics", _json_object(diagnostics))

    @property
    def frame(self) -> pd.DataFrame:
        return _clone_frame(self._frame)

    @property
    def predictions(self) -> Prediction:
        return self._predictions

    @property
    def diagnostics(self) -> dict:
        return _json_object(self._diagnostics)

    def to_dict(self) -> dict:
        return _json_object({
            "frame": self._frame.to_dict(orient="records"),
            "predictions": self._predictions.to_dict(),
            "diagnostics": self._diagnostics,
        })


__all__ = ["CapabilityReport", "Prediction", "EvaluationResult", "RecommendationResult"]
