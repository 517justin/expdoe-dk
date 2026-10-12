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


@dataclass(frozen=True)
class CapabilityReport:
    applied: tuple[str, ...]
    blocked: tuple[str, ...]
    diagnostics: tuple[dict, ...]

    def __post_init__(self) -> None:
        for name in ("applied", "blocked"):
            source = object.__getattribute__(self, name)
            values = tuple(source)
            if any(type(value) is not str or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        if set(self.applied) & set(self.blocked):
            raise ValueError("applied and blocked must be disjoint")
        diagnostics = object.__getattribute__(self, "diagnostics")
        object.__setattr__(self, "diagnostics", _json_rows(tuple(diagnostics)))

    def __getattribute__(self, name: str):
        value = object.__getattribute__(self, name)
        return _json_rows(value) if name == "diagnostics" else value

    def to_dict(self) -> dict:
        diagnostics = object.__getattribute__(self, "diagnostics")
        return _json_object({
            "applied": list(self.applied), "blocked": list(self.blocked),
            "diagnostics": list(diagnostics),
        })


@dataclass(frozen=True)
class Prediction:
    mean: np.ndarray
    variance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(object.__getattribute__(self, "mean"), dtype=float, copy=True)
        variance = np.array(object.__getattribute__(self, "variance"), dtype=float, copy=True)
        self._validate_arrays(mean, variance)
        mean.setflags(write=False)
        variance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)

    @staticmethod
    def _validate_arrays(mean: np.ndarray, variance: np.ndarray) -> None:
        if mean.ndim != 2 or mean.shape[1] != 1 or variance.shape != mean.shape:
            raise ValueError("Prediction arrays must have matching (n, 1) shape")
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError("Prediction arrays must be finite")
        if (variance < 0).any():
            raise ValueError("Prediction variance must be non-negative")

    def __getattribute__(self, name: str):
        value = object.__getattribute__(self, name)
        return value.copy() if name in {"mean", "variance"} else value

    def to_dict(self) -> dict:
        mean = object.__getattribute__(self, "mean")
        variance = object.__getattribute__(self, "variance")
        self._validate_arrays(mean, variance)
        return {"mean": mean.tolist(), "variance": variance.tolist()}


@dataclass(frozen=True)
class EvaluationResult:
    folds: tuple[tuple[int, ...], ...]
    rows: tuple[dict, ...]
    exclusions: tuple[dict, ...]

    def __post_init__(self) -> None:
        folds = tuple(tuple(fold) for fold in object.__getattribute__(self, "folds"))
        if any(any(type(index) is not int or index < 0 for index in fold) for fold in folds):
            raise ValueError("Fold indices must be non-negative integers")
        object.__setattr__(self, "folds", folds)
        for name in ("rows", "exclusions"):
            values = object.__getattribute__(self, name)
            object.__setattr__(self, name, _json_rows(tuple(values)))

    def __getattribute__(self, name: str):
        value = object.__getattribute__(self, name)
        return _json_rows(value) if name in {"rows", "exclusions"} else value

    def to_dict(self) -> dict:
        rows = object.__getattribute__(self, "rows")
        exclusions = object.__getattribute__(self, "exclusions")
        return _json_object({
            "folds": [list(fold) for fold in self.folds],
            "rows": list(rows), "exclusions": list(exclusions),
        })


@dataclass(frozen=True)
class RecommendationResult:
    frame: pd.DataFrame
    predictions: Prediction
    diagnostics: dict

    def __post_init__(self) -> None:
        frame = object.__getattribute__(self, "frame")
        predictions = object.__getattribute__(self, "predictions")
        diagnostics = object.__getattribute__(self, "diagnostics")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("RecommendationResult.frame must be a DataFrame")
        if not isinstance(predictions, Prediction):
            raise TypeError("RecommendationResult.predictions must be a Prediction")
        if len(frame) != predictions.mean.shape[0]:
            raise ValueError("Recommendation rows and prediction rows differ")
        object.__setattr__(self, "frame", _clone_frame(frame))
        object.__setattr__(self, "diagnostics", _json_object(diagnostics))

    def __getattribute__(self, name: str):
        value = object.__getattribute__(self, name)
        if name == "frame":
            return _clone_frame(value)
        if name == "diagnostics":
            return _json_object(value)
        return value

    def to_dict(self) -> dict:
        frame = object.__getattribute__(self, "frame")
        diagnostics = object.__getattribute__(self, "diagnostics")
        return _json_object({
            "frame": frame.to_dict(orient="records"),
            "predictions": self.predictions.to_dict(),
            "diagnostics": diagnostics,
        })


__all__ = ["CapabilityReport", "Prediction", "EvaluationResult", "RecommendationResult"]
