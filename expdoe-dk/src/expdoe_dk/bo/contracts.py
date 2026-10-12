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


@dataclass(frozen=True)
class CapabilityReport:
    applied: tuple[str, ...]
    blocked: tuple[str, ...]
    diagnostics: tuple[dict, ...]

    def __post_init__(self) -> None:
        for name in ("applied", "blocked"):
            values = tuple(getattr(self, name))
            if any(type(value) is not str or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        if set(self.applied) & set(self.blocked):
            raise ValueError("applied and blocked must be disjoint")
        object.__setattr__(self, "diagnostics", _json_rows(tuple(self.diagnostics)))

    def to_dict(self) -> dict:
        return _json_object({
            "applied": list(self.applied), "blocked": list(self.blocked),
            "diagnostics": list(self.diagnostics),
        })


@dataclass(frozen=True)
class Prediction:
    mean: np.ndarray
    variance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=float, copy=True)
        variance = np.array(self.variance, dtype=float, copy=True)
        if mean.ndim != 2 or mean.shape[1] != 1 or variance.shape != mean.shape:
            raise ValueError("Prediction arrays must have matching (n, 1) shape")
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError("Prediction arrays must be finite")
        if (variance < 0).any():
            raise ValueError("Prediction variance must be non-negative")
        mean.setflags(write=False)
        variance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "variance": self.variance.tolist()}


@dataclass(frozen=True)
class EvaluationResult:
    folds: tuple[tuple[int, ...], ...]
    rows: tuple[dict, ...]
    exclusions: tuple[dict, ...]

    def __post_init__(self) -> None:
        folds = tuple(tuple(fold) for fold in self.folds)
        if any(any(type(index) is not int or index < 0 for index in fold) for fold in folds):
            raise ValueError("Fold indices must be non-negative integers")
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "rows", _json_rows(tuple(self.rows)))
        object.__setattr__(self, "exclusions", _json_rows(tuple(self.exclusions)))

    def to_dict(self) -> dict:
        return _json_object({
            "folds": [list(fold) for fold in self.folds],
            "rows": list(self.rows), "exclusions": list(self.exclusions),
        })


@dataclass(frozen=True)
class RecommendationResult:
    frame: pd.DataFrame
    predictions: Prediction
    diagnostics: dict

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("RecommendationResult.frame must be a DataFrame")
        if not isinstance(self.predictions, Prediction):
            raise TypeError("RecommendationResult.predictions must be a Prediction")
        if len(self.frame) != self.predictions.mean.shape[0]:
            raise ValueError("Recommendation rows and prediction rows differ")
        object.__setattr__(self, "frame", self.frame.copy(deep=True).reset_index(drop=True))
        object.__setattr__(self, "diagnostics", _json_object(self.diagnostics))

    def to_dict(self) -> dict:
        return _json_object({
            "frame": self.frame.to_dict(orient="records"),
            "predictions": self.predictions.to_dict(),
            "diagnostics": self.diagnostics,
        })


__all__ = ["CapabilityReport", "Prediction", "EvaluationResult", "RecommendationResult"]
