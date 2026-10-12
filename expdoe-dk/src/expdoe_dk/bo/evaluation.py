"""Deterministic pooled out-of-fold evaluation; references never recommend."""
from __future__ import annotations

import numpy as np

from ..domain import ObservationBatch, Space
from ..errors import EngineError, ErrorCode
from ..knowledge import Knowledge
from .contracts import EvaluationResult
from .surrogates import _PhysicalFeatures, _eligible, _fit_surrogate, _model_inputs


def make_folds(n: int, *, seed: int = 0) -> tuple[tuple[int, ...], ...]:
    """Eligible-row positions: LOO for 3–10, shuffled five-fold otherwise."""
    if type(n) is not int or type(seed) is not int or seed < 0:
        raise EngineError(ErrorCode.CONFIG_INVALID, "n and seed must be integers; seed must be non-negative")
    if n < 3:
        raise EngineError(ErrorCode.INSUFFICIENT_DATA, "Evaluation requires at least three successful observations")
    if n <= 10:
        return tuple((i,) for i in range(n))
    return tuple(tuple(int(i) for i in chunk)
                 for chunk in np.array_split(np.random.default_rng(seed).permutation(n), 5))


def _reference_classes():
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as error:
        raise EngineError(ErrorCode.CONFIG_INVALID,
            "Reference evaluation requires the optional reference-models dependency",
            hints=('Install with: pip install "expdoe-dk[reference-models]"',)) from error
    return Ridge, RandomForestRegressor


def _subset(batch, indices):
    return ObservationBatch(batch.X.iloc[indices], batch.Y.iloc[indices],
        Yvar=None if batch.Yvar is None else batch.Yvar.iloc[indices],
        ids=tuple(batch.ids[i] for i in indices))


def evaluate_models(space: Space, observations: ObservationBatch, *,
                    tracks: dict[str, Knowledge | None], seed: int = 0) -> EvaluationResult:
    """Compare production GP tracks plus ridge/RF on identical physical OOF rows.

    Folds index the success-only row sequence; observation_ids retain the source
    mapping. Every row records fold-local transforms and pooled physical metrics.
    Ridge/RF are always appended internally and are not production track options.
    """
    if (not isinstance(tracks, dict) or not tracks or any(
        not isinstance(name, str) or not name or name in {"ridge", "random_forest"}
        or (knowledge is not None and not isinstance(knowledge, Knowledge))
        for name, knowledge in tracks.items())):
        raise EngineError(ErrorCode.CONFIG_INVALID,
            "tracks must contain named GP knowledge configurations; ridge/random_forest are reserved references")
    batch, exclusions = _eligible(space, observations, minimum=3)
    folds = make_folds(len(batch.X), seed=seed)
    Ridge, RandomForest = _reference_classes()
    names = [*tracks, "ridge", "random_forest"]
    predictions = {name: np.empty(len(batch.X)) for name in names}
    transforms = {name: [] for name in names}
    y = batch.Y.to_numpy(dtype=float)[:, 0]
    for fold_index, test in enumerate(folds):
        training_indices = [i for i in range(len(y)) if i not in test]
        training = _subset(batch, training_indices)
        test_X = batch.X.iloc[list(test)]
        for name, knowledge in tracks.items():
            try:
                fitted = _fit_surrogate(space, training, knowledge=knowledge, seed=seed)
                predictions[name][list(test)] = fitted.predict(test_X).mean[:, 0]
                transforms[name].append(fitted.transforms)
            except EngineError as error:
                raise EngineError(error.code, error.message,
                    details={**error.details, "track": name, "fold": fold_index}, hints=error.hints) from error
        features = _PhysicalFeatures(space, training.X)
        train_features = features(_model_inputs(space, training.X)).numpy()
        test_features = features(_model_inputs(space, test_X)).numpy()
        for name, estimator in (("ridge", Ridge(alpha=1.)),
                ("random_forest", RandomForest(n_estimators=200, random_state=seed, n_jobs=1))):
            try:
                estimator.fit(train_features, training.Y.to_numpy(dtype=float)[:, 0])
                output = estimator.predict(test_features)
                if not np.isfinite(output).all():
                    raise ValueError("Reference produced non-finite predictions")
                predictions[name][list(test)] = output
                transforms[name].append({**features.describe(), "response_transform": "physical_identity"})
            except Exception as error:
                raise EngineError(ErrorCode.MODEL_FIT_FAILED, f"{name} fitting failed: {error}",
                    details={"track": name, "fold": fold_index}) from error
    constant = bool(np.all(y == y[0]))
    total = float(np.sum((y - y.mean())**2))
    rows = []
    for name in names:
        residual = y - predictions[name]
        rows.append({"track": name, "role": "production" if name in tracks else "reference",
            "rmse": float(np.sqrt(np.mean(residual**2))), "mae": float(np.mean(abs(residual))),
            "r2": None if constant else float(1 - np.sum(residual**2) / total),
            "r2_reason": "constant target: pooled R2 is undefined" if constant else None,
            "objective": space.objectives[0], "units": "physical", "observation_ids": list(batch.ids),
            "oof_predictions": predictions[name].tolist(), "fold_transforms": transforms[name]})
    return EvaluationResult(folds, tuple(rows), exclusions)


__all__ = ["make_folds", "evaluate_models"]
