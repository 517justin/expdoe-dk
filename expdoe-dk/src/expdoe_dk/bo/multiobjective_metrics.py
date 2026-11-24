"""Observed maximization metrics; no fitting, preferences or oracle inference."""
from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module

import numpy as np
import torch

from expdoe_dk.errors import EngineError, ErrorCode


def _invalid(message):
    return EngineError(ErrorCode.CONFIG_INVALID, message)


def _finite_array(value, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise _invalid(f"{name} must be finite numeric values") from error
    if not np.isfinite(result).all():
        raise _invalid(f"{name} must be finite numeric values")
    return result


def _utilities(value):
    result = _finite_array(value, "utilities")
    if result.ndim != 2 or result.shape[1] < 2:
        raise _invalid("utilities must have shape (n, m), with m >= 2")
    return torch.tensor(result.copy(), dtype=torch.double)


def _capability(module, symbol):
    try:
        return getattr(import_module(module), symbol)
    except (ImportError, AttributeError) as error:
        raise EngineError(ErrorCode.ACQUISITION_INCOMPATIBLE,
                          f"Multiobjective metrics require {module}.{symbol}") from error


def pareto_mask(utilities: np.ndarray) -> np.ndarray:
    """Return all nondominated trials, including equal-utility replicates."""
    values = _utilities(utilities)
    nondominated = _capability("botorch.utils.multi_objective.pareto", "is_non_dominated")
    if not len(values):
        return np.zeros(0, dtype=bool)
    return nondominated(values, deduplicate=False).numpy()


def hypervolume(utilities, reference) -> float:
    """Dominated volume above a fixed reference; duplicate utilities count once."""
    values = _utilities(utilities)
    reference = _finite_array(reference, "reference")
    if reference.shape != (values.shape[1],):
        raise _invalid("reference must match the utility objective axis")
    hv = _capability("botorch.utils.multi_objective.hypervolume", "Hypervolume")
    if not len(values):
        return 0.
    front = values[pareto_mask(values.numpy())].unique(dim=0)
    volume = float(hv(ref_point=torch.tensor(reference, dtype=torch.double)).compute(front))
    if not np.isfinite(volume):
        raise _invalid("Hypervolume overflowed")
    return volume


def hypervolume_regret(observed, oracle, reference) -> float:
    """Benchmark-only absolute HV shortfall under the SAME utility geometry."""
    difference = hypervolume(oracle, reference) - hypervolume(observed, reference)
    if difference < -1e-10:
        raise _invalid("Observed hypervolume exceeds oracle; check reference and geometry")
    return max(difference, 0.)


def plateau_signal(history: Sequence[float]) -> dict | None:
    """Advisory only, after five consecutive relative gains <= 1e-3.

    The caller must supply comparable, fixed-geometry measurement updates;
    an empty feasible set or failed-only batches are not progress observations.
    """
    values = _finite_array(history, "hypervolume history")
    if values.ndim != 1 or (values < 0).any() or (np.diff(values) < 0).any():
        raise _invalid("Hypervolume history must be nonnegative and nondecreasing")
    if len(values) < 6:
        return None
    previous = values[-6:-1]
    gains = np.diff(values[-6:]) / np.maximum(np.abs(previous), 1e-12)
    if (gains <= 1e-3).all():
        return {"kind": "plateau", "advisory": True, "stop": False,
                "updates": 5, "relative_gain_threshold": 1e-3,
                "relative_gains": gains.tolist()}
    return None


__all__ = ["pareto_mask", "hypervolume", "hypervolume_regret", "plateau_signal"]
