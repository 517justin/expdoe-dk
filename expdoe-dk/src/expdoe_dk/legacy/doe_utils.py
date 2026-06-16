"""Deprecation shim for ax_doe_bo.doe_utils."""
from __future__ import annotations

import warnings

import numpy as np

from ..doe.lhs import (
    latin_hypercube_sample as _new_latin_hypercube_sample,
    optimize_lhs_maximin as _new_optimize_lhs_maximin,
)


def latin_hypercube_sample(
    n_samples: int, n_dims: int, seed: int = 42
) -> np.ndarray:
    warnings.warn(
        "ax_doe_bo.doe_utils.latin_hypercube_sample is deprecated; use "
        "expdoe_dk.doe.latin_hypercube_sample or expdoe_dk.suggest_design.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_latin_hypercube_sample(n_samples, n_dims, seed=seed)


def optimize_lhs_maximin(
    n_samples: int,
    n_dims: int,
    n_iterations: int = 1000,
    n_restarts: int = 10,
    seed: int = 42,
) -> np.ndarray:
    warnings.warn(
        "ax_doe_bo.doe_utils.optimize_lhs_maximin is deprecated; use "
        "expdoe_dk.suggest_design(space, n, method='lhs_maximin') which "
        "additionally supports constraints + discrete steps.",
        DeprecationWarning,
        stacklevel=2,
    )
    design, _ = _new_optimize_lhs_maximin(
        n_samples=n_samples,
        n_dims=n_dims,
        n_iterations=n_iterations,
        n_restarts=n_restarts,
        seed=seed,
    )
    return design
