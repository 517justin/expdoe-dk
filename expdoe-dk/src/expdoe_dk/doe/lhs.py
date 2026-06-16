"""
Basic Latin Hypercube Sampling (LHS) and maximin-optimized variant.

These operate in the unit hypercube [0,1]^d. Constraint and discrete handling
live in `constrained.py`; the public entry point is `doe.generate(space, ...)`.

Ported from ax_doe_bo/doe_utils.py with minor refactors:
- numpy → torch defaults removed (still numpy-internal for SA speed)
- max distance metric kept (Euclidean on raw [0,1]^d)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def latin_hypercube_sample(
    n_samples: int,
    n_dims: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Random LHS in the unit hypercube [0,1]^d.

    Each dim divided into n_samples equal-width bins; one point per bin per dim.
    """
    if n_samples < 1 or n_dims < 1:
        raise ValueError(f"n_samples={n_samples}, n_dims={n_dims} must be >= 1.")
    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, n_dims), dtype=np.float64)
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        samples[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples
    return samples


def optimize_lhs_maximin(
    n_samples: int,
    n_dims: int,
    n_iterations: int = 2000,
    n_restarts: int = 10,
    seed: int = 42,
    *,
    column_weights: np.ndarray | None = None,
    feasibility_fn=None,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Maximize the minimum pairwise distance of an LHS via Simulated Annealing
    with row-swap moves (LHS structure preserved).

    Parameters
    ----------
    n_samples, n_dims, n_iterations, n_restarts, seed
        Standard SA loop parameters.
    column_weights : np.ndarray | None
        Per-dimension weights for distance (after scaling to [0,1]). When the
        downstream space mixes physical units of different magnitudes (mL,
        °C), pass `1 / (high - low)` to keep them commensurable. Defaults
        to 1.0 (uniform).
    feasibility_fn : callable | None
        Function `f(design: np.ndarray) -> np.ndarray[bool]` returning which
        rows satisfy constraints. SA proposals creating any infeasible row
        are rejected. Used by `constrained.py`.
    verbose : bool
        If True, print final maximin distance.

    Returns
    -------
    (design, min_dist)
        design shape (n_samples, n_dims) in [0,1]^d; min_dist of best run.
    """
    rng = np.random.default_rng(seed)
    weights = (
        np.ones(n_dims, dtype=np.float64)
        if column_weights is None
        else np.asarray(column_weights, dtype=np.float64)
    )

    def weighted_min_pdist(design: np.ndarray) -> float:
        return float(np.min(pdist(design * weights)))

    best_design: np.ndarray | None = None
    best_min_dist = -np.inf

    for restart in range(n_restarts):
        design = latin_hypercube_sample(n_samples, n_dims, seed=seed + restart)
        # If a feasibility function is provided and the initial design has
        # infeasible rows, this restart is skipped (constrained module handles
        # repair before calling SA).
        if feasibility_fn is not None and not feasibility_fn(design).all():
            continue

        current = weighted_min_pdist(design)
        T_init, T_min = 1.0, 1e-4
        cooling = (T_min / T_init) ** (1.0 / max(1, n_iterations))
        T = T_init

        for _ in range(n_iterations):
            dim = int(rng.integers(n_dims))
            i, j = rng.choice(n_samples, size=2, replace=False)
            proposal = design.copy()
            proposal[i, dim], proposal[j, dim] = proposal[j, dim], proposal[i, dim]
            if feasibility_fn is not None and not feasibility_fn(proposal).all():
                T *= cooling
                continue
            new_dist = weighted_min_pdist(proposal)
            delta = new_dist - current
            if delta > 0 or rng.random() < np.exp(delta / T):
                design = proposal
                current = new_dist
            T *= cooling

        if current > best_min_dist:
            best_min_dist = current
            best_design = design.copy()

    if best_design is None:
        raise RuntimeError(
            "optimize_lhs_maximin: no feasible LHS found across all restarts. "
            "Consider relaxing constraints or reducing n_samples."
        )
    if verbose:
        print(
            f"[Optimized LHS] n={n_samples}, d={n_dims}, "
            f"maximin distance = {best_min_dist:.4f}"
        )
    return best_design, float(best_min_dist)
