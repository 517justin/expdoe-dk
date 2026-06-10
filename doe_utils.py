"""
doe_utils.py
============
Design of Experiments (DOE) utilities for initialising Bayesian optimisation.

Includes:
  - latin_hypercube_sample  : basic random LHS
  - optimize_lhs_maximin    : SA-optimised maximin LHS
"""

import numpy as np
from scipy.spatial.distance import pdist


def latin_hypercube_sample(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate a random Latin Hypercube Sample (LHS).

    Each dimension is divided into n_samples equal-width bins;
    exactly one point falls in each bin per dimension.

    Parameters
    ----------
    n_samples : int
    n_dims    : int
    seed      : int

    Returns
    -------
    np.ndarray of shape (n_samples, n_dims), values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        samples[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples
    return samples


def optimize_lhs_maximin(n_samples: int, n_dims: int,
                         n_iterations: int = 1000,
                         n_restarts: int = 10,
                         seed: int = 42) -> np.ndarray:
    """
    Optimise a LHS design to maximise the minimum pairwise distance
    (maximin criterion) using Simulated Annealing (SA).

    A maximin LHS spreads the initial points as uniformly as possible in the
    parameter space, helping the GP surrogate estimate length-scales and noise
    hyper-parameters more accurately before the BO phase begins.

    Parameters
    ----------
    n_samples    : int   – number of design points
    n_dims       : int   – number of parameters / dimensions
    n_iterations : int   – SA steps per restart
    n_restarts   : int   – number of independent SA restarts
    seed         : int   – base random seed

    Returns
    -------
    np.ndarray of shape (n_samples, n_dims), values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    best_design = None
    best_min_dist = -np.inf

    for restart in range(n_restarts):
        design = latin_hypercube_sample(n_samples, n_dims, seed=seed + restart)
        current_min_dist = np.min(pdist(design))

        T_init = 1.0
        T_min = 1e-4
        cooling = (T_min / T_init) ** (1.0 / n_iterations)
        T = T_init

        for _ in range(n_iterations):
            dim = rng.integers(n_dims)
            i, j = rng.choice(n_samples, size=2, replace=False)
            new_design = design.copy()
            new_design[i, dim], new_design[j, dim] = new_design[j, dim], new_design[i, dim]
            new_min_dist = np.min(pdist(new_design))
            delta = new_min_dist - current_min_dist
            if delta > 0 or rng.random() < np.exp(delta / T):
                design = new_design
                current_min_dist = new_min_dist
            T *= cooling

        if current_min_dist > best_min_dist:
            best_min_dist = current_min_dist
            best_design = design.copy()

    print(f"[Optimized LHS] n={n_samples}, d={n_dims}, "
          f"maximin distance = {best_min_dist:.4f}")
    return best_design
