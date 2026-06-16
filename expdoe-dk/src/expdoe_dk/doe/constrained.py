"""
Constrained + discrete DoE generation.

Implements the unified pipeline for all DoE methods:
  1. produce candidate points in [0,1]^d via the chosen method
  2. snap discrete dimensions to legal grid (in physical space)
  3. filter rows that violate LinearConstraints
  4. accept-reject / row-swap repair until n feasible rows accumulated
  5. method-specific post-processing (SA maximin for lhs_maximin only)
"""
from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from ..space import LinearConstraint, Space
from .lhs import latin_hypercube_sample, optimize_lhs_maximin


class InfeasibleDesignError(RuntimeError):
    """Raised when the requested n design points cannot be generated."""


# ----------------------------------------------------------------------- #
# Internal helpers
# ----------------------------------------------------------------------- #
def _unit_to_physical_array(U: np.ndarray, space: Space) -> np.ndarray:
    """[0,1]^d unit array → physical units, with discrete snap."""
    lo = space.lower.cpu().numpy()
    hi = space.upper.cpu().numpy()
    X = lo + U * (hi - lo)
    for j, p in enumerate(space.params):
        if p.kind == "discrete":
            X[:, j] = p.snap(X[:, j])
    return X


def _physical_to_unit_array(X: np.ndarray, space: Space) -> np.ndarray:
    lo = space.lower.cpu().numpy()
    hi = space.upper.cpu().numpy()
    return (X - lo) / (hi - lo)


def _feasibility_mask(X_phys: np.ndarray, space: Space) -> np.ndarray:
    if not space.constraints:
        return np.ones(X_phys.shape[0], dtype=bool)
    names = space.param_names
    mask = np.ones(X_phys.shape[0], dtype=bool)
    for c in space.constraints:
        coeffs = np.array([c.coeffs.get(n, 0.0) for n in names], dtype=np.float64)
        v = X_phys @ coeffs
        mask &= (v >= c.lower - 1e-9) & (v <= c.upper + 1e-9)
    return mask


def _feasibility_diagnostic(space: Space, n: int) -> str:
    """Estimate feasibility fraction via Monte Carlo; useful for error msg."""
    rng = np.random.default_rng(0)
    sample = rng.uniform(size=(10_000, space.n_dims))
    X = _unit_to_physical_array(sample, space)
    mask = _feasibility_mask(X, space)
    frac = mask.mean()
    if frac < 1e-3:
        est = "<1%"
    else:
        est = f"~{frac * 100:.1f}%"
    return (
        f"feasible region ≈ {est} of unconstrained space; "
        f"requested {n} points. Consider relaxing constraints or reducing n."
    )


# ----------------------------------------------------------------------- #
# Per-method primary samplers (no constraint enforcement here)
# ----------------------------------------------------------------------- #
def _draw_lhs_random(n: int, d: int, seed: int) -> np.ndarray:
    return latin_hypercube_sample(n, d, seed=seed)


def _draw_sobol(n: int, d: int, seed: int) -> np.ndarray:
    s = qmc.Sobol(d=d, scramble=True, seed=seed)
    return s.random(n).astype(np.float64)


def _draw_halton(n: int, d: int, seed: int) -> np.ndarray:
    s = qmc.Halton(d=d, scramble=True, seed=seed)
    return s.random(n).astype(np.float64)


def _draw_random_uniform(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(size=(n, d))


def _draw_d_optimal(n: int, space: Space, seed: int) -> np.ndarray:
    """
    D-Optimal via pyDOE3's coordinate exchange (linear model). Falls back to
    LHS maximin if pyDOE3 is unavailable or feasibility too tight.
    """
    try:
        from pyDOE3 import doe_optimal  # noqa: F401
    except Exception:
        warnings.warn(
            "pyDOE3 not available for d_optimal; falling back to lhs_maximin.",
            stacklevel=2,
        )
        design, _ = optimize_lhs_maximin(n, space.n_dims, seed=seed)
        return design

    # pyDOE3 expects candidate set in [-1, 1]; we sample feasible candidates
    # in unit space, transform, run exchange.
    rng = np.random.default_rng(seed)
    candidates_unit = rng.uniform(size=(max(200, 50 * n), space.n_dims))
    cand_phys = _unit_to_physical_array(candidates_unit, space)
    mask = _feasibility_mask(cand_phys, space)
    if mask.sum() < n:
        raise InfeasibleDesignError(
            f"d_optimal: only {mask.sum()} feasible candidates from "
            f"{len(mask)} draws; cannot select {n}. "
            + _feasibility_diagnostic(space, n)
        )
    feasible_unit = candidates_unit[mask]
    # Pick the n most spread-out feasible candidates via simple greedy maximin
    # (lightweight surrogate for full D-optimal exchange; acceptable for v0.1)
    chosen_idx = [int(rng.integers(feasible_unit.shape[0]))]
    for _ in range(n - 1):
        dists = np.min(
            np.linalg.norm(
                feasible_unit[:, None, :] - feasible_unit[chosen_idx][None, :, :],
                axis=-1,
            ),
            axis=1,
        )
        chosen_idx.append(int(np.argmax(dists)))
    return feasible_unit[chosen_idx]


# ----------------------------------------------------------------------- #
# Public dispatch
# ----------------------------------------------------------------------- #
MethodLiteral = Literal[
    "lhs_maximin",
    "lhs_random",
    "sobol",
    "halton",
    "d_optimal",
    "random_uniform",
]


def generate(
    space: Space,
    n: int,
    method: MethodLiteral = "lhs_maximin",
    *,
    n_iterations: int = 2000,
    n_restarts: int = 10,
    seed: int = 42,
    max_resample: int = 50,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate n design points in PHYSICAL units, satisfying all
    `LinearConstraint`s and snapping discrete dims to their grid.

    See module docstring for the unified pipeline. Method-specific notes:

    | Method         | Notes                                                     |
    |----------------|-----------------------------------------------------------|
    | lhs_maximin    | LHS + SA. Best space coverage. Slower (SA iterations).    |
    | lhs_random     | Standard LHS, no optimization.                            |
    | sobol          | Sobol low-discrepancy quasi-random.                       |
    | halton         | Halton low-discrepancy quasi-random.                      |
    | d_optimal      | Greedy maximin over feasible candidate pool (pyDOE3 stub).|
    | random_uniform | Pure random uniform (baseline only).                      |

    Returns
    -------
    pd.DataFrame, columns=space.param_names, n rows.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if method not in {
        "lhs_maximin",
        "lhs_random",
        "sobol",
        "halton",
        "d_optimal",
        "random_uniform",
    }:
        raise ValueError(f"Unknown method: {method}.")

    d = space.n_dims

    # Sanity: estimate feasible tuple count for discrete + constrained spaces
    if space.constraints:
        # Rough early warning (Monte Carlo)
        rng = np.random.default_rng(seed)
        probe_unit = rng.uniform(size=(min(5000, max(1000, 50 * n)), d))
        probe_phys = _unit_to_physical_array(probe_unit, space)
        frac = _feasibility_mask(probe_phys, space).mean()
        if frac < n / probe_phys.shape[0] / 10:
            warnings.warn(
                f"Feasible region is small ({frac * 100:.2f}% of space). "
                f"Design generation may be slow.",
                stacklevel=2,
            )

    # --------------------------------------------------------------- #
    # Method-specific primary sampling + accept-reject loop
    # --------------------------------------------------------------- #
    if method == "lhs_maximin":
        # Two-stage strategy:
        # (a) Try LHS + physical row-swap repair; if successful (likely for
        #     loose constraints), SA-optimize maximin.
        # (b) If LHS structure cannot satisfy constraints (e.g. discrete +
        #     tight A>B), fall back to candidate-pool + greedy maximin.
        design_phys = None
        attempt_seed = seed
        for attempt in range(max(3, max_resample // 5)):
            design_unit = latin_hypercube_sample(n, d, seed=attempt_seed)
            cand = _unit_to_physical_array(design_unit, space)
            if not _feasibility_mask(cand, space).all():
                cand = _repair_design_physical(
                    cand, space,
                    max_swaps=max(2000, n_iterations),
                    seed=attempt_seed,
                )
            if _feasibility_mask(cand, space).all():
                design_phys = cand
                break
            attempt_seed += 1

        if design_phys is None:
            # Fallback: candidate-pool + greedy maximin. Loses strict LHS
            # property but guarantees feasibility under tight constraints.
            design_phys = _pool_greedy_maximin(
                space, n=n, pool_factor=200, max_resample=max_resample,
                seed=seed,
            )

        # SA maximin polish (preserves feasibility via rejection).
        col_weights = 1.0 / (
            space.upper.cpu().numpy() - space.lower.cpu().numpy()
        )

        def feas_fn(design_arr: np.ndarray) -> np.ndarray:
            return _feasibility_mask(design_arr, space)

        design_phys, _dist = _sa_maximin_physical(
            design_phys,
            space=space,
            n_iterations=n_iterations,
            column_weights=col_weights,
            feasibility_fn=feas_fn,
            seed=seed + 1000,
        )

    elif method == "lhs_random":
        design_unit = latin_hypercube_sample(n, d, seed=seed)
        design_phys = _unit_to_physical_array(design_unit, space)
        if not _feasibility_mask(design_phys, space).all():
            design_phys = _repair_design_physical(
                design_phys, space, max_swaps=3000, seed=seed,
            )
            if not _feasibility_mask(design_phys, space).all():
                warnings.warn(
                    "lhs_random: strict LHS structure infeasible under "
                    "constraints; falling back to feasible random sample.",
                    stacklevel=2,
                )
                design_phys = _pool_greedy_maximin(
                    space, n=n, pool_factor=100, max_resample=max_resample,
                    seed=seed,
                )

    elif method in ("sobol", "halton", "random_uniform"):
        draw = {
            "sobol": _draw_sobol,
            "halton": _draw_halton,
            "random_uniform": _draw_random_uniform,
        }[method]
        feasible = []
        attempt_seed = seed
        for _ in range(max_resample):
            batch = draw(max(n * 4, 32), d, attempt_seed)
            phys = _unit_to_physical_array(batch, space)
            mask = _feasibility_mask(phys, space)
            for row in phys[mask]:
                feasible.append(row)
                if len(feasible) >= n:
                    break
            if len(feasible) >= n:
                break
            attempt_seed += 1
        if len(feasible) < n:
            raise InfeasibleDesignError(
                f"{method}: only {len(feasible)} feasible points after "
                f"{max_resample} accept-reject rounds. "
                + _feasibility_diagnostic(space, n)
            )
        design_phys = np.asarray(feasible[:n], dtype=np.float64)

    elif method == "d_optimal":
        design_unit = _draw_d_optimal(n, space, seed=seed)
        design_phys = _unit_to_physical_array(design_unit, space)
        if not _feasibility_mask(design_phys, space).all():
            raise InfeasibleDesignError(
                "d_optimal: discrete-snap produced infeasible rows. "
                + _feasibility_diagnostic(space, n)
            )

    if verbose:
        print(
            f"[expdoe_dk.doe.generate] method={method}, n={n}, d={d}, "
            f"constraints={len(space.constraints)}"
        )

    df = pd.DataFrame(design_phys, columns=space.param_names)
    return df


# ----------------------------------------------------------------------- #
# Candidate-pool greedy maximin: feasible random draws + greedy selection.
# Used as fallback when LHS structure cannot satisfy tight constraints
# (e.g. discrete dims with strict ordering A > B).
# ----------------------------------------------------------------------- #
def _pool_greedy_maximin(
    space: Space,
    n: int,
    pool_factor: int = 200,
    max_resample: int = 50,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool: list[np.ndarray] = []
    attempt = 0
    target = max(pool_factor * n, 1000)
    while len(pool) < target and attempt < max_resample:
        batch_unit = rng.uniform(size=(target, space.n_dims))
        batch_phys = _unit_to_physical_array(batch_unit, space)
        mask = _feasibility_mask(batch_phys, space)
        for row in batch_phys[mask]:
            pool.append(row)
            if len(pool) >= target:
                break
        attempt += 1
    if len(pool) < n:
        raise InfeasibleDesignError(
            f"Candidate pool only yielded {len(pool)} feasible points after "
            f"{attempt} batches; cannot pick {n}. "
            + _feasibility_diagnostic(space, n)
        )

    pool_arr = np.asarray(pool, dtype=np.float64)
    col_weights = 1.0 / (space.upper.cpu().numpy() - space.lower.cpu().numpy())

    # Greedy maximin selection.
    chosen = [int(rng.integers(len(pool_arr)))]
    for _ in range(n - 1):
        chosen_arr = pool_arr[chosen]
        # distance from each pool point to nearest already-chosen
        dists = np.linalg.norm(
            (pool_arr[:, None, :] - chosen_arr[None, :, :]) * col_weights[None, None, :],
            axis=-1,
        ).min(axis=1)
        dists[chosen] = -1.0
        chosen.append(int(np.argmax(dists)))
    return pool_arr[chosen]


# ----------------------------------------------------------------------- #
# Row-swap repair in PHYSICAL space (snapped). For discrete dims this
# preserves grid validity; for continuous dims values are real numbers.
# ----------------------------------------------------------------------- #
def _repair_design_physical(
    design_phys: np.ndarray,
    space: Space,
    max_swaps: int = 3000,
    seed: int = 0,
) -> np.ndarray:
    """Best-effort row-swap repair on physical (already snapped) design."""
    rng = np.random.default_rng(seed)
    n, d = design_phys.shape
    arr = design_phys.copy()

    mask = _feasibility_mask(arr, space)
    if mask.all():
        return arr

    for _ in range(max_swaps):
        infeasible = np.where(~mask)[0]
        if len(infeasible) == 0:
            return arr
        i = int(rng.choice(infeasible))
        # Try a random column swap with any other row.
        dim = int(rng.integers(d))
        j = int(rng.integers(n))
        if i == j:
            continue
        proposal = arr.copy()
        proposal[i, dim], proposal[j, dim] = proposal[j, dim], proposal[i, dim]
        new_mask = _feasibility_mask(proposal, space)
        # Accept if feasibility count weakly improves.
        if new_mask.sum() > mask.sum() or (
            new_mask.sum() == mask.sum() and rng.random() < 0.3
        ):
            arr = proposal
            mask = new_mask
        if mask.all():
            return arr
    return arr


# ----------------------------------------------------------------------- #
# SA maximin in PHYSICAL space with feasibility rejection and discrete snap.
# ----------------------------------------------------------------------- #
def _sa_maximin_physical(
    initial_phys: np.ndarray,
    space: Space,
    n_iterations: int,
    column_weights: np.ndarray,
    feasibility_fn,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n, d = initial_phys.shape
    design = initial_phys.copy()

    def wm(arr: np.ndarray) -> float:
        return float(np.min(pdist(arr * column_weights)))

    current = wm(design)
    T_init, T_min = 1.0, 1e-4
    cooling = (T_min / T_init) ** (1.0 / max(1, n_iterations))
    T = T_init

    for _ in range(n_iterations):
        dim = int(rng.integers(d))
        i, j = rng.choice(n, size=2, replace=False)
        proposal = design.copy()
        proposal[i, dim], proposal[j, dim] = proposal[j, dim], proposal[i, dim]
        # Discrete dims already on grid; swap preserves grid validity.
        if not feasibility_fn(proposal).all():
            T *= cooling
            continue
        new_dist = wm(proposal)
        delta = new_dist - current
        if delta > 0 or rng.random() < np.exp(delta / T):
            design = proposal
            current = new_dist
        T *= cooling

    return design, current
