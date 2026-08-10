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

from ..space import Space
from .lhs import latin_hypercube_sample, optimize_lhs_maximin


class InfeasibleDesignError(RuntimeError):
    """Raised when the requested n design points cannot be generated."""


# ----------------------------------------------------------------------- #
# Internal helpers
# ----------------------------------------------------------------------- #
def _unit_to_physical_array(U: np.ndarray, space: Space) -> pd.DataFrame:
    """Decode a numerical unit cube into an ordered physical DataFrame."""
    unit = np.asarray(U, dtype=np.float64)
    model_bounds = space.model_bounds_tensor.cpu().numpy()
    model = model_bounds[0] + unit * (model_bounds[1] - model_bounds[0])
    return space.model_to_physical(torch.as_tensor(model, dtype=torch.float64))


def _physical_to_unit_array(X: pd.DataFrame, space: Space) -> np.ndarray:
    """Encode physical rows as numeric model coordinates for distance math."""
    return space.physical_to_model(X).cpu().numpy()


def _feasibility_mask(X_phys: pd.DataFrame, space: Space) -> np.ndarray:
    return space.feasibility_mask(X_phys).cpu().numpy()


def _physical_frame_from_records(
    records: list[dict[str, object]], space: Space
) -> pd.DataFrame:
    """Build physical rows without pandas coercing heterogeneous levels."""
    data: dict[str, object] = {}
    for parameter in space.params:
        values = [record[parameter.name] for record in records]
        data[parameter.name] = (
            pd.Series(values, dtype=object)
            if parameter.kind in {"categorical", "ordinal"}
            else values
        )
    return pd.DataFrame(data, columns=space.param_names)


def _model_column_weights(space: Space) -> np.ndarray:
    bounds = space.model_bounds_tensor.cpu().numpy()
    spans = bounds[1] - bounds[0]
    return np.divide(1.0, spans, out=np.ones_like(spans), where=spans != 0)


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
    """Select a deterministic D-optimal design from feasible numeric candidates."""
    rng = np.random.default_rng(seed)
    candidates_unit = rng.uniform(size=(max(256, 64 * n), space.n_dims))
    cand_phys = _unit_to_physical_array(candidates_unit, space)
    mask = _feasibility_mask(cand_phys, space)
    if mask.sum() < n:
        raise InfeasibleDesignError(
            f"d_optimal: only {mask.sum()} feasible candidates from "
            f"{len(mask)} draws; cannot select {n}. "
            + _feasibility_diagnostic(space, n)
        )
    feasible_unit = candidates_unit[mask]
    matrix = np.column_stack(
        (np.ones(len(feasible_unit)), feasible_unit, feasible_unit**2)
    )
    information = np.eye(matrix.shape[1], dtype=np.float64) * 1e-12
    selected: list[int] = []
    remaining = np.ones(len(matrix), dtype=bool)
    for _ in range(n):
        best, best_score = -1, -np.inf
        for index in np.flatnonzero(remaining):
            sign, score = np.linalg.slogdet(
                information + np.outer(matrix[index], matrix[index])
            )
            if sign > 0 and score > best_score:
                best, best_score = int(index), float(score)
        if best < 0:
            raise InfeasibleDesignError("d_optimal: no positive determinant candidate")
        selected.append(best)
        information += np.outer(matrix[best], matrix[best])
        remaining[best] = False
    return feasible_unit[selected]


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


def _legacy_generate(
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
        model_design = _physical_to_unit_array(design_phys, space)
        col_weights = _model_column_weights(space)

        def feas_fn(model_arr: np.ndarray) -> np.ndarray:
            physical = space.model_to_physical(
                torch.as_tensor(model_arr, dtype=torch.float64)
            )
            return _feasibility_mask(physical, space)

        model_design, _dist = _sa_maximin_physical(
            model_design,
            space=space,
            n_iterations=n_iterations,
            column_weights=col_weights,
            feasibility_fn=feas_fn,
            seed=seed + 1000,
        )
        design_phys = space.model_to_physical(
            torch.as_tensor(model_design, dtype=torch.float64)
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
        feasible: list[dict[str, object]] = []
        attempt_seed = seed
        for _ in range(max_resample):
            batch = draw(max(n * 4, 32), d, attempt_seed)
            phys = _unit_to_physical_array(batch, space)
            mask = _feasibility_mask(phys, space)
            for row in phys.loc[mask].to_dict(orient="records"):
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
        design_phys = _physical_frame_from_records(feasible[:n], space)

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

    design_phys = design_phys.reset_index(drop=True)
    if not _feasibility_mask(design_phys, space).all():
        raise InfeasibleDesignError(
            f"{method}: final reconstructed design contains infeasible rows."
        )
    return design_phys


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
    """Compatibility wrapper over the diagnostic initial-design engine.

    Historical tuning arguments are accepted so existing callers continue to
    run; the v0.5 deterministic selector intentionally does not vary by them.
    """
    del n_iterations, n_restarts, max_resample, verbose
    from .design import suggest_design

    result = suggest_design(space, n, method=method, seed=seed)
    assert isinstance(result, pd.DataFrame)
    return result


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
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool: list[dict[str, object]] = []
    attempt = 0
    target = max(pool_factor * n, 1000)
    while len(pool) < target and attempt < max_resample:
        batch_unit = rng.uniform(size=(target, space.n_dims))
        batch_phys = _unit_to_physical_array(batch_unit, space)
        mask = _feasibility_mask(batch_phys, space)
        for row in batch_phys.loc[mask].to_dict(orient="records"):
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

    pool_frame = _physical_frame_from_records(pool, space)
    pool_arr = _physical_to_unit_array(pool_frame, space)
    col_weights = _model_column_weights(space)

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
    return pool_frame.iloc[chosen].reset_index(drop=True)


# ----------------------------------------------------------------------- #
# Row-swap repair in PHYSICAL space (snapped). For discrete dims this
# preserves grid validity; for continuous dims values are real numbers.
# ----------------------------------------------------------------------- #
def _repair_design_physical(
    design_phys: pd.DataFrame,
    space: Space,
    max_swaps: int = 3000,
    seed: int = 0,
) -> pd.DataFrame:
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
        left = proposal.iat[i, dim]
        proposal.iat[i, dim] = proposal.iat[j, dim]
        proposal.iat[j, dim] = left
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
    if n < 2:
        return design, math.inf

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
