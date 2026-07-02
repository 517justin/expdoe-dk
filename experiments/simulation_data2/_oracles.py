"""Lab-like constrained discrete oracles for simulation_data2."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

import expdoe_dk as ed


@dataclass(frozen=True)
class ProblemSpec:
    dim: int
    space: ed.Space
    oracle: Callable[..., np.ndarray]
    noiseless: Callable[[pd.DataFrame], np.ndarray]
    true_opt: float
    true_x_hint: dict[str, float]
    n_doe: int
    n_iter: int


def _with_noise(
    noiseless_fn: Callable[[pd.DataFrame], np.ndarray],
    df: pd.DataFrame,
    *,
    rng: np.random.Generator | None = None,
    noise_seed: int = 0,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(noise_seed)
    return noiseless_fn(df) + 0.01 * rng.standard_normal(size=len(df))


def _unit_peak(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def noiseless_4d(df: pd.DataFrame) -> np.ndarray:
    T = df["T"].to_numpy(dtype=float)
    time = df["time"].to_numpy(dtype=float)
    pH = df["pH"].to_numpy(dtype=float)
    catalyst = df["catalyst"].to_numpy(dtype=float)

    T_norm = (T - 60.0) / 60.0
    temp_gain = np.exp(-0.7 / np.clip(0.55 + T_norm, 1e-6, None))
    temp_degrade = np.exp(-0.35 * np.maximum(T - 105.0, 0.0) / 15.0)
    time_rise = 1.0 - np.exp(-3.0 * (time - 10.0) / 170.0)
    time_degrade = np.exp(-0.25 * np.maximum(time - 130.0, 0.0) / 50.0)
    ph_effect = _unit_peak(pH, center=7.0, width=1.1)
    cat_effect = _unit_peak(catalyst, center=2.5, width=0.9)
    return temp_gain * temp_degrade * time_rise * time_degrade * ph_effect * cat_effect


def noiseless_6d(df: pd.DataFrame) -> np.ndarray:
    base = noiseless_4d(df)
    solvent_A = df["solvent_A"].to_numpy(dtype=float)
    additive = df["additive"].to_numpy(dtype=float)
    return base * _formulation_effect(solvent_A, additive)


def _formulation_effect(solvent_A: np.ndarray, additive: np.ndarray) -> np.ndarray:
    solvent_peak = _unit_peak(solvent_A, center=55.0, width=13.0)
    additive_peak = _unit_peak(additive, center=5.0, width=2.2)
    interaction = 1.0 + 0.18 * np.exp(
        -0.5 * ((solvent_A - 55.0) / 10.0) ** 2
        - 0.5 * ((additive - 5.0) / 1.8) ** 2
    )
    penalty = np.exp(-0.08 * np.maximum(additive - 8.0, 0.0))
    return solvent_peak * additive_peak * interaction * penalty


def objective_4d(
    df: pd.DataFrame,
    *,
    rng: np.random.Generator | None = None,
    noise_seed: int = 0,
) -> np.ndarray:
    return _with_noise(noiseless_4d, df, rng=rng, noise_seed=noise_seed)


def objective_6d(
    df: pd.DataFrame,
    *,
    rng: np.random.Generator | None = None,
    noise_seed: int = 0,
) -> np.ndarray:
    return _with_noise(noiseless_6d, df, rng=rng, noise_seed=noise_seed)


def _space_4d() -> ed.Space:
    return ed.Space(
        params=[
            ed.Parameter("T", bounds=(60.0, 120.0), unit="C", kind="discrete", step=1.0),
            ed.Parameter("time", bounds=(10.0, 180.0), unit="min", kind="discrete", step=5.0),
            ed.Parameter("pH", bounds=(4.0, 10.0), unit="", kind="discrete", step=1.0),
            ed.Parameter(
                "catalyst",
                bounds=(0.5, 5.0),
                unit="mol%",
                kind="discrete",
                step=0.5,
            ),
        ],
        objectives="yield",
        maximize=True,
    )


def _space_6d() -> ed.Space:
    return ed.Space(
        params=[
            ed.Parameter("T", bounds=(60.0, 120.0), unit="C", kind="discrete", step=1.0),
            ed.Parameter("time", bounds=(10.0, 180.0), unit="min", kind="discrete", step=5.0),
            ed.Parameter("pH", bounds=(4.0, 10.0), unit="", kind="discrete", step=1.0),
            ed.Parameter(
                "catalyst",
                bounds=(0.5, 5.0),
                unit="mol%",
                kind="discrete",
                step=0.5,
            ),
            ed.Parameter(
                "solvent_A",
                bounds=(20.0, 80.0),
                unit="vol%",
                kind="discrete",
                step=5.0,
            ),
            ed.Parameter(
                "additive",
                bounds=(0.0, 10.0),
                unit="mol%",
                kind="discrete",
                step=1.0,
            ),
        ],
        constraints=[
            ed.LinearConstraint(
                coeffs={"solvent_A": 1.0, "additive": 1.0},
                upper=85.0,
            ),
            ed.LinearConstraint(
                coeffs={"solvent_A": 1.0, "additive": -3.0},
                lower=20.0,
            ),
            ed.LinearConstraint(
                coeffs={"catalyst": 1.0, "additive": 1.0},
                upper=12.0,
            ),
        ],
        objectives="yield",
        maximize=True,
    )


def _grid_dataframe(space: ed.Space) -> pd.DataFrame:
    levels = [param.levels for param in space.params]
    rows = [dict(zip(space.param_names, values)) for values in product(*levels)]
    df = pd.DataFrame(rows)
    mask = space.feasibility_mask(df).numpy()
    return df.loc[mask].reset_index(drop=True)


def _estimate_optimum(
    space: ed.Space,
    noiseless_fn: Callable[[pd.DataFrame], np.ndarray],
) -> tuple[float, dict[str, float]]:
    if space.param_names == ["T", "time", "pH", "catalyst", "solvent_A", "additive"]:
        return _estimate_optimum_6d(space)

    grid = _grid_dataframe(space)
    values = noiseless_fn(grid)
    idx = int(np.argmax(values))
    return float(values[idx]), {
        name: float(grid.iloc[idx][name]) for name in space.param_names
    }


def _estimate_optimum_6d(space: ed.Space) -> tuple[float, dict[str, float]]:
    process_space = _space_4d()
    process_grid = _grid_dataframe(process_space)
    process_values = noiseless_4d(process_grid)
    solvent_levels = space.param_by_name("solvent_A").levels
    additive_levels = space.param_by_name("additive").levels

    best_value = -np.inf
    best_hint: dict[str, float] | None = None
    catalyst = process_grid["catalyst"].to_numpy(dtype=float)
    for solvent_A, additive in product(solvent_levels, additive_levels):
        if solvent_A + additive > 85.0 or solvent_A < 3.0 * additive + 20.0:
            continue
        process_mask = catalyst + additive <= 12.0
        if not np.any(process_mask):
            continue
        feasible_values = process_values[process_mask]
        process_idx = int(np.flatnonzero(process_mask)[int(np.argmax(feasible_values))])
        value = float(
            process_values[process_idx]
            * _formulation_effect(
                np.asarray([solvent_A], dtype=float),
                np.asarray([additive], dtype=float),
            )[0]
        )
        if value > best_value:
            best_value = value
            best_hint = {
                name: float(process_grid.iloc[process_idx][name])
                for name in process_space.param_names
            }
            best_hint["solvent_A"] = float(solvent_A)
            best_hint["additive"] = float(additive)

    if best_hint is None:
        raise ValueError("No feasible 6D grid rows found.")
    return best_value, {name: best_hint[name] for name in space.param_names}


@lru_cache(maxsize=None)
def make_problem(dim: int) -> ProblemSpec:
    if dim == 4:
        space = _space_4d()
        true_opt, true_x_hint = _estimate_optimum(space, noiseless_4d)
        return ProblemSpec(
            dim=4,
            space=space,
            oracle=objective_4d,
            noiseless=noiseless_4d,
            true_opt=true_opt,
            true_x_hint=true_x_hint,
            n_doe=6,
            n_iter=15,
        )
    if dim == 6:
        space = _space_6d()
        true_opt, true_x_hint = _estimate_optimum(space, noiseless_6d)
        return ProblemSpec(
            dim=6,
            space=space,
            oracle=objective_6d,
            noiseless=noiseless_6d,
            true_opt=true_opt,
            true_x_hint=true_x_hint,
            n_doe=10,
            n_iter=20,
        )
    raise ValueError(f"Unsupported dim={dim}. Use 4 or 6.")
