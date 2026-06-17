"""
Shared canonical objectives used by the experiment scripts.

These are ports of the oracles from the sister project's empirical studies
(Plan 2 Exp-7 / Exp-9 / Exp-10 v2 in ``AGENT_KNOWLEDGE.md``):

  - ``reaction_objective_2d``  — Exp-7  (T × conc)
  - ``process_objective_4d``   — Exp-9  (T × conc × pH × t)
  - ``process_objective_6d_v2`` — Exp-10 v2 (4D + polar bimodal + rpm Gaussian)

Each "noisy" oracle returns positive yield with ``N(0, 0.01²)`` noise; each
has a companion noise-free function used for reporting clean gap metrics.

All inputs are :class:`pandas.DataFrame` rows with physical-unit values
keyed by the parameter names declared in the :class:`expdoe_dk.Space`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import expdoe_dk as ed


# --------------------------------------------------------------------- #
# Noiseless forms (used only for reporting clean gaps; not seen by BO)
# --------------------------------------------------------------------- #
def noiseless_2d(df: pd.DataFrame) -> np.ndarray:
    T = df["T"].to_numpy()
    c = df["conc"].to_numpy()
    rate = np.exp(-0.5 / np.clip(T / 600.0, 1e-3, None))
    eff = 4.0 * c * (1.0 - c)
    return rate * eff


def noiseless_4d(df: pd.DataFrame) -> np.ndarray:
    T = df["T"].to_numpy()
    c = df["conc"].to_numpy()
    pH = df["pH"].to_numpy()
    t = df["t"].to_numpy()
    rate_T = np.exp(-1.0 / np.clip(T / 800.0, 1e-3, None))
    eff_c = 4.0 * (c / 2.0) * (1.0 - c / 2.0)
    act_pH = np.exp(-((pH - 7.0) / 1.5) ** 2)
    yield_t = 1.0 - np.exp(-((t - 10.0) / 110.0) * 3.0)
    return rate_T * eff_c * act_pH * yield_t


def noiseless_6d(df: pd.DataFrame) -> np.ndarray:
    T = df["T"].to_numpy()
    c = df["conc"].to_numpy()
    pH = df["pH"].to_numpy()
    t = df["t"].to_numpy()
    polar = df["polar"].to_numpy()
    rpm = df["rpm"].to_numpy()
    rate_T = np.exp(-1.0 / np.clip(T / 800.0, 1e-3, None))
    eff_c = 4.0 * (c / 2.0) * (1.0 - c / 2.0)
    act_pH = np.exp(-((pH - 7.0) / 1.5) ** 2)
    yield_t = 1.0 - np.exp(-((t - 10.0) / 110.0) * 3.0)
    rpm_norm = (rpm - 100.0) / 900.0
    eff_rpm = np.exp(-((rpm_norm - 0.667) ** 2) / (2 * 0.15 ** 2))
    eff_polar = np.abs(np.sin(2.0 * polar * math.pi))
    return rate_T * eff_c * act_pH * yield_t * eff_polar * eff_rpm


# --------------------------------------------------------------------- #
# Noisy variants (used during the BO loop)
# --------------------------------------------------------------------- #
def _with_noise(noiseless_fn: Callable[[pd.DataFrame], np.ndarray],
                df: pd.DataFrame, *, noise_seed: int) -> np.ndarray:
    rng = np.random.default_rng(noise_seed)
    return noiseless_fn(df) + 0.01 * rng.standard_normal(size=len(df))


def reaction_objective_2d(df: pd.DataFrame, *, noise_seed: int = 0) -> np.ndarray:
    """Exp-7 oracle. Peak at T=600 K (boundary), conc=0.5 (interior).
    True noiseless optimum ≈ 0.6065."""
    return _with_noise(noiseless_2d, df, noise_seed=noise_seed)


def process_objective_4d(df: pd.DataFrame, *, noise_seed: int = 0) -> np.ndarray:
    """Exp-9 oracle. True noiseless optimum ≈ 0.34956."""
    return _with_noise(noiseless_4d, df, noise_seed=noise_seed)


def process_objective_6d_v2(df: pd.DataFrame, *, noise_seed: int = 0) -> np.ndarray:
    """Exp-10 v2 oracle. True noiseless optimum ≈ 0.34956."""
    return _with_noise(noiseless_6d, df, noise_seed=noise_seed)


# --------------------------------------------------------------------- #
# Problem specifications (Space + oracle + budget) shared by experiments
# --------------------------------------------------------------------- #
@dataclass
class ProblemSpec:
    dim: int
    space: ed.Space
    oracle: Callable[..., np.ndarray]
    noiseless: Callable[[pd.DataFrame], np.ndarray]
    true_opt: float
    n_doe: int
    n_iter: int


def make_problem(dim: int) -> ProblemSpec:
    """Return the canonical (Space, oracle, budget) for the given dim."""
    if dim == 2:
        space = ed.Space(
            params=[
                ed.Parameter("T",    bounds=(300.0, 600.0), unit="K"),
                ed.Parameter("conc", bounds=(0.0, 1.0),     unit="mol/L"),
            ],
            objectives="yield",
            maximize=True,
        )
        return ProblemSpec(
            dim=2, space=space, oracle=reaction_objective_2d,
            noiseless=noiseless_2d,
            true_opt=0.6065, n_doe=6, n_iter=15,
        )
    if dim == 4:
        space = ed.Space(
            params=[
                ed.Parameter("T",    bounds=(300.0, 800.0), unit="K"),
                ed.Parameter("conc", bounds=(0.0, 2.0),     unit="mol/L"),
                ed.Parameter("pH",   bounds=(4.0, 10.0),    unit=""),
                ed.Parameter("t",    bounds=(10.0, 120.0),  unit="min"),
            ],
            objectives="yield",
            maximize=True,
        )
        return ProblemSpec(
            dim=4, space=space, oracle=process_objective_4d,
            noiseless=noiseless_4d,
            true_opt=0.34956, n_doe=12, n_iter=30,
        )
    if dim == 6:
        space = ed.Space(
            params=[
                ed.Parameter("T",     bounds=(300.0, 800.0), unit="K"),
                ed.Parameter("conc",  bounds=(0.0, 2.0),     unit="mol/L"),
                ed.Parameter("pH",    bounds=(4.0, 10.0),    unit=""),
                ed.Parameter("t",     bounds=(10.0, 120.0),  unit="min"),
                ed.Parameter("polar", bounds=(0.0, 1.0),     unit=""),
                ed.Parameter("rpm",   bounds=(100.0, 1000.0), unit="rpm"),
            ],
            objectives="yield",
            maximize=True,
        )
        return ProblemSpec(
            dim=6, space=space, oracle=process_objective_6d_v2,
            noiseless=noiseless_6d,
            true_opt=0.34956, n_doe=18, n_iter=30,
        )
    raise ValueError(f"Unsupported dim={dim}. Use 2, 4, or 6.")
