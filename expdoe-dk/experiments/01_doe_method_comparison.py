"""
Experiment 01 — How much does the DoE method matter for BO?

Question
--------
Holding the BO budget and (no) domain knowledge fixed, how much does the
choice of initial-design method affect the final best yield?

Setup
-----
- Same toy chemistry oracle as the examples (4D, A−B≥1 constraint).
- 5 DoE methods: lhs_maximin / lhs_random / sobol / halton / random_uniform.
- 3 random seeds per method (kept small so the script runs in <2 minutes).
- n_doe = 8, n_iter = 12 (matches example 01).
- No injected knowledge (Campaign auto-applies the Cat ② random_augment
  default — same for every method, so the comparison is fair).

Output
------
- Prints a per-method summary table (best yield median ± best-of-3,
  improvement over the worst method).
- Saves the full history per seed/method to
  ``outputs/experiment_01_doe_methods.csv`` for further analysis.

Run with:
    python experiments/01_doe_method_comparison.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import expdoe_dk as ed


# --------------------------------------------------------------------- #
# Shared toy chemistry oracle (peak at T=95, time=120, A=7, B=3)
# --------------------------------------------------------------------- #
def oracle(df: pd.DataFrame) -> np.ndarray:
    T = df["T"].to_numpy()
    t = df["time"].to_numpy()
    A = df["conc_A"].to_numpy()
    B = df["conc_B"].to_numpy()
    rate = np.exp(-1.5 / (T / 100.0))
    sat = 1.0 - np.exp(-t / 60.0)
    eA = 1.0 - 0.04 * (A - 7.0) ** 2
    eB = 1.0 - 0.06 * (B - 3.0) ** 2
    rng = np.random.default_rng(0)
    return 95.0 * rate * sat * eA * eB + rng.normal(0.0, 0.3, size=len(df))


def _build_space() -> ed.Space:
    return ed.Space(
        params=[
            ed.Parameter("T",      bounds=(60.0, 120.0), unit="°C"),
            ed.Parameter("time",   bounds=(10.0, 180.0), unit="min"),
            ed.Parameter("conc_A", bounds=(1.0, 10.0), unit="mL",
                         kind="discrete", step=1.0),
            ed.Parameter("conc_B", bounds=(1.0, 10.0), unit="mL",
                         kind="discrete", step=1.0),
        ],
        constraints=[
            ed.LinearConstraint(
                coeffs={"conc_A": 1.0, "conc_B": -1.0}, lower=1.0,
                name="A − B ≥ 1 mL",
            ),
        ],
        objectives="yield_pct",
        maximize=True,
    )


# --------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------- #
METHODS = ["lhs_maximin", "lhs_random", "sobol", "halton", "random_uniform"]
SEEDS = [42, 43, 44]
N_DOE = 8
N_ITER = 12


def run_one(method: str, seed: int) -> dict:
    space = _build_space()
    campaign = ed.Campaign(space, knowledge=None, seed=seed)
    doe = campaign.suggest_doe(n=N_DOE, method=method)
    campaign.tell(doe, oracle(doe))
    for it in range(N_ITER):
        nxt = campaign.ask(q=1, iteration=it)
        campaign.tell(nxt, oracle(nxt))
    result = campaign.finalize()
    return {
        "method": method,
        "seed": seed,
        "best_y": result.best_y_physical,
        "history": result.history_df,
    }


def main() -> None:
    warnings.simplefilter("ignore")

    print("=" * 72)
    print(f" Experiment 01 — DoE method × BO outcome")
    print(f"   methods = {METHODS}")
    print(f"   seeds   = {SEEDS}   n_doe={N_DOE}, n_iter={N_ITER}")
    print(f"   oracle  = peak at T=95 °C, time=120 min, A=7 mL, B=3 mL")
    print("=" * 72)

    rows: list[dict] = []
    history_rows: list[pd.DataFrame] = []
    t0 = time.time()
    for method in METHODS:
        for seed in SEEDS:
            tic = time.time()
            res = run_one(method, seed)
            elapsed = time.time() - tic
            rows.append({"method": method, "seed": seed,
                         "best_y": res["best_y"], "secs": elapsed})
            h = res["history"].copy()
            h["method"] = method
            h["seed"] = seed
            h["trial"] = range(1, len(h) + 1)
            history_rows.append(h)
            print(f"  {method:<15} seed={seed}  "
                  f"best={res['best_y']:6.2f}  ({elapsed:4.1f}s)")
    total = (time.time() - t0) / 60.0
    print(f"\n  Total: {total:.1f} min")

    summary = pd.DataFrame(rows)
    agg = (summary.groupby("method")["best_y"]
                .agg(median="median", best="max", worst="min", std="std")
                .round(3)
                .sort_values("median", ascending=False))
    worst_overall = agg["median"].min()
    agg["vs worst"] = (agg["median"] - worst_overall).round(3)
    print("\n" + "─" * 72)
    print(" Summary (best yield across {} seeds):".format(len(SEEDS)))
    print("─" * 72)
    print(agg.to_string())

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    full = pd.concat(history_rows, axis=0, ignore_index=True)
    full.to_csv(out_dir / "experiment_01_doe_methods.csv", index=False)
    agg.to_csv(out_dir / "experiment_01_summary.csv")
    print(f"\nWrote:")
    print(f"  {out_dir / 'experiment_01_doe_methods.csv'}  ({len(full)} rows)")
    print(f"  {out_dir / 'experiment_01_summary.csv'}")


if __name__ == "__main__":
    main()
