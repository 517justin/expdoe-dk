"""
Experiment 02 — How much does the *type* of domain knowledge matter?

Question
--------
Holding the DoE method and BO budget fixed, which knowledge configuration
gives the best yield on the chemistry oracle?

We sample the five categories defined in the project's
``AGENT_KNOWLEDGE.md`` synthesis:

  ① Domain knowledge (correct)      — Arrhenius + monotone + quadratic peak
  ② Pure regularization             — only with_random_augment(n=20)
  ③ Weak knowledge (GP prior)       — only with_gp_prior("medium")
  ④ Dimension-sensitive (mean fn)   — Arrhenius alone (frozen)
  ⑤ Mono + prior combo              — monotone + gp_prior (uses v0.3 auto-rescue)

Plus an A: baseline that uses no knowledge — Campaign auto-applies the
random-augment Cat ② default, which doubles as a sanity check that the
default is competitive.

Setup
-----
- Same oracle and space as experiment 01 (peak at T=95, time=120, A=7, B=3).
- DoE: lhs_maximin, n_doe = 8.
- BO: n_iter = 12, q = 1.
- 3 seeds per knowledge config.

Output
------
- Per-config summary table (best yield median ± best/worst over seeds).
- Full history CSV under ``outputs/`` for downstream analysis.

Run with:
    python experiments/02_knowledge_comparison.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import expdoe_dk as ed


# --------------------------------------------------------------------- #
# Same toy oracle as experiment 01
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


def _space() -> ed.Space:
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
# Five knowledge configurations (factories — one per seed)
# --------------------------------------------------------------------- #
def _k_baseline() -> ed.Knowledge:
    # No items: Campaign will auto-apply with_random_augment(n=20) (Cat ②).
    return ed.Knowledge()


def _k_random_augment() -> ed.Knowledge:
    return ed.Knowledge().strict().with_random_augment(n=20)


def _k_gp_prior_only() -> ed.Knowledge:
    return ed.Knowledge().strict().with_gp_prior(lengthscale="medium")


def _k_full_correct() -> ed.Knowledge:
    return (
        ed.Knowledge()
        .strict()
        .with_arrhenius("T")
        .with_monotone("time", effect="increases_objective")
        .with_quadratic_peak("conc_A", center=7.0)
        .with_random_augment(n=20)
    )


def _k_arrhenius_only() -> ed.Knowledge:
    return ed.Knowledge().strict().with_arrhenius("T")


def _k_mono_plus_prior() -> ed.Knowledge:
    # Triggers v0.3 ε auto-rescue (combining monotone + strong prior with
    # default ε would otherwise raise EpsilonConflictError).
    return (
        ed.Knowledge()
        .strict()
        .with_monotone("time", effect="increases_objective", epsilon=0.02)
        .with_gp_prior(lengthscale="strong")
    )


CONFIGS: dict[str, Callable[[], ed.Knowledge]] = {
    "A: baseline (auto random_augment)": _k_baseline,
    "②: random_augment only": _k_random_augment,
    "③: gp_prior only": _k_gp_prior_only,
    "①: full domain knowledge": _k_full_correct,
    "④: Arrhenius mean only": _k_arrhenius_only,
    "⑤: monotone + gp_prior (rescued)": _k_mono_plus_prior,
}

SEEDS = [42, 43, 44]
N_DOE = 8
N_ITER = 12


def run_one(name: str, knowledge_factory: Callable[[], ed.Knowledge],
            seed: int) -> dict:
    space = _space()
    campaign = ed.Campaign(space, knowledge_factory(), seed=seed)
    doe = campaign.suggest_doe(n=N_DOE, method="lhs_maximin")
    campaign.tell(doe, oracle(doe))
    for it in range(N_ITER):
        nxt = campaign.ask(q=1, iteration=it)
        campaign.tell(nxt, oracle(nxt))
    res = campaign.finalize()
    return {
        "config": name,
        "seed": seed,
        "best_y": res.best_y_physical,
        "history": res.history_df,
    }


def main() -> None:
    warnings.simplefilter("ignore")

    print("=" * 72)
    print(" Experiment 02 — Knowledge configuration × BO outcome")
    print(f"   configs = {list(CONFIGS)}")
    print(f"   seeds   = {SEEDS}   n_doe={N_DOE}, n_iter={N_ITER}")
    print(f"   DoE method = lhs_maximin (held constant)")
    print("=" * 72)

    rows: list[dict] = []
    history_rows: list[pd.DataFrame] = []
    t0 = time.time()
    for name, factory in CONFIGS.items():
        for seed in SEEDS:
            tic = time.time()
            res = run_one(name, factory, seed)
            elapsed = time.time() - tic
            rows.append({"config": name, "seed": seed,
                         "best_y": res["best_y"], "secs": elapsed})
            h = res["history"].copy()
            h["config"] = name
            h["seed"] = seed
            h["trial"] = range(1, len(h) + 1)
            history_rows.append(h)
            print(f"  {name:<36} seed={seed}  "
                  f"best={res['best_y']:6.2f}  ({elapsed:4.1f}s)")
    total = (time.time() - t0) / 60.0
    print(f"\n  Total: {total:.1f} min")

    summary = pd.DataFrame(rows)
    agg = (summary.groupby("config")["best_y"]
                .agg(median="median", best="max", worst="min", std="std")
                .round(3)
                .sort_values("median", ascending=False))
    worst_overall = agg["median"].min()
    agg["vs worst"] = (agg["median"] - worst_overall).round(3)
    print("\n" + "─" * 72)
    print(" Summary (best yield across {} seeds):".format(len(SEEDS)))
    print("─" * 72)
    print(agg.to_string())

    print("\n Interpretation (refer to AGENT_KNOWLEDGE.md §6b for the 5-category framework):")
    print("   ① Correct domain knowledge — best ceiling in high-D, second in 4D")
    print("   ② Pure regularization      — safest default, +50–77% across 2D/4D/6D")
    print("   ③ Weak GP prior            — stable middle, modest gain")
    print("   ④ Mean fn only             — dimension-sensitive; may underperform")
    print("   ⑤ Mono + prior (rescued)   — depends on auto-rescued ε (see v0.3)")

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    full = pd.concat(history_rows, axis=0, ignore_index=True)
    full.to_csv(out_dir / "experiment_02_knowledge.csv", index=False)
    agg.to_csv(out_dir / "experiment_02_summary.csv")
    print(f"\nWrote:")
    print(f"  {out_dir / 'experiment_02_knowledge.csv'}  ({len(full)} rows)")
    print(f"  {out_dir / 'experiment_02_summary.csv'}")


if __name__ == "__main__":
    main()
