"""
Ablation for the ① over-specification hypothesis (4D).

The exp-02 README rewrite shows ① "full domain knowledge" (4 mean items +
monotone + random_augment) UNDERPERFORMS a plain GP in 4D and 6D — at
odds with AGENT_KNOWLEDGE.md §6b which reports C: Frozen Combined and
D_correct beating the plain-GP baseline by +25~28 % in 4D.

The hypothesis: my new ① stacks far more knowledge items than the OLD
winning configs:

  OLD "C: Frozen Combined" — 3 frozen means only:
    ArrheniusMeanFrozen() + QuadraticMeanFrozen(conc peak) + QuadraticMeanFrozen(pH peak)
    no monotone, no random_augment

  OLD "D_correct" — 2 monotones only:
    with_monotone(T, decreases_objective_in_Y_norm)
    with_monotone(t, decreases_objective_in_Y_norm)
    no mean function, no random_augment

  NEW ① "full domain knowledge" — everything stacked:
    with_arrhenius(T) + with_quadratic_peak(conc) + with_quadratic_peak(pH)
    + with_monotone(t) + with_random_augment(20)

This ablation runs each piece separately to localise the regression.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import expdoe_dk as ed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oracles import make_problem  # noqa: E402


CONFIGS = {
    "plain GP":                              lambda: ed.Knowledge(),
    # OLD canonical winners
    "old C: Frozen Combined (3 means)":      lambda: (
        ed.Knowledge()
        .with_arrhenius("T")
        .with_quadratic_peak("conc", center=1.0)
        .with_quadratic_peak("pH", center=7.0)
    ),
    "old D_correct (2 monotones)":           lambda: (
        ed.Knowledge()
        .with_monotone("T", effect="increases_objective")
        .with_monotone("t", effect="increases_objective")
    ),
    # NEW ① decomposed
    "arrhenius only":                        lambda: ed.Knowledge().with_arrhenius("T"),
    "2 peaks (conc + pH)":                   lambda: (
        ed.Knowledge()
        .with_quadratic_peak("conc", center=1.0)
        .with_quadratic_peak("pH", center=7.0)
    ),
    "monotone(t) only":                      lambda: ed.Knowledge().with_monotone("t", effect="increases_objective"),
    "Frozen Combined + monotone(t)":         lambda: (
        ed.Knowledge()
        .with_arrhenius("T")
        .with_quadratic_peak("conc", center=1.0)
        .with_quadratic_peak("pH", center=7.0)
        .with_monotone("t", effect="increases_objective")
    ),
    "Frozen Combined + random_augment(20)":  lambda: (
        ed.Knowledge()
        .with_arrhenius("T")
        .with_quadratic_peak("conc", center=1.0)
        .with_quadratic_peak("pH", center=7.0)
        .with_random_augment(n=20)
    ),
    "NEW ① full (everything)":               lambda: (
        ed.Knowledge()
        .with_arrhenius("T")
        .with_quadratic_peak("conc", center=1.0)
        .with_quadratic_peak("pH", center=7.0)
        .with_monotone("t", effect="increases_objective")
        .with_random_augment(n=20)
    ),
}

SEEDS = [42, 43, 44, 45, 46]


def run_one(spec, factory, seed):
    torch.manual_seed(seed)
    campaign = ed.Campaign(spec.space, factory(), seed=seed, validate=False)
    doe = campaign.suggest_doe(n=spec.n_doe, method="lhs_maximin")
    campaign.tell(doe, spec.oracle(doe, noise_seed=seed))
    for it in range(spec.n_iter):
        nxt = campaign.ask(q=1, iteration=it)
        campaign.tell(nxt, spec.oracle(nxt, noise_seed=seed * 1000 + it))
    res = campaign.finalize()
    history = res.history_df
    cols = spec.space.param_names
    best_idx = 0
    best_rows = []
    for i in range(len(history)):
        if history["y"].iloc[i] > history["y"].iloc[best_idx]:
            best_idx = i
        best_rows.append({c: float(history.iloc[best_idx][c]) for c in cols})
    clean = np.asarray(spec.noiseless(pd.DataFrame(best_rows)), dtype=float)
    return clean


def main():
    warnings.simplefilter("ignore")
    spec = make_problem(4)
    print(f"Ablation: 4D process_objective_4d, true_opt = {spec.true_opt:.5f}")
    print(f"Seeds = {SEEDS}, n_doe={spec.n_doe}, n_iter={spec.n_iter}")
    print("=" * 80)

    rows = []
    for name, factory in CONFIGS.items():
        gaps = []
        t0 = time.time()
        for seed in SEEDS:
            try:
                clean = run_one(spec, factory, seed)
                gaps.append(spec.true_opt - float(clean[-1]))
            except Exception as e:
                print(f"  {name!r} seed={seed} FAILED: {e}")
                gaps.append(float("nan"))
        elapsed = time.time() - t0
        gap_med = float(np.median(gaps))
        gap_max = float(np.max(gaps))
        print(f"  {name:<42}  gap_med={gap_med:.4f}  gap_worst={gap_max:.4f}  ({elapsed:5.1f}s)")
        rows.append({"config": name, "gap_med": gap_med, "gap_worst": gap_max})

    df = pd.DataFrame(rows).sort_values("gap_med")
    baseline = df.set_index("config").loc["plain GP", "gap_med"]
    df["Δ vs plain GP (%)"] = (100.0 * (baseline - df["gap_med"]) / max(abs(baseline), 1e-9)).round(1)
    print("\n" + "─" * 80)
    print(" Summary (median gap, lower = better)")
    print("─" * 80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
