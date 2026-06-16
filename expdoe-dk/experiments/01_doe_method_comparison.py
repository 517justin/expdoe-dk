"""
Experiment 01 — How much does the DoE method matter for BO?

Question
--------
Holding the knowledge configuration and BO budget fixed, how much does
the choice of initial-design method affect convergence to the optimum?

Setup
-----
Uses the canonical objectives from the sister project's empirical studies
(Exp-7 / Exp-9 / Exp-10 v2 in ``AGENT_KNOWLEDGE.md``), so the result is
comparable to ``02_knowledge_comparison.py`` (same oracles, same budgets,
same noise-free gap metric — only the experimental variable differs).

  --dim 2  → reaction_objective_2d  (n_doe=6,  n_iter=15)
  --dim 4  → process_objective_4d   (n_doe=12, n_iter=30)
  --dim 6  → process_objective_6d_v2 (n_doe=18, n_iter=30)

DoE methods compared:
  lhs_maximin / lhs_random / sobol / halton / random_uniform / d_optimal

No injected knowledge: every method runs with the Campaign's auto Cat ②
default (``with_random_augment(n=20)``), so the comparison is fair.

Headline metric
---------------
``gap_final`` = true_opt − noiseless_oracle(best_x_final).
Smaller is better. Robust to lucky / unlucky noise draws.

Run with:
    python experiments/01_doe_method_comparison.py                # 2D, ~3 min
    python experiments/01_doe_method_comparison.py --dim 4        # ~12 min
    python experiments/01_doe_method_comparison.py --dim 6        # ~17 min
    python experiments/01_doe_method_comparison.py --seeds 3
"""
from __future__ import annotations

import argparse
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


METHODS = [
    "lhs_maximin",
    "lhs_random",
    "sobol",
    "halton",
    "random_uniform",
    "d_optimal",
]


def run_one(spec, method: str, seed: int) -> dict:
    torch.manual_seed(seed)
    campaign = ed.Campaign(spec.space, knowledge=None, seed=seed,
                           validate=False)
    doe = campaign.suggest_doe(n=spec.n_doe, method=method)
    campaign.tell(doe, spec.oracle(doe, noise_seed=seed))
    for it in range(spec.n_iter):
        nxt = campaign.ask(q=1, iteration=it)
        campaign.tell(nxt, spec.oracle(nxt, noise_seed=seed * 1000 + it))
    res = campaign.finalize()
    history = res.history_df

    # Noise-free trajectory: re-evaluate oracle without noise at each best-so-far X.
    param_cols = spec.space.param_names
    best_idx = 0
    best_x_rows: list[dict] = []
    for i in range(len(history)):
        if history["y"].iloc[i] > history["y"].iloc[best_idx]:
            best_idx = i
        best_x_rows.append(
            {c: float(history.iloc[best_idx][c]) for c in param_cols}
        )
    clean = np.asarray(spec.noiseless(pd.DataFrame(best_x_rows)), dtype=float)
    return {"clean": clean.tolist(), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=[2, 4, 6], default=2,
                        help="Problem dimensionality. Default 2.")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds. Default 5.")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    spec = make_problem(args.dim)
    seeds = [42 + i for i in range(args.seeds)]
    total_evals = spec.n_doe + spec.n_iter
    target = spec.true_opt * 0.95

    print("=" * 80)
    print(f" Experiment 01 — DoE method × BO convergence on {args.dim}D")
    print(f"   oracle      = "
          f"{'reaction_objective_2d' if args.dim == 2 else f'process_objective_{args.dim}d'}"
          f"{'_v2' if args.dim == 6 else ''}")
    print(f"   true_opt    = {spec.true_opt:.4f}  (target = 95 % ≈ {target:.4f})")
    print(f"   n_doe       = {spec.n_doe}   n_iter = {spec.n_iter}   "
          f"(total {total_evals} evals)")
    print(f"   seeds       = {seeds}")
    print(f"   methods     = {METHODS}")
    print("=" * 80)

    rows: list[dict] = []
    traces: list[pd.DataFrame] = []
    t0 = time.time()
    for method in METHODS:
        for seed in seeds:
            tic = time.time()
            try:
                r = run_one(spec, method, seed)
                clean = np.asarray(r["clean"], dtype=float)
            except Exception as exc:
                print(f"  {method:<15} seed={seed}  FAILED: {exc}")
                clean = np.full(total_evals, np.nan)
            elapsed = time.time() - tic

            gap = spec.true_opt - clean
            reached = [i for i, v in enumerate(clean) if v >= target]
            trials_to_target = reached[0] + 1 if reached else None
            rows.append({
                "method": method, "seed": seed,
                "clean_final": float(clean[-1]) if len(clean) else float("nan"),
                "gap_final": float(gap[-1]) if len(gap) else float("nan"),
                "clean@doe_end": (float(clean[spec.n_doe - 1])
                                  if len(clean) >= spec.n_doe else float("nan")),
                "trials_to_95pct": trials_to_target,
                "secs": elapsed,
            })
            traces.append(pd.DataFrame({
                "method": method, "seed": seed,
                "trial": np.arange(1, len(clean) + 1),
                "clean_best_so_far": clean,
            }))
            ttg_str = str(trials_to_target) if trials_to_target else "—"
            print(f"  {method:<15} seed={seed}  "
                  f"clean={clean[-1]:.4f}  gap={gap[-1]:.4f}  "
                  f"hit95@{ttg_str:>4}  ({elapsed:4.1f}s)")
    total = (time.time() - t0) / 60.0
    print(f"\n  Total: {total:.1f} min")

    summary = pd.DataFrame(rows)
    agg_median = (
        summary.groupby("method")[["clean_final", "gap_final", "clean@doe_end"]]
        .median().round(4)
    )
    summary["_ttg"] = summary["trials_to_95pct"].fillna(total_evals + 1)
    agg_median["trials_to_95% (median)"] = (
        summary.groupby("method")["_ttg"].median().astype(int)
    )
    agg_median["%seeds_hit_95"] = (
        summary.groupby("method")["trials_to_95pct"]
        .apply(lambda s: int(round(100.0 * s.notna().mean())))
    )
    # Δ vs random_uniform (the natural worst-case baseline for DoE methods).
    if "random_uniform" in agg_median.index:
        baseline_gap = agg_median.loc["random_uniform", "gap_final"]
        agg_median["Δ gap vs random (%)"] = (
            100.0 * (baseline_gap - agg_median["gap_final"])
            / max(abs(baseline_gap), 1e-9)
        ).round(1)
    agg_median = agg_median.sort_values("gap_final", ascending=True)

    print("\n" + "─" * 80)
    print(f" Convergence summary (median over {len(seeds)} seeds; "
          f"target = 95 % of true opt ≈ {target:.4f})")
    print("─" * 80)
    print(agg_median.to_string())

    print("\n Interpretation:")
    print("   gap_final               — distance to true noiseless optimum (lower = better)")
    print("   clean@doe_end           — best yield after DoE only (before BO)")
    print("   trials_to_95% (median)  — speed; lower = faster")
    print("   %seeds_hit_95           — robustness across seeds")

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"experiment_01_doe_methods_{args.dim}d"
    summary.to_csv(out_dir / f"{stem}.csv", index=False)
    agg_median.to_csv(out_dir / f"{stem}_summary.csv")
    pd.concat(traces, ignore_index=True).to_csv(
        out_dir / f"{stem}_traces.csv", index=False,
    )
    print(f"\nWrote:")
    print(f"  {out_dir / f'{stem}.csv'}")
    print(f"  {out_dir / f'{stem}_summary.csv'}")
    print(f"  {out_dir / f'{stem}_traces.csv'}")


if __name__ == "__main__":
    main()
