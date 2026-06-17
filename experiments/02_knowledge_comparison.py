"""
Experiment 02 — How much does the *type* of domain knowledge matter?

The v1 of this experiment used a toy oracle whose optimum sat at the
corner of the search space, so every BO method reached it within budget
and knowledge configurations were indistinguishable. v2 rebuilds the
experiment around the canonical objectives from the sister project's
empirical studies (Exp-7 / Exp-9 / Exp-10 in `AGENT_KNOWLEDGE.md`):

  - `reaction_objective_2d`  — T × conc, the Plan-2 Exp-7 problem (2D)
  - `process_objective_4d`   — T × conc × pH × t, the Exp-9 problem (4D)
  - `process_objective_6d_v2`— 4D + polar (bimodal) + rpm (Gaussian peak)

These are the same objectives that produced the 5-category framework, so
running this script against them should reproduce the §6b pattern:

  ① domain knowledge — strong in 2D (+80%) and 6D (+91%), compressed in 4D
  ② regularization   — safest default across dimensions
  ⑤ mono + prior     — depends on v0.3 auto-rescue
  G  wrong direction — v0.2 Spearman validator warns; performance suffers

Run with:
    python experiments/02_knowledge_comparison.py             # 2D (default)
    python experiments/02_knowledge_comparison.py --dim 4
    python experiments/02_knowledge_comparison.py --dim 6
    python experiments/02_knowledge_comparison.py --seeds 3   # fewer seeds, faster
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

import expdoe_dk as ed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oracles import ProblemSpec, make_problem  # noqa: E402


# --------------------------------------------------------------------- #
# Per-dimension knowledge configurations
# --------------------------------------------------------------------- #
def _configs_2d() -> dict[str, Callable[[], ed.Knowledge]]:
    return {
        "A: baseline (auto Cat ②)": lambda: ed.Knowledge(),
        "②: random_augment only":   lambda: ed.Knowledge().strict().with_random_augment(n=20),
        "③: gp_prior only":         lambda: ed.Knowledge().strict().with_gp_prior(lengthscale="medium"),
        "①: full domain knowledge": lambda: (
            ed.Knowledge().strict()
            .with_arrhenius("T")
            .with_quadratic_peak("conc", center=0.5)
            .with_random_augment(n=20)
        ),
        "④: Arrhenius mean only":   lambda: ed.Knowledge().strict().with_arrhenius("T"),
        "⑤: mono + gp_prior (rescued)": lambda: (
            ed.Knowledge().strict()
            .with_monotone("T", effect="increases_objective", epsilon=0.02)
            .with_gp_prior(lengthscale="strong")
        ),
        "G: WRONG-direction monotone": lambda: (
            ed.Knowledge().strict()
            .with_monotone("T", effect="decreases_objective")
            .with_random_augment(n=20)
        ),
    }


def _configs_4d() -> dict[str, Callable[[], ed.Knowledge]]:
    return {
        "A: baseline (auto Cat ②)": lambda: ed.Knowledge(),
        "②: random_augment only":   lambda: ed.Knowledge().strict().with_random_augment(n=20),
        "③: gp_prior only":         lambda: ed.Knowledge().strict().with_gp_prior(lengthscale="medium"),
        "①: full domain knowledge": lambda: (
            ed.Knowledge().strict()
            .with_arrhenius("T")
            .with_quadratic_peak("conc", center=1.0)
            .with_quadratic_peak("pH",   center=7.0)
            .with_monotone("t", effect="increases_objective")
            .with_random_augment(n=20)
        ),
        "④: Arrhenius mean only":   lambda: ed.Knowledge().strict().with_arrhenius("T"),
        "⑤: mono + gp_prior (rescued)": lambda: (
            ed.Knowledge().strict()
            .with_monotone("T", effect="increases_objective", epsilon=0.02)
            .with_monotone("t", effect="increases_objective", epsilon=0.02)
            .with_gp_prior(lengthscale="strong")
        ),
        "G: WRONG-direction monotone": lambda: (
            ed.Knowledge().strict()
            .with_monotone("T", effect="decreases_objective")
            .with_monotone("t", effect="decreases_objective")
            .with_random_augment(n=20)
        ),
    }


def _configs_6d() -> dict[str, Callable[[], ed.Knowledge]]:
    return {
        "A: baseline (auto Cat ②)": lambda: ed.Knowledge(),
        "②: random_augment only":   lambda: ed.Knowledge().strict().with_random_augment(n=20),
        "③: gp_prior only":         lambda: ed.Knowledge().strict().with_gp_prior(lengthscale="medium"),
        "①: full domain knowledge": lambda: (
            ed.Knowledge().strict()
            .with_arrhenius("T")
            .with_quadratic_peak("conc", center=1.0)
            .with_quadratic_peak("pH",   center=7.0)
            .with_monotone("t", effect="increases_objective")
            .with_quadratic_peak("rpm",  center=700.0)
            .with_random_augment(n=20)
        ),
        "④: Arrhenius mean only":   lambda: ed.Knowledge().strict().with_arrhenius("T"),
        "⑤: mono + gp_prior (rescued)": lambda: (
            ed.Knowledge().strict()
            .with_monotone("T", effect="increases_objective", epsilon=0.02)
            .with_monotone("t", effect="increases_objective", epsilon=0.02)
            .with_gp_prior(lengthscale="strong")
        ),
        "G: WRONG-direction monotone": lambda: (
            ed.Knowledge().strict()
            .with_monotone("T", effect="decreases_objective")
            .with_monotone("t", effect="decreases_objective")
            .with_random_augment(n=20)
        ),
    }


_CONFIG_BUILDERS = {2: _configs_2d, 4: _configs_4d, 6: _configs_6d}


# --------------------------------------------------------------------- #
# Single run
# --------------------------------------------------------------------- #
def run_one(spec: ProblemSpec, name: str,
            knowledge_factory: Callable[[], ed.Knowledge],
            seed: int) -> dict:
    torch.manual_seed(seed)
    campaign = ed.Campaign(spec.space, knowledge_factory(), seed=seed,
                           validate=False)
    doe = campaign.suggest_doe(n=spec.n_doe, method="lhs_maximin")
    campaign.tell(doe, spec.oracle(doe, noise_seed=seed))
    for it in range(spec.n_iter):
        nxt = campaign.ask(q=1, iteration=it)
        campaign.tell(nxt, spec.oracle(nxt, noise_seed=seed * 1000 + it))
    res = campaign.finalize()
    history = res.history_df
    cum_best = history["y"].cummax().to_numpy()

    # Noise-free trajectory: for each iteration, evaluate the noiseless
    # oracle at the X corresponding to the best observed so far. This is
    # the standard BO-benchmark metric, robust to lucky noise draws.
    param_cols = spec.space.param_names
    best_x_so_far: list[dict] = []
    best_idx = 0
    for i in range(len(history)):
        if history["y"].iloc[i] > history["y"].iloc[best_idx]:
            best_idx = i
        best_x_so_far.append({c: float(history.iloc[best_idx][c])
                              for c in param_cols})
    noiseless_trace = spec.noiseless(pd.DataFrame(best_x_so_far))
    return {
        "best_y": float(cum_best[-1]),
        "cum_best": cum_best.tolist(),
        "noiseless_best_at_iter": noiseless_trace.tolist(),
        "history": history,
    }


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=[2, 4, 6], default=2,
                        help="Problem dimensionality (2, 4, or 6). Default 2.")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds (each starts from 42). Default 5.")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    spec = make_problem(args.dim)
    configs = _CONFIG_BUILDERS[args.dim]()
    seeds = [42 + i for i in range(args.seeds)]
    total_evals = spec.n_doe + spec.n_iter
    target = spec.true_opt * 0.95

    print("=" * 80)
    print(f" Experiment 02 — Knowledge comparison on {args.dim}D")
    print(f"   oracle      = "
          f"{'reaction_objective_2d' if args.dim == 2 else f'process_objective_{args.dim}d'}"
          f"{'_v2' if args.dim == 6 else ''}")
    print(f"   true_opt    = {spec.true_opt:.4f}  (target = 95 % ≈ {target:.4f})")
    print(f"   n_doe       = {spec.n_doe}   n_iter = {spec.n_iter}   "
          f"(total {total_evals} evals)")
    print(f"   seeds       = {seeds}")
    print(f"   configs     = {len(configs)}")
    print("=" * 80)

    rows: list[dict] = []
    traces: list[pd.DataFrame] = []
    t0 = time.time()
    for name, factory in configs.items():
        for seed in seeds:
            tic = time.time()
            try:
                r = run_one(spec, name, factory, seed)
                clean = np.asarray(r["noiseless_best_at_iter"], dtype=float)
            except Exception as exc:
                print(f"  {name:<36} seed={seed}  FAILED: {exc}")
                clean = np.full(total_evals, np.nan)
            elapsed = time.time() - tic

            # Gap from true noiseless optimum at each iteration.
            gap = spec.true_opt - clean
            # Convergence: first trial where the noise-free best is within
            # 5 % of the true optimum.
            reached = [i for i, v in enumerate(clean) if v >= target]
            trials_to_target = reached[0] + 1 if reached else None
            rows.append({
                "config": name, "seed": seed,
                "clean_final": float(clean[-1]) if len(clean) else float("nan"),
                "gap_final": float(gap[-1]) if len(gap) else float("nan"),
                "clean@doe_end": (float(clean[spec.n_doe - 1])
                                  if len(clean) >= spec.n_doe else float("nan")),
                "clean@mid": (float(clean[(spec.n_doe + spec.n_iter // 2) - 1])
                              if len(clean) >= spec.n_doe + spec.n_iter // 2
                              else float("nan")),
                "trials_to_95pct": trials_to_target,
                "secs": elapsed,
            })
            traces.append(pd.DataFrame({
                "config": name, "seed": seed,
                "trial": np.arange(1, len(clean) + 1),
                "clean_best_so_far": clean,
            }))
            ttg_str = str(trials_to_target) if trials_to_target else "—"
            final_val = clean[-1] if len(clean) else float("nan")
            print(f"  {name:<36} seed={seed}  "
                  f"clean={final_val:.4f}  gap={gap[-1] if len(gap) else float('nan'):.4f}  "
                  f"hit95@{ttg_str:>4}  ({elapsed:4.1f}s)")
    total = (time.time() - t0) / 60.0
    print(f"\n  Total: {total:.1f} min")

    summary = pd.DataFrame(rows)
    agg_median = (
        summary.groupby("config")[["clean_final", "gap_final",
                                   "clean@doe_end", "clean@mid"]]
        .median().round(4)
    )
    summary["_ttg"] = summary["trials_to_95pct"].fillna(total_evals + 1)
    agg_median["trials_to_95% (median)"] = (
        summary.groupby("config")["_ttg"].median().astype(int)
    )
    agg_median["%seeds_hit_95"] = (
        summary.groupby("config")["trials_to_95pct"]
        .apply(lambda s: int(round(100.0 * s.notna().mean())))
    )
    # Improvement vs baseline: relative reduction in gap_final
    # (smaller gap = better; same convention as AGENT_KNOWLEDGE.md §6b).
    baseline_gap = agg_median.loc["A: baseline (auto Cat ②)", "gap_final"]
    agg_median["Δ gap vs baseline (%)"] = (
        100.0 * (baseline_gap - agg_median["gap_final"])
        / max(abs(baseline_gap), 1e-9)
    ).round(1)
    agg_median = agg_median.sort_values("gap_final", ascending=True)

    print("\n" + "─" * 80)
    print(f" Convergence summary (median over {len(seeds)} seeds; "
          f"target = 95 % of true opt ≈ {target:.4f})")
    print("─" * 80)
    print(agg_median.to_string())

    print(f"\n Per AGENT_KNOWLEDGE.md §6b on this oracle:")
    if args.dim == 2:
        print("   ① full domain knowledge should lead by ~5× (+80% in Exp-7)")
    elif args.dim == 4:
        print("   ① / ② are roughly tied (+25-77% range in Exp-10)")
        print("   ② regularization typically wins on speed; ⑤ on stability")
    else:  # 6
        print("   ① / C: Frozen Combined dominate (+91% in Exp-10 6D v2)")
        print("   ② still strong, but knowledge advantage is clearest here")
    print("   G (wrong direction) should be slowest and least reliable")

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"experiment_02_knowledge_{args.dim}d"
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
