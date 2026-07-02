"""
Experiment 01 — DoE method comparison for simulation_data2.

Compares initial design methods on the constrained, discrete lab-like
simulation_data2 problems. Each run uses a plain GP after the initial DoE;
the experimental variable is only the initial design method.
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
from _oracles import ProblemSpec, make_problem  # noqa: E402


METHODS = [
    "lhs_maximin",
    "lhs_random",
    "sobol",
    "halton",
    "random_uniform",
    "d_optimal",
]


def _row_key(row: pd.Series, names: list[str]) -> tuple[float, ...]:
    return tuple(round(float(row[name]), 12) for name in names)


def lab_metrics(space: ed.Space, history: pd.DataFrame) -> dict[str, object]:
    """Return feasibility, grid, and duplicate diagnostics for lab validity."""
    param_cols = space.param_names
    if history.empty:
        return {
            "constraint_violations": 0,
            "step_grid_violations": 0,
            "duplicate_rows": 0,
            "all_rows_feasible": True,
            "all_rows_on_grid": True,
        }

    X = history[param_cols].copy()
    feasible = space.feasibility_mask(X).cpu().numpy().astype(bool)
    constraint_violations = int((~feasible).sum())

    grid_violation_mask = np.zeros(len(X), dtype=bool)
    for param in space.params:
        values = X[param.name].to_numpy(dtype=float)
        lo, hi = param.bounds
        on_grid = (values >= lo - 1e-9) & (values <= hi + 1e-9)
        if param.kind == "discrete":
            assert param.step is not None
            offsets = (values - lo) / param.step
            on_grid &= np.isclose(offsets, np.round(offsets), atol=1e-8, rtol=0.0)
        grid_violation_mask |= ~on_grid

    duplicate_rows = int(X.duplicated(keep="first").sum())
    return {
        "constraint_violations": constraint_violations,
        "step_grid_violations": int(grid_violation_mask.sum()),
        "duplicate_rows": duplicate_rows,
        "all_rows_feasible": bool(constraint_violations == 0),
        "all_rows_on_grid": bool(not grid_violation_mask.any()),
    }


def _clean_best_trace(spec: ProblemSpec, history: pd.DataFrame) -> np.ndarray:
    """Re-evaluate the best observed X so far with the noiseless oracle."""
    param_cols = spec.space.param_names
    best_idx = 0
    best_x_rows: list[dict[str, float]] = []
    for i in range(len(history)):
        if history["y"].iloc[i] > history["y"].iloc[best_idx]:
            best_idx = i
        best_x_rows.append(
            {col: float(history.iloc[best_idx][col]) for col in param_cols}
        )
    return np.asarray(spec.noiseless(pd.DataFrame(best_x_rows)), dtype=float)


def run_one(spec: ProblemSpec, method: str, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    campaign = ed.Campaign(spec.space, knowledge=None, seed=seed, validate=False)
    noise_rng = np.random.default_rng(seed)

    doe = campaign.suggest_doe(n=spec.n_doe, method=method)
    campaign.tell(doe, spec.oracle(doe, rng=noise_rng))
    for it in range(spec.n_iter):
        nxt = campaign.ask(q=1, iteration=it)
        campaign.tell(nxt, spec.oracle(nxt, rng=noise_rng))

    result = campaign.finalize()
    history = result.history_df
    return {
        "clean": _clean_best_trace(spec, history),
        "history": history,
        "lab_metrics": lab_metrics(spec.space, history),
    }


def _failure_lab_metrics(total_evals: int) -> dict[str, object]:
    return {
        "constraint_violations": total_evals,
        "step_grid_violations": total_evals,
        "duplicate_rows": np.nan,
        "all_rows_feasible": False,
        "all_rows_on_grid": False,
    }


def summarize(raw: pd.DataFrame, total_evals: int) -> pd.DataFrame:
    summary = (
        raw.groupby("method")[["clean_final", "gap_final", "clean@doe_end", "clean@mid"]]
        .median()
        .round(4)
    )
    raw = raw.copy()
    raw["_ttg"] = raw["trials_to_95pct"].fillna(total_evals + 1)
    summary["trials_to_95% median"] = raw.groupby("method")["_ttg"].median().astype(int)
    summary["% seeds hit 95"] = (
        raw.groupby("method")["trials_to_95pct"]
        .apply(lambda s: round(100.0 * s.notna().mean(), 1))
    )
    summary["% runs feasible"] = (
        raw.groupby("method")["all_rows_feasible"].apply(lambda s: round(100.0 * s.mean(), 1))
    )
    summary["% runs on grid"] = (
        raw.groupby("method")["all_rows_on_grid"].apply(lambda s: round(100.0 * s.mean(), 1))
    )
    summary["duplicate median"] = raw.groupby("method")["duplicate_rows"].median()

    if "random_uniform" in summary.index:
        baseline_gap = float(summary.loc["random_uniform", "gap_final"])
        if abs(baseline_gap) < 1e-4:
            summary["delta gap vs random_uniform"] = np.nan
        else:
            summary["delta gap vs random_uniform"] = (
                100.0 * (baseline_gap - summary["gap_final"]) / abs(baseline_gap)
            ).round(1)
    else:
        summary["delta gap vs random_uniform"] = np.nan

    return summary.sort_values("gap_final", ascending=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=[4, 6], default=4)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    spec = make_problem(args.dim)
    seeds = [42 + i for i in range(args.seeds)]
    total_evals = spec.n_doe + spec.n_iter
    target = spec.true_opt * 0.95
    mid_idx = (spec.n_doe + spec.n_iter // 2) - 1

    print("=" * 80)
    print(f" Experiment 01 — DoE method comparison on simulation_data2 {args.dim}D")
    print(f"   true_opt    = {spec.true_opt:.4f}  (target = 95% = {target:.4f})")
    print(f"   n_doe       = {spec.n_doe}   n_iter = {spec.n_iter}   total = {total_evals}")
    print(f"   seeds       = {seeds}")
    print(f"   methods     = {METHODS}")
    print("=" * 80)

    rows: list[dict[str, object]] = []
    traces: list[pd.DataFrame] = []
    started = time.time()
    for method in METHODS:
        for seed in seeds:
            tic = time.time()
            error = ""
            status = "OK"
            try:
                run = run_one(spec, method, seed)
                clean = np.asarray(run["clean"], dtype=float)
                metrics = dict(run["lab_metrics"])
            except Exception as exc:
                status = "ERROR"
                error = str(exc)
                clean = np.full(total_evals, np.nan)
                metrics = _failure_lab_metrics(total_evals)
                print(f"  {method:<15} seed={seed}  FAILED: {error}")
            elapsed = time.time() - tic

            gap = spec.true_opt - clean
            reached = np.flatnonzero(clean >= target)
            trials_to_target = int(reached[0] + 1) if len(reached) else None
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "status": status,
                    "error": error,
                    "clean_final": float(clean[-1]) if len(clean) else float("nan"),
                    "gap_final": float(gap[-1]) if len(gap) else float("nan"),
                    "clean@doe_end": (
                        float(clean[spec.n_doe - 1])
                        if len(clean) >= spec.n_doe
                        else float("nan")
                    ),
                    "clean@mid": (
                        float(clean[mid_idx]) if len(clean) > mid_idx else float("nan")
                    ),
                    "trials_to_95pct": trials_to_target,
                    "secs": elapsed,
                    **metrics,
                }
            )
            traces.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "seed": seed,
                        "trial": np.arange(1, len(clean) + 1),
                        "clean_best_so_far": clean,
                    }
                )
            )
            ttg = str(trials_to_target) if trials_to_target else "-"
            final = clean[-1] if len(clean) else float("nan")
            final_gap = gap[-1] if len(gap) else float("nan")
            print(
                f"  {method:<15} seed={seed}  status={status:<5}  "
                f"clean={final:.4f}  gap={final_gap:.4f}  hit95@{ttg:>3}  "
                f"({elapsed:4.1f}s)"
            )

    print(f"\n  Total: {(time.time() - started) / 60.0:.1f} min")

    raw = pd.DataFrame(rows)
    summary = summarize(raw, total_evals)

    print("\n" + "-" * 80)
    print(f" Summary (median over {len(seeds)} seed(s))")
    print("-" * 80)
    print(summary.to_string())

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"experiment_01_doe_methods_{args.dim}d"
    raw_path = out_dir / f"{stem}.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    traces_path = out_dir / f"{stem}_traces.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path)
    pd.concat(traces, ignore_index=True).to_csv(traces_path, index=False)

    print("\nWrote:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {traces_path}")


if __name__ == "__main__":
    main()
