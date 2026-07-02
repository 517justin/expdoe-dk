"""
Experiment 02 - knowledge comparison for simulation_data2.

Compares process-knowledge configurations on the constrained, discrete
simulation_data2 problems. The initial DoE method can be selected explicitly
or inferred from Experiment 01's summary via ``--doe-method auto``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

import expdoe_dk as ed

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _oracles import ProblemSpec, make_problem  # noqa: E402


def _load_experiment_01():
    path = THIS_DIR / "01_doe_method_comparison.py"
    spec = importlib.util.spec_from_file_location("simulation_data2_exp01", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Experiment 01 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXP01 = _load_experiment_01()
clean_best_trace = _EXP01.clean_best_trace
lab_metrics = _EXP01.lab_metrics
summarize = _EXP01.summarize


METHODS = [
    "lhs_maximin",
    "lhs_random",
    "sobol",
    "halton",
    "random_uniform",
    "d_optimal",
]

BASELINE_LABEL = "A: baseline (plain GP)"


def resolve_doe_method(requested: str, *, dim: int, out_dir: Path) -> str:
    """Resolve a manual or auto-selected DoE method."""
    if requested != "auto":
        if requested not in METHODS:
            raise ValueError(
                f"Unknown DoE method {requested!r}. Choose one of {METHODS} or 'auto'."
            )
        return requested

    summary_path = out_dir / f"experiment_01_doe_methods_{dim}d_summary.csv"
    try:
        summary = pd.read_csv(summary_path, index_col=0)
        gaps = pd.to_numeric(summary["gap_final"], errors="coerce")
        if gaps.dropna().empty:
            raise ValueError("No numeric gap_final values found")
        selected = str(gaps.idxmin())
        if selected not in METHODS:
            raise ValueError(f"Selected method {selected!r} is not supported")
        print(f"Auto-selected DoE method from {summary_path}: {selected}")
        return selected
    except Exception as exc:
        print(
            f"WARNING: could not auto-select DoE method from {summary_path}: "
            f"{exc}; falling back to sobol."
        )
        return "sobol"


def _process_knowledge() -> ed.Knowledge:
    return (
        ed.Knowledge()
        .with_quadratic_peak("pH", center=7.0)
        .with_quadratic_peak("catalyst", center=2.5)
        .with_quadratic_peak("time", center=130.0)
    )


def _partial_process_knowledge() -> ed.Knowledge:
    return (
        ed.Knowledge()
        .with_quadratic_peak("pH", center=7.0)
        .with_quadratic_peak("catalyst", center=2.5)
    )


def _wrong_process_knowledge() -> ed.Knowledge:
    return (
        ed.Knowledge()
        .with_quadratic_peak("pH", center=5.0)
        .with_monotone("time", effect="decreases_objective")
    )


def _full_mixed_4d() -> ed.Knowledge:
    return _process_knowledge().with_gp_prior(lengthscale="medium")


def _full_mixed_6d() -> ed.Knowledge:
    return (
        _full_mixed_4d()
        .with_quadratic_peak("solvent_A", center=55.0)
        .with_quadratic_peak("additive", center=5.0)
    )


def _configs_4d() -> dict[str, Callable[[], ed.Knowledge]]:
    return {
        BASELINE_LABEL: lambda: ed.Knowledge(),
        "B: random_augment only": lambda: ed.Knowledge().with_random_augment(n=20),
        "C: gp_prior only": lambda: ed.Knowledge().with_gp_prior(lengthscale="medium"),
        "D: process knowledge": _process_knowledge,
        "E: partial process knowledge": _partial_process_knowledge,
        "F: wrong process knowledge": _wrong_process_knowledge,
        "G: full mixed knowledge": _full_mixed_4d,
    }


def _configs_6d() -> dict[str, Callable[[], ed.Knowledge]]:
    configs = _configs_4d()
    configs["G: full mixed knowledge"] = _full_mixed_6d
    return configs


_CONFIG_BUILDERS = {4: _configs_4d, 6: _configs_6d}


def run_one(
    spec: ProblemSpec,
    knowledge_factory: Callable[[], ed.Knowledge],
    *,
    seed: int,
    doe_method: str,
) -> dict[str, object]:
    torch.manual_seed(seed)
    campaign = ed.Campaign(spec.space, knowledge_factory(), seed=seed, validate=False)
    noise_rng = np.random.default_rng(seed)

    doe = campaign.suggest_doe(n=spec.n_doe, method=doe_method)
    campaign.tell(doe, spec.oracle(doe, rng=noise_rng))
    for it in range(spec.n_iter):
        nxt = campaign.ask(q=1, iteration=it)
        campaign.tell(nxt, spec.oracle(nxt, rng=noise_rng))

    result = campaign.finalize()
    history = result.history_df
    return {
        "clean": clean_best_trace(spec, history),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=[4, 6], default=4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--doe-method", choices=["auto", *METHODS], default="auto")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    spec = make_problem(args.dim)
    out_dir = THIS_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    doe_method = resolve_doe_method(args.doe_method, dim=args.dim, out_dir=out_dir)
    configs = _CONFIG_BUILDERS[args.dim]()
    seeds = [42 + i for i in range(args.seeds)]
    total_evals = spec.n_doe + spec.n_iter
    target = spec.true_opt * 0.95
    mid_idx = (spec.n_doe + spec.n_iter // 2) - 1

    print("=" * 80)
    print(f" Experiment 02 - knowledge comparison on simulation_data2 {args.dim}D")
    print(f"   true_opt    = {spec.true_opt:.4f}  (target = 95% = {target:.4f})")
    print(f"   n_doe       = {spec.n_doe}   n_iter = {spec.n_iter}   total = {total_evals}")
    print(f"   seeds       = {seeds}")
    print(f"   doe_method  = {doe_method}")
    print(f"   configs     = {list(configs)}")
    print("=" * 80)

    rows: list[dict[str, object]] = []
    traces: list[pd.DataFrame] = []
    started = time.time()
    for config, factory in configs.items():
        for seed in seeds:
            tic = time.time()
            error = ""
            status = "OK"
            try:
                run = run_one(spec, factory, seed=seed, doe_method=doe_method)
                clean = np.asarray(run["clean"], dtype=float)
                metrics = dict(run["lab_metrics"])
            except Exception as exc:
                status = "ERROR"
                error = str(exc)
                clean = np.full(total_evals, np.nan)
                metrics = _failure_lab_metrics(total_evals)
                print(f"  {config:<30} seed={seed}  FAILED: {error}")
            elapsed = time.time() - tic

            gap = spec.true_opt - clean
            reached = np.flatnonzero(clean >= target)
            trials_to_target = int(reached[0] + 1) if len(reached) else None
            rows.append(
                {
                    "config": config,
                    "seed": seed,
                    "doe_method": doe_method,
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
                    "total_evals": total_evals,
                    "secs": elapsed,
                    **metrics,
                }
            )
            traces.append(
                pd.DataFrame(
                    {
                        "config": config,
                        "seed": seed,
                        "doe_method": doe_method,
                        "trial": np.arange(1, len(clean) + 1),
                        "clean_best_so_far": clean,
                    }
                )
            )
            ttg = str(trials_to_target) if trials_to_target else "-"
            final = clean[-1] if len(clean) else float("nan")
            final_gap = gap[-1] if len(gap) else float("nan")
            print(
                f"  {config:<30} seed={seed}  status={status:<5}  "
                f"clean={final:.4f}  gap={final_gap:.4f}  hit95@{ttg:>3}  "
                f"({elapsed:4.1f}s)"
            )

    print(f"\n  Total: {(time.time() - started) / 60.0:.1f} min")

    raw = pd.DataFrame(rows)
    summary = summarize(raw, "config", baseline_label=BASELINE_LABEL)

    print("\n" + "-" * 80)
    print(f" Summary (median over {len(seeds)} seed(s))")
    print("-" * 80)
    print(summary.to_string())

    stem = f"experiment_02_knowledge_{args.dim}d"
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
