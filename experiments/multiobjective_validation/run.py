"""Reproducible, fixed-budget acceptance, not a general optimizer ranking."""
from __future__ import annotations

import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import warnings

import numpy as np
import pandas as pd

from expdoe_dk import Campaign, ObservationBatch
from expdoe_dk.bo.multiobjective_metrics import hypervolume, hypervolume_regret
from . import fixtures as f

ROOT = Path(__file__).resolve().parents[2]


def provenance():
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for acceptance provenance")
    def read(*args):
        return subprocess.check_output([git, *args], cwd=ROOT, text=True).strip()
    status = read("status", "--porcelain")
    return {"source_commit": read("rev-parse", "HEAD"), "dirty": bool(status),
            "git_status": status, "python": platform.python_version(),
            "platform": platform.platform(), "dependencies": {name: version(name) for name in
            ("expdoe-dk", "torch", "botorch", "gpytorch", "ax-platform", "numpy", "scipy", "pandas")}}


def quality_summary(rows):
    if {row["method"] for row in rows} != set(f.METHODS):
        raise ValueError("Quality summary requires exactly the four declared methods")
    if any(not np.isfinite(row["hv_regret"]) or row["hv_regret"] < 0 for row in rows):
        raise ValueError("Quality summary requires finite nonnegative regret")
    means = {method: float(np.mean([row["hv_regret"] for row in rows if row["method"] == method]))
             for method in f.METHODS}
    gates = {method + "_vs_random": means[method] <= 1.10 * max(means["random"], 1e-12)
             for method in ("plain_gp", "knowledge_gp_v2")}
    gates["knowledge_gp_v2_vs_plain"] = means["knowledge_gp_v2"] <= 1.10 * max(means["plain_gp"], 1e-12)
    legacy = {"knowledge_gp_vs_random": means["knowledge_gp"] <= 1.10 * max(means["random"], 1e-12),
              "knowledge_gp_vs_plain": means["knowledge_gp"] <= 1.10 * max(means["plain_gp"], 1e-12)}
    return {"protocol": f.PROTOCOL, "protocol_version": f.PROTOCOL_VERSION,
            "mean_hv_regret": means, "tolerance": 1.10, "gates": gates, "passed": all(gates.values()),
            "legacy_v1": {"gates": legacy, "passed": all(legacy.values())}}


def _trajectory(seed, method, geometry, *, restart=None):
    rng = np.random.default_rng(seed)
    campaign = Campaign(f.space(), f.knowledge(method), seed=seed, validate=False, multiobjective=f.config())
    campaign.tell_batch(f.initial_batch(seed))
    batches, routes, checkpoint = [], [], None
    start = 0
    if restart is not None:
        campaign = Campaign.from_state(restart["campaign"])
        rng.bit_generator.state = restart["rng"]
        batches, routes = copy.deepcopy(restart["batches"]), list(restart["routes"])
        start = len(batches)
    catalog = f.catalog()
    ids = tuple(f"c{i:03}" for i in range(len(catalog)))
    for iteration in range(start, f.ROUNDS):
        if method == "random":
            seen = set(campaign.history_df().x)
            eligible = catalog.loc[(catalog.x <= .8) & ~catalog.x.isin(seen)]
            X = eligible.iloc[rng.choice(len(eligible), f.Q, replace=False)].reset_index(drop=True)
            route = "random_without_replacement"
        else:
            result = campaign.ask_with_diagnostics(q=f.Q, candidates=catalog, candidate_ids=ids)
            X, route = result.frame, result.diagnostics["acquisition_kind"]
            assert result.diagnostics["geometry"] == geometry.to_dict()
            assert route == "q_log_ehvi"
        assert len(X) == f.Q and X.x.is_unique and (X.x <= .8).all()
        assert not set(X.x) & set(campaign.history_df().x)
        campaign.tell_batch(ObservationBatch(X, f.measure(X), ids=tuple(f"round-{iteration}-{i}" for i in range(f.Q))))
        batches.append(X.x.tolist())
        routes.append(route)
        if iteration == 3:
            checkpoint = json.loads(json.dumps({"campaign": campaign.to_state(),
                "rng": rng.bit_generator.state, "batches": batches, "routes": routes}, allow_nan=False))
    return campaign, batches, routes, checkpoint


def run_matrix(*, seeds, output_dir) -> list[dict]:
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = provenance()
    rows = []
    with (output_dir / "matrix.jsonl").open("w") as stream:
        for seed in seeds:
            geometry = f.geometry(seed)
            truth = f.measure(f.catalog().query("x <= .8")).to_numpy()
            oracle_hv = hypervolume((truth - geometry.location) / geometry.scale, geometry.report_reference)
            for method in f.METHODS:
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter("always")
                    campaign, batches, routes, checkpoint = _trajectory(seed, method, geometry)
                    replay, replay_batches, _, _ = _trajectory(seed, method, geometry)
                    resumed, resumed_batches, _, _ = _trajectory(seed, method, geometry, restart=checkpoint)
                history = campaign.history_df()
                pd.testing.assert_frame_equal(history, replay.history_df(), check_exact=True)
                pd.testing.assert_frame_equal(history, resumed.history_df(), check_exact=True)
                values = history[f.space().objectives].to_numpy()
                hv = hypervolume((values - geometry.location) / geometry.scale, geometry.report_reference)
                row = {**source, "seed": seed, "method": method, "initial_x": f.initial(seed).x.tolist(),
                    "protocol": f.PROTOCOL, "protocol_version": f.PROTOCOL_VERSION,
                    "knowledge": f.knowledge(method).to_envelope(), "declared_noise_mode": f.config().noise_mode,
                    "budget": {"initial": f.INITIAL, "rounds": f.ROUNDS, "q": f.Q},
                    "geometry": geometry.to_dict(), "reference_raw": [0., 0.], "scale": list(geometry.scale),
                    "measurement_count": len(history), "batch_sizes": [len(b) for b in batches],
                    "batches": batches, "routes": routes, "runtime_seconds": time.perf_counter() - started,
                    "parameter_feasible_rate": float((history.x <= .8).mean()),
                    "duplicate_rate": float(history.x.duplicated().mean()),
                    "model_joint_feasibility": None, "outcome_safety_constraint": None,
                    "oracle_hv_raw": hypervolume(truth, [0., 0.]), "observed_hv_raw": hypervolume(values, [0., 0.]),
                    "oracle_hv": oracle_hv, "observed_hv": hv,
                    "hv_regret": hypervolume_regret((values - geometry.location) / geometry.scale,
                        (truth - geometry.location) / geometry.scale, geometry.report_reference),
                    "reproducible": batches == replay_batches, "restart_consistent": batches == resumed_batches,
                    "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in notices]}
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print(f"seed={seed} method={method} regret={row['hv_regret']:.9g} seconds={row['runtime_seconds']:.2f}", flush=True)
    summary = quality_summary(rows)
    summary.update({"seeds": seeds, "rows": len(rows), "runtime_seconds": sum(r["runtime_seconds"] for r in rows),
                    "provenance": source})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "multiobjective-validation")
    args = parser.parse_args()
    rows = run_matrix(seeds=f.SEEDS, output_dir=args.output_dir)
    summary = quality_summary(rows)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)
