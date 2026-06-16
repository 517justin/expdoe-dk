"""
Example 02: generate the v0.4 HTML campaign report.

Same chemistry scenario as example 01 (T × time × conc_A × conc_B with the
A−B ≥ 1 constraint). After the BO loop finishes we call
``Result.to_html(path)``, producing a single self-contained HTML file with:

  - Best yield + best conditions in physical units
  - Convergence chart (best-so-far stepped line + per-trial dots)
  - History table + one-click CSV download (embedded as a data: URL)
  - Knowledge applied: human-readable list

The page renders with Chart.js loaded from cdn.jsdelivr.net but degrades
gracefully when JavaScript is disabled — the summary / table / CSV all
still work.

Run with:
    python examples/02_html_report.py
    open report_example_02.html  # macOS
"""
from __future__ import annotations

import warnings
import webbrowser
from pathlib import Path

import numpy as np

import expdoe_dk as ed


# --------------------------------------------------------------------- #
# Same "lab" as example 01 — replaced by real measurements in production
# --------------------------------------------------------------------- #
def run_lab_experiments(df):
    T = df["T"].to_numpy()
    t = df["time"].to_numpy()
    A = df["conc_A"].to_numpy()
    B = df["conc_B"].to_numpy()
    rate = np.exp(-1.5 / (T / 100.0))
    saturating_time = 1.0 - np.exp(-t / 60.0)
    eff_A = 1.0 - 0.04 * (A - 7.0) ** 2
    eff_B = 1.0 - 0.06 * (B - 3.0) ** 2
    yield_pct = 95.0 * rate * saturating_time * eff_A * eff_B
    rng = np.random.default_rng(0)
    return yield_pct + rng.normal(0.0, 0.3, size=len(df))


# --------------------------------------------------------------------- #
# Space, knowledge, campaign
# --------------------------------------------------------------------- #
space = ed.Space(
    params=[
        ed.Parameter("T",      bounds=(60.0, 120.0), unit="°C"),
        ed.Parameter("time",   bounds=(10.0, 180.0), unit="min"),
        ed.Parameter("conc_A", bounds=(1.0, 10.0),   unit="mL",
                     kind="discrete", step=1.0),
        ed.Parameter("conc_B", bounds=(1.0, 10.0),   unit="mL",
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

knowledge = (
    ed.Knowledge()
    .with_arrhenius("T")
    .with_monotone("time", effect="increases_objective")
    .with_quadratic_peak("conc_A", center=7.0)
    .with_random_augment(n=20)
)


def main() -> None:
    warnings.simplefilter("ignore")

    print(space, "\n")

    campaign = ed.Campaign(space, knowledge, seed=42)
    result = campaign.run(run_lab_experiments, n_doe=8, n_iter=12)

    print(f"Best yield : {result.best_y_physical:.2f}%")
    print("Best conditions:")
    for k, v in result.best_x_physical.items():
        p = space.param_by_name(k)
        if p.kind == "discrete":
            print(f"  {k:8s} = {int(v):>4d} {p.unit}")
        else:
            print(f"  {k:8s} = {v:>7.2f} {p.unit}")

    out_path = Path(__file__).resolve().parent / "report_example_02.html"
    result.to_html(out_path, title="Reaction yield optimisation (example 02)")
    print(f"\nReport written: {out_path}")
    print(f"  size: {out_path.stat().st_size} bytes")
    print(f"  open with: open {out_path}")

    # Optional: open the report in the default browser. Suppressed when
    # running headless (CI). Comment in if you want this on every run.
    # webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
