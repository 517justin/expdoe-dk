"""
Example 01: Reaction optimization with discrete reagent volumes + constraint.

A chemist wants to maximize reaction yield. Knobs:
- Temperature T (continuous, 60–120 °C)
- Reaction time t (continuous, 10–180 min)
- Reagent A volume (discrete, integer mL in 1–10)
- Reagent B volume (discrete, integer mL in 1–10)

Constraint: A must exceed B by at least 1 mL (stoichiometric requirement).

Prior knowledge:
- Temperature follows Arrhenius behavior (rate rises with T).
- Time increases yield monotonically (longer reaction = more product, with
  diminishing returns).
- Reagent A has an optimum around 7 mL (peak, not monotone).

Budget: 8 initial DoE points + 15 BO iterations = 23 experiments.

Run with:
    python examples/01_reaction_optimization.py
"""
from __future__ import annotations

import warnings

import numpy as np

import expdoe_dk as ed


# --------------------------------------------------------------------- #
# 1. The "lab" — replaced by real measurements in production
# --------------------------------------------------------------------- #
def run_lab_experiments(df):
    """
    Simulated reaction yield. In real use, this is the chemist taking the
    DataFrame to the bench and measuring yield_pct for each row.
    """
    T = df["T"].to_numpy()
    t = df["time"].to_numpy()
    A = df["conc_A"].to_numpy()
    B = df["conc_B"].to_numpy()
    # Arrhenius-ish in T, monotone (saturating) in time, peaked at A=7, B=3.
    rate = np.exp(-1.5 / (T / 100.0))
    saturating_time = 1.0 - np.exp(-t / 60.0)
    eff_A = 1.0 - 0.04 * (A - 7.0) ** 2
    eff_B = 1.0 - 0.06 * (B - 3.0) ** 2
    yield_pct = 95.0 * rate * saturating_time * eff_A * eff_B
    rng = np.random.default_rng(0)
    return yield_pct + rng.normal(0.0, 0.3, size=len(df))


# --------------------------------------------------------------------- #
# 2. Declare the experiment in physical units
# --------------------------------------------------------------------- #
space = ed.Space(
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
# 3. Inject prior knowledge in physical-intuition language
# --------------------------------------------------------------------- #
knowledge = (
    ed.Knowledge()
    .with_arrhenius("T")
    .with_monotone("time", effect="increases_objective")
    .with_quadratic_peak("conc_A", center=7.0)
    .with_random_augment(n=20)
)


def main() -> None:
    warnings.simplefilter("ignore")
    print(space)
    print("\nKnowledge:")
    for it in knowledge.to_dict()["items"]:
        print(f"  - {it}")
    print()

    campaign = ed.Campaign(space, knowledge, seed=42)

    # ----------------------------------------------------------------- #
    # Initial DoE (constraint-aware, discrete-aware)
    # ----------------------------------------------------------------- #
    doe = campaign.suggest_doe(n=8, method="lhs_maximin", n_iterations=2000)
    print(f"Initial DoE (n=8):\n{doe.round(2).to_string(index=False)}\n")

    y_doe = run_lab_experiments(doe)
    campaign.tell(doe, y_doe)
    print(f"DoE yields: {np.round(y_doe, 2)}")
    print(f"Best DoE yield: {y_doe.max():.2f}\n")

    # ----------------------------------------------------------------- #
    # BO iterations
    # ----------------------------------------------------------------- #
    n_iter = 15
    for it in range(n_iter):
        next_pts = campaign.ask(q=1, iteration=it)
        y_new = run_lab_experiments(next_pts)
        campaign.tell(next_pts, y_new)
        best_so_far = campaign.history_df()["y"].max()
        print(
            f"BO iter {it+1:2d}: proposed T={next_pts.iloc[0]['T']:.1f} °C, "
            f"time={next_pts.iloc[0]['time']:.1f} min, "
            f"A={int(next_pts.iloc[0]['conc_A'])} mL, "
            f"B={int(next_pts.iloc[0]['conc_B'])} mL → y={y_new[0]:.2f} "
            f"(best={best_so_far:.2f})"
        )

    # ----------------------------------------------------------------- #
    # Finalize
    # ----------------------------------------------------------------- #
    result = campaign.finalize()
    print("\n" + "=" * 60)
    print(f"Best yield: {result.best_y_physical:.2f}%")
    print("Best conditions:")
    for k, v in result.best_x_physical.items():
        p = space.param_by_name(k)
        if p.kind == "discrete":
            print(f"  {k:8s} = {int(v):>4d} {p.unit}")
        else:
            print(f"  {k:8s} = {v:>7.2f} {p.unit}")
    print(f"Total experiments: {result.n_doe + result.n_bo}")
    print("=" * 60)


if __name__ == "__main__":
    main()
