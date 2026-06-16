"""
End-to-end smoke test of the Campaign loop on a small chemistry oracle.
"""
import warnings

import numpy as np
import pytest

from expdoe_dk import (
    Campaign,
    Knowledge,
    LinearConstraint,
    Parameter,
    Space,
)


def _chem_space():
    return Space(
        params=[
            Parameter("T", bounds=(60.0, 120.0), unit="°C"),
            Parameter("time", bounds=(10.0, 180.0), unit="min"),
            Parameter("conc_A", bounds=(1.0, 10.0), unit="mL",
                      kind="discrete", step=1.0),
            Parameter("conc_B", bounds=(1.0, 10.0), unit="mL",
                      kind="discrete", step=1.0),
        ],
        constraints=[
            LinearConstraint(coeffs={"conc_A": 1.0, "conc_B": -1.0}, lower=1.0),
        ],
        objectives="yield_pct",
        maximize=True,
    )


def _oracle(df):
    """Quadratic toy yield: peak at (T=95, time=120, A=7, B=3)."""
    T = df["T"].to_numpy()
    t = df["time"].to_numpy()
    A = df["conc_A"].to_numpy()
    B = df["conc_B"].to_numpy()
    y = (
        100.0
        - 0.02 * (T - 95.0) ** 2
        - 0.001 * (t - 120.0) ** 2
        - 1.5 * (A - 7.0) ** 2
        - 1.5 * (B - 3.0) ** 2
    )
    rng = np.random.default_rng(0)
    return y + rng.normal(0.0, 0.3, size=len(df))


@pytest.mark.slow
def test_campaign_runs_end_to_end_and_improves():
    warnings.simplefilter("ignore")
    space = _chem_space()
    knowledge = (
        Knowledge()
        .with_arrhenius("T")
        .with_monotone("time", effect="increases_objective")
        .with_quadratic_peak("conc_A", center=7.0)
        .with_random_augment(n=20)
    )
    campaign = Campaign(space, knowledge, seed=42)
    result = campaign.run(_oracle, n_doe=6, n_iter=8, q=1)

    # Feasibility of every recorded trial.
    assert bool(space.feasibility_mask(result.history_df).all().item())

    # Improvement: best y after BO must beat best y from DoE-only.
    doe_only = result.history_df[result.history_df["kind"] == "doe"]
    bo_only = result.history_df[result.history_df["kind"] == "bo"]
    assert bo_only["y"].max() >= doe_only["y"].max() - 1e-6


@pytest.mark.slow
def test_campaign_checkpoint_roundtrip(tmp_path):
    warnings.simplefilter("ignore")
    space = _chem_space()
    campaign = Campaign(space, Knowledge(), seed=42)
    doe = campaign.suggest_doe(n=6, method="lhs_random", n_iterations=200)
    y = _oracle(doe)
    campaign.tell(doe, y)

    ckpt = tmp_path / "campaign.json"
    campaign.save_checkpoint(ckpt)
    campaign2 = Campaign.load_checkpoint(ckpt)

    # History rows match.
    assert len(campaign2._X_phys) == len(campaign._X_phys)
    assert np.allclose(campaign2._y_internal, campaign._y_internal)
