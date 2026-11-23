"""Recommend only original catalog formulas; never expose unmeasured outcomes.

Run from the repository root with the current checkout installed, or:
    PYTHONPATH=expdoe-dk/src python examples/03_candidate_catalog.py
"""
from __future__ import annotations

import pandas as pd

import expdoe_dk as ed
from expdoe_dk.domain import LinearConstraint


def run_example() -> ed.RecommendationResult:
    space = ed.Space(
        [ed.Parameter("A", bounds=(0., 10.), unit="mL"),
         ed.Parameter("B", bounds=(0., 10.), unit="mL"),
         ed.Parameter("material", kind="categorical", values=("water", "ethanol"))],
        objectives=ed.Objective("yield_pct", "maximize", unit="%"),
        constraints=[LinearConstraint("total_volume", {"A": 1., "B": 1.}, "==", 10.)],
    )
    catalog = pd.DataFrame({
        "candidate_id": ["F01", "F02", "F03", "F04", "F05", "F06", "F07"],
        "A": [1., 3., 5., 6., 7., 8., 9.],
        "B": [9., 7., 5., 4., 3., 2., 1.],
        "material": ["water", "ethanol", "water", "ethanol", "water", "ethanol", "water"],
    }).set_index("candidate_id")
    measured = ("F01", "F02", "F03")
    observations = ed.ObservationBatch(
        X=catalog.loc[list(measured)],
        Y=pd.DataFrame({"yield_pct": [32., 48., 61.]}),
        ids=("trial-001", "trial-002", "trial-003"),
    )
    pending = ed.PendingBatch(("trial-004",), catalog.loc[["F04"]])
    result = ed.recommend(
        space, observations, pending=pending, candidates=catalog,
        candidate_ids=tuple(catalog.index), q=2, seed=17,
    )
    assert len(set(result.candidate_ids)) == 2
    assert set(result.candidate_ids) <= {"F05", "F06", "F07"}
    pd.testing.assert_frame_equal(
        result.frame, catalog.loc[list(result.candidate_ids)].reset_index(drop=True),
        check_exact=True,
    )
    assert space.feasibility_mask(result.frame).all()
    return result


if __name__ == "__main__":
    recommendation = run_example()
    display = recommendation.frame
    display.insert(0, "candidate_id", recommendation.candidate_ids)
    display["predicted_yield_pct"] = recommendation.predictions.mean[:, 0]
    print(display.to_string(index=False))
    print("Prediction only: no new laboratory measurements have been recorded.")
