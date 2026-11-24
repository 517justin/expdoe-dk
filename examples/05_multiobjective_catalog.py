"""Versioned multiobjective workflow: synthetic measurements, not lab safety.

Run with the checkout installed: python examples/05_multiobjective_catalog.py
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import expdoe_dk as ed
from expdoe_dk.domain import LinearConstraint


def measure(X):
    """A noiseless teaching oracle; never replace real lab returns with predictions."""
    return pd.DataFrame({"retention": 1 - X.x.to_numpy(), "gain": X.x.to_numpy()})


def run_example():
    space = ed.Space([ed.Parameter("x", bounds=(0., 1.))],
        objectives=[ed.Objective("gain", "maximize"), ed.Objective("retention", "maximize")],
        constraints=[LinearConstraint("limit", {"x": 1.}, "<=", .8)])
    catalog = pd.DataFrame({"x": np.arange(21) / 20})
    candidate_ids = tuple(f"recipe-{i:02}" for i in range(len(catalog)))
    config = ed.MultiObjectiveConfig(noise_mode="noiseless", reference_point={"gain": 0., "retention": 0.})
    knowledge = ed.Knowledge().add(ed.make_pattern_spec(
        pattern_id="gain-increases-v2", pattern="monotone", version="2.0",
        scope=ed.KnowledgeScope(factors=("x",), objectives=("gain",)), confidence=.8,
        parameters={"direction": "increasing", "n_pairs_per_dim": 3, "epsilon": .05, "softness": 1.}))
    campaign = ed.Campaign(space, knowledge, seed=7, validate=False, multiobjective=config)
    feasible = catalog.loc[catalog.x <= .8]
    initial = feasible.iloc[np.random.default_rng(7).choice(len(feasible), 6, replace=False)]
    campaign.tell(initial, measure(initial))
    with TemporaryDirectory() as directory:
        for iteration in range(2):
            suggestion = campaign.ask_with_diagnostics(q=2, candidates=catalog, candidate_ids=candidate_ids)
            assert suggestion.diagnostics["acquisition_kind"] == "q_log_ehvi"
            assert suggestion.diagnostics["objectives"]["gain"]["directional"]["status"] == "active"
            pd.testing.assert_frame_equal(suggestion.frame,
                catalog.iloc[[candidate_ids.index(i) for i in suggestion.candidate_ids]].reset_index(drop=True))
            trial_ids = tuple(f"measurement-{iteration}-{i}" for i in range(2))
            campaign.set_pending(ed.PendingBatch(trial_ids, suggestion.frame))
            path = Path(directory) / "campaign.json"
            campaign.save_checkpoint(path)
            campaign = ed.Campaign.load_checkpoint(path)
            campaign.tell_batch(ed.ObservationBatch(suggestion.frame, measure(suggestion.frame), ids=trial_ids))
        result = campaign.finalize()
        result.to_html(Path(directory) / "report.html", title="Synthetic gain and retention")
    assert len(result.history_df) == 10
    assert result.history_df.x.is_unique
    assert (result.history_df.x <= .8).all()
    return result


if __name__ == "__main__":
    result = run_example()
    print(result.history_df.to_string(index=False))
    print("Measured Pareto rows:", len(result.pareto_rows))
    print("Fixed-geometry observed hypervolume:", result.hypervolume_history)
    print("Synthetic noiseless measurements only; no laboratory safety guarantee.")
