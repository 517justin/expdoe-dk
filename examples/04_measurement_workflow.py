"""Example 04: track asynchronous lab measurements around a Campaign.

This single-process helper is not part of the expdoe_dk API. From the inner
package directory, run with: PYTHONPATH=src python ../examples/04_measurement_workflow.py
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import expdoe_dk as ed


@dataclass(frozen=True)
class Measurement:
    trial_id: str
    status: str
    value: float | None = None


class MeasurementWorkflow:
    """Own trial bookkeeping for a single-objective Campaign and a fixed catalog."""

    def __init__(self, space: ed.Space, catalog: pd.DataFrame) -> None:
        if space.n_objectives != 1:
            raise ValueError("This example requires exactly one objective.")
        if list(catalog.columns) != list(space.param_names):
            raise ValueError("Catalog columns must exactly match the physical parameter names.")
        if (catalog.empty or not catalog.index.is_unique
                or any(not isinstance(key, str) or not key for key in catalog.index)):
            raise ValueError("The catalog must have unique non-empty string candidate IDs.")
        space.physical_to_model(catalog)
        self.campaign = ed.Campaign(space)
        self._catalog = catalog.copy(deep=True)
        self._trial_candidates: dict[str, str] = {}
        self._pending_ids: tuple[str, ...] = ()
        self._cancellations: dict[str, str] = {}

    def _conditions(self, ids: Sequence[str], trials: dict[str, str]) -> pd.DataFrame:
        return self._catalog.loc[[trials[trial_id] for trial_id in ids]].reset_index(drop=True)

    def _pending_batch(self, ids: tuple[str, ...], trials: dict[str, str]) -> ed.PendingBatch | None:
        return ed.PendingBatch(ids, self._conditions(ids, trials)) if ids else None

    def _require_pending(self, trial_id: str) -> None:
        if not isinstance(trial_id, str) or trial_id not in self._trial_candidates:
            raise ValueError(f"Unknown trial ID: {trial_id!r}.")
        if trial_id not in self._pending_ids:
            raise ValueError(f"Trial {trial_id!r} already returned or was cancelled.")

    def dispatch(self, trial_id: str, candidate_id: str) -> None:
        """Register one new measurement, including intentional repeats of a formula."""
        if not isinstance(candidate_id, str) or candidate_id not in self._catalog.index:
            raise ValueError(f"Unknown candidate ID: {candidate_id!r}.")
        if not isinstance(trial_id, str) or not trial_id or trial_id in self._catalog.index:
            raise ValueError("Trial IDs must be non-empty strings distinct from candidate IDs.")
        if trial_id in self._trial_candidates:
            raise ValueError(f"Duplicate trial ID: {trial_id!r}.")
        if not self.campaign.space.feasibility_mask(self._catalog.loc[[candidate_id]]).all():
            raise ValueError("Dispatched formula violates a hard parameter constraint.")
        trials = {**self._trial_candidates, trial_id: candidate_id}
        pending_ids = (*self._pending_ids, trial_id)
        self.campaign.set_pending(self._pending_batch(pending_ids, trials))
        self._trial_candidates = trials
        self._pending_ids = pending_ids

    def record_returns(self, returns: Sequence[Measurement]) -> None:
        """Validate the whole return before recording any outcomes or closing any trials."""
        rows = tuple(returns)
        if not rows:
            return
        ids = tuple(row.trial_id for row in rows)
        for trial_id in ids:
            self._require_pending(trial_id)
        returned = set(ids)
        if len(returned) != len(ids):
            raise ValueError("Duplicate trial IDs in the same return.")

        # Preserve missing outcomes as missing; never substitute a prediction.
        outcomes = pd.DataFrame(
            [[pd.NA if row.value is None else row.value] for row in rows],
            columns=self.campaign.space.objectives, dtype=object,
        )
        observations = ed.ObservationBatch(
            self._conditions(ids, self._trial_candidates), outcomes,
            status=pd.Series([row.status for row in rows], dtype=object), ids=ids,
        )
        pending_ids = tuple(trial_id for trial_id in self._pending_ids if trial_id not in returned)
        pending = self._pending_batch(pending_ids, self._trial_candidates)
        self.campaign.tell_batch(observations)
        self.campaign.set_pending(pending)
        self._pending_ids = pending_ids

    def cancel(self, trial_id: str, reason: str) -> None:
        """Close a pending trial without inventing an observation."""
        self._require_pending(trial_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Cancellation requires a reason.")
        pending_ids = tuple(key for key in self._pending_ids if key != trial_id)
        self.campaign.set_pending(self._pending_batch(pending_ids, self._trial_candidates))
        self._pending_ids = pending_ids
        self._cancellations[trial_id] = reason

    def to_state(self) -> dict:
        return {
            "catalog": {"ids": self._catalog.index.tolist(),
                        "X": self._catalog.astype(object).values.tolist()},
            "trial_candidates": self._trial_candidates.copy(),
            "cancellations": self._cancellations.copy(),
            "campaign": self.campaign.to_state(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_state(), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> MeasurementWorkflow:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        campaign_state = state["campaign"]
        campaign = ed.Campaign.from_state(campaign_state)
        catalog = pd.DataFrame(state["catalog"]["X"], index=state["catalog"]["ids"],
                               columns=campaign.space.param_names, dtype=object)
        workflow = cls(campaign.space, catalog)
        workflow.campaign = campaign
        workflow._trial_candidates = state["trial_candidates"]
        workflow._cancellations = state["cancellations"]
        pending = campaign_state["pending"]
        workflow._pending_ids = () if pending is None else tuple(pending["ids"])
        workflow._validate_restored_audit(campaign_state)
        return workflow

    def _validate_restored_audit(self, campaign_state: dict) -> None:
        trials, cancelled = self._trial_candidates, self._cancellations
        if not isinstance(trials, dict) or not isinstance(cancelled, dict):
            raise ValueError("Invalid trial audit mappings.")
        if any(not isinstance(trial, str) or not trial or trial in self._catalog.index
               or not isinstance(candidate, str) or candidate not in self._catalog.index
               for trial, candidate in trials.items()):
            raise ValueError("Invalid trial-to-candidate mapping.")
        if any(trial not in trials or not isinstance(reason, str) or not reason.strip()
               for trial, reason in cancelled.items()):
            raise ValueError("Invalid cancellation audit.")

        seen = set(cancelled)
        batches = list(campaign_state["observations"])
        if campaign_state["pending"] is not None:
            batches.append(campaign_state["pending"])
        for batch in batches:
            if len(batch["ids"]) != len(batch["X"]):
                raise ValueError("Every recorded condition requires its trial ID.")
            for trial, actual in zip(batch["ids"], batch["X"]):
                if trial not in trials or trial in seen:
                    raise ValueError("Trial states overlap or refer to an unknown trial.")
                expected = self._catalog.loc[[trials[trial]]].to_dict(orient="records")[0]
                for parameter, value in zip(self.campaign.space.params, actual):
                    original = expected[parameter.name]
                    if value != original or isinstance(value, bool) != isinstance(original, bool):
                        raise ValueError("Trial conditions disagree with the dispatched formula.")
                seen.add(trial)
        if seen != set(trials):
            raise ValueError("Trial audit contains unaccounted measurements.")


def main() -> None:
    space = ed.Space(params=[ed.Parameter("temperature", bounds=(20, 100), unit="C")],
                     objectives="yield_pct", maximize=True)
    catalog = pd.DataFrame({"temperature": [35, 50, 75]},
                           index=["formula-a", "formula-b", "formula-c"])
    workflow = MeasurementWorkflow(space, catalog)
    workflow.dispatch("trial-1", "formula-a")
    workflow.dispatch("trial-2", "formula-b")
    workflow.dispatch("trial-3", "formula-c")

    # Mock lab returns, not model predictions. The third trial is still pending.
    workflow.record_returns([Measurement("trial-1", "success", 78.0),
                             Measurement("trial-2", "failed")])
    with TemporaryDirectory() as directory:
        path = Path(directory) / "measurement-workflow.json"
        workflow.save(path)
        workflow = MeasurementWorkflow.load(path)

    workflow.cancel("trial-3", "instrument unavailable")
    workflow.dispatch("trial-4", "formula-b")  # Retry under a fresh trial ID.
    workflow.record_returns([Measurement("trial-4", "warning", 97.0)])
    workflow.dispatch("trial-5", "formula-b")
    workflow.record_returns([Measurement("trial-5", "success", 83.0)])
    print("Successful measurements:")
    print(workflow.campaign.history_df().to_string(index=False))
    print("Cancellations:", workflow.to_state()["cancellations"])


if __name__ == "__main__":
    main()
