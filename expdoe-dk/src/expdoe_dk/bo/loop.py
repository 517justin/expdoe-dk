"""
Campaign — the end-to-end DoE + BO loop.

Maintains the experiment history as the user's "physical units" DataFrame
plus the in-frame internal Y values. All translation between frames is
funneled through `_frame.py` to prevent the D-bug class of errors.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..space import Space
from ..domain import ObservationBatch, PendingBatch
from ..knowledge import Knowledge
from ..domain.space import ENGINE_VERSION
from ..errors import EngineError, ErrorCode
from ..knowledge.validators import (
    check_monotone_assumption,
    check_shape_prior_fit,
)
from ..knowledge.shape import (
    ArrheniusMeanFrozen,
    QuadraticMeanFrozen,
)
from ..doe import generate as generate_doe
from .contracts import RecommendationResult
from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION, atomic_save, batch_to_state, read_json, restore,
    validate_frame, validate_state,
)

log = logging.getLogger("expdoe_dk.bo.Campaign")

# --------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------- #
@dataclass
class Result:
    best_x_physical: dict
    best_y_physical: float
    history_df: pd.DataFrame
    n_doe: int
    n_bo: int
    objectives: list[str]
    maximize: list[bool]
    knowledge_summary: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    param_units: dict[str, str] = field(default_factory=dict)
    param_kinds: dict[str, str] = field(default_factory=dict)
    knowledge: dict[str, object] | None = None

    def to_dict(self) -> dict:
        return {
            "best_x_physical": self.best_x_physical,
            "best_y_physical": self.best_y_physical,
            "n_doe": self.n_doe,
            "n_bo": self.n_bo,
            "objectives": self.objectives,
            "maximize": self.maximize,
            "knowledge_summary": self.knowledge_summary,
            "knowledge": self.knowledge,
            "notes": self.notes,
            "param_units": self.param_units,
            "param_kinds": self.param_kinds,
            "history_records": self.history_df.to_dict(orient="records"),
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def to_html(self, path: str | Path, title: str | None = None) -> Path:
        """
        Render a self-contained HTML report and write it to ``path``.

        Returns the resolved ``Path``. The page uses Chart.js via CDN for the
        convergence plot and degrades gracefully (table + summary) when JS
        is disabled. All numbers shown are in physical units.
        """
        from .report import write_html_report

        return write_html_report(self, path, title=title)

    def to_html_string(self, title: str | None = None) -> str:
        """Return the HTML report as a string (no file written)."""
        from .report import render_html

        return render_html(self, title=title)


# --------------------------------------------------------------------- #
# Campaign
# --------------------------------------------------------------------- #
class Campaign:
    """
    Stateful BO campaign in physical units.

    Internally minimizes a normalized Y. The public ask/tell/run methods
    accept and return DataFrames in physical units with the user's objective
    sign convention (maximize=True → higher is better).

    Parameters
    ----------
    space : Space
        Parameter space with constraints and objective directions.
    knowledge : Knowledge | None
        Optional domain knowledge spec. If None, the Campaign runs a plain
        GP with no injected structure — the conservative default. The
        library does NOT silently add regularization on your behalf;
        ``with_random_augment`` is opt-in (see the note below).
    seed : int
        Base random seed.

    Notes
    -----
    Earlier versions auto-applied ``with_random_augment(n=20)`` when no
    knowledge was given. That default has been removed: random-augment is
    pure regularization whose benefit is still under validation, and the
    useful ``n`` depends on the experiment's sample size — so applying it
    silently could mislead. Pass ``knowledge=Knowledge().with_random_augment(n=...)``
    explicitly if you want it.
    """

    def __init__(
        self,
        space: Space,
        knowledge: Knowledge | None = None,
        seed: int = 0,
        *,
        validate: bool = True,
        validation_interval: int = 5,
        validation_min_obs: int = 10,
        auto_rescue: bool = True,
    ) -> None:
        if space.n_objectives > 1:
            warnings.warn(
                "Multi-objective declared; v0.1 only optimizes the first "
                f"objective: {space.objectives[0]!r}. Other objectives are "
                f"stored in history but not driving acquisition.",
                stacklevel=2,
            )
        self.space = space
        # No auto-default: an empty Knowledge means "plain GP". The library
        # never injects regularization on the user's behalf (see class Notes).
        self.knowledge = knowledge if knowledge is not None else Knowledge()
        # v0.3: rescue (or raise on) Exp-14 epsilon conflicts before any
        # acquisition is built.
        self.auto_rescue = bool(auto_rescue)
        self.knowledge.validate(auto_rescue=self.auto_rescue)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # History stored as DataFrame (physical X) + tensor (internal Y).
        self._X_phys: list[pd.DataFrame] = []  # list of single-row frames
        self._y_internal: list[float] = []
        self._y_physical: list[float] = []
        self._trial_kind: list[str] = []  # "doe" / "bo"
        self._observation_batches: list[ObservationBatch] = []
        self._pending: PendingBatch | None = None
        self._ask_iteration = 0

        # v0.2 — empirical validators (run on tell, throttled by interval).
        self.validate_active = bool(validate)
        self.validation_interval = max(1, int(validation_interval))
        self.validation_min_obs = max(3, int(validation_min_obs))
        self._last_validation_n = 0
        # Cache of warnings already emitted so we don't spam every K obs.
        self._validation_emitted: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ #
    # Step 1: initial DoE
    # ------------------------------------------------------------------ #
    def suggest_doe(
        self,
        n: int,
        method: str = "lhs_maximin",
        n_iterations: int = 2000,
        n_restarts: int = 10,
    ) -> pd.DataFrame:
        """Return n design points in physical units (DataFrame)."""
        df = generate_doe(
            self.space,
            n=n,
            method=method,
            n_iterations=n_iterations,
            n_restarts=n_restarts,
            seed=self.seed,
        )
        return df

    # ------------------------------------------------------------------ #
    # Step 2: tell — record results
    # ------------------------------------------------------------------ #
    def tell(self, X_phys: pd.DataFrame, y: np.ndarray | list[float]) -> None:
        """
        Record results from user-run experiments.

        Parameters
        ----------
        X_phys : DataFrame
            Rows of physical-unit parameter values. Columns must match space.param_names.
        y : array-like
            Same length as X_phys. Single-objective scalar per row. In the
            user's frame (higher=better if maximize=True).
        """
        if list(X_phys.columns) != list(self.space.param_names):
            missing = set(self.space.param_names) - set(X_phys.columns)
            extra = set(X_phys.columns) - set(self.space.param_names)
            raise ValueError(
                f"X_phys columns must match space.param_names. "
                f"Missing: {sorted(missing)}, extra: {sorted(extra)}."
            )
        y_arr = np.asarray(y, dtype=np.float64).flatten()
        if len(y_arr) != len(X_phys):
            raise ValueError(
                f"len(y)={len(y_arr)} must match len(X_phys)={len(X_phys)}."
            )

        observations = ObservationBatch(X_phys, pd.DataFrame({self.space.objectives[0]: y_arr}))
        if self.space.n_objectives == 1:
            self.tell_batch(observations)
        else:
            # The historical first-objective archive remains readable/writable;
            # new Phase-1 entry points explicitly refuse this interpretation.
            validate_frame(self.space, observations.X)
            self._record_observations(observations)

    def tell_batch(self, observations: ObservationBatch) -> None:
        """Record physical outcomes, preserving every status and audit field.

        Only successful rows enter legacy history/results or model training;
        all rows remain visible to recommendation duplicate exclusion.
        """
        if self.space.n_objectives != 1:
            raise EngineError(ErrorCode.CONFIG_INVALID, "Phase-1 Campaign requires exactly one objective")
        if not isinstance(observations, ObservationBatch):
            raise EngineError(ErrorCode.CONFIG_INVALID, "Expected ObservationBatch")
        if list(observations.Y.columns) != self.space.objectives:
            raise EngineError(ErrorCode.CONFIG_INVALID, "Response columns must exactly match space.objectives")
        validate_frame(self.space, observations.X)
        previous_ids = {identifier for batch in self._observation_batches for identifier in batch.ids}
        if previous_ids.intersection(observations.ids):
            raise EngineError(ErrorCode.CONFIG_INVALID, "Observation IDs must be unique across batches")
        # Validate serializability before mutation, including unsuccessful values.
        from .checkpoint import _strict_json
        _strict_json(batch_to_state(observations))
        self._record_observations(observations)

    def _record_observations(self, observations: ObservationBatch) -> None:
        successful = observations.successful()
        X, Y = successful.X, successful.Y
        kind = "doe" if not self._X_phys else "bo"
        self._observation_batches.append(observations)
        for i in range(len(X)):
            self._X_phys.append(X.iloc[[i]].reset_index(drop=True))
            y_user = float(Y.iloc[i, 0])
            self._y_physical.append(y_user)
            self._y_internal.append(float(-self.space.objective_specs[0].to_utility(np.array([y_user]))[0]))
            self._trial_kind.append(kind)
        self._maybe_run_validators()

    def set_pending(self, pending: PendingBatch | None) -> None:
        """Replace pending work; repeated conditions remain auditable and excluded."""
        if self.space.n_objectives != 1:
            raise EngineError(ErrorCode.CONFIG_INVALID, "Phase-1 Campaign requires exactly one objective")
        if pending is not None:
            if not isinstance(pending, PendingBatch):
                raise EngineError(ErrorCode.CONFIG_INVALID, "Expected PendingBatch or None")
            validate_frame(self.space, pending.X)
        self._pending = pending

    def _combined_observations(self) -> ObservationBatch:
        batches = self._observation_batches
        # A partially known noise vector cannot be represented by ObservationBatch.
        # Preserve all input variances in state and fail explicitly at fit time,
        # rather than silently discarding known variances or inventing them.
        successful_batches = [b for b in batches if len(b.successful().X)]
        known = [b.Yvar is not None for b in successful_batches]
        if any(known) and not all(known):
            raise EngineError(ErrorCode.CONFIG_INVALID,
                "Mixed missing variance across successful batches is unsupported")
        variance = None
        if known and all(known):
            variance = pd.concat([b.Yvar if b.Yvar is not None else
                pd.DataFrame(0., index=range(len(b.X)), columns=self.space.objectives)
                for b in batches], ignore_index=True)
        status = pd.concat([b.status if b.status is not None else
            pd.Series(["success"] * len(b.X), dtype=object) for b in batches], ignore_index=True)
        identifiers = []
        used = {identifier for b in batches for identifier in b.ids}
        for index, batch in enumerate(batches):
            for row in range(len(batch.X)):
                identifier = batch.ids[row] if batch.ids else f"unidentified:{index}:{row}"
                if not batch.ids:
                    while identifier in used:
                        identifier += ":"
                    used.add(identifier)
                identifiers.append(identifier)
        return ObservationBatch(pd.concat([b.X for b in batches], ignore_index=True),
            pd.concat([b.Y for b in batches], ignore_index=True), Yvar=variance,
            status=status, ids=identifiers)

    # ------------------------------------------------------------------ #
    # Step 3: ask — propose next experiments
    # ------------------------------------------------------------------ #
    def ask(self, q: int = 1, iteration: int | None = None) -> pd.DataFrame:
        """Return physical candidates through the public diagnostic boundary."""
        return self.ask_with_diagnostics(q=q, iteration=iteration).frame

    def ask_with_diagnostics(self, q: int = 1, iteration: int | None = None) -> RecommendationResult:
        """Fit the production GP and return candidates plus truthful diagnostics."""
        from .recommendation import recommend

        if iteration is not None and (type(iteration) is not int or iteration < 0):
            raise EngineError(ErrorCode.CONFIG_INVALID, "iteration must be a non-negative integer")
        if not self._X_phys:
            raise EngineError(ErrorCode.INSUFFICIENT_DATA,
                "Campaign.ask requires initial observations via tell()")
        observations = self._combined_observations()
        iteration_seed = self.seed + (len(self._y_physical) + self._ask_iteration if iteration is None else iteration)
        result = recommend(self.space, observations, knowledge=self.knowledge,
            pending=self._pending, q=q, seed=iteration_seed)
        self._ask_iteration = self._ask_iteration + 1 if iteration is None else iteration + 1
        return result

    # ------------------------------------------------------------------ #
    # Step 4: run — full DoE + BO automation
    # ------------------------------------------------------------------ #
    def run(
        self,
        oracle: Callable[[pd.DataFrame], np.ndarray],
        n_doe: int = 12,
        n_iter: int = 20,
        q: int = 1,
        doe_method: str = "lhs_maximin",
        checkpoint: str | Path | None = None,
    ) -> Result:
        # Initial DoE.
        doe = self.suggest_doe(n=n_doe, method=doe_method)
        y_doe = np.asarray(oracle(doe), dtype=np.float64).flatten()
        self.tell(doe, y_doe)

        # BO iterations.
        for it in range(n_iter):
            next_pts = self.ask(q=q, iteration=it)
            y_next = np.asarray(oracle(next_pts), dtype=np.float64).flatten()
            self.tell(next_pts, y_next)
            if checkpoint:
                self.save_checkpoint(checkpoint)

        return self.finalize()

    # ------------------------------------------------------------------ #
    # Empirical validators (v0.2)
    # ------------------------------------------------------------------ #
    def _maybe_run_validators(self) -> None:
        """Throttled hook: run empirical validators after enough new data."""
        if not self.validate_active:
            return
        n = len(self._y_internal)
        if n < self.validation_min_obs:
            return
        if n < self._last_validation_n + self.validation_interval:
            return
        self._last_validation_n = n
        self._run_monotone_validators()
        self._run_shape_validators()

    def _run_monotone_validators(self) -> None:
        mono_items = self.knowledge.items_of("monotone")
        if not mono_items:
            return
        X_phys = self.history_df()[self.space.param_names].to_numpy()
        y_user = np.asarray(self._y_physical, dtype=np.float64)
        for item in mono_items:
            try:
                idx = self.space.param_names.index(item.param)
            except ValueError:
                continue
            result = check_monotone_assumption(
                X_user_units=X_phys,
                y_user=y_user,
                param_index=idx,
                param_name=item.param,
                declared_effect=item.effect,
                min_observations=self.validation_min_obs,
            )
            if result is None or not result.violation:
                continue
            key = ("monotone", item.param)
            if key in self._validation_emitted:
                continue
            self._validation_emitted.add(key)
            warnings.warn(result.to_warning(), stacklevel=3)

    def _run_shape_validators(self) -> None:
        shape_items = [
            it for it in self.knowledge.items
            if getattr(it, "kind", None) in ("arrhenius", "quadratic_peak")
        ]
        if not shape_items:
            return
        # Build mean predictions in unit / Y_norm space.
        X_phys = self.space.to_tensor(
            self.history_df()[self.space.param_names]
        )
        X_unit = self.space.physical_to_unit(X_phys)
        y_int = torch.tensor(self._y_internal, dtype=torch.float64)
        if y_int.std() < 1e-9:
            return
        y_norm = ((y_int - y_int.mean()) / y_int.std()).cpu().numpy()

        for item in shape_items:
            try:
                dim_idx = self.space.param_names.index(item.param)
            except ValueError:
                continue
            kind = getattr(item, "kind")
            if kind == "arrhenius":
                fn = ArrheniusMeanFrozen(
                    temp_dim_index=dim_idx,
                    activation_energy=item.activation_energy,
                    amplitude_init=item.amplitude_init,
                )
            else:  # quadratic_peak
                param = self.space.param_by_name(item.param)
                lo, hi = param.bounds
                center_unit = (item.center - lo) / (hi - lo)
                curvature_signs = [0.0] * self.space.n_dims
                if item.direction == "peak":
                    sign_internal = +1.0 if self.space.maximize[0] else -1.0
                else:
                    sign_internal = -1.0 if self.space.maximize[0] else +1.0
                curvature_signs[dim_idx] = sign_internal
                centers = [0.5] * self.space.n_dims
                centers[dim_idx] = float(center_unit)
                fn = QuadraticMeanFrozen(
                    input_dim=self.space.n_dims,
                    curvature_signs=curvature_signs,
                    centers=centers,
                )
            with torch.no_grad():
                m_pred = fn(X_unit).detach().cpu().numpy()
            result = check_shape_prior_fit(
                mean_predictions=m_pred,
                y_norm=y_norm,
                kind=kind,
                param=item.param,
                min_observations=self.validation_min_obs,
            )
            if result is None or not result.violation:
                continue
            key = (kind, item.param)
            if key in self._validation_emitted:
                continue
            self._validation_emitted.add(key)
            warnings.warn(result.to_warning(), stacklevel=3)

    # ------------------------------------------------------------------ #
    # Step 5: finalize — best point in physical frame
    # ------------------------------------------------------------------ #
    def finalize(self) -> Result:
        if not self._X_phys:
            raise RuntimeError("Campaign has no data; cannot finalize.")
        history = self.history_df()
        # Utilities choose the best; reports retain the measured physical value.
        idx = int(np.argmax(self.space.objective_specs[0].to_utility(history["y"].to_numpy())))
        best_row = history.iloc[idx]
        best_x_phys = {p: best_row[p].item() if isinstance(best_row[p], np.generic)
                       else best_row[p] for p in self.space.param_names}
        result = Result(
            best_x_physical=best_x_phys,
            best_y_physical=float(best_row["y"]),
            history_df=history,
            n_doe=sum(1 for k in self._trial_kind if k == "doe"),
            n_bo=sum(1 for k in self._trial_kind if k == "bo"),
            objectives=list(self.space.objectives),
            maximize=list(self.space.maximize),
            knowledge_summary=self.knowledge.to_dict()["items"],
            knowledge=self.knowledge.to_envelope(),
            param_units={p.name: p.unit for p in self.space.params},
            param_kinds={p.name: p.kind for p in self.space.params},
        )
        if not self.knowledge.items:
            result.notes.append(
                "No domain knowledge applied — plain GP baseline."
            )
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def history_df(self) -> pd.DataFrame:
        if not self._X_phys:
            return pd.DataFrame(columns=list(self.space.param_names) + ["y", "kind"])
        df = pd.concat(self._X_phys, axis=0).reset_index(drop=True)
        df["y"] = self._y_physical
        df["kind"] = self._trial_kind
        return df

    def save_checkpoint(self, path: str | Path) -> None:
        """Save 2.1 atomically, or the explicit lossless legacy archive route.

        Legacy multiobjective/opaque-provider archives retain schema 2.0 only
        while no 2.1-only metadata, pending work or RNG progression would be lost.
        """
        if self._requires_legacy_archive():
            state = self._legacy_archive_state()
            warnings.warn("Saving legacy archival schema 2.0; Phase-1 restore is unavailable for this declaration", stacklevel=2)
        else:
            state = self.to_state()
        atomic_save(path, state)

    def _requires_legacy_archive(self) -> bool:
        from ..knowledge.patterns import builtin_pattern_definitions
        builtins = {(d.pattern, d.version) for d in builtin_pattern_definitions()}
        provider_keys = {(d.pattern, d.version) for p in self.knowledge.provider_records for d in p.definitions}
        return self.space.n_objectives != 1 or any(
            s.enabled and (s.pattern, s.version) not in builtins and
            (s.pattern, s.version) in provider_keys for s in self.knowledge.specs)

    def _legacy_archive_state(self) -> dict:
        if (self._pending is not None or self._ask_iteration != 0 or
                self._rng.bit_generator.state != np.random.default_rng(self.seed).bit_generator.state or
                not self.validate_active or not self.auto_rescue or
                self.validation_interval != 5 or self.validation_min_obs != 10 or
                self._last_validation_n != 0 or self._validation_emitted or
                any(b.Yvar is not None or b.status is not None or b.ids for b in self._observation_batches)):
            raise EngineError(ErrorCode.CHECKPOINT_INCOMPATIBLE,
                "Cannot downgrade 2.1 observation/pending/RNG metadata to a legacy 2.0 archive")
        return {"schema_version": "2.0", "engine_version": ENGINE_VERSION,
            "space": self.space.to_dict(), "knowledge": self.knowledge.to_envelope(),
            "seed": self.seed, "history": self.history_df().to_dict(orient="records"),
            "trial_kind": list(self._trial_kind)}

    def to_state(self) -> dict:
        """Return detached, validated, single-objective schema-2.1 JSON state."""
        if self.space.n_objectives != 1:
            raise EngineError(ErrorCode.CHECKPOINT_INCOMPATIBLE,
                "Phase-1 Campaign state requires exactly one objective")
        pending = None if self._pending is None else {
            "ids": list(self._pending.ids), "X": self._pending.X.astype(object).values.tolist()}
        return validate_state({"schema_version": CHECKPOINT_SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION, "space": self.space.to_dict(),
            "knowledge": self.knowledge.to_envelope(), "seed": self.seed,
            "observations": [batch_to_state(b) for b in self._observation_batches],
            "pending": pending, "trial_kind": list(self._trial_kind),
            "rng_state": self._rng.bit_generator.state, "ask_iteration": self._ask_iteration,
            "validation": {"active": self.validate_active, "interval": self.validation_interval,
                "min_obs": self.validation_min_obs, "auto_rescue": self.auto_rescue,
                "last_n": self._last_validation_n, "emitted": [list(x) for x in sorted(self._validation_emitted)]}})

    @classmethod
    def from_state(cls, state: dict) -> "Campaign":
        """Restore declarative 2.1 state, or migrate a known legacy wire format."""
        return restore(state, cls)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "Campaign":
        return cls.from_state(read_json(path))
