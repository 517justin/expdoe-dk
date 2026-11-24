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
from collections.abc import Sequence
from copy import deepcopy
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
from .multiobjective import MultiObjectiveConfig
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
    best_x_physical: dict | None
    best_y_physical: float | None
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
    history_columns: dict[str, dict[str, str]] | None = None

    def to_dict(self) -> dict:
        result = {
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
        if self.history_columns is not None:
            result["history_columns"] = deepcopy(self.history_columns)
        return result

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
        multiobjective: MultiObjectiveConfig | None = None,
    ) -> None:
        if multiobjective is not None:
            if not isinstance(multiobjective, MultiObjectiveConfig):
                raise EngineError(ErrorCode.CONFIG_INVALID, "Expected MultiObjectiveConfig or None")
            multiobjective.validate_for_space(space)
        self._multiobjective = (multiobjective or MultiObjectiveConfig()) if space.n_objectives > 1 else None
        self._geometry = None
        self._catalog: pd.DataFrame | None = None
        self._catalog_ids: tuple[str, ...] | None = None
        self._resolved_declaration: dict | None = None
        self.space = space
        # No auto-default: an empty Knowledge means "plain GP". The library
        # never injects regularization on the user's behalf (see class Notes).
        self.knowledge = knowledge if knowledge is not None else Knowledge()
        # v0.3: rescue (or raise on) Exp-14 epsilon conflicts before any
        # acquisition is built.
        self.auto_rescue = bool(auto_rescue)
        if space.n_objectives > 1:
            self._validate_multiobjective_knowledge()
        else:
            self.knowledge.validate(auto_rescue=self.auto_rescue)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # History stored as DataFrame (physical X) + tensor (internal Y).
        self._X_phys: list[pd.DataFrame] = []  # list of single-row frames
        self._y_internal: list[float] = []
        self._y_physical: list[float] = []
        self._trial_kind: list[str] = []  # "doe" / "bo"
        self._observation_batches: list[ObservationBatch] = []
        self._observation_ids: list[tuple[str, ...]] = []
        self._batch_kinds: list[str] = []
        self._pending: PendingBatch | None = None
        self._ask_iteration = 0

        # v0.2 — empirical validators (run on tell, throttled by interval).
        self.validate_active = bool(validate)
        self.validation_interval = max(1, int(validation_interval))
        self.validation_min_obs = max(3, int(validation_min_obs))
        self._last_validation_n = 0
        # Cache of warnings already emitted so we don't spam every K obs.
        self._validation_emitted: set[tuple[str, ...]] = set()
        self._validation_diagnostics: list[dict] = []

    @property
    def multiobjective(self) -> MultiObjectiveConfig | None:
        return self._multiobjective

    @property
    def validation_diagnostics(self) -> list[dict]:
        return deepcopy(self._validation_diagnostics)

    def _declaration(self) -> dict:
        return {"space": self.space.to_dict(), "knowledge": self.knowledge.to_envelope(),
                "multiobjective": self.multiobjective.to_dict()}

    def _check_frozen_declaration(self) -> None:
        if self._resolved_declaration is not None and self._declaration() != self._resolved_declaration:
            raise EngineError(ErrorCode.CONFIG_INVALID,
                "Resolved multiobjective settings are frozen; create a new Campaign")

    def _successful_count(self) -> int:
        return sum(len(b.successful().X) for b in self._observation_batches)

    def _validate_multiobjective_knowledge(self) -> None:
        from .knowledge_runtime import compile_runtime
        from .objective_routing import project_objective
        compile_runtime(self.space, self.knowledge, None)
        envelope = self.knowledge.to_envelope()
        specs = {spec["pattern_id"]: spec for spec in envelope["specs"]}
        for name in self.space.objectives:
            _, _, projected = project_objective(self.space, None, self.knowledge, name)
            projected.validate(auto_rescue=self.auto_rescue)
            for spec in projected.specs:
                if spec.pattern == "monotone" and spec.version == "1.0":
                    original = specs[spec.pattern_id]["parameters"]
                    if original["epsilon"] != "auto":
                        original["epsilon"] = max(original["epsilon"], spec.parameters["epsilon"])
        self.knowledge = Knowledge.from_envelope(envelope, registry=self.knowledge.registry)

    @property
    def history_columns(self) -> dict[str, dict[str, str]]:
        """Map each named role to a unique flat history column (multiobjective)."""
        used = set(self.space.param_names) | set(self.space.objectives) | {"trial_id", "status", "kind"}
        assigned = set(self.space.objectives)
        columns = {"parameters": {}, "objectives": {name: name for name in self.space.objectives}, "metadata": {}}
        for role, names, prefix in (("parameters", self.space.param_names, "parameter:"),
                                     ("metadata", ("trial_id", "status", "kind"), "metadata:")):
            for name in names:
                column = name
                if column in assigned:
                    column = prefix + name
                    while column in used:
                        column = prefix + column
                columns[role][name] = column
                assigned.add(column)
                used.add(column)
        return columns

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
    def tell(self, X_phys: pd.DataFrame, y: np.ndarray | list[float] | pd.DataFrame) -> None:
        """
        Record results from user-run experiments.

        Parameters
        ----------
        X_phys : DataFrame
            Rows of physical-unit parameter values. Columns must match space.param_names.
        y : array-like or DataFrame
            Same length as X_phys, in physical units. Multiobjective responses
            require a DataFrame with exactly the named objective columns.
        """
        if self.space.n_objectives > 1:
            if not isinstance(y, pd.DataFrame):
                raise EngineError(ErrorCode.CONFIG_INVALID,
                    "Multiobjective tell requires a named response DataFrame")
            self.tell_batch(ObservationBatch(X_phys, y))
            return
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
        self.tell_batch(observations)

    def tell_batch(self, observations: ObservationBatch) -> None:
        """Record physical outcomes, preserving every status and audit field.

        Only successful rows enter legacy history/results or model training;
        all rows remain visible to recommendation duplicate exclusion.
        """
        if not isinstance(observations, ObservationBatch):
            raise EngineError(ErrorCode.CONFIG_INVALID, "Expected ObservationBatch")
        if self.space.n_objectives > 1:
            self._check_frozen_declaration()
            if set(observations.Y.columns) != set(self.space.objectives):
                raise EngineError(ErrorCode.CONFIG_INVALID, "Response columns must exactly match space.objectives")
            variance = observations.Yvar
            observations = ObservationBatch(observations.X, observations.Y[self.space.objectives],
                Yvar=None if variance is None else variance[self.space.objectives],
                status=observations.status, ids=observations.ids)
        if list(observations.Y.columns) != self.space.objectives:
            raise EngineError(ErrorCode.CONFIG_INVALID, "Response columns must exactly match space.objectives")
        validate_frame(self.space, observations.X)
        previous_ids = ({identifier for ids in self._observation_ids for identifier in ids}
                        if self.space.n_objectives > 1 else
                        {identifier for batch in self._observation_batches for identifier in batch.ids})
        if previous_ids.intersection(observations.ids):
            raise EngineError(ErrorCode.CONFIG_INVALID, "Observation IDs must be unique across batches")
        # Validate serializability before mutation, including unsuccessful values.
        from .checkpoint import _strict_json
        _strict_json(batch_to_state(observations))
        self._record_observations(observations)

    def _record_observations(self, observations: ObservationBatch) -> None:
        if self.space.n_objectives > 1:
            # Allocate once, separately from caller IDs, so appends cannot rename
            # geometry's source prefix or turn anonymous returns into pending IDs.
            used = {identifier for ids in self._observation_ids for identifier in ids}
            if self._pending is not None:
                used.update(self._pending.ids)
            identifiers = list(observations.ids)
            if not identifiers:
                for row in range(len(observations.X)):
                    identifier = f"unidentified:{len(self._observation_batches)}:{row}"
                    while identifier in used:
                        identifier += ":"
                    used.add(identifier)
                    identifiers.append(identifier)
            kind = "doe" if not self._observation_batches else "bo"
            previous_pending = self._pending
            previous_validation = (self._last_validation_n, self._validation_emitted.copy(),
                                   deepcopy(self._validation_diagnostics))
            trial_count = len(self._trial_kind)
            self._observation_batches.append(observations)
            self._observation_ids.append(tuple(identifiers))
            self._batch_kinds.append(kind)
            self._trial_kind.extend([kind] * len(observations.successful().X))
            try:
                if self._pending is not None and observations.ids:
                    keep = [i for i, identifier in enumerate(self._pending.ids) if identifier not in observations.ids]
                    self._pending = (PendingBatch(tuple(self._pending.ids[i] for i in keep), self._pending.X.iloc[keep])
                                     if keep else None)
                self._maybe_run_validators()
            except Exception:
                self._observation_batches.pop()
                self._observation_ids.pop()
                self._batch_kinds.pop()
                del self._trial_kind[trial_count:]
                self._pending = previous_pending
                self._last_validation_n, self._validation_emitted, self._validation_diagnostics = previous_validation
                raise
            return
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
        if (self.space.n_objectives > 1 and known and not any(known)
                and any(b.Yvar is not None and b.Yvar.notna().to_numpy().any() for b in batches)):
            raise EngineError(ErrorCode.CONFIG_INVALID,
                "Recorded unsuccessful variance cannot be combined with missing successful variance")
        variance = None
        if known and all(known):
            if self.space.n_objectives > 1 and any(
                    b.Yvar is None or b.Yvar.isna().to_numpy().any() for b in batches):
                # Object columns retain None / NaN / pd.NA across pandas versions;
                # absent unsuccessful-batch variances are unknown, never zero.
                variance = pd.DataFrame({name: pd.Series([
                    value for b in batches for value in
                    ([None] * len(b.X) if b.Yvar is None else b.Yvar[name].tolist())
                ], dtype=object) for name in self.space.objectives})
            else:
                variance = pd.concat([b.Yvar if b.Yvar is not None else
                    pd.DataFrame(0., index=range(len(b.X)), columns=self.space.objectives)
                    for b in batches], ignore_index=True)
        status = pd.concat([b.status if b.status is not None else
            pd.Series(["success"] * len(b.X), dtype=object) for b in batches], ignore_index=True)
        identifiers = []
        used = {identifier for b in batches for identifier in b.ids}
        for index, batch in enumerate(batches):
            if self.space.n_objectives > 1:
                identifiers.extend(self._observation_ids[index])
                continue
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
    def ask(self, q: int = 1, iteration: int | None = None, *,
            candidates: pd.DataFrame | None = None,
            candidate_ids: Sequence[str] | None = None) -> pd.DataFrame:
        """Return physical candidates through the public diagnostic boundary."""
        return self.ask_with_diagnostics(q=q, iteration=iteration,
            candidates=candidates, candidate_ids=candidate_ids).frame

    def ask_with_diagnostics(self, q: int = 1, iteration: int | None = None, *,
                             candidates: pd.DataFrame | None = None,
                             candidate_ids: Sequence[str] | None = None) -> RecommendationResult:
        """Fit the production GP and return candidates plus truthful diagnostics."""
        from .recommendation import recommend

        if iteration is not None and (type(iteration) is not int or iteration < 0):
            raise EngineError(ErrorCode.CONFIG_INVALID, "iteration must be a non-negative integer")
        multi = self.space.n_objectives > 1
        if not (self._successful_count() if multi else len(self._X_phys)):
            raise EngineError(ErrorCode.INSUFFICIENT_DATA,
                "Campaign.ask requires initial observations via tell()")
        observations = self._combined_observations()
        count = self._successful_count() if multi else len(self._y_physical)
        iteration_seed = self.seed + (count + self._ask_iteration if iteration is None else iteration)
        if multi:
            from .recommendation import _recommend_resolved
            from .objective_geometry import ObjectiveGeometry
            self._check_frozen_declaration()
            catalog, ids = self._prepare_campaign_catalog(candidates, candidate_ids)
            declaration = self._declaration()
            result = _recommend_resolved(self.space, observations, knowledge=self.knowledge,
                pending=self._pending, q=q, seed=iteration_seed, multiobjective=self.multiobjective,
                geometry=self._geometry, candidates=catalog, candidate_ids=ids)
            geometry = ObjectiveGeometry.from_dict(result.diagnostics["geometry"])
            self._geometry, self._catalog, self._catalog_ids = geometry, catalog, ids
            self._resolved_declaration = declaration
        else:
            result = recommend(self.space, observations, knowledge=self.knowledge,
                pending=self._pending, q=q, seed=iteration_seed,
                candidates=candidates, candidate_ids=candidate_ids)
        self._ask_iteration = self._ask_iteration + 1 if iteration is None else iteration + 1
        return result

    def _prepare_campaign_catalog(self, candidates, candidate_ids):
        from .candidates import prepare_catalog
        if candidates is None:
            if candidate_ids is not None:
                raise EngineError(ErrorCode.CONFIG_INVALID, "candidate_ids requires candidates")
            if self._catalog is None:
                return None, None
            candidates, candidate_ids = self._catalog, self._catalog_ids
        references = tuple(batch.X for batch in self._observation_batches)
        if self._pending is not None:
            references += (self._pending.X,)
        catalog, _ = prepare_catalog(self.space, candidates, candidate_ids, references=references)
        ids = tuple(sorted(candidate_ids))
        if self._catalog is not None and (ids != self._catalog_ids or not catalog.equals(self._catalog)):
            raise EngineError(ErrorCode.CONFIG_INVALID, "Campaign catalog is fixed; create a new Campaign")
        return catalog, ids

    # ------------------------------------------------------------------ #
    # Step 4: run — full DoE + BO automation
    # ------------------------------------------------------------------ #
    def run(
        self,
        oracle: Callable[[pd.DataFrame], np.ndarray | pd.DataFrame],
        n_doe: int = 12,
        n_iter: int = 20,
        q: int = 1,
        doe_method: str = "lhs_maximin",
        checkpoint: str | Path | None = None,
        *,
        candidates: pd.DataFrame | None = None,
        candidate_ids: Sequence[str] | None = None,
    ) -> Result:
        # Initial DoE.
        multi = self.space.n_objectives > 1
        if multi and checkpoint is not None:
            raise EngineError(ErrorCode.CHECKPOINT_INCOMPATIBLE,
                "Multiobjective checkpoint support is not available yet")
        if multi:
            self._check_frozen_declaration()
            candidates, candidate_ids = self._prepare_campaign_catalog(candidates, candidate_ids)
        elif candidate_ids is not None and candidates is None:
            raise EngineError(ErrorCode.CONFIG_INVALID, "candidate_ids requires candidates")
        if candidates is None:
            doe = self.suggest_doe(n=n_doe, method=doe_method)
        else:
            from .candidates import prepare_catalog
            from .knowledge_runtime import compile_runtime
            from ..doe.design import candidate_keys
            catalog, _ = prepare_catalog(self.space, candidates, candidate_ids)
            runtime = compile_runtime(self.space, self.knowledge, None)
            design_space = self.space.with_constraints(*runtime["parameter_constraints"])
            feasible = catalog.loc[design_space.feasibility_mask(catalog).cpu().numpy()].reset_index(drop=True)
            excluded = set()
            for batch in self._observation_batches:
                excluded.update(candidate_keys(batch.X, self.space))
            if self._pending is not None:
                excluded.update(candidate_keys(self._pending.X, self.space))
            feasible = feasible.iloc[[i for i in range(len(feasible))
                if not candidate_keys(feasible.iloc[[i]], self.space) & excluded]]
            if type(n_doe) is not int or n_doe < 1:
                raise EngineError(ErrorCode.CONFIG_INVALID, "n_doe must be a positive integer")
            if len(feasible) < n_doe:
                raise EngineError(ErrorCode.NO_FEASIBLE_CANDIDATE, "Catalog has too few initial feasible rows")
            doe = feasible.iloc[np.random.default_rng(self.seed).choice(len(feasible), n_doe, replace=False)].reset_index(drop=True)
        y_doe = oracle(doe)
        if not multi:
            y_doe = np.asarray(y_doe, dtype=np.float64).flatten()
        self.tell(doe, y_doe)

        # BO iterations.
        for it in range(n_iter):
            next_pts = self.ask(q=q, iteration=it, candidates=candidates, candidate_ids=candidate_ids)
            y_next = oracle(next_pts)
            if not multi:
                y_next = np.asarray(y_next, dtype=np.float64).flatten()
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
        n = self._successful_count() if self.space.n_objectives > 1 else len(self._y_internal)
        if n < self.validation_min_obs:
            return
        if n < self._last_validation_n + self.validation_interval:
            return
        self._last_validation_n = n
        if self.space.n_objectives > 1:
            self._run_multiobjective_validators()
            return
        self._run_monotone_validators()
        self._run_shape_validators()

    def _run_multiobjective_validators(self) -> None:
        from .knowledge_runtime import compile_runtime
        from .objective_routing import project_objective
        observations = self._combined_observations()
        compile_runtime(self.space, self.knowledge, observations)
        diagnostics = []
        for name in self.space.objectives:
            space, batch, knowledge = project_objective(self.space, observations, self.knowledge, name)
            successful = batch.successful()
            context = (space, successful, knowledge, name)
            checked = self._run_monotone_validators(context=context)
            shape_supported = (space.objective_specs[0].direction in {"minimize", "maximize"}
                               and all(p.kind in {"continuous", "integer"} and p.bounds is not None
                                       for p in space.params))
            if shape_supported:
                checked.update(self._run_shape_validators(context=context))
            for spec in knowledge.specs:
                if not spec.enabled:
                    continue
                results = [checked[spec.pattern_id]] if spec.pattern_id in checked else []
                supported = spec.version == "1.0" and spec.pattern in {"monotone", "arrhenius", "quadratic_peak"}
                unsupported_shape = spec.pattern in {"arrhenius", "quadratic_peak"} and not shape_supported
                supported = supported and not unsupported_shape
                status = "checked" if results and all(r is not None for r in results) else (
                    "not_checked" if supported else "unsupported")
                diagnostics.append({"objective_name": name, "pattern_id": spec.pattern_id,
                    "pattern": spec.pattern, "status": status,
                    "violation": any(r.violation for r in results if r is not None) if status == "checked" else None,
                    "reason": None if status == "checked" else (
                        "shape_validator_requires_bounded_numeric_min_max_space" if unsupported_shape else
                        "insufficient_or_inapplicable_data" if supported else "no_empirical_validator")})
        self._validation_diagnostics = diagnostics

    @staticmethod
    def _validator_items(knowledge, kinds, *, scoped):
        if not scoped:
            return [(None, item) for item in knowledge.items if item.kind in kinds]
        from ..knowledge import _legacy_item_from_spec
        # The legacy item view omits identity; keep each original spec beside it.
        return [(spec.pattern_id, item) for spec in knowledge.specs if spec.enabled
                if (item := _legacy_item_from_spec(spec)) is not None and item.kind in kinds]

    def _run_monotone_validators(self, *, context=None) -> dict:
        space, knowledge = self.space, self.knowledge
        objective_name = None
        if context is not None:
            space, batch, knowledge, objective_name = context
        mono_items = self._validator_items(knowledge, {"monotone"}, scoped=context is not None)
        checked = {}
        if not mono_items:
            return checked
        X_phys = (self.history_df()[space.param_names] if context is None else batch.X).to_numpy()
        y_user = np.asarray(self._y_physical if context is None else batch.Y.iloc[:, 0], dtype=np.float64)
        for pattern_id, item in mono_items:
            try:
                idx = space.param_names.index(item.param)
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
            checked[("monotone", item.param) if context is None else pattern_id] = result
            if result is None or not result.violation:
                continue
            key = ("monotone", item.param) if objective_name is None else (objective_name, pattern_id, "monotone", item.param)
            if key in self._validation_emitted:
                continue
            self._validation_emitted.add(key)
            message = result.to_warning()
            warnings.warn(message if objective_name is None else type(message)(f"{objective_name}: [{pattern_id}] {message}"), stacklevel=3)
        return checked

    def _run_shape_validators(self, *, context=None) -> dict:
        space, knowledge = self.space, self.knowledge
        objective_name = None
        if context is not None:
            space, batch, knowledge, objective_name = context
        checked = {}
        shape_items = self._validator_items(knowledge, {"arrhenius", "quadratic_peak"}, scoped=context is not None)
        if not shape_items:
            return checked
        # Build mean predictions in unit / Y_norm space.
        X_phys = space.to_tensor(
            self.history_df()[space.param_names] if context is None else batch.X
        )
        X_unit = space.physical_to_unit(X_phys)
        values = (self._y_internal if context is None else
                  -space.objective_specs[0].to_utility(batch.Y.iloc[:, 0].to_numpy(dtype=float)))
        y_int = torch.tensor(values, dtype=torch.float64)
        if y_int.std() < 1e-9:
            return checked
        y_norm = ((y_int - y_int.mean()) / y_int.std()).cpu().numpy()

        for pattern_id, item in shape_items:
            try:
                dim_idx = space.param_names.index(item.param)
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
                param = space.param_by_name(item.param)
                lo, hi = param.bounds
                center_unit = (item.center - lo) / (hi - lo)
                curvature_signs = [0.0] * space.n_dims
                if item.direction == "peak":
                    sign_internal = +1.0 if space.maximize[0] else -1.0
                else:
                    sign_internal = -1.0 if space.maximize[0] else +1.0
                curvature_signs[dim_idx] = sign_internal
                centers = [0.5] * space.n_dims
                centers[dim_idx] = float(center_unit)
                fn = QuadraticMeanFrozen(
                    input_dim=space.n_dims,
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
            checked[(kind, item.param) if context is None else pattern_id] = result
            if result is None or not result.violation:
                continue
            key = (kind, item.param) if objective_name is None else (objective_name, pattern_id, kind, item.param)
            if key in self._validation_emitted:
                continue
            self._validation_emitted.add(key)
            message = result.to_warning()
            warnings.warn(message if objective_name is None else type(message)(f"{objective_name}: [{pattern_id}] {message}"), stacklevel=3)
        return checked

    # ------------------------------------------------------------------ #
    # Step 5: finalize — best point in physical frame
    # ------------------------------------------------------------------ #
    def finalize(self) -> Result:
        multi = self.space.n_objectives > 1
        if not (self._observation_batches if multi else self._X_phys):
            raise RuntimeError("Campaign has no data; cannot finalize.")
        history = self.history_df()
        # Utilities choose the best; reports retain the measured physical value.
        best_x_phys, best_y_phys = None, None
        if not multi:
            idx = int(np.argmax(self.space.objective_specs[0].to_utility(history["y"].to_numpy())))
            best_row = history.iloc[idx]
            best_x_phys = {p: best_row[p].item() if isinstance(best_row[p], np.generic)
                           else best_row[p] for p in self.space.param_names}
            best_y_phys = float(best_row["y"])
        result = Result(
            best_x_physical=best_x_phys,
            best_y_physical=best_y_phys,
            history_df=history,
            n_doe=sum(1 for k in self._trial_kind if k == "doe"),
            n_bo=sum(1 for k in self._trial_kind if k == "bo"),
            objectives=list(self.space.objectives),
            maximize=list(self.space.maximize),
            knowledge_summary=self.knowledge.to_dict()["items"],
            knowledge=self.knowledge.to_envelope(),
            param_units={p.name: p.unit for p in self.space.params},
            param_kinds={p.name: p.kind for p in self.space.params},
            history_columns=self.history_columns if multi else None,
        )
        if not (any(spec.enabled for spec in self.knowledge.specs) if multi else self.knowledge.items):
            result.notes.append(
                "No domain knowledge applied — plain GP baseline."
            )
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def history_df(self) -> pd.DataFrame:
        if self.space.n_objectives > 1:
            columns = self.history_columns
            frames = []
            for batch, ids, kind in zip(self._observation_batches, self._observation_ids, self._batch_kinds):
                frame = pd.concat([batch.X.rename(columns=columns["parameters"]), batch.Y], axis=1)
                frame[columns["metadata"]["trial_id"]] = ids
                frame[columns["metadata"]["status"]] = "success" if batch.status is None else batch.status
                frame[columns["metadata"]["kind"]] = kind
                frames.append(frame)
            return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=[*columns["parameters"].values(), *self.space.objectives, *columns["metadata"].values()]))
        if not self._X_phys:
            return pd.DataFrame(columns=list(self.space.param_names) + ["y", "kind"])
        df = pd.concat(self._X_phys, axis=0).reset_index(drop=True)
        df["y"] = self._y_physical
        df["kind"] = self._trial_kind
        return df

    def save_checkpoint(self, path: str | Path) -> None:
        """Save 2.1 atomically, or the explicit lossless legacy archive route.

        Scalar opaque-provider archives retain schema 2.0 only while no newer
        metadata would be lost. Multiobjective checkpoint writing is unavailable.
        """
        if self.space.n_objectives > 1:
            raise EngineError(ErrorCode.CHECKPOINT_INCOMPATIBLE,
                "Multiobjective checkpoint support is not available; cannot downgrade measured vectors")
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
