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
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..space import Space
from ..knowledge import Knowledge
from ..knowledge._frame import (
    flip_for_minimize,
    internal_to_physical,
    physical_to_internal_best,
)
from ..knowledge.validators import (
    check_monotone_assumption,
    check_shape_prior_fit,
)
from ..knowledge.shape import (
    ArrheniusMeanFrozen,
    QuadraticMeanFrozen,
)
from ..doe import generate as generate_doe
from .gp import build_gp

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

    def to_dict(self) -> dict:
        return {
            "best_x_physical": self.best_x_physical,
            "best_y_physical": self.best_y_physical,
            "n_doe": self.n_doe,
            "n_bo": self.n_bo,
            "objectives": self.objectives,
            "maximize": self.maximize,
            "knowledge_summary": self.knowledge_summary,
            "notes": self.notes,
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

        kind = "doe" if not self._X_phys else "bo"
        for i in range(len(X_phys)):
            self._X_phys.append(X_phys.iloc[[i]].reset_index(drop=True))
            y_user = float(y_arr[i])
            self._y_physical.append(y_user)
            y_int = float(
                physical_to_internal_best(y_user, self.space.maximize[0])
            )
            self._y_internal.append(y_int)
            self._trial_kind.append(kind)

        # v0.2: run empirical validators (Spearman monotone, shape fit) on a
        # cadence so each warning surfaces at most ~once until new data comes.
        self._maybe_run_validators()

    # ------------------------------------------------------------------ #
    # Step 3: ask — propose next experiments
    # ------------------------------------------------------------------ #
    def ask(self, q: int = 1, iteration: int | None = None) -> pd.DataFrame:
        if not self._X_phys:
            raise RuntimeError(
                "Campaign.ask() called before any tell(). Provide initial "
                "data via tell(suggest_doe(n), y_observed) first."
            )

        # Build tensors in unit / Y_norm space.
        X_phys_df = pd.concat(self._X_phys, axis=0).reset_index(drop=True)
        X_phys_t = self.space.to_tensor(X_phys_df)
        X_unit = self.space.physical_to_unit(X_phys_t)
        y_int = torch.tensor(self._y_internal, dtype=torch.float64).unsqueeze(-1)
        y_mean = y_int.mean()
        y_std = y_int.std().clamp(min=1e-6)
        y_norm = (y_int - y_mean) / y_std

        # Augment with virtual points if knowledge has monotone.
        iter_idx = iteration if iteration is not None else len(self._y_internal)
        model, augmenter = build_gp(self.space, self.knowledge, X_unit, y_norm)
        if augmenter is not None:
            n_pairs = max(
                (m.n_pairs_per_dim for m in self.knowledge.items_of("monotone")),
                default=5,
            )
            X_aug, Y_aug = augmenter.augment(
                X_unit, y_norm,
                n_pairs_per_dim=n_pairs,
                iteration=iter_idx,
                bo_seed=self.seed,
            )
            # Rebuild model with augmented data (same kernel / mean / lik).
            model, _ = build_gp(self.space, self.knowledge, X_aug, Y_aug)

        # Random augment (Cat ②): append zero-Y_norm anchor points.
        ra_items = self.knowledge.items_of("random_augment")
        if ra_items:
            n_ra = ra_items[-1].n
            torch.manual_seed(self.seed * 10000 + iter_idx)
            X_ra = torch.rand(n_ra, self.space.n_dims, dtype=torch.float64)
            Y_ra = torch.full((n_ra, 1), float(y_norm.mean()), dtype=torch.float64)
            if model.train_inputs is None:
                X_all, Y_all = X_unit, y_norm
            else:
                X_all = torch.cat([model.train_inputs[0], X_ra], dim=0)
                Y_all = torch.cat([model.train_targets.unsqueeze(-1), Y_ra], dim=0)
            model, _ = build_gp(self.space, self.knowledge, X_all, Y_all)

        # Fit hyperparameters.
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            log.warning(f"GP fit fallback (continuing with defaults): {e}")

        # Acquisition: minimize internal y_norm.
        best_f = y_norm.min()
        acqf = LogExpectedImprovement(model=model, best_f=best_f, maximize=False)
        unit_bounds = torch.stack(
            [
                torch.zeros(self.space.n_dims, dtype=torch.float64),
                torch.ones(self.space.n_dims, dtype=torch.float64),
            ]
        )

        # Optimize acquisition with rejection sampling for constraints.
        candidates: list[pd.DataFrame] = []
        tries = 0
        max_tries = max(20, q * 10)
        while len(candidates) < q and tries < max_tries:
            tries += 1
            try:
                cand_unit, _ = optimize_acqf(
                    acqf,
                    bounds=unit_bounds,
                    q=1,
                    num_restarts=10,
                    raw_samples=256,
                )
            except Exception as e:
                log.warning(f"optimize_acqf failed: {e}; random fallback.")
                cand_unit = torch.rand(1, self.space.n_dims, dtype=torch.float64)

            cand_phys = self.space.unit_to_physical(cand_unit)
            df_cand = self.space.to_dataframe(cand_phys)
            if not bool(self.space.feasibility_mask(df_cand).all().item()):
                # Try one local random nudge.
                continue
            candidates.append(df_cand)

        if len(candidates) < q:
            # Fallback: random feasible sample to fill the batch.
            missing = q - len(candidates)
            extra = generate_doe(
                self.space, n=missing, method="random_uniform",
                seed=self.seed + tries,
            )
            candidates.append(extra)

        return pd.concat(candidates, axis=0).reset_index(drop=True).head(q)

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
        # Best in user frame:
        if self.space.maximize[0]:
            idx = int(history["y"].idxmax())
        else:
            idx = int(history["y"].idxmin())
        best_row = history.iloc[idx]
        best_x_phys = {p: float(best_row[p]) for p in self.space.param_names}
        result = Result(
            best_x_physical=best_x_phys,
            best_y_physical=float(best_row["y"]),
            history_df=history,
            n_doe=sum(1 for k in self._trial_kind if k == "doe"),
            n_bo=sum(1 for k in self._trial_kind if k == "bo"),
            objectives=list(self.space.objectives),
            maximize=list(self.space.maximize),
            knowledge_summary=self.knowledge.to_dict()["items"],
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
        p = Path(path)
        state = {
            "space": self.space.to_dict(),
            "knowledge": self.knowledge.to_dict(),
            "seed": self.seed,
            "history": self.history_df().to_dict(orient="records"),
            "trial_kind": list(self._trial_kind),
        }
        p.write_text(json.dumps(state, indent=2))

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "Campaign":
        state = json.loads(Path(path).read_text())
        space = Space.from_dict(state["space"])
        knowledge = Knowledge.from_dict(state["knowledge"])
        c = cls(space=space, knowledge=knowledge, seed=state["seed"])
        # Re-tell from history.
        history = pd.DataFrame(state["history"])
        if not history.empty:
            X_phys = history[list(space.param_names)]
            y = history["y"].to_numpy()
            c.tell(X_phys, y)
            c._trial_kind = list(state.get("trial_kind", c._trial_kind))
        return c
