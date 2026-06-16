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
        Optional domain knowledge spec. If None (and not `Knowledge().strict()`),
        the Campaign auto-applies `with_random_augment(n=20)` and logs a notice.
    seed : int
        Base random seed.
    """

    def __init__(
        self,
        space: Space,
        knowledge: Knowledge | None = None,
        seed: int = 0,
    ) -> None:
        if space.n_objectives > 1:
            warnings.warn(
                "Multi-objective declared; v0.1 only optimizes the first "
                f"objective: {space.objectives[0]!r}. Other objectives are "
                f"stored in history but not driving acquisition.",
                stacklevel=2,
            )
        self.space = space
        self.knowledge = knowledge if knowledge is not None else Knowledge()
        if not self.knowledge.items and not self.knowledge.is_strict():
            self.knowledge.with_random_augment(n=20)
            self._auto_default_applied = True
        else:
            self._auto_default_applied = False
        # Validate now so any conflict is caught before we run anything.
        self.knowledge.validate()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # History stored as DataFrame (physical X) + tensor (internal Y).
        self._X_phys: list[pd.DataFrame] = []  # list of single-row frames
        self._y_internal: list[float] = []
        self._y_physical: list[float] = []
        self._trial_kind: list[str] = []  # "doe" / "bo"

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
        if self._auto_default_applied:
            log.info(
                "expdoe_dk: no knowledge provided; auto-applied "
                "with_random_augment(n=20) (Category ② safe default). "
                "Use Knowledge().strict() to disable."
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
        )
        if self._auto_default_applied:
            result.notes.append(
                "Auto-applied with_random_augment(n=20) (Category ② default)."
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
