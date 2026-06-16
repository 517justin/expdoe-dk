"""
Parameter / LinearConstraint / Space — physical-units parameter space.

This is the chemist's view of the experiment: parameters in real units
(°C, mL, min), discrete steps for lab convenience, linear constraints
like "A must be >= B".

Internally BO always minimizes a normalized scalar in [0,1] unit space.
Frame translation (physical ↔ unit, maximize ↔ minimize) is centralized
in `knowledge/_frame.py` to prevent the D-bug class of errors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor


@dataclass(frozen=True)
class Parameter:
    """
    A single experimental factor in physical units.

    Parameters
    ----------
    name : str
        Identifier used as DataFrame column name. Must be unique within a Space.
    bounds : tuple[float, float]
        (low, high) in physical units, inclusive.
    unit : str
        Display unit, e.g. "°C", "mL", "min". Used in reports only.
    kind : "continuous" or "discrete"
        Continuous = any real value in bounds. Discrete = values on a grid.
    step : float | None
        Required when kind="discrete". Spacing of the discrete grid in
        physical units (e.g. step=1.0 → integer mL, step=0.5 → half-mL).
        For kind="continuous" it must be None.
    log_scale : bool
        If True, sampling and internal modeling treat this dimension on
        log axis. Useful for time, concentration spanning orders of
        magnitude. v0.1: stored but only used by future report layer.
    """

    name: str
    bounds: tuple[float, float]
    unit: str = ""
    kind: Literal["continuous", "discrete"] = "continuous"
    step: float | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        lo, hi = self.bounds
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"Parameter {self.name}: bounds must be finite.")
        if lo >= hi:
            raise ValueError(
                f"Parameter {self.name}: bounds={self.bounds} invalid "
                f"(low must be < high)."
            )
        if self.kind == "discrete":
            if self.step is None or self.step <= 0:
                raise ValueError(
                    f"Parameter {self.name}: discrete requires step > 0, "
                    f"got step={self.step}."
                )
            if self.step > (hi - lo):
                raise ValueError(
                    f"Parameter {self.name}: step={self.step} larger than "
                    f"range {hi - lo}; no valid levels."
                )
        else:
            if self.step is not None:
                raise ValueError(
                    f"Parameter {self.name}: step must be None for continuous."
                )

    @property
    def levels(self) -> np.ndarray:
        """
        Discrete grid points. Raises for continuous params.
        Includes both endpoints when reachable by step.
        """
        if self.kind != "discrete":
            raise AttributeError(f"Parameter {self.name} is continuous.")
        lo, hi = self.bounds
        # Use a small tolerance so the right endpoint is included when it
        # lands exactly on the grid (avoids float arange off-by-one).
        n = int(round((hi - lo) / self.step)) + 1
        return lo + np.arange(n, dtype=np.float64) * self.step

    def snap(self, x: np.ndarray | Tensor) -> np.ndarray | Tensor:
        """Snap continuous values to the nearest valid discrete level."""
        if self.kind != "discrete":
            return x
        lo, _ = self.bounds
        idx = torch.round((torch.as_tensor(x) - lo) / self.step)
        n_levels = len(self.levels)
        idx = idx.clamp(min=0, max=n_levels - 1)
        snapped = lo + idx * self.step
        if isinstance(x, np.ndarray):
            return snapped.cpu().numpy()
        return snapped


@dataclass(frozen=True)
class LinearConstraint:
    """
    Linear inequality constraint on parameter values in PHYSICAL UNITS:

        lower <= sum(coeffs[name] * x[name]) <= upper

    Example: A must be at least 1 mL larger than B:
        LinearConstraint(coeffs={"A": 1.0, "B": -1.0}, lower=1.0)

    Parameters
    ----------
    coeffs : dict[str, float]
        Maps parameter name to its coefficient. Missing names get 0.
    lower, upper : float
        Default ±inf so the constraint becomes one-sided when only one
        bound is given.
    name : str
        Optional label for error messages.
    """

    coeffs: dict[str, float]
    lower: float = -math.inf
    upper: float = math.inf
    name: str = ""

    def __post_init__(self) -> None:
        if not self.coeffs:
            raise ValueError("LinearConstraint: coeffs must be non-empty.")
        if self.lower > self.upper:
            raise ValueError(
                f"LinearConstraint: lower={self.lower} > upper={self.upper}."
            )

    def evaluate(self, row: dict[str, float]) -> float:
        return sum(coef * row.get(k, 0.0) for k, coef in self.coeffs.items())

    def satisfied(self, row: dict[str, float], tol: float = 1e-9) -> bool:
        v = self.evaluate(row)
        return (self.lower - tol) <= v <= (self.upper + tol)

    def describe(self) -> str:
        if self.name:
            return self.name
        terms = " + ".join(f"{coef:g}·{k}" for k, coef in self.coeffs.items())
        lo = "" if not math.isfinite(self.lower) else f"{self.lower:g} ≤ "
        hi = "" if not math.isfinite(self.upper) else f" ≤ {self.upper:g}"
        return f"{lo}{terms}{hi}"


class Space:
    """
    The experiment's parameter space.

    Keeps physical-unit Parameters and Linear constraints together with the
    objective name(s) and direction. Provides physical ↔ unit ([0,1]^d)
    transforms used internally by the BO loop.

    Parameters
    ----------
    params : list[Parameter]
        Ordered list of factors. Order is preserved in DataFrame columns
        and tensor dimensions.
    constraints : list[LinearConstraint] | None
        Optional linear inequalities. Applied row-wise during design
        generation and acquisition optimization.
    objectives : str | list[str]
        Single string for single-objective; list for multi-objective. v0.1
        only uses the first if multiple are given (forward-compat field).
    maximize : bool | list[bool]
        True = "higher is better" (yield, purity); False = "lower is better"
        (impurity, cost). For multi-objective, a list per objective.
    """

    def __init__(
        self,
        params: list[Parameter],
        constraints: list[LinearConstraint] | None = None,
        objectives: str | list[str] = "y",
        maximize: bool | list[bool] = True,
    ) -> None:
        if not params:
            raise ValueError("Space: at least one parameter is required.")
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise ValueError(f"Space: parameter names must be unique, got {names}.")

        self.params: list[Parameter] = list(params)
        self.constraints: list[LinearConstraint] = list(constraints or [])
        self.objectives: list[str] = (
            [objectives] if isinstance(objectives, str) else list(objectives)
        )
        if isinstance(maximize, bool):
            self.maximize: list[bool] = [maximize] * len(self.objectives)
        else:
            self.maximize = list(maximize)
            if len(self.maximize) != len(self.objectives):
                raise ValueError(
                    f"Space: len(maximize)={len(self.maximize)} must match "
                    f"len(objectives)={len(self.objectives)}."
                )

        # Validate constraints reference real parameter names
        param_names = set(names)
        for c in self.constraints:
            for k in c.coeffs:
                if k not in param_names:
                    raise ValueError(
                        f"LinearConstraint references unknown param '{k}'. "
                        f"Known: {sorted(param_names)}."
                    )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def n_dims(self) -> int:
        return len(self.params)

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]

    def param_by_name(self, name: str) -> Parameter:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"Unknown parameter: {name}. Available: {self.param_names}.")

    # ------------------------------------------------------------------ #
    # Bounds tensors
    # ------------------------------------------------------------------ #
    @property
    def lower(self) -> Tensor:
        return torch.tensor([p.bounds[0] for p in self.params], dtype=torch.float64)

    @property
    def upper(self) -> Tensor:
        return torch.tensor([p.bounds[1] for p in self.params], dtype=torch.float64)

    @property
    def bounds_tensor(self) -> Tensor:
        """Shape (2, d): row 0 = lower, row 1 = upper in physical units."""
        return torch.stack([self.lower, self.upper], dim=0)

    # ------------------------------------------------------------------ #
    # Physical ↔ unit transforms
    # ------------------------------------------------------------------ #
    def physical_to_unit(self, X_phys: Tensor) -> Tensor:
        """Linearly map physical-unit values to [0,1]^d."""
        X = torch.as_tensor(X_phys, dtype=torch.float64)
        return (X - self.lower) / (self.upper - self.lower)

    def unit_to_physical(self, X_unit: Tensor) -> Tensor:
        """Map [0,1]^d back to physical units; snap discrete dims."""
        X = torch.as_tensor(X_unit, dtype=torch.float64)
        X_phys = self.lower + X * (self.upper - self.lower)
        for j, p in enumerate(self.params):
            if p.kind == "discrete":
                X_phys[..., j] = p.snap(X_phys[..., j])
        return X_phys

    # ------------------------------------------------------------------ #
    # Feasibility checks (physical units)
    # ------------------------------------------------------------------ #
    def feasibility_mask(self, X_phys: Tensor | pd.DataFrame) -> Tensor:
        """Return bool tensor of shape (n,) indicating which rows are feasible."""
        if isinstance(X_phys, pd.DataFrame):
            rows = X_phys[self.param_names].to_dict(orient="records")
        else:
            X_phys = torch.as_tensor(X_phys, dtype=torch.float64)
            if X_phys.ndim == 1:
                X_phys = X_phys.unsqueeze(0)
            rows = [
                {p.name: float(X_phys[i, j]) for j, p in enumerate(self.params)}
                for i in range(X_phys.shape[0])
            ]
        if not self.constraints:
            return torch.ones(len(rows), dtype=torch.bool)
        return torch.tensor(
            [all(c.satisfied(r) for c in self.constraints) for r in rows],
            dtype=torch.bool,
        )

    # ------------------------------------------------------------------ #
    # DataFrame helpers
    # ------------------------------------------------------------------ #
    def to_dataframe(self, X_phys: Tensor) -> pd.DataFrame:
        X = torch.as_tensor(X_phys, dtype=torch.float64).cpu().numpy()
        return pd.DataFrame(X, columns=self.param_names)

    def to_tensor(self, df: pd.DataFrame) -> Tensor:
        return torch.tensor(df[self.param_names].to_numpy(), dtype=torch.float64)

    # ------------------------------------------------------------------ #
    # JSON serialization (Skill/MCP-ready)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "params": [
                {
                    "name": p.name,
                    "bounds": list(p.bounds),
                    "unit": p.unit,
                    "kind": p.kind,
                    "step": p.step,
                    "log_scale": p.log_scale,
                }
                for p in self.params
            ],
            "constraints": [
                {
                    "coeffs": dict(c.coeffs),
                    "lower": c.lower if math.isfinite(c.lower) else None,
                    "upper": c.upper if math.isfinite(c.upper) else None,
                    "name": c.name,
                }
                for c in self.constraints
            ],
            "objectives": list(self.objectives),
            "maximize": list(self.maximize),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Space":
        params = [
            Parameter(
                name=p["name"],
                bounds=tuple(p["bounds"]),
                unit=p.get("unit", ""),
                kind=p.get("kind", "continuous"),
                step=p.get("step"),
                log_scale=p.get("log_scale", False),
            )
            for p in d["params"]
        ]
        constraints = [
            LinearConstraint(
                coeffs=dict(c["coeffs"]),
                lower=c["lower"] if c.get("lower") is not None else -math.inf,
                upper=c["upper"] if c.get("upper") is not None else math.inf,
                name=c.get("name", ""),
            )
            for c in d.get("constraints", [])
        ]
        return cls(
            params=params,
            constraints=constraints,
            objectives=d.get("objectives", "y"),
            maximize=d.get("maximize", True),
        )

    def __repr__(self) -> str:
        lines = [f"Space(n_dims={self.n_dims}, objectives={self.objectives}, maximize={self.maximize})"]
        for p in self.params:
            extra = ""
            if p.kind == "discrete":
                extra = f", step={p.step}"
            lines.append(f"  - {p.name}: [{p.bounds[0]}, {p.bounds[1]}] {p.unit} ({p.kind}{extra})")
        for c in self.constraints:
            lines.append(f"  ! {c.describe()}")
        return "\n".join(lines)
