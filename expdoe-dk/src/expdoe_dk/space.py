"""Compatibility facade for the composed experiment-space domain model.

``Parameter`` and ``Space`` are the domain implementations. The public
``LinearConstraint`` remains the v0.4 two-sided adapter; the v0.5 declarative
one-sided type is exported from :mod:`expdoe_dk.domain`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .domain import Parameter, Space


@dataclass(frozen=True)
class LinearConstraint:
    """Legacy two-sided physical-unit linear constraint.

    The declarative v0.5 constraint has the distinct constructor
    ``domain.LinearConstraint(name, coefficients, operator, bound, ...)``.
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

    def evaluate(self, row: dict[str, object]) -> float:
        return sum(
            coefficient * float(row.get(name, 0.0))
            for name, coefficient in self.coeffs.items()
        )

    def satisfied(self, row: dict[str, object], tol: float = 1e-9) -> bool:
        value = self.evaluate(row)
        return (self.lower - tol) <= value <= (self.upper + tol)

    def describe(self) -> str:
        if self.name:
            return self.name
        terms = " + ".join(
            f"{coefficient:g}·{name}" for name, coefficient in self.coeffs.items()
        )
        lower = "" if not math.isfinite(self.lower) else f"{self.lower:g} ≤ "
        upper = "" if not math.isfinite(self.upper) else f" ≤ {self.upper:g}"
        return f"{lower}{terms}{upper}"


__all__ = ["LinearConstraint", "Parameter", "Space"]
