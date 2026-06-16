"""
Knowledge composition API.

Chemists declare what they know in physical-units intuition. The Campaign
later resolves each item into GP-frame artifacts (mean function, virtual
points, hyperparameter priors), routing every translation through
`_frame.flip_for_minimize` to avoid the D-bug class of errors.

Lessons from DOEGP Plan 2 hard-coded as safer defaults:
  - `frozen=True` is the default for Arrhenius and Quadratic means (avoid
    MLE absorption from Exp-7).
  - `epsilon="auto"` for monotone is recommended (Exp-14 lengthscale rule).
  - `with_random_augment(n=20)` is the safest baseline (Cat ②, +52~77%
    across 2D/4D/6D).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

from ._frame import PhysicalEffect, flip_for_minimize, InternalDirection
from .monotone import epsilon_from_prior
from .validators import (
    MonotoneCheckResult,
    MonotoneViolationWarning,
    ShapeCheckResult,
    ShapePriorMismatchWarning,
    check_monotone_assumption,
    check_shape_prior_fit,
)

# --------------------------------------------------------------------- #
# Knowledge item dataclasses (JSON-serializable, no torch tensors stored)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ArrheniusItem:
    kind: str = "arrhenius"
    param: str = ""
    frozen: bool = True
    activation_energy: float = 1.0
    amplitude_init: float = -1.0


@dataclass(frozen=True)
class _QuadraticPeakItem:
    kind: str = "quadratic_peak"
    param: str = ""
    center: float = 0.0
    direction: str = "peak"  # "peak" or "valley"
    frozen: bool = True


@dataclass(frozen=True)
class _MonotoneItem:
    kind: str = "monotone"
    param: str = ""
    effect: PhysicalEffect = "increases_objective"
    n_pairs_per_dim: int = 5
    epsilon: float | Literal["auto"] = "auto"
    delta_norm: float = 0.5


@dataclass(frozen=True)
class _RandomAugmentItem:
    kind: str = "random_augment"
    n: int = 20


@dataclass(frozen=True)
class _GPPriorItem:
    kind: str = "gp_prior"
    lengthscale: Literal["weak", "medium", "strong"] = "medium"


# Map of lengthscale prior strength → (Gamma(a, b) parameters, mode).
# Modes ((a-1)/b) used by `epsilon=auto` resolution.
GP_PRIOR_PRESETS: dict[str, dict[str, Any]] = {
    "weak":   {"ls": (1.5, 0.5),  "os": (2.0, 0.5), "noise": (1.0, 50.0)},
    "medium": {"ls": (3.0, 6.0),  "os": (3.0, 1.5), "noise": (2.0, 200.0)},
    "strong": {"ls": (6.0, 15.0), "os": (3.0, 1.5), "noise": (3.0, 500.0)},
}


def _gamma_mode(a: float, b: float) -> float:
    return max(0.01, (a - 1.0) / b) if a > 1 else 0.05


# --------------------------------------------------------------------- #
# Knowledge class
# --------------------------------------------------------------------- #


class Knowledge:
    """
    Composable container of domain-knowledge specs.

    All `with_*` methods return self for chaining and the spec is stored as
    a frozen dataclass (no torch tensors, JSON-serializable). The Campaign
    resolves them at fit time, applying frame translation and ε auto-tune.
    """

    def __init__(self) -> None:
        self._items: list[Any] = []
        self._strict = False  # set by Knowledge.strict() to disable auto-defaults

    # ------------------------------------------------------------------ #
    # Composition (chainable)
    # ------------------------------------------------------------------ #
    def with_arrhenius(
        self,
        param: str,
        *,
        frozen: bool = True,
        activation_energy: float = 1.0,
        amplitude_init: float = -1.0,
    ) -> "Knowledge":
        if not frozen:
            warnings.warn(
                "Arrhenius with frozen=False is the LearnableMeanAbsorption "
                "pitfall (Exp-7): the GP MLE absorbs the mean parameters and "
                "the prior adds no value. Keep frozen=True unless you have "
                "a specific reason.",
                stacklevel=2,
            )
        self._items.append(
            _ArrheniusItem(
                param=param,
                frozen=frozen,
                activation_energy=activation_energy,
                amplitude_init=amplitude_init,
            )
        )
        return self

    def with_quadratic_peak(
        self,
        param: str,
        *,
        center: float,
        direction: Literal["peak", "valley"] = "peak",
        frozen: bool = True,
    ) -> "Knowledge":
        if not frozen:
            warnings.warn(
                "QuadraticPeak with frozen=False risks MLE absorption "
                "(Exp-7). Keep frozen=True unless you have a specific reason.",
                stacklevel=2,
            )
        if direction not in ("peak", "valley"):
            raise ValueError(
                f"direction must be 'peak' or 'valley', got {direction!r}."
            )
        self._items.append(
            _QuadraticPeakItem(
                param=param, center=float(center), direction=direction,
                frozen=frozen,
            )
        )
        return self

    def with_monotone(
        self,
        param: str,
        effect: PhysicalEffect,
        *,
        n_pairs_per_dim: int = 5,
        epsilon: float | Literal["auto"] = "auto",
        delta_norm: float = 0.5,
    ) -> "Knowledge":
        if effect not in ("increases_objective", "decreases_objective"):
            raise ValueError(
                f"effect must be 'increases_objective' or "
                f"'decreases_objective', got {effect!r}."
            )
        self._items.append(
            _MonotoneItem(
                param=param,
                effect=effect,
                n_pairs_per_dim=n_pairs_per_dim,
                epsilon=epsilon,
                delta_norm=delta_norm,
            )
        )
        return self

    def with_random_augment(self, n: int = 20) -> "Knowledge":
        self._items.append(_RandomAugmentItem(n=n))
        return self

    def with_gp_prior(
        self,
        lengthscale: Literal["weak", "medium", "strong"] = "medium",
    ) -> "Knowledge":
        if lengthscale not in GP_PRIOR_PRESETS:
            raise ValueError(
                f"lengthscale preset must be one of {list(GP_PRIOR_PRESETS)}."
            )
        self._items.append(_GPPriorItem(lengthscale=lengthscale))
        return self

    def strict(self) -> "Knowledge":
        """Disable any implicit default (e.g. auto random_augment)."""
        self._strict = True
        return self

    # ------------------------------------------------------------------ #
    # Removal (chainable, used by validators' remediation messages)
    # ------------------------------------------------------------------ #
    def drop_monotone(self, param: str | None = None) -> "Knowledge":
        """
        Remove ``with_monotone`` item(s).

        - ``drop_monotone()`` removes all monotone items.
        - ``drop_monotone("T")`` removes only items declared for parameter ``T``.
        """
        self._items = [
            it
            for it in self._items
            if not (
                getattr(it, "kind", None) == "monotone"
                and (param is None or getattr(it, "param", None) == param)
            )
        ]
        return self

    def drop(self, kind: str, param: str | None = None) -> "Knowledge":
        """Remove items of a given kind (and optionally a specific param)."""
        self._items = [
            it
            for it in self._items
            if not (
                getattr(it, "kind", None) == kind
                and (param is None or getattr(it, "param", None) == param)
            )
        ]
        return self

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def items(self) -> list[Any]:
        return list(self._items)

    def has_kind(self, kind: str) -> bool:
        return any(getattr(it, "kind", None) == kind for it in self._items)

    def items_of(self, kind: str) -> list[Any]:
        return [it for it in self._items if getattr(it, "kind", None) == kind]

    def is_strict(self) -> bool:
        return self._strict

    # ------------------------------------------------------------------ #
    # Validation (called by Campaign once all with_* are done)
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise EpsilonConflictError or similar if config is unsafe."""
        gp_prior_items = self.items_of("gp_prior")
        monotone_items = self.items_of("monotone")

        if gp_prior_items and monotone_items:
            preset = gp_prior_items[-1].lengthscale
            a, b = GP_PRIOR_PRESETS[preset]["ls"]
            ls_mode = _gamma_mode(a, b)
            min_eps = 0.3 * ls_mode
            for m in monotone_items:
                if m.epsilon != "auto":
                    if float(m.epsilon) < min_eps - 1e-9:
                        raise EpsilonConflictError(
                            f"with_monotone(param={m.param!r}, "
                            f"epsilon={m.epsilon}) is too small for "
                            f"with_gp_prior(lengthscale={preset!r}) "
                            f"(prior mode={ls_mode:.3f}, minimum safe "
                            f"ε={min_eps:.3f}). Use epsilon='auto' or "
                            f"raise epsilon to ≥ {min_eps:.3f}. "
                            f"See AGENT_KNOWLEDGE.md Exp-14 for the rule."
                        )

    def resolve_epsilon(self, monotone: _MonotoneItem) -> float:
        """Translate epsilon='auto' to a concrete value given any GP prior."""
        if monotone.epsilon != "auto":
            return float(monotone.epsilon)
        gp_prior_items = self.items_of("gp_prior")
        if gp_prior_items:
            a, b = GP_PRIOR_PRESETS[gp_prior_items[-1].lengthscale]["ls"]
            return epsilon_from_prior(_gamma_mode(a, b))
        return epsilon_from_prior(None)

    # ------------------------------------------------------------------ #
    # JSON serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "strict": self._strict,
            "items": [
                {**it.__dict__} for it in self._items
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Knowledge":
        k = cls()
        if d.get("strict", False):
            k.strict()
        for raw in d.get("items", []):
            kind = raw.get("kind")
            if kind == "arrhenius":
                k.with_arrhenius(
                    raw["param"],
                    frozen=raw.get("frozen", True),
                    activation_energy=raw.get("activation_energy", 1.0),
                    amplitude_init=raw.get("amplitude_init", -1.0),
                )
            elif kind == "quadratic_peak":
                k.with_quadratic_peak(
                    raw["param"],
                    center=raw["center"],
                    direction=raw.get("direction", "peak"),
                    frozen=raw.get("frozen", True),
                )
            elif kind == "monotone":
                k.with_monotone(
                    raw["param"],
                    effect=raw["effect"],
                    n_pairs_per_dim=raw.get("n_pairs_per_dim", 5),
                    epsilon=raw.get("epsilon", "auto"),
                    delta_norm=raw.get("delta_norm", 0.5),
                )
            elif kind == "random_augment":
                k.with_random_augment(n=raw.get("n", 20))
            elif kind == "gp_prior":
                k.with_gp_prior(lengthscale=raw.get("lengthscale", "medium"))
            else:
                warnings.warn(f"Unknown knowledge kind: {kind!r}", stacklevel=2)
        return k


# --------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------- #
class EpsilonConflictError(ValueError):
    """Raised when monotone ε is too small relative to the GP lengthscale prior."""


class LearnableMeanAbsorptionWarning(UserWarning):
    """Issued when a learnable mean function risks MLE absorption."""


__all__ = [
    "Knowledge",
    "EpsilonConflictError",
    "LearnableMeanAbsorptionWarning",
    "MonotoneViolationWarning",
    "ShapePriorMismatchWarning",
    "MonotoneCheckResult",
    "ShapeCheckResult",
    "check_monotone_assumption",
    "check_shape_prior_fit",
    "GP_PRIOR_PRESETS",
]
