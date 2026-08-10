"""Objective definitions and legacy objective normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from expdoe_dk.errors import EngineError, ErrorCode


@dataclass(frozen=True)
class Objective:
    """A measured response and how optimization should value it."""

    name: str
    direction: Literal["maximize", "minimize", "target"]
    target: float | tuple[float, float] | None = None
    unit: str = ""
    priority: int = 0

    def __post_init__(self) -> None:
        if self.direction not in ("maximize", "minimize", "target"):
            raise EngineError(
                ErrorCode.CONFIG_INVALID,
                f"Objective {self.name}: unknown direction {self.direction!r}",
            )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise EngineError(
                ErrorCode.CONFIG_INVALID,
                "Objective priority must be a non-negative integer rank",
            )
        if self.priority < 0:
            raise EngineError(
                ErrorCode.CONFIG_INVALID,
                "Objective priority must be a non-negative integer rank",
            )
        if self.direction == "target":
            if self.target is None:
                raise EngineError(
                    ErrorCode.CONFIG_INVALID,
                    "Target objectives require a target value",
                )
            if isinstance(self.target, tuple):
                if len(self.target) != 2 or self.target[0] > self.target[1]:
                    raise EngineError(
                        ErrorCode.CONFIG_INVALID,
                        "Target ranges must be ordered (low, high) pairs",
                    )
        elif self.target is not None:
            raise EngineError(
                ErrorCode.CONFIG_INVALID,
                f"Objective {self.direction} does not accept a target",
            )

    def to_utility(self, values: np.ndarray) -> np.ndarray:
        """Return higher-is-better utilities in the objective's native units."""
        y = np.asarray(values, dtype=np.float64)
        if self.direction == "maximize":
            return y
        if self.direction == "minimize":
            return -y
        if isinstance(self.target, tuple):
            lo, hi = self.target
            return -np.maximum.reduce((lo - y, y - hi, np.zeros_like(y)))
        return -np.abs(y - float(self.target))


def normalize_objectives(
    objectives: str | Objective | Sequence[str | Objective],
    maximize: bool | Sequence[bool],
) -> tuple[Objective, ...]:
    """Convert legacy objective names/directions to explicit objectives."""
    raw = (objectives,) if isinstance(objectives, (str, Objective)) else tuple(objectives)
    flags = (maximize,) * len(raw) if isinstance(maximize, bool) else tuple(maximize)
    if len(raw) != len(flags):
        raise EngineError(ErrorCode.CONFIG_INVALID, "objectives and maximize lengths differ")
    return tuple(
        item if isinstance(item, Objective) else Objective(item, "maximize" if flag else "minimize")
        for item, flag in zip(raw, flags, strict=True)
    )
