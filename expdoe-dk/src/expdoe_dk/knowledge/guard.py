"""Knowledge validation and compatibility result contracts."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


ValidationState = Literal["valid", "warning", "invalid", "insufficient_data"]
_VALIDATION_STATES = frozenset(
    {"valid", "warning", "invalid", "insufficient_data"}
)


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be an ordered sequence of strings")
    result = tuple(value)
    if any(type(item) is not str for item in result):
        raise TypeError(f"{context} must contain only strings")
    return result  # type: ignore[return-value]


def _confidence(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{context} must be a finite non-boolean real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{context} must be finite and in [0, 1]")
    return normalized


@dataclass(frozen=True)
class KnowledgeValidationResult:
    pattern_id: str
    state: ValidationState
    summary: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    effective_confidence: float | None = None

    def __post_init__(self) -> None:
        if type(self.pattern_id) is not str or not self.pattern_id:
            raise ValueError("KnowledgeValidationResult.pattern_id must be a non-empty string")
        if type(self.state) is not str or self.state not in _VALIDATION_STATES:
            raise ValueError(
                "KnowledgeValidationResult.state must be one of valid, warning, "
                "invalid, insufficient_data"
            )
        if type(self.summary) is not str:
            raise TypeError("KnowledgeValidationResult.summary must be a string")
        object.__setattr__(
            self,
            "errors",
            _string_tuple(self.errors, "KnowledgeValidationResult.errors"),
        )
        object.__setattr__(
            self,
            "warnings",
            _string_tuple(self.warnings, "KnowledgeValidationResult.warnings"),
        )
        if self.effective_confidence is not None:
            object.__setattr__(
                self,
                "effective_confidence",
                _confidence(
                    self.effective_confidence,
                    "KnowledgeValidationResult.effective_confidence",
                ),
            )

    @property
    def valid(self) -> bool:
        return self.state in {"valid", "warning", "insufficient_data"}


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.compatible) is not bool:
            raise TypeError("CompatibilityResult.compatible must be a bool")
        object.__setattr__(
            self,
            "reasons",
            _string_tuple(self.reasons, "CompatibilityResult.reasons"),
        )


__all__ = [
    "CompatibilityResult",
    "KnowledgeValidationResult",
    "ValidationState",
]
