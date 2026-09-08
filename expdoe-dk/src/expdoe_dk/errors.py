"""Stable errors raised by the optimization engine."""
from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    CONFIG_INVALID = "CONFIG_INVALID"
    KNOWLEDGE_INVALID = "KNOWLEDGE_INVALID"
    KNOWLEDGE_CONFLICT = "KNOWLEDGE_CONFLICT"
    SPACE_INFEASIBLE = "SPACE_INFEASIBLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    MODEL_FIT_FAILED = "MODEL_FIT_FAILED"
    ACQUISITION_INCOMPATIBLE = "ACQUISITION_INCOMPATIBLE"
    NO_FEASIBLE_CANDIDATE = "NO_FEASIBLE_CANDIDATE"
    CHECKPOINT_INCOMPATIBLE = "CHECKPOINT_INCOMPATIBLE"
    EXTENSION_NOT_ALLOWED = "EXTENSION_NOT_ALLOWED"


class EngineError(RuntimeError):
    """An error with a stable, serializable code and optional context."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict | None = None,
        hints: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.hints = hints
