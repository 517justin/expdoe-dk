"""Core domain types for experiment optimization."""

from .objective import Objective, normalize_objectives
from .parameter import Parameter

__all__ = ["Objective", "Parameter", "normalize_objectives"]
