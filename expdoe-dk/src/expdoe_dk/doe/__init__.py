"""DoE generation: capability-aware initial design and legacy helpers."""
from .constrained import generate, InfeasibleDesignError, MethodLiteral
from .design import (
    METHOD_KINDS,
    DesignBatch,
    DesignDiagnostics,
    candidate_keys,
    compatible_design_methods,
    resolve_design_method,
    suggest_design,
)
from .lhs import latin_hypercube_sample, optimize_lhs_maximin

__all__ = [
    "generate",
    "suggest_design",
    "InfeasibleDesignError",
    "MethodLiteral",
    "METHOD_KINDS",
    "DesignBatch",
    "DesignDiagnostics",
    "candidate_keys",
    "compatible_design_methods",
    "resolve_design_method",
    "latin_hypercube_sample",
    "optimize_lhs_maximin",
]
