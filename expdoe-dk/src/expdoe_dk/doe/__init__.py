"""DoE generation: LHS variants, Sobol, Halton, D-Optimal, with constraints."""
from .constrained import generate, InfeasibleDesignError, MethodLiteral
from .lhs import latin_hypercube_sample, optimize_lhs_maximin

__all__ = [
    "generate",
    "InfeasibleDesignError",
    "MethodLiteral",
    "latin_hypercube_sample",
    "optimize_lhs_maximin",
]
