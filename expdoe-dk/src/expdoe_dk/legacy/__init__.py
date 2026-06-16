"""
Backward-compatibility shims for `ax_doe_bo`.

Old code:
    from ax_doe_bo import run_ax_bo
    cum_best = run_ax_bo(design, bench_fn, n_bo=20, seed=42)

Still works (via this module) but emits DeprecationWarning pointing to
the new `expdoe_dk.Campaign` API.
"""
from .ax_doe_bo import run_ax_bo, run_pure_botorch
from .doe_utils import latin_hypercube_sample, optimize_lhs_maximin

__all__ = [
    "run_ax_bo",
    "run_pure_botorch",
    "latin_hypercube_sample",
    "optimize_lhs_maximin",
]
