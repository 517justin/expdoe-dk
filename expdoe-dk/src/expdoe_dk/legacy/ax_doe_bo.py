"""
Deprecation shim for ax_doe_bo.ax_doe_bo.

These thin wrappers preserve the old call signatures for users who built
on the previous package. New code should use `expdoe_dk.Campaign`.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import torch


def run_ax_bo(
    design: np.ndarray,
    bench_fn: Callable,
    n_bo: int,
    seed: int = 42,
    surrogate: str = "botorch",
) -> np.ndarray:
    warnings.warn(
        "ax_doe_bo.run_ax_bo is deprecated; use expdoe_dk.Campaign:\n"
        "    space = Space([Parameter(...)])\n"
        "    campaign = Campaign(space, knowledge=None, seed=seed)\n"
        "    result = campaign.run(oracle, n_doe=..., n_iter=n_bo)\n"
        "See AGENT_KNOWLEDGE.md for migration details.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Best-effort: route through a simple Campaign on a unit hypercube.
    from ..space import Parameter, Space
    from ..bo import Campaign

    d = design.shape[1]
    space = Space(
        params=[Parameter(f"x{i}", bounds=(0.0, 1.0)) for i in range(d)],
        objectives="y",
        maximize=False,  # bench_fn historically returns a value to minimize
    )
    campaign = Campaign(space=space, knowledge=None, seed=seed)

    # Inject design as initial DoE.
    import pandas as pd

    df0 = pd.DataFrame(design, columns=space.param_names)
    y0 = np.asarray([float(bench_fn(torch.tensor(row).unsqueeze(0))) for row in design])
    campaign.tell(df0, y0)

    cum_best = [float(np.min(y0))]
    for it in range(n_bo):
        next_pts = campaign.ask(q=1, iteration=it)
        y_new = float(bench_fn(torch.tensor(next_pts.values)))
        campaign.tell(next_pts, np.array([y_new]))
        cum_best.append(min(cum_best[-1], y_new))
    return np.asarray(cum_best, dtype=np.float64)


def run_pure_botorch(
    design: np.ndarray,
    bench_fn: Callable,
    n_bo: int,
    seed: int = 42,
) -> np.ndarray:
    warnings.warn(
        "ax_doe_bo.run_pure_botorch is deprecated; expdoe_dk.Campaign "
        "internally uses pure BoTorch and produces equivalent results.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_ax_bo(design, bench_fn, n_bo, seed=seed)
