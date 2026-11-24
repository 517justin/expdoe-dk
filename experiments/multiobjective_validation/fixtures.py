"""Predeclared noiseless catalog oracle; truth is never used to fit geometry."""
import numpy as np
import pandas as pd

from expdoe_dk import (Knowledge, KnowledgeScope, MultiObjectiveConfig, Objective,
                       ObservationBatch, Parameter, Space, make_pattern_spec)
from expdoe_dk.domain import LinearConstraint
from expdoe_dk.bo.knowledge_runtime import compile_runtime
from expdoe_dk.bo.objective_geometry import resolve_geometry

SEEDS = (2, 5, 11, 23, 47)
METHODS = ("random", "plain_gp", "knowledge_gp")
INITIAL, ROUNDS, Q = 6, 8, 2


def catalog():
    return pd.DataFrame({"x": np.arange(101) / 100})


def space():
    return Space([Parameter("x", bounds=(0., 1.))],
        objectives=[Objective("gain", "maximize"), Objective("retention", "maximize")],
        constraints=[LinearConstraint("limit", {"x": 1.}, "<=", .8)])


def measure(X):
    x = X.x.to_numpy()
    return pd.DataFrame({"gain": x, "retention": 1 - x})


def initial(seed):
    feasible = catalog().loc[lambda X: X.x <= .8]
    return feasible.iloc[np.random.default_rng(seed).choice(len(feasible), INITIAL, replace=False)].reset_index(drop=True)


def initial_batch(seed):
    X = initial(seed)
    return ObservationBatch(X, measure(X), ids=tuple(f"initial-{i}" for i in range(INITIAL)))


def config():
    return MultiObjectiveConfig(noise_mode="noiseless", reference_point={"gain": 0., "retention": 0.})


def knowledge(method):
    result = Knowledge()
    if method == "knowledge_gp":
        result.add(make_pattern_spec(pattern_id="gain-increases", pattern="monotone", version="1.0",
            scope=KnowledgeScope(factors=("x",), objectives=("gain",)),
            parameters={"direction": "increasing", "epsilon": "auto", "delta_norm": .5,
                        "n_pairs_per_dim": 3}, confidence=.8))
    return result


def geometry(seed):
    declaration = space()
    return resolve_geometry(declaration, initial_batch(seed),
                            compile_runtime(declaration, Knowledge(), None), config())
