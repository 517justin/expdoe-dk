"""Deterministic, registry-valid Phase 1 numerical contract cases."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from expdoe_dk import Knowledge, Objective, ObservationBatch, Parameter, Space
from expdoe_dk.knowledge.artifacts import OptimizationArtifact, OptimizationArtifacts
from expdoe_dk.knowledge.guard import CompatibilityResult, KnowledgeValidationResult
from expdoe_dk.knowledge.registry import KnowledgePatternDefinition, PatternRegistry
from expdoe_dk.knowledge.specs import KnowledgeScope, make_pattern_spec


@pytest.fixture
def mixed_case() -> Space:
    return Space(
        [
            Parameter("x", bounds=(1.0, 10.0), unit="K", transform="log"),
            Parameter("z", bounds=(1.0, 10.0)),
            Parameter("count", bounds=(1, 5), kind="integer"),
            Parameter("grade", kind="ordinal", values=("low", "medium", "high")),
            Parameter("material", kind="categorical", values=("A", "B", "C")),
        ],
        objectives=(Objective("y", "maximize"),),
    )


@pytest.fixture
def target_case(mixed_case) -> Space:
    return Space(mixed_case.params, objectives=(Objective("y", "target", target=(0.4, 0.6)),))


@pytest.fixture
def pattern_records() -> list[dict]:
    path = Path(__file__).parent / "fixtures" / "contract_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def pattern_cases(pattern_records, mixed_case):
    multi = Space(
        mixed_case.params,
        objectives=(Objective("y", "maximize"), Objective("cost", "minimize")),
    )
    result = []
    for record in pattern_records:
        space = multi if record["space"] == "multi" else mixed_case
        knowledge = Knowledge().add_spec(
            make_pattern_spec(
                pattern_id=record["id"], pattern=record["pattern"], version="1.0",
                parameters=record["parameters"], confidence=0.8,
                scope=KnowledgeScope(
                    factors=tuple(record["factors"]), objectives=tuple(record["objectives"]),
                ),
            )
        )
        result.append((space, knowledge))
    return tuple(result)


def _artifact_case(space, *, kind: str, category: str):
    def compile_artifact(spec, space, observations=None):
        artifact = OptimizationArtifact(
            kind=kind, payload={}, source_pattern_id=spec.pattern_id,
            source_pattern=spec.pattern, source_version=spec.version,
        )
        return OptimizationArtifacts(**{category: (artifact,)})

    registry = PatternRegistry()
    registry.register(KnowledgePatternDefinition(
        pattern="test_unknown", version="1.0", family="test", schema={"type": "object"},
        compiler=compile_artifact,
        validator=lambda spec, space, observations=None: KnowledgeValidationResult(
            pattern_id=spec.pattern_id, state="valid", summary="valid",
        ),
        compatibility=lambda spec, space: CompatibilityResult(compatible=True),
        renderer=lambda spec, space: "test unknown",
    ))
    knowledge = Knowledge(registry=registry).add_spec(make_pattern_spec(
        pattern_id="KP-unknown", pattern="test_unknown", version="1.0", parameters={},
        scope=KnowledgeScope(), confidence=0.8,
    ))
    return space, knowledge


@pytest.fixture
def unsupported_artifact_case(mixed_case):
    return _artifact_case(
        mixed_case, kind="unknown_consumer_kind", category="mean_components",
    )


@pytest.fixture
def wrong_category_artifact_case(mixed_case):
    return _artifact_case(mixed_case, kind="gp_prior", category="mean_components")


def _batch(n: int, shape: str) -> ObservationBatch:
    rng = np.random.default_rng(17)
    x = rng.uniform(1.0, 10.0, n)
    z = rng.uniform(1.0, 10.0, n)
    X = pd.DataFrame({
        "x": x, "z": z, "count": rng.integers(1, 6, n),
        "grade": rng.choice(["low", "medium", "high"], n),
        "material": rng.choice(["A", "B", "C"], n),
    })
    if shape == "linear":
        y = 0.4 * x + 0.2 * z
    elif shape == "smooth":
        y = np.sin(x / 2) + 0.1 * z
    else:
        y = 0.2 * x * z + np.sin(x)
    return ObservationBatch(X, pd.DataFrame({"y": y}), ids=tuple(f"obs-{i}" for i in range(n)))


@pytest.fixture
def linear_batch() -> ObservationBatch:
    return _batch(12, "linear")


@pytest.fixture
def smooth_batch() -> ObservationBatch:
    return _batch(16, "smooth")


@pytest.fixture
def interaction_batch() -> ObservationBatch:
    return _batch(20, "interaction")
