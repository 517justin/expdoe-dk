# Domain Model and Knowledge Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backward-compatible mixed experiment-domain types and a declarative Knowledge Pattern Registry to `expdoe-dk`.

**Architecture:** New focused modules under `expdoe_dk/domain/` own parameters, objectives, constraints, spaces, observations, and pending points; the existing `expdoe_dk.space` module becomes a compatibility facade. Registry-backed knowledge specs live under `expdoe_dk/knowledge/registry/`, while the existing `Knowledge` fluent API translates legacy helpers into those specs.

**Tech Stack:** Python 3.10+, dataclasses, NumPy, pandas, PyTorch, SciPy, JSON Schema-compatible dictionaries, pytest.

## Global Constraints

- Target release is `expdoe-dk 0.5.0`.
- Existing `Parameter`, `LinearConstraint`, `Space`, `Knowledge`, `Campaign`, and `Result` imports remain valid.
- Version 0.4 checkpoint and result payloads remain readable.
- Serialized constraints and knowledge contain no callable, module path, lambda, or source code.
- Built-in registry definitions load without Python entry-point discovery.
- External registry providers load only from an explicit allow-list.
- Hard constraints and safety patterns are never silently relaxed.
- Physical-to-model transforms must be reversible within declared snapping tolerance.
- No test in this plan requires AI credentials, network access, or equipment.
- Use test-driven development and commit after every task.

---

## File Responsibility Map

```text
expdoe-dk/src/expdoe_dk/
├── errors.py                         # stable typed engine errors
├── domain/
│   ├── __init__.py                   # domain exports
│   ├── parameter.py                  # Parameter kinds, encoding, transforms, snapping
│   ├── objective.py                  # Objective directions and target utility
│   ├── constraints.py                # declarative constraint types and safe AST
│   ├── observation.py                # ObservationBatch and PendingBatch
│   └── space.py                      # Space composition and serialization
├── space.py                          # v0.4 import compatibility facade
└── knowledge/
    ├── __init__.py                   # fluent compatibility API
    ├── artifacts.py                  # OptimizationArtifacts and provenance
    ├── specs.py                      # KnowledgePatternSpec, scope, evidence
    ├── guard.py                      # validation states and guard policy
    ├── registry/
    │   ├── __init__.py               # public registry API
    │   ├── definition.py             # definition protocols and results
    │   ├── registry.py               # registration and exact resolution
    │   └── providers.py              # explicit entry-point provider loading
    └── patterns/
        ├── __init__.py               # built-in registration
        ├── shape.py                  # shape-family definitions
        ├── physics.py                # Arrhenius definition
        ├── interaction.py            # interaction definitions
        ├── categorical.py            # categorical definitions
        ├── feasibility.py            # safe/forbidden regions
        ├── multiobjective.py         # target/priority/tradeoff patterns
        └── prior.py                  # GP prior and random augment compatibility
```

## Task 1: Freeze the v0.4 Compatibility Surface

**Files:**
- Create: `expdoe-dk/tests/fixtures/v04_checkpoint.json`
- Create: `expdoe-dk/tests/test_v04_compatibility.py`
- Modify: `expdoe-dk/tests/conftest.py`

**Interfaces:**
- Consumes: current `expdoe_dk.Parameter`, `LinearConstraint`, `Space`, `Knowledge`, and `Campaign.load_checkpoint()`.
- Produces: characterization tests that every later task must keep green.

- [ ] **Step 1: Add a representative v0.4 checkpoint fixture**

```json
{
  "space": {
    "params": [{"name": "x", "bounds": [0.0, 1.0], "unit": "", "kind": "continuous", "step": null, "log_scale": false}],
    "constraints": [],
    "objectives": ["yield"],
    "maximize": [true]
  },
  "knowledge": {"strict": false, "items": []},
  "seed": 7,
  "history": [{"x": 0.25, "y": 1.5, "kind": "doe"}],
  "trial_kind": ["doe"]
}
```

- [ ] **Step 2: Write compatibility assertions**

```python
from pathlib import Path

import expdoe_dk as ed


def test_v04_public_imports_and_checkpoint_remain_readable():
    assert ed.Parameter.__name__ == "Parameter"
    assert ed.LinearConstraint.__name__ == "LinearConstraint"
    assert ed.Space.__name__ == "Space"
    checkpoint = Path(__file__).parent / "fixtures" / "v04_checkpoint.json"
    campaign = ed.Campaign.load_checkpoint(checkpoint)
    assert campaign.space.objectives == ["yield"]
    assert campaign.history_df()["y"].tolist() == [1.5]


def test_v04_knowledge_payload_round_trips():
    payload = {
        "strict": False,
        "items": [
            {"kind": "monotone", "param": "x", "effect": "increases_objective", "n_pairs_per_dim": 5, "epsilon": "auto", "delta_norm": 0.5}
        ],
    }
    restored = ed.Knowledge.from_dict(payload)
    assert restored.to_dict() == payload
```

- [ ] **Step 3: Run the characterization suite**

Run: `cd expdoe-dk && pytest -q tests/test_v04_compatibility.py tests/test_campaign_smoke.py tests/test_constrained_lhs.py tests/test_validators.py tests/test_frame_translation.py`

Expected: PASS; this records the pre-refactor compatibility boundary.

- [ ] **Step 4: Commit the baseline**

```bash
git add expdoe-dk/tests/fixtures/v04_checkpoint.json expdoe-dk/tests/test_v04_compatibility.py expdoe-dk/tests/conftest.py
git commit -m "test: freeze v0.4 compatibility surface"
```

## Task 2: Add Typed Errors and Explicit Objectives

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/errors.py`
- Create: `expdoe-dk/src/expdoe_dk/domain/__init__.py`
- Create: `expdoe-dk/src/expdoe_dk/domain/objective.py`
- Create: `expdoe-dk/tests/test_objectives.py`
- Modify: `expdoe-dk/src/expdoe_dk/__init__.py`

**Interfaces:**
- Produces: `EngineError(code, message, details, hints)`, `Objective(name, direction, target, unit, priority)`, `Objective.to_utility(values)`, and `normalize_objectives(objectives, maximize)`.
- Consumed by: Tasks 5-10 and the optimization pipeline plan.

- [ ] **Step 1: Write failing objective and error tests**

```python
import numpy as np
import pytest

from expdoe_dk import Objective
from expdoe_dk.errors import EngineError, ErrorCode


def test_target_range_utility_is_zero_inside_and_negative_outside():
    objective = Objective("viscosity", direction="target", target=(10.0, 12.0))
    actual = objective.to_utility(np.array([9.0, 10.0, 11.0, 12.0, 14.0]))
    assert actual.tolist() == [-1.0, 0.0, 0.0, 0.0, -2.0]


def test_objective_rejects_target_for_maximize():
    with pytest.raises(EngineError) as caught:
        Objective("yield", direction="maximize", target=90.0)
    assert caught.value.code is ErrorCode.CONFIG_INVALID
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/test_objectives.py`

Expected: FAIL with import errors for `Objective` and `expdoe_dk.errors`.

- [ ] **Step 3: Implement stable errors and objective utility**

```python
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
    def __init__(self, code: ErrorCode, message: str, *, details: dict | None = None, hints: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.hints = hints
```

```python
@dataclass(frozen=True)
class Objective:
    name: str
    direction: Literal["maximize", "minimize", "target"]
    target: float | tuple[float, float] | None = None
    unit: str = ""
    priority: int = 0

    def to_utility(self, values: np.ndarray) -> np.ndarray:
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
    raw = (objectives,) if isinstance(objectives, (str, Objective)) else tuple(objectives)
    flags = (maximize,) * len(raw) if isinstance(maximize, bool) else tuple(maximize)
    if len(raw) != len(flags):
        raise EngineError(ErrorCode.CONFIG_INVALID, "objectives and maximize lengths differ")
    return tuple(
        item if isinstance(item, Objective) else Objective(item, "maximize" if flag else "minimize")
        for item, flag in zip(raw, flags, strict=True)
    )
```

- [ ] **Step 4: Export the types and run focused tests**

Run: `cd expdoe-dk && pytest -q tests/test_objectives.py`

Expected: PASS.

- [ ] **Step 5: Run existing frame tests**

Run: `cd expdoe-dk && pytest -q tests/test_frame_translation.py tests/test_v04_compatibility.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/errors.py expdoe-dk/src/expdoe_dk/domain expdoe-dk/src/expdoe_dk/__init__.py expdoe-dk/tests/test_objectives.py
git commit -m "feat: add explicit objective domain model"
```

## Task 3: Generalize Parameters and Physical/Model Transforms

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/domain/parameter.py`
- Create: `expdoe-dk/tests/test_parameter_types.py`
- Modify: `expdoe-dk/src/expdoe_dk/domain/__init__.py`

**Interfaces:**
- Produces: `Parameter` with kinds `continuous`, `integer`, `discrete`, `categorical`, `ordinal`; `encode()`, `decode()`, `snap()`, `model_bounds`, and `cardinality`.
- Consumed by: `domain.space.Space`, mixed DoE, and candidate optimization.

- [ ] **Step 1: Write failing round-trip tests for all five kinds**

```python
import numpy as np
import pytest

from expdoe_dk.domain import Parameter


@pytest.mark.parametrize(
    ("parameter", "physical"),
    [
        (Parameter("x", kind="continuous", bounds=(1.0, 100.0), transform="log"), [1.0, 10.0, 100.0]),
        (Parameter("count", kind="integer", bounds=(1, 9), step=2), [1, 5, 9]),
        (Parameter("dose", kind="discrete", values=[0.1, 0.3, 0.8]), [0.1, 0.8]),
        (Parameter("solvent", kind="categorical", values=["water", "ethanol"]), ["water", "ethanol"]),
        (Parameter("grade", kind="ordinal", values=["low", "medium", "high"]), ["low", "high"]),
    ],
)
def test_parameter_encode_decode_round_trip(parameter, physical):
    encoded = parameter.encode(physical)
    assert parameter.decode(encoded) == physical


def test_log_transform_rejects_nonpositive_bounds():
    with pytest.raises(ValueError, match="positive"):
        Parameter("x", kind="continuous", bounds=(0.0, 10.0), transform="log")
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/test_parameter_types.py`

Expected: FAIL because the generalized `Parameter` does not exist.

- [ ] **Step 3: Implement the tagged parameter and encoding rules**

```python
@dataclass(frozen=True)
class Parameter:
    name: str
    bounds: tuple[float, float] | None = None
    unit: str = ""
    kind: Literal["continuous", "integer", "discrete", "categorical", "ordinal"] = "continuous"
    step: float | int | None = None
    values: tuple[object, ...] | list[object] | None = None
    transform: Literal["linear", "log"] = "linear"
    log_scale: bool = False

    def encode(self, values: Sequence[object]) -> list[float]:
        if self.kind in {"categorical", "ordinal"}:
            levels = list(self.values or ())
            return [float(levels.index(value)) for value in values]
        numeric = np.asarray(values, dtype=np.float64)
        if self.transform == "log" or self.log_scale:
            numeric = np.log(numeric)
        lo, hi = self._model_limits()
        return ((numeric - lo) / (hi - lo)).tolist()

    def decode(self, encoded: Sequence[float]) -> list[object]:
        if self.kind in {"categorical", "ordinal"}:
            levels = list(self.values or ())
            return [levels[int(round(value))] for value in encoded]
        lo, hi = self._model_limits()
        numeric = lo + np.asarray(encoded, dtype=np.float64) * (hi - lo)
        if self.transform == "log" or self.log_scale:
            numeric = np.exp(numeric)
        return np.asarray(self.snap(numeric)).tolist()

    def _model_limits(self) -> tuple[float, float]:
        if self.kind in {"categorical", "ordinal"}:
            return 0.0, float(len(self.values or ()) - 1)
        low, high = self.bounds or (0.0, 1.0)
        if self.transform == "log" or self.log_scale:
            return math.log(float(low)), math.log(float(high))
        return float(low), float(high)

    def snap(self, values: np.ndarray) -> np.ndarray:
        numeric = np.asarray(values, dtype=np.float64)
        if self.kind == "continuous":
            return numeric
        levels = np.asarray(self.numeric_levels, dtype=np.float64)
        indices = np.abs(numeric[..., None] - levels).argmin(axis=-1)
        return levels[indices]
```

Implement `numeric_levels` for integer/discrete kinds. Implement `__post_init__()` with the exact validation rules from the design and normalize `values` to a tuple using `object.__setattr__`.

- [ ] **Step 4: Run parameter tests**

Run: `cd expdoe-dk && pytest -q tests/test_parameter_types.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/domain/parameter.py expdoe-dk/src/expdoe_dk/domain/__init__.py expdoe-dk/tests/test_parameter_types.py
git commit -m "feat: support mixed parameter types and transforms"
```

## Task 4: Add Declarative Constraints and a Safe Expression AST

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/domain/constraints.py`
- Create: `expdoe-dk/tests/test_constraints.py`
- Modify: `expdoe-dk/src/expdoe_dk/domain/__init__.py`

**Interfaces:**
- Produces: `Constraint` protocol, `LinearConstraint`, `ExpressionConstraint`, `CategoricalCombinationConstraint`, `OutcomeConstraint`, `constraint_from_dict()`.
- Consumed by: `Space.feasibility_mask()`, DoE generation, knowledge safety patterns, and optimization post-processing.

- [ ] **Step 1: Write failing parser and semantics tests**

```python
import pytest

from expdoe_dk.domain import ExpressionConstraint, CategoricalCombinationConstraint


def test_expression_constraint_evaluates_allowlisted_math():
    constraint = ExpressionConstraint("ratio", "a / b <= 2", hard=True)
    assert constraint.satisfied({"a": 4.0, "b": 2.0})
    assert not constraint.satisfied({"a": 5.0, "b": 2.0})


@pytest.mark.parametrize("expression", ["__import__('os')", "x.__class__", "x[0]"])
def test_expression_constraint_rejects_executable_syntax(expression):
    with pytest.raises(ValueError, match="not allowed"):
        ExpressionConstraint("unsafe", expression)


def test_forbidden_category_combination():
    constraint = CategoricalCombinationConstraint(
        "unstable_pair", forbidden=({"binder": "A", "solvent": "water"},)
    )
    assert not constraint.satisfied({"binder": "A", "solvent": "water"})
    assert constraint.satisfied({"binder": "B", "solvent": "water"})
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/test_constraints.py`

Expected: FAIL with missing constraint classes.

- [ ] **Step 3: Implement the allow-list visitor and constraints**

```python
_ALLOWED_CALLS = {"abs": abs, "min": min, "max": max, "log": math.log, "exp": math.exp}
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Call,
)


def _compile_expression(expression: str, allowed_names: set[str] | None = None) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Expression node {type(node).__name__} is not allowed")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("Only direct allow-listed calls are allowed")
        if isinstance(node, ast.Call) and node.func.id not in _ALLOWED_CALLS:
            raise ValueError(f"Call {node.func.id!r} is not allowed")
        if isinstance(node, ast.Name) and allowed_names is not None and node.id not in allowed_names | set(_ALLOWED_CALLS):
            raise ValueError(f"Name {node.id!r} is not allowed")
    return tree
```

`ExpressionConstraint.satisfied()` evaluates the compiled AST with `{"__builtins__": {}}`, `_ALLOWED_CALLS`, and the row values. Soft constraints require `penalty` and finite positive `weight`; hard constraints reject penalty fields.

- [ ] **Step 4: Run constraint tests**

Run: `cd expdoe-dk && pytest -q tests/test_constraints.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/domain/constraints.py expdoe-dk/src/expdoe_dk/domain/__init__.py expdoe-dk/tests/test_constraints.py
git commit -m "feat: add declarative experiment constraints"
```

## Task 5: Compose the New Space and Preserve `expdoe_dk.space`

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/domain/space.py`
- Replace: `expdoe-dk/src/expdoe_dk/space.py`
- Modify: `expdoe-dk/src/expdoe_dk/doe/constrained.py`
- Modify: `expdoe-dk/src/expdoe_dk/__init__.py`
- Create: `expdoe-dk/tests/test_mixed_space.py`
- Modify: `expdoe-dk/tests/test_constrained_lhs.py`

**Interfaces:**
- Produces: `Space(params, constraints, objectives, maximize, outcome_constraints)`, mixed physical/model DataFrame transforms, schema-versioned `to_dict()`/`from_dict()`.
- Consumed by: all existing DoE and BO code.

- [ ] **Step 1: Write failing mixed-space and legacy-constructor tests**

```python
import pandas as pd

from expdoe_dk import Objective, Parameter, Space, suggest_design


def test_mixed_space_round_trip_and_sobol_design():
    space = Space(
        params=[
            Parameter("temperature", bounds=(300.0, 400.0)),
            Parameter("grade", kind="ordinal", values=["low", "medium", "high"]),
            Parameter("binder", kind="categorical", values=["A", "B"]),
        ],
        objectives=[Objective("strength", "maximize")],
    )
    frame = pd.DataFrame({"temperature": [325.0], "grade": ["medium"], "binder": ["B"]})
    assert space.model_to_physical(space.physical_to_model(frame)).to_dict("records") == frame.to_dict("records")
    design = suggest_design(space, n=6, method="sobol", seed=4)
    assert set(design["binder"]) <= {"A", "B"}


def test_legacy_space_constructor_still_exposes_objective_lists():
    space = Space([Parameter("x", bounds=(0.0, 1.0))], objectives="yield", maximize=True)
    assert space.objectives == ["yield"]
    assert space.maximize == [True]
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/test_mixed_space.py`

Expected: FAIL because mixed transforms are unavailable.

- [ ] **Step 3: Implement `domain.space.Space` and the compatibility facade**

```python
class Space:
    def __init__(self, params, constraints=None, objectives="y", maximize=True, outcome_constraints=None):
        self.params = list(params)
        self.constraints = list(constraints or ())
        self.objective_specs = normalize_objectives(objectives, maximize)
        self.outcome_constraints = list(outcome_constraints or ())

    @property
    def objectives(self) -> list[str]:
        return [objective.name for objective in self.objective_specs]

    @property
    def maximize(self) -> list[bool]:
        return [objective.direction != "minimize" for objective in self.objective_specs]

    def physical_to_model(self, frame: pd.DataFrame) -> torch.Tensor:
        columns = [parameter.encode(frame[parameter.name].tolist()) for parameter in self.params]
        return torch.tensor(np.column_stack(columns), dtype=torch.float64)

    def model_to_physical(self, tensor: torch.Tensor) -> pd.DataFrame:
        data = {parameter.name: parameter.decode(tensor[:, index].tolist()) for index, parameter in enumerate(self.params)}
        return pd.DataFrame(data, columns=self.param_names)
```

Replace `expdoe_dk/space.py` with re-exports from `expdoe_dk.domain`. Keep `physical_to_unit()` and `unit_to_physical()` as numerical-space aliases that delegate to the mixed transform for legacy callers.

```python
def physical_to_unit(self, X_phys: torch.Tensor) -> torch.Tensor:
    frame = pd.DataFrame(torch.as_tensor(X_phys).cpu().numpy(), columns=self.param_names)
    return self.physical_to_model(frame)


def unit_to_physical(self, X_unit: torch.Tensor) -> torch.Tensor:
    frame = self.model_to_physical(torch.as_tensor(X_unit, dtype=torch.float64))
    if any(parameter.kind in {"categorical", "ordinal"} for parameter in self.params):
        raise TypeError("unit_to_physical tensor output is unavailable for string-valued spaces; use model_to_physical")
    return torch.tensor(frame.to_numpy(dtype=float), dtype=torch.float64)
```

- [ ] **Step 4: Update DoE conversion helpers to use Space transforms**

```python
def _unit_to_physical_array(U: np.ndarray, space: Space) -> pd.DataFrame:
    return space.model_to_physical(torch.as_tensor(U, dtype=torch.float64))


def _feasibility_mask(frame: pd.DataFrame, space: Space) -> np.ndarray:
    return space.feasibility_mask(frame).cpu().numpy()
```

Convert method internals back to a numeric model tensor only for distance calculations. Categorical and ordinal columns are encoded before weighted distance is computed.

- [ ] **Step 5: Run mixed and legacy suites**

Run: `cd expdoe-dk && pytest -q tests/test_mixed_space.py tests/test_constrained_lhs.py tests/test_v04_compatibility.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/domain/space.py expdoe-dk/src/expdoe_dk/space.py expdoe-dk/src/expdoe_dk/doe/constrained.py expdoe-dk/src/expdoe_dk/__init__.py expdoe-dk/tests/test_mixed_space.py expdoe-dk/tests/test_constrained_lhs.py
git commit -m "feat: generalize experiment spaces"
```

## Task 6: Add Observation and Pending Batch Contracts

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/domain/observation.py`
- Create: `expdoe-dk/tests/test_observation_batches.py`
- Modify: `expdoe-dk/src/expdoe_dk/domain/__init__.py`
- Modify: `expdoe-dk/src/expdoe_dk/__init__.py`

**Interfaces:**
- Produces: `ObservationBatch(X, Y, Yvar, status, ids)`, `ObservationBatch.successful()`, `PendingBatch(ids, X)`.
- Consumed by: `OptimizationProblem`, `Campaign.tell()`, `Campaign.ask()`.

- [ ] **Step 1: Write failing validation tests**

```python
import pandas as pd
import pytest

from expdoe_dk import ObservationBatch, PendingBatch


def test_successful_filters_failed_rows_without_imputation():
    batch = ObservationBatch(
        X=pd.DataFrame({"x": [0.1, 0.2]}),
        Y=pd.DataFrame({"yield": [10.0, float("nan")]}),
        status=pd.Series(["success", "failed"]),
        ids=("o1", "o2"),
    )
    successful = batch.successful()
    assert successful.ids == ("o1",)
    assert successful.Y["yield"].tolist() == [10.0]


def test_pending_ids_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        PendingBatch(ids=("p1", "p1"), X=pd.DataFrame({"x": [0.1, 0.2]}))
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/test_observation_batches.py`

Expected: FAIL with missing batch types.

- [ ] **Step 3: Implement immutable validated batches**

```python
@dataclass(frozen=True)
class ObservationBatch:
    X: pd.DataFrame
    Y: pd.DataFrame
    Yvar: pd.DataFrame | None = None
    status: pd.Series | None = None
    ids: tuple[str, ...] = ()

    def successful(self) -> "ObservationBatch":
        status = self.status if self.status is not None else pd.Series(["success"] * len(self.X))
        mask = status.reset_index(drop=True).eq("success")
        return ObservationBatch(
            X=self.X.loc[mask].reset_index(drop=True),
            Y=self.Y.loc[mask].reset_index(drop=True),
            Yvar=None if self.Yvar is None else self.Yvar.loc[mask].reset_index(drop=True),
            status=status.loc[mask].reset_index(drop=True),
            ids=() if not self.ids else tuple(identifier for identifier, keep in zip(self.ids, mask, strict=True) if keep),
        )
```

Validate equal row counts, unique IDs, declared statuses, non-negative finite `Yvar`, and no missing objective values in successful rows.

- [ ] **Step 4: Run tests**

Run: `cd expdoe-dk && pytest -q tests/test_observation_batches.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/domain/observation.py expdoe-dk/src/expdoe_dk/domain/__init__.py expdoe-dk/src/expdoe_dk/__init__.py expdoe-dk/tests/test_observation_batches.py
git commit -m "feat: add observation and pending batch contracts"
```

## Task 7: Build the Registry Core and Declarative Specs

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/knowledge/specs.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/artifacts.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/guard.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/registry/__init__.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/registry/definition.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/registry/registry.py`
- Create: `expdoe-dk/tests/knowledge/conftest.py`
- Create: `expdoe-dk/tests/knowledge/test_registry.py`
- Create: `expdoe-dk/tests/knowledge/test_specs.py`

**Interfaces:**
- Produces: `KnowledgePatternSpec`, `KnowledgeScope`, `Evidence`, `KnowledgePatternDefinition`, `PatternRegistry`, `OptimizationArtifacts`, `KnowledgeValidationResult`, and `CompatibilityResult`.
- Consumed by: built-in patterns, fluent `Knowledge`, surrogate factory, and external providers.

- [ ] **Step 1: Write failing spec round-trip and registry tests**

```python
from expdoe_dk.knowledge.registry import PatternRegistry
from expdoe_dk.knowledge.specs import Evidence, KnowledgePatternSpec, KnowledgeScope


def test_pattern_spec_round_trip_preserves_provenance():
    spec = KnowledgePatternSpec(
        pattern="monotone",
        version="1.0",
        parameters={"direction": "increasing"},
        scope=KnowledgeScope(factors=("time",), objectives=("yield",)),
        confidence=0.8,
        evidence=(Evidence(kind="expert_experience", reference="ten prior runs"),),
    )
    assert KnowledgePatternSpec.from_dict(spec.to_dict()) == spec


def test_registry_rejects_duplicate_pattern_version(fake_definition):
    registry = PatternRegistry()
    registry.register(fake_definition)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(fake_definition)
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/knowledge/test_specs.py tests/knowledge/test_registry.py`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement frozen JSON-safe specs and artifact provenance**

```python
JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class KnowledgePatternSpec:
    pattern: str
    version: str
    parameters: dict[str, JSONValue]
    scope: KnowledgeScope
    confidence: float
    evidence: tuple[Evidence, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        json.dumps(self.to_dict(), allow_nan=False)
```

```python
@dataclass(frozen=True)
class OptimizationArtifact:
    kind: str
    payload: dict[str, JSONValue]
    source_pattern: str
    source_version: str


@dataclass(frozen=True)
class OptimizationArtifacts:
    input_transforms: tuple[OptimizationArtifact, ...] = ()
    outcome_transforms: tuple[OptimizationArtifact, ...] = ()
    mean_components: tuple[OptimizationArtifact, ...] = ()
    kernel_components: tuple[OptimizationArtifact, ...] = ()
    priors: tuple[OptimizationArtifact, ...] = ()
    virtual_observations: tuple[OptimizationArtifact, ...] = ()
    parameter_constraints: tuple[OptimizationArtifact, ...] = ()
    outcome_constraints: tuple[OptimizationArtifact, ...] = ()
    acquisition_preferences: tuple[OptimizationArtifact, ...] = ()
    diagnostics: tuple[OptimizationArtifact, ...] = ()
```

```python
@dataclass(frozen=True)
class KnowledgeValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    effective_confidence: float | None = None


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reasons: tuple[str, ...] = ()
```

```python
@dataclass(frozen=True)
class KnowledgePatternDefinition:
    pattern: str
    version: str
    family: str
    schema: dict[str, JSONValue]
    compiler: Callable[[KnowledgePatternSpec, Space, ObservationBatch | None], OptimizationArtifacts]
    validator: Callable[[KnowledgePatternSpec, Space, ObservationBatch | None], KnowledgeValidationResult]
    renderer: Callable[[KnowledgePatternSpec, Space], str]
    compatibility: Callable[[KnowledgePatternSpec, Space], CompatibilityResult]
```

In `tests/knowledge/conftest.py`, define `fake_definition`, `builtin_registry`, `numeric_space`, and `make_spec` using no-op compiler/renderer plus an always-valid `KnowledgeValidationResult`. Later pattern tests reuse these exact fixtures.

- [ ] **Step 4: Implement exact registry resolution**

```python
class PatternRegistry:
    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], KnowledgePatternDefinition] = {}

    def register(self, definition: KnowledgePatternDefinition) -> None:
        key = (definition.pattern, definition.version)
        if key in self._definitions:
            raise ValueError(f"Pattern {key} already registered")
        self._definitions[key] = definition

    def resolve(self, pattern: str, version: str) -> KnowledgePatternDefinition:
        try:
            return self._definitions[(pattern, version)]
        except KeyError as error:
            raise EngineError(ErrorCode.KNOWLEDGE_INVALID, f"Unknown pattern {pattern}@{version}") from error

    def definitions(self) -> tuple[KnowledgePatternDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    def validate_many(self, specs, space, observations=None) -> tuple[KnowledgeValidationResult, ...]:
        return tuple(self.resolve(spec.pattern, spec.version).validator(spec, space, observations) for spec in specs)

    def compile_many(self, specs, space, observations=None) -> OptimizationArtifacts:
        return merge_artifacts(
            self.resolve(spec.pattern, spec.version).compiler(spec, space, observations)
            for spec in specs
        )

    def render(self, spec, space) -> str:
        return self.resolve(spec.pattern, spec.version).renderer(spec, space)
```

- [ ] **Step 5: Run registry tests**

Run: `cd expdoe-dk && pytest -q tests/knowledge/test_specs.py tests/knowledge/test_registry.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/knowledge/specs.py expdoe-dk/src/expdoe_dk/knowledge/artifacts.py expdoe-dk/src/expdoe_dk/knowledge/guard.py expdoe-dk/src/expdoe_dk/knowledge/registry expdoe-dk/tests/knowledge
git commit -m "feat: add knowledge pattern registry core"
```

## Task 8: Migrate Existing Knowledge Helpers to the Registry

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/knowledge/patterns/prior.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/patterns/physics.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/patterns/shape.py`
- Modify: `expdoe-dk/src/expdoe_dk/knowledge/__init__.py`
- Modify: `expdoe-dk/src/expdoe_dk/bo/gp.py`
- Create: `expdoe-dk/tests/knowledge/test_legacy_helpers.py`
- Modify: `expdoe-dk/tests/test_eps_auto_rescue.py`
- Modify: `expdoe-dk/tests/test_validators.py`

**Interfaces:**
- Produces: registry definitions for `arrhenius@1.0`, `quadratic_peak@1.0`, `monotone@1.0`, `gp_prior@1.0`, and `random_augment@1.0`; `Knowledge.specs`.
- Consumes: registry core and existing GP components.

- [ ] **Step 1: Write failing legacy-to-spec tests**

```python
from expdoe_dk import Knowledge


def test_legacy_helpers_emit_versioned_specs_and_old_payload():
    knowledge = Knowledge().with_arrhenius("temperature").with_monotone(
        "time", effect="increases_objective"
    )
    assert [(spec.pattern, spec.version) for spec in knowledge.specs] == [
        ("arrhenius", "1.0"),
        ("monotone", "1.0"),
    ]
    restored = Knowledge.from_dict(knowledge.to_dict())
    assert restored.to_dict() == knowledge.to_dict()
```

- [ ] **Step 2: Run the test to verify failure**

Run: `cd expdoe-dk && pytest -q tests/knowledge/test_legacy_helpers.py`

Expected: FAIL because `Knowledge.specs` is unavailable.

- [ ] **Step 3: Make the fluent API construct specs**

```python
def with_monotone(self, param: str, effect: PhysicalEffect, **options) -> "Knowledge":
    direction = "increasing" if effect == "increases_objective" else "decreasing"
    self._specs.append(
        KnowledgePatternSpec(
            pattern="monotone",
            version="1.0",
            parameters={"direction": direction, **options},
            scope=KnowledgeScope(factors=(param,)),
            confidence=1.0,
        )
    )
    return self
```

Keep `items`, `items_of()`, `drop()`, `resolve_epsilon()`, and legacy `to_dict()` behavior through adapters that render registered specs into the old item view. Add `to_specs_dict()` for the new schema instead of changing old payloads in place.

Implement `Knowledge.compile(space, observations=None)` as a thin call to the configured `PatternRegistry.compile_many(self.specs, space, observations)`. The constructor accepts an optional registry for tests and provider-enabled applications; otherwise it uses a fresh registry populated only with built-in definitions.

- [ ] **Step 4: Route GP construction through compiled artifacts**

```python
artifacts = knowledge.compile(space=space, observations=None)
mean_function = build_mean_from_artifacts(space, artifacts.mean_components)
likelihood, covar = build_prior_modules(artifacts.priors, train_X_unit.shape[1])
augmenter = build_virtual_observation_augmenter(space, artifacts.virtual_observations)
```

The helper functions reuse `ArrheniusMeanFrozen`, `QuadraticMeanFrozen`, `MonotonicAugmenter`, and `GP_PRIOR_PRESETS`; this task changes routing, not numerical formulas.

- [ ] **Step 5: Run compatibility and knowledge tests**

Run: `cd expdoe-dk && pytest -q tests/knowledge/test_legacy_helpers.py tests/test_eps_auto_rescue.py tests/test_epsilon_validator.py tests/test_validators.py tests/test_campaign_smoke.py tests/test_v04_compatibility.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/knowledge expdoe-dk/src/expdoe_dk/bo/gp.py expdoe-dk/tests/knowledge/test_legacy_helpers.py expdoe-dk/tests/test_eps_auto_rescue.py expdoe-dk/tests/test_validators.py
git commit -m "refactor: route legacy knowledge through registry"
```

## Task 9: Add the Complete Built-in Pattern Taxonomy

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/knowledge/patterns/__init__.py`
- Expand: `expdoe-dk/src/expdoe_dk/knowledge/patterns/shape.py`
- Expand: `expdoe-dk/src/expdoe_dk/knowledge/patterns/physics.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/patterns/interaction.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/patterns/categorical.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/patterns/feasibility.py`
- Create: `expdoe-dk/src/expdoe_dk/knowledge/patterns/multiobjective.py`
- Create: `expdoe-dk/tests/knowledge/test_builtin_patterns.py`
- Create: `expdoe-dk/tests/knowledge/test_pattern_composition.py`

**Interfaces:**
- Produces: all built-ins listed in the approved taxonomy and chainable helpers `with_saturation()`, `with_threshold()`, `with_interaction()`, `with_safe_region()`, and `with_tradeoff()`.
- Consumed by: optimization artifact compilation and `ai-doe knowledge` commands.

- [ ] **Step 1: Write a registry coverage test**

```python
EXPECTED = {
    "monotone", "saturation", "threshold", "quadratic_peak",
    "quadratic_valley", "optimum_range", "power_law", "exponential",
    "periodic", "arrhenius", "synergy", "antagonism",
    "conditional_effect", "ratio_optimum", "ordinal_categories",
    "category_similarity", "safe_region", "forbidden_region",
    "target_range", "objective_priority", "tradeoff", "gp_prior",
    "random_augment",
}


def test_builtin_registry_contains_complete_taxonomy(builtin_registry):
    assert {definition.pattern for definition in builtin_registry.definitions()} == EXPECTED
```

- [ ] **Step 2: Write representative validation and composition tests**

```python
def test_arrhenius_requires_temperature_compatible_scope(numeric_space):
    spec = make_spec("arrhenius", factors=("pressure",), parameters={})
    result = builtin_registry.validate(spec, numeric_space, observations=None)
    assert result.state == "invalid"


def test_safe_and_forbidden_artifacts_intersect(builtin_registry, numeric_space):
    safe = make_spec("safe_region", parameters={"expression": "x >= 0.2"})
    forbidden = make_spec("forbidden_region", parameters={"expression": "x > 0.8"})
    artifacts = builtin_registry.compile_many((safe, forbidden), numeric_space, None)
    assert len(artifacts.parameter_constraints) == 2
```

- [ ] **Step 3: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/knowledge/test_builtin_patterns.py tests/knowledge/test_pattern_composition.py`

Expected: FAIL because the full taxonomy is not registered.

- [ ] **Step 4: Implement definitions with a shared constructor**

```python
def definition(pattern, family, schema, compiler, validator, renderer, compatibility):
    return KnowledgePatternDefinition(
        pattern=pattern,
        version="1.0",
        family=family,
        schema=schema,
        compiler=compiler,
        validator=validator,
        renderer=renderer,
        compatibility=compatibility,
    )
```

Each module defines explicit JSON-compatible schemas and returns provenance-bearing artifacts. Shape compilers use `mean_component` or `virtual_observation`; interaction compilers use kernel/mean components; categorical compilers emit similarity/ordering artifacts; safety compilers emit hard constraints; multi-objective compilers emit bounded acquisition preferences.

- [ ] **Step 5: Add typed convenience helpers**

```python
def with_saturation(self, param: str, *, direction: str, half_response: float, confidence: float = 1.0) -> "Knowledge":
    return self.add_spec(
        KnowledgePatternSpec(
            pattern="saturation", version="1.0",
            parameters={"direction": direction, "half_response": half_response},
            scope=KnowledgeScope(factors=(param,)), confidence=confidence,
        )
    )


def with_interaction(self, first: str, second: str, *, kind: str, confidence: float = 1.0) -> "Knowledge":
    if kind not in {"synergy", "antagonism"}:
        raise ValueError("kind must be synergy or antagonism")
    return self.add_spec(
        KnowledgePatternSpec(
            pattern=kind, version="1.0", parameters={},
            scope=KnowledgeScope(factors=(first, second)), confidence=confidence,
        )
    )
```

- [ ] **Step 6: Run pattern and full knowledge tests**

Run: `cd expdoe-dk && pytest -q tests/knowledge tests/test_validators.py tests/test_eps_auto_rescue.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/knowledge/patterns expdoe-dk/src/expdoe_dk/knowledge/__init__.py expdoe-dk/tests/knowledge
git commit -m "feat: add built-in domain knowledge patterns"
```

## Task 10: Add Explicit Provider Loading, Serialization Versions, and Exports

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/knowledge/registry/providers.py`
- Create: `expdoe-dk/tests/knowledge/test_provider_loading.py`
- Modify: `expdoe-dk/src/expdoe_dk/knowledge/registry/__init__.py`
- Modify: `expdoe-dk/src/expdoe_dk/__init__.py`
- Modify: `expdoe-dk/pyproject.toml`
- Modify: `README.md`
- Modify: `README_zh.md`

**Interfaces:**
- Produces: `load_pattern_providers(allowed: Collection[str], registry: PatternRegistry) -> ProviderLoadReport` and public version `0.5.0` exports.
- Consumed by: `ai-doe validate` and experiment configuration.

- [ ] **Step 1: Write failing allow-list tests with mocked entry points**

```python
def test_provider_loader_imports_only_allowed_distributions(monkeypatch, registry):
    loaded = []
    monkeypatch.setattr(providers, "entry_points", lambda group: fake_entry_points(loaded))
    report = load_pattern_providers({"approved-patterns"}, registry)
    assert loaded == ["approved-patterns"]
    assert report.providers[0].name == "approved-patterns"


def test_provider_loader_rejects_unlisted_name(registry):
    with pytest.raises(EngineError) as caught:
        load_pattern_providers({"not-installed"}, registry)
    assert caught.value.code is ErrorCode.EXTENSION_NOT_ALLOWED
```

Define `fake_entry_points(loaded)` in this test module with one approved fake entry point and one unapproved fake entry point. Its `.load()` appends only the selected entry-point name to `loaded` before returning a provider factory, so the assertion proves that unlisted provider code never executes.

- [ ] **Step 2: Run the tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/knowledge/test_provider_loading.py`

Expected: FAIL because provider loading is unavailable.

- [ ] **Step 3: Implement explicit entry-point loading**

```python
ENTRY_POINT_GROUP = "expdoe_dk.knowledge_patterns"


def load_pattern_providers(allowed: Collection[str], registry: PatternRegistry) -> ProviderLoadReport:
    available = {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)}
    missing = sorted(set(allowed) - set(available))
    if missing:
        raise EngineError(ErrorCode.EXTENSION_NOT_ALLOWED, "Pattern providers are not installed", details={"providers": missing})
    records = []
    for name in sorted(allowed):
        provider = available[name].load()
        definitions = tuple(provider())
        for item in definitions:
            registry.register(item)
        records.append(provider_record(name, available[name], definitions))
    return ProviderLoadReport(tuple(records))
```

- [ ] **Step 4: Export APIs, bump version, and document migration**

Set `project.version = "0.5.0"`, `expdoe_dk.__version__ = "0.5.0"`, export new public types, and update both READMEs with the mixed-space and registry APIs. Replace the old roadmap entries with delivered `0.5.0` capabilities and keep future observer/device work outside this repository.

- [ ] **Step 5: Run the complete fast suite**

Run: `cd expdoe-dk && pytest -q`

Expected: all fast tests PASS; slow-marked tests remain skipped unless `--run-slow` is supplied.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk expdoe-dk/tests/knowledge/test_provider_loading.py expdoe-dk/pyproject.toml README.md README_zh.md
git commit -m "feat: publish extensible knowledge registry API"
```

## Plan Completion Gate

Run:

```bash
cd expdoe-dk
pytest -q
python -m compileall src
```

Expected: fast tests pass, compilation exits 0, v0.4 compatibility tests remain green, and no provider loads without an explicit allow-list. Do not begin the optimization pipeline plan until this gate passes.
