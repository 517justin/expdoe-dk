[中文版 README](./README_zh.md)

# expdoe-dk

Design of Experiments (DoE) and Bayesian optimization for physical-unit,
mixed-variable laboratory workflows, with explicit constraints and versioned
domain knowledge.

## v0.5 capabilities

- Continuous, integer, discrete, categorical, and ordinal parameters with
  physical/model transforms.
- Explicit maximize, minimize, and target objectives.
- Declarative linear, expression, categorical-combination, and outcome
  constraints.
- Capability-aware LHS, Sobol, Halton, random, and D-optimal initial designs
  with deterministic diagnostics.
- Immutable observation/pending batches and schema-versioned space payloads.
- A versioned knowledge-pattern registry with built-ins and an explicit,
  distribution-name allow-list for external providers.
- The v0.4 `Campaign`, `Knowledge`, checkpoint, report, and top-level import
  surface remain compatible.

## Install

```bash
pip install -e ./expdoe-dk
```

Python 3.10–3.12 is supported. Runtime dependencies are declared in
[`expdoe-dk/pyproject.toml`](./expdoe-dk/pyproject.toml), including the direct
`jsonschema>=4.18` dependency used to validate knowledge definitions.

## Mixed-space design with objectives and constraints

This example is runnable from an environment where the package is installed:

```python
import expdoe_dk as ed
from expdoe_dk.domain import LinearConstraint

space = ed.Space(
    params=[
        ed.Parameter("temperature", bounds=(300.0, 400.0), unit="K"),
        ed.Parameter("cycles", kind="integer", bounds=(1, 9), step=2),
        ed.Parameter("dose", kind="discrete", values=[0.1, 0.3, 0.8]),
        ed.Parameter("solvent", kind="categorical", values=["water", "ethanol"]),
        ed.Parameter("grade", kind="ordinal", values=["low", "medium", "high"]),
    ],
    constraints=[
        LinearConstraint(
            "temperature_limit",
            coefficients={"temperature": 1.0},
            operator="<=",
            bound=390.0,
        ),
        ed.CategoricalCombinationConstraint(
            "avoid_low_ethanol",
            forbidden=[{"solvent": "ethanol", "grade": "low"}],
        ),
    ],
    objectives=[
        ed.Objective("yield", "maximize", unit="%", priority=0),
        ed.Objective("waste", "minimize", unit="g", priority=1),
    ],
)

batch = ed.suggest_design(
    space, n=8, method="auto", seed=7, return_diagnostics=True
)
print(batch.frame)
print(batch.diagnostics.effective_method)
```

`expdoe_dk.LinearConstraint` remains the v0.4 two-sided adapter. New code that
wants the declarative one-sided form should import
`expdoe_dk.domain.LinearConstraint`, as above.

## Knowledge registry and explicit provider loading

Built-ins and external providers use the same exact `(pattern, version)`
registry. Provider discovery never runs implicitly: imports, `Knowledge`,
validation, and compilation do not execute provider code.

```python
from expdoe_dk import Knowledge, PatternRegistry, load_pattern_providers
from expdoe_dk.knowledge.patterns import builtin_pattern_definitions

registry = PatternRegistry()
for definition in builtin_pattern_definitions():
    registry.register(definition)

# Keep empty when no external distribution is approved. Replace with an
# installed distribution name such as {"my-lab-patterns"} after review.
approved_distributions: set[str] = set()
report = load_pattern_providers(approved_distributions, registry)
print(report.to_dict())

knowledge = Knowledge(registry).with_monotone(
    "temperature", effect="increases_objective"
)
print([(spec.pattern, spec.version) for spec in knowledge.specs])
```

Allow-list matching uses deterministic PEP 503-style distribution names, so
case and `.`, `_`, or `-` punctuation variants are equivalent. Entry-point
names are provenance only and never grant authorization. Missing requested
distributions and malformed providers raise typed `EngineError` values before
the registry is mutated.

## Migrating from v0.4

Legacy construction remains valid:

```python
import expdoe_dk as ed

legacy_space = ed.Space(
    [ed.Parameter("x", bounds=(0.0, 1.0))],
    objectives="yield",
    maximize=True,
)
legacy_constraint = ed.LinearConstraint(coeffs={"x": 1.0}, upper=0.8)
```

For v0.5, prefer explicit objectives, declarative constraints, and versioned
knowledge specs:

```python
import expdoe_dk as ed
from expdoe_dk.domain import LinearConstraint

space = ed.Space(
    [ed.Parameter("x", bounds=(0.0, 1.0))],
    constraints=[LinearConstraint("x_limit", {"x": 1.0}, "<=", 0.8)],
    objectives=[ed.Objective("yield", "maximize")],
)
knowledge = ed.Knowledge().with_saturation(
    "x", direction="increasing", half_response=0.4
)
payload = space.to_dict()
restored = ed.Space.from_dict(payload)
assert restored.to_dict() == payload
```

Observer and laboratory-device integration is future external work and is not
implemented in this repository.

## Development

```bash
cd expdoe-dk
pip install -e ".[dev]"
pytest -q
```

Slow integration tests run only when explicitly selected with `--run-slow`.

## License

Apache License, Version 2.0. See [`LICENSE`](./LICENSE) and
[`NOTICE`](./NOTICE).
