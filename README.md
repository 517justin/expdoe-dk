[中文版 README](./README_zh.md)

# expdoe-dk

**Design of Experiments (DoE) + Bayesian Optimization (BO), with optional domain-knowledge injection for chemistry, materials, and lab workflows.**

`expdoe-dk` helps experimentalists plan a small initial design, respect real lab constraints, and continue with Gaussian-process Bayesian optimization. The package is built for physical units, discrete operating steps, linear constraints, and domain knowledge such as monotonic trends, Arrhenius-like temperature effects, and peak-shaped factors.

## What It Provides

| Capability | Summary |
|------------|---------|
| Constrained DoE | Generate initial points with LHS, Sobol, Halton, D-optimal, or random designs while respecting bounds, discrete steps, and linear constraints. |
| Bayesian optimization | Continue after DoE with an ask/tell campaign loop backed by GP models. |
| Domain knowledge | Encode expected trends with `Knowledge().with_arrhenius(...)`, `with_monotone(...)`, `with_quadratic_peak(...)`, and related helpers. |
| Lab-facing output | Work in physical units and export shareable HTML campaign reports. |
| Safe defaults | If no knowledge is given, the campaign uses a plain GP rather than injecting hidden assumptions. |

## Quick Example

```python
import expdoe_dk as ed

space = ed.Space(
    params=[
        ed.Parameter("T", bounds=(60, 120), unit="degC", kind="discrete", step=1),
        ed.Parameter("time", bounds=(10, 180), unit="min", kind="discrete", step=5),
        ed.Parameter("pH", bounds=(4, 10), kind="discrete", step=1),
        ed.Parameter("conc_A", bounds=(1, 10), unit="mL", kind="discrete", step=1),
    ],
    objectives="yield_pct",
    maximize=True,
)

knowledge = (
    ed.Knowledge()
    .with_arrhenius("T")
    .with_monotone("time", effect="increases_objective")
    .with_quadratic_peak("pH", center=7)
)

campaign = ed.Campaign(space, knowledge, seed=42)

doe = campaign.suggest_doe(n=12)
y_doe = run_lab_experiments(doe)
campaign.tell(doe, y_doe)

for _ in range(20):
    x_next = campaign.ask(q=1)
    y_next = run_lab_experiments(x_next)
    campaign.tell(x_next, y_next)

result = campaign.finalize()
result.to_html("campaign_report.html")
```

## Install

For normal development from the repository root:

```bash
pip install -r requirements.txt
pip install -e ./expdoe-dk
```

Or install directly from the package directory:

```bash
cd expdoe-dk
pip install -e .
```

The runtime requirements are listed in [`requirements.txt`](./requirements.txt) and mirrored in [`expdoe-dk/pyproject.toml`](./expdoe-dk/pyproject.toml): Python 3.10+, PyTorch, BoTorch, GPyTorch, Ax, NumPy, SciPy, pandas, matplotlib, and pyDOE3.

For development and tests:

```bash
cd expdoe-dk
pip install -e ".[dev]"
pytest -q
```

Slow integration tests can be enabled with:

```bash
pytest -q --run-slow
```

## Repository Layout

```text
expdoe-dk/                          # Publishable Python package
  src/expdoe_dk/
    space.py                        # Parameter, LinearConstraint, Space
    doe/                            # DoE methods
    knowledge/                      # Knowledge specs and frame translation
    bo/                             # Campaign loop and HTML report
    legacy/                         # ax_doe_bo compatibility shims
  tests/                            # Unit and integration tests
  pyproject.toml                    # Package metadata and dependencies

examples/                           # End-to-end usage demos
experiments/                        # Reproducible studies and result notes
docs/superpowers/specs/              # Design specs for planned experiments

ax_doe_bo.py / doe_utils.py / benchmarks.py
                                    # Historical research framework kept for reproducibility
```

## Examples And Experiments

| Path | Purpose |
|------|---------|
| [`examples/01_reaction_optimization.py`](./examples/01_reaction_optimization.py) | End-to-end DoE to BO reaction optimization example. |
| [`examples/02_html_report.py`](./examples/02_html_report.py) | Demonstrates `Result.to_html(...)`. |
| [`experiments/simulation_data1/`](./experiments/simulation_data1/) | Clean synthetic oracle studies for DoE method and knowledge comparisons. |
| [`experiments/simulation_data2/`](./experiments/simulation_data2/) | Discrete, constrained, lab-like simulation study. |
| [`docs/superpowers/specs/`](./docs/superpowers/specs/) | Planning documents for future simulation datasets. |

The README intentionally keeps experiment results short. Detailed methods, tables, and conclusions live under [`experiments/`](./experiments/), especially [`experiments/README.md`](./experiments/README.md) and each simulation dataset README.

## Practical Guidance

- Start with `Campaign(space)` and no knowledge when you do not have a specific physical assumption.
- Use domain knowledge when the assumption is strong and stated in physical terms, such as "temperature increases yield" or "pH peaks near 7".
- Treat `with_random_augment(...)` as exploratory regularization, not a default.
- Keep the number of active optimization factors modest when lab budget is tight.
- Export `result.to_html(...)` when you need a compact report for a notebook or collaborator.

## Roadmap

| Version | Adds | Status |
|---------|------|--------|
| v0.1 | Constrained DoE, knowledge composition, campaign loop, first example | Released |
| v0.2 | Empirical validators for monotonicity and frozen-mean shape checks | Released |
| v0.3 | Epsilon auto-rescue for monotone knowledge plus GP priors | Released |
| v0.4 | Self-contained HTML report | Released |
| v0.5 | Claude Code skill packaging | Planned |
| v0.6 | MCP server interface | Planned |
| v0.7 | Multi-objective BO | Planned |
| v0.8 | Multi-fidelity BO | Planned |
| v1.0 | Stable API and legacy shim removal | Planned |

## License

Apache License, Version 2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

The historical files `ax_doe_bo.py`, `doe_utils.py`, and `benchmarks.py` were originally MIT-licensed. They are retained for reproducibility, and the original terms remain available in git history.
