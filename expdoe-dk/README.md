# expdoe-dk

**Experimental Design of Experiments + Bayesian Optimization, with Domain Knowledge injection — for chemistry, materials, and lab experimentalists.**

If you're a chemist or materials researcher running a few dozen experiments to find the best conditions, this library gives you:

1. **Constrained, discrete-step DoE** — initial designs that respect "A must be ≥ B + 1 mL" and "this dial only steps in 0.5 mL increments". No more rounding off post-hoc.
2. **Bayesian optimization** that drives the experiment forward after the initial DoE, using a Gaussian Process surrogate.
3. **Domain-knowledge injection** — tell the optimizer that "temperature increases yield (Arrhenius)" or "pH peaks at 7" and it will use that hint, not fight you.
4. **Safe defaults** baked in from two months of empirical research on a sister project (see [AGENT_KNOWLEDGE.md](../AGENT_KNOWLEDGE.md) for the lessons).

```python
import expdoe_dk as ed

space = ed.Space(
    params=[
        ed.Parameter("T",      bounds=(60, 120), unit="°C"),
        ed.Parameter("time",   bounds=(10, 180), unit="min"),
        ed.Parameter("conc_A", bounds=(1, 10), unit="mL", kind="discrete", step=1.0),
        ed.Parameter("conc_B", bounds=(1, 10), unit="mL", kind="discrete", step=1.0),
    ],
    constraints=[
        ed.LinearConstraint(coeffs={"conc_A": 1, "conc_B": -1}, lower=1.0),
    ],
    objectives="yield_pct",
    maximize=True,
)

knowledge = (ed.Knowledge()
             .with_arrhenius("T")
             .with_monotone("time", effect="increases_objective")
             .with_quadratic_peak("conc_A", center=7.0))

campaign = ed.Campaign(space, knowledge, seed=42)

doe   = campaign.suggest_doe(n=12)         # DataFrame in °C / min / mL
y_doe = run_lab_experiments(doe)           # chemist measures
campaign.tell(doe, y_doe)

for _ in range(20):
    next_pts = campaign.ask(q=1)
    y_next   = run_lab_experiments(next_pts)
    campaign.tell(next_pts, y_next)

result = campaign.finalize()
print(result.best_x_physical, result.best_y_physical)
```

---

## Why does this exist?

We ran a series of 16 experiments comparing 10 BO methods across 2D, 4D, and 6D synthetic chemistry objectives (see the parent [DOEGP](..) project). Two things became clear:

- **Domain knowledge done right** is the biggest single lever — a correctly-encoded monotonicity prior makes BO **13× better** in 6D.
- **Domain knowledge done wrong** is catastrophic — using physical intuition naively in the BO frame degrades performance by **−653%** in 4D.

The difference between the two is one sign flip the user can't see. `expdoe-dk` makes that sign flip impossible: you write "T increases yield"; the library handles the rest.

---

## What's in the box

| Module                              | What it does                                                                        |
|-------------------------------------|-------------------------------------------------------------------------------------|
| `expdoe_dk.Space` / `Parameter`     | Physical-unit parameter space with discrete steps                                   |
| `expdoe_dk.LinearConstraint`        | Linear inequalities like `A − B ≥ 1`                                                |
| `expdoe_dk.suggest_design`          | DoE generator: LHS maximin, Sobol, Halton, D-Optimal, random — all respect constraints |
| `expdoe_dk.Knowledge`               | Composable domain-knowledge spec (Arrhenius, peaks, monotonicity, random augment)   |
| `expdoe_dk.Campaign`                | End-to-end DoE → BO loop, ask/tell/run interface                                    |

---

## Five categories of knowledge — what to use when

Based on cross-dimension experiments (2D, 4D, 6D):

| Category                                 | API                                                | Best for                                    |
|------------------------------------------|----------------------------------------------------|---------------------------------------------|
| ① Domain knowledge (correct)             | `with_arrhenius`, `with_quadratic_peak`, `with_monotone` | When you have real physics on the system   |
| ② Pure regularization (safest default)   | `with_random_augment(n=20)`                        | When you have no specific knowledge        |
| ③ Weak knowledge (GP prior alone)        | `with_gp_prior("medium")`                          | When you want hyperparameter tuning hints  |
| ④ Avoid (learnable means)                | `with_arrhenius(frozen=False)` (warns)             | Triggers a warning — usually use frozen    |
| ⑤ Avoid (mono + prior with default ε)    | `with_monotone(epsilon=0.02)` + strong prior (errors) | Will raise `EpsilonConflictError`         |

The "avoid" categories are blocked or warned against by the library; users naturally end up in ① or ②.

---

## Install

```bash
pip install -e .
# or after publish:
# pip install expdoe-dk
```

Requires Python 3.10+, BoTorch ≥ 0.11, Ax ≥ 1.2.4.

---

## Migration from `ax_doe_bo`

```python
# Old:
from ax_doe_bo import run_ax_bo
cum_best = run_ax_bo(design, bench_fn, n_bo=20, seed=42)

# New:
import expdoe_dk as ed
space = ed.Space([ed.Parameter("x0", bounds=(0, 1)), ed.Parameter("x1", bounds=(0, 1))])
campaign = ed.Campaign(space)
result = campaign.run(bench_fn, n_doe=8, n_iter=20)
```

The old import path still works via `expdoe_dk.legacy.ax_doe_bo`, but emits `DeprecationWarning`.

---

## Roadmap

| Version | Adds |
|---------|------|
| v0.1 (this) | Constrained DoE + Knowledge composition + Campaign loop + 1 example |
| v0.2 | Empirical validator (auto-warn when monotone assumption disagrees with data) |
| v0.3 | F_eps auto-rescue (Exp-14 rule applied transparently) |
| v0.4 | HTML report (`result.to_html()`) |
| v0.5 | Claude Code skill packaging |
| v0.6 | MCP server (stateless tools) |
| v0.7 | Multi-objective (qLogEHVI, Pareto reports) |
| v1.0 | Stable API, remove legacy shim |

---

## Project lineage

- Sister project: [DOEGP](..) (research notebooks + experiments 7–16)
- Original: `ax_doe_bo` (BO research framework, now deprecated)
- License: MIT
