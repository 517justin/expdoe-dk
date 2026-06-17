# expdoe-dk

**Experimental Design of Experiments + Bayesian Optimization, with Domain Knowledge injection — for chemistry, materials, and lab experimentalists.**

If you're a chemist or materials researcher running a few dozen experiments to find the best conditions, this library gives you:

1. **Constrained, discrete-step DoE** — initial designs that respect "A must be ≥ B + 1 mL" and "this dial only steps in 0.5 mL increments". No more rounding off post-hoc.
2. **Bayesian optimization** that drives the experiment forward after the initial DoE, using a Gaussian Process surrogate.
3. **Domain-knowledge injection** — tell the optimizer that "temperature increases yield (Arrhenius)" or "pH peaks at 7" and it will use that hint, not fight you.

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

## HTML report (v0.4)

`Result.to_html(path)` produces a single self-contained HTML file you can attach to a lab notebook or share with collaborators:

```python
result = campaign.finalize()
result.to_html("campaign_report.html", title="Reaction yield optimisation")
```

Contains:

- **Best conditions** in physical units (with the parameter unit string and discrete steps formatted as integers)
- **Convergence chart** — best-so-far yield vs trial number, with DoE and BO points coloured differently. Renders with Chart.js loaded from `cdn.jsdelivr.net`; degrades gracefully to the summary + table if JavaScript is disabled
- **History table** with a one-click CSV download (embedded as a `data:` URL — no file dependency)
- **Knowledge applied** — human-readable listing of every `with_*` item, plus any notes (e.g. auto-applied `random_augment` default)
- Light / dark mode aware via `prefers-color-scheme`

`result.to_html_string()` returns the HTML as a string if you want to post-process it.

---

## Epsilon auto-rescue (v0.3)

`with_monotone` combined with `with_gp_prior` used to raise `EpsilonConflictError` when the virtual-point spacing was too small for the prior lengthscale (Exp-14). v0.3 makes this *transparent by default*: the conflicting ε is silently bumped to `0.3 × prior_lengthscale_mode` and an `EpsilonAutoRescueNotice` is emitted once per rescued item.

```python
campaign = ed.Campaign(space, knowledge,
                       auto_rescue=True)   # default — soft rescue + notice
campaign = ed.Campaign(space, knowledge,
                       auto_rescue=False)  # strict — raise EpsilonConflictError
```

The rescue is idempotent: a second `validate()` is silent.

---

## Empirical validation (v0.2)

After every K observations, the Campaign quietly cross-checks your knowledge spec against the data and warns when it sees signs of trouble:

| Validator                  | Triggers a warning when                                             |
|----------------------------|---------------------------------------------------------------------|
| `MonotoneViolationWarning` | Spearman correlation between a monotone param and Y has the opposite sign, p < 0.05 |
| `ShapePriorMismatchWarning`| A frozen Arrhenius / quadratic peak shape correlates < 0.30 with Y |

The warning text always points to a concrete remediation (`Knowledge.drop_monotone(param)`, falling back to `with_random_augment`, etc.). Each warning surfaces at most once per parameter, so a stale assumption doesn't spam the log.

Tune or disable via `Campaign(..., validate=..., validation_interval=..., validation_min_obs=...)`. Default: validate every 5 observations once you have ≥ 10 data points.

```python
campaign = ed.Campaign(space, knowledge,
                       validate=True,
                       validation_interval=5,   # how often
                       validation_min_obs=10)   # how early
```

---

## Five categories of knowledge — what to use when

Derived from sweeping the same knowledge configurations across three
synthetic chemistry oracles of increasing dimensionality (i.e. number of
process parameters being optimised at once):

- **2D** — temperature × concentration (21 evals)
- **4D** — temperature × concentration × pH × time (42 evals)
- **6D** — 4D + solvent polarity + stirring rpm (48 evals)

The reproducible study is in [`experiments/02_knowledge_comparison.py`](../experiments/02_knowledge_comparison.py); summary tables for each dimension live in [`experiments/README.md`](../experiments/README.md).

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
| v0.1 | Constrained DoE + Knowledge composition + Campaign loop + 1 example |
| v0.2 | Empirical validators (Spearman monotone + frozen-mean shape) auto-running every K observations |
| v0.3 | ε auto-rescue: combining `with_monotone` + `with_gp_prior` now transparently raises ε to the Exp-14 safe value |
| v0.4 (this) | HTML report (`result.to_html(path)`) — self-contained file with best point, convergence chart, history table, CSV download, knowledge spec |
| v0.5 | Claude Code skill packaging |
| v0.6 | MCP server (stateless tools) |
| v0.7 | Multi-objective (qLogEHVI, Pareto reports) |
| v1.0 | Stable API, remove legacy shim |

---

## Project lineage

- Sister project: [DOEGP](..) (research notebooks + experiments 7–16)
- Original: `ax_doe_bo` (BO research framework, now deprecated)

## License

Apache License, Version 2.0 — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

The historical code in the parent repository (`ax_doe_bo.py`, `doe_utils.py`,
`benchmarks.py`) was originally MIT-licensed; this rebrand re-licenses the
project under Apache 2.0. The original MIT terms remain in the git history
prior to the relicense commit.
