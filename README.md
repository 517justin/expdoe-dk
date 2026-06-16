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
result.to_html("campaign_report.html")     # share-ready HTML
```

The package lives in [`expdoe-dk/`](./expdoe-dk/). The historical research framework (Ax+BoTorch wrapper) is kept in this directory as `ax_doe_bo.py` / `doe_utils.py` / `benchmarks.py` for reproducibility; new work should use `expdoe-dk`.

---

## Install

```bash
cd expdoe-dk
pip install -e .          # editable install
```

Requires Python 3.10+, BoTorch ≥ 0.11, Ax ≥ 1.2.4.

---

## Repository layout

```
expdoe-dk/                          # ★ the package — use this
  src/expdoe_dk/
    space.py                        # Parameter, LinearConstraint, Space
    doe/                            # 6 DoE methods (LHS maximin / Sobol / Halton / ...)
    knowledge/                      # Knowledge composition + frame translator
    bo/                             # Campaign + HTML report
    legacy/                         # ax_doe_bo backward-compat shims
  examples/
    01_reaction_optimization.{py,ipynb}    # chemistry workflow end-to-end
    02_html_report.py                       # → v0.4 HTML report
  experiments/
    01_doe_method_comparison.py             # six DoE methods × same oracle
    02_knowledge_comparison.py              # five knowledge categories × same oracle
  tests/                            # 53 unit + integration tests
  LICENSE / NOTICE                  # Apache 2.0

ax_doe_bo.py / doe_utils.py / benchmarks.py    # historical research framework
README.md                          # this file
```

---

## Examples and experiments

| Path | What it does |
|------|--------------|
| [`expdoe-dk/examples/01_reaction_optimization.py`](./expdoe-dk/examples/01_reaction_optimization.py) | A chemist runs DoE → BO end-to-end with knowledge injection. Finds the true optimum in 23 evals. |
| [`expdoe-dk/examples/02_html_report.py`](./expdoe-dk/examples/02_html_report.py) | Reproduces the v0.4 HTML report (`Result.to_html`). |
| [`expdoe-dk/experiments/01_doe_method_comparison.py`](./expdoe-dk/experiments/01_doe_method_comparison.py) | Holds knowledge fixed, varies the DoE method. Quantifies how `lhs_maximin` vs Sobol vs random impacts BO. |
| [`expdoe-dk/experiments/02_knowledge_comparison.py`](./expdoe-dk/experiments/02_knowledge_comparison.py) | Holds DoE fixed, varies knowledge injection (baseline / random_augment / mean function / monotone / combo). Reproduces the 5-category framework empirically. |

Run any of them:

```bash
python expdoe-dk/examples/01_reaction_optimization.py
python expdoe-dk/experiments/01_doe_method_comparison.py
```

---

## Five categories of knowledge — what to use when

Based on cross-dimension experiments (2D, 4D, 6D):

| Category                                | API                                                | Best for                                    |
|-----------------------------------------|----------------------------------------------------|---------------------------------------------|
| ① Domain knowledge (correct)            | `with_arrhenius`, `with_quadratic_peak`, `with_monotone` | When you have real physics on the system   |
| ② Pure regularization (safest default)  | `with_random_augment(n=20)`                        | When you have no specific knowledge        |
| ③ Weak knowledge (GP prior alone)       | `with_gp_prior("medium")`                          | When you want hyperparameter tuning hints  |
| ④ Avoid (learnable means)               | `with_arrhenius(frozen=False)` (warns)             | Triggers a warning — usually use frozen    |
| ⑤ Avoid (mono + prior, default ε)       | `with_monotone(epsilon=0.02)` + strong prior        | Now auto-rescued; opt out with `auto_rescue=False` |

Steering happens through API defaults: users naturally end up in ① or ②.

---

## Safe-by-default behaviours

| Pitfall (found empirically — see AGENT_KNOWLEDGE.md) | What `expdoe-dk` does |
|------|------|
| `monotone_dims={dim: "increasing"}` is in user / yield space, but BO minimizes `-yield`, so the GP sees the reversed direction | `with_monotone(effect="increases_objective")` is in physical space; `_frame.flip_for_minimize` translates internally |
| `MonotonicGPWithDerivatives` ε=0.02 conflicts with Gamma(3,6) lengthscale prior → 13× worse fit | `epsilon="auto"` resolves to `0.3 × prior_lengthscale_mode`; explicit small ε now auto-rescues with a notice (`v0.3`) |
| Learnable mean parameters get absorbed by MLE → mean function adds no signal | `Arrhenius`, `QuadraticMean` default to `frozen=True`; learnable variants emit `LearnableMeanAbsorptionWarning` |
| Wrong monotone assumption silently hurts | After K observations the Campaign runs a Spearman check and warns with `MonotoneViolationWarning` (`v0.2`) |
| No knowledge given at all → user thinks they need to learn the whole API | Campaign auto-applies `with_random_augment(n=20)` and logs a one-liner |

---

## Roadmap

| Version | Adds | Status |
|---------|------|------|
| v0.1 | Constrained DoE + Knowledge composition + Campaign loop + 1 example | [released](https://github.com/517justin/expdoe-dk/releases/tag/v0.1.0) |
| v0.2 | Empirical validators (Spearman monotone + frozen-mean shape) auto-running every K observations | [released](https://github.com/517justin/expdoe-dk/releases/tag/v0.2.0) |
| v0.3 | ε auto-rescue: `with_monotone` + `with_gp_prior` now transparently raises ε to the Exp-14 safe value | [released](https://github.com/517justin/expdoe-dk/releases/tag/v0.3.0) |
| v0.4 | HTML report (`Result.to_html()`) | [released](https://github.com/517justin/expdoe-dk/releases/tag/v0.4.0) |
| v0.5 | Claude Code skill packaging | pending |
| v0.6 | MCP server | pending |
| v0.7 | Multi-objective (qLogEHVI, Pareto) | pending |
| v1.0 | Stable API, remove legacy shim | pending |

---

## License

Apache License, Version 2.0 — see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

The historical code in this directory (`ax_doe_bo.py`, `doe_utils.py`, `benchmarks.py`) was originally MIT-licensed; the rebrand re-licenses the repository under Apache 2.0. The original MIT terms remain available in the git history.

---

## Appendix — historical research framework (`ax_doe_bo.py`)

Before the rebrand, this repository was a research framework studying DoE → BO bridging via Ax + BoTorch (`ax_doe_bo.py` with 5 parts: GenerationStrategy / Ax-vs-pure-BoTorch / SAASBO / batch BO / JSON checkpoint).

The five parts and their results are preserved in [`ax_doe_bo.py`](./ax_doe_bo.py); the convergence figures live in `outputs/`. Quick re-run:

```bash
python ax_doe_bo.py            # runs Parts B / C / D / E sequentially
```

Highlights:
- Ax-BoTorch vs Pure BoTorch on 4 benchmarks (Branin / Hartmann / Rosenbrock / Ackley) — equivalent or better with Ax's abstraction.
- SAASBO for 4D underperforms standard GP (designed for ≥ 20D).
- Batch BO q=4 achieves 16× better gap than sequential q=1 on Branin 2D.
- JSON checkpoint/resume works end-to-end.

The lessons from this framework (esp. the EI direction sign-flip and frozen mean function rules) are now baked into `expdoe-dk`'s API defaults.
