# Experiments

Small reproducible studies built on top of the `expdoe-dk` API. Each is
self-contained and runs in 2–3 minutes on a laptop.

| Script | Question |
|--------|----------|
| [`01_doe_method_comparison.py`](./01_doe_method_comparison.py) | Holding knowledge fixed, how much does the *DoE method* affect BO outcome? |
| [`02_knowledge_comparison.py`](./02_knowledge_comparison.py) | Holding DoE fixed, how much does the *type of domain knowledge* affect BO outcome? |

Both use the same toy chemistry oracle as `examples/01_reaction_optimization.py`
(4D, peak at T=95 °C, time=120 min, conc_A=7 mL, conc_B=3 mL, with the
A − B ≥ 1 mL constraint). Each runs 3 seeds × small budget.

Run:

```bash
python experiments/01_doe_method_comparison.py
python experiments/02_knowledge_comparison.py
```

Results write to `experiments/outputs/*.csv` (git-ignored — re-run to
reproduce).

---

## Experiment 01 — DoE method comparison

Setup: `n_doe = 8`, `n_iter = 12`, `knowledge = None` (Campaign auto-applies
`with_random_augment(n=20)` — same Cat ② default for every method, so the
comparison is fair).

Best yield across 3 seeds (oracle optimum = 25.90 %):

| Method            | median | best   | worst  | std   |
|-------------------|-------:|-------:|-------:|------:|
| `halton`          | 25.90  | 25.90  | 25.51  | 0.23  |
| `lhs_maximin`     | 25.90  | 25.90  | 21.53  | 2.52  |
| `sobol`           | 25.83  | 25.90  | 23.65  | 1.28  |
| `lhs_random`      | 24.87  | 25.90  | 24.35  | 0.79  |
| `random_uniform`  | 24.67  | 25.90  | 24.18  | 0.89  |

Takeaways:
- Three QMC-style methods (`halton`, `lhs_maximin`, `sobol`) tie for top
  median — all land on the true optimum across seeds.
- `lhs_maximin` has the widest spread (one seed got stuck at 21.5) because
  the SA optimisation favours coverage even when feasibility regions are
  narrow under the A−B≥1 constraint.
- `random_uniform` underperforms — confirming the standard textbook
  recommendation that quasi-random initialisation beats pure random for
  small BO budgets.

The takeaway for chemists: **use `lhs_maximin` by default**, switch to
`sobol`/`halton` if you have ≥ 20 initial points and want lower variance.

---

## Experiment 02 — Knowledge comparison

Setup: `lhs_maximin` DoE held constant, `n_doe = 8`, `n_iter = 12`.

Six knowledge configurations corresponding to the five categories from
`AGENT_KNOWLEDGE.md` §6b, plus an `A: baseline` that uses no knowledge
(Campaign auto-applies `with_random_augment(n=20)`, so it doubles as a
sanity check that the Cat ② default is competitive).

Best yield across 3 seeds:

| Config                                | median | best   | worst  | std  |
|---------------------------------------|-------:|-------:|-------:|-----:|
| `A: baseline (auto random_augment)`   | 25.90  | 25.90  | 21.53  | 2.52 |
| `②: random_augment only`              | 25.90  | 25.90  | 21.53  | 2.52 |
| `④: Arrhenius mean only`              | 25.90  | 25.90  | 19.69  | 3.58 |
| `③: gp_prior only`                    | 25.90  | 25.90  | 25.90  | 0.00 |
| `⑤: monotone + gp_prior (rescued)`    | 25.90  | 25.90  | 25.67  | 0.13 |
| `①: full domain knowledge`            | 24.97  | 25.90  | 13.36  | 6.98 |

Takeaways:
- **`A:` and `②:` are identical** by construction: with no knowledge given,
  Campaign auto-applies `with_random_augment(n=20)`. This is the documented
  Cat ② safe default.
- **`③: gp_prior only` and `⑤: monotone + gp_prior` are the most stable**
  in 4D — std is 0 / 0.13, both consistently reach the optimum. This is
  consistent with AGENT_KNOWLEDGE.md's finding that Cat ③ gives "stable
  middle" performance.
- **`①: full domain knowledge` looks worst here in 4D** — one seed (44)
  landed at 13.36, dragging the median down. This is the *known* 4D
  weakness of the correct-domain-knowledge category from AGENT_KNOWLEDGE.md
  §6b: it shines in 2D and 6D but is *compressed* in 4D where pure
  regularisation can catch up. Expect ① to dominate in higher-D problems
  with the same setup.
- **`⑤: monotone + gp_prior` does NOT crash here** because v0.3
  `auto_rescue=True` (the default) silently bumps ε to the Exp-14 safe
  value. Try `auto_rescue=False` to reproduce the original conflict.

The takeaway for chemists:
- If you have **real domain knowledge** that's known to be correct in the
  *BO frame* (`with_monotone(effect="increases_objective")` handles the
  translation), use ① in high-dimensional problems.
- If you're unsure, **let the default (`Knowledge()` → auto Cat ②)** carry
  you. It will not crash, will not silently mislead, and is competitive
  in mid-dim problems.
- Avoid learnable mean functions (`frozen=False`) — the library will warn,
  but `④` here used `frozen=True` and was still wide-spread because the
  Arrhenius shape alone is dimension-sensitive.
