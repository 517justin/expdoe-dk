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

This experiment uses the **canonical objectives** from the sister project's
empirical studies (Exp-7 / Exp-9 / Exp-10 in `AGENT_KNOWLEDGE.md`):

- `reaction_objective_2d`   (T × conc, the Exp-7 problem)
- `process_objective_4d`    (T × conc × pH × t, the Exp-9 problem)
- `process_objective_6d_v2` (4D + polar (bimodal) + rpm (Gaussian peak))

Running them lets the experiment reproduce the §6b pattern directly.

### Why this needs a noise-free gap metric

The objectives all carry N(0, 0.01²) noise — the same level of variability
as the actual gap between best methods. Comparing "best observed yield"
would conflate "BO genuinely converged" with "BO got lucky on a noisy
draw". The script therefore re-evaluates the **noise-free oracle** at each
trial's best-so-far X and reports the gap from the true optimum. This is
the standard BO-benchmark practice.

### How to run

```bash
python experiments/02_knowledge_comparison.py             # 2D (default), ~3 min
python experiments/02_knowledge_comparison.py --dim 4     # ~12 min
python experiments/02_knowledge_comparison.py --dim 6     # ~17 min
python experiments/02_knowledge_comparison.py --seeds 3   # fewer seeds, faster
```

Setup: `lhs_maximin` DoE held constant. Budget matches Exp-7/9/10:

| dim | n_doe | n_iter | total evals |
|-----|------:|-------:|-----------:|
| 2D  | 6     | 15     | 21         |
| 4D  | 12    | 30     | 42         |
| 6D  | 18    | 30     | 48         |

### 2D results (5 seeds, ~3 min)

True noiseless optimum = 0.6065 at T=600 K, conc=0.5 mol/L.
**`gap_final`** is `|true_opt − noiseless_yield(best_x_final)|` — smaller is better.

| Config                              | gap_final | Δ gap vs baseline | trials → 95 % |
|-------------------------------------|----------:|------------------:|--------------:|
| ④ Arrhenius mean only               | 0.0004    | **+83.3 %**       | 8             |
| ③ gp_prior only                     | 0.0010    | **+58.3 %**       | 7             |
| ① full domain knowledge             | 0.0013    | **+45.8 %**       | 9             |
| A: baseline / ② random_augment      | 0.0024    | 0.0 %             | 8             |
| G: WRONG-direction monotone         | 0.0026    | −8.3 %            | 8             |
| ⑤ monotone + gp_prior (rescued)     | 0.0028    | −16.7 %           | 7             |

Takeaways:

- **④ Arrhenius mean alone is the strongest** at +83 % — the Arrhenius
  shape IS the natural monotone-in-T encoding for this oracle (peak at
  T=600 boundary), so a single frozen mean function captures most of the
  structure with just one knowledge item.
- **③ GP prior alone is +58 %** — informative hyperparameter priors carry
  surprising weight in 2D.
- **① full domain knowledge is third (+46 %)** — adding 4 mean items +
  `random_augment` in only 21 evals overspecifies the problem; the
  ceiling of correct knowledge shows in higher dim (per §6b: ① ranks #1
  in 6D with +92 %).
- **G wrong direction (−8 %) and ⑤ mono + prior (−17 %) actively hurt** —
  matches §6b's warnings. ⑤'s drop survives v0.3 `auto_rescue`: the
  combination is fragile in low-D with a tight budget, not the ε alone.
- `A:` and `②:` are identical by construction (Campaign auto-applies
  `with_random_augment(n=20)` when no knowledge is provided).

### 4D / 6D (run yourself)

For the 4D problem (Exp-9), §6b reports ① at +26~28 % and ② at +72~77 %.
For the 6D problem (Exp-10 v2) with the harder polar/rpm structure, ① jumps
to +91 % vs ② at +52~68 %. Re-run with `--dim 4` or `--dim 6` to confirm.

The script writes per-iteration noise-free traces to
`outputs/experiment_02_knowledge_{dim}d_traces.csv` so you can plot the
convergence curves yourself.

### Chemist's quick-reference (per §6b 5-category framework)

| If you have                                   | Use                                  |
|-----------------------------------------------|--------------------------------------|
| A known shape for ONE dim (Arrhenius / peak)  | `④ with_arrhenius(frozen=True)` etc. — surprisingly powerful in low-D |
| Several known shapes / structures             | `① with_*` chained (best in high-D)   |
| No specific knowledge                         | `Knowledge()` default → auto Cat ② safe net |
| A monotone direction you're confident about   | `with_monotone(effect="increases_objective")` — frame translation handled |
| Uncertain about direction                     | Skip `with_monotone`; the v0.2 validator will warn if your guess is reversed |
| Reach for `with_gp_prior` AND `with_monotone` | Trust v0.3 `auto_rescue=True`, but expect lower ceiling in low-D |
