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

Uses the same canonical objectives as experiment 02 (Exp-7 / Exp-9 /
Exp-10 v2). Knowledge config held constant (Campaign auto-applies Cat ②
default), so the comparison is fair.

```bash
python experiments/01_doe_method_comparison.py             # 2D (default)
python experiments/01_doe_method_comparison.py --dim 4     # ~12 min
python experiments/01_doe_method_comparison.py --dim 6     # ~17 min
```

Headline metric is the same noise-free `gap_final` as experiment 02:
`true_opt − noiseless_oracle(best_x_final)`. Smaller is better.

### 2D results on `reaction_objective_2d` (5 seeds, ~3 min)

True noiseless optimum = 0.6065 at T=600 K, conc=0.5 mol/L.

| Method             | gap_final | Δ vs random_uniform | trials → 95 % |
|--------------------|----------:|---------------------:|--------------:|
| `lhs_random`       | 0.0013    | **+72.9 %**          | 10            |
| `lhs_maximin`      | 0.0024    | **+50.0 %**          | 8             |
| `d_optimal`        | 0.0036    | +25.0 %              | 8             |
| `sobol`            | 0.0037    | +22.9 %              | 11            |
| `random_uniform`   | 0.0048    | 0.0 %                | 8             |
| `halton`           | 0.0055    | −14.6 %              | 8             |

### 4D results on `process_objective_4d` (5 seeds, ~7 min)

True noiseless optimum ≈ 0.34956.

| Method             | gap_final | Δ vs random_uniform | trials → 95 % | %seeds hit 95 |
|--------------------|----------:|---------------------:|--------------:|--------------:|
| `sobol`            | 0.0050    | **+70.1 %**          | 33            | 80 %          |
| `d_optimal`        | 0.0066    | **+60.5 %**          | 37            | 60 %          |
| `lhs_random`       | 0.0100    | +40.1 %              | 25            | 100 %         |
| `halton`           | 0.0124    | +25.7 %              | 26            | 60 %          |
| `lhs_maximin`      | 0.0130    | +22.2 %              | 22            | 100 %         |
| `random_uniform`   | 0.0167    | 0.0 %                | 31            | 60 %          |

### 6D results on `process_objective_6d_v2` (5 seeds, ~11 min)

True noiseless optimum ≈ 0.34956. The hardened oracle (bimodal `polar`,
Gaussian peak on `rpm`) is far more punishing for DoE-only initialisation.

| Method             | gap_final | Δ vs random_uniform | %seeds hit 95 |
|--------------------|----------:|---------------------:|--------------:|
| `random_uniform`   | 0.0111    | 0.0 %                | 80 %          |
| `halton`           | 0.0121    | −9.0 %               | 80 %          |
| `lhs_random`       | 0.0188    | −69.4 %              | 40 %          |
| `sobol`            | 0.0284    | −155.9 %             | 20 %          |
| `lhs_maximin`      | 0.0844    | −660.4 %             | 0 %           |
| `d_optimal`        | 0.3309    | −2881 %              | 20 %          |

> **There is no single "best" DoE method.** Every structural method that
> wins in 2D / 4D underperforms `random_uniform` here, and `d_optimal`
> degrades by almost 30×. A method's coverage strategy is tuned to a
> particular landscape — once the landscape changes (bimodal dim,
> off-centre Gaussian peak), the strategy backfires.
>
> **Practical implication: avoid using DoE on too many dimensions at
> once.** 6 parameters is already past the point where well-spaced
> initial designs reliably help. If you have ≥ 5 active factors:
>
> 1. Reduce dimensionality first — fix the dims you have least uncertainty
>    on, screen the rest with a cheap factorial (Plackett–Burman /
>    fractional-factorial), then run the BO loop on the surviving 2–4.
> 2. If you must stay in 5D+, pick `random_uniform` or `halton` — they
>    don't get stuck on the wrong structural assumption — and lean
>    heavily on knowledge (experiment 02 below) and a larger BO budget.

### Cross-dim takeaways

- **The DoE-method ranking changes with dimension.** No single method
  dominates 2D / 4D / 6D simultaneously.
  - 2D: LHS variants top, halton bottom
  - 4D: Sobol top, lhs_maximin / lhs_random mid
  - 6D: random_uniform is the most robust; structural QMC methods
    collapse on the bimodal `polar` × Gaussian `rpm` landscape
- **`lhs_maximin`'s strength is robustness, not the best gap**: it has
  100 % seeds-hit-95 % in 2D and 4D — it never lands in catastrophe even
  if it doesn't reach the smallest gap.
- **`d_optimal` fails in 6D** — the greedy maximin selection over a
  feasible candidate pool concentrates near boundaries on
  multimodal/Gaussian-peaked dims, missing all the interior optima.

### Recommendations

| Setting | Suggested default |
|---------|-------------------|
| 2D / n_init ≤ 10 (interior peaks) | `lhs_random` / `lhs_maximin` |
| 3–4D mid-budget | `sobol` for the cleanest gap, `lhs_maximin` for robustness |
| 5–6D high-noise / multi-modal | **Reduce dimensionality first.** If you must stay high-D, prefer `random_uniform` or `halton` — structural methods backfire. |
| Linear / quadratic surrogate assumed | `d_optimal` (NOT for GP BO) |
| Sanity-check baseline | `random_uniform` |

**Headline reading from this experiment:** at high dimensionality the
benefit of clever DoE shrinks (and can flip negative). Spend the
engineering effort on dimensionality reduction and on injecting correct
domain knowledge (experiment 02 below) before tuning the DoE method.

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

### 4D results on `process_objective_4d` (5 seeds, ~9 min)

True noiseless optimum ≈ 0.34956.

| Config                              | gap_final | Δ gap vs baseline | trials → 95 % | %seeds hit 95 |
|-------------------------------------|----------:|------------------:|--------------:|--------------:|
| ④ Arrhenius mean only               | 0.0020    | **+84.6 %**       | 19            | 100 %         |
| ③ gp_prior only                     | 0.0063    | **+51.5 %**       | 25            | 100 %         |
| ① full domain knowledge             | 0.0076    | **+41.5 %**       | 37            | 60 %          |
| ⑤ mono+prior (rescued)              | 0.0088    | +32.3 %           | 30            | 80 %          |
| A / ②: baseline / random_augment    | 0.0130    | 0.0 %             | 22            | 100 %         |
| G: WRONG-direction monotone         | 0.0716    | **−450.8 %**      | 43            | 20 %          |

### 6D results on `process_objective_6d_v2` (5 seeds, ~13 min)

| Config                              | gap_final | Δ gap vs baseline | %seeds hit 95 |
|-------------------------------------|----------:|------------------:|--------------:|
| ① full domain knowledge             | 0.0345    | **+59.1 %**       | 40 %          |
| ③ gp_prior only                     | 0.0427    | **+49.4 %**       | 20 %          |
| ⑤ mono+prior (rescued)              | 0.0741    | +12.2 %           | 0 %           |
| A / ②: baseline / random_augment    | 0.0844    | 0.0 %             | 0 %           |
| G: WRONG-direction monotone         | 0.1490    | −76.5 %           | 0 %           |
| ④ Arrhenius mean only               | 0.1924    | **−128.0 %**      | 40 %          |

### The U-shape — knowledge value vs dimension

`Δ gap vs baseline (%)` across all three dimensions, for the same configs:

| Config                              | 2D       | 4D       | 6D       | Pattern |
|-------------------------------------|---------:|---------:|---------:|---------|
| ① full domain knowledge             | +45.8 %  | +41.5 %  | **+59.1 %** | mild U |
| ④ Arrhenius mean only               | **+83.3 %** | **+84.6 %** | **−128.0 %** | dim-sensitive collapse |
| ③ gp_prior only                     | +58.3 %  | +51.5 %  | +49.4 %  | most stable across dims |
| ⑤ mono+prior (rescued)              | −16.7 %  | +32.3 %  | +12.2 %  | auto_rescue value rises with dim |
| G WRONG-direction monotone          | −8.3 %   | **−450.8 %** | −76.5 %  | wrong knowledge worst in mid-dim |

This reproduces the AGENT_KNOWLEDGE.md §6b finding:

- **①** is a mild U (45.8 → 41.5 → 59.1): mid-compression, high-dim rise.
- **④** is the textbook dimension-sensitive case: a single learnable mean
  alone wins big in 2D / 4D but **catastrophically fails in 6D** (−128 %).
  This is exactly why `frozen=True` is the default and learnable variants
  warn — and why §6b puts mean-function-only configurations in Cat ④.
- **③** is the steady performer across dims (+58 / +51 / +49 %) — Cat ③
  "stable middle".
- **⑤** with v0.3 auto_rescue goes from a low-D liability (−17 % in 2D)
  to a high-D contributor (+12 % in 6D), showing the ε-rescue rule from
  Exp-14 actually pays off as the GP fit gets harder.
- **G** wrong direction is worst exactly where knowledge matters most
  (4D, −451 %). The v0.2 Spearman validator would have warned here.

### Chemist's quick-reference (per §6b 5-category framework)

| If you have                                   | Use                                  |
|-----------------------------------------------|--------------------------------------|
| A known shape for ONE dim (Arrhenius / peak)  | `④ with_arrhenius(frozen=True)` etc. — surprisingly powerful in low-D |
| Several known shapes / structures             | `① with_*` chained (best in high-D)   |
| No specific knowledge                         | `Knowledge()` default → auto Cat ② safe net |
| A monotone direction you're confident about   | `with_monotone(effect="increases_objective")` — frame translation handled |
| Uncertain about direction                     | Skip `with_monotone`; the v0.2 validator will warn if your guess is reversed |
| Reach for `with_gp_prior` AND `with_monotone` | Trust v0.3 `auto_rescue=True`, but expect lower ceiling in low-D |
