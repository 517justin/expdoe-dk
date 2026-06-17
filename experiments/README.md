# Experiments

Small reproducible studies built on top of the `expdoe-dk` API.

| Script | Question |
|--------|----------|
| [`01_doe_method_comparison.py`](./01_doe_method_comparison.py) | Holding knowledge fixed (plain GP), how much does the *DoE method* affect BO outcome? |
| [`02_knowledge_comparison.py`](./02_knowledge_comparison.py) | Holding the DoE method fixed, how much does the *type of domain knowledge* affect BO outcome? |

Both run on the same three canonical objectives (ports of the sister
project's Exp-7/9/10 oracles), selectable with `--dim {2,4,6}`, with the
noise-free gap-from-optimum as the headline metric. Each config is run over
5 random seeds.

```bash
python experiments/01_doe_method_comparison.py --dim 2   # ~3 min
python experiments/02_knowledge_comparison.py --dim 4    # ~9 min
```

Results write to `experiments/outputs/*.csv` (git-ignored — re-run to
reproduce).

> **Read these as directional, not definitive.** They are single-oracle-family
> runs over 5 seeds. The breadth needed to make firm recommendations is
> tracked in roadmap issues #19 / #20 / #21.

---

## Experiment 01 — DoE method comparison

Uses the same canonical objectives as experiment 02 (Exp-7 / Exp-9 /
Exp-10 v2). Knowledge is held constant at *none* (plain GP), so the only
variable is the DoE method and the comparison is fair.

```bash
python experiments/01_doe_method_comparison.py             # 2D (default)
python experiments/01_doe_method_comparison.py --dim 4     # ~12 min
python experiments/01_doe_method_comparison.py --dim 6     # ~17 min
```

Headline metric is the same noise-free `gap_final` as experiment 02:
`true_opt − noiseless_oracle(best_x_final)`. Smaller is better.

All knowledge held at *none* (plain GP). `Δ vs random_uniform` is the
relative gap reduction against the random-uniform baseline.

### 2D results on `reaction_objective_2d` (5 seeds, ~3 min)

True noiseless optimum = 0.6065 at T=600 K, conc=0.5 mol/L.

| Method             | gap_final | Δ vs random_uniform | %seeds hit 95 |
|--------------------|----------:|---------------------:|--------------:|
| `lhs_random`       | 0.0002    | **+75.0 %**          | 100 %         |
| `sobol`            | 0.0006    | +25.0 %              | 100 %         |
| `random_uniform`   | 0.0008    | 0.0 %                | 100 %         |
| `d_optimal`        | 0.0014    | −75.0 %              | 100 %         |
| `halton`           | 0.0015    | −87.5 %              | 100 %         |
| `lhs_maximin`      | 0.0019    | −137.5 %             | 100 %         |

### 4D results on `process_objective_4d` (5 seeds, ~7 min)

True noiseless optimum ≈ 0.34956.

| Method             | gap_final | Δ vs random_uniform | %seeds hit 95 |
|--------------------|----------:|---------------------:|--------------:|
| `lhs_random`       | 0.0011    | **+60.7 %**          | 100 %         |
| `lhs_maximin`      | 0.0028    | 0.0 %                | 100 %         |
| `random_uniform`   | 0.0028    | 0.0 %                | 100 %         |
| `sobol`            | 0.0033    | −17.9 %              | 100 %         |
| `d_optimal`        | 0.0037    | −32.1 %              | 100 %         |
| `halton`           | 0.0082    | −192.9 %             | 100 %         |

### 6D results on `process_objective_6d_v2` (5 seeds, ~11 min)

True noiseless optimum ≈ 0.34956. The hardened oracle (bimodal `polar`,
Gaussian peak on `rpm`) is the most punishing for DoE-only initialisation.

| Method             | gap_final | Δ vs random_uniform | %seeds hit 95 |
|--------------------|----------:|---------------------:|--------------:|
| `halton`           | 0.0064    | **+54.3 %**          | 80 %          |
| `lhs_random`       | 0.0074    | **+47.1 %**          | 80 %          |
| `random_uniform`   | 0.0140    | 0.0 %                | 80 %          |
| `sobol`            | 0.0160    | −14.3 %              | 80 %          |
| `lhs_maximin`      | 0.0224    | −60.0 %              | 60 %          |
| `d_optimal`        | 0.0536    | −282.9 %             | 0 %           |

> **There is no single "best" DoE method.** The ranking reshuffles every
> time the dimensionality changes:
> - 2D: `lhs_random` wins (+75 %); `lhs_maximin` is *last*.
> - 4D: `lhs_random` wins again (+61 %); `sobol`/`halton` slip below
>   `random_uniform`.
> - 6D: `halton` wins (+54 %); `lhs_random` second; `lhs_maximin` and
>   `d_optimal` are well below baseline.
>
> No method is in the top tier across all three dimensions except
> `lhs_random`, which is consistently strong but never guaranteed best.
> `d_optimal` degrades monotonically with dimension and is the worst in
> 6D — its greedy boundary-seeking selection misses interior optima.

> **Practical implication: be cautious about running DoE on many
> dimensions at once.** The benefit of a clever space-filling design
> shrinks and becomes more method-dependent as dimensionality grows
> (note how the gaps and the spread between methods both widen from 2D to
> 6D, and `%seeds hit 95` starts dropping). If you have ≥ 5 active
> factors, the higher-leverage move is usually to **reduce
> dimensionality first** — fix the parameters you are most certain about,
> screen the rest with a cheap factorial (Plackett–Burman /
> fractional-factorial), and run the BO loop on the surviving 2–4. If you
> must stay high-D, `lhs_random` or `halton` are the safer picks and
> `d_optimal` should be avoided.

### Cross-dim takeaways

- **`lhs_random` is the most consistent performer** — top of the table in
  2D and 4D, second in 6D. If you want one default, it's the safest.
- **`lhs_maximin` is not the automatic winner.** The SA-optimised maximin
  design is mid-pack here and *last* in 2D; its main virtue is robustness
  (100 % seeds-hit-95 % in 2D/4D), not the smallest gap.
- **`d_optimal` degrades with dimension** — fine-ish in 2D/4D, worst by a
  wide margin in 6D. Its greedy boundary-seeking selection concentrates
  near the edges and misses interior / multimodal optima. Not recommended
  for GP BO.
- The earlier, more dramatic "everything collapses in 6D" reading was
  partly an artefact of the old auto-`random_augment` baseline; against a
  plain GP the high-D picture is milder but the no-single-winner
  conclusion stands.

### Recommendations

| Setting | Suggested default |
|---------|-------------------|
| 2D–4D, small budget | `lhs_random` (top in both) |
| 6D+ if you stay high-D | `halton` or `lhs_random`; avoid `lhs_maximin` / `d_optimal` |
| ≥ 5 active factors | Reduce dimensionality first (screen, then BO on 2–4) |
| Linear / quadratic surrogate assumed | `d_optimal` (NOT for GP BO) |
| Sanity-check baseline | `random_uniform` — a respectable baseline, not embarrassing |

**Headline reading from this experiment:** there is no universally best
DoE method; `lhs_random` is the most reliable single choice, the benefit
of any DoE method narrows at high dimensionality, and `d_optimal` is the
one to avoid for GP-based BO.

---

## Experiment 02 — Knowledge comparison

This experiment uses the **canonical objectives** from the sister project's
empirical studies (Exp-7 / Exp-9 / Exp-10 in `AGENT_KNOWLEDGE.md`):

- `reaction_objective_2d`   (T × conc, the Exp-7 problem)
- `process_objective_4d`    (T × conc × pH × t, the Exp-9 problem)
- `process_objective_6d_v2` (4D + polar (bimodal) + rpm (Gaussian peak))

> **Baseline change (important).** Earlier versions of this experiment used
> a baseline that silently auto-applied `with_random_augment(n=20)`. That
> auto-default has been **removed** — `A: baseline` is now a genuine plain
> GP (`knowledge=None`). The corrected baseline tells a very different (and
> more honest) story: a plain GP is a *strong* competitor, and several
> knowledge configs — including `random_augment` itself — can do worse than
> doing nothing on these oracles. All `Δ gap` numbers below are now relative
> to the plain-GP baseline.

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
`Δ gap` is relative to the plain-GP baseline.

| Config                              | gap_final | Δ gap vs baseline | %seeds hit 95 |
|-------------------------------------|----------:|------------------:|--------------:|
| ④ Arrhenius mean only               | 0.0004    | **+78.9 %**       | 100 %         |
| ③ gp_prior only                     | 0.0010    | **+47.4 %**       | 100 %         |
| ① full domain knowledge             | 0.0013    | **+31.6 %**       | 100 %         |
| **A: baseline (plain GP)**          | 0.0019    | 0.0 %             | 100 %         |
| ② random_augment only               | 0.0024    | −26.3 %           | 100 %         |
| G: WRONG-direction monotone         | 0.0026    | −36.8 %           | 100 %         |
| ⑤ monotone + gp_prior (rescued)     | 0.0028    | −47.4 %           | 100 %         |

In 2D, three knowledge configs beat the plain GP: **④ Arrhenius** (+79 %, the
Arrhenius shape is the right encoding for the T peak), **③ gp_prior** (+47 %),
and **① full domain knowledge** (+32 %). Everything below the baseline line —
including `② random_augment` — does *worse than doing nothing*.

### 4D results on `process_objective_4d` (5 seeds, ~9 min)

True noiseless optimum ≈ 0.34956.

| Config                              | gap_final | Δ gap vs baseline | %seeds hit 95 |
|-------------------------------------|----------:|------------------:|--------------:|
| ④ Arrhenius mean only               | 0.0020    | **+28.6 %**       | 100 %         |
| **A: baseline (plain GP)**          | 0.0028    | 0.0 %             | 100 %         |
| ③ gp_prior only                     | 0.0063    | −125 %            | 100 %         |
| ① full domain knowledge             | 0.0076    | −171 %            | 60 %          |
| ⑤ mono+prior (rescued)              | 0.0088    | −214 %            | 80 %          |
| ② random_augment only               | 0.0130    | −364 %            | 100 %         |
| G: WRONG-direction monotone         | 0.0716    | −2457 %           | 20 %          |

In 4D, **only ④ Arrhenius beats the plain GP**. The plain GP is already a
strong learner here (100 % seeds reach 95 % of the optimum), and most
knowledge configs — including our hand-built ① "full" config — get in its
way. `② random_augment` is 4.6× worse than the baseline.

### 6D results on `process_objective_6d_v2` (5 seeds, ~13 min)

| Config                              | gap_final | Δ gap vs baseline | %seeds hit 95 |
|-------------------------------------|----------:|------------------:|--------------:|
| **A: baseline (plain GP)**          | 0.0224    | 0.0 %             | 60 %          |
| ① full domain knowledge             | 0.0345    | −54 %             | 40 %          |
| ③ gp_prior only                     | 0.0427    | −91 %             | 20 %          |
| ⑤ mono+prior (rescued)              | 0.0741    | −231 %            | 0 %           |
| ② random_augment only               | 0.0844    | −277 %            | 0 %           |
| G: WRONG-direction monotone         | 0.1490    | −565 %            | 0 %           |
| ④ Arrhenius mean only               | 0.1924    | −759 %            | 0 %           |

In 6D, **nothing beats the plain GP** on this oracle. ① full domain knowledge
is the least-bad knowledge config; ④ Arrhenius-alone — the 2D/4D winner —
collapses to last place. This is the dimension-sensitivity pattern: a single
mean shape that captures most of a low-D landscape becomes actively
misleading when 5 other dimensions interact.

### What this actually says

The corrected (plain-GP) baseline overturns the tidy "knowledge always
helps" story and replaces it with three honest, more useful findings:

1. **`random_augment` hurt on all three oracles** (−26 % / −364 % / −277 %).
   This is the empirical basis for removing it as an auto-default. It is a
   real example of why "add some regularization for free" is not safe —
   see roadmap issue #20 for the planned validation sweep over `n` and
   more datasets.
2. **A plain GP is a strong baseline** — stronger than we credited while
   the old default was quietly diluting it. Knowledge has to *earn* its
   place against it.
3. **Correctly-shaped knowledge still wins where it matches the landscape**:
   ④ Arrhenius is the clear 2D/4D winner because the oracle's temperature
   term *is* Arrhenius. But the same config is the worst in 6D — so the
   shape has to match the problem, and a hand-built "full" config of
   stacked priors (①) is not automatically better.

These are single-oracle-family results on 5 seeds; treat them as
directional. The breadth needed to make firm recommendations is tracked in
roadmap issues #19 (DoE across datasets) and #20 (knowledge across
datasets, esp. pure regularization). The fact that our purpose-built
primitives (`with_arrhenius` etc.) match these oracles but may not match a
new dataset is exactly what roadmap issue #21 (generalising the
knowledge API) addresses.

### Chemist's quick-reference (per §6b 5-category framework)

| If you have                                   | Use                                  |
|-----------------------------------------------|--------------------------------------|
| A known shape for ONE low-D parameter (Arrhenius / peak) | `with_arrhenius` / `with_quadratic_peak` (`frozen=True`) — the clear winner in 2D/4D when the shape matches |
| No specific knowledge                         | **Nothing — `Campaign(space)` runs a plain GP.** It's a strong baseline; don't add structure you can't justify |
| A monotone direction you're confident about   | `with_monotone(effect="increases_objective")` — frame translation handled; v0.2 validator warns if the data disagrees |
| The temptation to "regularise for free"       | Avoid `with_random_augment` for now — on these oracles it hurt vs a plain GP. It's opt-in and under validation (#20) |
| `with_gp_prior` AND `with_monotone` together  | Trust v0.3 `auto_rescue=True`; expect a lower ceiling in low-D |
| A high-dimensional problem (≥5 active factors) | Reduce dimensionality first; a plain GP was the best config in our 6D test |
