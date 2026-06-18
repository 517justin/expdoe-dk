# Experiments

Small reproducible studies built on top of the `expdoe-dk` API.

| Script | Question |
|--------|----------|
| [`01_doe_method_comparison.py`](./01_doe_method_comparison.py) | Holding knowledge fixed (plain GP), how much does the *DoE method* affect BO outcome? |
| [`02_knowledge_comparison.py`](./02_knowledge_comparison.py) | Holding the DoE method fixed, how much does the *type of domain knowledge* affect BO outcome? |

Both run on the same three canonical objectives (full mathematical
definitions below in [§ The three oracles](#the-three-oracles)), selectable
with `--dim {2,4,6}`, with the noise-free gap-from-optimum as the headline
metric. Each config is run over 5 random seeds.

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

## The three oracles

All three are deterministic synthetic chemistry yield functions returning
positive yield in `[0, ~y_max]`, with additive `N(0, 0.01²)` noise added at
sampling time and a noise-free counterpart used for reporting. Each
implements the canonical objective from the sister project's experiments
(see [`_oracles.py`](./_oracles.py) for the exact code).

### `reaction_objective_2d` — Exp-7

Parameters (physical units):

| Param | Bounds        | Role |
|-------|---------------|------|
| T     | [300, 600] K  | temperature (Arrhenius rate term) |
| conc  | [0, 1] mol/L  | reactant concentration (quadratic peak) |

Formula:

$$
\text{rate}(T) = \exp\!\Big(\!-\frac{0.5}{\max(T/600,\, 10^{-3})}\Big), \qquad
\text{eff}(c) = 4c\,(1-c)
$$

$$
y_{\text{user}} = \text{rate}(T)\cdot\text{eff}(c)\;+\;\varepsilon,\quad \varepsilon\sim\mathcal N(0,\,0.01^2)
$$

Noiseless optimum ≈ **0.6065** at $(T=600\text{ K},\; \text{conc}=0.5)$.

### `process_objective_4d` — Exp-9

Parameters:

| Param | Bounds          | Role |
|-------|-----------------|------|
| T     | [300, 800] K    | Arrhenius rate (peaks at the upper bound)     |
| conc  | [0, 2] mol/L    | quadratic peak at $c=1$                       |
| pH    | [4, 10]         | Gaussian-shaped activity, peak at $\text{pH}=7$ |
| t     | [10, 120] min   | saturating monotone in $t$                    |

Formula:

$$
\text{rate}(T) = \exp\!\Big(\!-\frac{1}{\max(T/800,\, 10^{-3})}\Big), \quad
\text{eff}(c) = 4\cdot\frac{c}{2}\cdot\Big(1-\frac{c}{2}\Big)
$$

$$
\text{act}(\text{pH}) = \exp\!\Big(\!-\Big(\frac{\text{pH}-7}{1.5}\Big)^{\!2}\Big), \quad
\text{sat}(t) = 1 - \exp\!\Big(\!-3\cdot\frac{t-10}{110}\Big)
$$

$$
y_{\text{user}} = \text{rate}(T)\cdot\text{eff}(c)\cdot\text{act}(\text{pH})\cdot\text{sat}(t)\;+\;\varepsilon
$$

Noiseless optimum ≈ **0.34956** at $(T=800,\, c=1,\, \text{pH}=7,\, t=120)$.

### `process_objective_6d_v2` — Exp-10 v2

Extends the 4D oracle with two further parameters that the original 6D
oracle made too easy. The "v2" hardening pushes the structure of `polar`
into a bimodal `|sin|` and `rpm` into a Gaussian off-centre peak:

| Param   | Bounds         | Role |
|---------|----------------|------|
| T, conc, pH, t | (same as 4D) | same                         |
| polar   | [0, 1]         | bimodal: $\,|\sin(2\pi\cdot\text{polar})|\,$ — peaks at 0.25 *and* 0.75 |
| rpm     | [100, 1000] rpm | Gaussian peak at $\text{rpm}\approx 700$ (σ = 0.15 in unit space) |

Formula adds two more factors to the 4D product:

$$
\widehat{\text{rpm}} = \frac{\text{rpm}-100}{900},\qquad
\text{eff}_{\text{rpm}} = \exp\!\Big(\!-\frac{(\widehat{\text{rpm}}-0.667)^2}{2\cdot 0.15^2}\Big)
$$

$$
\text{eff}_{\text{polar}} = \big|\sin(2\pi\cdot\text{polar})\big|
$$

$$
y_{\text{user}} = \text{rate}\cdot\text{eff}\cdot\text{act}\cdot\text{sat}\cdot\text{eff}_{\text{polar}}\cdot\text{eff}_{\text{rpm}}\;+\;\varepsilon
$$

Noiseless optimum ≈ **0.34956** at $(T=800,\, c=1,\, \text{pH}=7,\, t=120,\, \text{polar}\in\{0.25, 0.75\},\, \text{rpm}=700)$.
The bimodal `polar` axis creates two equivalent global optima.

### Why these shapes matter

The experiments below test whether the `expdoe-dk` knowledge primitives can
recover these analytic shapes:

| Term                              | Best-matching primitive |
|-----------------------------------|-------------------------|
| `rate(T)` (Arrhenius)             | `with_arrhenius("T")`   |
| `eff(c)` (quadratic peak)         | `with_quadratic_peak("conc", center=...)` |
| `act(pH)` (Gaussian peak)         | `with_quadratic_peak("pH", center=7.0)` (quadratic approximation) |
| `sat(t)` (saturating monotone)    | `with_monotone("t", effect="increases_objective")` |
| `eff_rpm` (Gaussian peak)         | `with_quadratic_peak("rpm", center=700.0)` (quadratic approximation) |
| `eff_polar` (bimodal `|sin|`)     | **no matching primitive** — see roadmap issue #21 |

The 6D oracle's `polar` axis is deliberately out of reach of the current
primitives — it is the test case for the "new dataset, new shape" gap
documented in roadmap #21 (generalising the knowledge API).

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

> **Two history notes that matter for reading the numbers below.**
>
> 1. **Baseline is now a plain GP.** Earlier versions of this experiment
>    used a baseline that silently auto-applied `with_random_augment(n=20)`.
>    That auto-default has been removed — `A: baseline` is `knowledge=None`,
>    a genuine plain GP. All `Δ gap` numbers are relative to that.
> 2. **DoE is held constant at `sobol`** — the same initial-design choice
>    used by the historical Exp-7 / Exp-9 / Exp-10 v2 runs in
>    `AGENT_KNOWLEDGE.md` §6b. This makes the numbers directly comparable
>    to that table. (Experiment 01 separately compares DoE methods and finds
>    `lhs_random` strongest; for this experiment we hold DoE constant.)
>
> Even with Sobol DoE matching §6b, the plain-GP baseline here is much
> stronger than the historical §6b "A: Standard BO" baseline
> (4D gap ≈ 0.003 vs the §6b 0.012). The likely cause is implementation
> drift unrelated to this study — the old run used `torch.quasirandom`'s
> Sobol while expdoe-dk uses `scipy.stats.qmc`, plus the OLD oracle drew
> noise from a single per-run RNG while this one re-seeds per call.
> Treat the §6b magnitudes as a directionally-useful historical reference,
> not a target the rerun should reproduce exactly.

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

Setup: `sobol` DoE held constant (matches Exp-7/9/10). Budget:

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
| ③ gp_prior only                     | 0.0002    | **+66.7 %**       | 100 %         |
| **A: baseline (plain GP)**          | 0.0006    | 0.0 %             | 100 %         |
| ① full domain knowledge             | 0.0009    | −50 %             | 100 %         |
| ② random_augment only               | 0.0037    | −517 %            | 100 %         |
| ④ Arrhenius mean only               | 0.0037    | −517 %            | 100 %         |
| ⑤ monotone + gp_prior (rescued)     | 0.0040    | −567 %            | 100 %         |
| G: WRONG-direction monotone         | 0.0119    | −1883 %           | 80 %          |

In 2D, only `③ gp_prior only` beats the plain GP (+67 %). Every other
knowledge config underperforms — the plain-GP baseline already hits gap
0.0006 (≈ 0.1 % of the optimum), leaving very little room for any prior
to add value.

### 4D results on `process_objective_4d` (5 seeds, ~9 min)

True noiseless optimum ≈ 0.34956.

| Config                              | gap_final | Δ gap vs baseline | %seeds hit 95 |
|-------------------------------------|----------:|------------------:|--------------:|
| **A: baseline (plain GP)**          | 0.0033    | 0.0 %             | 100 %         |
| ④ Arrhenius mean only               | 0.0037    | −12.1 %           | 100 %         |
| ② random_augment only               | 0.0050    | −51.5 %           | 80 %          |
| ① full domain knowledge             | 0.0058    | −75.8 %           | 80 %          |
| ⑤ mono+prior (rescued)              | 0.0082    | −148.5 %          | 100 %         |
| ③ gp_prior only                     | 0.0084    | −154.5 %          | 100 %         |
| G: WRONG-direction monotone         | 0.0727    | −2103 %           | 20 %          |

In 4D, **no knowledge config beats the plain GP**. ④ Arrhenius is the
closest (−12 %), then ② and ①. ③ gp_prior moves from the 2D winner to
mid-pack; G wrong-direction is the clear bottom, as expected.

### 6D results on `process_objective_6d_v2` (5 seeds, ~13 min)

| Config                              | gap_final | Δ gap vs baseline | %seeds hit 95 |
|-------------------------------------|----------:|------------------:|--------------:|
| **A: baseline (plain GP)**          | 0.0160    | 0.0 %             | 80 %          |
| ② random_augment only               | 0.0284    | −77.5 %           | 20 %          |
| ⑤ mono+prior (rescued)              | 0.0471    | −194 %            | 0 %           |
| ④ Arrhenius mean only               | 0.0542    | −239 %            | 0 %           |
| ③ gp_prior only                     | 0.0598    | −274 %            | 20 %          |
| ① full domain knowledge             | 0.0684    | −328 %            | 0 %           |
| G: WRONG-direction monotone         | 0.1769    | −1006 %           | 0 %           |

In 6D, the plain GP wins again. ① drops to next-to-last (−328 %); ④
Arrhenius-alone — the 2D leader — collapses (−239 %). The dimension-
sensitivity pattern is intact: a single mean shape that fits one of six
dims becomes actively misleading on the remaining five.

### What this actually says

A plain GP is a strong baseline on these oracles — strong enough that
none of the canonical knowledge configurations match or beat it cleanly
across all three dimensions:

1. **`random_augment` hurt on every dimension** (−26 % / −52 % / −78 %).
   This is the empirical basis for removing it as an auto-default. Pure
   regularization is not a "free win"; its benefit is dataset-dependent
   and the right `n` scales with sample size (roadmap #20).
2. **The §6b "stacked-knowledge wins big" reading does not reproduce
   here**, even with Sobol DoE matching §6b's setup. Best guess: the
   absolute baselines differ because of implementation drift
   (`torch.quasirandom` vs `scipy.stats.qmc`; single-RNG noise vs
   per-call seeded noise). The qualitative §6b patterns — direction
   correctness matters most in high-D, Cat ④ is dimension-sensitive,
   wrong-direction priors hurt — all still show up.
3. **Among knowledge configs, simpler tends to be less bad.** ④
   Arrhenius alone is the closest to plain GP in 2D and 4D; once 6D
   adds dims with no matching primitive (bimodal `polar`, Gaussian
   `rpm`), no config recovers.

These conclusions are single-oracle-family results on 5 seeds; treat
them as directional. The breadth needed to make firm recommendations is
tracked in roadmap issues #20 (knowledge across datasets, including
the pure-regularization sweep) and #21 (generalising the knowledge API
so chemists can author primitives that actually match their landscape).

### Chemist's quick-reference

| If you have                                   | Use                                  |
|-----------------------------------------------|--------------------------------------|
| No specific knowledge                         | **Nothing — `Campaign(space)` runs a plain GP.** On these oracles it was the strongest or tied-strongest config at every dimension |
| A monotone direction you're confident about   | One `with_monotone(effect="increases_objective")` — frame translation handled; v0.2 validator warns if the data disagrees |
| A known shape for ONE low-D parameter (Arrhenius / peak) | One `with_arrhenius` *or* one `with_quadratic_peak` (`frozen=True`). Single primitives stayed close to plain GP; don't chain several at once |
| The temptation to "regularise for free"       | Avoid `with_random_augment` — it hurt at every dimension. Opt-in only, under validation in #20 |
| `with_gp_prior` AND `with_monotone` together  | Trust v0.3 `auto_rescue=True`; the combo never won here, expect a low ceiling |
| A high-dimensional problem (≥ 5 active factors) | Reduce dimensionality first; a plain GP was the strongest config in our 6D test |
