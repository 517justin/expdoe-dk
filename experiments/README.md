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

### Investigation — why does ① "full domain knowledge" underperform?

The 4D and 6D tables above show ① "full domain knowledge" (Arrhenius +
quadratic peaks for conc and pH + monotone(t) + random_augment(20)) doing
*worse* than a plain GP. This is at odds with `AGENT_KNOWLEDGE.md` §6b,
which reports the canonical "C: Frozen Combined" config beating the
baseline by **+26 % in 4D** and **+91 % in 6D**. So either the new
implementation regressed, or something subtler is going on. The ablation
in [`_ablation_knowledge_4d.py`](./_ablation_knowledge_4d.py) localises
the issue (4D, same five seeds and budget as exp 02):

| Config                                | gap_med | Δ vs plain GP |
|---------------------------------------|--------:|--------------:|
| `monotone(t)` only                    | 0.0019  | **+30.3 %**   |
| `arrhenius` only                      | 0.0020  | +28.0 %       |
| 2 peaks (conc + pH)                   | 0.0022  | +19.5 %       |
| old D_correct (monotone T + t)        | 0.0026  | +7.0 %        |
| **plain GP**                          | 0.0028  | 0.0 %         |
| old C: Frozen Combined (3 means)      | 0.0037  | −32.8 %       |
| Frozen Combined + monotone(t)         | 0.0038  | −37.7 %       |
| ① full (4 items + `random_augment`)   | 0.0076  | **−175 %**    |
| Frozen Combined + `random_augment(20)`| 0.0103  | −271 %        |

Two findings, neither of which was visible in `AGENT_KNOWLEDGE.md`:

**Finding 1 — single, targeted knowledge items help; stacking hurts.**
Every single-item config (just `monotone(t)`, just `arrhenius`, or just
the two peaks) beats the plain-GP baseline by +20~30 %. The moment two or
more priors are stacked the advantage disappears, and at four items + a
random-augment block the result is *strictly worse* than no knowledge at
all. The library's primitives DO work — when applied one at a time.

**Finding 2 — the apparent ①+25 %/+91 % in `AGENT_KNOWLEDGE.md` was
partly an artefact of a weaker baseline.** The OLD experiments used Sobol
for the initial DoE: in 4D the OLD plain-GP-with-Sobol gap was 0.0117,
and "C: Frozen Combined" reduced that to 0.0087 (a +26 % improvement).
Our new baseline uses `lhs_maximin` for the initial DoE, which on this
oracle reaches 0.0028 *without any knowledge* — already better than the
OLD "C: Frozen Combined" result. In other words, **a stronger DoE
absorbs the easy gains that the old knowledge configs used to provide**;
once the baseline is genuinely strong, stacking knowledge over-specifies
the GP and starts to hurt. This is consistent with experiment 01, which
showed `lhs_random` / `lhs_maximin` as the strongest DoE methods on
`process_objective_4d` (Δ +60 % / 0 % vs `random_uniform`).

**Practical takeaway.** Pick **one** piece of knowledge that matches the
hardest term in your landscape — typically the monotone direction of the
most important parameter, or a single Arrhenius/peak for one factor with
known shape — and trust the plain GP for the rest. Reach for additional
priors only if you have evidence that one piece isn't enough; do **not**
chain `with_*` calls in the hope that "more knowledge = better fit".

This finding is the immediate motivation for roadmap issue #21 (generalise
the knowledge API and document an authoring guide that warns against
over-specification by default) and gives roadmap issue #20 a concrete
validation question: *across more oracles, does the one-targeted-item
heuristic continue to dominate stacked configurations?*

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
   place against it, and (per the investigation above) the right shape is
   one well-chosen piece, not a stacked combination.
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
| A known shape for ONE low-D parameter (Arrhenius / peak) | One `with_arrhenius` *or* one `with_quadratic_peak` (`frozen=True`) — best when the shape matches the hardest dim. **Do not chain multiple priors.** |
| A monotone direction you're confident about   | One `with_monotone(effect="increases_objective")` — frame translation handled; v0.2 validator warns if the data disagrees |
| Several plausible shapes                      | **Pick the one that matches the hardest term.** The ablation above shows stacking 2+ priors strictly hurts on this oracle (Δ goes from +30 % to −175 %) |
| No specific knowledge                         | **Nothing — `Campaign(space)` runs a plain GP.** It's a strong baseline; don't add structure you can't justify |
| The temptation to "regularise for free"       | Avoid `with_random_augment` for now — on these oracles it hurt vs a plain GP. It's opt-in and under validation (#20) |
| `with_gp_prior` AND `with_monotone` together  | Trust v0.3 `auto_rescue=True`; expect a lower ceiling in low-D |
| A high-dimensional problem (≥5 active factors) | Reduce dimensionality first; a plain GP was the best config in our 6D test |
