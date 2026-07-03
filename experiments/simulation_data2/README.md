# Simulation Data 2

`simulation_data2` is a lab-like constrained, discrete experiment suite for
checking whether `expdoe-dk` remains usable when an optimization problem has
realistic formulation and process restrictions.

Unlike the canonical and `simulation_data1`-style experiments, this dataset
emphasizes engineering validity:

- every factor is discrete;
- 6D includes linear formulation constraints;
- summaries report feasibility, grid validity, duplicate rows, and clean
  optimization metrics; raw CSV rows include per-run runtime as `secs`, and the
  console log prints elapsed time;
- Experiment 02 can choose its DoE method from Experiment 01 output.

## Problems

### 4D process-only

| Parameter | Bounds | Step |
| --- | ---: | ---: |
| `T` | 60-120 C | 1 |
| `time` | 10-180 min | 5 |
| `pH` | 4-10 | 1 |
| `catalyst` | 0.5-5.0 mol% | 0.5 |

Budget: `n_doe=6`, `n_iter=15`.

### 6D formulation + process

Adds:

| Parameter | Bounds | Step |
| --- | ---: | ---: |
| `solvent_A` | 20-80 vol% | 5 |
| `additive` | 0-10 mol% | 1 |

Constraints:

- `solvent_A + additive <= 85`
- `solvent_A >= 3 * additive + 20`
- `catalyst + additive <= 12`

Budget: `n_doe=10`, `n_iter=20`.

## Commands

```bash
python experiments/simulation_data2/01_doe_method_comparison.py --dim 4
python experiments/simulation_data2/01_doe_method_comparison.py --dim 6
python experiments/simulation_data2/02_knowledge_comparison.py --dim 4 --doe-method auto
python experiments/simulation_data2/02_knowledge_comparison.py --dim 6 --doe-method auto
```

For faster smoke runs:

```bash
python experiments/simulation_data2/01_doe_method_comparison.py --dim 4 --seeds 1
python experiments/simulation_data2/01_doe_method_comparison.py --dim 6 --seeds 1
python experiments/simulation_data2/02_knowledge_comparison.py --dim 4 --seeds 1 --doe-method auto
python experiments/simulation_data2/02_knowledge_comparison.py --dim 6 --seeds 1 --doe-method auto
```

## Interpreting Results

Use this dataset as a feasibility check, not as proof of a universal best
algorithm. A successful run should have:

- zero constraint violations;
- zero grid violations;
- duplicate suggestions recorded in `duplicate_rows`;
- all scripts completing without crashes;
- 6D visibly harder than 4D;
- wrong knowledge not consistently beating full or partial knowledge.

When baseline gap is near zero, relative gap improvement is reported as
`NaN` to avoid misleading percentages.

## Results Snapshot

The latest checked run used 3 seeds for each 4D/6D script. All rows were
feasible and on-grid, and the median duplicate count was zero.

### Experiment 01 - DoE Method Comparison

Knowledge is held fixed at a plain GP, so the only variable is the initial
DoE method.

#### 4D

| Method | clean_final | gap_final | clean@doe_end | clean@mid | trials_to_95 median | %seeds_hit_95 | Gap improvement vs `random_uniform` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sobol` | 0.4886 | 0.0249 | 0.2676 | 0.4096 | 19 | 66.7 | 69.4% |
| `halton` | 0.4846 | 0.0289 | 0.1987 | 0.3355 | 22 | 33.3 | 64.5% |
| `lhs_random` | 0.4743 | 0.0391 | 0.1677 | 0.3247 | 22 | 0.0 | 52.0% |
| `lhs_maximin` | 0.4534 | 0.0600 | 0.1650 | 0.1846 | 22 | 0.0 | 26.3% |
| `random_uniform` | 0.4320 | 0.0814 | 0.2351 | 0.3320 | 22 | 33.3 | 0.0% |
| `d_optimal` | 0.4160 | 0.0974 | 0.0208 | 0.2345 | 22 | 0.0 | -19.6% |

#### 6D

| Method | clean_final | gap_final | clean@doe_end | clean@mid | trials_to_95 median | %seeds_hit_95 | Gap improvement vs `random_uniform` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lhs_maximin` | 0.5217 | 0.0842 | 0.0559 | 0.3054 | 31 | 33.3 | 78.7% |
| `sobol` | 0.4949 | 0.1109 | 0.1072 | 0.2946 | 31 | 0.0 | 71.9% |
| `halton` | 0.4194 | 0.1865 | 0.1401 | 0.2082 | 31 | 0.0 | 52.8% |
| `lhs_random` | 0.3269 | 0.2790 | 0.0500 | 0.1392 | 31 | 0.0 | 29.4% |
| `random_uniform` | 0.2108 | 0.3951 | 0.0492 | 0.0952 | 31 | 33.3 | -0.0% |
| `d_optimal` | 0.1670 | 0.4388 | 0.0207 | 0.1286 | 31 | 0.0 | -11.1% |

Result: the preferred DoE method changes with dimensionality. `sobol` is the
best 4D choice in this run, while `lhs_maximin` is best in 6D. Experiment 02
uses these Experiment 01 summaries when `--doe-method auto` is selected.

### Experiment 02 - Knowledge Comparison

Experiment 02 uses the best available DoE method from Experiment 01 and
compares knowledge configurations.

#### 4D

| Config | clean_final | gap_final | clean@doe_end | clean@mid | trials_to_95 median | %seeds_hit_95 | Gap improvement vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D: process knowledge` | 0.5098 | 0.0036 | 0.2676 | 0.4811 | 18 | 66.7 | 85.5% |
| `C: gp_prior only` | 0.5062 | 0.0072 | 0.2676 | 0.4369 | 20 | 100.0 | 71.1% |
| `G: full mixed knowledge` | 0.4964 | 0.0170 | 0.2676 | 0.4644 | 21 | 66.7 | 31.6% |
| `F: wrong process knowledge` | 0.4921 | 0.0213 | 0.2676 | 0.3822 | 18 | 66.7 | 14.4% |
| `B: random_augment only` | 0.4920 | 0.0214 | 0.2676 | 0.4534 | 16 | 66.7 | 14.0% |
| `E: partial process knowledge` | 0.4909 | 0.0225 | 0.2676 | 0.4437 | 18 | 100.0 | 9.5% |
| `A: baseline (plain GP)` | 0.4886 | 0.0249 | 0.2676 | 0.4096 | 19 | 66.7 | -0.1% |

#### 6D

| Config | clean_final | gap_final | clean@doe_end | clean@mid | trials_to_95 median | %seeds_hit_95 | Gap improvement vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `G: full mixed knowledge` | 0.5982 | 0.0076 | 0.0559 | 0.4922 | 23 | 100.0 | 91.0% |
| `A: baseline (plain GP)` | 0.5217 | 0.0842 | 0.0559 | 0.3054 | 31 | 33.3 | -0.0% |
| `B: random_augment only` | 0.3923 | 0.2136 | 0.0559 | 0.1664 | 31 | 0.0 | -153.7% |
| `D: process knowledge` | 0.3688 | 0.2370 | 0.0559 | 0.0764 | 31 | 0.0 | -181.5% |
| `F: wrong process knowledge` | 0.3566 | 0.2493 | 0.0559 | 0.1423 | 31 | 0.0 | -196.1% |
| `E: partial process knowledge` | 0.2652 | 0.3407 | 0.0559 | 0.1200 | 31 | 33.3 | -304.7% |
| `C: gp_prior only` | 0.2626 | 0.3432 | 0.0559 | 0.0806 | 31 | 0.0 | -307.6% |

Result: process knowledge is already useful in 4D, but the full formulation
plus process knowledge is most valuable in 6D. The 6D case is harder than 4D,
yet the full mixed knowledge configuration reached the 95% target in all
tested seeds.

## Takeaways

- The tool remains usable under realistic discrete grids and linear
  formulation constraints.
- DoE method choice matters: no single method is best across 4D and 6D.
- Knowledge injection matters more in the harder 6D mixed formulation/process
  setting.
- The feasibility checks passed cleanly: zero constraint violations, zero
  grid violations, and zero median duplicate rows.
