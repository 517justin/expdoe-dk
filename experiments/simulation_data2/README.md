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
