# Simulation Data 3 Design

## Purpose

`simulation_data3` will test `expdoe-dk` on dirty experimental execution
data rather than a clean oracle-only benchmark.

The scenario is polymer/resin curing. It models what a lab might actually
record after running specimens: discrete process settings, metadata,
batch/day/operator/material-lot effects, failed runs, missing results, low
measurement values, and replicate measurements.

The goal is to answer three practical questions:

1. Which initial DoE method is robust when the response data includes
   failed/missing/limited measurements?
2. Does one fixed material-process knowledge configuration improve over a
   plain GP?
3. How much do dirty-data handling strategies affect BO results?

`simulation_data4` and `simulation_data5` are reserved for future coating
and battery/electrode scenarios. This spec covers only `simulation_data3`.

## Relationship To Existing Simulation Data

- `simulation_data1` is a canonical clean-oracle benchmark.
- `simulation_data2` adds realistic discrete grids and linear formulation
  constraints.
- `simulation_data3` keeps BO single-objective, but shifts the hard part to
  experimental data quality and execution metadata.

Unlike `simulation_data1` and `simulation_data2`, the generated raw data
will include rows that are not directly usable by BO until a processing
strategy converts or filters them.

## Optimizable Factors

The BO search space has five discrete controllable factors:

| Factor | Bounds | Step | Role |
| --- | ---: | ---: | --- |
| `curing_temp` | 60-180 C | 10 | Curing temperature; too low under-cures, too high degrades. |
| `curing_time` | 30-240 min | 15 | Cure duration; too short under-cures, too long embrittles. |
| `mix_ratio` | 80-120 phr | 5 | Resin/hardener stoichiometry; best near 100 phr. |
| `humidity` | 20-80 %RH | 10 | Environmental moisture; higher humidity generally hurts. |
| `filler_loading` | 0-40 wt% | 5 | Reinforcing filler; moderate loading helps, too much defects. |

These are the only columns passed to `expdoe_dk.Space`.

## Metadata Columns

The raw experimental table will also include non-optimizable metadata:

| Column | Purpose |
| --- | --- |
| `batch_id` | Captures batch-to-batch process shift. |
| `day` | Captures day-level drift. |
| `operator` | Captures operator-specific offset/noise. |
| `material_lot` | Captures raw-material lot variation. |
| `replicate_id` | Identifies repeated measurements at the same condition. |

Metadata is used for reporting and dirty-data simulation. It is not passed
as BO input.

## Response Columns

The primary BO objective remains single-objective:

- `tensile_strength`: maximize, reported in MPa. The clean response should
  roughly span 20-100 MPa before execution noise and data-quality effects.

The raw data also records engineering outcomes:

- `modulus`
- `defect_rate`
- `failure_mode`
- `measurement_status`
- `is_valid`

`measurement_status` values should include:

- `valid`
- `failed`
- `missing`
- `below_lod`

`failure_mode` should be empty or `none` for valid rows and contain labels
such as `under_cured`, `over_cured`, `voids`, `brittle`, or `instrument_error`
for problematic rows.

## Dirty Data Severity

Use medium severity:

- About 10-15% of attempted rows should be failed, missing, or below the
  measurement floor.
- Low-quality process regions should fail more often than good regions.
- Higher humidity, extreme mix ratio, excessive filler, under-curing, and
  over-curing should increase the chance of failed or defective specimens.
- Batch/day/operator/material-lot offsets should affect observed values.
- Some conditions should have replicate rows.

The simulator should remain deterministic for a given seed.

## Underlying Clean Response

The implementation may still use an internal deterministic clean response
function to score `clean_final` and `gap_final`, but the public experiment
should treat observed data as dirty experimental records.

Expected response shape:

- `curing_temp` has a broad peak near 130 C.
- `curing_time` has a broad peak near 150 min.
- `mix_ratio` peaks near 100 phr.
- `humidity` decreases objective as it increases.
- `filler_loading` peaks near 20 wt%.
- Interactions should make very high humidity plus high filler especially
  defective.

This clean response is used for benchmark reporting only, not exposed as
clean observations during BO.

## Material Knowledge Configuration

Experiment 02 compares only two knowledge configurations:

1. `plain_gp`
   - No knowledge injection.
2. `with_material_knowledge`
   - `curing_temp`: quadratic peak centered near 130 C.
   - `curing_time`: quadratic peak centered near 150 min.
   - `mix_ratio`: quadratic peak centered near 100 phr.
   - `humidity`: monotone decreases objective.
   - `filler_loading`: quadratic peak centered near 20 wt%.
   - Add `with_gp_prior(lengthscale="medium")`.

Do not compare multiple knowledge recipes in `simulation_data3`. The point
is plain GP versus one reasonable material-process knowledge package.

## Dirty Data Strategies

The data processing module should expose four strategies:

### `valid_only`

Use only rows with `is_valid == True` and numeric `tensile_strength`.
Failed, missing, and below-LOD rows remain in raw reports but are not used
for BO training.

### `penalty_impute`

Convert failed/missing/below-LOD rows into a low score so BO learns to avoid
bad process regions. Use a deterministic penalty of `0.0` MPa for
`tensile_strength` in processed BO training data, while preserving the raw
status columns for reporting.

### `lod_floor`

For `below_lod` rows, use a measurement floor value of `10.0` MPa. Failed
and missing rows are still excluded by this strategy.

### `replicate_mean`

Aggregate rows with identical optimizable factors. Use the mean of valid
replicate strengths. Report replicate counts and failure counts. If all
replicates at a condition are invalid, exclude that condition from processed
BO training data and count it in the data-quality summary.

## Experiment Scripts

Place files under `experiments/simulation_data3/`.

### `_data.py`

Defines:

- the 5D `Space`
- raw data generation helpers
- deterministic clean response helpers for benchmark reporting
- execution simulator that returns dirty experimental records
- problem spec factory

The module should make it obvious which columns are optimizable factors and
which columns are metadata/response/reporting fields.

### `_processing.py`

Defines:

- dirty-data strategy implementations
- validation that processed BO input rows have only optimizable factors plus
  a numeric objective
- summary helpers for data quality metrics

### `01_doe_method_comparison.py`

Question: with `plain_gp` and the default `valid_only` processing strategy,
which DoE method is most robust?

Compare the same DoE method list used by `simulation_data2`:

- `lhs_maximin`
- `lhs_random`
- `sobol`
- `halton`
- `random_uniform`
- `d_optimal`

### `02_gp_vs_knowledge_comparison.py`

Question: under dirty experimental data, does fixed material knowledge
improve over a plain GP?

Compare:

- `plain_gp`
- `with_material_knowledge`

Use a default processing strategy of `replicate_mean`, because Experiment 02
should represent a realistic material workflow where repeated specimens are
summarized before modeling.

### `03_data_strategy_comparison.py`

Question: how much does dirty-data handling change BO results?

Compare:

- `valid_only`
- `penalty_impute`
- `lod_floor`
- `replicate_mean`

Use the best available DoE method from Experiment 01 when possible, with a
safe fallback to `sobol`. Use the fixed material knowledge package for all
dirty-data strategies so the only experimental variable is data processing.

### `README.md`

Document:

- factor ranges and metadata columns
- response columns and status meanings
- dirty-data generation behavior
- experiment commands
- all summary tables after verification runs
- takeaways about feasibility, knowledge value, and strategy sensitivity

## Metrics

Keep the BO headline metrics consistent with prior datasets:

- `clean_final`
- `gap_final`
- `clean@doe_end`
- `clean@mid`
- `trials_to_95pct`
- `%seeds_hit_95`

Add data quality metrics:

- `n_valid_rows`
- `n_failed_rows`
- `n_missing_rows`
- `n_lod_rows`
- `n_replicates`
- `failure_rate`
- `missing_rate`
- `lod_rate`
- `duplicate_conditions`
- `batch_count`
- `operator_count`

Add engineering report metrics:

- `modulus_final`
- `defect_rate_final`
- `failure_mode_final`
- `is_engineering_feasible_final`

## Success Criteria

Implementation is successful if:

- all scripts complete without crashing on failed/missing/below-LOD rows
- `valid_only` filters bad rows correctly
- `penalty_impute` turns bad rows into a deterministic low score
- `lod_floor` handles below-LOD rows without dropping useful information
- `replicate_mean` aggregates repeated conditions and reports replicate
  counts
- Experiment 02 shows whether `with_material_knowledge` is more stable than
  `plain_gp`
- Experiment 03 shows whether dirty-data strategy materially changes BO
  results
- README clearly states this is dirty experimental execution data, not a
  clean oracle benchmark

## Out Of Scope

- No `simulation_data4` or `simulation_data5` implementation in this cycle.
- No multi-objective BO core changes.
- No new public `Campaign` API.
- No use of metadata columns as BO factors.
- No attempt to model every possible experimental data problem.

## Recommended Verification

The implementation plan should include:

- focused unit tests for data generation and processing strategies
- smoke tests for all three experiment scripts with `--seeds 1`
- a final verification run with at least 3 seeds for Experiment 01/02/03
- `pytest -q --run-slow` from the package root
- `git diff --check`
