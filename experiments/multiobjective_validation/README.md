# Multiobjective Acceptance

**Approved protocol v2 pending validation; 0.6.0 release remains blocked.**
Protocol `fixed-catalog-monotone`, version `2.0`, reports four methods:
`random`, `plain_gp`, legacy `knowledge_gp`, and `knowledge_gp_v2`.
Only the latter adds `monotone@2.0`, increasing on gain, confidence .8,
three pairs, epsilon .05 and softness 1. Initial design, seeds, catalog,
reference, objective functions, geometry, budget and 1.10 tolerances are unchanged.
Release gates require Plain <= 1.10 * Random, V2 <= 1.10 * Random and
V2 <= 1.10 * Plain (each denominator floored at 1e-12). Legacy comparisons
remain explicitly separate under `legacy_v1`; they do not become v2 successes.
Every method still requires exact replay/restart, feasibility 1, duplicates 0,
and eight q=2 batches. Declared noiseless mode is unchanged: ordinary GPyTorch
fixed-noise likelihoods clamp zero declared variance to 1e-6 internally.
This protocol does not claim exact zero fitted noise.

## Preserved Historical V1 Failure

**2026-11-24: original quality gate FAILED; release blocked at that revision.** This is a named
sequential experiment, not general multiobjective cross-validation. Package and
engine versions remain 0.5.1. No successful release acceptance is claimed.

## Frozen Protocol

Catalog: `x=i/100`, i=0..100. Maximize `[x, 1-x]`, hard parameter limit `x<=0.8`.
The 81 feasible recipes have exact raw-utility HV **0.476**, reference `[0,0]`.
The fixture test proves this to absolute tolerance 1e-12 before recommendations.

Seeds are `(2,5,11,23,47)`. Each seed samples six initial recipes without
replacement from the hard-feasible catalog, using only x and that seed. All three
methods share those measurements. Each receives eight q=2 batches, for 22 actual
measurements. The baseline samples remaining feasible catalog recipes uniformly
without replacement; Plain GP uses no knowledge; Knowledge GP uses one increasing
monotone pattern scoped only to gain (`epsilon="auto"`, `delta_norm=.5`,
`n_pairs_per_dim=3`, confidence=.8). Both GP methods explicitly use noiseless
mode and route to qLogEHVI. These parameters were fixed before any run and were
not tuned after seeing the failures.

Geometry is resolved once per seed from the initial six measured rows. Both GPs
must report exactly that geometry. Random, both GPs and the oracle reference set
are scored under the same location, scale and reference. Oracle values never
select initial points or fit geometry. Raw and normalized HV are recorded; raw
regret is `oracle_hv_raw-observed_hv_raw`. Normalized regret is computed from the
transformed measured and oracle arrays, not by subtracting raw 0.476.

Each trajectory is repeated from scratch, then resumed from a JSON midpoint
checkpoint. Both complete measurement frames and every recommended batch must
match exactly. Runtime includes replay and restart verification; the 22-row
budget counts the primary trajectory only, not duplicated verification work.

## Results

Source commit `3aae0f583d9a8dfa20cd09e5e2d6f8915fc8d4c8`, clean tracked/untracked
state at both standalone matrix starts. Initial Task10 base is
`22fc8ba40f7a2eeffad5eb64a272579105ea3108`. Current and lower environments produced
the same trajectories and metrics. Runtime: current **65.79s**, lower **75.31s**.

| Seed | Scale (each objective) | Random regret | Plain GP regret | Knowledge GP regret |
| ---: | ---: | ---: | ---: | ---: |
| 2 | .57 | .0643274854 | .0341643583 | .1735918744 |
| 5 | .62 | .0616545265 | .0309573361 | .1636316337 |
| 11 | .53 | .0857956568 | .0416518334 | .2328230687 |
| 23 | .51 | .2037677816 | .0445982314 | .4063821607 |
| 47 | .52 | .1142751479 | .0417899408 | .2740384615 |
| Mean | per-seed fixed | **.1059641197** | **.0386323400** | **.2500934398** |

| Seed | Random raw regret | Plain raw regret | Knowledge raw regret |
| ---: | ---: | ---: | ---: |
| 2 | .0209 | .0111 | .0564 |
| 5 | .0237 | .0119 | .0629 |
| 11 | .0241 | .0117 | .0654 |
| 23 | .0530 | .0116 | .1057 |
| 47 | .0309 | .0113 | .0741 |

The fixed quality rule is each GP mean regret <= 1.10 times
`max(random_mean_regret,1e-12)`, plus Knowledge <= 1.10 times
`max(plain_mean_regret,1e-12)`. Plain passes; Knowledge fails both comparisons.
The baseline threshold is .1165605316 and the Knowledge-vs-Plain threshold is
.0424955740. No seed was dropped or replaced, and no tolerance was relaxed.

All 15 rows per stack pass the functional checks: parameter feasible rate 1.0,
duplicate rate 0.0, eight full q=2 batches, exact reproducibility and restart
consistency. This fixture has no outcome safety constraint; model joint
feasibility is recorded as null, not confused with parameter feasibility.

Post-gate diagnosis suggests the existing monotone pseudo-response model is
overconfident under noiseless fitting: pseudo pairs are centered on a global
mean independently of x, and inherit zero known measurement variance. At seed 2,
first fit, the six pseudo physical responses differ from the synthetic truth by
up to .5947069905 despite the correct monotone direction and scope. This is
diagnostic evidence, not an approved core-model change or a universal explanation
of knowledge performance. Acceptance parameters and production models were not
modified to improve these numbers.

## Environments and Warnings

| Dependency | Current | Resolver-clean lower |
| --- | --- | --- |
| Python | 3.12.13 | 3.11.15 |
| Ax | 1.3.1 | 1.2.4 |
| BoTorch | 0.18.1 | 0.17.2 |
| GPyTorch | 1.15.2 | 1.15.2 |
| Torch | 2.12.1 | 2.2.0 |
| NumPy | 2.5.0 | 1.24.4 |
| SciPy | 1.18.0 | 1.12.0 |
| pandas | 3.0.3 | 2.0.3 |

Both `pip check` calls passed. Both matrices captured 400 small-noise
NumericalWarnings across all primary/replay/resume fits. Current additionally
captured one optional fused-qLogEHVI extension warning (ninja unavailable), with
BoTorch's own pure-Python acquisition implementation used. This is not a random
recommendation fallback. Warnings are recorded per row, not globally suppressed.

The legacy BoTorch 0.11.0/GPyTorch 1.11 overlay passes 170 scalar/acquisition
capability tests (15.40s, five small-noise warnings), including real computation
and simulated missing-API typed errors. It conflicts with Ax metadata and is
not a resolver-clean support claim. Ax 1.2.4 requires Python >=3.11, pending the
gated metadata correction. No speculative direct BoTorch floor increase was made.

Only local macOS 26.7.3 ARM64 was tested. Optional PyArrow dtypes, other operating
systems, GPU, Colab and textbook datasets remain unverified. Torch 2.2 lacks
uint64 tensors, so that one precise capability test skips there; current Torch
still exercises it. See [API and reporting limits](../../docs/multiobjective.md).

Full slow suite at the implementation revision: current **1971 passed, 1 failed,
0 skipped, 174 warnings, 344.49s**; lower **1970 passed, 1 failed, 1 skipped,
171 warnings, 471.70s**. The sole failure in both is this quality gate; the lower
skip is unavailable Torch uint64. Existing 0.5.1 wheel/sdist legal/dependency
tests passed within those suites, but no 0.6.0 artifact was built or installed.
The subsequently added example 05 regression passed separately on both stacks;
all three examples ran outside the source directory with installed editable
provenance and no PYTHONPATH. Final focused current tests: 196 passed, one slow
gate skipped by default, 27 warnings, 10.84s. This does not turn slow acceptance
green. Controller-owned offline HTML screenshots passed a representative
1280x900/390x844 layout check, not comprehensive accessibility certification.

## Reproduce

From the repository root, with an editable installation and working git in PATH:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TORCH_EXTENSIONS_DIR="$(mktemp -d)" \
  python -m experiments.multiobjective_validation.run --output-dir outputs/multiobjective-validation
```

The CLI returns nonzero if v2 quality fails. `run_matrix(*, seeds, output_dir)`
returns all four-method rows; `quality_summary(rows)` applies the approved v2
gates and reports legacy comparisons separately. JSONL
contains source SHA, dirty status/details, dependencies/platform, geometry,
reference, scale, seed, method, budget, batches, runtime, actual counts,
feasibility/duplicates, raw and normalized HV, regret, replay/restart results
and warnings. `summary.json` records aggregate gates. Default output is under
git-ignored `outputs/`; do not commit large generated traces.

From the inner package directory:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TORCH_EXTENSIONS_DIR="$(mktemp -d)" \
  python -m pytest -q --run-slow
```

The full slow suite exercises the approved v2 quality gate. Fast tests
prove oracle/geometry and actual 2/3/4/5/6-objective routing with named-column,
catalog, pending, noise and hard-constraint checks; they do not replace quality
acceptance. Logs and full JSONL evidence for this local run are in the ignored
Task10 SDD handoff directory; exact commands and results are in task-10-report.md.
