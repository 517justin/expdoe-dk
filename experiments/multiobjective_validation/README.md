# Multiobjective Acceptance

**0.6.0 passed the approved protocol and complete current/lower slow suites.**
Final production source is clean `4761df9824ffaace1a84a4a412aefe8bddcfbd81`;
fresh installed-wheel, metadata, examples and artifact checks also passed.
The final documentation-only delta is composed with that source's evidence;
the full suites and artifacts are not attributed to a later prose revision.
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

### Final-Review Fix Verification

The single consolidated fix wave binds multiobjective v2 empirical evidence to
its actual declaration/Space and preserves exact physical values and missing
markers across batches. It changes neither scalar concatenation nor the
numerical, search, noise, version or quality protocol. All 54 new regressions
pass on both stacks. Complete suites ran once each at clean `4761df9`, with
no duplicate standalone matrix:

| Stack | Passed | Skipped | Warnings | Full suite | Matrix including replay/restart |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 2646 | 0 | 188 | 1559.31s | 1046.97s |
| Lower | 2645 | 1 | 202 | 1997.35s | 1242.33s |

All nine slow cases ran. The sole lower skip remains Torch 2.2's missing uint64
tensor capability. Both 20-row matrices retain the four-method/five-seed results
below: all three release gates pass, v2 ties Plain, and legacy v1 still fails
separately. Controller-owned independent physical-HV/normalization/provenance
audits pass both matrices, maximum discrepancy 2.64e-16. No general superiority
claim follows. Timings overlap and are not isolated performance benchmarks.

Current warnings comprise 36 jitter, 47 fixed-noise clamps, 102 negative-variance,
two Torch deprecations and one extreme-domain overflow. Lower has 50 jitter,
47 fixed-noise clamps, 103 negative-variance, one NumPy corruption-test deprecation
and one overflow. Each matrix separately captures 600 fixed-noise-clamp warnings;
those are not included in the pytest warning totals. Passing is not warning-free.

Retained evidence under `.superpowers/sdd/2026-11-24-monotone-v2-implementation/`:
`final-fix-{current,floor}-4761df9.log`,
`final-fix-{current,floor}-4761df9/test_full_fixed_seed_matrix0/{matrix.jsonl,summary.json}`,
and `final-fix-{current,floor}-4761df9-independent-audit.json`.
Commands, RED/GREEN chronology, failures and decisions are in `final-fix-report.md`.

Fresh wheel/sdist legal, dependency and source checks pass. All 63 package files
match committed source in both artifacts and the noneditable installation.
Outside-source verification with PYTHONPATH unset passed examples 03/04/05,
historical checkpoint/Space reads and R1/R2 replacement, exact-value and
missing-marker smoke checks. Current/lower editable examples and all three pip
checks pass. The BoTorch-0.11 overlay passes 170 capability tests, not a supported
Ax dependency resolution.

Artifacts in `final-fix-artifacts-4761df9/`:
- `expdoe_dk-0.6.0-py3-none-any.whl`, SHA256
  `b68f2bbad5835f574dc802a971806fea12d0ad66f09a6ce6345cd1893da363fc`.
- `expdoe_dk-0.6.0.tar.gz`, SHA256
  `5a1947a53270e92bd39108807c54a597606e782bd3b857d93c7073b5c2a0d4eb`.

**Parked N1 compatibility limitation:** unpublished intermediate 0.6.0
schema-2.2 checkpoints from `93a920a`, `412fd2d` or `96e9013` that combine integer
Yvar with missing failed-row variance may fail frozen geometry-digest checking
under the final representation. These development candidates are not promised
universal internal-build continuity. Do not edit checkpoint files or hashes to
force restoration. Preserve originals and, if necessary, recreate a Campaign
from original measurements and re-establish geometry; this is not same-campaign
resume. Supported historical 0.4/0.5.x routes are unchanged, and final
same-revision restart passes. N1 is explicitly excluded/parked, not fixed; see
the [checkpoint compatibility guide](../../docs/multiobjective.md#persistence-and-environment).

The later root/documentation-only update is not packaged. Production, tests,
package README and metadata remain byte-identical to the fully tested and
packaged revision; final provenance evidence composes these results rather than
claiming a new full-suite run on prose. Prior failures below remain preserved.

### Approved V2 Results

Pre-release source `226379f3bef0e2ab1bbf41468d7fad37a4f0944e`, clean at both matrix starts.
Each complete `pytest --run-slow` suite includes the fixed 20-row matrix; no
duplicate standalone matrix was run on the same source. Both stacks passed
all three quality gates and every row's replay/restart, feasibility 1.0,
duplicate rate 0.0, 22 measured rows and eight q=2 batches. V2 ties Plain on
this fixture; no superiority or general optimizer-ranking claim is made.

| Seed | Scale (each objective) | Random regret | Plain regret | Legacy v1 regret | V2 regret |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | .57 | .0643274854 | .0341643583 | .1735918744 | .0341643583 |
| 5 | .62 | .0616545265 | .0309573361 | .1636316337 | .0309573361 |
| 11 | .53 | .0857956568 | .0416518334 | .2328230687 | .0416518334 |
| 23 | .51 | .2037677816 | .0445982314 | .4063821607 | .0445982314 |
| 47 | .52 | .1142751479 | .0417899408 | .2740384615 | .0417899408 |
| Mean | per-seed fixed | **.1059641197** | **.0386323400** | **.2500934398** | **.0386323400** |

These are normalized regrets, not raw HV differences. V2 raw regrets are
.0111, .0119, .0117, .0116 and .0113 in seed order, equal to Plain. Reference,
geometry, seeds, knowledge parameters, declared noise and tolerances did not
change. The same-frozen-GP source-construction repair resolves numerical
cancellation without new pseudo data, noise, rank cutoffs or modified fitting.

Current full suite: **2582 passed, 0 skipped, 188 warnings, 1497.75s**.
Lower full suite: **2581 passed, 1 skipped, 200 warnings, 1934.10s**; sole skip is
Torch 2.2's unavailable uint64 dtype. Both include all nine slow tests and
wheel/sdist legal/dependency checks. Matrix runtimes including replay/restart
were 1017.96s/current and 1204.62s/lower under overlapping suite load, not an
isolated performance benchmark. Per-row warnings remain in JSONL; full-suite
warning summaries remain in the retained logs. Each matrix captured 600 expected
fixed-noise-clamp warnings across primary/replay/restart fits; these are separate
from the pytest warning-summary counts. Both use mpmath 1.3.0 in addition
to the dependency versions listed below. The legacy BoTorch overlay passed 170
tests; examples 03/04/05 passed outside source on both editable environments.

Retained pre-release artifacts are under
`.superpowers/sdd/2026-11-24-monotone-v2-implementation/task-6-pre-{current,floor}-226379f/`:
`test_full_fixed_seed_matrix0/{matrix.jsonl,summary.json}`. Controller-owned
stdlib-only audits independently recomputed physical rectangle-union HV and
normalization; both passed, maximum metric discrepancy 2.64e-16. They do not
replace full suites or fresh installed-wheel verification.

V2 is opt-in and can be slow. Bounded source-adapter forward/backward observations
reached 131-133s excluding setup, not full-recommendation throughput; see
[numerical scope and cost](../../docs/multiobjective.md#numerical-scope-and-cost).

### Prior 0.6.0 Candidate Verification

Both complete post-bump suites ran on clean source
`412fd2d4ff89ec836f6889bdfebe489dff532f1a`, with installed metadata 0.6.0:

| Stack | Passed | Skipped | Warnings | Full suite | Matrix including replay/restart |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 2592 | 0 | 188 | 1591.86s | 1071.71s |
| Lower | 2591 | 1 | 200 | 2016.40s | 1258.85s |

All nine slow cases ran, including packaging; the only lower skip is unavailable
Torch uint64. Timings overlap across stacks and are not isolated benchmarks.
Both 20-row matrices reproduce the table above exactly, with the same locked
parameters, all three release gates passing and legacy v1 failing separately.
Controller-owned independent audits recomputed physical HV and normalized
regrets, passing with maximum discrepancy 2.64e-16. Each matrix separately
captured 600 fixed-noise-clamp warnings. Suite warnings remain visible: jitter,
fixed-noise clamps, negative-variance warnings, existing overflow and dependency
deprecations. A passing suite does not mean warning-free numerical execution.

Prior candidate logs and retained matrices are under
`.superpowers/sdd/2026-11-24-monotone-v2-implementation/`:
`task-6-final-{current,floor}-412fd2d.log` and
`task-6-final-{current,floor}-412fd2d/test_full_fixed_seed_matrix0/{matrix.jsonl,summary.json}`.
Audit files are `task-6-final-{current,floor}-412fd2d-independent-audit.json`.

Fresh wheel and sdist built from that same clean source passed legal/dependency
and source-content checks. Offline noneditable wheel installation was checked
outside source with PYTHONPATH removed: actual site-packages import, matching
package/metadata/engine 0.6.0, exact artifact hash, historical checkpoint/Space
reads, rejected unreviewed writers, and examples 03/04/05 all passed. The same
examples passed on both refreshed editables; all three pip checks passed. The
legacy API overlay passed 170 capability tests, not a supported Ax resolution.

Artifacts in `task-6-artifacts-412fd2d/`:
- `expdoe_dk-0.6.0-py3-none-any.whl`, SHA256
  `c984742fd3258046db67cad0dd091bf52719ea51b60fea5302d5b6435e50a78a`.
- `expdoe_dk-0.6.0.tar.gz`, SHA256
  `871b994331251eaaf89d64ad1ac5781a901d9b5b92f41ee73d40a4676daac7b5`.

Review identified a masked dtype-regression test: its old schema-2.2 fixture
failed at the writer check instead of conversion. After full gates completed,
the test was corrected to validate the separate 0.6.0 fixture before corruption and assert
the intended conversion error; historical fixtures remain unchanged. The new
assertion first failed for all three X/Y/Yvar cases, then the entire checkpoint
test file passed on current and lower (72 each). This test-only correction and
the final prose delta do not change the production package or built wheel;
their focused verification is composed with, not called a rerun of, the full
412fd2d gates. No tag, push or publication was performed.

### Preserved Post-Bump Compatibility Failure

The earlier 0.6.0 candidate `93a920aef20db1529ddb557180250e9f0ef6d9de` passed
both matrices and their independent audits, but each full suite failed 40
historical 0.5.1 Space reads: current 2549 passed/40 failed/105 warnings;
lower 2548 passed/40 failed/1 skipped/122 warnings. Those are not full passes.
The controller-approved correction accepts exactly standalone Space writers
0.5.1 and 0.6.0, keeps 0.5.0 migration Campaign-owned, rejects unreviewed/future
versions, and leaves every historical fixture unchanged. Full corrected-source
results are above; the original `task-6-post-{current,floor}-v060` logs/matrices
and `task-6-artifacts-93a920a` archives remain preserved, not reused as final
artifact acceptance. No optimizer, quality, noise or numerical policy changed.

### Historical V2 Execution Hold

Both fixed matrices started from clean source
`80e7a1b5b2907cd5244d2dbe2760e4acd193212f`. Each completed seed 2's random,
plain and legacy methods with full replay/restart, then stopped in v2 with
`MODEL_FIT_FAILED`, reason `negative_covariance`, on gain. There is no complete
v2 quality summary or release acceptance at that revision. All protocol
parameters stayed locked; package and engine versions remained 0.5.1.

| Stack | V2 failing round (zero-based) | Measurements | Source covariance minimum eigenvalue | PSD tolerance |
| --- | ---: | ---: | ---: | ---: |
| Current | 2 | 10 | -1.7544614871e-11 | 1.4210707455e-19 |
| Lower | 1 | 8 | -1.4331936525e-11 | 1.1369598784e-19 |

The failure occurs while validating the base GP query covariance, before the
directional update. The ill-conditioned measured-data fit subtracts large prior
and conditioning terms to obtain a small latent covariance. A diagnostic-only
80-digit recomputation of the frozen Matern-2.5 kernel, fitted hyperparameters,
physical features and fitted 1e-6 noise gives positive minimum eigenvalues:
5.0287595946e-13 (current), 6.0729325677e-12 (lower). On the current stack,
double-precision direct solve and Cholesky subtraction both still yield negative
eigenvalues; lower-stack direct solve also remains negative.
This supports a covariance-construction cancellation diagnosis, not permission
to widen PSD/rank tolerances or change the noise, oracle, seeds or fit parameters.
The separately approved source-construction repair subsequently resolved this
hold, as recorded above; the fail-closed numerical policy remains.

## Preserved Historical V1 Failure

**2026-11-24: original quality gate FAILED; release blocked at that revision.** This is a named
sequential experiment, not general multiobjective cross-validation. Package and
engine versions remained 0.5.1 at that revision. No successful acceptance is
attributed to this historical v1 run.

### Historical Frozen Protocol

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

### Historical Results

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

### Historical Environments and Warnings

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
Task 6 SDD directory named above; commands, source provenance, failures and
results are in task-6-report.md. Historical v1 evidence remains in the original
Task10 handoff directory and task-10-report.md.
