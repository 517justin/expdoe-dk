# Four-pattern v2 sensitivity audit

This directory records the F6 numerical-policy evidence. It is not an E6 release
approval, a calibration study, or a claim that correct knowledge always improves
prediction. Read [REPORT.md](REPORT.md) for the observed outcomes and
[audit.json](audit.json) for the complete finite machine-readable record.

## Reproduce

From the engine repository root, with the canonical environment:

```bash
PYTHONPATH=expdoe-dk/src /opt/homebrew/Caskroom/miniconda/base/envs/expdoe-dk-codex/bin/python experiments/four_pattern_v2_sensitivity.py --output-dir docs/validation/four-pattern-v2
```

The equivalent explicit grid is `--seeds 17 29 43 --scale-multipliers .5 1 2
--width-centers .025 .05 .10`. `main(argv=None)` also accepts `--fixtures NAME ...`
for small regression runs, which are explicitly not full acceptance. Output
directories contain `audit.json` and `REPORT.md`. A nonzero exit indicates a
default noiseless, signed/normalization/isolation, replay, or units gate failure;
the evidence is still written. Unexpected harness exceptions abort rather than
becoming successful fit records.

## Frozen experimental definitions

The script declares all data, generating coefficients, fold memberships, and
physical prediction grids before fitting. There are 19 fixtures: saturation in
both directions, all six differing threshold branch pairs, synergy/antagonism in
both higher/lower orientations, additive-only, zero-effect, weak/noisy,
changed-units, missing-transition, composed saturation-plus-interaction, and
composed saturation-plus-threshold-plus-interaction cases. Every fixture has
separately declared correct and wrong knowledge.

There are two fixed complementary even/odd validation folds. One-dimensional
fixtures have 16 distinct rows; two-dimensional fixtures have 32 fixed factorial
rows (64 for the two-shape composition, covering transitions on both factors).
Each admissible training fold has at least two distinct values on each side
of the transition, and the interaction/complete mean basis is checked by the
engine before optimization. The missing-transition fixture deliberately violates
coverage; its `INSUFFICIENT_DATA` failures are not default optimizer failures.

Fifteen noiseless fixtures are declared mandatory before fitting. Additive-only,
zero-effect, weak/noisy, and missing-transition remain separately identified
stress cases. The generating threshold width is 0.08, not the fitted prior center;
effects, noise seed, known variance and transformations are recorded in each
fixture definition. Constant response folds use training scale S=1.

Each correct/wrong track actually fits all 27 policies per fold (including
inactive width settings for non-threshold fixtures). M0 and the existing Ridge
and 200-tree RF reference definitions run once per seed on exactly the same
folds. They do not depend on the v2 magnitude/width policies. All response and
feature normalization is fitted to training rows only. Default first-fold correct
fits are independently repeated for every seed and checked for exact equality.

## Reading the results

- Each structured run records its policy, physical means/variances, held-out
  metrics, typed failure when applicable, all start statuses, selected objective,
  fitted physical component parameters, prior/bound diagnostics and invariants.
- Pooled RMSE/MAE use observed physical responses and require every fold to
  succeed. Incomplete tracks have null pooled scores, never a flattering subset.
- GP 95% intervals use latent variance and are checked against the noiseless
  generator. They are conditional on fitted MAP parameters, not calibrated
  observation intervals. The noisy case is therefore not an observation-coverage
  claim. Ridge/RF coverage is null because uncertainty is unavailable.
- Sensitivity uses the largest pairwise absolute mean difference at any fixed
  audit-grid point. A flag requires a strict difference greater than one training
  response scale; equality does not flag. Zero/one successful configuration is
  `not_evaluable`. This engineering threshold is not a significance test.
- Component anchors, derivative/branch signs, width bounds, physical decoding,
  training-only response statistics and disjoint train/test IDs are checked.
  The mean/residual decomposition is not certified identifiable, and the full
  residual-GP prediction may contradict a component's sign.
- The changed-units fixture applies x'=100+1000x and Y'=273.15+100Y to the same
  saturation data. De-transformed predictions are compared to the original
  run with a predeclared tolerance of 0.001 training response scales. Differences
  and any failures are reported, not used to tune policies.

The JSON includes the exact command, source commit and dirty status, script
checksum, Python/dependency versions, complete serialized fixture definitions and
their SHA-256 checksums. The script uses one PyTorch thread for small-matrix
repeatability. Optimizer warnings retained in fitted diagnostics must be read
alongside fit status; convergence does not certify a scientific interpretation.

### Observed full-grid outcome

The recorded sweep contains 2,052 structured records: 1,944 finite successes and
108 expected missing-transition coverage failures. All 90 mandatory default
noiseless fits and all 342 M0/Ridge/RF fits succeeded. All 57 same-seed replay
checks matched exactly; no signed/normalization/isolation or units check failed.
The maximum de-transformed unit difference was 1.94345e-6 original response units.
Eighteen individual starts failed; the other starts supplied valid converged
selections, and every failure remains in the diagnostics.

Seven track/fold groups are `prior_sensitive`: correct synergy-higher fold 1,
correct synergy-lower folds 0/1, correct antagonism-higher folds 0/1, correct
antagonism-lower fold 1, and wrong composed fold 1. These findings are retained;
they do not trigger automatic tuning or a claim of robust scientific inference.

The default correct composed and two-shape-composed cases have **zero latent
95% truth coverage**, despite small pooled RMSE (about 0.00114 and 0.00190).
This is important evidence of conditional point-MAP uncertainty limitations,
not a calibrated uncertainty success. Passing the mandatory numerical-fit gate
does not establish calibration or release readiness.

## Test matrix recorded on 2026-10-25

From the package directory, using
`PYTHONPATH=src /opt/homebrew/Caskroom/miniconda/base/envs/expdoe-dk-codex/bin/python -m pytest`:

| Arguments | Actual result |
|---|---|
| `tests/phase_one/test_four_pattern_sensitivity.py -q` | 5 passed, 2 warnings in 9.00s |
| `tests -q` | 1,224 passed, 8 skipped, 58 warnings in 152.40s |
| `tests --run-slow -q` | 2 failed, 1,230 passed, 108 warnings in 162.66s |

The failed slow cases are
`tests/test_campaign_smoke.py::test_campaign_runs_end_to_end_and_improves` and
`tests/test_campaign_smoke.py::test_campaign_checkpoint_roundtrip`.
Their existing `lhs_maximin` / `lhs_random` generation fails to return the six
requested unique feasible design points. F6 does not repair or waive them.
Warnings include installed Torch deprecation, GPyTorch tiny-negative-variance
rounding, and Cholesky jitter. Eight default opt-in skips were exercised by the
slow run; it is still red and cannot approve release.

## Downstream contract and remaining gates

The separate public golden fixtures are
`expdoe-dk/tests/phase_one/fixtures/four_pattern_v2_cases.json`; historical v1
storage cases remain in `contract_cases.json`. The public evaluation contract
adds `fold_diagnostics` keyed by canonical JSON validation-index lists (for
example `"[3,7]"`). Entries contain training/validation indices and IDs,
training-only response location/scale, fit status and `fitted_diagnostics`.
The latter includes `structured_mean` coverage, starts, parameters, policy,
`parameter_identifiability: not_certified` and `uncertainty: conditional_on_MAP`.
A failed fold retains its typed reason/coverage and has no complete OOF score.

The four pattern names now have numerical v2 consumers, while their v1
descriptor semantics and opt-in migration remain unchanged. Registry inventory
is 23 distinct names / 27 versioned definitions, not 27 supported scientific
models. Legacy shape means/virtual augmenters mixed with v2 remain blocked;
unimplemented descriptors and wide-integer precision restrictions remain.

E6 still owns independent review of this evidence, the complete test matrix
(including the two historically failing slow LHS cases), the wide-integer support
contract, consistent version/compatibility updates, build, and clean offline
wheel smoke with reference dependencies. No CLI/adapter task, release version,
wheel approval, merge, push, publication or E6 approval is implied here.
