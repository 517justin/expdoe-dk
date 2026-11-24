# Monotone v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add explicitly versioned soft directional knowledge without changing legacy behavior, pass the approved fixed acceptance protocol, and finish the held 0.6.0 release work.

**Architecture:** Fit the existing measured-data GP, then condition its joint latent posterior with finite-difference probit factors using a whitened Laplace approximation. Keep monotone 1.0 and old APIs unchanged; new 2.0 declarations opt in to a separate consumer. Reuse domain constraints, exact knowledge envelopes, physical posterior adapters and the existing multiobjective search.

**Tech Stack:** Python, Torch/GPyTorch/BoTorch, SciPy, NumPy, pandas, pytest; no new mandatory dependency.

**Spec:** docs/superpowers/specs/2026-11-24-monotone-v2-compatibility-design.md (approved 2026-11-24).

## Global Constraints

- Do not redirect old definitions, add an implicit latest-version resolution, or change the existing with_monotone signature/default/version.
- V2 contributes NO fabricated function-value rows and NO Yvar entries.
- Keep the base GP hyperparameters fixed during the directional conditioning step.
- Confidence is a fractional evidence weight, NOT a calibrated probability that the rule is true and NOT a measurement variance.
- The directional log factor is (confidence / n) * sum_j log Phi(sign_j * D_j / (softness * h_j)).
- n_pairs_per_dim in [1,32], 0 < epsilon < 0.5, softness > 0, confidence in [0,1], at most 64 active pairs/output, at most 4096 generated centers/spec.
- Defaults: n_pairs_per_dim=5, epsilon=0.05, softness=1.0, confidence=1.0; materialize all defaults in specs. No v2 delta_norm or epsilon="auto".
- Unknown versions, impossible pairs, nonconvergence or invalid covariance fail explicitly; no old-method/plain/random fallback.
- Full joint-q covariance, physical units, scope isolation, safety and original measurement metadata remain correct.
- Existing failed acceptance evidence remains; original seeds, budget, reference, objective functions, and quality tolerances are not relaxed.
- Version/README release claims only after acceptance. No merge/push/tag/publish authorization.

## Review Focus

1. Confidence-zero or legacy-only calls must not change predictions, RNG, noise or serialized declarations; test in Tasks1/4/5.
2. Singular near-noiseless differences and repeated endpoints must not create fabricated uncertainty or silently flatten covariance; test in Tasks2/3/4.
3. Equality constraints, discrete snapping, global parameter knowledge and colliding parameter/objective names must retain truthful scope and identity; test in Tasks2/4/5.
4. Wrong or duplicated directional knowledge must be distinguished from measured facts; reject duplication and expose conflict diagnostics without rewriting data; test in Tasks1/3/5.
5. Historical archives and revised acceptance protocol must not be relabeled as new-model success; test in Tasks5/6 and inspect release docs.

## Workspace and Verification

Existing isolated worktree: .worktrees/multiobjective-v060, branch codex/multiobjective-v060. Starting source6f83bcb; original multiobjective branch base197cf49. Main/unrelated worktrees untouched.

Use fallback Git /Users/justnichen/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git (system Git license unavailable).
From inner expdoe-dk run ../.venv/bin/python -m pytest with PYTHONPATH=src,
OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, fallback Git first PATH, unique TORCH_EXTENSIONS_DIR.
Focused tests during iteration; full fast once before each task commit.
The existing full slow suite has one EXPECTED-TO-FIX legacy quality-gate failure,
not a green release baseline. Do not xfail it. Task6 implements approved protocol revision.

Prepared lower environment, legacy API overlay and clean offline wheel environment
remain under .superpowers/sdd/2026-11-24-multiobjective-recommendation-addendum:
.venv-floor, legacy-api, .venv-wheel, offline-wheelhouse. Reuse these environments,
write NEW evidence only to this plan's workspace. Do not re-download dependencies
or treat an editable install as clean wheel validation.

## File Responsibilities

- knowledge/patterns/physics.py + knowledge/__init__.py: versioned spec/helper.
- bo/directional_pairs.py: deterministic feasible pairs and detached metadata.
- bo/directional_posterior.py: likelihood, numerical state and joint posterior adapter.
- bo/surrogates.py: narrow fitting/conditioning integration.
- bo/knowledge_runtime.py + capabilities.py: payload/composition/fail-closed route.
- bo/loop.py + checkpoint.py + report.py: v2 validation, persistence and diagnostics only.
- tests/monotone_v2/: new focused tests, with __init__.py to prevent pytest basename collisions.
- experiments/multiobjective_validation/: explicit v2 protocol and preserved legacy result.
- Existing release docs/version files: gated final Task6.

## Task 1: Versioned declaration and strict policy contract

**Files:** Modify knowledge/patterns/physics.py, patterns/__init__.py, knowledge/__init__.py, bo/knowledge_runtime.py; Create tests/monotone_v2/__init__.py and test_contracts.py (all source/tests inside expdoe-dk).

**Interfaces:** Add monotone_v2_definition(); Knowledge.with_monotone_v2(param, effect, *, objective=None, n_pairs_per_dim=5, epsilon=0.05, softness=1.0, confidence=1.0) -> Knowledge.
Compiler emits virtual_observations kind monotone_sign_v2, payload {dim,direction,n_pairs_per_dim,epsilon,softness,confidence}; direction is internal frame.
Add check_directional_composition(artifacts) -> None in knowledge_runtime.py for later wiring. Registry validation/helper usable now; capability remains blocked until numerical integration in Task4.

- [ ] RED test_v2_defaults_and_exact_envelope: assert pattern/version, exact payload and immutable defaults, old helper remains1.0.
- [ ] RED parameterized boundary tests bool/NaN/inf/count0/33/epsilon0/.5/softness0/auto/delta_norm/unknown objective; roundtrip both versions, old item view excludes2.0.
- [ ] Run pytest -q tests/monotone_v2/test_contracts.py and capture genuine missing-interface RED.
- [ ] Implement schema/compiler/helper and direct composition validator. Reject duplicate same-factor/output declarations, opposing directions, v2 plus legacy virtual-response on same output; allow distinct outputs/factors. Count only active v2 factors toward64; zero-confidence is inactive but malformed declarations still reject.
- [ ] GREEN focused existing knowledge registry/envelope tests and new contract tests. Preserve no-wired-consumer behavior beforeTask4.
- [ ] Fullfast, self-review and commit feat: declare opt-in monotone v2 knowledge.

## Task 2: Deterministic feasible directional pairs

**Files:** Create bo/directional_pairs.py, tests/monotone_v2/test_pairs.py.

**Interfaces:** frozen DirectionalPairs with endpoints Tensor[n,2,d] in Space model coordinates, signs Tensor[n], widths Tensor[n], softness Tensor[n], weights Tensor[n], pattern_ids tuple[str,...], diagnostics detached list[dict].
build_directional_pairs(space, observations, artifacts, *, seed, parameter_constraints=()) -> DirectionalPairs.
Artifacts are already validated projected v2 artifacts; observations are successful scalar measurement batch; constraints include full global RuntimeParameterConstraints. No response values used for locations.

- [ ] RED test_seeded_pairs_respect_physical_domain: same seed/data produces equal endpoints, first observed centers before Sobol, unchanged global RNG/data, widths reflect actual snap.
- [ ] RED stepped/log inputs, categorical context, equality/no-free-direction, duplicate centers, requestedcount unavailable after4096, global Knowledge constraint, malformed types and64total limit.
- [ ] Run pytest -q tests/monotone_v2/test_pairs.py; then implement exact specSection6 algorithm through existing Space transforms/domain feasibility. Sort specs bystable ID, isolate seeded Sobol; no latent outcome or catalog membership filter.
- [ ] Zero-confidence creates no pairs/consumes no RNG and diagnosticinactive; duplicate/collapsed pairs excluded. Compare actual endpoint coordinates, not approximate category conversion.
- [ ] GREEN newpairs+domain regression; test parameter names containing trial_id/objective names as normal names.
- [ ] Fullfast/self-review/commit feat: generate feasible directional knowledge pairs.

## Task 3: Soft-direction Laplace posterior engine

**Files:** Create bo/directional_posterior.py, tests/monotone_v2/test_posterior.py.

**Interfaces:** condition_directional_posterior(model: BoTorch Model, pairs: DirectionalPairs) -> tuple[Model, dict].
Returned model consumes the SAME internal standardized GP response and Space model coordinates as inputmodel; num_outputs=1 and original batch_shape.
Empty active pairs returns the original model with inactive diagnostics, no optimizer/RNG.
Expose private helper _fit_directional_laplace(mean, covariance, signs, scales, weights) for independent numeric unit tests, returning a frozen state with rank-aware whitening, mode, Hessian factor and diagnostics. scales=softness*widths.

- [ ] RED test_one_difference_laplace_matches_independent_mode_and_hessian; score logPhi using vetted stable primitives, not clamped probability or rawlog(cdf).
- [ ] RED test_joint_posterior_matches_block_formula, repeated/singular pairs, zero-rank covariance, materialnegativeeigenvalue/NaN/nonconvergence/nullspace inconsistency typederror.
- [ ] Run pytest -q tests/monotone_v2/test_posterior.py; implement specSection7 mean/covariance with rank-aware factorization, SciPy optimizer and Torch derivatives. Record numerical tolerances/solver rationale BEFOREqualityrunner.
- [ ] Use actualbasejoint endpoint posterior (not independentpredict marginals), finite nonnegative Hessian, linear solves. Retained-space/nullspace checks must not silently discard material covariance.
- [ ] GREEN independent quadrature approximation checks with explicit chosen tolerance and reference regime; no claim Laplace equals exact moments. Wrongdirection/offset/weight tests, finite-difference querygradients, sample mean/fullcovariance, q>1 t-batches andrepeatedqueries.
- [ ] Fullfast/self-review/commit feat: condition GP posteriors on soft directional factors.

## Task 4: Shared scalar/multiobjective fitting integration

**Files:** Modify bo/surrogates.py, capabilities.py, knowledge_runtime.py, objective_routing.py onlyifneeded; Create tests/monotone_v2/test_integration.py.

**Interfaces:** fit_surrogate/recommend public signatures stayunchanged. Internal _fit_surrogate gains optional private parameter_constraints context if needed for projected global constraints; samecontext reaches allprojectedoutput fits withoutresponseleakage. Call Task2 builder and Task3 conditioner after existing measured-data fit.

- [ ] RED real scalar/multi scoped fits assert NOv2 function-value rows, real Yvar exact, zero-confidence posterior/RNGidentical, oldv1 independent reference identical.
- [ ] RED newartifact inspect_capabilities becomesapplied onlyafterwired consumer; duplicate/legacyconflicts failCONFIG_INVALID withpatternIDs; globalconstraints surviveprojection.
- [ ] Run pytest -q tests/monotone_v2/test_integration.py, then integrate. V2 must not route through legacyaugmenter; registeredmean/kernel/prior behavior stillworks.
- [ ] Addconsumer manifest entry, runtimepayloadchecks, callcompositionvalidator forscalarandallmulti routes. Preserve exact versions/sourceIDs and old error behavior outsidev2.
- [ ] GREEN positiveaffine responseinvariance/nonconstant, directionflips, unscopedoutputunchanged, actual0/known/inferredvariancefits; predict agreesphysicalacquisitionposterior.
- [ ] Real qLogEHVI/NEHVI/NParEGO forward/backward/sampling withv2 andphysicaloutcomesafety; no duplicateddirectionalsafetyfactor. Oldscalar regression and missingcapabilitytypederrors.
- [ ] Fullfast/self-review/commit feat: integrate versioned directional GP conditioning.

## Task 5: Campaign diagnostics, persistence and compatibility

**Files:** Modify bo/loop.py, checkpoint.py, report.py narrowly; Create tests/monotone_v2/test_campaign.py, test_checkpoint.py, test_report.py; existingliteralfixtures remainhistorical.

**Interfaces:** publicCampaign/Result/checkpoint signatures unchanged. V2 declarations stayexactenvelope; adddetached currentmodel/validationdiagnostics through existingdiagnosticfields, not inventedhistoricalmodelstate.

- [ ] RED test_v2_resume_same_next_batch: scalar andmulti checkpoint restore givesidenticalnextIDs/physicalvalues/pairs/weights/geometry; unchangedv1archiveproducesoldalgorithm.
- [ ] RED validation/reporting distinguishesversion/objective/patternID; v2neverbecomeslegacyitem orgetsauto-epsilonrescue. Namedmeasuredtrendchecks mayreusevalidator mathematics through explicitv2adapter, notlegacymutation.
- [ ] Run pytest -q tests/monotone_v2/test_campaign.py tests/monotone_v2/test_checkpoint.py tests/monotone_v2/test_report.py.
- [ ] Implementexactcodecusing existingKnowledgeenvelope whereverpossible. Existingcheckpointdiagnosticschema mayneedfiniteJSON optional/variant expansion forv2diagnostics; no arbitraryextra fields orpickles, no scalarlegacywire/key churn.
- [ ] GREEN unknownv2versions/invalidparams/scopescorruption reject; historycollisionmap andunsuccessfulmissingmarkers retained; refitfailingaskdoesnotcommitcampaignstate.
- [ ] Report observeddata separate fromdirectionalfactors, confidenceweightnotprobability, pairlimits/approximation; emptyhistory andinactiveconfidence handled.
- [ ] Fullfast/self-review/commit feat: preserve directional policy through campaign checkpoints.

## Task 6: Approved quality protocol and 0.6.0 release

**Files:** Modify experiments/multiobjective_validation/{fixtures.py,run.py,README.md}, tests/multiobjective/test_acceptance.py, docs/multiobjective.md, examples/05_multiobjective_catalog.py; Create/Update docs/roadmap.md; Modify rootREADME.md/README_zh.md, expdoe-dk/README.md, docs/textbook-improvements.md, expdoe-dk/pyproject.toml, src/expdoe_dk/__init__.py, domain/space.py, knowledge/__init__.py, checkpoint.py andversion/release tests. Narrowexpectedwarning assertions allowed innewtests; noglobalsuppression.

**Interfaces:** run_matrix(*, seeds, output_dir) remains; explicitprotocol/version and methodknowledge_gp_v2 added. Fourreportedmethods random/plain_gp/knowledge_gp/knowledge_gp_v2; legacyFAIL reportedseparately, gates nowapplyapprovedv2protocol. Historicalfailedresulttable retainedwithsourceSHA.

- [ ] RED protocol test assert exactmethodidentity, v2paramsconfidence.8/pairs3/epsilon.05/softness1, unchangedinitial6/seeds(2,5,11,23,47)/rounds8/q2/reference/catalog/oracleHV.476 andgeometry.
- [ ] Run focusedtests, implement explicitprotocol andnewreleasegate withoutxfail; keep oldfailure asdeclaredhistoricalbaseline andv1behaviorregressions.
- [ ] Commit implementation/tests beforeexpensivematrix forcleanprovenance; run current andlower fullfixedmatrices including replay/restart, feasibility1, duplicates0, exactq.
- [ ] Require plain<=1.10*max(random,1e-12), v2<=same andv2<=1.10*max(plain,1e-12). No post-resultparameter/seed/noise/tolerance tuning. Failedgate meansholdanddiagnose, notfakegreen.
- [ ] Run full pytest -q --run-slow current/lower, oldBoTorch capabilityoverlayprobe andexamples03/04/05. Recordeveryfailure/skip/warning; lowerTorch2.2 uint64 testexistingcapabilityskip allowed.
- [ ] ONLYafterfunctionalqualitypass updateconciseREADMEEN/ZH/packageREADME, roadmapandtextbookcurrentstatus; detailedper-methodresults inexperimentREADME. Clearlydocumentv1opt-incompatibility/v2approximation andunsupportedlimits.
- [ ] Updateversions0.6.0 coherently: pyproject/__version__/ENGINE_VERSION +exactreaderallowlists, old0.5.0/.5.1 fixturetests; schema2.2releasewriter0.6.0only. Historical2.0/2.1fixturesremainold. CorrectPythonfloor>=3.11 fromverifiedAx1.2.4dependency.
- [ ] Reinstallprimary/lowereditablemetadata; rerunfullsuites; buildwheel/sdist andinstallnewwheel --no-index into preparedclean .venv-wheel. Outside source/PYTHONPATH verifyimportfile/metadata/engine0.6.0, examples03/04/05 andoldcheckpointread. Existinglegal/dependencyartifactchecks mustpass.
- [ ] Commit feat: release validated versioned monotone workflows; taskreview thenwholebranchreview fromoriginal197cf49, coveringbothmultitargetandmonotonework. No merge/push/tag/publish.

## Self-review and execution

SpecSections1–4 -> Task1; 5/7 -> Task3/4; 6 -> Task2/4; 8 -> Tasks1/4/5;
9A -> Tasks2/3/4, 9B -> Tasks4/5, 9C/D ->Task6; risks/limits ->Task6docs.
AllfiveReviewFocusitems haveownedtests. Interface names/types align acrossTasks.
Task1advertisesno consumer untilTask4; Tasks2/3 independentlytest numerichelpers.
Userexplicitlyrequested implementation afterfullspec andpreviouslyselectedSDD.
Execute thisderivedplan withoutanother approvalprompt; recordanymaterialdesignchange.
