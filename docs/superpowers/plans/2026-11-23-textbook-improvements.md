# Textbook Improvement Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans and test-driven-development. Independent example/documentation tasks may be delegated with disjoint file ownership.

**Goal:** Implement the approved textbook improvements except multi-objective recommendation.

**Architecture:** Extend the existing single-objective `recommend` boundary with a supplied candidate catalog, retaining the same GP, conditional acquisition, hard constraints and pending exclusion. Teach experiment lifecycle using existing Campaign APIs and a tested example rather than changing checkpoint schemas. Document log-response semantics without adding a new transformation API.

**Tech Stack:** Python, pandas, PyTorch, BoTorch, pytest; existing conda environment.

**Spec:** https://github.com/517justin/mi-textbook/blob/phase3-ch12/expdoe-dk-improvement-plan.md (read 2026-11-23), plus the user's approval to implement all but multi-objective recommendation.

## Global Constraints

- Base revision: `3d430962272ed14482674e33a0ba72117145a692`.
- No multi-objective implementation, dependency/version churn, commit or push in this task.
- Use `PYTHONPATH=src` from the inner package to avoid the old editable install in another checkout.
- Preserve physical candidate values and stable IDs. No candidate response columns enter selection.
- Preserve existing single-objective behavior when no catalog is supplied.
- Colab execution and textbook chapters 9/10/12 are external acceptance work; never claim them from local tests.

## Review Focus

1. Catalog ID/condition ambiguity and reordered rows: require explicit unique string IDs, reject duplicate physical conditions, sort by ID before scoring.
2. Leakage and precision: reject extra/missing columns, preserve original rows and integer/object dtypes rather than decoding candidates.
3. Hard constraints and safety: use the existing runtime and final checks, including equality-constrained catalog rows.
4. Partial returns and retries: distinct trial and candidate IDs; validate before mutation; remove only returned/cancelled pending trials.
5. Numerical claims: separate invariant correctness from ranking reproducibility, latent uncertainty from measurement noise, and optimization utility from response units.

## Task 1: Candidate Catalog Recommendation

**Files:** `bo/recommendation.py`, new `bo/candidates.py`, `bo/contracts.py`, `tests/phase_one/test_candidate_catalog.py`, `examples/03_candidate_catalog.py`, `docs/candidate-catalog.md` (package paths under `expdoe-dk/src/expdoe_dk` and `expdoe-dk/tests`).

**Interfaces:** `recommend(..., candidates: pd.DataFrame | None = None, candidate_ids: Sequence[str] | None = None)`; `RecommendationResult.candidate_ids: tuple[str, ...] = ()`. Existing result construction remains compatible; serialize IDs only for catalog results.

- [x] Write and run failing tests for exact catalog membership, deterministic ID alignment after row/column permutation, exclusion by physical condition, q exhaustion, strict validation, duplicate-condition rejection, mixed parameters, min/max, hard/knowledge/outcome safety.
- [x] Implement catalog validation and selection as an alternate pool feeding the existing scoring loop, not a new GP or continuous search. Return original values and ordered IDs.
- [x] Run new tests and existing recommendation/contract tests (23 new + 38 existing recommendation tests passed; broader contract checks remain in final suite).
- [x] Add runnable example and document duplicate/retry policy, local reproducibility and external Colab acceptance (example execution remains Task 5).

## Task 2: Measurement Lifecycle Teaching Example

**Files:** new `examples/04_measurement_workflow.py`, `docs/measurement-workflow.md`, `expdoe-dk/tests/test_measurement_workflow_example.py` only.

**Interfaces:** A clearly example-only workflow around `Campaign.tell_batch`, `set_pending`, `to_state`/`from_state`. Candidate IDs identify formulas; unique trial IDs identify measurements. No new public package API.

- [x] Write and run failing behavior tests for known/unknown IDs, duplicate returns, missing success values, failed/warning outcomes, repeated measurements with new trial IDs, partial returns and cancellation, plus save/restore.
- [x] Implement explicit transitions and validation before mutation; persist trial mapping/cancellation audit together with Campaign state; never fill measurement columns from predictions.
- [x] Run example tests and checkpoint regression tests; add documentation and a minimal executable demonstration (84 passed, additional model-rebuild regression included in final suite).

## Task 3: Constraint and Knowledge Regression

**Files:** `bo/recommendation.py` and focused `tests/phase_one/test_recommendation_constraints.py`; existing knowledge test modules remain the regression baseline.

- [x] Reproduce continuous linear equality search failure with a returned-candidate assertion, not only a feasibility-mask assertion.
- [x] Translate supported affine physical linear constraints into model-coordinate constraints for BoTorch; reject unsupported non-affine equality search explicitly, retaining catalog selection as the supported alternative.
- [x] Test mixed finite/continuous equalities, infeasible constraints and preservation of original single-objective behavior. Use the existing knowledge matrix for supported/conflicting/unsupported artifacts (75 focused and 215 knowledge regression tests passed in implementer's verification).

## Task 4: Target Scale and Roadmap Documentation

**Files:** new `docs/log-objectives.md`, `docs/textbook-improvements.md`; brief links in root English/Chinese READMEs (controller-owned).

- [x] Explain positive response units/reference scale, zero/detection-limit policy, lognormal mean vs median, latent vs new-observation variance, acquisition scale and optimization direction. No transform API added.
- [x] Record multi-objective deferral with exact original plan references: optimization pipeline Task 3 (qLogEHVI/qLogNEHVI, >=5 objectives scalarization), Task 8 (Pareto/hypervolume results); validation matrix metrics/oracles. Do not relabel planned features as implemented.
- [x] Separate local acceptance, model-quality assessment and still-unrun textbook/Colab acceptance.

## Task 5: Integrated Verification and Review

- [x] Run all tests including `--run-slow` from inner package with the selected source tree (final fix pass: 1328 passed, 117 warnings, 0 failed/skipped, 204.60s).
- [x] Run both new executable examples, inspect git diff/check and obtain a fresh code review.
- [x] Fix important review findings with regression tests. Report exact results and external limitations; leave changes uncommitted.

## Execution Record

- Scope and API decisions above are implementation choices following the approved improvement scope; lifecycle stays example-only to avoid promising a complete laboratory information system.
- Original multi-objective plan: `docs/superpowers/plans/2026-08-05-optimization-pipeline-implementation-plan.md`; no multi-objective guards are removed.
- Existing conda environment imports the old DOEGP checkout by default; all verification explicitly selects this checkout.
- Baseline: 1242 passed, 8 skipped, 2 Git-dependent sensitivity tests failed because system Git requires Xcode license acceptance. Running that complete sensitivity file with the bundled fallback Git first on PATH passed all 5 tests. No system license/settings were changed.
- Candidate RED: 23 failures from the missing `candidates` keyword after correcting the fixture to import the declarative constraint from `expdoe_dk.domain`. GREEN: candidate + existing recommendation tests, 61 passed in 13.01s.
- Documentation review corrected the original-plan result-reporting reference from Task 7 to Task 8.
- User selected an isolated worktree after initial edits. All task edits were transferred and verified byte-identical to `.worktrees/textbook-improvements` on `codex/textbook-improvements`; the source checkout is clean on `main`. Native worktree creation/attachment failed to recognize the checkout; fallback Git created and verified the linked worktree. No commit or push.
- Task 2 RED: 14 missing-example failures. GREEN: 84 example/checkpoint tests. Shared suite found one intentional result-contract field-list mismatch after adding candidate IDs; updated that test to exercise snapshots with and without IDs, preserving legacy `to_dict` shape.
- Task 3 RED reproduced false no-candidate errors for both equality APIs, mixed subspaces and fixed coordinates. GREEN covers physical constraints, redundant equalities and fail-closed optimization warnings. Scope is affine hard linear equalities; non-affine log-continuous equalities and finite products above 100000 direct users to a supplied catalog.
- Verification environment ruling: the conda environment has stale 0.4.0 distribution metadata. Created a worktree-local `.venv` using its Python with `--system-site-packages`, then installed this package with `--no-deps --no-build-isolation`. This reuses dependencies without replacing the original conda installation or weakening version tests.
- Integrated full slow-suite verification: first run 1304 passed / 1 stale-distribution-metadata failure. The version test passed after isolated installation; pre-review full run 1305 passed / 117 warnings / 0 skipped in 201.46s. New catalog/workflow examples and the log-response documentation snippet also executed successfully.
- Fresh review found eight actionable edge cases. Fix-pass RED reproductions: returned ordinal big-integer coercion, object-column batch coercion, model-coordinate collisions mislabeled as duplicates/exhaustion, saved ordinal catalog coercion, six audit/condition disagreements, and dispatch bypassing hard constraints. Boundary-affine-hull, equality tolerance and old-BoTorch imports are handled in the constraint task's regression tests.
- Review ruling: all eight findings are accepted. Multi-objective, Colab/predictive-quality acceptance and concurrent/atomic LIMS workflows remain outside scope as explicitly documented; ordinary checkpoint fidelity and audit consistency are in scope. No minor findings were deferred.
- Review fix-pass GREEN: 144 candidate/batch/workflow/checkpoint tests passed; the constraint implementer reports 89 constraint/recommendation/catalog tests passed. Both executable examples were rerun successfully. Exact object scalar types are preserved through copies and checkpoints; catalog/reference encoding collisions are explicit errors; restored audit mappings and dispatched hard constraints are validated. Equality fixes cover tight boundary faces, physically consistent redundant rows and near-boundary equations without relaxing hard inequalities. Old sampler API behavior is simulated, not a complete dependency-floor certification.
- Final controller verification after all production fixes: `pytest -q --run-slow` exited 0 with 1328 passed, 117 warnings, 0 failed/skipped in 204.60s. Both executable examples and the log-response guide snippet were rerun successfully. `git diff --check` passed; original main checkout remains clean. Changes remain uncommitted in the isolated worktree. Colab, textbook actual-data acceptance and a full dependency-floor environment remain unrun.
