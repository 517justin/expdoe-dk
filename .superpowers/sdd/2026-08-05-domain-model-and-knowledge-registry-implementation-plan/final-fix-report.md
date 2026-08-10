# expdoe-dk v0.5 final fix report

Date: 2026-08-11
Review base: `4de10c9099da099b0745a9475a11631158f0d60b`
Implementation commit: `4dda5f8673eed41863e31024a533f78a098ddd15` (`fix: close v0.5 final review findings`)
Outcome: all 12 Critical/Important findings closed; release artifacts verified locally; nothing uploaded or pushed.

## Scope and governing resolutions

- Preserved the exact v0.4 `Knowledge.to_dict()` projection and unversioned Campaign checkpoint migration reader.
- Added a separate bidirectional versioned knowledge envelope and Campaign checkpoint schema rather than changing the legacy payload.
- Accepted safety source expressions only at the helper/`ExpressionConstraint` construction boundary. Persistent state, IDs, checkpoints, and artifacts contain only canonical constraint AST data.
- Kept disabled declarations in the versioned audit envelope while excluding them from legacy effective items, validation, compilation, and application.
- Adopted the established v0.4 integer midpoint rule: exact half-grid ties select the higher grid index for decode and Python/NumPy/Torch snapping.
- Used SciPy's declared optimization backends for certified categorical balance and affine feasibility. Resource limits return typed, explicit failure or inconclusive diagnostics; there is no heuristic safety/balance fallback.
- Did not edit the SDD progress ledger, historical plan/spec records, adapters/plugins, algorithms outside the findings, or observer/device work.

## Per-finding RED/GREEN evidence

### 1. Versioned knowledge in checkpoints/results

RED: a checkpoint/result serialized only `Knowledge.to_dict()`. A safety spec, disabled audit spec, and provider declaration disappeared on round-trip; provider executable availability was not distinguishable from declaration provenance.

GREEN:

- `test_versioned_knowledge_envelope_is_bidirectional_and_keeps_audit_specs`
- `test_checkpoint_and_result_preserve_versioned_knowledge_and_provider_provenance`
- unknown knowledge/checkpoint schema and engine versions now fail with `CHECKPOINT_INCOMPATIBLE`
- restored provider declarations require an exact explicit reload; a manual same-key definition, distribution-version mismatch, or schema-digest mismatch fails with `EXTENSION_NOT_ALLOWED`, while the exact loaded record succeeds
- focused envelope/provider run: `48 passed, 2 warnings in 0.17s`

### 2. Adjacent large integral bounds/levels

RED: `(2**53, 2**53 + 1)` was converted to identical floats. Integer bounds failed as unordered and explicit integral discrete levels failed as duplicates. Adjacent integral log levels also produced a zero logarithmic span.

GREEN:

- exact integral validation/storage is retained through encode, decode, snap, and payload round-trip
- stable `log1p`/`expm1` relative arithmetic preserves adjacent integral log levels
- `test_adjacent_integral_domains_above_float_precision_remain_distinct`
- `test_adjacent_integral_log_levels_preserve_their_model_span`

### 3. Integer midpoint tie disagreement

RED: with `base = 2**53`, bounds `(base, base + 4)`, step `4`, and midpoint `base + 2`, decode selected `base` while snap selected `base + 4`.

GREEN: exact rational grid-index rounding now uses half-up everywhere. `test_integer_midpoint_ties_choose_the_upper_level_on_every_backend` covers decode and Python, NumPy, and Torch integer/floating inputs without sacrificing arbitrary precision. Existing v0.4 half-up fixtures remain unchanged.

### 4. Successful observation values

RED: successful Y cells containing `"1.25"`, `True`, or `1+2j` were accepted through float coercion; complex input could emit a warning while discarding the imaginary part.

GREEN:

- successful outcomes accept only actual finite real, non-boolean numeric scalar values
- string, boolean, complex, NaN, and infinity fail without coercion
- failed-row missing-value semantics are unchanged
- `test_successful_rows_reject_coercible_nonnumeric_boolean_or_complex_values`
- `test_successful_rows_accept_finite_real_numeric_scalar_types`

### 5. Target objective validation

RED: booleans, strings, non-finite values, malformed ranges, and wrong-shaped values survived construction or failed later in utility evaluation. An integer too large for the model float leaked `OverflowError`.

GREEN: construction normalizes one finite non-boolean real point or exactly two finite ordered endpoints and raises typed `CONFIG_INVALID` for malformed/unrepresentable declarations. The malformed-target matrix, valid NumPy scalar normalization, and utility fixtures pass.

### 6. Certified categorical/ordinal balance

RED: for `list(itertools.product([0, 1], repeat=5))[:23]`, five binary ordinal factors, and `n=6`, `comb(23, 6) == 100947` crossed the old cutoff. The heuristic selected 4/2 marginals for two factors although a 3/3 solution existed.

GREEN:

- SciPy MILP certifies minimum maximum deviation, then minimum total deviation
- deterministic maximin feasibility is certified by distance-threshold MILPs, followed by lexicographically smallest pool-index selection
- the exact regression independently enumerates all 100,947 subsets and verifies both global balance/maximin and deterministic tie-breaking
- a pre-allocation coefficient limit fails explicitly with typed solver/resource provenance rather than attempting an unbounded dense pairwise formulation
- design plus v0.4 focused run: `73 passed, 2 warnings in 7.95s`

### 7. `n_restarts` forwarding

RED: the constrained compatibility wrapper deleted `n_restarts`, so wrapper output differed from the direct design-engine call.

GREEN: the wrapper forwards `n_restarts` unchanged while retaining only the documented compatibility no-ops. `test_legacy_doe_generate_forwards_n_restarts_to_the_design_engine` compares the wrapper and direct deterministic result.

### 8. Disabled/invalid knowledge compiler gating

RED: disabled declarations could resolve/validate/compile, and structurally invalid declarations could reach their compiler. `insufficient_data` was not cleanly distinguished from invalidity.

GREEN:

- disabled specs remain in `to_envelope()` but are absent from v0.4 items and all effective registry paths
- enabled specs pass JSON schema, compatibility, and validator gates before compiler execution
- `invalid` raises a typed error; `insufficient_data` may compile and carries deterministic diagnostics
- safety validation no longer calls its compiler internally
- regressions use compiler counters and explicit invalid/insufficient states

### 9. Quadratic peak model coordinates

RED: a discrete-log parameter crashed because the compiler unpacked `bounds=None`; continuous-log center `10` over `(1, 100)` produced `0.090909...` instead of `0.5`. A center inside a discrete range but absent from its levels reached a raw compiler `ValueError`.

GREEN: compilation and validation use `Parameter.encode()`. Continuous-log and discrete-log centers both compile to `0.5`, and an undeclared discrete center fails as typed `KNOWLEDGE_INVALID` before compiler execution.

### 10. Hard-safety feasibility

RED: only scalar interval contradictions were proved. `x != x`, finite-domain `x + y == 0.5`, and contradictory continuous affine conjunctions were not rejected; unsupported forms lacked explicit provenance.

GREEN:

- all-finite domains at or below 100,000 combinations are enumerated exactly
- supported bounded continuous affine conjunctions are solved deterministically with `scipy.optimize.linprog`, including strict inequalities through a shared maximized margin
- `x != x` is structurally false; multivariate contradictions are typed `KNOWLEDGE_CONFLICT`; tautologies certify cleanly
- unsupported nonlinear ASTs, finite enumeration limits, solver uncertainty, and mixed domains with non-float-exact integral coordinates return explicit provenance-bearing inconclusive diagnostics
- internal-review REDs for tiny positive strict margin (`x < 1e-10`), frozen boolean/call ASTs, exact values above `2**53`, mixed-domain precision, and validator/compiler order are all covered
- final knowledge run: `269 passed, 2 warnings in 2.22s`

### 11. Canonical safety persistence

RED: helper-generated specs/artifacts retained `source_expression`, and spelling-equivalent expressions could affect deterministic persistence/identity.

GREEN:

- the helper parses source immediately and stores only `{kind: expression, hard: true, ast: ...}`
- persisted source-shaped parameters are rejected by schema
- artifacts/checkpoints contain no source text
- canonical-equivalent source spellings yield the same deterministic identity
- direct and JSON-envelope boolean/call AST rehydration are covered after immutable tuple freezing

### 12. Release artifact cleanup

RED: the package NOTICE/LICENSE contained the prohibited private attribution, both NOTICE inputs retained the removed legacy dependency attribution, and root `requirements.txt` still installed that dependency. Those license inputs were configured for wheel inclusion.

GREEN:

- package LICENSE/NOTICE and root NOTICE/requirements were corrected
- the source-input scan and built-archive scan are automated in `tests/test_release_artifacts.py`
- final wheel and sdist contain clean LICENSE/NOTICE, clean dependency metadata, version `0.5.0`, and expected registry/source files
- slow release gate: `2 passed in 0.25s`

## Architecture and migration details

### Knowledge envelope and provider provenance

The v0.5 knowledge envelope is:

```text
schema_version = 1.0
engine_version = 0.5.0
strict
specs[]
providers[]
```

`Knowledge.to_dict()/from_dict()` remain the exact v0.4 projection. `to_envelope()/from_envelope()` are the separate reversible path. Provider records are frozen and deterministically sorted. Each record contains original and PEP-503-canonical distribution names, distribution version, entry-point name, exact pattern/version declarations, and a SHA-256 digest of canonical JSON schema. No provider executable code is serialized. Before compiling an enabled restored provider declaration, the complete recorded provider record must exactly match a record attached by explicit loading.

Campaign checkpoint schema `2.0` adds the engine version and knowledge envelope. An unversioned payload dispatches to the unchanged v0.4 reader. `Result.to_dict()` retains `knowledge_summary` and adds the full `knowledge` envelope.

### Safety and registry ordering

The registry effective path is schema -> compatibility -> validation -> compiler. Invalid results stop before compilation; insufficient-data results compile with diagnostics. Safety validation operates on parsed canonical constraints, not a compiler-produced artifact. Combined safety artifacts are certified after compilation without source text.

### Numeric and design behavior

Integral physical values remain Python integers until a model-coordinate boundary. Exact rational arithmetic controls integer grids and half-up ties. Stable relative logarithmic arithmetic avoids subtracting rounded large logarithms. The balance path is exact enumeration below the declared cutoff and certified MILP above it; when the exact secondary formulation would exceed its declared resource budget, it fails explicitly.

## Verification record

All commands ran from `expdoe-dk/` with `../.venv/bin/python` unless noted.

### Focused and full tests

```text
../.venv/bin/python -m pytest -q tests/test_parameter_types.py tests/test_observation_batches.py tests/test_objectives.py
113 passed, 2 warnings in 1.51s

../.venv/bin/python -m pytest -q tests/doe/test_design_invariants.py tests/test_v04_compatibility.py
73 passed, 2 warnings in 7.95s

../.venv/bin/python -m pytest -q tests/knowledge/test_envelope.py tests/knowledge/test_provider_loading.py
48 passed, 2 warnings in 0.17s

../.venv/bin/python -m pytest -q tests/knowledge
269 passed, 2 warnings in 2.22s

../.venv/bin/python -m pytest -q
684 passed, 4 skipped, 2 warnings in 37.79s
```

The two warnings are the previously deferred dependency-owned Torch JIT deprecation warnings.

### Compile, diff, and source gates

```text
../.venv/bin/python -m compileall -q src
exit 0; no output

git diff --check
exit 0; no output

git diff --name-only | rg 'progress\.md$'
exit 1; no matches (ledger untouched)

git grep -n -i -E 'MI-6|pyDOE3' -- ':!docs/superpowers/**' ':!.superpowers/**'
exit 1; no shipped-source/dependency-input matches
```

### Offline builds

The environment's optional PyPA frontend was unavailable:

```text
../.venv/bin/python -m build --no-isolation --wheel --sdist
No module named build.__main__; 'build' is a package and cannot be directly executed
```

The project-declared backend was then invoked directly, offline and without build isolation:

```text
../.venv/bin/python -c 'from setuptools.build_meta import build_sdist, build_wheel; print(build_wheel("dist")); print(build_sdist("dist"))'
expdoe_dk-0.5.0-py3-none-any.whl
expdoe_dk-0.5.0.tar.gz
exit 0

../.venv/bin/python -m pytest -q --run-slow tests/test_release_artifacts.py
2 passed in 0.25s
```

Final SHA-256 values:

```text
ba00d80fd7223ad6f063b4960c9093529ed580ba573243b6b44052575f5abf70  expdoe_dk-0.5.0-py3-none-any.whl
f40fc0ff02eff993482ff3c4c75d422918a23ac30186cc877dc16e34e7574a5a  expdoe_dk-0.5.0.tar.gz
```

Archive inspection found 46 wheel members and the expected sdist tree, including `expdoe_dk/__init__.py`, knowledge registry/provider modules, LICENSE, NOTICE, README, `pyproject.toml`, PKG-INFO/METADATA, and selected tests. Programmatic content inspection decoded every archive member and found neither prohibited/stale text.

Both METADATA and PKG-INFO report:

```text
Metadata-Version: 2.4
Name: expdoe-dk
Version: 0.5.0
Requires-Python: >=3.10
Requires-Dist: jsonschema>=4.18
Requires-Dist: torch>=2.0
Requires-Dist: botorch>=0.11.0
Requires-Dist: gpytorch>=1.11
Requires-Dist: ax-platform>=1.2.4
Requires-Dist: numpy>=1.24
Requires-Dist: scipy>=1.12
Requires-Dist: pandas>=2.0
Requires-Dist: matplotlib>=3.7
```

The root requirements inspection includes `scipy>=1.12` with no removed legacy dependency. Wheel and sdist NOTICE contents match the cleaned attribution/dependency list.

### Installed-wheel/version/no-implicit-provider probe

The wheel was installed with `--no-deps --no-index` into a fresh `/private/tmp` target. Entry-point enumeration was replaced with an assertion-raising sentinel, then `Knowledge()` and `PatternRegistry()` were constructed.

```text
0.5.0 0.5.0 0.5.0 1.0 2.0 /private/tmp/expdoe-final-wheel.1gCjaf/expdoe_dk/__init__.py
```

Fields are package `__version__`, installed distribution version, engine version, knowledge schema version, and checkpoint schema version. The sentinel was never called.

## Files changed

- Domain: `domain/parameter.py`, `domain/observation.py`, `domain/objective.py`
- Design: `doe/design.py`, `doe/constrained.py`
- Knowledge/envelopes/checkpoints: `knowledge/__init__.py`, `knowledge/guard.py`, `bo/loop.py`, package `__init__.py`
- Safety/patterns/registry/providers: `knowledge/patterns/feasibility.py`, `knowledge/patterns/shape.py`, `knowledge/registry/providers.py`, `knowledge/registry/registry.py`
- Release inputs: root/package NOTICE, package LICENSE, root requirements
- Regression coverage: domain, design, v0.4 compatibility, registry/provider/pattern, envelope/checkpoint, and release-artifact tests
- This report is the only SDD record changed; the progress ledger was not edited.

## Self-review

- Independently rechecked the exact 23-row balance counterexample and global maximin/lexicographic witness.
- Reconciled midpoint behavior with v0.4 fixtures and retained half-up rather than changing legacy semantics.
- Added a balance resource gate after review identified dense pairwise/exclusion-row growth.
- Added post-review regressions for frozen boolean/call safety AST rehydration, tiny strict margins, large integral finite and mixed safety, validator-before-compiler ordering, exact provider provenance matching, discrete quadratic membership, and huge target overflow.
- Rebuilt artifacts after the final source change and repeated archive, installed-wheel, compile, full-test, diff, ledger, and source scans.
- No known Critical or Important findings remain. The only observed warnings are dependency-owned and explicitly deferred by the authoritative findings.

## Commits

- `4dda5f8673eed41863e31024a533f78a098ddd15` — `fix: close v0.5 final review findings`
- Report commit — the commit containing this file; its hash is included in the final handoff because a Git commit cannot embed its own final hash.

## Exception Wave — Authoritative provider provenance and atomic publication

Date: 2026-08-11

This user-authorized exception wave addressed only the two re-review blockers. It did not change the SDD progress ledger or take on any deferred Minor work.

### 1. Restored provenance remains authoritative

RED independently reproduced the same-entry replacement counterexample with literal, distinct provider v1/v2 distribution versions and schema digests. Before the fix, serializing a restored v1 envelope against a registry containing v2 emitted v2 before compilation, silently rewriting the checkpoint's expected provenance.

GREEN changes the audit merge precedence so restored provider records remain the authoritative expected state. Full-record verification still compares original and canonical distribution names, distribution version, entry-point name, exact pattern/version declarations, and canonical-schema SHA-256 digests. A missing or mismatched explicitly loaded provider raises typed `EXTENSION_NOT_ALLOWED` without mutating the expected records. Focused tests also prove input/output detachment, repeated serialize/restore stability before and after a failed compile, exact-match compilation, and fresh `Knowledge` snapshots when no restored expectation exists.

```text
../.venv/bin/python -m pytest -q tests/knowledge/test_envelope.py
13 passed, 2 warnings
```

### 2. Provider definitions and records publish atomically

RED introduced real-state transaction regressions. Against the pre-fix loader they produced `3 failed, 1 passed`: a post-provenance-assignment exception escaped raw after leaving definitions installed, the loader made zero transaction calls, and a coordinated reader observed definitions before provenance. The already-atomic definitions-assignment case was the single pass.

GREEN adds one registry transaction that, while holding the same registry lock used by `resolve()`, `definitions()`, and provider-record snapshots, validates both inputs, builds both replacement mappings, and publishes definitions plus provenance together. If either assignment mutates and then raises, base-level rollback restores the exact original mapping objects for both collections before the loader sanitizes the failure as deterministic `KNOWLEDGE_INVALID` commit provenance. The loader invokes exactly one transaction after preflight. Existing `register()` and `register_many()` behavior is unchanged.

Focused tests cover failure immediately after definitions assignment and after provenance assignment, pre-populated state preservation, exact private mapping identity restoration, deterministic multi-provider publication, a coordinated concurrent reader, sanitized error details, and legacy registry behavior.

```text
new atomic transaction regressions (GREEN)
4 passed, 2 warnings in 0.22s

../.venv/bin/python -m pytest -q tests/knowledge/test_provider_loading.py tests/knowledge/test_registry.py
58 passed, 2 warnings in 0.28s

../.venv/bin/python -m pytest -q tests/knowledge/test_envelope.py tests/knowledge/test_provider_loading.py tests/knowledge/test_registry.py
71 passed, 2 warnings in 0.42s
```

### Exception-wave verification

```text
../.venv/bin/python -m pytest -q tests/knowledge
274 passed, 2 warnings in 2.62s

../.venv/bin/python -m pytest -q tests/test_v04_compatibility.py tests/test_campaign_smoke.py tests/test_mixed_space.py
101 passed, 2 skipped, 2 warnings in 10.89s

../.venv/bin/python -m pytest -q
689 passed, 4 skipped, 2 warnings in 39.55s

../.venv/bin/python -m compileall -q src
exit 0; no output

git diff --check
exit 0; no output

git diff --name-only | rg 'progress\.md$'
exit 1; no matches (ledger untouched)

git grep -n -i -E 'MI-6|pyDOE3' -- ':!docs/superpowers/**' ':!.superpowers/**'
exit 1; no shipped-source/dependency-input matches
```

The final source state was rebuilt offline through the declared setuptools backend, without build isolation:

```text
../.venv/bin/python -c 'from setuptools.build_meta import build_sdist, build_wheel; print(build_wheel("dist")); print(build_sdist("dist"))'
expdoe_dk-0.5.0-py3-none-any.whl
expdoe_dk-0.5.0.tar.gz
exit 0

../.venv/bin/python -m pytest -q --run-slow tests/test_release_artifacts.py
2 passed in 0.27s
```

Final exception-wave artifact SHA-256 values:

```text
1e076eb92c77fe8b605a39d1aab2a364ce48bda5335d008adc63887884e34af9  expdoe_dk-0.5.0-py3-none-any.whl
23016284fb57e88bf96733c237fdc2bca09401a39462297fb0240c689a700989  expdoe_dk-0.5.0.tar.gz
```

The archive gate rechecked wheel/sdist contents, metadata, NOTICE, requirements, expected files, and prohibited/stale text. The rebuilt wheel was then installed with `--no-deps --no-index` into a fresh target. Provider entry-point enumeration was replaced by an assertion-raising sentinel before constructing `Knowledge()` and `PatternRegistry()`:

```text
0.5.0 0.5.0 0.5.0 1.0 2.0 /private/tmp/expdoe-exception-wheel.tH1eYm/expdoe_dk/__init__.py
```

The sentinel was never called. The values are package version, installed distribution version, engine version, knowledge schema version, and checkpoint schema version.

### Exception-wave files and self-review

- Authoritative audit state: `knowledge/__init__.py`, `tests/knowledge/test_envelope.py`
- Atomic registry publication: `knowledge/registry/registry.py`, `knowledge/registry/providers.py`, `tests/knowledge/test_provider_loading.py`
- Evidence: this report only; the progress ledger remains untouched

Self-review confirmed that restored provenance never accepts current registry replacement; failed verification is non-mutating; fresh knowledge retains the explicit-load snapshot behavior; transaction validation and both mapping builds precede publication; all readers use the publication lock; rollback bypasses hostile assignment hooks and restores both original object identities; loader errors remain sanitized and deterministic; and legacy registry APIs retain their previous semantics. No known concerns remain.

Exception-wave commit — the commit containing this section; its hash is included in the final handoff.
