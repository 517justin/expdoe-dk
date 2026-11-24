# Multiobjective Workflows (Unreleased)

This branch implements named multiobjective recommendations, but **0.6.0 is
blocked by the fixed-seed knowledge-quality gate**. The package remains 0.5.1.
See the [acceptance evidence](../experiments/multiobjective_validation/README.md).
Functional coverage is not evidence that knowledge improves every experiment.

## Measurements and Recommendations

Declare two or more `Objective` values in `Space`. Use a DataFrame with exactly
the objective names in `Campaign.tell(X, Y)` or `ObservationBatch(X, Y, Yvar=...)`.
Columns are aligned by name, not position. At least three successful complete
measurements are required before recommendation. Failure/warning rows remain
in history but do not train the models. Partial objective returns and mixed
known/missing variances across successful batches are unsupported.

Each objective gets an independent physical-response GP. Predictions retain
physical units and shape `(n, m)`. There is no cross-output correlation model,
first-objective shortcut, reference-estimator fallback, or random fallback for
failed fitting. Knowledge affecting responses requires explicit objective scope.
For example, a monotone `PatternSpec` uses
`KnowledgeScope(factors=("x",), objectives=("gain",))`; an unscoped response
assumption is rejected rather than applied to every objective.

The [standalone catalog example](../examples/05_multiobjective_catalog.py) shows
named returns, stable recipe IDs, distinct measurement IDs, pending trials,
checkpoint restoration and measured Pareto reporting. It uses a teaching oracle,
not posterior predictions as observations. Examples 03 and 04 remain scalar.

## Routing and Noise

Pass `multiobjective=MultiObjectiveConfig(...)` as a keyword to `Campaign` or
`recommend`. Do not pass multiobjective configuration to a scalar space.

| Configuration | Automatic route |
| --- | --- |
| 2-4 objectives, explicit noiseless, no pending or preferences | qLogEHVI |
| 2-4 objectives, inferred/known noise or pending, no preferences | qLogNEHVI |
| 5+ objectives, or any explicit preference | qLogNParEGO-style |

`noise_mode="auto"` means known when complete Yvar exists, inferred otherwise.
Missing Yvar never means noiseless. Explicit noiseless mode accepts only zero
recorded variances; the GP's numerical noise floor is not fabricated measurement
metadata. Known mode requires complete variances; inferred rejects supplied ones.
Incompatible explicit acquisitions raise a typed error.

Positive preference weights must cover every objective. Config, knowledge and
unequal `Objective.priority` declarations must agree after normalization;
conflicting sources reject. Unequal priorities map to normalized
`1 / (1 + priority)`, not lexicographic optimization. With no preferences,
NParEGO weights are reproducible per iteration and greedy slot. A knowledge
target range is a soft scoped penalty, never permission to violate hard limits.

## Geometry, Reports and Safety

The first successful ask freezes utility location/range from initial successful
measurements, never unmeasured catalog outcomes. Constant outputs have a finite
fallback scale. An explicit reference is a mapping of physical values; at least
one measured feasible baseline must strictly dominate it in every objective.
Without one, reference utilities are derived from feasible measured minima
minus 0.1 scale. Report and acquisition references are named separately; soft
knowledge penalties do not alter report HV.

Pareto rows and hypervolume use successful, hard-feasible measurements only.
Duplicate utility vectors contribute to HV once, while trial audit rows remain.
Observed noisy HV is not latent truth. Scalar best fields are null for multiple
objectives. Five sufficiently small comparable HV updates produce an advisory,
not an automatic stop. Before first ask, report-only geometry may change between
finalize calls and must not be compared as one fixed history. In particular,
mixed supplied/missing Yvar across successful batches before first ask raises
`CONFIG_INVALID`; not every archivable input combination is fit/reportable.
After geometry is frozen, such later variance mixtures retain measured metrics
but expose routing unavailability.

Hard parameter constraints, catalog membership and measured/pending exclusions
are enforced at the physical candidate boundary. Exactly q unique feasible
candidates are returned or a typed error is raised. Supported outcome safety
uses named physical `<=`/`>=` intervals and model-based joint probability under
independent outputs, default threshold 0.95. **This is not a measurement safety
guarantee.** Soft/equality outcome constraints and nonlinear cross-output safety
are not supported. Numeric optimization failures are not reported as proof that
the feasible domain is empty.

## Persistence and Environment

Scalar campaigns continue to write schema 2.1. Multiobjective schema 2.2 records
all objective values/variances, metadata, pending rows, fixed geometry, config,
catalog/IDs and iteration state. A catalog binds after the first successful ask;
later equivalent reordering is allowed, replacement is not. New preferences or
reference values require a new campaign. Historical schema 2.0 multiobjective
archives missing objectives are archival-only; no data are imputed.

The pre-release writer is still 0.5.1; 0.6.0 writer compatibility is not yet
enabled. Never assume all 0.x payloads are compatible. Exact writer allow-lists
and schema validators govern restoration.

The resolver-clean lower environment is Python 3.11.15, Ax 1.2.4, BoTorch 0.17.2,
GPyTorch 1.15.2, Torch 2.2.0, NumPy 1.24.4, SciPy 1.12.0 and pandas 2.0.3.
Ax 1.2.4 itself requires Python >=3.11; the old project's Python >=3.10 metadata
is pending release correction. Current verification uses Python 3.12.13, Ax
1.3.1, BoTorch 0.18.1, Torch 2.12.1 and pandas 3.0.3. The separate BoTorch 0.11
overlay is capability-only and conflicts with installed Ax metadata; it is not
a supported dependency resolution. Missing acquisition/sampler APIs yield typed
capability errors without breaking scalar import. No direct BoTorch floor was
raised based on that overlay.

Python-backed StringDtype and nullable UInt64 checkpoint coverage exists in the
tested stacks. PyArrow is not installed and its optional dtype backend is
unverified. Torch 2.2 lacks uint64 tensors: that one capability-specific test is
skipped there, not on current Torch. Windows, Linux, CUDA, Colab, real textbook
datasets, arbitrary objective-count performance and global optimality are not
certified by this local macOS ARM64 acceptance.
