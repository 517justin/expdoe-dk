# Multiobjective Workflows (0.6.0)

Named multiobjective recommendations and opt-in monotone v2 pass the fixed
current/lower quality protocol. V2 ties Plain on this fixture, not outperforms it.
The [acceptance evidence](../experiments/multiobjective_validation/README.md)
keeps the historical v1 failure and earlier numerical hold separate.
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
not posterior predictions as observations, and explicitly selects monotone v2
for gain only. Examples 03 and 04 remain scalar.

## Versioned Monotone Evidence

`with_monotone` retains its v1 defaults, pseudo-response implementation and
stored version. It does not upgrade old declarations. Use `with_monotone_v2`
for an explicit scalar opt-in, or name the version and objective scope:

```python
import expdoe_dk as ed

knowledge = ed.Knowledge().add(ed.make_pattern_spec(
    pattern_id="gain-increases", pattern="monotone", version="2.0",
    scope=ed.KnowledgeScope(factors=("x",), objectives=("gain",)),
    confidence=.8,
    parameters={"direction": "increasing", "n_pairs_per_dim": 3,
                "epsilon": .05, "softness": 1.},
))
```

V2 adds a tempered-probit factor on finite directional differences, then uses a
Laplace approximation with full joint-query covariance. It creates no response
rows or Yvar entries and does not refit the base hyperparameters during
conditioning. Confidence is a fractional evidence weight, **not** a calibrated
probability or measurement variance. Zero confidence is inactive and preserves
the ordinary fit and RNG behavior. Wrong knowledge can hurt results; marginal
measured-trend diagnostics are not causal proof and never rewrite observations.

Pair count per dimension is 1-32, epsilon is strictly between 0 and .5, softness
is positive and confidence is in [0,1]. Limits are 64 active pairs per output and
4096 generated centers per spec. Unsupported composition, duplicated direction
declarations, impossible feasible pairs, invalid covariance and nonconvergence
fail explicitly. V1 virtual-response and v2 declarations cannot be combined on
the same output, including inactive declarations. Historical specs retain exact
pattern versions.

Ask diagnostics expose `result.diagnostics["directional"]` for scalar results
and `result.diagnostics["objectives"][name]["directional"]` for multiobjective
results. They separate declared from fitted noise, actual measurements from
directional evidence, accepted pair geometry and the approximate solver state.
Checkpoint restore refits when needed; it does not deserialize a historical GP
pickle or claim that report-only restoration reran a fit.

### Numerical Scope and Cost

The supported active-v2 exact-GP source computes the same frozen GP moments
using bounded multiprecision construction and checked physical transforms.
It does not change the fitted prior/noise, PSD/rank tolerances, seeds or search
policy. Its error checks are engineering estimates, not rigorous interval
certificates. Original Torch feature/mean operations and the downstream
Laplace/root/sampling stages retain their own numerical boundaries. First and
second reverse-mode query derivatives are tested; third/higher derivatives,
forward AD, torch.func/vmap and torch.compile are not supported claims.

**V2 is opt-in and can be slow; no interactive-latency guarantee is made.**
Bounded source-adapter observations for n=32, d=6, q=2, 128 endpoints and 32
t-batches took about 131-133s for forward plus backward, excluding 35-37s setup.
With six endpoints, forward plus backward took about 29-30s. These runs had
timing-overlap and interim guard-refinement caveats; they are not full
recommendation/optimizer-throughput benchmarks. Large catalogs, pair sets and
repeated optimizer evaluations can be costly. No hardware SLA or permission to
reduce numerical/search fidelity follows from the fixed quality result.

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
Ordinary zero-Yvar GPyTorch likelihoods clamp fitted normalized noise to 1e-6;
declared noiseless mode is not a claim of exact zero fitted likelihood noise.

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

The release writer is 0.6.0. Knowledge and scalar schema 2.0/2.1 readers accept
the exact reviewed engine versions 0.5.0, 0.5.1 and 0.6.0. Standalone versioned
Space reads accept exactly 0.5.1 and 0.6.0, while 0.5.0 Space normalization is
owned by Campaign restoration. Readers leave historical input payloads unchanged;
writers emit 0.6.0. Schema 2.2 accepts **only** release writer 0.6.0 with
matching nested envelopes. The retained 0.5.1 prerelease schema-2.2 fixture is
not silently relabelled or accepted as a released archive. Never assume all
0.x payloads are compatible. Historical schema 2.0/2.1 fixtures stay historical.

The resolver-clean lower environment is Python 3.11.15, Ax 1.2.4, BoTorch 0.17.2,
GPyTorch 1.15.2, Torch 2.2.0, NumPy 1.24.4, SciPy 1.12.0 and pandas 2.0.3.
Ax 1.2.4 itself requires Python >=3.11, now the package's declared floor.
Both tested stacks use mpmath 1.3.0 (declared >=1.3,<1.4).
Current verification uses Python 3.12.13, Ax
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
