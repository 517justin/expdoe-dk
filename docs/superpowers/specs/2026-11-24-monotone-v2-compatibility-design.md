# Monotone v2: soft directional knowledge with legacy compatibility

Date: 2026-11-24
Status: design proposal for approval; direction selected, implementation NOT authorized by this document yet.
Builder check: independent planning review completed; no material unanswered planning questions identified. This is not implementation, mathematical proof, or acceptance evidence.
Base: codex/multiobjective-v060 at 46a0b376dbc058867a60392789edda6a29f8cfb5.

## 1. Intent and approved direction

The user selected a new knowledge-processing method while retaining legacy compatibility.
The purpose is to represent increasing/decreasing knowledge without inventing exact
response values, then resume the blocked 0.6.0 acceptance work. This is a model
repair project, not a benchmark replacement or a small noise-floor adjustment.

User-approved decisions:

- New directional knowledge processing and legacy compatibility must coexist.
- Existing failed acceptance evidence remains; original seeds, budget, reference,
  objective functions, and quality tolerances are not relaxed.
- README EN/ZH and 0.6.0 release updates remain conditional on acceptance.

The detailed API, mathematical approximation, defaults, and initial support limits
below are the author's proposals, not decisions already approved by the user.
No merge, push, tag, or package publication is included.

## 2. Evidence and current code

The fixed benchmark has mean normalized HV regrets:
random 0.1059641197, plain GP 0.0386323400, legacy knowledge GP 0.2500934398.
Both tested dependency stacks produced the same trajectories. An independent
audit recomputed all 30 saved results within 8.9e-16.

The inherited monotone augmenter places absolute pseudo-responses around the
global training mean. Individual pairs have the right direction, but their levels
can contradict genuine measured ordering. Known variance is extended to these
rows using the mean real variance, including zero for noiseless observations.
An inferred-noise control remains inaccurate; a positive noise floor alone does
not address the invented absolute response levels.

Relevant implementation boundaries:

- knowledge/patterns/physics.py registers monotone 1.0.
- Knowledge.with_monotone builds exactly that legacy version.
- knowledge/registry resolves exact (pattern, version) keys.
- bo/surrogates.py fits projected scalar GPs and exposes joint physical posteriors.
- bo/checkpoint.py stores the exact Knowledge envelope and rebuilds models.

Detailed retained evidence is in the existing multiobjective SDD workspace:
task-10-report.md, task-10-acceptance-review.md, task-10-root-cause.md.
The original acceptance README remains a historical failed baseline until updated
with separately identified new-policy results; it must not be overwritten as if
the old method had passed.

## 3. Approaches and selection

| Approach | Benefit | Cost / reason |
| --- | --- | --- |
| Increase pseudo-response noise only | Small code change | Retains false absolute-level assumptions; diagnosis does not support it as the quality repair. Rejected. |
| Replace the shared legacy algorithm | One implementation for all calls | Changes existing scalar numerical behavior and historical continuation. Not selected. |
| Add versioned soft directional inference | Encodes the supplied information and retains legacy semantics | New posterior adapter and numerical validation required. User selected this compatibility direction. |

Within the selected direction, propose finite differences plus a Gaussian Laplace
approximation rather than a derivative-kernel rewrite. It works through the
existing GP joint posterior, supports stepped numeric factors, and avoids
requiring differentiability of every kernel/feature transform.
It is an approximation and does not guarantee global monotonicity or better BO.

## 4. Versioned declaration and opt-in

Register pattern="monotone", version="2.0" alongside version="1.0".
Do not redirect old definitions, add an implicit latest-version resolution, or
change the existing with_monotone signature/default/version.

Proposed new convenience method:

```python
Knowledge.with_monotone_v2(
    param, effect, *,
    objective=None,
    n_pairs_per_dim=5,
    epsilon=0.05,
    softness=1.0,
    confidence=1.0,
)
```

The helper emits an ordinary immutable spec with all defaults materialized.
The existing make_pattern_spec API remains the full declaration interface,
including scopes spanning multiple named objectives.

Exact v2 parameters: direction, n_pairs_per_dim, epsilon, softness.
Confidence stays the existing top-level spec field. No delta_norm field:
the new method specifies neither response levels nor a required response increment.
No epsilon="auto" in v2: the legacy lengthscale-prior heuristic is not silently
carried into the new inference method.

Validation:

- direction is increasing or decreasing; one scoped numeric factor.
- n_pairs_per_dim is an integer in [1, 32], excluding bool.
- epsilon is finite and 0 < epsilon < 0.5; softness is finite and > 0.
- confidence is finite in [0, 1], excluding bool.
- Multiobjective response scopes are explicit and retain original pattern IDs.
- Scoped factor kinds: bounded continuous, integer, discrete numeric.
  Nominal/ordinal direction semantics are not introduced here.
- Other factors may retain existing categorical/ordinal/numeric encodings.
- Simultaneous legacy virtual-response artifacts and v2 on the SAME output
  reject with CONFIG_INVALID; unrelated outputs may use different versions.
- A repeated v2 declaration for the same objective/factor rejects explicitly,
  rather than silently multiplying identical evidence. Opposite directions
  on that objective/factor are an explicit composition error.
- Different factors on one output are supported and conditioned jointly.
- Zero confidence is an explicit inactive factor, reported as such; the base GP
  and its recommendations must be unchanged, not silently labeled knowledge-applied.

The initial v2 conditioning budget is at most 64 accepted pairs per output across
all active v2 patterns. Exceeding it is CONFIG_INVALID, never silent truncation.
These limits bound dense covariance work; they are not claims about maximum
possible scientific model size.

## 5. Meaning of the new knowledge

Fit a base GP on actual successful measurements using the existing physical
normalization, variance handling, registered means/kernels/priors and deterministic
seed. V2 contributes NO fabricated function-value rows and NO Yvar entries.
Keep the base GP hyperparameters fixed during the directional conditioning step.
This two-stage approximation avoids fitting the same real observations twice.

For pair j changing only the declared factor, let:

```text
D_j = f(x_plus_j) - f(x_minus_j)
h_j = model_coordinate(x_plus_j) - model_coordinate(x_minus_j) > 0
t_j = sign_j * D_j / (softness * h_j)
```

f is the existing standardized scalar GP response; sign_j is translated once
from physical direction into its internal response frame. h is the actual
post-snap difference in that factor's Space model coordinates, not expanded
kernel features. For log inputs, it is consequently a log-coordinate difference.
All other coordinates are identical between the two endpoints.

The directional log factor for a spec with n accepted pairs is:

```text
(confidence / n) * sum_j log Phi(t_j)
```

Phi is the standard normal CDF. Confidence is a fractional evidence weight,
NOT a calibrated probability that the rule is true and NOT a measurement variance.
Softness is a scale for uncertain slope in standardized-response/model-coordinate
units. Larger softness weakens a directional preference at a given slope.
Confidence=1 still describes a soft rule, not a hard constraint.
Dividing by n prevents requesting more locations from multiplying a spec's
total evidence weight solely through discretization.

Author-proposed default softness=1 means a slope scale of one observed response
standard deviation per full model-coordinate span. It is an interpretable,
predeclared modeling policy, NOT a parameter estimated from the acceptance oracle.
It is not uniquely scientifically optimal. Its suitability remains an acceptance
question. Default changes after approval require a documented policy revision,
not an unreported sweep to pass the fixture.

Positive affine response-unit equivariance is required for nonconstant measured
responses, with the existing standardization. Constant-response data retain the
existing unit-scale fallback and must be explicitly tested/documented rather than
claiming unrestricted unit equivariance in that degenerate case.

## 6. Pair locations and hard parameter feasibility

Locations are derived only from declared parameter domains, known hard parameter
constraints, seed, and actual observed input coordinates. No oracle response,
predicted safety label, or unmeasured outcome is used to select locations.

Use a deterministic, versioned generator shared by scalar and multiobjective
paths, ordered by stable pattern identity. It may reuse existing domain sampling,
snapping and feasibility helpers; do not modify the legacy augmenter's RNG.

Proposed algorithm:

1. Attempt centers from the successful observed X rows in stable input order.
2. Add seeded Sobol centers from the full declared domain when more are needed.
3. Form plus/minus epsilon endpoints in the scoped model coordinate, bounded
   within its range, then apply existing physical snapping/roundtrip rules.
4. Reject collapsed pairs, changes to another coordinate, duplicate pairs, or
   either endpoint violating known hard parameter constraints.
5. Keep the first requested number of distinct valid pairs. Use at most 4096
   generated centers per spec. If insufficient, raise CONFIG_INVALID with
   reason=no_feasible_directional_pairs, requested/accepted counts and pattern ID.

Continuous domain constraints include global Knowledge parameter constraints,
not only Space constraints. The full runtime's parameter-feasibility context
must reach projected scalar fitting; response-dependent/global constraints must
not be blended into the other objective's likelihood.

Finite catalog membership is not a scientific definition of the latent domain.
Pairs may lie off the recommendation catalog but must remain in the declared
physical domain and obey known parameter constraints. Only real recommendations
must belong to that catalog. State this distinction in diagnostics/documentation.
No outcome-safety guarantee is inferred for virtual locations.

Equality restrictions may make changing one factor with all others fixed
impossible. Fail explicitly; do not move a second factor and claim a partial
monotonic effect. No hidden reduction in requested pairs or dropping of knowledge.

## 7. Inference and posterior contract

Evaluate the base GP's JOINT latent posterior at all pair endpoints. Construct
the difference mean m and covariance K, including cross-pair dependence.
Independent marginal variances are insufficient and forbidden here.

Use a whitened representation D = m + B u with u ~ N(0, I), where B is obtained
from a symmetric PSD factorization of K. Solve the strictly convex negative
log posterior in u, then use its Hessian at the mode for a Gaussian Laplace
approximation. Optimize with a proven SciPy optimizer and Torch differentiation;
reuse stable log-CDF/numerical primitives from the verified stack. Do not
hand-write a new nonlinear optimizer.

For query values f_X, define A = Cov(f_X, D) U Lambda^(-1/2) on the retained
positive eigenspace K = U Lambda U^T. If u_hat is the mode and H its Hessian:

```text
mean_v2(X) = mean_base(X) + A u_hat
cov_v2(X)  = cov_base(X) - A A^T + A H^(-1) A^T
```

This includes covariance between every query point; it is not a pointwise mean
correction. Use linear solves/factorizations, not explicit dense inverses.
Numerical tolerances, convergence status, covariance rank and conditioning
diagnostics must be explicit and covered by tests. Singular redundant directions
are handled through rank-aware factorization, not invented observation noise.
Material negative eigenvalues, nonfinite values, inconsistent nullspace cross
covariance, or nonconvergence raise MODEL_FIT_FAILED.

The implementer may choose the supported SciPy solver and tested numerical
tolerances, documenting them before running quality acceptance. These are
implementation choices, not permission to change the likelihood, use score
clipping that changes its gradients, or tune quality thresholds.

Return a BoTorch-compatible Gaussian posterior adapter, wrapped through the
existing physical-response adapter. Requirements:

- predict and acquisition see the SAME conditioned posterior.
- Full joint-q and batch shapes, differentiable query inputs and reparameterized
  samples work for qLogEHVI, qLogNEHVI, and qLogNParEGO.
- No extra physical safety probability factor per directional rule.
- Independent response models remain independent; conditioning does not introduce
  cross-output correlations.
- Real noiseless/known/inferred variance metadata remains unchanged.
- Never fall back to the old algorithm, plain GP, or random recommendations on
  failure. Return the existing typed capability/model-fit error.

The current direct PairwiseProbitLikelihood is not a drop-in measured+directional
model. Its inspected implementation fixes a noise scale and clamps scores; do
not substitute it without checking the exact desired objective and derivatives.
Using vetted numerical primitives with a separately tested adapter is preferable
to pretending an incompatible ready-made model implements this contract.

## 8. Integration, diagnostics and persistence

Use the existing artifact envelope with a distinct kind, proposed
("virtual_observations", "monotone_sign_v2"). The category is retained to avoid
changing the public artifact JSON envelope; this kind means a directional factor,
not virtual measured function values. Its consumer must be separate from
build_virtual_observation_augmenter and advertised only after it is wired.

Explicitly register capability, runtime validation and composition checks.
Preserve legacy item views: v2 is not converted into _MonotoneItem, and old
epsilon auto-rescue must not mutate its parameters or reset its version.
Provide v2 validation diagnostics rather than claiming it was checked through
the legacy item adapter.

Diagnostics identify pattern ID/version, objective/factor, seed, requested and
accepted pairs, actual physical endpoints, direction, confidence/evidence weight,
softness/units, actual-measurement count, base-noise mode, approximation policy,
solver convergence and covariance rank. No invented y values or measurement IDs.
Reports distinguish declaration, model conditioning and measured Pareto/HV.

Pattern version 2.0 binds these likelihood/default/generation semantics. Existing
exact-version Knowledge envelopes preserve them. New/default helper parameters
are stored explicitly, so restoration never consults a changed default.
Do not pickle fitted models; restore exact declarations and rebuild deterministically.

Schema 2.1 scalar and 2.2 multiobjective checkpoints continue to store Knowledge
envelopes. Validate whether existing envelopes suffice before adding any wire
fields; avoid an unnecessary schema bump. Unknown pattern versions remain errors.
Old archives/with_monotone/v1 specs restore into v1, never silently upgrade.
The unreleased interim 2.2/engine0.5.1 file remains development evidence; final
0.6.0 writer allowlists follow the already approved release plan.

## 9. Acceptance and legacy compatibility

Acceptance has four independent gates; functional passage alone is insufficient.

### A. Scientific/numerical invariants

- No v2 function-value pseudo rows; original Y/Yvar/status/IDs are immutable.
- Confidence=0 is exactly the base posterior and does not perturb RNG.
- Translation and nonconstant positive affine output-unit changes preserve
  physical conclusions; maximize/minimize flips retain physical direction.
- Shared-offset changes do not change a pair's directional factor.
- Full pair covariance, singular/repeated/near-deterministic pairs and numerical
  failure paths are tested; no fake diagonal independence.
- On a one-dimensional latent-difference example, compare mode and Laplace
  Hessian to independently derived formulas; compare approximate posterior
  moments to numerical quadrature with declared approximation tolerances.
- Correct/wrong knowledge and varying confidence have mechanism-level tests,
  not only an easy oracle fixture. A contradictory rule need not improve BO.
- Joint samples match implemented covariance; input gradients match finite
  differences, including batch q>1 and all three acquisition families.

### B. Existing behavior and persistence

- V1 scalar recommendations/JSON/RNG remain numerically identical in the SAME
  dependency environment; use independent legacy reference paths, not new goldens.
- V2 scalar and projected multiobjective use the same semantics and agree.
- Unscoped outputs and non-monotone patterns remain unchanged.
- Old literal archives remain readable under their original compatibility rules.
- V2 save/load produces the same next candidate IDs, weights, pair locations,
  declarations and report geometry; unknown versions fail closed.
- Known hard parameter restrictions, discrete snapping and catalog identity hold.

### C. Original fixed quality benchmark

Keep the existing legacy failure as an explicit historical baseline. Add a
separately named knowledge_gp_v2 method and a protocol identifier; do not relabel
knowledge_gp as if its old result changed.

Reuse seeds (2,5,11,23,47), initial6, eight rounds, q2, the exact oracle/catalog,
shared initial geometry, reference and explicit noiseless GP setting.
Plain and random baselines remain unchanged. V2 uses the same direction/scope,
confidence .8, three pairs, numeric epsilon .05, proposed softness1.
The semantic method change is disclosed; this is a revised knowledge protocol,
not an identical rerun of the legacy consumer.

Required comparisons:
plain <= 1.10 * max(random_mean_regret, 1e-12);
v2 <= 1.10 * max(random_mean_regret, 1e-12);
v2 <= 1.10 * max(plain_mean_regret, 1e-12).
The legacy arm remains reported as FAIL, not required to magically pass and not
hidden with xfail/seed removal. Replace the old release-gate assertion only after
the new protocol is explicitly approved; retain its failed evidence and a legacy
mechanism regression. Do not leave a known failing legacy gate in the required
release suite and then claim the complete suite passed.

Record source SHA/dirty state, all four method identities where compared,
per-seed results, warnings, actual measurements, feasibility1, duplicates0,
q completeness, replay and restart equality.
No oracle-guided tuning of softness, confidence, epsilon, seeds or solver settings.
If the v2 gate fails, report it and retain the release hold.

### D. Release

Repeat current and resolver-consistent lower-stack full --run-slow suites,
legacy capability probes, examples and artifact checks. Capability-only old
BoTorch overlay must not be called a supported package installation.
Only then finish the original Task10: roadmap, root EN/ZH and package README,
Python floor correction, exact engine/schema compatibility tests, 0.6.0 metadata,
post-bump full validation, wheel/sdist and fresh offline-wheel examples.
Task/whole-branch independent review remains required.

## 10. Non-goals and risks

No global monotonicity guarantee, hard monotone GP, derivative-kernel family,
correlated-output model, automatic confidence calibration, unrestricted factor
kinds, all-provider compatibility, benchmark-specific analytic shortcut, or
claim that knowledge always improves recommendations.

The main risks are approximation quality, conditioning of near-deterministic
differences, joint posterior performance, and the usability of explicit opt-in.
The baseline GP is fitted before knowledge, so v2 does not jointly optimize kernel
hyperparameters with directional factors; describe this limitation rather than
implying exact Bayesian inference.

Builder-owned choices: small-module layout, numerical solver/tolerances,
reuse of existing sampling/constraint helpers, test organization and diagnostic
presentation. These choices cannot change the declared scientific policy,
backward-compatibility rule, or quality gate without a documented design revision.

## 11. Sources and attribution

- Riihimaki and Vehtari (2010), Gaussian processes with monotonicity information:
  https://proceedings.mlr.press/v9/riihimaki10a.html
  Supports derivative-sign knowledge with approximate inference. The paper uses
  expectation propagation; this proposed finite-difference, tempered Laplace
  conditioning policy is OUR design, not a claimed reproduction of that paper.
- BoTorch Model/posterior interfaces:
  https://botorch.readthedocs.io/en/stable/models.html
  Joint query shape and posterior interface reference. Dependency-specific
  compatibility must be established through local tests, not documentation alone.
- Local PairwiseProbitLikelihood source inspected in the prepared environment
  on 2026-11-24; it is not adopted wholesale.

## 12. Approval checkpoint

User has selected versioned new knowledge with legacy compatibility.
This full proposal still requires approval, particularly:

1. Explicit opt-in v2 while old with_monotone stays v1.
2. Two-stage soft directional Laplace approximation and confidence semantics.
3. Declared support limits and errors instead of silently ignoring knowledge.
4. Separate v2 acceptance protocol retaining the legacy failed baseline.

After builder review and user approval, write an implementation plan and execute
it with the previously requested subagent-driven workflow in the existing worktree.
