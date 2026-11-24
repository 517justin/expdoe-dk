# Stable GP Source Construction Addendum

Date: 2026-11-25
Status: controller-authorized implementation under the approved same-fixed-GP design, after independent mathematical review and two scoped design-fix rounds. This is not release acceptance or a user approval of numerical details.
Binding precedence: Section 14 supersedes Section 13 where stated; both supersede earlier provisional policy. Existing monotone-v2 constraints and relative rank policy remain unchanged. Task 6 release stays held until all original gates pass.

The following is the retained design and feasibility evidence, including its historical proposal wording.

# Task 6A: stable fixed-GP source construction

Status: **FEASIBLE, phased implementation proposal; not a production or release approval.**

This is the requested advisory design draft, published for controller review.
The bounded experiments establish feasibility for both saved failing fits and
for a real first/second-derivative bridge. They do NOT implement or verify the
entire production support adapter. No production files, tests, experiment
settings, fitted hyperparameters, Git state, or historical evidence were edited.
No fitting, full matrix, suite, subagent, installation, or download was run.

## 1. Decision requested

Authorize a small private exact-GP moment helper at
`bo/directional_posterior.py::_joint_latent_moments`, preferably implemented in
one private sibling module. Reconstruct the supported analytic kernel and solve
the SAME frozen Gaussian conditional in mpmath, then round the completed means,
covariances, and their analytic derivatives to Torch double. Retain the existing
Laplace optimizer, evidence, whitening/rank policy, joint posterior, and root
construction. Do not introduce a new user-facing model/backend choice.

The approved rank policy remains **64*n*eps*eigenscale**, unchanged. The controller
ruling expressly permits numerical nullspaces under this relative policy.
"Preserve tiny positive covariance" here prohibits new floors, blanket clipping,
or source-construction flattening; it does not promise retention of every
mathematical eigenmode below float64 resolution. No root/rank redesign is proposed.

Recommend explicit dependency `mpmath>=1.3,<1.4`, lazily imported only for an
active supported v2 source. Both existing runtimes have mpmath 1.3.0. The existing
offline wheelhouse contains `mpmath-1.3.0-py3-none-any.whl`. This makes an existing
transitive dependency explicit; it is still a packaging change requiring approval.

Phasing is engineering sequencing, NOT permission to narrow advertised v2
composition support silently:

1. Implement/review the numerical helper and frozen-fit regressions first.
2. Complete the bounded model/kernel/transform adapter and invalid-source guards
   below before wiring all production active-v2 routes or resuming Task 6.
3. Resume the unchanged acceptance work only after that reviewed repair passes.

If only a one-function covariance value patch is authorized, this proposal is
not sufficient: correct autograd, source identity, and caching require a private
state/adapter boundary. If the controller requires an already fully validated
general SingleTaskGP backend, that broader deliverable remains **not established**.

## 2. Evidence and diagnosis

Read: binding-context.md; task-6-report.md including diagnosis/high-precision
sections; both failure.json and before-failure.json snapshots; the original
high-precision script/logs; directional_posterior.py, gp.py, surrogates.py,
structured means/composition and relevant existing monotone-v2 tests.

Worktree HEAD verified as `79c62cbf11656d03d816e1e0895df6ad6b8a9eda`.
Frozen numerical artifacts were produced at
`80e7a1b5b2907cd5244d2dbe2760e4acd193212f`; the later commit is documentation only.

The rounded prior is already slightly indefinite. Float64 inverse replacement,
Cholesky subtraction, or block Cholesky of that rounded prior cannot recover
information lost when evaluating the kernel. Current source Cxx has minimum
eigenvalue -1.75446e-11, but the analytic fixed-parameter GP has +5.02876e-13.
The lower-stack frozen fit has the same mechanism. The proposal reconstructs the
kernel from coordinates and constrained parameters, not from the saved rounded
prior, and does not pretend that accepting a negative residual repairs it.

### New bounded results

Artifacts in this directory:

- `task-6a-probe.py`: diagnostic-only frozen Matern-5/2 source, analytic first and
  second derivatives, custom Torch reverse/double-backward bridge, existing
  directional adapter exercised without patching production functions.
- `task-6a-probe-current.log`: both saved fits evaluated on Torch 2.12.1, exit 0.
- `task-6a-probe-floor.log`: both saved fits evaluated on Torch 2.2.0, exit 0.
- `task-6a-cost-boundary-probe.py` and `.log`: endpoint-cache cost, isolated
  covariance derivatives, unchanged generic-provider adversarial checks; exit 0.

Candidate: 50 decimal digits, mpmath Cholesky of A followed by library solves.
Oracle: 80 digits, independently evaluated inverse-based conditional, matching
the original analytic oracle. Float inputs are imported exactly as binary64
values. Feature z-scores, response normalization, mean, fitted noise, and fitted
parameters are reconstructed from the snapshots; no optimizer is called.

| Quantity | Current frozen fit | Lower frozen fit |
| --- | ---: | ---: |
| Training n / joint query count | 10 / 16 | 8 / 14 |
| Max covariance error, 50-digit vs 80-digit, before cast | 2.4713e-46 | 2.0415e-46 |
| Max covariance error after float64 cast | 2.6338e-23 | 2.3517e-23 |
| Max mean error after cast | 1.0920e-16 | 1.0301e-16 |
| Cxx minimum eigenvalue, returned double | 5.0287595942e-13 | 6.0729325677e-12 |
| Existing Laplace fit rank | 3 | 3 |
| Final directional covariance minimum eigenvalue | 5.0287595939e-13 | 6.0729325676e-12 |
| Mixed mean/covariance VJP absolute error | 4.1324e-17 | 1.1110e-16 |
| Mixed directional second derivative absolute error | 1.3810e-21 | 1.4229e-21 |

Both current and lower runtimes passed the spike's gradcheck and gradgradcheck,
including repeated input coordinates. Joint batches had shape [2,2,1]; repeated
covariance rows were bitwise equal. The frozen source retained the repeated
endpoint identity. Fixed-base-sample loss gradients through the existing
directional posterior agreed with finite differences within 4.44e-11.

The supplementary covariance-only check prevents a large mean derivative from
hiding an inaccurate small covariance derivative: VJP oracle scale 2.28965e-7,
absolute error 1.59013e-23; second-directional derivative 4.66818e-7, error
3.21618e-23. Covariance scaled by 1e6 also passed gradcheck/gradgradcheck.

A separate analytic noiseless microprobe returned an exactly zero known row,
exact known mean, and a NONZERO cross-query derivative -0.0168157390. The
neighbor at distance 1e-6 retained variance 4.97808e-13. Thus no detached
value-only repair or zero-gradient mask was needed.

Representation limit, not a requested change: two close distinct queries in that
microprobe have true minimum eigenvalue 2.65121e-18; the existing relative policy
retains rank 1 rather than mathematical rank 2. This is explicitly allowed by
the controller ruling. Preserve and document this behavior.

These are feasibility microprobes, not regression tests or acceptance results.
The scalar probe does not implement the production kernel-tree adapter, full
model-batch support, registered-mean integration, or the eligibility guards.

## 3. Mathematical construction

In the source GP's fitted response frame, let training features be T, query plus
endpoint features be Z, and let the frozen likelihood covariance be diagonal N:

```
A = k(T,T) + N
B = k(Z,T)
D = k(Z,Z)
alpha = solve(A, y - m(T))
P = solve(A, B.T)
mu = m(Z) + B alpha
C = D - B P
```

Evaluate A, B, D, both solves, and the cancellation at multiprecision. Only then
convert mu and C to double. This is Gaussian conditioning, not a replacement
model. Cholesky requires A positive definite; it does not require C positive
definite. Singular query covariances and repeated endpoints remain admissible.

Use constrained `.lengthscale`, `.outputscale`, and `.noise` values, not raw
parameter assumptions. The probe's large raw positive values equal their double
softplus transforms; that shortcut must not become the production extraction API.
Preserve the fitted 1e-6 noise in these snapshots, not declared zero measurement
variance. Never add a solve jitter, reset a constraint, or refit parameters.

Matern-5/2 with ARD M=diag(ell^-2), delta=z-z', r=sqrt(delta.T M delta), a=sqrt(5):

```
k = s * (1 + a*r + 5*r*r/3) * exp(-a*r)
g = d k / d z = -(5*s/3) * exp(-a*r) * (1+a*r) * M delta
H = d2 k / d z2
  = -(5*s/3) * exp(-a*r) * ((1+a*r)*M - 5*(M delta)(M delta).T)
```

These expressions have their ordinary continuous limits at r=0: g=0 and
H=-(5*s/3)M. No division by distance or arbitrary near-zero radius is needed.
For RBF, k=s*exp(-r*r/2), g=-k*M delta,
H=k*((M delta)(M delta).T-M). A categorical-table product multiplies k,g,H
by its fixed category-pair coefficient; categories have no continuous derivative.

The supported production kernel grammar has at most one continuous stationary
factor, so a general symbolic kernel compiler is unnecessary.

## 4. Exact proposed support boundary

The repair path is for audited scalar, CPU-double **SingleTaskGP** sources,
including independent model batches, after fitting. It reads frozen parameters
and observations. No hyperparameter/training-data derivative API is promised:
those are explicitly fixed during directional inference. Query derivatives are
first and second order wherever the existing mean/transform is differentiable.

| Component | Repair adapter scope |
| --- | --- |
| Stationary covariance | Exact MaternKernel(nu=2.5) and RBFKernel; scalar or ARD lengthscales; active_dims and kernel/model batches |
| Scale | ScaleKernel, including nesting/order actually emitted by build_gp; read constrained outputscale |
| Local features | _FeatureKernel with _PhysicalFeatures, evaluated in Torch before MP kernel arithmetic; physical z-scores, log decoding, nominal one-hot, ordinal rank, large-integer origin logic remain unchanged |
| Categories | Existing _CategorySimilarityKernel and ProductKernel compositions emitted by build_kernel_from_artifacts/build_gp; tables read unchanged and category indices validated; with or without a continuous factor and outer scale |
| Means | ConstantMean/ZeroMean and existing registered ArrheniusMeanFrozen, QuadraticMeanFrozen, _PhysicalFormulaMean, CombinedMean, StructuredMean; execute original mean modules in Torch, do not translate their formulas into a second implementation |
| Priors | Existing GP-prior presets and mean MAP priors remain fitting-only; retain all fitted values and prior registration; no new optimization |
| Likelihood | GaussianLikelihood or FixedNoiseGaussianLikelihood with finite nonnegative diagonal fitted noise; the latter's public noise includes second_noise when present |
| Input transforms | None; eval-frozen Normalize and InputStandardize; Log10 and chains of these with validated shapes; use existing Torch transform methods |
| Outcome transforms | None; scalar trained Standardize and affine chains of supported standardizations; use existing untransform_posterior on a lazy dense Gaussian without requesting its default variance floor |
| Outer physical response | Existing FittedSurrogate/_PhysicalAcquisitionModel sign, location, scale unchanged; covariance scales by scale squared and samples by signed scale |

Local production compositions are required before this repair is accepted as
general active-v2 production wiring. In particular, the existing integration
test combines power_law OR saturation, category similarity, and GP priors; a
Matern-only frozen-fixture patch does not fulfill that obligation.

Do not accept arbitrary user kernel subclasses, custom forward/posterior hooks,
warps, non-Gaussian likelihoods, correlated training noise, multitask/fantasy
semantics, sparse/variational sources, nonlinear outcome transforms, GPU/float32,
or other Matern smoothness values as if covered by this helper.

Unsupported providers retain their existing strictly validated moment-provider
route when selected BEFORE evaluation; they do not receive this repair. A failed
repair must never fall through to that route, or to plain/random/legacy models.
If the public composition contract requires an unlisted source, stop for a scope
decision instead of quietly declaring it supported.

Training A must be SPD. Exact redundant noiseless training rows or other
singular A are explicit unsupported/invalid-source failures in this bounded
proposal; no pseudoinverse, rank-threshold training reduction, or noise is added.
This does not exclude singular QUERY covariances or zero-noise observations with
nonsingular A. Exotic deterministic relations not certified below may fail closed.

## 5. Private API and transforms

Suggested internal state: `_FrozenExactGPSource`, owned by the directional
adapter, with one state per base model batch. Suggested calls:
`prepare(source, endpoints)` and `joint_moments(X) -> (mean, covariance)`.
No public signature or model-selection setting changes.

Create it only after the existing empty/zero-active-evidence returns. Confidence
zero therefore returns the IDENTICAL model without evaluation, mpmath import,
cache creation, parameter touching, or RNG use. Legacy-only paths never enter it.

Call public `model.eval()` to establish BoTorch's eval training inputs. Training
`train_inputs[0]` is then already input-transformed; do not transform it twice.
Apply `model.transform_inputs(X)` once to queries. The mean uses these coordinates;
_FeatureKernel performs its separate feature expansion for the covariance.
Preserve active_dims ordering: a leaf's selection follows the same transform
placement as its actual kernel tree. Category lookup remains in its own original
model-coordinate branch, not in an accidentally expanded one-hot branch.

Use explicit shape broadcasting to model.batch_shape and query t-batches;
evaluate independent MP blocks and let Torch expand/reduce implement the chain
rule. Do not fuse different output GPs or base batches. Frozen endpoints permit
cached moments/cross-solves, not detaching differentiable query coordinates.

Mean boundary: pass the original Torch query mean m(Z) as an additional tensor
input to the MP moment Function, and freeze m(T) evaluated once at training
coordinates. Compute m(Z)+B alpha in MP; backward returns the upstream mean
cotangent unchanged to m(Z), plus the kernel derivative to its feature inputs.
This preserves registered mean and nonlinear input-transform chain rules,
including second derivatives, without hand-writing those means' gradients.

Retain outcome untransformation and outer physical transformations exactly once.
Return full covariance, not diagonal-only or separately repaired marginal/cross
blocks. Leave directional pair differencing and downstream PSD checks unchanged.

## 6. Differentiation contract

Let b_i=k(z_i,T), P_j=A^-1 b_j.T, d_i=grad b_i, e_i=Hessian b_i.
For fixed training state and a constant source mean (add original mean chain rule
as above for general means), define:

```
u_i = d_i alpha
J_ij = grad_1 k(z_i,z_j) - d_i P_j
U_ij = Hessian_11 k(z_i,z_j) - e_i P_j
V_ij = Hessian_12 k(z_i,z_j) - d_i A^-1 d_j.T
Hessian_12 k = -Hessian_11 k  # stationary continuous leaf
```

These differences must ALSO be formed in MP, not by differentiating the inaccurate
float64 Schur subtraction. For incoming mean cotangent h and covariance cotangent G:

```
grad_z_i = h_i*u_i + sum_j (G_ij + G_ji)*J_ij
```

Double backward in query direction v, holding h/G fixed, uses:

```
HVP_i = h_i*(e_i alpha)*v_i
      + sum_j (G_ij+G_ji)*(U_ij v_i + V_ij v_j)
```

It must ALSO return derivatives with respect to h and G:
`dh_i = dot(u_i,v_i)` and
`dG_ij = dot(J_ij,v_i)+dot(J_ji,v_j)`.
The spike implements this explicit second custom Function; it is not a detached
Jacobian advertised as supporting double backward. Kernel derivatives are
analytic and checked against independent MP finite differences of the forward.
Finite differences are an ORACLE only, not the proposed production backward.

Known noiseless query rows: certify using exact transformed equality with a
training observation whose TOTAL fitted noise is zero; then set covariance row
and column to exact zero and its mean to the known target. Zero-prior rows of a
validated categorical kernel are another algebraic identity. Apply these to
VALUES inside the MP Function, not by masking its backward. A derivative cross
covariance at a known point need not be zero, as the probe demonstrates.

Repeated points can share value computations but must retain separate query
arguments for derivative assembly. Do not use unique/scatter in a way that makes
the second occurrence's derivative belong to the first coordinate.

Sampling uses the existing differentiable root after rounding, with existing
rank policy. Full-rank fixed-base-sample gradients were probed. Exact rank changes
retain the existing policy's local/branch semantics; this design does not promise
a globally smooth sample factor across a change of retained rank. Third and
higher derivatives, torch.func/vmap, forward-mode AD, and torch.compile are not
established by this spike and must not be silently advertised.

## 7. Error policy and adversarial sources

Use private MP contexts, never process-global mp.dps. Suggested initial precision
policy: compare complete 50- and 80-digit constructions, including required
derivative residuals; increase to a bounded 110/160-digit pair if unresolved;
otherwise raise a typed numerical failure. Endpoint/training caches are keyed
by precision and frozen model identity. The cap is an engineering proposal, not
a model parameter or PSD threshold. It requires implementation review.

Require convergence to binary64 accuracy, checking component/block-scaled
differences without a max(1, covariance_scale) unit floor; inspect solve residuals
at the higher precision. Known-zero identities are exact, not threshold-derived.
If the model implies a positive diagonal but computation rounds it to zero at
working precision, do not certify that as a noiseless identity: increase precision
or fail. Nonfinite, inconsistent shapes, underflow of a meaningful returned
variance, invalid kernel parameters, negative noise, or unresolved precision fail
explicitly. No eigenvalue repair is permitted. Existing `_psd` remains the final
double-precision acceptance/rank authority.

Two-precision agreement is a practical numerical validation, NOT a rigorous
interval certificate for every possible extreme configuration. This design uses
standard arbitrary-precision arithmetic, not an unimplemented certification claim.
mpmath interval arithmetic is not proposed as a second backend.

Source identity is crucial: reconstructing parameters while ignoring a custom
posterior could silently replace an invalid supplied model with a valid one.
Use exact audited class/tree/callable identities, not just isinstance(SingleTaskGP).
Modified/subclassed providers must stay on the generic validated route selected
before preparation, or fail typed unsupported-source. Recheck identity and frozen
parameter/buffer versions before using cached state; mutation after preparation
fails, never silently changes evaluation route. Do not compare raw and repaired
covariances with a broad prior-scale budget to decide that corruption is benign.
Such a comparison cannot distinguish a real 1e-16 injected negative from roundoff.

Keep the existing adversarial tests. For generic providers the supplementary
probe confirms `negative_covariance` for both -1 and -1e-16 perturbations,
`asymmetric_covariance` for asymmetry, and `nonfinite` for NaN. These tests do NOT
prove the not-yet-implemented eligibility/mutation guard. That guard is a mandatory
new test target. A mutation-specific typed reason may require controller review
of an exact error-reason assertion; accepting the corrupt covariance is never OK.

Category tables deserve a specific guard: the upstream declaration validator
allows symmetry residual 1e-12 and eigenvalues down to -1e-10. The helper must not
reinterpret genuinely indefinite/asymmetric binary64 tables as exact PSD kernels
or clip them. Reject invalid source tables; unresolved near-degenerate cases fail
closed. Supporting tolerated-but-mathematically-invalid tables would require a
separate semantic decision, not a numerical source repair.

## 8. Cost and dependency boundary

Use only existing mpmath public arithmetic/matrix APIs: private cloned context,
mpf conversion, exp/sqrt/fsum, matrix, cholesky, and solve operations. The spike
uses `lu_solve` on each triangular Cholesky factor, not a homemade linear solver;
it is conservative but does extra factorization work. Do not assume Torch has
extended precision, NumPy longdouble is wider on this host, or SciPy solves retain
arbitrary precision. No GPU extension or compiler requirement is proposed.

One-off current-runtime timings (not a throughput guarantee):

| Frozen n=10 operation | 50 digits | 80 digits |
| --- | ---: | ---: |
| q=2 plus 6 endpoints, uncached | 17.50 ms | 19.42 ms |
| same, cached endpoints, mean of 3 | 4.85 ms | 5.43 ms |
| q=2 plus 128 endpoints, uncached | 731.84 ms | 803.71 ms |
| same, cached endpoints, mean of 3 | 14.08 ms | 15.58 ms |
| 128-endpoint cache preparation | 0.962 s | 1.002 s |

The 128-endpoint cost case repeats saved endpoints to test shape only; it does
not alter a declaration, campaign, fitted model, or acceptance fixture. Cached
and uncached MP results agreed exactly. The original 87-point catalog-shaped
joint source call took 0.38 s/current-frozen-fit and 0.31 s/lower-frozen-fit at
50 digits. Existing lower-runtime timings were similar.

Two-precision verification roughly doubles the cached source cost: about 10 ms
for q=2/e=6 and 30 ms for q=2/e=128 at this n, before derivatives/Laplace/root
work. These are estimates from the measured separate calls, not measured total
production acquisition times. Each t-batch/model batch costs another MP block;
there is no Torch-vectorized arbitrary-precision kernel. Large restart batches
and optimizer loops can therefore be materially slower.

Cache A/factors/alpha and endpoint B/P/moments once per fitted model and precision.
For variable queries recompute only query-query and query-endpoint blocks. Keep
caches bounded to the fitted adapter lifetime, not an unbounded query memoizer.
Build derivative blocks only when backward needs them. The spike's public solves
can refactor per RHS, so do not claim optimal O(n^2) cached triangular-solve cost
without a separately reviewed implementation. Increasing n and feature dimension
is an unmeasured risk; n=8/10 timings do not certify campaign-scale throughput.

The installed dependency is already transitive through Torch -> SymPy -> mpmath.
An explicit bound is preferable to relying on that incidental path. No extra
SymPy, JAX, arbitrary-precision Torch, state-space, or compiled package is needed.

## 9. Alternatives considered

1. **Multiprecision analytic source helper: recommended.** Same fixed GP algebra;
   bounded existing kernel grammar; supported gradient bridge demonstrated;
   CPU cost and adapter verification are the real tradeoffs.
2. **Float64 block Cholesky / QR on existing prior: rejected as a repair.** The
   input prior already contains construction error larger than the true mode.
   Higher-precision solves on the rounded prior alone have the same limitation.
3. **Exact state-space Matern representation: defer.** Established for suitable
   one-dimensional Markov kernels, but it introduces different ordering/state/
   query derivative machinery and does not directly cover the existing ARD,
   RBF, feature/category composition scope. No bounded local implementation was
   demonstrated here. Feature approximation, jitter, clipping, and retuning are
   excluded rather than presented as alternatives preserving the contract.

## 10. Implementation steps and acceptance oracles

1. Add frozen fixtures from the saved operands/checkpoints and RED tests for
   both Cxx failures against independent 80+ digit references; assert full joint
   means, marginal and cross covariance, repeated endpoints, and no data changes.
2. Add the private analytic source state and audited extraction grammar. Tests
   must cover constrained-parameter extraction (including non-large raw values),
   actual fitted noise, unsupported/corrupt source rejection, and mutation guards.
3. Implement the first/second derivative Functions above. Compare covariance-only
   derivatives and HVPs with MP finite differences, not merely finite/nonzero
   gradients. Include coincident and noiseless coordinates, ARD, active_dims,
   registered means, nonlinear physical input chain rules, and affine outcomes.
4. Wire the active-v2-only boundary and endpoint cache. Test zero confidence/v1
   object identity, predictions, serialization, parameters and RNG exactly.
   Preserve physical/measurement metadata and existing per-objective isolation.
5. Exercise existing registered-mean/category/prior integration, Normalize and
   Standardize source cases, all model/t-batch combinations, joint-q sampling,
   amplitudes 1e-20/1/1e20, repeated queries and exact-zero gradients. Existing
   invalid-source cases must still fail; no threshold/fixture weakening.
6. Confirm the unchanged Laplace/rank/nullspace checks accept both repaired frozen
   fits on current and lower stacks. Benchmark representative frozen acquisition
   forward/backward batch sizes before claiming acceptable production latency.
7. Controller reviews the source repair; only then commit it and let the original
   Task 6 worker resume the original fixed matrices and remaining release gates.

No production step above has been executed in this advisory task. A sensible
first implementation review should reject scope creep into general GP backends,
rank-policy changes, or source behavior not explicitly listed here.

## 11. Remaining risks / controller choices

- Approve the explicit mpmath dependency and private-state/autograd helper, not
  just a detached covariance correction. The numerical method itself is viable.
- Require completion of the listed local composition adapter before release;
  only the 1D frozen numerical bridge has been experimentally implemented here.
- Set an acceptable acquisition-latency budget. Endpoint caching makes the small
  case plausible, but large batches/dimensions/training sets remain unmeasured.
- Review the bounded precision-escalation/convergence policy and typed failure
  details. It must not change the approved relative rank policy.
- Resolve only if encountered: mathematically invalid category tables that pass
  upstream loose validation; singular training A; unsupported custom transforms.
  The conservative answer is explicit failure, not silent source reinterpretation.
- This method corrects evaluation of the frozen fitted GP, not any numerical
  error incurred while fitting its hyperparameters. No quality-gate success is
  inferred from successful construction of these two saved fits.

## 12. References and reproduction

Library basis: [mpmath matrix factorizations and linear solves](https://mpmath.org/doc/current/matrices.html)
and [arbitrary-precision contexts](https://mpmath.org/doc/current/contexts.html).
The custom derivative boundary follows PyTorch's documented
[double-backward pattern for external computations](https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html).
The state-space alternative is established in
[Nickisch et al., 2018](https://proceedings.mlr.press/v80/nickisch18a/nickisch18a.pdf).
Derivative-oracle finite differences use high precision, not a production
finite-difference estimator; see [mpmath differentiation documentation](https://mpmath.org/doc/current/calculus/differentiation.html).
The formulas and scope decisions above are this task's derivation/proposal,
not claims that a library provides a ready-made GP adapter.

Commands, cwd the named worktree, with S denoting this report's directory:

```
env PYTHONPATH=expdoe-dk/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python S/task-6a-probe.py
env PYTHONPATH=expdoe-dk/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .superpowers/sdd/2026-11-24-multiobjective-recommendation-addendum/.venv-floor/bin/python S/task-6a-probe.py
env PYTHONPATH=expdoe-dk/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python S/task-6a-cost-boundary-probe.py
```

Actual executions redirected stdout/stderr to the corresponding new logs listed
above. All three subprocesses completed with exit 0. There was no suite/matrix
run. All new files are confined to S; the original Task 6 evidence is untouched.

## 13. Binding policy addendum: review fixes B1/B2 (2026-11-25)

This addendum supersedes the incomplete guarding/convergence language in sections
5, 7 and 11. It resolves only B1/B2 of `task-6a-design-review.md`; it is a design
contract, not implemented verification or release approval. All other support
and implementation-test obligations remain. **64*n*eps*eigenscale is unchanged**;
the rules below govern source construction, not downstream numerical rank.

### B1. Owned semantic state and guards

After the one intentional public `eval()` transition, prepare an explicit manifest
for the audited grammar. `state_dict`, parameter/buffer enumeration and tensor
version counters alone are NOT the manifest. Include:

- Every stored training-input/target tensor, including ordinary attributes and
  original/preprocessed input state; all constituent fixed/learned noise tensors
  and the derived total fitted noise; raw/constrained parameters and constraints.
- Complete kernel/mean/feature/transform tree, child order, active dimensions,
  category tables, endpoints and directional fit state used by the computation.
- Typed non-tensor semantic content: mean artifacts/component payloads, feature
  params/integral_limits/origins, dimensions, normalization scalars, transform
  coefficients/options and constraint bounds/transforms. Include train/eval mode,
  transform_on_train/eval/fantasize, reverse flags and outcome output selection.
- Audited classes, relevant method/function identities and hook registries,
  including child forward/pre/backward hooks and applicable global hooks. Reject
  uninspectable callbacks/closures or unsupported hooks rather than trusting them.
  Explicitly enumerate permitted nonsemantic prediction caches; unknown mutable
  state is not silently omitted. Honor actual train/query transform flags rather
  than assuming both frames received the same preprocessing.

Own detached, storage-independent copies of ALL fixed tensors and typed deep
copies of semantic payloads. Use audited owned Torch mean/feature/transform
instances with these copies for query evaluation; their parameters are frozen,
but query autograd is retained. No live source tensor/payload may supply a cached
solve or query expression. Do not modify the ORIGINAL model's `requires_grad`
flags. Check that copying introduced no shared mutable storage or payloads.

For each stored source tensor retain object/storage identity, layout metadata,
dtype/device/shape, requires_grad flag and an owned byte-exact content snapshot.
Compare identities AND content: replacements, `.data`, NumPy/external-storage
writes and ordinary in-place changes must be detected even without a version
increment. Derived totals/constrained values require exact value comparison, not
identity of freshly computed tensors. Compare typed payload contents and tree,
method, hook, flag and constraint manifests exactly; do not use a hash alone.

Guard before AND after each forward/cache use, backward and double-backward
calculation, including owned Torch query evaluations. Each autograd context holds
the preparation generation and guard reference; saved query/cotangent operands
also get owned values and mutation checks where needed across derivative passes.
Use the owned forward state for derivatives, never newly read source values.
Any mismatch raises MODEL_FIT_FAILED with reason `source_state_changed` (or
`autograd_operand_changed` for saved query/cotangent mutation). Never reprepare
under the old Laplace state, invalidate-and-continue, or reroute after a mismatch.
Copy/load requires fresh preparation of the whole directional adapter, including
its Laplace state, not reuse of a source cache/graph from another generation.

Entry/exit checks plus owned computation are a fixed-model contract, not a
security sandbox or a guarantee to observe transient hostile concurrent writes.
Concurrent source mutation is unsupported; no live-source computation is allowed
between checks. Acceptance must exercise every manifest class after preparation
and between forward/backward/double-backward, including external-storage writes.

### B2. Precision, local accuracy and representable range

Use isolated MP contexts and the fixed comparison schedule **(50,80), (80,110),
(110,160) decimal digits**. Return the higher-precision result of the first pair
passing ALL checks. Rebuild operands/solves at each precision; never promote a
low-precision factor. Escalate unresolved numerical Cholesky/sign/convergence
failures within this route; after the last pair raise `source_precision_exhausted`.
Proven invalid parameters/noise/tables or exact singular training A fail directly
as `invalid_source`/`singular_training_covariance`. No jitter or pseudoinverse.

Let u=2^-53 and tau=32*u. For each scalar result r define an error estimate
`E(r)=abs(r_high-r_low)+e_high(r)`. The latter includes arithmetic, operand and
conditioned-solve contributions, NOT only a solve residual. Use positive
first-order error propagation through the actual expression, with elementary
MP-roundoff allowance `8*gamma_k*F`, `gamma_k=k*u_p/(1-k*u_p)`, where u_p is the
context's binary epsilon, k bounds the evaluated primitive operation count and
F is that expression's local absolute-term scale. Propagate exp/sqrt input
sensitivity too; analytic coincident-point limits avoid singular distance
derivatives. If k*u_p>=1 or a bound is undefined, escalate/fail. This is a
conservative engineering estimate plus precision comparison, NOT interval proof.

Compare A, each RHS, B, D and their derivative operands across precisions before
using them: their per-entry uncertainty includes the absolute precision difference
and local arithmetic allowance. For each solve `A*x=f`, including alpha, every P
column and derivative RHS, compute the higher-precision residual and its own
roundoff allowance. Compute an audit inverse Z by library solves against I;
do not replace the production solves with Z. Require

```
rho = ||I-A_high*Z||_inf + residual_roundoff + ||DeltaA||_inf*||Z||_inf < 1/2
M = ||Z||_inf / (1-rho)
kappa_est = ||A_high||_inf*M
8*gamma_k(high)*kappa_est < 1/16
E_x = M * ||abs(f_high-A_high*x_hat) + residual_roundoff_vector
             + Deltaf + DeltaA*abs(x_hat)||_inf
```

All quantities on the right are nonnegative estimates computed in MP; use the
bound per RHS column, not one residual shared across different RHS scales.
DeltaA/Deltaf include operand comparisons above. For example the covariance
solve/operand contribution is bounded entrywise by
`DeltaD + DeltaB*abs(P_hat) + (abs(B_high)+DeltaB)*E_P`, plus the local matrix
product/subtraction allowance. Apply the same product rule to mean/J/U/V and
Hessian expressions. A global inverse-norm bound may conservatively force a
failure; it must never enlarge the LOCAL acceptance budgets below.

Accept only these separate scalar budgets (all computed in MP, with no unit floor):

- Each non-certified diagonal v_i must satisfy v_high>0 and `E(v_i)<=tau*v_high`.
  Thus its own positive scale, not another large variance, controls acceptance.
- Each off-diagonal satisfies `E(C_ij)<=tau*sqrt(v_i*v_j)`. A certified zero row
  has exact zero entries, not a nonzero tolerance borrowed from other rows.
  Additionally require symmetry and cross-precision consistency after scaling
  the stochastic block by its OWN diagonal standard deviations: normalized
  difference row sums <= q*tau. Keep the existing final `_psd` check unchanged.
- Each mean entry, EACH feature-coordinate kernel-gradient/residual entry and
  EACH Hessian entry is checked independently: `E(r)<=tau*F_r`, with F_r the
  sum of absolute summands in THAT scalar formula before cancellation. For
  example `F_Jij,a=abs(g_ij,a)+sum_t abs(d_i,a,t*P_t,j)`; mean uses
  `abs(m_i)+sum_t abs(B_i,t*alpha_t)`, and U/V use their own corresponding terms.
  No other coordinate, large mean, Hessian, or batch contributes to F_r.

This promises formula-local ABSOLUTE accuracy for means/derivatives, not relative
accuracy in an arbitrarily cancelled derivative. It promises relative accuracy
for each positive variance and local-standard-deviation accuracy for covariance.
Perform VJP/HVP contractions in MP using the binary64 cotangents as exact inputs;
propagate operand errors and check each contracted scalar against its own
absolute-term scale before casting. This includes the dh/dG returns. Do not claim
that casting analytic entries and contracting them in unchecked double arithmetic
has the same guarantee. Original Torch mean/feature evaluations and their chain
rules remain the existing numerical boundary, not newly MP-certified formulas.

Zero handling is explicit: an agreeing pair of numerical zeros, F_r==0 at working
precision, or a small derivative is NOT a certificate. Exact zero may be returned
only from an audited algebraic identity over owned operands, for example unused
coordinates, a coincident stationary first derivative, a proved noiseless match,
or a validated zero categorical prior row. Identities must be justified per
VALUE/DERIVATIVE entry: a known-point covariance row does not imply zero moving-
point cross derivatives or Hessians. For a zero categorical prior row the mean
is m(z), not an observation target. Non-certified zero, or nonzero with unresolved
sign (`abs(r_high)<=E(r)`), escalates then fails `unresolved_source_zero_or_sign`.
This can reject a genuine but unrecognized exact zero; it must not fabricate a
deterministic identity. No threshold sets small entries to zero.

Successful nonzero numerical outputs use a **normal-only binary64 range policy**:
for every MP->Torch cast require `2^-1022 <= abs(r) <= (2-2^-52)*2^1023`, finite
conversion and round-trip error <= u*abs(r). Certified exact zeros are exempt.
Reject subnormal outputs even when representable: no subnormal relative-accuracy
promise is made. Reject nonzero-to-zero underflow and overflow, including gradient,
Hessian, VJP/HVP and auxiliary dh/dG outputs, as `source_output_range`. Do not flush
or saturate. Existing binary64 input operands are imported exactly, not floored.

Apply the same range/error preflight at EVERY supported outcome/physical affine
stage, including means, full covariance, samples and associated derivative scale
products. Form reference products/sums in MP before the actual Torch operation;
check intermediates and actual results, propagate incoming construction error,
and use the stage's local budgets. Reject overflow/underflow caused by an
intermediate such as scale squared even if rearrangement could rescue the final
value; no silent arithmetic variant is introduced here. A fitted-frame variance
1e-330 FAILS before casting even if a later scale of 1e100 would rescue it.
These checks require a narrow guard at the existing affine adapters, not a new
model or transform backend. If that boundary cannot be guarded, the affected
transform route is not approved for this repair. No claim covers precision
already lost inside existing input-feature/mean Torch evaluation.

Required focused implementation evidence: B1 mutations across all three passes;
B2 heterogeneous variances (including 1e20 beside 1e-100), zero identities versus
uncertified zeros, isolated derivative cancellation, ill-conditioned solves,
escalation/cap failure, subnormal/range failures before and after affine stages.
No new probe or implementation was performed for this policy-only fix round.

## 14. Superseding B2 zero/sign policy (design fix round 2)

Binding controller ruling for the finding in `task-6a-design-fix-1-review.md`.
**B1 remains closed and unchanged.** Retain section 13's precision schedule,
conditioning/operand checks, local scales and tau=32*2^-53. The following replaces
its universal zero/sign rejection, normal-only signed-output rule and overbroad
affine-certification wording. The approved relative PSD/rank policy is unchanged.

### Acceptance by quantity type

Let r_H be the computed higher-precision result and f its actual binary64
round-to-nearest conversion. Use `E_total=E(r)+abs(MP(f)-r_H)`; casting error is
part of, not additional tolerance beyond, the applicable local budget.

- **Non-certified variance:** require a strictly positive, normal-range r_H AND f,
  and `E_total<=tau*r_H`. No numerical zero certifies a deterministic observation.
  Unresolved positivity/precision escalates; subnormal variance, underflow or
  overflow fails typed. The fitted-frame variance 1e-330 example still fails,
  irrespective of whether later affine scaling could rescue it.
- **Signed means, ordinary off-diagonals, derivative entries, VJP/HVP and dh/dG:**
  accept f whenever finite and E_total meets its local ABSOLUTE budget, even when
  r_H or f is zero or `abs(r_H)<=E(r)`. Off-diagonals retain
  `tau*sqrt(v_i*v_j)`; means/derivatives/contractions retain their own scalar
  formula-local absolute-term budgets. There is NO separate sign/zero certificate
  requirement for these quantities. Ordinary cancellations are valid calls.
  Permit signed subnormals and rounding of a nonzero signed result to zero when
  the measured casting error fits that same budget; otherwise escalate/fail typed.
  Overflow/nonfinite output is never accepted and is a range failure, not evidence
  that the source GP itself is invalid.
- **Certified deterministic covariance identities:** retain exact known-point/
  zero-prior-row value enforcement and its independently derived derivatives.
  Never derive such an identity, derivative mask or rank decision from a returned
  numerical zero. No small-value threshold replaces r_H by zero; f is its actual
  conversion. Training-SPD and downstream semantic positivity checks remain.

For a zero local formula scale F, do not invent a unit floor and do not demand
the sign of zero. If the evaluated zero operands/identity have zero propagated
error, E_total=0 passes a zero budget. Otherwise expand the SAME scalar formula's
term scale before cancellation/grouping and propagate its operand uncertainties
(for example, expand `(G_ij+G_ji)*J_ij` into its two contributions). Use that
expanded local scale, not any other entry's magnitude. If no nonzero local scale
is available and error remains positive, escalate and ultimately fail the error
budget. Agreement of numerical zeros alone never proves zero operand error.
This rule accepts accurate cancellation without requiring a symbolic zero prover.

### Bounded error-estimation model

Implement fixed estimate formulas only for the two stationary leaves, supported
category/scale products, conditional solves, J/U/V blocks and MP contractions.
No general expression tracer or audit of arbitrary Torch/mpmath internals is
required. For gamma_k, count rounded scalar +,-,*,/ in the evaluated formula;
a length-n dot product uses the conservative count 2*n and a residual row adds
one subtraction. Use the conservative per-entry leaf count 256+32*d for either
supported stationary leaf's value/first/second derivative formula. Product/scale
nodes add their actual scalar-operation counts; unimplemented formulas are not
covered merely by assigning them a large k.

The engineering error model assumes finite MP elementary arithmetic behaves at
the context precision and exp/sqrt evaluations have local output error estimated
by 8*u_p times their magnitude. Propagate input sensitivity separately; use the
analytic coincident-point limits. These are explicit MODEL ASSUMPTIONS, not
library-provided rigorous bounds. Opaque factorization/solve error is assessed
with the section 13 residual, operand-comparison and inverse/conditioning checks,
not claimed certified by counting hidden solver operations. Cache the audit
inverse by owned source and precision. Precision agreement plus these estimates
is practical validation, not proof or an interval enclosure. Input-feature and
mean Torch evaluation remains the previously stated numerical input boundary.

### Local affine boundary and focused regressions

Bound affine checks to the supported maps on their ACTUAL binary64 inputs:
`a*x+b`, covariance scale products and the corresponding derivative products.
Evaluate local reference arithmetic in MP, check finite results/intermediates
and actual binary64 rounding against the applicable local budgets above. For
moments, propagate the available incoming moment-error estimates. For samples,
the already-generated binary64 sample is the input boundary: check only its
local affine arithmetic, not an MP certificate for the upstream Laplace root,
sample path or Monte Carlo accuracy. Signed cancelled affine values follow the
signed rule, not the strict positive-variance rule.

These fixed-map checks stay active-v2-only, preserve autograd/RNG/sample identity
and root/rank behavior, and do not introduce a general tracing backend. Their
per-sample cost is additional, not covered by the old source-forward timings.
All already-required transform compositions remain mandatory before production
wiring; this narrower boundary is not permission to waive one silently.

Required focused regressions: symmetric noncoincident training points with an
exactly zero posterior mean derivative; zero and skew-symmetric covariance
cotangents; cancelling mixed losses/shared-query contributions; auxiliary dh/dG
and HVP zeros in double backward. Also distinguish accepted budget-compliant
signed subnormal/rounded-underflow results from strict positive-variance range
failures, and verify no deterministic/rank inference from signed numerical zeros.
This round changes only the design policy; no new probe, research or production
implementation establishes those regressions yet.

