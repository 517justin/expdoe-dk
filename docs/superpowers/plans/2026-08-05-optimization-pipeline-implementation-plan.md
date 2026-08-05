# Optimization Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic single-objective `Campaign.ask()` path with a typed, diagnostic-rich optimization pipeline supporting noise, pending points, constraints, mixed spaces, and multiple objectives.

**Architecture:** `Campaign` remains the public facade but delegates to an immutable `OptimizationProblem` and focused analyzer, surrogate, knowledge compiler, acquisition, candidate optimizer, post-processing, diagnostics, and stopping modules. Each pipeline stage returns typed output; numerical failures become stable engine errors rather than silent random/model fallback.

**Tech Stack:** Python 3.10+, PyTorch, BoTorch 0.17-compatible APIs, GPyTorch, NumPy, pandas, SciPy, pytest.

## Global Constraints

- Complete the domain-model and knowledge-registry plan first.
- Preserve positional single-objective `Campaign.tell(X, y)` and `Campaign.ask(q)` behavior.
- Use log improvement acquisitions instead of legacy EI/NEI/EHVI/NEHVI variants.
- Use `Yvar` when supplied and condition acquisitions on pending points.
- Ridge Regression and Random Forest are not production-surrogate fallbacks.
- Fit or acquisition failure returns a typed error with diagnostics.
- Every final candidate is converted to physical units, snapped, hard-constraint checked, and deduplicated.
- Fewer than `q` feasible unique points returns `NO_FEASIBLE_CANDIDATE`; never relax hard constraints silently.
- Fixed seeds reproduce candidate order within documented numerical tolerance.
- No test requires network access, AI credentials, or equipment.
- Use test-driven development and commit after every task.

## Official API References

- BoTorch acquisition overview: `https://botorch.org/docs/acquisition`
- BoTorch multi-objective optimization: `https://botorch.org/docs/v0.17.2/tutorials/multi_objective_bo`
- BoTorch optimizer API: `https://botorch.readthedocs.io/en/latest/optim.html`
- BoTorch model API: `https://botorch.readthedocs.io/en/latest/models.html`

---

## File Responsibility Map

```text
expdoe-dk/src/expdoe_dk/bo/
├── problem.py          # immutable optimization inputs and analysis
├── surrogate.py        # lazy blueprint, artifact-aware model construction and fit
├── acquisition.py      # policy routing and acquisition constructors
├── candidates.py       # continuous/discrete/mixed optimization dispatch
├── postprocess.py      # physical conversion, feasibility, uniqueness, diversity
├── diagnostics.py      # candidate and pipeline diagnostic types
├── stopping.py         # evidence-backed stop suggestions
├── checkpoint.py       # schema-v2 checkpoint serialization and v0.4 migration
├── loop.py             # backward-compatible Campaign and Result facade
└── gp.py               # compatibility wrapper around SurrogateFactory
```

## Task 1: Define Optimization Problem, Strategy, and Analyzer

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/bo/problem.py`
- Create: `expdoe-dk/tests/bo/conftest.py`
- Create: `expdoe-dk/tests/bo/test_problem_analyzer.py`
- Modify: `expdoe-dk/src/expdoe_dk/bo/__init__.py`

**Interfaces:**
- Consumes: `Space`, `ObservationBatch`, `PendingBatch`, `KnowledgePatternSpec`.
- Produces: `Budget`, `OptimizationStrategy`, `OptimizationProblem`, `ProblemAnalysis`, `analyze_problem(problem)`.

- [ ] **Step 1: Write failing analyzer routing tests**

```python
from expdoe_dk.bo.problem import Budget, OptimizationProblem, OptimizationStrategy, analyze_problem


def test_analyzer_requests_doe_when_successful_data_is_insufficient(numeric_space, empty_observations):
    problem = OptimizationProblem(
        space=numeric_space,
        observations=empty_observations,
        pending=None,
        budget=Budget(total=20, used=0),
        knowledge=(),
        strategy=OptimizationStrategy(),
        seed=5,
    )
    analysis = analyze_problem(problem)
    assert analysis.route == "constrained_doe"
    assert analysis.successful_rows == 0


def test_analyzer_marks_noisy_pending_multiobjective_problem(multiobjective_problem):
    analysis = analyze_problem(multiobjective_problem)
    assert analysis.route == "bayesian_optimization"
    assert analysis.has_known_noise
    assert analysis.has_pending
    assert analysis.n_objectives == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/bo/test_problem_analyzer.py`

Expected: FAIL with missing `bo.problem`.

- [ ] **Step 3: Implement immutable problem and analysis types**

```python
@dataclass(frozen=True)
class OptimizationStrategy:
    mode: Literal["balanced", "explore", "exploit"] = "balanced"
    acquisition: str | None = None
    surrogate: Literal["default", "saas"] = "default"
    batch_mode: Literal["joint", "sequential"] = "sequential"
    raw_samples: int = 256
    num_restarts: int = 10
    ucb_beta: float = 0.2


@dataclass(frozen=True)
class OptimizationProblem:
    space: Space
    observations: ObservationBatch
    pending: PendingBatch | None
    budget: Budget
    knowledge: tuple[KnowledgePatternSpec, ...]
    strategy: OptimizationStrategy
    seed: int
```

```python
@dataclass(frozen=True)
class ProblemAnalysis:
    route: Literal["constrained_doe", "bayesian_optimization"]
    successful_rows: int
    n_objectives: int
    has_known_noise: bool
    has_pending: bool
    has_outcome_constraints: bool
    categorical_dims: tuple[int, ...]
    discrete_dims: tuple[int, ...]
    fully_discrete: bool
    has_mixed_dimensions: bool
    space_cardinality: int | None
    discrete_combinations: int
    high_dimensional: bool
    hard_constraints: tuple[Constraint, ...]
```

```python
def analyze_problem(problem: OptimizationProblem) -> ProblemAnalysis:
    successful = problem.observations.successful()
    return ProblemAnalysis(
        route="constrained_doe" if len(successful.X) < max(2, problem.space.n_dims + 1) else "bayesian_optimization",
        successful_rows=len(successful.X),
        n_objectives=problem.space.n_objectives,
        has_known_noise=successful.Yvar is not None,
        has_pending=problem.pending is not None and len(problem.pending.X) > 0,
        has_outcome_constraints=bool(problem.space.outcome_constraints),
        categorical_dims=problem.space.categorical_indices,
        discrete_dims=problem.space.discrete_indices,
        fully_discrete=problem.space.is_fully_discrete,
        has_mixed_dimensions=problem.space.has_mixed_dimensions,
        space_cardinality=problem.space.cardinality,
        discrete_combinations=problem.space.discrete_cardinality,
        high_dimensional=problem.space.n_dims >= 20,
        hard_constraints=tuple(constraint for constraint in problem.space.constraints if constraint.hard),
    )
```

In `tests/bo/conftest.py`, provide `numeric_space`, `multi_space`, `empty_observations`, `observed_X`, `observed_Y`, `observed_Yvar`, `problem_factory`, `multiobjective_problem`, `mixed_problem`, `noisy_problem`, `high_dimensional_problem`, `single_objective_problem`, `discrete_problem`, `large_mixed_problem`, `postprocess_problem`, `tiny_problem`, `pending_frame`, `simple_campaign`, `acquisition`, `raw_candidates`, `duplicate_raw_candidates`, `fake_surrogate()`, `fake_acquisition()`, and the `state(**overrides)` stopping-state builder from small deterministic DataFrames. Reuse this fixture module throughout Tasks 2-8.

- [ ] **Step 4: Run analyzer tests**

Run: `cd expdoe-dk && pytest -q tests/bo/test_problem_analyzer.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/bo/problem.py expdoe-dk/src/expdoe_dk/bo/__init__.py expdoe-dk/tests/bo/test_problem_analyzer.py
git commit -m "feat: add optimization problem analyzer"
```

## Task 2: Build Lazy Surrogate Blueprints and Artifact-Aware Models

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/bo/surrogate.py`
- Create: `expdoe-dk/tests/bo/test_surrogate_factory.py`
- Modify: `expdoe-dk/src/expdoe_dk/bo/gp.py`

**Interfaces:**
- Produces: `SurrogateBlueprint`, `FittedSurrogate`, `SurrogateFactory.blueprint(problem, analysis)`, `fit_surrogate(blueprint, artifacts, observations)`.
- Consumed by: acquisition construction and model comparison.

- [ ] **Step 1: Write failing blueprint-selection tests**

```python
from expdoe_dk.bo.surrogate import SurrogateFactory


def test_factory_selects_mixed_gp_for_categorical_space(mixed_problem):
    blueprint = SurrogateFactory().blueprint(mixed_problem, analyze_problem(mixed_problem))
    assert blueprint.family == "mixed_single_task_gp"
    assert blueprint.categorical_dims == (2,)


def test_factory_preserves_yvar_for_fixed_noise(noisy_problem):
    blueprint = SurrogateFactory().blueprint(noisy_problem, analyze_problem(noisy_problem))
    fitted = fit_surrogate(blueprint, OptimizationArtifacts(), noisy_problem.observations)
    assert fitted.metadata.known_noise is True


def test_factory_uses_saas_only_when_explicit(high_dimensional_problem):
    default = SurrogateFactory().blueprint(high_dimensional_problem, analyze_problem(high_dimensional_problem))
    explicit_problem = replace(high_dimensional_problem, strategy=replace(high_dimensional_problem.strategy, surrogate="saas"))
    explicit = SurrogateFactory().blueprint(explicit_problem, analyze_problem(explicit_problem))
    assert default.family == "single_task_gp"
    assert explicit.family == "saas_fully_bayesian_gp"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/bo/test_surrogate_factory.py`

Expected: FAIL with missing surrogate factory.

- [ ] **Step 3: Implement capability-based blueprints**

```python
class SurrogateFactory:
    def blueprint(self, problem: OptimizationProblem, analysis: ProblemAnalysis) -> SurrogateBlueprint:
        if problem.strategy.surrogate == "saas":
            if analysis.categorical_dims:
                raise EngineError(ErrorCode.CONFIG_INVALID, "SAAS is not available for categorical spaces")
            if not analysis.high_dimensional:
                raise EngineError(ErrorCode.CONFIG_INVALID, "SAAS requires a high-dimensional problem")
            family = "saas_fully_bayesian_gp"
        elif analysis.categorical_dims:
            family = "mixed_single_task_gp"
        elif problem.space.n_objectives > 1:
            family = "model_list_gp"
        else:
            family = "single_task_gp"
        return SurrogateBlueprint(
            family=family,
            categorical_dims=analysis.categorical_dims,
            objective_names=tuple(problem.space.objectives),
            known_noise=analysis.has_known_noise,
            seed=problem.seed,
        )
```

- [ ] **Step 4: Finalize blueprints after Knowledge compilation**

```python
def _single_model(train_X, train_Y, train_Yvar, blueprint, modules):
    kwargs = modules.as_botorch_kwargs()
    if blueprint.family == "mixed_single_task_gp":
        return MixedSingleTaskGP(train_X, train_Y, cat_dims=list(blueprint.categorical_dims), train_Yvar=train_Yvar, **kwargs)
    if blueprint.family == "saas_fully_bayesian_gp":
        return SaasFullyBayesianSingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar, **kwargs)
    return SingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar, **kwargs)


def fit_surrogate(blueprint, artifacts, observations):
    tensors = observations_to_model_tensors(observations, blueprint)
    modules = compile_model_modules(artifacts, blueprint)
    models = [_single_model(tensors.X, tensors.Y[:, index:index + 1], tensors.yvar_for(index), blueprint, modules.for_output(index)) for index in range(tensors.Y.shape[1])]
    model = models[0] if len(models) == 1 else ModelListGP(*models)
    for component in models:
        if isinstance(component, SaasFullyBayesianSingleTaskGP):
            fit_fully_bayesian_model_nuts(component, warmup_steps=256, num_samples=128, thinning=16)
        else:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(component.likelihood, component))
    return FittedSurrogate(model=model, metadata=build_model_metadata(blueprint, artifacts, observations))
```

Implement `observations_to_model_tensors()` so it first orders `Y`/`Yvar` by `blueprint.objective_names`, applies each `Objective.to_utility()` to its own output, then standardizes that utility column while transforming its variance with the same scale. Preserve the inverse outcome transforms and scaling statistics in `FittedSurrogate.metadata` so candidate diagnostics and reports are emitted in physical objective units.

Catch fit exceptions and raise `EngineError(ErrorCode.MODEL_FIT_FAILED, "Surrogate fitting failed", details={"model_family": blueprint.family, "exception_type": type(error).__name__})`; do not continue with defaults.

- [ ] **Step 5: Keep `build_gp()` as a compatibility wrapper**

`build_gp(space, knowledge, train_X_unit, train_Y_norm)` constructs a single-objective observation batch and delegates to the factory, returning the single BoTorch model plus the legacy augmenter view.

- [ ] **Step 6: Run surrogate and legacy campaign tests**

Run: `cd expdoe-dk && pytest -q tests/bo/test_surrogate_factory.py tests/test_campaign_smoke.py tests/test_v04_compatibility.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/bo/surrogate.py expdoe-dk/src/expdoe_dk/bo/gp.py expdoe-dk/tests/bo/test_surrogate_factory.py
git commit -m "feat: add artifact-aware surrogate factory"
```

## Task 3: Implement Acquisition Policy Routing and Constructors

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/bo/acquisition.py`
- Create: `expdoe-dk/tests/bo/test_acquisition_policy.py`

**Interfaces:**
- Produces: `AcquisitionKind`, `AcquisitionDecision`, `AcquisitionPolicy.choose(problem, analysis)`, `build_acquisition(decision, fitted, problem)`.
- Consumed by: candidate optimizer.

- [ ] **Step 1: Write the complete routing truth table**

```python
@pytest.mark.parametrize(
    ("n_objectives", "noise", "pending", "constraints", "expected"),
    [
        (1, False, False, False, "q_log_ei"),
        (1, True, False, False, "q_log_nei"),
        (1, False, True, False, "q_log_nei"),
        (1, False, False, True, "constrained_q_log_ei"),
        (2, False, False, False, "q_log_ehvi"),
        (2, True, False, False, "q_log_nehvi"),
        (2, False, True, False, "q_log_nehvi"),
        (5, True, True, False, "q_log_nparego"),
    ],
)
def test_policy_routes_supported_problem(problem_factory, n_objectives, noise, pending, constraints, expected):
    problem, analysis = problem_factory(n_objectives, noise, pending, constraints)
    assert AcquisitionPolicy().choose(problem, analysis).kind.value == expected


def test_explicit_explore_mode_routes_to_ucb(single_objective_problem):
    problem = replace(single_objective_problem, strategy=replace(single_objective_problem.strategy, mode="explore"))
    assert AcquisitionPolicy().choose(problem, analyze_problem(problem)).kind.value == "q_ucb"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/bo/test_acquisition_policy.py`

Expected: FAIL with missing acquisition policy.

- [ ] **Step 3: Implement explicit routing and override validation**

```python
def choose(self, problem, analysis):
    if problem.strategy.acquisition:
        return self._validate_override(problem.strategy.acquisition, analysis)
    if problem.strategy.mode == "explore":
        return AcquisitionDecision(AcquisitionKind.Q_UCB, "explicit exploration strategy")
    if analysis.n_objectives >= 5:
        return AcquisitionDecision(AcquisitionKind.Q_LOG_NPAREGO, "many-objective scalarization")
    if analysis.n_objectives > 1:
        kind = AcquisitionKind.Q_LOG_NEHVI if analysis.has_known_noise or analysis.has_pending else AcquisitionKind.Q_LOG_EHVI
        return AcquisitionDecision(kind, "multi-objective default")
    if analysis.has_outcome_constraints:
        kind = AcquisitionKind.CONSTRAINED_Q_LOG_NEI if analysis.has_known_noise or analysis.has_pending else AcquisitionKind.CONSTRAINED_Q_LOG_EI
        return AcquisitionDecision(kind, "outcome constraints active")
    kind = AcquisitionKind.Q_LOG_NEI if analysis.has_known_noise or analysis.has_pending else AcquisitionKind.Q_LOG_EI
    return AcquisitionDecision(kind, "single-objective default")
```

- [ ] **Step 4: Implement BoTorch constructors**

```python
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound


def build_acquisition(decision, fitted, problem):
    X_baseline = model_X(problem.observations.successful(), problem.space)
    X_pending = None if problem.pending is None else model_X(problem.pending.X, problem.space)
    if decision.kind is AcquisitionKind.Q_LOG_EI:
        return qLogExpectedImprovement(fitted.model, best_f=best_feasible_utility(problem), X_pending=X_pending)
    if decision.kind is AcquisitionKind.Q_LOG_NEI:
        return qLogNoisyExpectedImprovement(fitted.model, X_baseline=X_baseline, X_pending=X_pending)
    if decision.kind is AcquisitionKind.Q_LOG_EHVI:
        return qLogExpectedHypervolumeImprovement(fitted.model, ref_point=reference_point(problem), partitioning=partitioning(problem))
    if decision.kind is AcquisitionKind.Q_LOG_NEHVI:
        return qLogNoisyExpectedHypervolumeImprovement(fitted.model, ref_point=reference_point(problem), X_baseline=X_baseline, X_pending=X_pending)
    if decision.kind is AcquisitionKind.Q_UCB:
        return qUpperConfidenceBound(fitted.model, beta=problem.strategy.ucb_beta, X_pending=X_pending)
    return build_constrained_or_scalarized_log_acquisition(decision, fitted, problem, X_baseline, X_pending)
```

Implement `build_constrained_or_scalarized_log_acquisition()` as follows: constrained single-objective decisions pass outcome-constraint callables to qLogEI/qLogNEI; qLogNParEGO-style behavior draws deterministic simplex weights from the campaign seed, constructs a Chebyshev `GenericMCObjective`, and passes it to `qLogNoisyExpectedImprovement`. Record weights in diagnostics.

```python
def build_log_nparego(fitted, problem, X_baseline, X_pending):
    weights = sample_simplex(d=problem.space.n_objectives, n=1, qmc=True, seed=problem.seed).squeeze(0)
    scalarization = get_chebyshev_scalarization(weights, observed_utility(problem))
    objective = GenericMCObjective(lambda samples, X=None: scalarization(samples))
    acquisition = qLogNoisyExpectedImprovement(fitted.model, X_baseline=X_baseline, X_pending=X_pending, objective=objective)
    return acquisition, {"weights": weights.tolist()}
```

- [ ] **Step 5: Run acquisition tests**

Run: `cd expdoe-dk && pytest -q tests/bo/test_acquisition_policy.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/bo/acquisition.py expdoe-dk/tests/bo/test_acquisition_policy.py
git commit -m "feat: add log acquisition policy"
```

## Task 4: Dispatch Continuous, Discrete, and Mixed Candidate Optimization

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/bo/candidates.py`
- Create: `expdoe-dk/tests/bo/test_candidate_optimizer.py`

**Interfaces:**
- Produces: `CandidateOptimizer.optimize(acquisition, problem, analysis, q) -> RawCandidateBatch`.
- Consumed by: post-processing pipeline.

- [ ] **Step 1: Write failing dispatch tests with mocked BoTorch optimizers**

```python
def test_small_discrete_space_enumerates(monkeypatch, discrete_problem, acquisition):
    calls = []
    monkeypatch.setattr(candidates, "optimize_acqf_discrete", lambda **kwargs: calls.append(kwargs) or fake_result(discrete_problem, 2))
    CandidateOptimizer().optimize(acquisition, discrete_problem, analyze_problem(discrete_problem), q=2)
    assert calls[0]["unique"] is True


def test_large_mixed_space_uses_alternating(monkeypatch, large_mixed_problem, acquisition):
    calls = []
    monkeypatch.setattr(candidates, "optimize_acqf_mixed_alternating", lambda **kwargs: calls.append(kwargs) or fake_result(large_mixed_problem, 1))
    CandidateOptimizer().optimize(acquisition, large_mixed_problem, analyze_problem(large_mixed_problem), q=1)
    assert calls[0]["sequential"] is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/bo/test_candidate_optimizer.py`

Expected: FAIL with missing candidate optimizer.

- [ ] **Step 3: Implement cardinality-based dispatch**

```python
from botorch.optim import optimize_acqf, optimize_acqf_discrete, optimize_acqf_mixed
from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating


def optimize(self, acquisition, problem, analysis, q):
    if analysis.fully_discrete and analysis.space_cardinality <= 10_000:
        return self._discrete(acquisition, problem, q)
    if analysis.has_mixed_dimensions and analysis.discrete_combinations <= 10:
        return self._mixed_enumerated(acquisition, problem, q)
    if analysis.has_mixed_dimensions:
        return self._mixed_alternating(acquisition, problem, q)
    return self._continuous(acquisition, problem, q)
```

`_continuous()` uses `optimize_acqf` and sets `sequential=(problem.strategy.batch_mode == "sequential")`; `_discrete()` uses `optimize_acqf_discrete` with `unique=True`, the same batch-mode choice where supported, and observed/pending tensors passed as `X_avoid`; `_mixed_enumerated()` uses `optimize_acqf_mixed` and honors the requested batch mode; `_mixed_alternating()` uses `optimize_acqf_mixed_alternating` with `sequential=True` and raises `ACQUISITION_INCOMPATIBLE` when joint mode is explicitly requested. All receive seed-controlled initial conditions and supported linear constraints.

- [ ] **Step 4: Verify dispatch and deterministic seeds**

Run: `cd expdoe-dk && pytest -q tests/bo/test_candidate_optimizer.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/bo/candidates.py expdoe-dk/tests/bo/test_candidate_optimizer.py
git commit -m "feat: optimize candidates across mixed spaces"
```

## Task 5: Post-process Feasibility, Uniqueness, Diversity, and Diagnostics

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/bo/diagnostics.py`
- Create: `expdoe-dk/src/expdoe_dk/bo/postprocess.py`
- Create: `expdoe-dk/tests/bo/test_postprocess.py`

**Interfaces:**
- Produces: `CandidateDiagnostic`, `CandidateBatch`, `RejectionSummary`, `postprocess_candidates(raw, problem, fitted, acquisition, q)`.
- Consumed by: `Campaign.ask()` and platform serialization.

- [ ] **Step 1: Write failing invariant tests**

```python
def test_postprocess_snaps_filters_and_deduplicates(postprocess_problem, raw_candidates):
    batch = postprocess_candidates(raw_candidates, postprocess_problem, fake_surrogate(), fake_acquisition(), q=2)
    assert len(batch.candidates) == 2
    assert len({candidate.key for candidate in batch.candidates}) == 2
    assert all(postprocess_problem.space.feasibility_mask(pd.DataFrame([candidate.conditions])).item() for candidate in batch.candidates)
    assert all(candidate.diagnostic.seed == postprocess_problem.seed for candidate in batch.candidates)


def test_postprocess_raises_when_q_feasible_unique_points_do_not_exist(tiny_problem, duplicate_raw_candidates):
    with pytest.raises(EngineError) as caught:
        postprocess_candidates(duplicate_raw_candidates, tiny_problem, fake_surrogate(), fake_acquisition(), q=3)
    assert caught.value.code is ErrorCode.NO_FEASIBLE_CANDIDATE
    assert caught.value.details["rejections"]["duplicate"] > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/bo/test_postprocess.py`

Expected: FAIL with missing post-processing types.

- [ ] **Step 3: Implement the ordered post-processing pipeline**

```python
def postprocess_candidates(raw, problem, fitted, acquisition, q):
    physical = problem.space.model_to_physical(raw.X)
    accepted, seen, rejected = [], observed_pending_keys(problem), Counter()
    for index, row in physical.iterrows():
        normalized = problem.space.snap_row(row.to_dict())
        if not problem.space.feasibility_mask(pd.DataFrame([normalized])).item():
            rejected["hard_constraint"] += 1
            continue
        key = problem.space.row_key(normalized)
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        accepted.append(build_candidate(normalized, raw, index, fitted, acquisition, problem))
    selected = diversity_select(accepted, q=q, space=problem.space, mode=problem.strategy.mode)
    if len(selected) < q:
        raise EngineError(ErrorCode.NO_FEASIBLE_CANDIDATE, f"Requested {q}, found {len(selected)}", details={"rejections": dict(rejected)})
    return CandidateBatch(tuple(selected), RejectionSummary(dict(rejected)))
```

`build_candidate()` records physical conditions, predicted means, uncertainty, feasibility probability, acquisition value, role, knowledge provenance, model/acquisition IDs, seed, checkpoint digest, and warnings.

- [ ] **Step 4: Run post-processing tests**

Run: `cd expdoe-dk && pytest -q tests/bo/test_postprocess.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/bo/diagnostics.py expdoe-dk/src/expdoe_dk/bo/postprocess.py expdoe-dk/tests/bo/test_postprocess.py
git commit -m "feat: validate and explain final candidates"
```

## Task 6: Refactor `Campaign` to Orchestrate the Pipeline

**Files:**
- Modify: `expdoe-dk/src/expdoe_dk/bo/loop.py`
- Modify: `expdoe-dk/src/expdoe_dk/bo/__init__.py`
- Create: `expdoe-dk/tests/bo/test_campaign_generalized.py`
- Modify: `expdoe-dk/tests/test_campaign_smoke.py`

**Interfaces:**
- Produces: extended `Campaign.tell(X, y=None, *, Y=None, Yvar=None, status=None, observation_ids=None)`, `Campaign.ask(q=1, *, pending=None, strategy=None, return_diagnostics=False)`, `Campaign.validate_knowledge()`.
- Preserves: `Campaign.ask(q) -> DataFrame` by default.

- [ ] **Step 1: Write failing multi-output, noise, pending, and compatibility tests**

```python
def test_campaign_accepts_named_multiobjective_y_and_variance(multi_space, observed_X, observed_Y, observed_Yvar):
    campaign = Campaign(multi_space, seed=3)
    campaign.tell(observed_X, Y=observed_Y, Yvar=observed_Yvar, observation_ids=[f"o{i}" for i in range(len(observed_X))])
    assert campaign.observations.Y.columns.tolist() == multi_space.objectives
    assert campaign.observations.Yvar.equals(observed_Yvar)


def test_campaign_passes_pending_points_and_returns_diagnostics(simple_campaign, pending_frame):
    batch = simple_campaign.ask(q=2, pending=PendingBatch(("p1",), pending_frame), return_diagnostics=True)
    assert len(batch.candidates) == 2
    assert all(candidate.conditions != pending_frame.iloc[0].to_dict() for candidate in batch.candidates)


def test_legacy_ask_still_returns_dataframe(simple_campaign):
    assert isinstance(simple_campaign.ask(q=1), pd.DataFrame)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/bo/test_campaign_generalized.py`

Expected: FAIL because the extended signatures are unavailable.

- [ ] **Step 3: Replace internal scalar lists with `ObservationBatch` state**

```python
def tell(self, X_phys, y=None, *, Y=None, Yvar=None, status=None, observation_ids=None):
    frame_y = normalize_Y(self.space, y=y, Y=Y, row_count=len(X_phys))
    incoming = ObservationBatch(
        X=X_phys.reset_index(drop=True),
        Y=frame_y,
        Yvar=normalize_Yvar(self.space, Yvar, len(X_phys)),
        status=normalize_status(status, len(X_phys)),
        ids=normalize_ids(observation_ids, len(X_phys), start=len(self.observations.X)),
    )
    self.observations = concatenate_observations(self.observations, incoming)
    self._maybe_run_validators()
```

Implement `normalize_Y()`, `normalize_Yvar()`, `normalize_status()`, `normalize_ids()`, and `concatenate_observations()` in `bo/loop.py` as private boundary helpers. They enforce objective-column order, non-negative finite variances, unique stable observation IDs, allowed status values, and exact row alignment before constructing an immutable `ObservationBatch`; no helper silently drops or reorders input rows.

- [ ] **Step 4: Orchestrate the approved pipeline in `ask()`**

```python
def ask(self, q=1, iteration=None, *, pending=None, strategy=None, return_diagnostics=False):
    problem = self._optimization_problem(pending=pending, strategy=strategy)
    analysis = analyze_problem(problem)
    if analysis.route == "constrained_doe":
        frame = self.suggest_doe(n=q)
        return diagnostic_doe_batch(frame, problem) if return_diagnostics else frame
    blueprint = self.surrogate_factory.blueprint(problem, analysis)
    validation = self.knowledge_validator.validate(problem)
    artifacts = self.knowledge_compiler.compile(validation.effective_specs, problem)
    fitted = fit_surrogate(blueprint, artifacts, problem.observations.successful())
    decision = self.acquisition_policy.choose(problem, analysis)
    acquisition = build_acquisition(decision, fitted, problem)
    raw = self.candidate_optimizer.optimize(acquisition, problem, analysis, q)
    batch = postprocess_candidates(raw, problem, fitted, acquisition, q)
    return batch if return_diagnostics else batch.to_frame(self.space.param_names)
```

Initialize `surrogate_factory`, `knowledge_validator`, `knowledge_compiler`, `acquisition_policy`, and `candidate_optimizer` in `Campaign.__init__()` with optional dependency-injection arguments for tests. Implement `diagnostic_doe_batch()` in `bo/diagnostics.py`; its candidate role is `initial_design`, prediction fields are `None`, and its conditions still pass through the same physical feasibility/uniqueness checks.

- [ ] **Step 5: Run generalized and legacy campaign suites**

Run: `cd expdoe-dk && pytest -q tests/bo/test_campaign_generalized.py tests/test_campaign_smoke.py tests/test_validators.py tests/test_v04_compatibility.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/bo/loop.py expdoe-dk/src/expdoe_dk/bo/__init__.py expdoe-dk/tests/bo/test_campaign_generalized.py expdoe-dk/tests/test_campaign_smoke.py
git commit -m "refactor: orchestrate typed optimization pipeline"
```

## Task 7: Add Schema-v2 Checkpoints with v0.4 Migration

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/bo/checkpoint.py`
- Create: `expdoe-dk/tests/bo/test_checkpoint_v2.py`
- Modify: `expdoe-dk/src/expdoe_dk/bo/loop.py`
- Modify: `expdoe-dk/tests/test_v04_compatibility.py`

**Interfaces:**
- Produces: `CampaignCheckpointV2`, `checkpoint_to_dict(campaign)`, `campaign_from_checkpoint(payload)`, v0.4-to-v2 migration.
- Consumed by: platform campaign repository and candidate digests.

- [ ] **Step 1: Write failing version/provenance and migration tests**

```python
def test_v2_checkpoint_records_policy_registry_and_pending(tmp_path, generalized_campaign):
    path = tmp_path / "checkpoint.json"
    generalized_campaign.save_checkpoint(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "2.0"
    assert payload["engine_version"] == "0.5.0"
    assert payload["optimization"]["strategy"]["mode"] == "balanced"
    assert "registry" in payload["knowledge"]


def test_v04_fixture_migrates_without_losing_history(v04_checkpoint_path):
    campaign = Campaign.load_checkpoint(v04_checkpoint_path)
    assert campaign.history_df()["y"].tolist() == [1.5]
    assert campaign.checkpoint_dict()["schema_version"] == "2.0"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/bo/test_checkpoint_v2.py tests/test_v04_compatibility.py`

Expected: FAIL because checkpoints have no schema version.

- [ ] **Step 3: Implement pure migration and deterministic serialization**

```python
def campaign_from_checkpoint(payload):
    version = payload.get("schema_version", "0.4")
    if version == "0.4":
        payload = migrate_v04_to_v2(payload)
    if payload["schema_version"] != "2.0":
        raise EngineError(ErrorCode.CHECKPOINT_INCOMPATIBLE, f"Unsupported checkpoint {payload['schema_version']}")
    return build_campaign_from_v2(payload)


def canonical_checkpoint_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
```

`build_campaign_from_v2()` reconstructs `Space`, `Knowledge`, observations, pending batch, budget, strategy, and seed through public constructors, then restores validation/model metadata as immutable checkpoint records. It never calls `tell()` in a way that reclassifies historical trial kinds.

Record space/objective schemas, successful/failed observations, `Yvar`, pending IDs, budget, strategy, seed/random state, provider versions, pattern validation states, model/acquisition metadata, and data digests.

- [ ] **Step 4: Run checkpoint tests**

Run: `cd expdoe-dk && pytest -q tests/bo/test_checkpoint_v2.py tests/test_v04_compatibility.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/bo/checkpoint.py expdoe-dk/src/expdoe_dk/bo/loop.py expdoe-dk/tests/bo/test_checkpoint_v2.py expdoe-dk/tests/test_v04_compatibility.py
git commit -m "feat: version campaign checkpoints"
```

## Task 8: Add Evidence-backed Stopping Suggestions and Result Reporting

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/bo/stopping.py`
- Create: `expdoe-dk/tests/bo/test_stopping.py`
- Modify: `expdoe-dk/src/expdoe_dk/bo/loop.py`
- Modify: `expdoe-dk/src/expdoe_dk/bo/report.py`
- Modify: `expdoe-dk/tests/test_html_report.py`

**Interfaces:**
- Produces: `StopSignal(kind, hard, evidence)`, `StoppingMonitor.evaluate(campaign_state)`, multi-objective-aware `Result`.
- Consumed by: platform `status` and `report` commands.

- [ ] **Step 1: Write failing stopping tests**

```python
def test_budget_exhaustion_is_hard_stop():
    signals = StoppingMonitor().evaluate(state(budget_total=10, budget_used=10))
    assert signals[0].kind == "budget_exhausted"
    assert signals[0].hard is True


def test_plateau_is_advisory_with_evidence():
    signals = StoppingMonitor(plateau_window=4, min_relative_gain=0.01).evaluate(
        state(best_history=[10.0, 10.01, 10.01, 10.02, 10.02])
    )
    plateau = next(signal for signal in signals if signal.kind == "plateau")
    assert plateau.hard is False
    assert plateau.evidence["window"] == 4


def test_uncertainty_hypervolume_and_warning_signals_are_advisory():
    signals = StoppingMonitor().evaluate(state(
        posterior_uncertainty=0.001,
        hypervolume_improvement=0.0001,
        repeated_warnings=3,
    ))
    assert {signal.kind for signal in signals} >= {"low_uncertainty", "low_hypervolume_improvement", "repeated_warnings"}
    assert all(signal.hard is False for signal in signals)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/bo/test_stopping.py`

Expected: FAIL with missing stopping monitor.

- [ ] **Step 3: Implement stopping signals and extend `Result`**

```python
class StoppingMonitor:
    def evaluate(self, state):
        signals = []
        if state.budget.used >= state.budget.total:
            signals.append(StopSignal("budget_exhausted", True, {"used": state.budget.used, "total": state.budget.total}))
        if has_plateau(state.best_history, self.plateau_window, self.min_relative_gain):
            signals.append(StopSignal("plateau", False, {"window": self.plateau_window, "threshold": self.min_relative_gain}))
        if state.expected_improvement is not None and state.expected_improvement < self.min_expected_improvement:
            signals.append(StopSignal("low_expected_improvement", False, {"value": state.expected_improvement}))
        if state.posterior_uncertainty is not None and state.posterior_uncertainty < self.max_low_uncertainty:
            signals.append(StopSignal("low_uncertainty", False, {"value": state.posterior_uncertainty}))
        if state.hypervolume_improvement is not None and state.hypervolume_improvement < self.min_hypervolume_improvement:
            signals.append(StopSignal("low_hypervolume_improvement", False, {"value": state.hypervolume_improvement}))
        if state.repeated_warnings >= self.warning_repeat_threshold:
            signals.append(StopSignal("repeated_warnings", False, {"count": state.repeated_warnings}))
        return tuple(signals)
```

Add Pareto rows, hypervolume history, candidate diagnostics, knowledge validation, and stop signals to `Result.to_dict()` while retaining existing single-objective fields.

- [ ] **Step 4: Run stopping and report tests**

Run: `cd expdoe-dk && pytest -q tests/bo/test_stopping.py tests/test_html_report.py`

Expected: PASS.

- [ ] **Step 5: Run full fast and selected slow suites**

Run: `cd expdoe-dk && pytest -q`

Expected: all fast tests PASS.

Run: `cd expdoe-dk && pytest -q --run-slow tests/bo/test_campaign_generalized.py tests/test_html_report.py`

Expected: selected slow tests PASS.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/bo/stopping.py expdoe-dk/src/expdoe_dk/bo/loop.py expdoe-dk/src/expdoe_dk/bo/report.py expdoe-dk/tests/bo/test_stopping.py expdoe-dk/tests/test_html_report.py
git commit -m "feat: report optimization diagnostics and stopping signals"
```

## Plan Completion Gate

Run:

```bash
cd expdoe-dk
pytest -q
pytest -q --run-slow tests/bo tests/test_campaign_smoke.py tests/test_html_report.py
python -m compileall src
```

Expected: all selected tests pass, compilation exits 0, pending points are never repeated, and every returned candidate passes physical-frame feasibility and uniqueness checks. Do not begin benchmark acceptance work until this gate passes.
