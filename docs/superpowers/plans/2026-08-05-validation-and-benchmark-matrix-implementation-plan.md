# Validation and Benchmark Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add transparent Ridge/Random Forest/Plain GP/Knowledge GP evaluation, knowledge degradation guards, and deterministic synthetic benchmark acceptance gates.

**Architecture:** A focused `evaluation/` package computes predictive and optimization metrics without changing production-surrogate selection. Synthetic fixture oracles live under `experiments/validation_matrix/`; small deterministic acceptance tests call them through a reusable benchmark runner, while expensive repetitions remain slow tests and reproducible scripts.

**Tech Stack:** Python 3.10+, scikit-learn 1.5+, NumPy, pandas, SciPy, PyTorch, BoTorch, pytest, JSON, CSV.

## Global Constraints

- Complete the domain/registry and optimization pipeline plans first.
- Ridge Regression and Random Forest are reference models, never silent production fallbacks.
- Compare random/Sobol, Ridge, Random Forest, Plain GP, and Knowledge GP where applicable.
- Report RMSE, MAE, R², calibration, simple regret/best-so-far, hypervolume regret, feasibility, duplicates, runtime, and observations consumed.
- Correct knowledge must improve data efficiency or stay within a declared fixture-specific non-inferiority tolerance.
- Deliberately wrong knowledge must warn and be downgraded or disabled by declared guard policy.
- Safety and forbidden-region patterns are not disabled solely by predictive cross-validation.
- All fixture data is synthetic, redistributable, deterministic, and free of device configuration.
- Fast CI tests use one seed and small budgets; full benchmark scripts use declared seed sets.
- Use test-driven development and commit after every task.

---

## File Responsibility Map

```text
expdoe-dk/src/expdoe_dk/evaluation/
├── __init__.py              # public evaluation exports
├── metrics.py               # predictive, calibration, regret, feasibility metrics
├── references.py            # Ridge and Random Forest evaluators
├── cross_validation.py      # deterministic split plans and model comparison
├── knowledge_guard.py       # predictive degradation policy and reports
└── benchmark.py             # sequential benchmark runner and result schema

expdoe-dk/tests/evaluation/conftest.py
                              # shared spaces, specs, problems, and fixture paths

experiments/validation_matrix/
├── README.md                # commands, budgets, tolerances, interpretation
├── fixtures.py              # four typed synthetic fixture definitions
├── run_predictive.py        # cross-model predictive matrix
├── run_sequential.py        # random/Sobol/GP sequential optimization matrix
└── expected/
    └── acceptance.json      # fixture thresholds, not frozen numerical outputs
```

## Task 1: Add Reference Model Evaluators as an Optional Capability

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/evaluation/__init__.py`
- Create: `expdoe-dk/src/expdoe_dk/evaluation/references.py`
- Create: `expdoe-dk/tests/evaluation/conftest.py`
- Create: `expdoe-dk/tests/evaluation/test_reference_models.py`
- Modify: `expdoe-dk/pyproject.toml`

**Interfaces:**
- Produces: `ReferenceModelKind`, `ReferencePrediction`, `ReferenceModelEvaluator.fit_predict(kind, train_X, train_Y, test_X, seed)`.
- Consumed by: deterministic cross-validation and `ai-doe model`.

- [ ] **Step 1: Add the explicit optional dependency**

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
reference-models = ["scikit-learn>=1.5,<2"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "jupyterlab>=4.0",
    "scikit-learn>=1.5,<2",
]
```

In `tests/evaluation/conftest.py`, define deterministic `reference_problem`, `non_safety_spec`, `safety_spec`, `numeric_space`, and helper payloads reused by all evaluation tests.

- [ ] **Step 2: Write failing reference-model tests**

```python
import numpy as np

from expdoe_dk.evaluation import ReferenceModelEvaluator, ReferenceModelKind


def test_ridge_reproduces_linear_signal():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    prediction = ReferenceModelEvaluator().fit_predict(ReferenceModelKind.RIDGE, X[:15], y[:15], X[15:], seed=7)
    assert np.max(np.abs(prediction.mean - y[15:])) < 1e-6
    assert prediction.production_eligible is False


def test_random_forest_is_reproducible():
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(30, 2))
    y = np.sin(X[:, 0] * 4) + X[:, 1]
    first = ReferenceModelEvaluator().fit_predict(ReferenceModelKind.RANDOM_FOREST, X[:20], y[:20], X[20:], seed=9)
    second = ReferenceModelEvaluator().fit_predict(ReferenceModelKind.RANDOM_FOREST, X[:20], y[:20], X[20:], seed=9)
    assert np.array_equal(first.mean, second.mean)
```

- [ ] **Step 3: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_reference_models.py`

Expected: FAIL with missing evaluation package.

- [ ] **Step 4: Implement lazy sklearn imports and reference-only metadata**

```python
class ReferenceModelEvaluator:
    def fit_predict(self, kind, train_X, train_Y, test_X, seed):
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as error:
            raise EngineError(ErrorCode.CONFIG_INVALID, "Install expdoe-dk[reference-models] to evaluate Ridge and Random Forest") from error
        if kind is ReferenceModelKind.RIDGE:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        else:
            model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1)
        model.fit(train_X, train_Y)
        return ReferencePrediction(mean=np.asarray(model.predict(test_X)), production_eligible=False, metadata={"kind": kind.value, "seed": seed})
```

- [ ] **Step 5: Run tests**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_reference_models.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/evaluation expdoe-dk/tests/evaluation/test_reference_models.py expdoe-dk/pyproject.toml
git commit -m "feat: add ridge and random forest references"
```

## Task 2: Implement Predictive, Calibration, and Optimization Metrics

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/evaluation/metrics.py`
- Create: `expdoe-dk/tests/evaluation/test_metrics.py`
- Modify: `expdoe-dk/src/expdoe_dk/evaluation/__init__.py`

**Interfaces:**
- Produces: `PredictiveMetrics`, `OptimizationMetrics`, `predictive_metrics()`, `simple_regret()`, `hypervolume_regret()`, `feasibility_rate()`, `duplicate_rate()`.
- Consumed by: cross-validation, guard policy, benchmark runner, platform model reports.

- [ ] **Step 1: Write failing exact-value tests**

```python
import numpy as np
import pytest

from expdoe_dk.evaluation.metrics import predictive_metrics, simple_regret, feasibility_rate


def test_predictive_metrics_exact_values():
    actual = predictive_metrics(
        y_true=np.array([1.0, 2.0, 3.0]),
        y_mean=np.array([1.0, 2.0, 4.0]),
        lower=np.array([0.5, 1.5, 2.5]),
        upper=np.array([1.5, 2.5, 2.9]),
    )
    assert actual.rmse == pytest.approx((1.0 / 3.0) ** 0.5)
    assert actual.mae == pytest.approx(1.0 / 3.0)
    assert actual.coverage == pytest.approx(2.0 / 3.0)


def test_simple_regret_and_feasibility():
    assert simple_regret(best_seen=8.0, optimum=10.0, maximize=True) == 2.0
    assert feasibility_rate(np.array([True, False, True])) == pytest.approx(2.0 / 3.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_metrics.py`

Expected: FAIL with missing metrics module.

- [ ] **Step 3: Implement finite, direction-aware metrics**

```python
def predictive_metrics(y_true, y_mean, lower=None, upper=None):
    truth = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(y_mean, dtype=np.float64)
    residual = mean - truth
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / denominator if denominator > 0 else float("nan")
    coverage = None if lower is None or upper is None else float(np.mean((truth >= lower) & (truth <= upper)))
    calibration_error = None if coverage is None else abs(coverage - 0.95)
    return PredictiveMetrics(rmse, mae, r2, coverage, calibration_error)
```

Use BoTorch `Hypervolume` with an explicitly recorded reference point for hypervolume regret. Reject non-finite input rather than dropping rows silently.

- [ ] **Step 4: Run metric tests**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_metrics.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/evaluation/metrics.py expdoe-dk/src/expdoe_dk/evaluation/__init__.py expdoe-dk/tests/evaluation/test_metrics.py
git commit -m "feat: add model and optimization metrics"
```

## Task 3: Add Deterministic Cross-validation and Model Comparison

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/evaluation/cross_validation.py`
- Create: `expdoe-dk/tests/evaluation/test_cross_validation.py`
- Modify: `expdoe-dk/src/expdoe_dk/bo/loop.py`

**Interfaces:**
- Produces: `CrossValidationPlan`, `ModelEvaluation`, `ModelComparison`, `compare_models(problem, plan)` and `Campaign.compare_models()`.
- Consumed by: knowledge guard and `ai-doe model`.

- [ ] **Step 1: Write failing split and comparison tests**

```python
def test_split_plan_is_deterministic_and_covers_each_row_once():
    plan = CrossValidationPlan.kfold(n_rows=20, folds=5, repeats=2, seed=12)
    assert plan == CrossValidationPlan.kfold(n_rows=20, folds=5, repeats=2, seed=12)
    first_repeat = [index for split in plan.splits if split.repeat == 0 for index in split.test_indices]
    assert sorted(first_repeat) == list(range(20))


def test_comparison_labels_reference_models(reference_problem):
    comparison = compare_models(reference_problem, CrossValidationPlan.kfold(20, 5, 1, 3))
    assert comparison.models["ridge"].production_eligible is False
    assert comparison.models["random_forest"].production_eligible is False
    assert comparison.models["plain_gp"].production_eligible is True
    assert comparison.models["knowledge_gp"].production_eligible is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_cross_validation.py`

Expected: FAIL with missing cross-validation types.

- [ ] **Step 3: Implement seeded split plans and evaluator adapters**

```python
@dataclass(frozen=True)
class CrossValidationPlan:
    splits: tuple[Split, ...]
    folds: int
    repeats: int
    seed: int

    @classmethod
    def kfold(cls, n_rows, folds, repeats, seed):
        rng = np.random.default_rng(seed)
        splits = []
        for repeat in range(repeats):
            order = rng.permutation(n_rows)
            for fold, test in enumerate(np.array_split(order, folds)):
                train = np.setdiff1d(order, test, assume_unique=True)
                splits.append(Split(repeat, fold, tuple(train.tolist()), tuple(test.tolist())))
        return cls(tuple(splits), folds, repeats, seed)
```

`compare_models()` evaluates four named tracks on identical splits. GP predictions include 95% intervals; reference models return means and explicitly report interval metrics as unavailable. Store per-fold metrics and aggregate mean/standard deviation.

To avoid a validation cycle, the Knowledge GP track receives only structurally and empirically eligible specs and does not invoke the predictive guard while cross-validating. The guard consumes the completed Plain GP versus Knowledge GP comparison afterward; `Campaign.ask()` receives the resulting final effective spec set.

- [ ] **Step 4: Run cross-validation tests**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_cross_validation.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/evaluation/cross_validation.py expdoe-dk/src/expdoe_dk/bo/loop.py expdoe-dk/tests/evaluation/test_cross_validation.py
git commit -m "feat: compare reference and gp models"
```

## Task 4: Implement Predictive Knowledge Guard Policy

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/evaluation/knowledge_guard.py`
- Create: `expdoe-dk/tests/evaluation/test_knowledge_guard.py`
- Modify: `expdoe-dk/src/expdoe_dk/knowledge/guard.py`
- Modify: `expdoe-dk/src/expdoe_dk/bo/loop.py`

**Interfaces:**
- Produces: `KnowledgeGuardPolicy`, `PatternGuardDecision`, `KnowledgeValidationReport`, `apply_predictive_guard(specs, structural_results, comparison, policy, observations)`.
- Consumed by: `Campaign.validate_knowledge()` and Knowledge GP fitting.

- [ ] **Step 1: Write the policy boundary tests**

```python
def test_guard_waits_for_minimum_observations(non_safety_spec):
    policy = KnowledgeGuardPolicy()
    decision = policy.decide(non_safety_spec, structural="valid", n_observations=7, active_numeric_dimensions=2, relative_rmse_degradation=0.5, calibration_error_increase=0.5, repeated_exceedances=2)
    assert decision.state == "insufficient_data"
    assert decision.enabled_for_fit is True


def test_guard_disables_repeated_predictive_degradation(non_safety_spec):
    policy = KnowledgeGuardPolicy(max_relative_rmse_degradation=0.10, max_calibration_error_increase=0.10)
    decision = policy.decide(non_safety_spec, structural="valid", n_observations=20, active_numeric_dimensions=2, relative_rmse_degradation=0.25, calibration_error_increase=0.12, repeated_exceedances=2)
    assert decision.state == "invalid"
    assert decision.enabled_for_fit is False


def test_safety_pattern_is_not_disabled_by_cv(safety_spec):
    decision = KnowledgeGuardPolicy().decide(safety_spec, structural="valid", n_observations=20, active_numeric_dimensions=2, relative_rmse_degradation=1.0, calibration_error_increase=1.0, repeated_exceedances=2)
    assert decision.enabled_for_fit is True
    assert decision.requires_human_revision is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_knowledge_guard.py`

Expected: FAIL with missing predictive guard.

- [ ] **Step 3: Implement the approved default policy**

```python
@dataclass(frozen=True)
class KnowledgeGuardPolicy:
    min_observations_floor: int = 8
    observations_per_active_dimension: int = 2
    max_relative_rmse_degradation: float = 0.10
    max_calibration_error_increase: float = 0.10
    required_repeated_exceedances: int = 2

    def minimum_observations(self, dimensions: int) -> int:
        return max(self.min_observations_floor, self.observations_per_active_dimension * dimensions)
```

Structural invalidity disables immediately. Hard empirical contradiction disables non-safety patterns. Predictive thresholds disable only after the required deterministic repeats. The report preserves original confidence, effective confidence, evidence, state, reasons, and enabled-for-fit state.

- [ ] **Step 4: Integrate with `Campaign.validate_knowledge()` and fit selection**

```python
report = apply_predictive_guard(
    specs=self.knowledge.specs,
    structural_results=self.registry.validate_many(self.knowledge.specs, self.space, self.observations),
    comparison=self.compare_models(plan),
    policy=self.guard_policy,
    observations=self.observations.successful(),
)
self._knowledge_validation_report = report
effective_specs = tuple(item.spec for item in report.decisions if item.enabled_for_fit)
```

- [ ] **Step 5: Run guard and legacy validator tests**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_knowledge_guard.py tests/test_validators.py tests/test_eps_auto_rescue.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/evaluation/knowledge_guard.py expdoe-dk/src/expdoe_dk/knowledge/guard.py expdoe-dk/src/expdoe_dk/bo/loop.py expdoe-dk/tests/evaluation/test_knowledge_guard.py
git commit -m "feat: guard against harmful domain knowledge"
```

## Task 5: Define Four Synthetic Fixture Families and Acceptance Thresholds

**Files:**
- Create: `experiments/validation_matrix/__init__.py`
- Create: `experiments/validation_matrix/fixtures.py`
- Create: `experiments/validation_matrix/expected/acceptance.json`
- Create: `expdoe-dk/tests/evaluation/test_fixture_oracles.py`

**Interfaces:**
- Produces: `SyntheticFixture`, `numerical_process()`, `mixed_formulation()`, `constrained_multiobjective_process()`, `known_pattern_functions()`.
- Consumed by: predictive and sequential benchmark runners.

- [ ] **Step 1: Write oracle determinism and truth tests**

```python
def test_known_pattern_fixture_encodes_declared_truth():
    fixture = known_pattern_functions()
    low = fixture.evaluate(pd.DataFrame({"time": [0.2], "dose": [0.5], "catalyst": ["A"]}))
    high = fixture.evaluate(pd.DataFrame({"time": [0.8], "dose": [0.5], "catalyst": ["A"]}))
    assert high["response"].iloc[0] > low["response"].iloc[0]
    assert fixture.correct_knowledge[0].pattern == "monotone"
    assert fixture.wrong_knowledge[0].parameters["direction"] == "decreasing"


def test_constrained_multiobjective_oracle_has_feasible_pareto_reference():
    fixture = constrained_multiobjective_process()
    assert fixture.optimum.hypervolume > 0
    assert fixture.space.feasibility_mask(fixture.optimum.points).all()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_fixture_oracles.py`

Expected: FAIL because fixture definitions do not exist.

- [ ] **Step 3: Implement typed deterministic fixtures**

```python
class FixtureOracle(Protocol):
    def __call__(self, frame: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        raise NotImplementedError


@dataclass(frozen=True)
class SyntheticFixture:
    name: str
    space: Space
    evaluate: FixtureOracle
    correct_knowledge: tuple[KnowledgePatternSpec, ...]
    wrong_knowledge: tuple[KnowledgePatternSpec, ...]
    optimum: KnownOptimum
    initial_budget: int
    sequential_budget: int
    non_inferiority_tolerance: float
```

The numerical process uses smooth continuous/discrete factors; mixed formulation includes categorical and ordinal effects; constrained multi-objective process has an analytic feasible Pareto reference; known-pattern functions isolate monotone, saturation, threshold, peak, and interaction truth. Oracle noise is generated from the provided seed, not global random state.

- [ ] **Step 4: Declare thresholds rather than frozen scores**

```json
{
  "schema_version": "1.0",
  "fixtures": {
    "numerical_single_objective": {"relative_rmse_non_inferiority": 0.10, "max_duplicate_rate": 0.0},
    "mixed_materials_formulation": {"relative_rmse_non_inferiority": 0.15, "max_duplicate_rate": 0.0},
    "constrained_multiobjective": {"max_infeasible_rate": 0.0, "max_hypervolume_regret_ratio": 1.10},
    "known_patterns": {"relative_rmse_non_inferiority": 0.10, "wrong_knowledge_must_warn": true}
  }
}
```

- [ ] **Step 5: Run fixture tests**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_fixture_oracles.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/validation_matrix expdoe-dk/tests/evaluation/test_fixture_oracles.py
git commit -m "test: add generic optimization fixtures"
```

## Task 6: Build Predictive and Sequential Benchmark Runners

**Files:**
- Create: `expdoe-dk/src/expdoe_dk/evaluation/benchmark.py`
- Create: `experiments/validation_matrix/run_predictive.py`
- Create: `experiments/validation_matrix/run_sequential.py`
- Create: `expdoe-dk/tests/evaluation/test_benchmark_runner.py`

**Interfaces:**
- Produces: `BenchmarkConfig`, `BenchmarkRecord`, `run_predictive_matrix()`, `run_sequential_matrix()`.
- Consumed by: acceptance tests, reproducible experiment reports, release gate.

- [ ] **Step 1: Write failing fast-runner tests**

```python
def test_predictive_runner_emits_all_model_tracks(tmp_path):
    config = BenchmarkConfig(seeds=(3,), folds=3, repeats=1, fast=True)
    records = run_predictive_matrix((numerical_process(),), config, output_dir=tmp_path)
    assert {record.method for record in records} == {"ridge", "random_forest", "plain_gp", "knowledge_gp_correct", "knowledge_gp_wrong"}
    assert (tmp_path / "predictive_records.jsonl").exists()


def test_sequential_runner_emits_search_and_gp_tracks(tmp_path):
    config = BenchmarkConfig(seeds=(3,), folds=3, repeats=1, fast=True)
    records = run_sequential_matrix((numerical_process(),), config, output_dir=tmp_path)
    assert {record.method for record in records} == {"random", "sobol", "plain_gp", "knowledge_gp_correct", "knowledge_gp_wrong"}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_benchmark_runner.py`

Expected: FAIL with missing benchmark runner.

- [ ] **Step 3: Implement append-only result records**

```python
@dataclass(frozen=True)
class BenchmarkConfig:
    seeds: tuple[int, ...]
    folds: int = 5
    repeats: int = 2
    fast: bool = False


@dataclass(frozen=True)
class BenchmarkRecord:
    fixture: str
    method: str
    seed: int
    metrics: dict[str, float | int | None]
    warnings: tuple[str, ...]
    elapsed_seconds: float
    package_versions: dict[str, str]


@dataclass(frozen=True)
class AggregateBenchmarkMetrics:
    rmse: float | None
    simple_regret: float | None
    hypervolume_regret: float | None
    feasibility_rate: float
    duplicate_rate: float
    knowledge_warning_rate: float
    disabled_pattern_rate: float
```

Each runner sorts fixture/method/seed order, records configuration and package versions, writes canonical JSON Lines plus a CSV summary, and never reuses modeled predictions as oracle observations.

`aggregate(records)` groups by fixture and method, computes means for numeric metrics, and computes warning/disabled rates from record diagnostics. It returns `dict[str, AggregateBenchmarkMetrics]`.

- [ ] **Step 4: Implement sequential ask/evaluate/tell loop**

```python
for seed in config.seeds:
    campaign = build_method_campaign(fixture, method, seed)
    initial = campaign.suggest_doe(fixture.initial_budget, method="sobol")
    campaign.tell(initial, Y=fixture.evaluate(initial))
    for step in range(fixture.sequential_budget):
        candidate = propose_for_method(method, campaign, fixture, seed, step)
        observed = fixture.evaluate(candidate, seed=seed * 10_000 + step)
        campaign.tell(candidate, Y=observed)
    records.append(score_campaign(fixture, method, seed, campaign))
```

- [ ] **Step 5: Run fast benchmark tests**

Run: `cd expdoe-dk && pytest -q tests/evaluation/test_benchmark_runner.py`

Expected: PASS within the fast-test budget.

- [ ] **Step 6: Commit**

```bash
git add expdoe-dk/src/expdoe_dk/evaluation/benchmark.py experiments/validation_matrix/run_predictive.py experiments/validation_matrix/run_sequential.py expdoe-dk/tests/evaluation/test_benchmark_runner.py
git commit -m "feat: add reproducible benchmark runners"
```

## Task 7: Enforce Acceptance Gates and Document Results

**Files:**
- Create: `expdoe-dk/tests/evaluation/test_acceptance_matrix.py`
- Create: `experiments/validation_matrix/README.md`
- Modify: `experiments/README.md`
- Modify: `README.md`
- Modify: `README_zh.md`

**Interfaces:**
- Consumes: benchmark records and `acceptance.json`.
- Produces: fast CI acceptance checks and documented full-run commands.

- [ ] **Step 1: Write acceptance assertions against fresh fast runs**

```python
@pytest.mark.slow
def test_correct_and_wrong_knowledge_acceptance(tmp_path):
    records = run_predictive_matrix((known_pattern_functions(),), BenchmarkConfig(seeds=(2, 5, 11), folds=5, repeats=2), tmp_path)
    by_method = aggregate(records)
    assert by_method["knowledge_gp_correct"].rmse <= by_method["plain_gp"].rmse * 1.10
    assert by_method["knowledge_gp_wrong"].knowledge_warning_rate == 1.0
    assert by_method["knowledge_gp_wrong"].disabled_pattern_rate > 0.0


@pytest.mark.slow
def test_multiobjective_candidates_are_feasible_and_nonduplicated(tmp_path):
    records = run_sequential_matrix((constrained_multiobjective_process(),), BenchmarkConfig(seeds=(2, 5, 11)), tmp_path)
    for record in records:
        assert record.metrics["feasibility_rate"] == 1.0
        assert record.metrics["duplicate_rate"] == 0.0
```

- [ ] **Step 2: Run the fast suite first**

Run: `cd expdoe-dk && pytest -q`

Expected: all fast tests PASS.

- [ ] **Step 3: Run the full acceptance matrix**

Run: `cd expdoe-dk && pytest -q --run-slow tests/evaluation/test_acceptance_matrix.py`

Expected: PASS for correct-knowledge non-inferiority, wrong-knowledge detection, feasibility, uniqueness, and reproducibility.

- [ ] **Step 4: Document exact reproduction commands and interpretation**

Document:

```bash
python experiments/validation_matrix/run_predictive.py --seeds 2 5 11 23 47 --output experiments/validation_matrix/results/predictive
python experiments/validation_matrix/run_sequential.py --seeds 2 5 11 23 47 --output experiments/validation_matrix/results/sequential
```

Explain that Ridge and Random Forest are references, individual seeds are not scientific conclusions, safety patterns require human revision, and generated result directories are ignored unless intentionally published as a release artifact.

- [ ] **Step 5: Commit**

```bash
git add expdoe-dk/tests/evaluation/test_acceptance_matrix.py experiments/validation_matrix/README.md experiments/README.md README.md README_zh.md
git commit -m "test: enforce optimization benchmark matrix"
```

## Plan Completion Gate

Run:

```bash
cd expdoe-dk
pytest -q
pytest -q --run-slow tests/evaluation
python -m compileall src
```

Expected: all tests pass, compilation exits 0, correct knowledge meets declared tolerance, wrong knowledge is surfaced and guarded, reference models remain ineligible for production recommendation, and all final optimization candidates are feasible and unique.
