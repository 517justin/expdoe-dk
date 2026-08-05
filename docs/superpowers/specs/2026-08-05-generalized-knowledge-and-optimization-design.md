# Generalized Knowledge and Optimization Design

**Date:** 2026-08-05
**Status:** Approved design
**Repository:** `517justin/expdoe-dk`
**Target release:** `0.5.0`
**Primary consumer:** `ai-doe`

## 1. Purpose

Extend `expdoe-dk` from its current numerical, single-objective campaign into a backward-compatible engine for arbitrary supported experiments. The engine must represent mixed experiment spaces, multiple objectives, constraints, observation noise, pending trials, and domain experience expressed as validated knowledge patterns.

The core design principle is separation:

- project files contain declarative scientific intent;
- registries validate and compile that intent;
- optimization components consume typed artifacts;
- `Campaign` coordinates the components without hiding assumptions;
- adapters and CLIs remain outside this repository.

## 2. Current Baseline

Version 0.4.0 already provides:

- continuous and stepped numerical parameters;
- linear constraints and constrained initial designs;
- a single-objective GP campaign loop;
- knowledge composition for Arrhenius, monotone, quadratic peak, random augmentation, and GP prior behavior;
- empirical validators and HTML reporting.

The extension keeps these behaviors available. Existing imports, helper calls, result objects, and readable checkpoints are regression-tested before internal refactoring.

## 3. Scope

### 3.1 In scope

- Continuous, integer/discrete, categorical, and ordinal parameters.
- Linear and log parameter transforms between physical and model frames.
- Linear, declarative nonlinear, categorical-combination, hard, and soft constraints.
- Explicit maximize, minimize, and target-range objectives.
- Single- and multi-objective observations, optional known variance, and pending points.
- A versioned Knowledge Pattern Registry with built-in and installed extensions.
- GP construction for numerical and mixed spaces.
- Policy-based acquisition selection and mixed-space candidate optimization.
- Diagnostics, stopping suggestions, serialization, and backward compatibility.
- Ridge Regression and Random Forest reference-model evaluation.

### 3.2 Deferred

- Multi-fidelity and cost-aware optimization.
- Time-dependent or streaming surrogate models.
- Equipment control and observation collection.
- Arbitrary Python callables in serialized constraints or knowledge.
- Automatic scientific literature retrieval.

## 4. Public Domain Model

### 4.1 Parameters

`Parameter` becomes a tagged, serializable definition:

```python
Parameter(
    name: str,
    kind: Literal["continuous", "integer", "discrete", "categorical", "ordinal"],
    bounds: tuple[float, float] | None = None,
    values: list[Scalar] | None = None,
    step: float | int | None = None,
    unit: str | None = None,
    transform: Literal["linear", "log"] = "linear",
)
```

Rules:

- Continuous parameters require finite ordered bounds and do not use `values`.
- Integer parameters use integral bounds and an optional integral step.
- Discrete parameters use explicit numeric values or numerical bounds plus a step.
- Categorical parameters use unique string values.
- Ordinal parameters use an ordered list of unique scalar values.
- A log transform requires positive physical values.
- Existing `kind="discrete", bounds=..., step=...` calls retain their behavior.
- Existing continuous parameters without a step remain valid.

Every parameter exposes reversible `to_model()` and `to_physical()` transforms. Snapping occurs only when returning to the physical frame and is revalidated afterward.

### 4.2 Objectives

`Objective` replaces implicit string-plus-boolean semantics:

```python
Objective(
    name: str,
    direction: Literal["maximize", "minimize", "target"],
    target: float | tuple[float, float] | None = None,
    unit: str | None = None,
    priority: int = 0,
)
```

- `maximize` and `minimize` reject `target`.
- `target` accepts a point or inclusive range and is transformed into a maximized internal utility: negative absolute distance for a point, or zero inside a range and negative distance to the nearest boundary outside it. The utility is scaled with the same outcome transform used for modeling, while reports retain physical meaning.
- Objective order is stable and controls tensor columns.
- Existing `Space(objectives="yield", maximize=True|False)` is translated to one explicit objective.

### 4.3 Constraints

Four serializable constraint types are supported:

1. `LinearConstraint`: weighted factor sum with `<=`, `>=`, or `==`.
2. `ExpressionConstraint`: an allow-listed expression AST using arithmetic, comparisons, `abs`, `min`, `max`, `log`, and `exp` over declared factors.
3. `CategoricalCombinationConstraint`: allowed or forbidden partial category assignments.
4. `OutcomeConstraint`: a bound on an observed or modeled objective.

Each constraint declares `hard: bool`. A hard constraint defines feasibility. A soft constraint also declares its penalty form and finite weight, compiles to an explicit penalty or preference, and can never override a hard constraint. Serialized constraints contain no module paths, lambdas, or source code.

Expression parsing rejects unknown names, calls outside the allow-list, attribute access, indexing, and non-finite constants. Registry extensions may add constraint kinds only through explicitly installed entry points and versioned schemas.

### 4.4 Space

`Space` contains ordered parameters, ordered objectives, parameter constraints, outcome constraints, and optional metadata. Construction performs cross-reference and basic feasibility validation. Full feasibility validation may sample or solve the space and returns structured diagnostics.

### 4.5 Observations and pending points

The typed observation boundary is:

```python
ObservationBatch(
    X: DataFrame,
    Y: DataFrame,
    Yvar: DataFrame | None = None,
    status: Series | None = None,
)
```

Only successful rows enter model fitting. Failed and warning rows may inform feasibility when explicitly configured, but missing objectives are never imputed as observed values.

`PendingBatch` contains candidate identifiers and physical conditions. `Campaign.ask()` conditions acquisitions on pending points to avoid duplicate batch recommendations.

### 4.6 Initial-design engine contract

`suggest_design()` is the sole public engine entry point for initial experimental designs. It accepts a `Space`, requested count, requested method, seed, compiled parameter constraints, knowledge-source identifiers, and optional existing/pending conditions to avoid. The legacy call returning a physical-unit DataFrame remains valid; callers may request a `DesignBatch` containing the same frame plus diagnostics.

The engine distinguishes the requested method from the effective method. `method="auto"` is an explicit deterministic policy: it selects `lhs_maximin` for spaces without nominal categorical factors and `sobol` for spaces containing categorical factors. The result records both names and the capability rule used. An explicitly requested incompatible method fails with `CONFIG_INVALID` and lists compatible methods; it never changes algorithms implicitly.

Method capabilities in v0.5 are:

| Method | Supported factor kinds | Required behavior |
|---|---|---|
| `lhs_maximin`, `lhs_random` | continuous, integer, discrete, ordinal | Preserve numerical/ordered stratification after snapping |
| `sobol`, `halton` | all supported kinds | Deterministic decoding with balanced categorical/ordinal level counts when feasibility permits |
| `d_optimal` | continuous, integer, discrete | Use the declared deterministic design matrix; reject categorical or ordinal spaces |
| `random_uniform` | all supported kinds | Reference baseline, explicitly labeled as such |

The engine removes the current `d_optimal` dependency-based fallback. Missing optional capability or incompatible factor types are errors. Every successful `DesignBatch` records requested/effective method, seed, factor encodings, constraint digest, knowledge-source identifiers, categorical/ordinal level counts, minimum model-space distance, rejection counts, and candidate keys.

Initial design consumes only hard parameter constraints compiled from approved knowledge, including `safe_region` and `forbidden_region`. Shape, interaction, physics, outcome, and acquisition-preference artifacts remain recorded but do not bias initial sampling in v0.5. Conflicting or invalid safety artifacts fail before sampling.

After decoding and snapping, the engine rechecks bounds and every hard constraint, removes duplicates against the batch and supplied existing/pending conditions, and returns exactly `n` feasible unique rows. If the feasible finite cardinality or sampled feasible set cannot supply `n`, it raises `SPACE_INFEASIBLE` with requested count, available/estimated cardinality, rejection counts, method, and seed. Hard constraints are never relaxed.

## 5. Knowledge Pattern Registry

### 5.1 Serialized pattern envelope

All persisted knowledge uses `KnowledgePatternSpec`:

```python
KnowledgePatternSpec(
    pattern_id: str,
    pattern: str,
    version: str,
    parameters: dict[str, JSONValue],
    scope: KnowledgeScope,
    confidence: float,
    evidence: list[Evidence],
    enabled: bool = True,
)
```

`pattern_id` identifies one immutable declaration instance and is preserved in validation results and compiled artifacts; `pattern` identifies the registered pattern type. Explicit IDs must be unique in a knowledge set. Fluent helpers generate a deterministic `KP-<12 hex chars>` identifier from normalized content. `scope` names the affected factor or factors, objectives, optional categorical conditions, and optional physical region. `confidence` is in `[0, 1]`. Evidence records a type such as expert experience, internal data, literature, or physical law plus a citation or note. Evidence content is provenance; it is not executable.

Validation state is campaign output, not user-authored truth. The original pattern remains immutable while validation results and effective confidence evolve in checkpoints.

### 5.2 Registry entry contract

Each registered pattern provides:

```python
KnowledgePatternDefinition(
    pattern: str,
    version: str,
    family: str,
    schema: JSONSchema,
    compiler: KnowledgeCompiler,
    validator: KnowledgeValidator,
    renderer: KnowledgeRenderer,
    compatibility: CompatibilityRule,
)
```

- `schema` validates declarative parameters.
- `compiler` produces typed optimization artifacts.
- `validator` checks structural, physical-frame, and empirical consistency.
- `renderer` explains normalized meaning and effective model influence.
- `compatibility` declares supported factor types, objective types, combinations, and engine versions.

Registration rejects duplicate pattern/version pairs. Resolution is exact by default; migrations are explicit. Built-in definitions load without entry-point discovery. External definitions load only from installed providers explicitly requested by the caller.

### 5.3 Built-in taxonomy

The initial registry includes:

| Family | Patterns | Required semantics |
|---|---|---|
| Shape | `monotone`, `saturation`, `threshold`, `quadratic_peak`, `quadratic_valley`, `optimum_range`, `power_law`, `exponential`, `periodic` | One-factor response shape in a declared scope |
| Physics | `arrhenius` | Temperature-dependent rate or response behavior with unit/frame validation |
| Interaction | `synergy`, `antagonism`, `conditional_effect`, `ratio_optimum` | Two-or-more-factor effects or conditional behavior |
| Categorical | `ordinal_categories`, `category_similarity` | Ordered levels or similarity groupings |
| Feasibility | `safe_region`, `forbidden_region` | Hard or soft domain regions with explicit safety semantics |
| Multi-objective | `target_range`, `objective_priority`, `tradeoff` | Desired outcome regions and acquisition preferences |
| Prior | `gp_prior` | Explicit mean, scale, noise, or kernel prior parameters |
| Experimental | `random_augment` | Opt-in exploratory regularization retained for compatibility |

### 5.4 Pattern-specific validation

Validators include these minimum checks:

- `monotone`: valid direction, numeric factor, scoped data trend and uncertainty.
- `saturation`: valid direction, scale or half-response parameter within/near scope.
- `threshold`: threshold in physical bounds and valid below/above behavior.
- Peak/valley/range: ordered locations and width consistent with factor bounds.
- `power_law` and `exponential`: domain and sign constraints compatible with transforms.
- `periodic`: positive period and enough covered phase to claim empirical validation.
- `arrhenius`: temperature factor, recognized unit conversion to Kelvin, and positive absolute temperature.
- Interaction patterns: distinct factors and adequate joint coverage.
- `ratio_optimum`: positive denominator domain and feasible ratio range.
- Categorical patterns: referenced levels exist; similarity is symmetric and bounded.
- Safe/forbidden regions: declarative constraint validity and non-empty remaining feasible space.
- Multi-objective patterns: referenced objectives exist and ranges/priorities are coherent.

Conflicting hard safety patterns fail with `KNOWLEDGE_CONFLICT`. Other conflicts produce diagnostics, reduce effective confidence, and remain visible to the caller.

### 5.5 Compiled artifacts

The registry compiles patterns into `OptimizationArtifacts`:

```python
OptimizationArtifacts(
    input_transforms=(),
    outcome_transforms=(),
    mean_components=(),
    kernel_components=(),
    priors=(),
    virtual_observations=(),
    parameter_constraints=(),
    outcome_constraints=(),
    acquisition_preferences=(),
    diagnostics=(),
)
```

Compilation is deterministic for a fixed pattern set, data, engine version, and seed. Each artifact records `source_pattern_id`, registered pattern type, and version so design/candidate diagnostics can identify the exact declaration that influenced the result.

Multiple artifacts combine through declared composition rules. Hard constraints intersect. Mean/kernel components use registered combiners. Acquisition preferences are normalized and bounded. Unsupported combinations fail rather than relying on registration order.

### 5.6 Convenience API and compatibility

`Knowledge` remains the ergonomic Python facade:

```python
knowledge = (
    ed.Knowledge()
    .with_monotone("time", effect="increases_objective")
    .with_saturation("time", direction="increasing", half_response=45)
    .with_arrhenius("temperature")
    .with_interaction("catalyst", "temperature", kind="synergy")
)
```

Each helper constructs a registry-backed `KnowledgePatternSpec`. Existing helpers preserve their signatures and behavior where possible. Deprecations emit actionable warnings and remain readable through the 0.x compatibility window.

## 6. Knowledge Validation and Degradation

Knowledge is evaluated in three stages:

1. Structural validation before fitting.
2. Empirical validation against available observations.
3. Predictive comparison between Plain GP and Knowledge GP.

Each pattern receives `valid`, `warning`, `invalid`, or `insufficient_data`, plus an effective confidence and reasons.

Default policy:

- Structural or physical invalidity disables compilation immediately and preserves the pattern for audit.
- `insufficient_data` preserves the declared confidence but marks the effect unverified.
- Empirical contradiction reduces effective confidence according to the validator's registered policy.
- A hard empirical contradiction or predictive degradation beyond the configured non-inferiority tolerance disables that pattern for the current fit.
- Safety/forbidden-region knowledge is never disabled solely by outcome-model cross-validation; it requires explicit human revision because it may encode hazards not represented in data.
- Disabling never deletes the original specification.

The default `KnowledgeGuardPolicy` waits for at least `max(8, 2 * active_numeric_dimensions)` successful observations before predictive degradation can disable a pattern. It warns and reduces effective confidence when Knowledge GP cross-validated RMSE is more than 10% worse than Plain GP or calibration error increases by more than 0.10. It disables the contributing non-safety pattern for the current fit when either limit is exceeded in two deterministic repeated folds, or when a pattern validator reports a hard contradiction. Callers may configure these declared thresholds; checkpoints record the complete policy.

The campaign emits a deterministic `KnowledgeValidationReport`. Callers can choose strict mode, which fails on any disabled non-safety pattern, or guarded mode, which fits with the effective validated set and reports exclusions.

## 7. Optimization Problem

`Campaign.ask()` first constructs an immutable `OptimizationProblem`:

```python
OptimizationProblem(
    space: Space,
    observations: ObservationBatch,
    pending: PendingBatch,
    budget: Budget,
    knowledge: tuple[KnowledgePatternSpec, ...],
    strategy: OptimizationStrategy,
    seed: int,
)
```

The problem explicitly carries objectives, parameter and outcome constraints, parameter types, noise, pending points, budget, and knowledge. No optimizer reads global state or infers an undeclared objective.

## 8. Optimization Pipeline

```text
Campaign.ask()
  -> Problem Analyzer
  -> Surrogate Factory
  -> Knowledge Compiler
  -> Acquisition Policy
  -> Candidate Optimizer
  -> Feasibility and Diversity Post-processing
  -> Candidate Diagnostics
```

Every stage has a typed input/output and may be unit-tested independently.

### 8.1 Problem Analyzer

The analyzer determines:

- factor and objective types;
- available successful observations and known variance;
- active hard/soft constraints;
- pending and already observed points;
- feasible-space evidence;
- eligible model and acquisition families;
- whether data is sufficient for BO.

If BO is not supported by available data, the analyzer requests constrained DoE through an explicit result. It does not pretend a GP was fitted.

### 8.2 Surrogate Factory

Production model selection is capability-based. At this pipeline stage, the factory returns a lazy `SurrogateBlueprint` describing the eligible model family, transforms, and output structure:

| Problem | Model family |
|---|---|
| Numerical single-output | `SingleTaskGP`-style model |
| Mixed numerical/categorical | `MixedSingleTaskGP`-style model |
| Multiple outputs/objectives | Independent per-output models composed as a model list unless a registered correlated model is selected |
| Known observation variance | Fixed-noise behavior using `Yvar` |
| High-dimensional sparse regime | Optional SAAS/sparse family when explicitly configured and supported |

The following Knowledge Compiler stage compiles `OptimizationArtifacts` and finalizes the blueprint into the fitted surrogate. This preserves the approved pipeline order while ensuring the selected model consumes knowledge explicitly. Model metadata records training rows, transforms, kernel/mean configuration, package versions, seed, and pattern provenance.

Ridge Regression and Random Forest live in `ReferenceModelEvaluator`, outside the production surrogate factory. They produce benchmark metrics and diagnostics but cannot become silent fallback models.

### 8.3 Acquisition Policy

Default acquisition selection is explicit and inspectable:

| Situation | Default family |
|---|---|
| Single objective, low noise | qLogEI |
| Single objective, noisy observations or pending points | qLogNEI |
| Single objective with outcome constraints | Constrained log-EI/log-NEI variant |
| Multi-objective, low noise | qLogEHVI |
| Multi-objective, noisy observations or pending points | qLogNEHVI |
| Many-objective or scalarization policy | qLogNParEGO-style policy |
| Explicit exploration strategy | UCB-family policy |
| Insufficient model data | Constrained DoE, not BO |

Callers may select another registered strategy or acquisition family. Selection rejects incompatible combinations with an explanation. The policy never evaluates arbitrary serialized code.

Acquisition preferences from knowledge are bounded modifiers or reference-point/weight guidance. They cannot remove hard feasibility constraints.

### 8.4 Candidate Optimizer

Candidate optimization uses factor-aware strategies:

- Continuous dimensions use multistart acquisition optimization.
- Small finite discrete/categorical spaces may be enumerated exactly.
- Larger mixed spaces use alternating mixed optimization with registered neighborhood generation.
- Linear and supported nonlinear constraints are enforced during generation when possible and always rechecked afterward.
- Batch candidates may be generated jointly or sequentially; pending points are included in fantasization or equivalent acquisition conditioning.

All random starts and tie breaks derive from the campaign seed. Optimizer diagnostics include restart counts, raw samples, convergence warnings, and elapsed time.

### 8.5 Feasibility and diversity post-processing

Post-processing occurs in this order:

1. Convert candidates to physical units.
2. Snap integer/discrete/ordinal dimensions.
3. Re-evaluate all hard parameter and knowledge constraints.
4. Remove duplicates against observations, pending points, and the current batch.
5. Apply safe and forbidden regions.
6. Apply the configured diversity policy without violating feasibility.
7. Recompute or retain acquisition diagnostics for the final physical points.

If fewer than `q` feasible unique candidates remain, return `NO_FEASIBLE_CANDIDATE` with rejection counts by reason. Bounds, safety regions, and other hard constraints are never relaxed silently.

### 8.6 Candidate diagnostics

Each returned candidate includes:

```text
conditions in physical units
predicted objective means
predictive uncertainty
probability of feasibility
acquisition value
exploration/exploitation role
active knowledge effects and source patterns
model/acquisition identifiers
seed and checkpoint digest
warnings
```

This is an engine object that the platform serializes. Diagnostics clearly distinguish predictions from observations.

## 9. Campaign API

The current workflow remains recognizable:

```python
campaign = ed.Campaign(space, knowledge=knowledge, seed=42)
doe = campaign.suggest_doe(n=12)
campaign.tell(X, Y, Yvar=Yvar)
candidates = campaign.ask(q=4, pending=pending, strategy="balanced")
result = campaign.finalize()
```

Extensions:

- `tell()` accepts named objectives, optional variance, and observation identifiers.
- `ask()` accepts pending points, strategy/acquisition overrides, and batch mode.
- `compare_models()` evaluates reference baselines, Plain GP, and Knowledge GP.
- `validate_knowledge()` returns the full validation report.
- `checkpoint()` serializes problem state, policies, data digests, registry/provider versions, and random state.

For backward compatibility, positional single-objective `tell(X, y)` and `ask(q)` continue to work.

## 10. Model Evaluation

The standard comparison contains:

- Ridge Regression;
- Random Forest;
- Plain GP;
- Knowledge GP using the effective validated pattern set.

Applicable metrics are:

- RMSE, MAE, and R²;
- predictive interval coverage and calibration error;
- negative log predictive density when defined;
- fit and prediction time;
- cross-validation split and seed provenance.

Optimization benchmarks additionally record:

- simple regret or best-so-far;
- hypervolume regret for multi-objective cases;
- feasibility and duplicate rates;
- observations consumed and wall time.

Comparison output identifies reference-only models. A consumer must make an explicit policy choice to use a non-GP model for recommendations; v0.5 does not provide acquisition support for Ridge or Random Forest.

## 11. Stopping Monitor

The campaign produces stop suggestions based on:

- exhausted experiment budget;
- best-so-far or hypervolume plateau;
- posterior uncertainty in the region of interest;
- expected improvement or hypervolume improvement below a configured threshold;
- insufficient feasible candidates;
- repeated model or knowledge warnings.

Only budget exhaustion is a default hard stop. Other signals are recommendations with evidence. The engine does not decide whether a physical program should terminate.

## 12. Serialization and Compatibility

### 12.1 Versioned payloads

Serialized `Space`, knowledge, observations, candidates, model comparisons, and campaign checkpoints include schema and engine versions. Migrations are pure functions with fixture tests.

### 12.2 Existing public API

- Existing import paths remain exported.
- Current `Parameter`, `LinearConstraint`, `Space`, `Knowledge`, `Campaign`, and `Result` construction remains supported.
- Existing knowledge helpers map to registry specs internally.
- Version 0.4 checkpoints and results remain readable through a compatibility loader.
- Removed behavior requires a deprecation cycle and migration message.

### 12.3 Extension security

Python extensions register through a named entry-point group. The library never imports all installed providers by default. A caller supplies an allow-list of provider names. Provider name, distribution version, pattern versions, and schema digest are recorded in checkpoints.

## 13. Error Model

Typed engine errors include:

```text
CONFIG_INVALID
KNOWLEDGE_INVALID
KNOWLEDGE_CONFLICT
SPACE_INFEASIBLE
INSUFFICIENT_DATA
MODEL_FIT_FAILED
ACQUISITION_INCOMPATIBLE
NO_FEASIBLE_CANDIDATE
CHECKPOINT_INCOMPATIBLE
EXTENSION_NOT_ALLOWED
```

Errors carry a stable code, message, structured details, and optional remediation hints. Numerical warnings are preserved in diagnostics; they are not converted into a different model without explicit policy.

## 14. Testing

### 14.1 Regression baseline

Before refactoring, freeze v0.4 behavior for:

- public imports and constructor signatures;
- constrained LHS and existing validators;
- epsilon rescue and frame translation;
- campaign ask/tell smoke behavior;
- report generation;
- legacy checkpoint/result loading.

### 14.2 Unit tests

- Every parameter kind and transform round trip.
- Every constraint schema, parser rejection, and feasibility path.
- Objective direction and target utility transformation.
- Pattern schema, compiler, validator, renderer, compatibility, and composition.
- Registry duplicate/version/provider rules.
- Knowledge validation and degradation states.
- Problem analysis and acquisition-policy routing.
- Candidate post-processing, snapping, deduplication, and hard-constraint preservation.
- Serialization migrations and deterministic seeds.

### 14.3 Synthetic fixture matrix

Use four redistributable fixture families:

| Fixture | Coverage |
|---|---|
| Numerical single-objective process | Continuous/discrete factors, maximize/minimize, baseline regret |
| Mixed materials formulation | Numerical plus categorical/ordinal factors and interactions |
| Constrained multi-objective process | Parameter/outcome constraints, target ranges, hypervolume |
| Known-pattern functions | Correct and incorrect monotone, saturation, threshold, peak, and interaction knowledge |

Run random/Sobol, Ridge, Random Forest, Plain GP, and Knowledge GP where applicable. Correct knowledge must improve or remain within a declared fixture-specific non-inferiority tolerance. Deliberately wrong knowledge must produce a warning and be downgraded or disabled under the declared policy.

### 14.4 Optimization invariants

- Fixed seeds reproduce candidate order and diagnostics within documented numerical tolerance.
- Pending points are not recommended again.
- Final batches contain no internal duplicates.
- Every final point satisfies hard constraints after physical-frame snapping.
- Requesting more points than feasible returns `NO_FEASIBLE_CANDIDATE`.
- Multi-objective reference points and objective direction are reported explicitly.
- Ridge and Random Forest never appear as an unrequested fallback.

## 15. Performance Expectations

Performance is reported, not hidden behind automatic model substitution. Default fast tests use small datasets and skip expensive integration cases. Slow tests cover mixed and multi-objective acquisition paths under an explicit marker.

Budget/configuration controls include model-fit timeout guidance, candidate raw samples, optimizer restarts, and maximum enumeration size. Exceeding a configured limit returns a typed warning or error with a suggested explicit alternative.

## 16. Delivery Sequence

1. Freeze v0.4 API, checkpoint, and numerical regression fixtures.
2. Introduce versioned domain types and compatibility translation.
3. Implement registry infrastructure and migrate existing knowledge helpers.
4. Add shape, physics, interaction, categorical, feasibility, and multi-objective patterns.
5. Add mixed spaces, explicit objectives, constraints, `Yvar`, and pending batches.
6. Refactor optimization into analyzer, factory, compiler, policy, optimizer, post-processor, and diagnostics stages.
7. Add multi-objective acquisition paths and mixed candidate optimization.
8. Add reference-model evaluation and correct/wrong knowledge benchmarks.
9. Complete serialization, reporting, performance checks, and the `0.5.0` release.

Each step lands with focused tests and keeps the existing public surface usable.

## 17. Acceptance Criteria

Version 0.5.0 is ready when:

- old single-objective examples and v0.4 regression tests pass;
- all five declared parameter kinds and their transforms round-trip correctly;
- explicit single/multi-objective and target-range campaigns work;
- hard and soft constraints retain distinct semantics;
- knowledge is registry-backed, serializable, explainable, and validator-controlled;
- all approved built-in pattern families have schema/compiler/validator/renderer coverage;
- correct and deliberately wrong knowledge behavior passes the benchmark policy;
- `Yvar` and pending points affect model/acquisition behavior correctly;
- candidate output is feasible, unique, diagnostic-rich, and reproducible;
- Ridge and Random Forest are available as transparent reference baselines only;
- no engine path controls equipment, requires AI credentials, or executes serialized arbitrary code;
- `ai-doe` can consume the public engine API without reaching into private modules.
