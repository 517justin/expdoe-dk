# Roadmap

## 0.6.0

- Named multiobjective measurements and physical-unit independent GP predictions.
- Fixed catalogs, pending/measurement identities and qLogEHVI, qLogNEHVI and
  qLogNParEGO-style routing; measured Pareto and fixed-geometry HV reports.
- Multiobjective schema 2.2 with release writer 0.6.0; exact historical scalar
  reader allowlists, with no reconstruction of missing objective measurements.
- Explicit monotone v2 tempered-probit Laplace conditioning, preserving measured
  values/variances and fitted hyperparameters. V1 remains explicitly compatible;
  no helper, stored definition or default silently resolves to v2.
- Fixed five-seed current/lower acceptance passes. V2 ties Plain on this fixture;
  legacy v1 failures and the earlier numerical failure remain documented.

See [the workflow guide](multiobjective.md) and
[per-method evidence](../experiments/multiobjective_validation/README.md).

## Limits and Deferred Work

Opt-in v2 can be expensive: bounded source-adapter observations reached roughly
131-133 seconds for forward/backward, excluding setup. This is not a full
recommendation benchmark or an interactive guarantee. Performance improvements
must preserve the same numerical and search policies, not weaken acceptance.

External observer/device integration, MCP/platform adapters, cross-output
correlation, partial-objective learning, arbitrary nonlinear cross-output safety,
GPU/other-OS certification and textbook/Colab validation remain outside this
release. No schedule or general optimality/knowledge-superiority claim is made.
