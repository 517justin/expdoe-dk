# DOE → Bayesian Optimisation with Ax + BoTorch

> **Note (2026-06-15)**: this research framework is being rebranded as **[`expdoe-dk`](./expdoe-dk/)** — a chemistry-friendly DoE+BO library with discrete-step parameters, linear constraints, and safe domain-knowledge injection. The original code below is preserved for reproducibility; new work should use `expdoe-dk` (see `./expdoe-dk/README.md`).

A unified framework for studying **Design of Experiments (DOE) initialisation strategies** combined with **Gaussian Process Bayesian Optimisation** using Meta's [Ax](https://ax.dev/) Service API and [BoTorch](https://botorch.org/).

---

## Overview

When running expensive black-box optimisations (e.g. materials/chemical formulation experiments with a budget of 20–40 evaluations), the choice of initial design points can significantly affect how quickly Bayesian Optimisation converges.

This repository implements and benchmarks the **DOE → BO bridging pipeline** using the Ax Service API, demonstrating:

| Part | Topic |
|------|-------|
| **A** | Core architecture: `GenerationStrategy` (DOE → BO auto-transition) + `attach_trial()` for injecting custom designs |
| **B** | Equivalence validation: Ax-BoTorch vs pure BoTorch on 4 benchmarks |
| **C** | SAASBO: sparse axis-aligned subspace BO for high-dimensional problems |
| **D** | Batch BO: `get_next_trials(max_trials=q)` for parallel acquisition (q=4) |
| **E** | JSON serialisation: `save_to_json_file` / `load_from_json_file` for checkpoint-resume |

---

## Key Results

### Part B — Ax-BoTorch vs Pure BoTorch (3-seed median final gap)

| Benchmark | Ax-BoTorch | Pure BoTorch | Verdict |
|-----------|-----------|-------------|---------|
| Branin 2D | 0.031 | 0.010 | Comparable |
| Hartmann 3D | **0.0005** | 0.010 | Ax better |
| Rosenbrock 4D | **11.88** | 17.08 | Ax better |
| Ackley 4D | 6.66 | 6.59 | Equivalent |

→ **Ax's abstraction layer does not degrade optimisation quality.**

### Part C — SAASBO vs Standard GP (4D benchmarks)

| Benchmark | Ax-BoTorch | Ax-SAASBO |
|-----------|-----------|----------|
| Rosenbrock 4D | **11.88** | 18.84 |
| Ackley 4D | 6.66 | **5.14** |

→ **SAASBO shines in ≥20D settings.** For 4D, it underperforms because: (1) its MCMC inference (NUTS) takes 5–10 min per iteration, and (2) the sparsity assumption hurts in low dimensions. Use standard GP for d ≤ 10.

### Part D — Batch BO (q=4 vs q=1, Branin 2D)

| Strategy | Total evals | Final gap |
|----------|------------|-----------|
| q=1 sequential | 16 | 0.031 |
| **q=4 batch** | 40 | **0.002** |

→ Batch BO with q=4 achieves **16× better gap** by using more parallel evaluations — ideal for experimental setups where parallel runs are possible.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ax-doe-bo.git
cd ax-doe-bo

# 2. Create and activate a conda environment (recommended)
conda create -n doegp python=3.11
conda activate doegp

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note**: `ax-platform` pulls in `torch`, `botorch`, and `gpytorch` automatically. Ensure you have a compatible CUDA environment if running on GPU (CPU is fine for these benchmarks).

---

## Usage

### Run all experiments

```bash
python ax_doe_bo.py
```

This runs all five parts sequentially and saves outputs to `outputs/`:

```
outputs/
├── ax_comparison.png      # Part B convergence curves (Ax vs Pure BoTorch)
├── saasbo_comparison.png  # Part C SAASBO vs standard GP
├── batch_demo.png         # Part D batch BO (q=1 vs q=4)
└── experiment_state.json  # Part E JSON checkpoint demo
```

> **Runtime**: ~12 hours total on CPU (SAASBO in Part C dominates due to MCMC inference). To skip SAASBO, comment out Part C in `__main__`.

### Use as a library

```python
from benchmarks import branin_2d, hartmann_3d, rosenbrock_nd, ackley_nd
from doe_utils import optimize_lhs_maximin
from ax_doe_bo import run_ax_bo, make_ax_generation_strategy, inject_doe_trials

# Generate an Opt-LHS initial design
design = optimize_lhs_maximin(n_samples=12, n_dims=4, n_iterations=200,
                               n_restarts=3, seed=42)

# Run DOE → BO with Ax
cum_best = run_ax_bo(
    design=design,
    bench_fn=rosenbrock_nd,
    n_bo=20,
    seed=42,
    surrogate="botorch",   # or "saasbo" for high-dimensional problems
)

print(f"Final gap: {abs(cum_best[-1] - 0.0):.4f}")
```

### Bring your own objective function

Replace the benchmark function with your own:

```python
def my_experiment(X: np.ndarray) -> np.ndarray:
    """
    X: shape (n, d), values in [0, 1]^d
    Returns: shape (n,) — objective values to MINIMISE
    """
    # Call your simulator / experimental evaluation here
    return your_simulator(X)

design = optimize_lhs_maximin(n_samples=10, n_dims=5, seed=42)
cum_best = run_ax_bo(design, my_experiment, n_bo=20, seed=42)
```

---

## Project Structure

```
ax-doe-bo/
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── .gitignore
│
├── ax_doe_bo.py         # Main script — all five experimental parts
├── benchmarks.py        # Benchmark functions (Branin / Hartmann / Rosenbrock / Ackley)
└── doe_utils.py         # DOE utilities (Random LHS, Maximin-Optimised LHS via SA)
```

---

## Architecture Details

### GenerationStrategy: DOE → BO auto-transition

```python
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.dispatch_utils import Generators
from ax.generation_strategy.transition_criterion import MinTrials

doe_node = GenerationNode(
    name="doe_node",
    generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL, ...)],
    transition_criteria=[
        MinTrials(
            threshold=n_init,
            transition_to="bo_node",
            use_all_trials_in_exp=True,      # manually attached trials count!
            count_only_trials_with_data=True,
        )
    ],
)
bo_node = GenerationNode(
    name="bo_node",
    generator_specs=[GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)],
)
gs = GenerationStrategy(nodes=[doe_node, bo_node])
```

### Injecting a custom DOE via `attach_trial()`

```python
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
ax_client.create_experiment(
    parameters=[{"name": f"x{i}", "type": "range", "bounds": [0.0, 1.0],
                 "value_type": "float"} for i in range(d)],
    objectives={"y": ObjectiveProperties(minimize=True)},
)

# Bypass Ax's internal DOE generator — inject our own design
for row in design:
    params = {f"x{j}": float(row[j]) for j in range(d)}
    _, trial_idx = ax_client.attach_trial(parameters=params)
    y_val = float(objective_fn(row.reshape(1, -1)).squeeze())
    ax_client.complete_trial(trial_index=trial_idx, raw_data={"y": (y_val, None)})
```

After injecting `n_init` trials, `MinTrials` triggers and the next `ax_client.get_next_trial()` call automatically uses the BO node (BOTORCH_MODULAR).

### JSON Checkpoint/Resume

```python
# Save after Phase 1
ax_client.save_to_json_file("experiment_state.json")

# Resume in Phase 2
ax_client2 = AxClient.load_from_json_file("experiment_state.json")
params, trial_idx = ax_client2.get_next_trial()   # continues BO seamlessly
```

---

## Benchmark Functions

All functions accept `X ∈ [0,1]^d` and return scalar values (minimisation).

| Function | Dim | Global Optimum | Characteristics |
|----------|-----|---------------|-----------------|
| `branin_2d` | 2 | 0.3979 (3 optima) | Multimodal, classic benchmark |
| `hartmann_3d` | 3 | −3.8628 | 4 local minima |
| `rosenbrock_nd` | n | 0.0 | Narrow curved valley, hard to optimise |
| `ackley_nd` | n | 0.0 | ~1000s of local minima, highly multimodal |

---

## When to Use SAASBO

SAASBO (`Generators.SAASBO`) uses a **fully Bayesian sparse GP** with MCMC (No-U-Turn Sampler) to identify relevant dimensions. Recommended when:

- Dimensionality ≥ 20
- Only a small subset of dimensions is truly active
- You can afford long per-iteration compute time (1–10 min per BO step)

For d ≤ 10, use `Generators.BOTORCH_MODULAR` (standard GP with MLE hyperparameters).

---

## Compatibility

Tested with:

| Package | Version |
|---------|---------|
| Python | 3.11 |
| ax-platform | 1.2.4 |
| botorch | 0.11.x |
| torch | 2.x |
| scipy | 1.12+ |

> **Note**: `AxClient` is deprecated as of Ax 1.4.0. The modern API uses `ax.api.Client`. This codebase targets Ax 1.2.x; migration to the new API is straightforward — see the [Ax migration guide](https://ax.dev).

---

## License

MIT License. See `LICENSE` for details.
