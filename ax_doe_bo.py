"""
plan4_ax_unified.py
===================
Ax + BoTorch 統一框架 — DOE 初始化策略研究

架構說明
--------
本腳本以 Ax 1.2.x 的 Service API 建構統一的 DOE→BO 流程，與 Exp-5 純 BoTorch
手動迴圈形成對比，並展示 Ax 獨有的進階功能：

  Part A  核心架構
          ├── GenerationStrategy（DOE node → BO node 狀態切換）
          ├── attach_trial()：注入任意初始設計（Opt LHS / Sobol / D-Optimal）
          └── get_next_trial()：Ax 管理的 BO 迴圈

  Part B  與 Exp-5 對比
          └── Opt LHS + Ax BoTorch  vs  Opt LHS + 純 BoTorch（驗證等效性）

  Part C  SAASBO：高維稀疏貝氏 GP
          └── Rosenbrock 4D / Ackley 4D（標準 GP 困難的場景）

  Part D  批次 BO（q=4 平行採集）
          └── Branin 2D，每輪提議 4 個點（模擬平行實驗）

  Part E  JSON 序列化（斷點續跑）
          └── save_to_json_file / load_from_json_file

輸出
----
  outputs/plan4_ax_comparison.png      — Part B 收斂曲線
  outputs/plan4_saasbo_comparison.png  — Part C SAASBO vs 標準 GP
  outputs/plan4_batch_demo.png         — Part D 批次 BO 示範
  outputs/plan4_experiment_state.json  — Part E 序列化範例
"""

import sys, pathlib, warnings, json, time
warnings.filterwarnings("ignore")

import numpy as np
import torch

_DIR = pathlib.Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

OUTPUT_DIR = _DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Benchmark 函數 ────────────────────────────────────────────
from benchmarks import branin_2d, hartmann_3d, rosenbrock_nd, ackley_nd

# ── DOE 設計函數（從 Plan 1/3 匯入）────────────────────────
from doe_utils import latin_hypercube_sample, optimize_lhs_maximin
from scipy.stats.qmc import Sobol as QMCSobol

# ── Ax imports ───────────────────────────────────────────────
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.dispatch_utils import Generators
from ax.generation_strategy.transition_criterion import MinTrials

# ── Pure BoTorch（用於 Part B 基線比較）───────────────────
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


# ══════════════════════════════════════════════════════════════
# Part A：核心架構函數
# ══════════════════════════════════════════════════════════════

def make_ax_generation_strategy(n_init: int, surrogate: str = "botorch",
                                 seed: int = 42) -> GenerationStrategy:
    """
    建立 GenerationStrategy：
      Node 1 "doe_node"  — SOBOL（作為佔位符，實際由 attach_trial 覆蓋）
                           MinTrials(n_init) 達標後自動轉換至 BO
      Node 2 "bo_node"   — BOTORCH_MODULAR 或 SAASBO

    注意：attach_trial() 產生的試驗同樣計入 MinTrials 門檻，
    因此注入 n_init 個 DOE 點後，首次 get_next_trial() 就直接進入 BO 節點。
    """
    surrogate_enum = (
        Generators.SAASBO
        if surrogate == "saasbo"
        else Generators.BOTORCH_MODULAR
    )

    doe_node = GenerationNode(
        name="doe_node",
        generator_specs=[
            GeneratorSpec(
                generator_enum=Generators.SOBOL,
                generator_kwargs={"seed": seed, "scramble": True},
            )
        ],
        transition_criteria=[
            MinTrials(
                threshold=n_init,
                transition_to="bo_node",
                use_all_trials_in_exp=True,   # attach_trial 也計入
                count_only_trials_with_data=True,
            )
        ],
    )

    bo_node = GenerationNode(
        name="bo_node",
        generator_specs=[
            GeneratorSpec(
                generator_enum=surrogate_enum,
                generator_kwargs={},
            )
        ],
    )

    return GenerationStrategy(nodes=[doe_node, bo_node])


def inject_doe_trials(ax_client: AxClient,
                      design: np.ndarray,
                      bench_fn,
                      param_names: list[str]) -> list[float]:
    """
    透過 attach_trial() 將 DOE 矩陣注入 Ax 實驗。

    Parameters
    ----------
    design : (n_init, d) array，值域 [0, 1]
    bench_fn : callable，輸入 (1, d) array，返回純量
    param_names : ['x0', 'x1', ...] 參數名稱列表

    Returns
    -------
    y_list : DOE 階段的目標函數值列表
    """
    y_list = []
    n_init, d = design.shape
    for i, row in enumerate(design):
        params = {param_names[j]: float(row[j]) for j in range(d)}
        _, trial_idx = ax_client.attach_trial(parameters=params)
        y_val = float(bench_fn(row.reshape(1, -1)).squeeze())
        ax_client.complete_trial(
            trial_index=trial_idx,
            raw_data={"y": (y_val, None)},
        )
        y_list.append(y_val)
    return y_list


def run_ax_bo(design: np.ndarray,
              bench_fn,
              n_bo: int,
              seed: int,
              surrogate: str = "botorch") -> np.ndarray:
    """
    Ax + BoTorch 完整 DOE→BO 流程。

    Parameters
    ----------
    design   : (n_init, d) 初始 DOE 矩陣
    bench_fn : benchmark 函數（最小化）
    n_bo     : BO 迭代次數
    seed     : 隨機種子
    surrogate: "botorch"（BOTORCH_MODULAR）或 "saasbo"（SAASBO）

    Returns
    -------
    cum_best : 長度 n_init + n_bo 的 cumulative best 陣列
    """
    n_init, d = design.shape
    param_names = [f"x{i}" for i in range(d)]

    # ── 建立 GenerationStrategy ─────────────────────────────
    gs = make_ax_generation_strategy(n_init=n_init, surrogate=surrogate, seed=seed)

    # ── 建立 AxClient + Experiment ──────────────────────────
    ax_client = AxClient(
        generation_strategy=gs,
        random_seed=seed,
        verbose_logging=False,
    )
    ax_client.create_experiment(
        parameters=[
            {
                "name": pn,
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float",
            }
            for pn in param_names
        ],
        objectives={"y": ObjectiveProperties(minimize=True)},
    )

    # ── Step 1：注入 DOE 初始設計 ────────────────────────────
    y_doe = inject_doe_trials(ax_client, design, bench_fn, param_names)

    # DOE 階段 cumulative best
    running_min = float("inf")
    cum_best = []
    for y in y_doe:
        running_min = min(running_min, y)
        cum_best.append(running_min)

    # ── Step 2：Ax 管理的 BO 迴圈 ───────────────────────────
    for _ in range(n_bo):
        try:
            params_next, trial_idx = ax_client.get_next_trial()
            x_next = np.array([params_next[pn] for pn in param_names])
            y_val = float(bench_fn(x_next.reshape(1, -1)).squeeze())
            ax_client.complete_trial(
                trial_index=trial_idx,
                raw_data={"y": (y_val, None)},
            )
            running_min = min(running_min, y_val)
        except Exception as e:
            print(f"    [BO 迭代錯誤] {e}")
        cum_best.append(running_min)

    return np.array(cum_best)


# ── 純 BoTorch 手動迴圈（Part B 基線，與 Exp-5 相同）─────────
def run_pure_botorch(design: np.ndarray,
                     bench_fn,
                     n_bo: int,
                     seed: int) -> np.ndarray:
    n_init, n_dims = design.shape
    Y_init = [float(bench_fn(x.reshape(1, -1)).squeeze()) for x in design]
    running_min = float("inf")
    cum_best = []
    for y in Y_init:
        running_min = min(running_min, y)
        cum_best.append(running_min)

    train_X = torch.tensor(design, dtype=torch.double)
    train_Y = torch.tensor(Y_init, dtype=torch.double).unsqueeze(-1)
    torch.manual_seed(seed)

    for _ in range(n_bo):
        Ys = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-8)
        Yn = -Ys
        model = SingleTaskGP(train_X, Yn)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        best_f = float(Yn.max())
        acqf = qLogExpectedImprovement(model, best_f=best_f)
        bounds = torch.stack([
            torch.zeros(n_dims, dtype=torch.double),
            torch.ones(n_dims, dtype=torch.double),
        ])
        try:
            X_next, _ = optimize_acqf(
                acqf, bounds=bounds, q=1, num_restarts=4, raw_samples=128
            )
        except Exception:
            cum_best.append(cum_best[-1])
            continue
        val = float(bench_fn(X_next.detach().numpy().reshape(1, -1)).squeeze())
        train_X = torch.cat([train_X, X_next.detach()], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[val]], dtype=torch.double)], dim=0)
        running_min = min(running_min, val)
        cum_best.append(running_min)

    return np.array(cum_best)


# ══════════════════════════════════════════════════════════════
# Part B：Ax 框架 vs 純 BoTorch（等效性驗證 + 收斂比較）
# ══════════════════════════════════════════════════════════════

BENCHMARKS_B = [
    dict(name="Branin 2D",     fn=branin_2d,     dims=2, global_opt=0.397887,  n_init=8,  n_bo=20),
    dict(name="Hartmann 3D",   fn=hartmann_3d,   dims=3, global_opt=-3.8628,   n_init=8,  n_bo=20),
    dict(name="Rosenbrock 4D", fn=rosenbrock_nd, dims=4, global_opt=0.0,       n_init=12, n_bo=20),
    dict(name="Ackley 4D",     fn=ackley_nd,     dims=4, global_opt=0.0,       n_init=12, n_bo=20),
]
SEEDS_B = [42, 43, 44]

def run_part_b():
    print("\n" + "═" * 68)
    print("  Part B：Ax-BoTorch vs 純 BoTorch 等效性驗證")
    print("  （相同 Opt LHS 初始設計，相同 n_init/n_bo/seeds）")
    print("═" * 68)

    results = {}

    for bm in BENCHMARKS_B:
        name = bm["name"]
        results[name] = {
            "Ax-BoTorch":   {"curves": [], "gaps": []},
            "Pure-BoTorch": {"curves": [], "gaps": []},
        }
        print(f"\n  {name}  n_init={bm['n_init']} n_bo={bm['n_bo']}")

        for seed in SEEDS_B:
            design = optimize_lhs_maximin(
                bm["n_init"], bm["dims"], n_iterations=200, n_restarts=3, seed=seed
            )

            # Ax-BoTorch
            try:
                c_ax = run_ax_bo(design, bm["fn"], bm["n_bo"], seed, surrogate="botorch")
                gap_ax = abs(c_ax[-1] - bm["global_opt"])
                results[name]["Ax-BoTorch"]["curves"].append(c_ax)
                results[name]["Ax-BoTorch"]["gaps"].append(gap_ax)
                print(f"    seed={seed}  Ax-BoTorch   gap={gap_ax:.5f}")
            except Exception as e:
                print(f"    seed={seed}  Ax-BoTorch   ERROR: {e}")
                results[name]["Ax-BoTorch"]["curves"].append(None)
                results[name]["Ax-BoTorch"]["gaps"].append(float("inf"))

            # Pure BoTorch
            c_pure = run_pure_botorch(design, bm["fn"], bm["n_bo"], seed)
            gap_pure = abs(c_pure[-1] - bm["global_opt"])
            results[name]["Pure-BoTorch"]["curves"].append(c_pure)
            results[name]["Pure-BoTorch"]["gaps"].append(gap_pure)
            print(f"    seed={seed}  Pure-BoTorch gap={gap_pure:.5f}")

    _plot_part_b(results)
    return results


def _plot_part_b(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#FAFAFA")
    colors = {"Ax-BoTorch": "#D32F2F", "Pure-BoTorch": "#1565C0"}
    styles = {"Ax-BoTorch": "-", "Pure-BoTorch": "--"}
    markers = {"Ax-BoTorch": "s", "Pure-BoTorch": "o"}

    for ax, bm in zip(axes.ravel(), BENCHMARKS_B):
        name = bm["name"]
        for method in ["Ax-BoTorch", "Pure-BoTorch"]:
            curves = [c for c in results[name][method]["curves"] if c is not None]
            if not curves:
                continue
            n = max(len(c) for c in curves)
            padded = [np.pad(c, (0, n - len(c)), constant_values=c[-1]) for c in curves]
            med = np.median(np.vstack(padded), axis=0)
            x_axis = np.arange(len(med))
            ax.plot(x_axis, med,
                    color=colors[method], linestyle=styles[method],
                    marker=markers[method], markersize=4, markevery=4,
                    linewidth=2, label=method, zorder=3)

        # 標示 DOE/BO 分界線
        ax.axvline(x=bm["n_init"] - 0.5, color="#888", linestyle=":", linewidth=1.5,
                   label=f"DOE→BO (n={bm['n_init']})")
        ax.set_yscale("log")
        ax.set_xlabel("累積評估次數", fontsize=10)
        ax.set_ylabel("Cumulative Best（log）", fontsize=10)
        ax.set_title(f"{name}\n（n_init={bm['n_init']}, n_bo={bm['n_bo']}, 3 seeds 中位數）",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_facecolor("#F8F8F8")

    fig.suptitle("Part B：Ax-BoTorch vs 純 BoTorch 收斂比較\n（相同 Opt LHS 初始設計）",
                 fontsize=12, fontweight="bold", color="#1F4E79", y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "plan4_ax_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"\n  [圖片] {out}")


# ══════════════════════════════════════════════════════════════
# Part C：SAASBO 高維對比
# ══════════════════════════════════════════════════════════════

BENCHMARKS_C = [
    dict(name="Rosenbrock 4D", fn=rosenbrock_nd, dims=4, global_opt=0.0,
         n_init=12, n_bo=20),
    dict(name="Ackley 4D",     fn=ackley_nd,     dims=4, global_opt=0.0,
         n_init=12, n_bo=20),
]
SEEDS_C = [42, 43, 44]

def run_part_c():
    print("\n" + "═" * 68)
    print("  Part C：SAASBO vs BOTORCH_MODULAR（高維 4D benchmark）")
    print("  SAASBO = Sparse Axis-Aligned Subspace BO（全貝氏稀疏 GP）")
    print("═" * 68)

    results = {}

    for bm in BENCHMARKS_C:
        name = bm["name"]
        results[name] = {
            "Ax-BoTorch": {"curves": [], "gaps": []},
            "Ax-SAASBO":  {"curves": [], "gaps": []},
        }
        print(f"\n  {name}  n_init={bm['n_init']} n_bo={bm['n_bo']}")

        for seed in SEEDS_C:
            design = optimize_lhs_maximin(
                bm["n_init"], bm["dims"], n_iterations=200, n_restarts=3, seed=seed
            )

            for surrogate in ["botorch", "saasbo"]:
                label = "Ax-SAASBO" if surrogate == "saasbo" else "Ax-BoTorch"
                try:
                    c = run_ax_bo(design, bm["fn"], bm["n_bo"], seed, surrogate=surrogate)
                    gap = abs(c[-1] - bm["global_opt"])
                    results[name][label]["curves"].append(c)
                    results[name][label]["gaps"].append(gap)
                    print(f"    seed={seed}  {label:12s}  gap={gap:.4f}")
                except Exception as e:
                    print(f"    seed={seed}  {label:12s}  ERROR: {e}")
                    results[name][label]["curves"].append(None)
                    results[name][label]["gaps"].append(float("inf"))

    _plot_part_c(results)
    return results


def _plot_part_c(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#FAFAFA")
    colors = {"Ax-BoTorch": "#1565C0", "Ax-SAASBO": "#6A1B9A"}
    styles = {"Ax-BoTorch": "--", "Ax-SAASBO": "-"}
    markers = {"Ax-BoTorch": "o", "Ax-SAASBO": "^"}

    for ax, bm in zip(axes, BENCHMARKS_C):
        name = bm["name"]
        for method in ["Ax-BoTorch", "Ax-SAASBO"]:
            curves = [c for c in results[name][method]["curves"] if c is not None]
            if not curves:
                continue
            n = max(len(c) for c in curves)
            padded = [np.pad(c, (0, n - len(c)), constant_values=c[-1]) for c in curves]
            med = np.median(np.vstack(padded), axis=0)
            ax.plot(np.arange(len(med)), med,
                    color=colors[method], linestyle=styles[method],
                    marker=markers[method], markersize=5, markevery=4,
                    linewidth=2.2, label=method, zorder=3)

        gaps = {m: [g for g in results[name][m]["gaps"] if not np.isinf(g)]
                for m in ["Ax-BoTorch", "Ax-SAASBO"]}
        med_gaps = {m: np.median(v) if v else float("inf") for m, v in gaps.items()}
        ax.axvline(x=bm["n_init"] - 0.5, color="#888", linestyle=":", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("累積評估次數", fontsize=10)
        ax.set_ylabel("Cumulative Best（log）", fontsize=10)
        gap_str = "  ".join(f"{m.split('-')[1]}: {v:.3f}" for m, v in med_gaps.items())
        ax.set_title(f"{name}\n中位數 final gap — {gap_str}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_facecolor("#F8F8F8")

    fig.suptitle("Part C：SAASBO vs 標準 GP BoTorch（4D Benchmarks, n_init=12）",
                 fontsize=12, fontweight="bold", color="#4A148C", y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "plan4_saasbo_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"\n  [圖片] {out}")


# ══════════════════════════════════════════════════════════════
# Part D：批次 BO（q=4 平行採集）
# ══════════════════════════════════════════════════════════════

def run_part_d():
    """
    示範 get_next_trials(max_trials=q) 的批次評估：
    每輪提議 q 個點（模擬平行實驗），比較 q=1 和 q=4 的收斂軌跡。
    """
    print("\n" + "═" * 68)
    print("  Part D：批次 BO（q=4 平行採集）— Branin 2D 示範")
    print("═" * 68)

    bm = dict(fn=branin_2d, dims=2, global_opt=0.397887, n_init=8)
    N_ROUNDS = 8           # 每次提議 q 個點，共 N_ROUNDS 輪
    Q = 4                  # 批次大小
    SEED = 42

    results_batch = {}

    for q, label in [(1, "q=1 (sequential)"), (Q, f"q={Q} (batch)")]:
        design = optimize_lhs_maximin(bm["n_init"], bm["dims"],
                                      n_iterations=200, n_restarts=3, seed=SEED)
        n_init, d = design.shape
        param_names = [f"x{i}" for i in range(d)]

        gs = make_ax_generation_strategy(n_init=n_init, surrogate="botorch", seed=SEED)
        ax_client = AxClient(generation_strategy=gs, random_seed=SEED,
                             verbose_logging=False)
        ax_client.create_experiment(
            parameters=[{"name": pn, "type": "range", "bounds": [0.0, 1.0],
                         "value_type": "float"} for pn in param_names],
            objectives={"y": ObjectiveProperties(minimize=True)},
        )

        # 注入 DOE
        y_doe = inject_doe_trials(ax_client, design, bm["fn"], param_names)
        running_min = min(y_doe)
        cum_best = [min(y_doe[:i+1]) for i in range(len(y_doe))]

        # BO 迴圈（批次）
        total_evals = n_init
        for rnd in range(N_ROUNDS):
            try:
                batch_params, _ = ax_client.get_next_trials(max_trials=q)
                for tid, params_next in batch_params.items():
                    x_next = np.array([params_next[pn] for pn in param_names])
                    y_val = float(bm["fn"](x_next.reshape(1, -1)).squeeze())
                    ax_client.complete_trial(
                        trial_index=tid,
                        raw_data={"y": (y_val, None)},
                    )
                    running_min = min(running_min, y_val)
                    cum_best.append(running_min)
                    total_evals += 1
            except Exception as e:
                print(f"    [batch round {rnd} ERROR] {e}")
                break

        results_batch[label] = np.array(cum_best)
        final_gap = abs(cum_best[-1] - bm["global_opt"])
        print(f"  {label:22s} 總評估次數={total_evals:3d}  final gap={final_gap:.5f}")

    _plot_part_d(results_batch, bm["n_init"], Q, bm["global_opt"])
    return results_batch


def _plot_part_d(results, n_init, q, global_opt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#FAFAFA")
    colors = {f"q=1 (sequential)": "#1565C0", f"q={q} (batch)": "#C62828"}

    for label, cum in results.items():
        ax.plot(np.arange(len(cum)), cum,
                color=colors.get(label, "#333"),
                linewidth=2.2, marker="o" if "q=1" in label else "^",
                markersize=5, markevery=4,
                label=label, zorder=3)

    ax.axhline(y=global_opt, color="#2E7D32", linestyle="--", linewidth=1.5,
               label=f"Global opt ({global_opt:.4f})")
    ax.axvline(x=n_init - 0.5, color="#888", linestyle=":", linewidth=1.5,
               label=f"DOE→BO (n={n_init})")
    ax.set_yscale("log")
    ax.set_xlabel("累積評估次數", fontsize=11)
    ax.set_ylabel("Cumulative Best（log）", fontsize=11)
    ax.set_title(f"Part D：批次 BO（q={q}）vs 序列 BO（q=1）— Branin 2D\n"
                 f"Opt LHS 初始化，n_init={n_init}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_facecolor("#F8F8F8")

    plt.tight_layout()
    out = OUTPUT_DIR / "plan4_batch_demo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"\n  [圖片] {out}")


# ══════════════════════════════════════════════════════════════
# Part E：JSON 序列化（斷點續跑示範）
# ══════════════════════════════════════════════════════════════

def run_part_e():
    """
    示範 save_to_json_file / load_from_json_file：
    先跑 5 次 BO，儲存狀態，再載入繼續跑 5 次，驗證連續性。
    """
    print("\n" + "═" * 68)
    print("  Part E：JSON 序列化（斷點續跑示範）— Branin 2D")
    print("═" * 68)

    bm = dict(fn=branin_2d, dims=2, global_opt=0.397887)
    N_INIT, N_BO_PHASE1, N_BO_PHASE2 = 6, 5, 5
    SEED = 42
    json_path = str(OUTPUT_DIR / "plan4_experiment_state.json")

    param_names = ["x0", "x1"]
    design = optimize_lhs_maximin(N_INIT, 2, n_iterations=200, n_restarts=3, seed=SEED)

    # ── Phase 1：跑 N_BO_PHASE1 次 BO 後儲存 ─────────────────
    print(f"\n  Phase 1：DOE({N_INIT}) + BO({N_BO_PHASE1})，然後儲存...")
    gs = make_ax_generation_strategy(n_init=N_INIT, surrogate="botorch", seed=SEED)
    ax_client = AxClient(generation_strategy=gs, random_seed=SEED, verbose_logging=False)
    ax_client.create_experiment(
        parameters=[{"name": pn, "type": "range", "bounds": [0.0, 1.0],
                     "value_type": "float"} for pn in param_names],
        objectives={"y": ObjectiveProperties(minimize=True)},
    )

    y_doe = inject_doe_trials(ax_client, design, bm["fn"], param_names)
    best_after_doe = min(y_doe)
    print(f"    DOE best = {best_after_doe:.5f}")

    for i in range(N_BO_PHASE1):
        try:
            params_next, tid = ax_client.get_next_trial()
            x_next = np.array([params_next[pn] for pn in param_names])
            y_val = float(bm["fn"](x_next.reshape(1, -1)).squeeze())
            ax_client.complete_trial(trial_index=tid, raw_data={"y": (y_val, None)})
        except Exception as e:
            print(f"    BO iter {i} error: {e}")

    best_phase1, _ = ax_client.get_best_parameters()
    print(f"    Phase 1 best params: {best_phase1}")

    ax_client.save_to_json_file(json_path)
    print(f"    ✅ 儲存至：{json_path}")

    # ── Phase 2：載入並繼續跑 N_BO_PHASE2 次 ─────────────────
    print(f"\n  Phase 2：載入 → 繼續 BO({N_BO_PHASE2})...")
    ax_client2 = AxClient.load_from_json_file(json_path)
    print(f"    已載入，目前 trials 數：{len(ax_client2.experiment.trials)}")

    for i in range(N_BO_PHASE2):
        try:
            params_next, tid = ax_client2.get_next_trial()
            x_next = np.array([params_next[pn] for pn in param_names])
            y_val = float(bm["fn"](x_next.reshape(1, -1)).squeeze())
            ax_client2.complete_trial(trial_index=tid, raw_data={"y": (y_val, None)})
        except Exception as e:
            print(f"    BO iter {i} error: {e}")

    best_phase2, _ = ax_client2.get_best_parameters()
    best_trial = ax_client2.get_best_trial()
    if best_trial is not None and best_trial[2] is not None:
        best_y2 = best_trial[2][0]["y"]  # (trial_idx, params, (mean_dict, covar_dict))
    else:
        # 退回從 raw data 取最佳值
        best_y2 = ax_client2.experiment.fetch_data().df["mean"].min()
    gap = abs(best_y2 - bm["global_opt"])
    print(f"    Phase 2 final best y = {best_y2:.5f}  gap = {gap:.5f}")
    print(f"    ✅ 斷點續跑成功！總 trials = {len(ax_client2.experiment.trials)}")

    return ax_client2


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 68)
    print("  Plan 4：Ax + BoTorch 統一框架示範")
    print(f"  輸出目錄：{OUTPUT_DIR}")
    print("=" * 68)

    t0 = time.time()

    # Part B：等效性驗證（主要比較）
    results_b = run_part_b()

    # Part C：SAASBO 高維對比
    results_c = run_part_c()

    # Part D：批次 BO
    results_d = run_part_d()

    # Part E：JSON 序列化
    run_part_e()

    elapsed = time.time() - t0
    print(f"\n{'='*68}")
    print(f"  全部完成！耗時 {elapsed/60:.1f} 分鐘")
    print(f"  輸出：")
    for f in sorted(OUTPUT_DIR.glob("plan4_*.png")):
        print(f"    {f.name}")
    print(f"    plan4_experiment_state.json")
    print("=" * 68)
