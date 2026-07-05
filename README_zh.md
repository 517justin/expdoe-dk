[English README](./README.md)

# expdoe-dk

**實驗設計 (DoE) + 貝葉斯優化 (BO)，支援選擇性領域知識注入，適合化學、材料與實驗室流程。**

`expdoe-dk` 協助實驗者規劃少量初始實驗，遵守真實實驗限制，並接著用高斯過程貝葉斯優化推進下一批條件。套件以物理單位、離散操作刻度、線性限制與領域知識為核心，例如單調趨勢、Arrhenius 溫度效應與峰值型因子。

## 功能概覽

| 功能 | 說明 |
|------|------|
| 限制式 DoE | 使用 LHS、Sobol、Halton、D-optimal 或 random 產生初始點，同時遵守邊界、離散步階與線性限制。 |
| 貝葉斯優化 | 初始 DoE 後，以 GP 模型和 ask/tell campaign loop 繼續建議實驗條件。 |
| 領域知識 | 用 `Knowledge().with_arrhenius(...)`、`with_monotone(...)`、`with_quadratic_peak(...)` 等 helper 編碼已知趨勢。 |
| 實驗室輸出 | 使用物理單位操作，並可輸出可分享的 HTML campaign report。 |
| 安全預設 | 若未提供知識，Campaign 使用純 GP，不會偷偷注入假設。 |

## 快速範例

```python
import expdoe_dk as ed

space = ed.Space(
    params=[
        ed.Parameter("T", bounds=(60, 120), unit="degC", kind="discrete", step=1),
        ed.Parameter("time", bounds=(10, 180), unit="min", kind="discrete", step=5),
        ed.Parameter("pH", bounds=(4, 10), kind="discrete", step=1),
        ed.Parameter("conc_A", bounds=(1, 10), unit="mL", kind="discrete", step=1),
    ],
    objectives="yield_pct",
    maximize=True,
)

knowledge = (
    ed.Knowledge()
    .with_arrhenius("T")
    .with_monotone("time", effect="increases_objective")
    .with_quadratic_peak("pH", center=7)
)

campaign = ed.Campaign(space, knowledge, seed=42)

doe = campaign.suggest_doe(n=12)
y_doe = run_lab_experiments(doe)
campaign.tell(doe, y_doe)

for _ in range(20):
    x_next = campaign.ask(q=1)
    y_next = run_lab_experiments(x_next)
    campaign.tell(x_next, y_next)

result = campaign.finalize()
result.to_html("campaign_report.html")
```

## 安裝

從 repo 根目錄進行一般開發安裝：

```bash
pip install -r requirements.txt
pip install -e ./expdoe-dk
```

或直接從套件目錄安裝：

```bash
cd expdoe-dk
pip install -e .
```

執行時需求列在 [`requirements.txt`](./requirements.txt)，並同步寫在 [`expdoe-dk/pyproject.toml`](./expdoe-dk/pyproject.toml)：Python 3.10+、PyTorch、BoTorch、GPyTorch、Ax、NumPy、SciPy、pandas、matplotlib、pyDOE3。

開發與測試工具：

```bash
cd expdoe-dk
pip install -e ".[dev]"
pytest -q
```

若要執行較慢的整合測試：

```bash
pytest -q --run-slow
```

## 目錄結構

```text
expdoe-dk/                          # 可發布的 Python 套件
  src/expdoe_dk/
    space.py                        # Parameter, LinearConstraint, Space
    doe/                            # DoE 方法
    knowledge/                      # 知識規格與座標翻譯
    bo/                             # Campaign loop 與 HTML 報告
    legacy/                         # ax_doe_bo 相容 shim
  tests/                            # 單元與整合測試
  pyproject.toml                    # 套件 metadata 與 dependencies

examples/                           # 端對端使用範例
experiments/                        # 可重現研究與結果說明
docs/superpowers/specs/              # 規劃中實驗的設計文件

ax_doe_bo.py / doe_utils.py / benchmarks.py
                                    # 保留供重現的歷史研究框架
```

## 範例與實驗

| 路徑 | 用途 |
|------|------|
| [`examples/01_reaction_optimization.py`](./examples/01_reaction_optimization.py) | DoE 到 BO 的端對端反應優化範例。 |
| [`examples/02_html_report.py`](./examples/02_html_report.py) | 示範 `Result.to_html(...)`。 |
| [`experiments/simulation_data1/`](./experiments/simulation_data1/) | 乾淨 synthetic oracle 的 DoE 方法與知識比較研究。 |
| [`experiments/simulation_data2/`](./experiments/simulation_data2/) | 離散、限制式、接近實驗室情境的 simulation study。 |
| [`docs/superpowers/specs/`](./docs/superpowers/specs/) | 未來 simulation datasets 的規劃文件。 |

首頁 README 只保留實驗入口與簡短說明。詳細方法、表格與結論放在 [`experiments/`](./experiments/)，特別是 [`experiments/README.md`](./experiments/README.md) 與各 simulation dataset 的 README。

## 使用建議

- 沒有明確物理假設時，先使用 `Campaign(space)`，也就是純 GP。
- 有強假設時再加入領域知識，例如「溫度提高產率」或「pH 約在 7 達峰值」。
- `with_random_augment(...)` 視為探索性正則化，不要當作預設。
- 實驗預算有限時，主動優化的因子數不要太多。
- 需要放入實驗紀錄或分享給合作者時，使用 `result.to_html(...)` 輸出報告。

## 路線圖

| 版本 | 新增功能 | 狀態 |
|------|----------|------|
| v0.1 | 限制式 DoE、知識組合、Campaign loop、第一個範例 | 已發布 |
| v0.2 | 單調性與 frozen-mean shape 的經驗驗證器 | 已發布 |
| v0.3 | monotone knowledge + GP prior 的 epsilon 自動救援 | 已發布 |
| v0.4 | 單檔 HTML report | 已發布 |
| v0.5 | Claude Code skill 封裝 | 規劃中 |
| v0.6 | MCP server 介面 | 規劃中 |
| v0.7 | 多目標 BO | 規劃中 |
| v0.8 | 多保真度 BO | 規劃中 |
| v1.0 | 穩定 API 並移除 legacy shim | 規劃中 |

## 授權

Apache License, Version 2.0。詳見 [`LICENSE`](./LICENSE) 與 [`NOTICE`](./NOTICE)。

歷史檔案 `ax_doe_bo.py`、`doe_utils.py`、`benchmarks.py` 原為 MIT 授權，保留作為重現用途；原始條款可在 git 歷史中查到。
