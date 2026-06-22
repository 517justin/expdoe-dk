# expdoe-dk

**實驗設計 (DoE) + 貝葉斯優化 (BO)，支援領域知識注入 — 專為化學、材料及實驗研究者打造。**

如果你是化學家或材料研究者，需要在幾十次實驗內找到最佳條件，這個函式庫提供：

1. **支援限制條件與離散刻度的 DoE** — 初始設計自動滿足「A 必須 ≥ B + 1 mL」、「這個旋鈕只能以 0.5 mL 為單位調整」等限制，不需要事後手動取整。
2. **貝葉斯優化** — 在初始 DoE 之後用高斯過程 (GP) 代理模型驅動後續實驗。
3. **領域知識注入** — 告訴優化器「溫度提高產率 (Arrhenius)」或「pH 在 7 時達到峰值」，它會利用這些提示而非與你對抗。

```python
import expdoe_dk as ed

space = ed.Space(
    params=[
        ed.Parameter("T",      bounds=(60, 120), unit="°C"),
        ed.Parameter("time",   bounds=(10, 180), unit="min"),
        ed.Parameter("conc_A", bounds=(1, 10), unit="mL", kind="discrete", step=1.0),
        ed.Parameter("conc_B", bounds=(1, 10), unit="mL", kind="discrete", step=1.0),
    ],
    constraints=[
        ed.LinearConstraint(coeffs={"conc_A": 1, "conc_B": -1}, lower=1.0),
    ],
    objectives="yield_pct",
    maximize=True,
)

knowledge = (ed.Knowledge()
             .with_arrhenius("T")
             .with_monotone("time", effect="increases_objective")
             .with_quadratic_peak("conc_A", center=7.0))

campaign = ed.Campaign(space, knowledge, seed=42)

doe   = campaign.suggest_doe(n=12)         # 回傳 DataFrame，單位為 °C / min / mL
y_doe = run_lab_experiments(doe)           # 化學家實測
campaign.tell(doe, y_doe)

for _ in range(20):
    next_pts = campaign.ask(q=1)
    y_next   = run_lab_experiments(next_pts)
    campaign.tell(next_pts, y_next)

result = campaign.finalize()
result.to_html("campaign_report.html")     # 可分享的 HTML 報告
```

套件位於 [`expdoe-dk/`](./expdoe-dk/)。歷史研究框架（Ax+BoTorch wrapper）保留在 `ax_doe_bo.py` / `doe_utils.py` / `benchmarks.py` 供重現；新工作請使用 `expdoe-dk`。

---

## 安裝

```bash
cd expdoe-dk
pip install -e .          # 可編輯安裝
```

需要 Python 3.10+、BoTorch >= 0.11、Ax >= 1.2.4。

---

## 目錄結構

```
expdoe-dk/                          # ★ 可發布的 Python 套件
  src/expdoe_dk/
    space.py                        # Parameter, LinearConstraint, Space
    doe/                            # 6 種 DoE 方法 (LHS maximin / Sobol / Halton / ...)
    knowledge/                      # 知識組合 + 座標翻譯器
    bo/                             # Campaign + HTML 報告
    legacy/                         # ax_doe_bo 向後相容 shim
  tests/                            # 53 個單元+整合測試
  LICENSE / NOTICE                  # Apache 2.0
  pyproject.toml                    # 建置 + 依賴規格

examples/                           # ★ 化學家導向的使用示範
  01_reaction_optimization.{py,ipynb}   # 化學工作流端對端示範
  02_html_report.py                     # v0.4 HTML 報告示範

experiments/                        # ★ 基於套件的可重現研究
  01_doe_method_comparison.py           # 6 種 DoE 方法 × 標準 2D/4D/6D
  02_knowledge_comparison.py            # 5 類知識配置 × 標準 2D/4D/6D
  _oracles.py                           # 共用的反應/製程目標函數
  README.md                             # 結果表格 + 解讀
```

---

## 範例與實驗

| 路徑 | 功能 |
|------|------|
| [`examples/01_reaction_optimization.py`](./examples/01_reaction_optimization.py) | 化學家端對端執行 DoE → BO + 知識注入，23 次評估找到真正最佳值 |
| [`examples/02_html_report.py`](./examples/02_html_report.py) | 重現 v0.4 HTML 報告 (`Result.to_html`) |
| [`experiments/01_doe_method_comparison.py`](./experiments/01_doe_method_comparison.py) | 固定知識設定，比較 DoE 方法對 BO 結果的影響 |
| [`experiments/02_knowledge_comparison.py`](./experiments/02_knowledge_comparison.py) | 固定 DoE 方法，比較知識注入類型的效果 |

執行方式：

```bash
python examples/01_reaction_optimization.py
python experiments/01_doe_method_comparison.py
```

---

## 實驗結果摘要

使用三個合成化學目標函數 (2D / 4D / 6D) 測試，統一預算
n_doe=6、n_iter=15（共 21 次評估），5 個隨機種子。完整表格見
[`experiments/README.md`](./experiments/README.md)。

| 配置 | 2D | 4D | 6D |
|------|:--:|:--:|:--:|
| **A: 純 GP** | 第 3 (gap 0.0005) | 第 3 (0.0092) | 第 2 (0.1480) |
| ① 完整領域知識 | 第 2 (0.0003) | 第 5 (0.0256) | **第 1 (0.1012)** |
| ③ 僅 GP 先驗 | **第 1 (0.0002)** | 第 2 (0.0087) | 第 3 (0.1782) |
| G: 錯誤方向 | 最後 | 最後 | 最後 |

**結論：** 純 GP 是安全的預設選擇（所有維度都在前三名）。領域知識在資料相對維度不足時幫助最大（6D）。堆疊多個知識原語可能有害（① 在 4D 崩潰）。錯誤方向的先驗穩定墊底。

---

## 知識類別

| 類別 | API | 適用時機 |
|------|-----|----------|
| ① 領域知識（正確） | `with_arrhenius`、`with_quadratic_peak`、`with_monotone` | 高維問題且有已知物理/化學知識 |
| ② 純正則化（未驗證） | `with_random_augment(n=...)` | 探索性選項 — 預算緊時可能有害 |
| ③ 弱知識（僅 GP 先驗） | `with_gp_prior("medium")` | 低維問題；超參數調整提示 |
| ④ 避免使用（可學習均值） | `with_arrhenius(frozen=False)`（會觸發警告） | 請改用 frozen 版本 |
| ⑤ 避免使用（單調 + 先驗） | `with_monotone(epsilon=0.02)` + 強先驗 | 已有自動救援機制 |

如果沒有特定知識，保守的預設選擇是 **純 GP**（`Campaign(space)` 搭配 `knowledge=None`）。

---

## 安全預設行為

| 陷阱（經實驗發現） | expdoe-dk 的處理方式 |
|-------------------|---------------------|
| 單調方向在使用者/產率空間定義，但 BO 最小化 `-yield`，GP 看到的方向相反 | `with_monotone(effect="increases_objective")` 使用物理空間語義；`_frame.flip_for_minimize` 在內部自動翻轉 |
| `MonotonicGPWithDerivatives` 的 epsilon=0.02 與 Gamma(3,6) lengthscale 先驗衝突，導致 13 倍劣化 | `epsilon="auto"` 解析為 `0.3 x prior_lengthscale_mode`；明確設定過小的 epsilon 會自動救援 (v0.3) |
| 可學習均值參數被 MLE 吸收，均值函數失去作用 | `Arrhenius`、`QuadraticMean` 預設 `frozen=True`；可學習版本會發出 `LearnableMeanAbsorptionWarning` |
| 錯誤的單調假設無聲地傷害結果 | 每 K 次觀測後 Campaign 執行 Spearman 檢驗，不一致時發出 `MonotoneViolationWarning` (v0.2) |
| 未給定任何知識 | Campaign 執行 **純 GP** — 不會自動注入任何結構。保守、無意外的預設 |
| 把 `with_random_augment` 當作「免費」預設 | 這是純正則化，效益**仍在驗證中**。函式庫不會自動套用 — 必須手動啟用 |

---

## 路線圖

| 版本 | 新增功能 | 狀態 |
|------|---------|------|
| v0.1 | 限制式 DoE + 知識組合 + Campaign 迴圈 + 1 個範例 | [已發布](https://github.com/517justin/expdoe-dk/releases/tag/v0.1.0) |
| v0.2 | 經驗驗證器（Spearman 單調檢驗 + frozen-mean 形狀檢驗）每 K 次觀測自動執行 | [已發布](https://github.com/517justin/expdoe-dk/releases/tag/v0.2.0) |
| v0.3 | epsilon 自動救援：`with_monotone` + `with_gp_prior` 自動提升 epsilon 至安全值 | [已發布](https://github.com/517justin/expdoe-dk/releases/tag/v0.3.0) |
| v0.4 | HTML 報告 (`Result.to_html()`) | [已發布](https://github.com/517justin/expdoe-dk/releases/tag/v0.4.0) |
| v0.5 | Claude Code skill 封裝（`.claude/skills/`、無狀態 API） | 待開發 |
| v0.6 | MCP server（FastMCP、JSON 工具介面） | 待開發 |
| v0.7 | 多目標優化（qLogEHVI、Pareto 前沿） | 待開發 |
| v0.8 | 多保真度 BO（低成本篩選 → 高成本實驗，MFKG） | 待開發 |
| v1.0 | 穩定 API、移除 legacy shim | 待開發 |

---

## 授權

Apache License, Version 2.0 — 見 [`LICENSE`](./LICENSE) 及 [`NOTICE`](./NOTICE)。

此目錄中的歷史程式碼（`ax_doe_bo.py`、`doe_utils.py`、`benchmarks.py`）原為 MIT 授權；改版後統一採用 Apache 2.0。原 MIT 條款保留在 git 歷史中。
