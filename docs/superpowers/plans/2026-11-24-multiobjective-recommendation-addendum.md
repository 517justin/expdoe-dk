# Multi-objective Recommendation Addendum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 依目前 main 補齊多目標推薦、量測與存檔流程，保留已實作的單目標行為、候選清單及限制檢查。

**Architecture:** 擴充現有 `surrogates.py`、`recommendation.py`、`contracts.py`、`loop.py`，只將新增的多目標設定、目標尺度、推薦方法與結果指標分成獨立模組。各目標沿用既有物理反應 GP，推薦時才把 posterior samples 轉成效用；不重建舊計畫中的整套 pipeline。

**Tech Stack:** 現有 Python、PyTorch、BoTorch、GPyTorch、NumPy、pandas、SciPy、pytest；正式支援版本以 Task 10 的環境矩陣驗證，不以最新文件代替相容性證據。

**Spec:** [原始設計](../specs/2026-08-05-generalized-knowledge-and-optimization-design.md)、[原 optimization pipeline 計畫](2026-08-05-optimization-pipeline-implementation-plan.md)、[原驗證計畫](2026-08-05-validation-and-benchmark-matrix-implementation-plan.md)，以及本文件「設計補正」。有衝突時以經使用者確認後的本增補為準。

**Status:** 使用者於2026-11-24確認本計畫，指定subagent-driven-development逐項實作，完成驗收後更新中英文README與版本至0.6.0。實作進行中；未通過的功能不得標成已驗收。

**Baseline:** 2026-11-24 檢視本機 main `197cf49b64fa2ac7aaef1998f41aeaab7e72bccf`，套件 `0.5.1`，checkpoint schema `2.1`。前一輪完整測試紀錄為 1,328 passed / 117 warnings；不是本計畫的新驗收結果。

## Global Constraints

- 不破壞單目標 `tell(X, y)`、`ask(q)`、`recommend()`、預測物理單位及既有結果 JSON；新參數只能是 keyword-only。
- 原定方法保留：qLogEHVI、qLogNEHVI、qLogNParEGO-style。明確偏好是可解釋的路由條件，不得默默只最佳化第一個目標。
- 不把預測值當成實測回填、不以測試／oracle 真值擬合尺度，不以 Ridge／RF 或隨機配方作推薦失敗備援。
- 回傳恰好 q 個唯一可行條件；不足則 `NO_FEASIBLE_CANDIDATE`，數值失敗不得偽裝成無解。
- 固定清單保留原始數值、資料型別與 candidate ID，沿用已測／待測排除與座標碰撞檢查；不展開 Cartesian product。
- 物理硬限制不能被偏好、權重或目標轉換放寬；量測結果安全機率不是實驗安全保證。
- 單目標目前的 BoTorch 下限與使用方式不得因新功能的頂層 import 被破壞；多目標缺必要 API 時先回報 typed capability error。
- 不重寫歷史設計與歷史驗收結果。README 與版本更新是最後的發布關卡，不能把「計畫完成」寫成「功能完成」。
- 本輪依使用者確認開始實作；建立獨立worktree，逐Task RED/GREEN、review、commit。Push、tag、發布套件另外取得授權。

## Review Focus

1. 同名目標被重排、缺欄或 Yvar 錯位：按名稱對齊或明確拒絕，不能把成本變異套到產率。Task 1、3、7。
2. target／target-range 不可逆、常數目標及極端單位：建模與報告維持物理值，尺度有限，不能由效用猜回實測值。Task 3、4。
3. 偏好改變或 noisy／pending 分支繞過安全限制：每條 acquisition 路徑都接限制，最後做相同物理檢查。Task 5、6。
4. 存檔還原造成尺度、參考點、權重種子或 catalog ID 改變：重新建模後結果可重現，損壞狀態在變更 campaign 前拒絕。Task 7、8。
5. 歷史多目標宣告其實只存第一目標，或新 API 不被舊依賴支援：不能捏造其餘量測，不能讓單目標 import 失敗。Task 1、8、10。

## Roadmap 核對

沒有獨立的現行 roadmap 檔；目前路線分散於 2026-08-05 設計／計畫、README 與 `docs/textbook-improvements.md`。

| 階段 | 實際狀態 | 本增補的處置 |
| --- | --- | --- |
| 原 0.5.0 全功能願景 | 包含多目標，但不等於目前均已發布 | 保留歷史內容；不得按版本號推定完成 |
| 現行 0.5.1 | 單目標 GP、knowledge、混合空間、候選清單、等式補強已進 main | 作回歸基線，不重做 |
| 多目標模型與資料契約 | 尚未接通正式推薦 | Task 1–4 |
| 多目標推薦與候選安全 | 尚未實作 | Task 5–6 |
| Campaign、checkpoint、Pareto／HV 報告 | 尚有單目標限制 | Task 7–9 |
| 驗收、雙語 README、版本與套件發布準備 | 待上述功能通過 | Task 10；提議下一功能版為 0.6.0，待確認 |

對應舊計畫：模型 Task 2 → 本 Task 2–3；acquisition Task 3 → 本 Task 4–5；搜尋 Task 4–5 → 本 Task 6；Campaign Task 6 → 本 Task 7；checkpoint Task 7 → 本 Task 8；結果 Task 8 → 本 Task 9。原驗證 Task 2／5／7 → 本 Task 9–10。

### 已有實作與不可照抄處

- 現行 `bo/candidates.py` 負責固定清單驗證，不能照舊計畫覆寫成搜尋 dispatcher。
- 現行檔名是 `bo/surrogates.py`，不是舊計畫的 `surrogate.py`；已有物理 posterior adapter 和 structured knowledge fitting，必須沿用。
- `Prediction` 只接受 `(n, 1)`；`fit_surrogate`、`recommend`、`tell_batch`、`set_pending`、checkpoint 明確限制單目標。
- `Campaign.run()` 會 flatten oracle 輸出；`finalize()` 與 HTML 報告使用單一最佳值，這些都必須改，不只拿掉 recommendation guard。
- `inspect_capabilities()` 目前對多目標的所有啟用 knowledge 直接阻擋；v2 形狀知識也有單目標檢查，需按目標投影後驗證，不能全部放行。
- 版本分散於 `pyproject.toml`、`__init__.py`、`domain/space.py`，並影響 knowledge／checkpoint allow-list 和 release 測試。

## 設計補正（待確認提案）

### A. 採用的實作方式

建議沿現有模組增量擴充：保留單目標路徑，讓多目標共用候選池、最後篩選與物理 posterior。替代方案是照 8 月計畫重建整個 pipeline，會同時移動已驗證的 catalog、structured GP 與 checkpoint，回歸面較大；只刪除 guard 則無法處理資料、報告與存檔。本文件採增量方案作待核准提案。

### B. 公開 API 與資料形狀

新增匯出的 frozen `MultiObjectiveConfig`，只用於 `m >= 2`：

```python
MultiObjectiveConfig(
    acquisition="auto",             # auto | q_log_ehvi | q_log_nehvi | q_log_nparego
    noise_mode="auto",              # auto | inferred | known | noiseless
    reference_point=None,            # Mapping[objective_name, physical_value]
    preference_weights=None,         # Mapping[objective_name, positive_weight]
    min_joint_feasibility=0.95,
)
recommend(..., multiobjective=None)   # 其他參數保持現狀
Campaign(..., multiobjective=None)    # 新增 keyword-only
Campaign.ask_with_diagnostics(..., candidates=None, candidate_ids=None)
Campaign.ask(..., candidates=None, candidate_ids=None)
```

所有 mapping 複製成不可變內部表示；名字必須恰好對應 `space.objectives`，拒絕 NaN、inf、bool 數值及未知欄位。單目標傳非空 config 明確報錯，不默默忽略。

成功的量測列必須有全部 m 個有限反應；至少三列成功量測才推薦。Yvar 若提供，成功列全部 m 個值必須非負且有限。首版不支援「同一 trial 只回填其中部分目標」，也不支援成功列混合已知／缺失變異。不同 trial 分批回填仍支援。失敗／warning 的缺值保留，但不訓練。

入口按名稱重排 Y／Yvar，缺欄、多欄、重複欄名皆錯誤；不得按位置猜測。Prediction 擴充為 `(n, m)`，單目標 JSON 不增加欄位；多目標結果在 diagnostics 明確列 `objective_names`。物理量測與預測仍分開存。

### C. 模型與 knowledge

每個目標用目前的單目標物理 GP 建模，保留各自訓練統計與 Yvar。合併各物理 posterior adapter 為 BoTorch `ModelList`，使用支援 `PosteriorList` 的 sampler，`cache_root=False`；不把 generic adapter 強塞進要求 GP 介面的 `ModelListGP`。獨立輸出模型不宣稱學到跨目標相關性。

目標 min/max/target/target-range 的效用在 posterior sample 上計算，不能先轉 target utility 再聲稱可反轉物理反應。Yvar 僅隨實際 GP 的仿射標準化以 scale 平方換算。

新增 `project_objective(space, observations, knowledge, objective_name)`：回傳一個單目標 Space、單欄 ObservationBatch 與該目標 Knowledge。參數定義和編碼保持一致；outcome constraints 與 acquisition preferences 留在全域推薦層，不塞進個別 GP。

- 有明確 scope 的 mean／virtual observations／prior 只作用到對應目標；跨多目標 scope 分別投影，但每個目標仍需通過現有驗證。
- 沒有 objective scope 的 response-dependent mean／virtual observations／prior 在多目標模式拒絕，避免同一個物理假設被默默套到所有反應。
- 只影響輸入編碼的 categorical similarity／硬參數限制可共用；v2 pattern 的投影保留原 pattern ID 和 provenance，不能改成不明來源的暫時 ID。
- 全域 `target_range` 是軟偏好：在對應目標原始效用加上現有負區間距離，不變成硬性安全排除。scope 外的目標不得受影響。
- `objective_priority`、`tradeoff` 及 `Objective.priority` 必須真的有數值 consumer；unknown artifact 保持 fail closed，不藉放行多目標而接受其他未支援 pattern。

### D. 固定尺度、參考點與偏好

新增 `ObjectiveGeometry`，在第一次成功建立推薦狀態時由成功量測計算並固定：每個目標原始效用的 min 作 location，range 作 scale。常數列用 `max(abs(location), 1.0)`，不除以零；不剪裁後續超過初始範圍的效用。尺度與初始資料 digest 一起存檔。

預設 report reference 為初始「實測且符合硬限制」效用逐欄最小值減 `0.1 * scale`。沒有任何實測可行 baseline 時沿用 `INSUFFICIENT_DATA`，details.reason為 `no_feasible_baseline`，不虛構現有enum。使用者提供的 reference 是具名物理值，經各 Objective 轉成效用並套相同尺度；要求至少一個可行 baseline 在全部目標嚴格優於 reference，否則 CONFIG_INVALID，不自動搬動 reference。

維持兩套明確具名的值：`report_reference` 使用原始 Objective utility；`acquisition_reference` 使用已加入 target_range 軟偏好的 utility。兩者以相同初始資料、固定 scale 求得並持久化。診斷列出物理參考值（若使用者有提供）、效用參考值、尺度與產生方法；自動推定的效用 reference 不聲稱能反推出物理值。

無偏好時，2–4 目標用 HV 路徑，>=5 目標用 seeded simplex 權重的 NParEGO-style。不要只乘上正權重後宣稱 weighted HV 能改變取捨排序。

明確偏好時使用 NParEGO-style，包括 2–4 目標；diagnostics 記錄 `preference_override`。來源依序是 config、knowledge、Objective.priority，但多個有效來源不默默覆蓋：相同正規化權重可合併 provenance，不同則 CONFIG_INVALID。knowledge 首版只接受涵蓋所有目標且每個權重 > 0 的單一有效偏好；較窄 scope、零權重及多組衝突明確 blocked。這比 registry 接受的語法窄，需清楚報告。

若所有 priority 相同，視為無明確偏好；否則使用 `1 / (1 + priority)` 後正規化，代表軟權重，不是字典序優先。這是待確認的新增規則。明確偏好權重固定；無偏好 NParEGO 每個 greedy slot 以 campaign iteration seed 與 slot index 產生可重現的新權重並記錄。

首次解析後，Campaign 的設定／尺度不可原地變更；要換偏好或 reference 需另建 campaign 並明確導入觀測。獨立 `recommend()` 可每次解析，但 diagnostics 必須回傳完整 geometry；不可把不同 geometry 的 HV 當同一歷史比較。

### E. 路由、噪音與安全

| 條件 | auto 路由 |
| --- | --- |
| 2–4 目標，明確 noiseless、無 pending、無偏好 | qLogEHVI |
| 2–4 目標，inferred／known noise 或有 pending、無偏好 | qLogNEHVI |
| >=5 目標，或有明確偏好 | qLogNParEGO-style |

修正原示例把「沒有 Yvar」視為無噪音的歧義：auto有完整Yvar解析為known，否則inferred，並非noiseless；顯式inferred遇到已提供Yvar拒絕，known必須有完整Yvar。noiseless若提供Yvar必須全零，未提供時只在模型內建立零噪音設定，不捏造原始量測metadata。顯式 qLogEHVI override 搭配 noisy／pending 必須拒絕；顯式 HV override 搭配非等權偏好也拒絕。單目標路由不變。

所有方法都接收原始物理 samples 上的 outcome constraint callables，採 BoTorch `g(samples) <= 0` convention，最後篩選仍檢查物理限制。首版 outcome safety 限制為目前可支援的具名目標硬性 `<=`／`>=` 區間；軟性OutcomeConstraint與`==`維持不支援，knowledge target_range的軟偏好則依前述獨立規則處理。同一目標多個區間先取交集，空交集拒絕；同目標區間機率只計一次，再乘各獨立目標機率作 joint feasibility。門檻預設 0.95。跨輸出非線性限制維持明確不支援，不能直接把各條 constraint 的機率相乘。

model-based joint feasibility 與 oracle 真實可行率分別報告。即使篩選機率 >=0.95，也不能宣稱新量測 100% 安全；驗收 100% 指測試中指定的硬參數限制與已知 oracle 可行性案例。

### F. Campaign、存檔與結果

- observation batches 是權威資料；多目標不得維持只含第一目標的私有列表再當成完整歷史。舊單目標欄位與亂數進程保留。
- 多目標Campaign首次成功的catalog ask綁定清單與ID；之後未傳catalog時重用，傳入重排列但ID對應相同可接受，換配方／ID則拒絕並要求新campaign。不以修改清單破壞resume；首次ask失敗不提交catalog狀態。直接`recommend()`仍為無狀態呼叫。
- `tell(X, y)` 對單目標保持不變；多目標只接受具名 DataFrame 或 `tell_batch`。`run()` 的多目標 oracle 必須回傳完整具名 DataFrame，禁止 flatten。
- history 用具名目標欄位；多目標不造出單一 best_y。`Result` 的既有 best 欄位在多目標為 null，新增 Pareto rows、固定尺度 HV history、diagnostics 與 advisory stop signals。單目標 `to_dict()` 保持原鍵集合。
- 提議 schema `2.2` 僅用於多目標 campaign：保存 config、geometry、catalog 與 candidate IDs（若有）、trial metadata、全部 Y／Yvar、pending、iteration／RNG。單目標仍寫 2.1，讀者精確支援 2.1 與 2.2。
- 2.0 歷史 archive 只存第一目標的情況保持 archival-only；無法升級完整多目標 campaign，不補零、不補預測、不丟目標。
- Pareto／HV 僅使用成功且實測可行資料，去除重複效用後計算 HV；保留所有 trial 稽核。報告 noisy-observed HV，不宣稱是真實 Pareto front。預測 Pareto（若提供）須另標示，首版不列為必要。
- 有固定 geometry 且連續 5 個更新的 HV 相對增益都 <=1e-3 時給 advisory plateau；分母 `max(abs(previous_hv), 1e-12)`。不自動中止 campaign，無資料不發 plateau。既有單目標 stopping 行為不擴寫。

## File Responsibility Map

以下 `src/` 與 `tests/` 均相對 repo 內層 `expdoe-dk/`。

| 檔案 | 處置與責任 |
| --- | --- |
| `src/expdoe_dk/bo/multiobjective.py` | 新增 config、route decision、immutable resolved state；不放 GP fitting |
| `src/expdoe_dk/bo/objective_geometry.py` | 新增向量 utility、固定尺度、reference 與偏好解析 |
| `src/expdoe_dk/bo/objective_routing.py` | 新增按 objective 投影 observation／knowledge／space |
| `src/expdoe_dk/bo/acquisition.py` | 新增多目標 acquisition constructors 與 typed capability checks |
| `src/expdoe_dk/bo/multiobjective_metrics.py` | 新增 Pareto、HV、regret、plateau 純函式 |
| `src/expdoe_dk/bo/contracts.py`, `surrogates.py` | 擴充多輸出物理預測與 model adapter |
| `src/expdoe_dk/bo/knowledge_runtime.py`, `capabilities.py` | 按目標／artifact 回報支援，不全域放行 |
| `src/expdoe_dk/bo/recommendation.py` | 接入 acquisition；共用 catalog、線性搜尋與最後檢查 |
| `src/expdoe_dk/bo/loop.py`, `checkpoint.py`, `report.py` | 完整量測、重現、保存與多目標報告 |
| `src/expdoe_dk/__init__.py`, `bo/__init__.py` | 公開 config；不要不必要地匯出內部 helpers |
| `tests/multiobjective/` | 新增明確目標的回歸測試，不取代 phase_one tests |
| `experiments/multiobjective_validation/` | 固定 oracle／seed／預算的實測驗收與結果摘要 |
| `docs/roadmap.md`, root READMEs、版本來源與 release tests | Task 10 的發布更新，不在功能完成前標成已提供 |

## 執行順序與共同測試環境

Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10。Task 4 的純函式可在 Task 1 後獨立開發，但 API 先固定；同一個 recommendation／loop 檔不要由多個 agent 同時修改。

從內層套件執行，使用指向執行 worktree 的虛擬環境與 editable install：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../.venv/bin/python -m pytest -q tests/multiobjective
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../.venv/bin/python -m pytest -q --run-slow
```

若系統 Git 仍受 Xcode 授權提示阻擋，將已存在且可用的 fallback Git 放到 PATH；不要略過依賴 Git 的驗收。先記錄實際 Python／dependency／source 路徑，不覆寫舊 DOEGP conda editable install。

## Task 1: 設定與多輸出結果契約

**Files:** Create `bo/multiobjective.py`；Modify `bo/contracts.py`、public exports；Test `tests/multiobjective/test_contracts.py`、既有 `tests/phase_one/test_capabilities.py`。

**Interfaces:** 產出上述 `MultiObjectiveConfig`，提供 `to_dict()`／`from_dict(payload)`；`Prediction(mean, variance)` 允許 `(n,m)`、m>=1，仍深複製且唯讀；新增 `AcquisitionDecision(kind: str, reason: str)` frozen dataclass。

- [ ] RED：新增 `test_prediction_keeps_two_physical_outputs`，assert `Prediction(np.zeros((3,2)), np.ones((3,2))).mean.shape == (3,2)`；測試 shape mismatch、負 variance、NaN、回傳陣列修改不污染內部。新增 config unknown field、錯名字、bool、非有限值、權重 <=0、單目標誤用案例。
- [ ] 執行 `pytest -q tests/multiobjective/test_contracts.py`，確認失敗來自缺介面／舊單欄限制。
- [ ] GREEN：實作設定驗證與 contracts；單目標 Prediction／RecommendationResult JSON 快照不改；多目標 objective_names 由推薦結果 diagnostics 給出，不重複藏在 unnamed ndarray。
- [ ] 跑新測試與 `tests/phase_one/test_capabilities.py`；修改預期只能對應新增契約，不能刪舊單目標斷言。
- [ ] Review 後 commit：`feat: define multiobjective recommendation contracts`。

## Task 2: Knowledge 的目標路由

**Files:** Create `bo/objective_routing.py`；Modify `bo/capabilities.py`、`bo/knowledge_runtime.py`；Test `tests/multiobjective/test_knowledge_routing.py`；沿用 `tests/knowledge/` 與 `tests/phase_one/test_knowledge_runtime.py`。

**Interfaces:** `project_objective(space, observations, knowledge, objective_name) -> tuple[Space, ObservationBatch, Knowledge]`；`inspect_capabilities` 回傳型別不變，diagnostics 增加 objective_name／scope；`compile_runtime` 的全域 constraints／preferences 不變成單一輸出內容。

- [ ] RED：`test_scoped_mean_does_not_leak_to_other_response` assert 產率投影包含原 pattern ID、成本投影不含它；`test_v2_projection_retains_provenance` assert saturation@2.0 能在其單目標投影驗證；無 response scope 必須 blocked，不能假定所有輸出共用同一規律。
- [ ] 執行 `pytest -q tests/multiobjective/test_knowledge_routing.py`，確認現行全域多目標阻擋造成預期失敗。
- [ ] GREEN：只對現有 consumer 可用的 pattern 投影；新增 `tradeoff_preference` 的嚴格 payload 驗證。全域安全限制只編譯一次。scope、priority、target_range 衝突在建模前報錯，錯誤帶原 pattern IDs。
- [ ] 測試 shape／physics／prior／category similarity 的代表案例與不支援案例；執行上述 knowledge 回歸集。assert 同一 knowledge 在 reorder objectives 後仍作用到正確名稱。
- [ ] Review 後 commit：`feat: route knowledge by objective scope`。

## Task 3: 多目標物理 GP 與 posterior

**Files:** Modify `bo/surrogates.py`；Test `tests/multiobjective/test_surrogates.py`；必要時在同檔抽出既有單輸出 fitting helper，不重寫 `bo/gp.py`。

**Interfaces:** `fit_surrogate(space, observations, *, knowledge=None, seed=0)` 簽名保留，回傳仍符合 `FittedSurrogate.predict(X) -> Prediction`、`acquisition_model() -> Model`、diagnostics／transforms／exclusions。多目標 wrapper 持有各單目標 fitted objects，不複製一份 GP 實作。

- [ ] RED：`test_named_outputs_align_y_and_yvar` 用 Y 欄位逆序 assert 預測仍按 space 名稱；`test_target_model_preserves_both_sides` 用 target=5、原始觀測3及7，assert 訓練及物理預測未先折成相同 utility；變異 scale 平方與常數輸出皆有限。
- [ ] 執行 `pytest -q tests/multiobjective/test_surrogates.py`，確認舊單目標 guard 失敗。
- [ ] GREEN：Task 2 投影後逐目標沿用現有 fitting；組合物理 adapters，不暴露內部標準化反應。部分模型失敗時整體 MODEL_FIT_FAILED，diagnostics 指明 objective，不回傳殘缺模型。
- [ ] 測試後驗 sampling 形狀 `sample_shape x batch x q x m`、輸入 gradient 有限、跨 q 聯合抽樣、不固定成獨立點邊際；Yvar 按目標保留。驗證缺 BoTorch sampler API 會 typed error，不影響單目標 import。
- [ ] 跑新測試與 `tests/phase_one/test_evaluation.py`；evaluation API 在本版仍單目標，須在入口明確拒絕多目標，不能放寬 fit 後使既有 metrics 偷取第一欄。新增此回歸。
- [ ] Review 後 commit：`feat: fit physical multioutput GP posteriors`。

## Task 4: 固定效用尺度、參考點與偏好

**Files:** Create `bo/objective_geometry.py`；Modify `bo/multiobjective.py`；Test `tests/multiobjective/test_geometry.py`。

**Interfaces:** `resolve_geometry(space, observations, runtime, config) -> ObjectiveGeometry`；frozen geometry 含 objective_names、location、scale、report_reference、acquisition_reference、acquisition_ideal、source_digest、preference_weights／provenance，支援嚴格 `to_dict/from_dict`。acquisition_ideal為初始標準化偏好後utility逐欄最大值，供Task5使用。`objective_samples(samples, space, geometry, runtime, *, with_preferences: bool) -> Tensor` 保留最後 m 維且可微。

- [ ] RED：`test_mixed_directions_and_target_utilities` 用 max=3、min=4、target5的觀測7 assert 原始效用為 `[3,-4,-2]`；`test_fixed_geometry_ignores_future_outcomes` assert 後續更佳值不改 reference／scale；常數列、非有限 reference、錯名字、零權重與衝突來源皆拒絕或依既定 fallback 處理。
- [ ] 執行 `pytest -q tests/multiobjective/test_geometry.py`，確認缺 helper 失敗。
- [ ] GREEN：按設計 D 實作，全部 tensor utility 與 `Objective.to_utility` 的有限輸入數值一致；target_range margin 單獨按其目標作用。明確保存兩種 reference，避免把偏好後 HV 報成原始科學品質。
- [ ] 測試非恆定欄位單位等比例改變並同步變換reference／noise時，正規化效用一致；常數欄位只驗證既定fallback有限，不虛稱在未知物理尺度下單位不變。優先序 `[0,1]` 得 `[2/3,1/3]`；相同來源權重合併、不同來源拒絕；無可行baseline使用`INSUFFICIENT_DATA`及`no_feasible_baseline` reason。
- [ ] Review 後 commit：`feat: resolve fixed multiobjective geometry`。

## Task 5: 推薦路由與安全限制

**Files:** Create `bo/acquisition.py`；Modify `bo/multiobjective.py`、`bo/knowledge_runtime.py`；Test `tests/multiobjective/test_acquisition.py`。

**Interfaces:** `choose_multiobjective_acquisition(n_objectives, config, *, has_pending, has_preferences) -> AcquisitionDecision`；`build_multiobjective_acquisition(space, fitted, observations, pending, runtime, config, geometry, *, seed, slot) -> tuple[AcquisitionFunction, dict]`；`joint_feasibility(space, fitted, X) -> np.ndarray`。

- [ ] RED：參數化 m=2,3,4,5,6；inferred／known／noiseless；pending on/off；偏好 on/off，assert 路由與設計 E 完全相同。加入 known 缺 Yvar、noiseless 非零 Yvar、非法 override 測試，不能只測 class 名稱。
- [ ] 執行 `pytest -q tests/multiobjective/test_acquisition.py`，確認缺方法失敗。
- [ ] GREEN：使用 BoTorch log acquisitions；HV objective 用 `GenericMCMultiOutputObjective`；NParEGO 用固定 geometry 下的 Chebyshev scalarization 與 `GenericMCObjective`／qLogNEI，不可再次以全資料動態正規化。明定標準化向量z的分數為 `min_i(w_i * (z_i - ideal_i)) + 0.05 * sum_i(w_i * (z_i - ideal_i))`；ideal取初始標準化acquisition utilities逐欄最大值。以BoTorch scalarization作數值對照，若使用其helper，須證明內部正規化與上述公式等價。相同helper同時處理baseline及samples。
- [ ] 真實小模型跑 forward／backward；所有分支必須傳 physical sample constraints、pending 與 seeded sampler。測試高分但不安全點被抑制、空區間失敗、同目標兩個限制先合併；assert joint feasibility 不等於重複相乘同一目標機率。模型安全不足不得轉 random。
- [ ] 測試顯式偏好確實改變非對稱 tradeoff toy case 排名，且不修改 report_reference；每 slot 權重可重現。使用的 BoTorch 介面不存在時 ACQUISITION_INCOMPATIBLE，訊息含所需能力。
- [ ] Review 後 commit：`feat: add constrained multiobjective log acquisitions`。

## Task 6: 接入現有候選搜尋

**Files:** Modify `bo/recommendation.py`；Test `tests/multiobjective/test_recommendation.py`；回歸 `tests/phase_one/test_candidate_catalog.py`、`tests/phase_one/test_recommendation_constraints.py`。

**Interfaces:** 公開 `recommend(..., multiobjective: MultiObjectiveConfig | None = None)`；新增內部 `_recommend_resolved(..., multiobjective, geometry)` 給 Campaign 共用，參數其餘同公開入口；多目標 diagnostics 包含 geometry、route、objective_names、weights_by_slot、feasibility 及 source IDs。

- [ ] RED：catalog 含混合因素與反應欄洩漏案例；`test_multiobjective_catalog_retains_ids_and_physical_values` assert q=2、原值原 ID 對齊且已測／待測皆排除。加入2目標及5目標 q=2 真實 acquisition，不以 mock 排名取代整合證據。
- [ ] 執行 `pytest -q tests/multiobjective/test_recommendation.py`，確認舊 guard 失敗。
- [ ] GREEN：只在模型與 acquisition／安全計分處分流，沿用 `_search_pool`、catalog、linear_search 與最後物理驗證。外部 pending 及已選點逐 slot 條件化；不得因 EHVI 初始無 pending 而漏掉 batch 中已選點。
- [ ] 測試 continuous、discrete、integer、categorical、ordinal、等式邊界、禁止組合、q耗盡；min/max／target 交叉搭配。精度碰撞仍 CONFIG_INVALID，數值求解失敗仍 typed error，不回傳半批。
- [ ] 跑新測試與完整 phase_one recommendation／catalog／constraints；單目標固定種子結果與診斷保留。
- [ ] Review 後 commit：`feat: integrate multiobjective candidate recommendation`。

## Task 7: Campaign 全資料流程

**Files:** Modify `bo/loop.py`；Test `tests/multiobjective/test_campaign.py`；回歸 `tests/test_campaign_smoke.py`、`tests/test_v04_compatibility.py`、`tests/test_observation_batches.py`。

**Interfaces:** Campaign 保存 config／resolved geometry；多目標 `tell(X, y: pd.DataFrame)` 與 `tell_batch` 共用驗證；`ask`／`ask_with_diagnostics` 轉送 catalog keyword。`run(oracle, ...)` 多目標 oracle 回傳具名 DataFrame，返回同一 Result API。

- [ ] RED：`test_run_never_flattens_multiobjective_response` assert 3列x2目標保持3 trials；`test_failed_ask_does_not_advance_iteration` assert seed計數與 geometry 不變；`test_partial_trial_returns_preserve_other_pending` 驗證只移除已回填 trial，不猜同條件重測的 ID。
- [ ] 執行 `pytest -q tests/multiobjective/test_campaign.py`，確認現行入口拒絕／flatten 問題。
- [ ] GREEN：用 observation batches 作多目標歷史來源；未來 geometry 只在成功 ask 後原子提交。成功 tell 以 observation.ids 精確移除同 ID pending；沒有 ID 不按條件猜測。拒絕成功列缺一個目標，不以 prediction 補缺。
- [ ] 舊單目標 `tell(X, list)` 與 iteration seed 演算法保留；多目標採成功量測列數，不乘上目標數。empirical validators 按 Task 2 投影執行，不能只驗第一目標；不支援的 validator 清楚記錄，不能宣稱已檢驗。
- [ ] 測試 callback 出錯、欄位錯位、重複trial ID、故意重測、failed／warning排除、pending空與多點；固定catalog 的 ask_with_diagnostics 返回 IDs。若要 `run` 使用固定catalog，新增 keyword-only candidates／candidate_ids 並以清單中可行唯一列做初始抽樣，不呼叫空間 DoE 產生清單外配方。
- [ ] Review 後 commit：`feat: support multiobjective campaign observations`。

## Task 8: Checkpoint 與相容性

**Files:** Modify `bo/checkpoint.py`、`bo/loop.py`；Create `tests/multiobjective/test_checkpoint.py`、`tests/fixtures/multiobjective_checkpoint_2_2.json`；回歸 `tests/phase_one/test_checkpoint.py`。

**Interfaces:** schema2.2 包含設計 F 欄位；`validate_state`／`restore`／`to_state` 簽名保留。單目標2.1 write與現有舊archive讀取仍可用；引擎版本與 schema 版本分開處理。

- [ ] RED：`test_resume_next_catalog_batch_and_weights` 連續run vs save/load，assert下一批 candidate IDs、物理值、geometry及權重一致；缺失目標欄、錯長度、非有限reference、重複ID、混合schema欄位皆拒絕。
- [ ] 執行 `pytest -q tests/multiobjective/test_checkpoint.py`，確認目前不接受2.2。
- [ ] GREEN：嚴格 JSON schema／semantic validation、既有 atomic_save。保留 object 數值型別與大整數，禁止內嵌 callable／pickle。保存 catalog config／IDs，未成功 ask 的 unresolved geometry 可為null。
- [ ] 測試2.1歷史正常讀寫、2.0多目標archive不得合成缺值升級、未知future schema／engine拒絕。測試已測與pending同條件不同trial合法、同trial狀態重疊非法。
- [ ] 檢查 package 與 nested Space／Knowledge engine versions 一致；正式0.6.0 allow-list擴充留到 Task10，先保持目前版本常數，fixture中的engine_version使用當前值。
- [ ] Review 後 commit：`feat: persist reproducible multiobjective campaigns`。

## Task 9: Pareto、HV 與可理解的結果

**Files:** Create `bo/multiobjective_metrics.py`；Modify `bo/loop.py`、`bo/report.py`；Test `tests/multiobjective/test_metrics.py`、`tests/multiobjective/test_report.py`、既有 `tests/test_html_report.py`。

**Interfaces:** `pareto_mask(utilities: np.ndarray) -> np.ndarray`；`hypervolume(utilities, reference) -> float`；`hypervolume_regret(observed, oracle, reference) -> float`；`plateau_signal(history: Sequence[float]) -> dict | None`。使用 BoTorch 非支配判斷與 Hypervolume，不手寫幾何引擎。Result 新增 `pareto_rows`、`hypervolume_history`、`multiobjective_diagnostics`、`stop_signals`；只有多目標才加新 JSON 鍵。

- [ ] RED：`test_exact_two_dimensional_hypervolume` assert `HV([[1,2],[2,1]], [0,0]) == 3`；加入重複、被支配、reference下方、空集合、NaN拒絕。`test_multiobjective_result_has_no_scalar_winner` assert best_y／best_x為null而非第一欄最佳。
- [ ] 執行 `pytest -q tests/multiobjective/test_metrics.py tests/multiobjective/test_report.py`，確認缺介面失敗。
- [ ] GREEN：以固定 report geometry 的原始utility計算，不把偏好權重或 posterior mean當量測。噪音資料標為observed；oracle regret僅benchmark可算，不出現在一般campaign冒充已知真值。若尚未ask就finalize，有可行資料則以同一resolve規則建立報告geometry；無可行資料則空Pareto、HV=0、reference unresolved，不聲稱有可比較的HV歷史。observed集合若超過宣稱oracle HV超過1e-10，驗證程式報錯，不以截零掩蓋reference錯配。
- [ ] HTML／JSON顯示逐目標名稱、方向、單位、Pareto配方及trial ID、reference／scale／route。可行量測集合為空時回空Pareto與HV=0並標註，不捏造最佳值。單目標報告快照不改。
- [ ] plateau用設計F規則；測試5次低增益只給advisory，早期或資料不足不停止。報告target-range與knowledge軟偏好分開，確保使用者能看懂哪些是量測、哪些是模型判斷。
- [ ] Review 後 commit：`feat: report Pareto outcomes and hypervolume progress`。

## Task 10: 驗收、Roadmap、README 與版本

**Files:** Create `experiments/multiobjective_validation/{fixtures.py,run.py,README.md}`、`tests/multiobjective/test_acceptance.py`、`docs/multiobjective.md`、`docs/roadmap.md`、`examples/05_multiobjective_catalog.py`；Modify root `README.md`／`README_zh.md`、`docs/textbook-improvements.md`、`expdoe-dk/pyproject.toml`、`requirements.txt`（僅若實際依賴決策需要）、`src/expdoe_dk/__init__.py`、`domain/space.py`、`knowledge/__init__.py`、version相關 tests。

**Interfaces:** `run_matrix(*, seeds, output_dir) -> list[dict]`，記錄source commit、dirty state、依賴、方法、reference、scale、seed、預算、runtime、實測次數、可行／重複率、HV與regret；輸出JSONL與摘要，生成大檔預設不追蹤。

- [ ] RED：先建立可解析oracle fixture：catalog `x=i/100, i=0..100`，目標max `[x,1-x]`，硬限制 `x<=0.8`。可行catalog真HV在reference `[0,0]`為0.476，assert誤差<=1e-12；先證明oracle再評估推薦器。
- [ ] 執行 `pytest -q tests/multiobjective/test_acceptance.py`，確認fixture／runner缺失；建立共享起始6點、8輪、q=2，種子 `(2,5,11,23,47)`，比較random catalog baseline、Plain GP、正確scoped knowledge GP，三者用同樣初始量測與reference，不借用oracle選初始點。
- [ ] 每seed的6個起始點只從已知硬參數限制允許的catalog以該seed無放回抽樣；從共享初始量測解析一次report geometry供三種方法與oracle參考集合共用。fixture的0.476是原始utility單位；benchmark若用正規化HV，oracle也必須經同一geometry，不能拿0.476直接相減。記錄兩種單位以供核對。
- [ ] GREEN：小型fast smoke驗證2／3／4／5／6目標路由、欄位順序、catalog、pending、noise與約束；slow完整矩陣驗證參數可行率1.0、重複率0.0、每批q、重現性及重啟一致。
- [ ] 此fixture用無噪音oracle，兩個GP方法都顯式noiseless；correct knowledge為只作用於第一目標的遞增monotone，Plain GP不注入任何pattern。品質門檻沿用原1.10容忍度：各GP方法跨固定seeds的平均HV regret <= `1.10 * max(random_mean_regret, 1e-12)`；Knowledge GP對Plain GP同樣非劣門檻。未達成就記錄失敗與原因，不調seed、刪壞案例或用功能測試取代品質驗收。此門檻為本fixture，不宣稱所有任務優於random。
- [ ] 執行完整 `pytest -q --run-slow`、兩個舊新範例與05範例；獨立review，重要問題必須有RED/GREEN回歸。記錄實際通過數、跳過理由、警告、耗時及尚未驗收的平台。
- [ ] 執行最低支援依賴環境與已驗證環境矩陣：檢查多目標所需log acquisition／ModelList posterior sampler能力。若舊下限欠缺，單目標import仍可用，多目標明確capability error；文件列多目標實測最低版本。只有經驗證且明確決策才提高整體依賴下限，不憑版本印象修改requirements。
- [ ] 功能與品質關卡通過後更新 `docs/roadmap.md` 的已完成／後續項目、中英文README的簡短能力與文件連結；結果細節放驗收README。修改 `docs/textbook-improvements.md` 的當前狀態，保留11月23日未實作的歷史範圍，不改舊計畫checkbox冒充當時完成。
- [ ] 版本提議0.6.0：經使用者確認後，同步pyproject、`__version__`、`ENGINE_VERSION`、版本斷言與metadata測試；新增精確舊版本0.5.0／0.5.1讀取測試，allow-list不可改成接受所有0.x。新的2.2只允許列明的writer版本；舊2.0／2.1仍依其格式讀取。歷史fixture保持舊版本以真正驗證migration，不全部搜尋取代。
- [ ] 再跑全套、建wheel／sdist、乾淨離線環境安裝wheel後執行單目標與多目標範例，驗證import／metadata／engine版本一致與舊checkpoint可讀。沿用 `tests/test_release_artifacts.py` 的legal／dependency檢查。README不得寫「支援所有版本」或「量測保證安全」。
- [ ] Review 後commit：`feat: release validated multiobjective workflows`；完成本機整合選項交付，push／tag／PyPI另待使用者授權。

## 完成定義與非目標

全部10 Tasks與發布關卡通過才可宣稱多目標功能完成；計畫完成不授權立即把版本改成0.6.0。此次不包含跨輸出相關GP、部分目標缺失訓練、RF／Ridge推薦、非線性跨輸出安全機率、LIMS連線、Colab／教材實際資料驗收、任意多目標數量的效能保證或全域最佳解保證。一般 `evaluate_models` 的多目標CV另案處理，這次新增的驗收是具名多目標循序實驗，不能偷取第一欄當多目標評估。

## 自我檢查與待確認事項

- [x] 以main實際檔案核對責任與API；未覆寫catalog模組或把registry存在視為runtime已支援。
- [x] 原Task2／3／6／7／8及驗證指標均有對應Task；單目標回歸與dependency-floor獨立列出。
- [x] 五類Review Focus均有具名測試；各Task有RED／GREEN／review／commit步驟。
- [x] 使用者確認本計畫，包括inferred-noise預設、偏好覆寫路由、reference固定策略、首版完整目標回填限制與版本0.6.0。
- [x] 使用者指定subagent-driven-development；已建立獨立worktree。

官方API核對於2026-11-24：BoTorch [模型介面](https://botorch.org/docs/models) 與 [acquisition API](https://botorch.readthedocs.io/en/stable/acquisition.html)。確認多輸出模型與MC objective／constraint／pending介面可供整合；具體版本相容性仍由Task10實測。規格中其餘預設值、限制與驗收門檻是本repo的設計提案，不是BoTorch的保證。
