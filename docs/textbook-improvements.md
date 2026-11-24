# 教科書改進：範圍與驗收

本文件保留 [2026-11-23 實作計畫](superpowers/plans/2026-11-23-textbook-improvements.md) 的單目標改進紀錄。後續 0.6.0 已加入具名多目標推薦與明確選用的 monotone v2，詳見[多目標指南](multiobjective.md)。[對數反應值指南](log-objectives.md) 說明既有反應尺度介面，並未新增轉換 API。

## 本次實作

- **[候選清單](candidate-catalog.md)（本次 Task 1）：** 單目標 `recommend` 已加入 `candidates`／`candidate_ids`，沿用現有 GP 與採集函數。專項測試確認回傳清單原值與候選 ID、重排後 ID 對齊、不接受反應欄、排除已觀測或待回填條件，且符合適用限制。另有本機合成的 294 列清單測試，不等同教材原始資料驗收。
- **[量測流程範例](measurement-workflow.md)（本次 Task 2）：** 以現有 Campaign 介面示範回填、部分回填、取消和狀態還原。候選 ID 識別配方，trial ID 識別單次量測；重測須使用新的 trial ID，不以預測值代替實測值。這是教學範例，不是新的套件 API；已測試恢復狀態後重新建模並產生一致推薦。
- **限制與知識回歸（本次 Task 3）：** 連續搜尋已加入仿射線性等式的限制轉換、可行起點取樣與受限最佳化，支援宣告式及舊版雙側 `LinearConstraint`。測試直接檢查回傳候選的可行性與唯一性，涵蓋混合參數、無解子空間及完全固定的連續座標；原有知識支援、衝突與不支援設定則沿用既有回歸測試。

## 等式搜尋的支援邊界

限制先在物理單位定義，再轉成模型座標；離散或整數因素先代入該組合的物理值。以 BoTorch 的可行區域取樣與受限最佳化產生候選，最後仍回到物理單位重新檢查，不以模型座標的可行性取代原始限制。[BoTorch 限制介面](https://botorch.readthedocs.io/en/stable/optim.html)

- 支援 `a + b == total` 等仿射線性等式，也支援同時存在的可轉換線性不等式。只有不等式而沒有線性等式的搜尋維持既有流程。
- 會檢查邊界造成的固定方向，保留其餘可變方向；相依等式則依原始物理單位的容差檢查一致性。需要使用容差時，搜尋的是容差內的一個可行等式切面，不是整個容差帶，也不會放寬硬性不等式。
- 等式搜尋需要 BoTorch 提供支援 `seed` 與 `n_thinning` 的 polytope sampler。缺少這個介面的舊版本會明確報錯並提示改用候選清單，不會因新版錯誤類別不存在而阻止套件匯入。舊介面測試採模擬方式，尚未在完整 BoTorch 0.11 環境驗收。
- 等式若涉及採用 log 編碼的連續參數，模型座標中就不再是線性等式；目前明確拒絕這種連續搜尋，請改提供已符合條件的候選清單。不會忽略等式。
- 混合搜尋的有限組合數上限為 100,000；超過上限的等式搜尋也須改提供候選清單。
- 其他 expression／知識限制仍會在物理單位檢查，但此補強不是任意非線性等式求解器。有限的取樣與多起點搜尋不保證找到所有可行配方或全域最佳解。
- 無法組成完整且不重複的批次就報錯；數值求解失敗也不以隨機配方替代。這是當時單目標補強的紀錄；現行多目標支援邊界見[多目標指南](multiobjective.md)。

## 0.6.0 多目標實作與驗收

原規劃的具名多目標推薦、qLogEHVI／qLogNEHVI／qLogNParEGO 式路由、實測 Pareto／固定尺度超體積報告、schema 2.2 checkpoint 已實作。0.6.0 原始碼 `412fd2d` 在乾淨工作樹通過固定五種子的 current／lower 品質協定與完整 slow suites：current 2592 passed；lower 2591 passed、1 項 Torch uint64 能力限制跳過。發布前 `226379f` 的驗收與 `93a920a` 的版本相容性失敗另行保留，不混作最終通過證據。後續只有測試與文件修正；checkpoint 測試檔另於兩個環境各通過 72 項，套件程式碼及已驗證 wheel 仍對應 `412fd2d`。此案例的 monotone v2 與 Plain 成績相同，不能宣稱優於 Plain。歷史 v1 與數值失敗證據仍保留，詳見[逐方法結果](../experiments/multiobjective_validation/README.md)。

`with_monotone` 仍為 v1，不自動升級；v2 以柔性方向證據調整已擬合 GP，不捏造量測值。v2 計算可能很慢，不保證互動速度；已觀察的 source-adapter 耗時並非完整推薦流程基準。最低 Python 版本已依 Ax 相依條件修正為 3.11。教材原始資料、Colab、其他作業系統與 GPU 仍未驗收。

以下保留原計畫對應，描述當時尚待執行的項目，不代表目前功能仍未實作：

- 原 [optimization pipeline Task 3](superpowers/plans/2026-08-05-optimization-pipeline-implementation-plan.md#task-3-implement-acquisition-policy-routing-and-constructors) 規劃 2 至 4 個目標使用 qLogEHVI；有已知噪音或待回填實驗時改走 qLogNEHVI。5 個以上目標規劃以 qLogNParEGO 方式做 Chebyshev 純量化，並記錄權重。應測試各條件的路由與建構結果。
- Pareto 候選列、超體積歷史與停止訊號的結果回報屬於原 [optimization pipeline Task 8](superpowers/plans/2026-08-05-optimization-pipeline-implementation-plan.md#task-8-add-evidence-backed-stopping-suggestions-and-result-reporting)。同一份計畫的 Task 7 是 schema-v2 checkpoint 與 v0.4 遷移，不是 Pareto 結果回報。
- 原 [validation matrix Task 2](superpowers/plans/2026-08-05-validation-and-benchmark-matrix-implementation-plan.md#task-2-implement-predictive-calibration-and-optimization-metrics) 要求以明確記錄的參考點計算超體積遺憾（hypervolume regret）。[Task 5](superpowers/plans/2026-08-05-validation-and-benchmark-matrix-implementation-plan.md#task-5-define-four-synthetic-fixture-families-and-acceptance-thresholds) 規劃用具解析可行 Pareto 前緣的合成 oracle 驗證：確認參考解可行且超體積為正，以一致的效用方向、尺度及參考點檢查非支配集合和超體積遺憾，並依該任務設定的門檻比較。[Task 7](superpowers/plans/2026-08-05-validation-and-benchmark-matrix-implementation-plan.md#task-7-enforce-acceptance-gates-and-document-results) 另規劃在循序基準測試中驗收可行率 100%、重複率 0%。這些都是待執行的測試與門檻，不是已取得的成績。

## 三種驗收各自回答什麼

| 驗收 | 要回答的問題 |
| --- | --- |
| 本機功能驗收 | 在內層套件目錄用 `PYTHONPATH=src` 指向本次 checkout，跑回歸測試及兩個範例。檢查清單成員與 ID、限制、重複排除及回填／還原是否正確；結果列於下方。 |
| 模型品質評估 | 在相同資料切分、種子與實驗預算下比較模型；分開報告預測誤差（RMSE、MAE、$R^2$）、校準與循序最佳化成效（例如最佳值進度、簡單遺憾）。標明評估尺度，並區分潛在反應與新量測的區間。功能測試通過不代表預測更準或候選排名一定相同。 |
| 教科書／Colab 驗收 | 教科書第 9、10、12 章均未執行，Colab 也未執行。本機測試或範例即使通過，也不能代替教科書流程在 Colab 的驗收。 |

## 本機驗證環境

2026-11-23 使用 `expdoe-dk-codex` conda 的 Python 與既有相依套件，再於 worktree 的 `.venv` 安裝本次套件。原因是原 conda 環境的 editable install 與 distribution metadata 仍指向舊版；使用 `PYTHONPATH=src` 雖能選取新原始碼，卻不足以保證安裝版本資訊一致。原 conda 安裝未被覆寫。

版本為 Python 3.12.13、NumPy 2.5.0、pandas 3.0.3、SciPy 1.18.0、PyTorch 2.12.1、BoTorch 0.18.1、GPyTorch 1.15.2、pytest 9.1.1。這不是所有相依版本下限的相容性證明。

在內層套件目錄執行完整測試：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../.venv/bin/python -m pytest -q --run-slow
```

部分既有稽核測試會呼叫 Git。本機系統 Git 當時被 Xcode 授權提示阻擋，因此驗證時將可用的獨立 Git 與 conda 環境的執行檔目錄放在 `PATH` 前方；未修改系統授權或略過這些測試。

## 2026-11-23 本機結果（歷史）

- 獨立 review 的八項問題修正後，完整 `pytest -q --run-slow`：**1,328 passed**，0 failed、0 skipped，耗時 204.60 秒。
- 候選清單、量測流程兩個範例都執行成功；對數反應值指南的 Python 範例也已執行。
- 共 117 個警告，包含 PyTorch 棄用提示、GP 的 jitter／微小負變異數修正，以及既有極端範圍測試的 overflow 提示；不宣稱沒有數值警告。
- 本機單一環境的功能通過，不等於教材實際資料、Colab、相依版本下限或模型預測精度已驗收。
