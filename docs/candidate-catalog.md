# 從既有配方清單推薦

若實驗只能從一份已驗證的配方清單中選取，使用 `recommend` 的 `candidates` 和 `candidate_ids`。這與在參數範圍內搜尋新配方不同：不組合各欄的可取值、不把 ID 當作數字特徵，也不把搜尋結果映射成最接近的配方。

完整可執行範例：[03_candidate_catalog.py](../examples/03_candidate_catalog.py)。在 repo 根目錄執行：

```bash
PYTHONPATH=expdoe-dk/src python examples/03_candidate_catalog.py
```

## 輸入與輸出

```python
result = ed.recommend(
    space, observations,
    knowledge=knowledge,       # 可省略
    pending=pending,           # 可省略；PendingBatch 的 ID 是量測 ID
    candidates=catalog[space.param_names],
    candidate_ids=tuple(catalog["candidate_id"]),
    q=2, seed=17,
)
selected_ids = result.candidate_ids
selected_formulas = result.frame
predictions = result.predictions
```

- `candidates` 必須是 DataFrame，欄位恰好是 `Space` 的全部參數。ID、備註與反應值另外存放；多餘或缺少欄位會報錯。
- `candidate_ids` 是與資料列**位置對齊**的唯一非空字串序列。DataFrame index 不拿來推定 ID；pandas 欄位或 index 請明確轉成 `tuple`。
- `result.frame` 按 `space.param_names` 排列欄位，保留被選配方的原始用量與類別值；與 `result.candidate_ids` 一一對應。`to_dict()` 也包含候選 ID。
- 不提供候選清單時，既有搜尋方式不變，`candidate_ids` 為空 tuple。
- 須至少有三筆成功量測。最大化、最小化或目標區間沿用 `Objective`，目前只支援單一目標。

## 排除、重複與限制

排除依**實驗條件**比對，不是依 ID：即使重新命名 ID，已觀測或待回填的配方仍不會被推薦。`failed` 與 `warning` 不參與模型訓練，但其條件也會被排除。若要刻意重測或重試失敗配方，應由實驗者建立新的量測 ID，參考[量測流程範例](measurement-workflow.md)；推薦器不會自動決定重試。

同一份清單中，相同條件使用不同 ID 也會被拒絕；由使用者先確認並合併重複配方。不用候選 ID 代替量測 ID，因此同一配方的不同量測仍可分別保留。

實體配方不同、但差異小到目前模型座標無法區分時，會明確回報 `Model-coordinate collision`。這是編碼精度限制，不代表配方重複，也不應合併成同一配方；此檢查也涵蓋清單與已測／待測條件之間的碰撞。

所有配方沿用現有參數限制、知識提供的硬限制及結果安全篩選。等式限制也直接以原始配方檢查。結果安全機率來自模型，不是實驗安全保證。清單篩選後不足 `q` 筆時，回報 `NO_FEASIBLE_CANDIDATE` 與 requested/available 數量，不回傳半個批次、不補隨機配方。

知識和 GP 使用同一條既有流程；不額外建立第二個模型。批次選取會把待測與本批已選點加入採集函數的條件，並在最後重新檢查限制和重複。

## 重現性與驗收

清單先按穩定 ID 排序再評分，相同環境、量測、種子與 ID 對應下，改變資料列順序不應改變結果。同分時採固定 ID 順序；不同依賴版本或硬體造成的近似同分排序差異，不能僅靠固定 seed 消除。

功能驗收檢查清單成員、原始值、ID 對齊、限制、排除和批次完整性。跨平台比較另記錄 Python／NumPy／PyTorch／BoTorch 版本、種子與數值容差；模型預測誤差及不確定性校準另行評估。此範例使用小型示範資料，不是教材的 294 配方資料，也不代表已完成 Colab 或教材第 9、10、12 章驗收。
