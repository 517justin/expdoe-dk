# 在對數尺度上建模反應值

## 尺度與資料前處理

若反應值為正，且實驗上較關心倍數變化而非絕對差值，可在套件外先定義 $z = \ln(y/y_0)$。$y_0 > 0$ 是固定參考值，須與 $y$ 使用同一單位；比值和 $z$ 都沒有量綱。例如濃度用 mg/L 記錄時，可選 $y_0 = 1\ \mathrm{mg/L}$，而 $z = 0$ 表示濃度等於參考值，不是零濃度。換單位時須一起換算 $y$ 和 $y_0$。保留原始量測，記錄參考值、單位與對數底數，並在同一實驗流程中沿用。

只對有限且嚴格大於零的實測值取對數。先判明零或負值的來源：若是記錄錯誤，回查原始資料；若是真零，考慮原尺度模型，或分開建模「是否為正」與「正值大小」；若是未檢出或低於報告限值，保留設限標記和限值，不要當成精確的零或精確的限值。若有正的左設限上界 $L$，在 log 尺度它仍是左設限，界線為 $\ln(L/y_0)$。現行 GP 不會把這類標記當作設限觀測處理；需要時應另用支援設限的模型。若排除這些資料，應揭露規則及可能造成的偏差。不要加任意的 $\epsilon$ 來讓零或負值可取對數。

## 現行 API 回傳什麼

[`fit_surrogate` 與 `FittedSurrogate`](../expdoe-dk/src/expdoe_dk/bo/surrogates.py) 以提供的 `ObservationBatch.Y` 擬合 GP；內部做中心化、縮放及符號調整，`predict` 和 `acquisition_model` 再還原至 **輸入反應欄的尺度**。若自行把 $z$ 當成反應，應讓欄名、目標和相關設定都指向這個反應；API 回傳的是 $z$ 的潛在後驗平均與變異數（採集模型也在此尺度取樣），不會替使用者取對數或轉回 $y$。參數的 log 編碼只作用於輸入 $X$，與反應值轉換無關。

`Yvar` 是訓練觀測的已知量測噪音變異數，不是 `predict` 回傳的潛在變異數。若 `Y` 是 $z$，`Yvar` 也須在 $z$ 尺度，優先用轉換後的重複量測估計，並對應到實際輸入的量測或平均值。若原尺度量測誤差相對小，可用一階近似 $\operatorname{Var}(Z_{\rm obs}) \approx \operatorname{Var}(Y_{\rm obs})/y^2$（$y$ 取代表性的正反應值）；不能直接沿用原單位的變異數，也不能對變異數取 log。若提供 `Yvar`，每筆成功觀測都須有有限且非負的值；不提供時由模型估計噪音。

以下是可獨立執行的小型示範，原始濃度仍保存在 `measured_mg_L`，不覆寫成預測值：

```python
import numpy as np
import pandas as pd
import expdoe_dk as ed

measured_mg_L = np.array([8.0, 6.0, 4.5, 3.0])
reference_mg_L = 1.0
if not np.all(np.isfinite(measured_mg_L) & (measured_mg_L > 0)):
    raise ValueError("Log responses require finite positive measurements")
space = ed.Space(
    [ed.Parameter("time", bounds=(10.0, 60.0), unit="min")],
    objectives=ed.Objective("log_concentration", "minimize"),
)
observations = ed.ObservationBatch(
    pd.DataFrame({"time": [10.0, 20.0, 30.0, 40.0]}),
    pd.DataFrame({"log_concentration": np.log(measured_mg_L / reference_mg_L)}),
)
model = ed.fit_surrogate(space, observations, seed=17)
prediction = model.predict(pd.DataFrame({"time": [45.0, 50.0]}))
mu, variance = prediction.mean[:, 0], prediction.variance[:, 0]
median_mg_L = reference_mg_L * np.exp(mu)
mean_mg_L = reference_mg_L * np.exp(mu + variance / 2)
print(pd.DataFrame({"latent_median_mg_L": median_mg_L,
                    "latent_mean_mg_L": mean_mg_L}))
```

## 回到原單位：平均數不等於中位數

在固定實驗條件下，若潛在對數反應的後驗為 $Z \mid \mathcal D \sim \mathcal N(\mu,v)$，則潛在原尺度反應 $Y = y_0 e^Z$ 服從對數常態：

| 原尺度摘要 | 公式 |
| --- | --- |
| 中位數 | $y_0 e^\mu$ |
| 算術平均數 | $y_0 e^{\mu + v/2}$ |
| 變異數 | $y_0^2 e^{2\mu+v}(e^v - 1)$ |

`exp(mu)` 乘上參考值只得到中位數，不是一般情況下的平均數。等尾區間可把 $Z$ 的上下分位數分別做 $y_0 e^z$ 反轉換。

上表用 `predict` 的 $v$ 時，描述的是潛在反應，不包含下一次量測的噪音。若新量測在 log 尺度另有獨立、零均值的高斯噪音，且在該條件下變異數為 $\tau_*^2$，才可用 $v + \tau_*^2$ 代入上表，描述新量測的預測分布。不可把訓練資料的 `Yvar` 直接當成新點的 $\tau_*^2$；若噪音模型不同，須另外推導或用取樣計算。

## 最佳化方向、知識與限制

- **採集函數不是反應轉換。** [`recommendation.py`](../expdoe-dk/src/expdoe_dk/bo/recommendation.py) 用 qLogNEI（`qLogNoisyExpectedImprovement`）評估改善效用；名稱中的 Log 指採集函數的數值計算，不會把反應值取 log。改用 log 反應會改變模型與改善效用，不保證候選排序相同。
- **方向不反轉。** $z$ 隨 $y$ 單調遞增，`maximize` 仍對應最大化、`minimize` 仍對應最小化，不必因 GP 內部符號再反轉一次。但在 log 尺度最佳化不等於最佳化原尺度的後驗算術平均數。正的目標值或區間兩端也要轉為 log；例如 $y_0 = 1\ \mathrm{mg/L}$ 時，目標區間 2 至 8 mg/L 應填入 $(\ln 2, \ln 8)$。距離目標的懲罰此時按 log 差值計算，不再按絕對濃度差計算。
- **硬性 outcome constraint 同尺度。** 正的上限或下限 $b$ 應改為 $\ln(b/y_0)$，不等號方向不變。以 $y>0$ 的模型而言，非正下限恆成立，非正上限無法滿足，不能用 $\epsilon$ 轉成有限界線。現行推薦只支援硬性單邊不等式，可用上下兩條限制表示區間；安全篩選用的是潛在後驗機率，不保證下一次帶噪量測一定合格。只涉及 $X$ 的參數限制不因反應取 log 而改變。
- **知識也要重新檢查。** 反應導向的知識須對應轉換後的目標欄位。正的反應界線（如 `target_range` 的端點）同樣轉換；此知識在推薦中是軟性懲罰，不能代替硬限制。單調方向仍保留，但斜率、差值、曲率和加法效應一般不保留。若正值公式 $m(x)$ 描述的是要建模的潛在反應曲線，可重新表達為 $\ln(m(x)/y_0)$；若它描述帶噪 $Y$ 的算術平均，不能逕當成 $\operatorname{E}[Z]$。只沿用在新尺度仍成立且受支援的知識，不應直接複製原尺度係數或虛擬反應值。

以上是使用既有介面的資料準備與結果解讀方式，並未新增反應轉換 API。
