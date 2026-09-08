[English README](./README.md)

# expdoe-dk

為實驗室的物理單位與混合變數流程提供實驗設計（DoE）與貝葉斯優化，並支援明確限制式及有版本的領域知識。

## v0.5 功能

- 連續、整數、離散、類別與序位參數，以及物理／模型座標轉換。
- 明確的最大化、最小化與目標區間 objective。
- 宣告式線性、expression、類別組合與 outcome constraints。
- 具能力檢查與確定性 diagnostics 的 LHS、Sobol、Halton、random 與
  D-optimal 初始設計。
- 不可變的 observation/pending batches，以及具 schema 版本的 space payload。
- 內建 knowledge patterns 與明確 distribution-name allow-list 的知識 registry。
- 保留 v0.4 的 `Campaign`、`Knowledge`、checkpoint、report 與頂層 import 相容性。

## 安裝

```bash
pip install -e ./expdoe-dk
```

支援 Python 3.10–3.12。執行時相依套件宣告於
[`expdoe-dk/pyproject.toml`](./expdoe-dk/pyproject.toml)，其中包含驗證知識定義所需的直接相依 `jsonschema>=4.18`。

## 混合空間、objectives 與 constraints

以下範例可在已安裝套件的環境直接執行：

```python
import expdoe_dk as ed
from expdoe_dk.domain import LinearConstraint

space = ed.Space(
    params=[
        ed.Parameter("temperature", bounds=(300.0, 400.0), unit="K"),
        ed.Parameter("cycles", kind="integer", bounds=(1, 9), step=2),
        ed.Parameter("dose", kind="discrete", values=[0.1, 0.3, 0.8]),
        ed.Parameter("solvent", kind="categorical", values=["water", "ethanol"]),
        ed.Parameter("grade", kind="ordinal", values=["low", "medium", "high"]),
    ],
    constraints=[
        LinearConstraint(
            "temperature_limit",
            coefficients={"temperature": 1.0},
            operator="<=",
            bound=390.0,
        ),
        ed.CategoricalCombinationConstraint(
            "avoid_low_ethanol",
            forbidden=[{"solvent": "ethanol", "grade": "low"}],
        ),
    ],
    objectives=[
        ed.Objective("yield", "maximize", unit="%", priority=0),
        ed.Objective("waste", "minimize", unit="g", priority=1),
    ],
)

batch = ed.suggest_design(
    space, n=8, method="auto", seed=7, return_diagnostics=True
)
print(batch.frame)
print(batch.diagnostics.effective_method)
```

`expdoe_dk.LinearConstraint` 仍是 v0.4 的雙側相容 adapter；新程式若要使用宣告式單側版本，請如上例 import `expdoe_dk.domain.LinearConstraint`。

## Knowledge registry 與明確 provider 載入

內建與外部 provider 共用同一個精確 `(pattern, version)` registry。Provider discovery 絕不會隱式執行；import、`Knowledge`、validation 與 compilation 都不會執行 provider code。

```python
from expdoe_dk import Knowledge, PatternRegistry, load_pattern_providers
from expdoe_dk.knowledge.patterns import builtin_pattern_definitions

registry = PatternRegistry()
for definition in builtin_pattern_definitions():
    registry.register(definition)

# 沒有核准外部 distribution 時維持空集合；審查並安裝後可改為
# {"my-lab-patterns"}。
approved_distributions: set[str] = set()
report = load_pattern_providers(approved_distributions, registry)
print(report.to_dict())

knowledge = Knowledge(registry).with_monotone(
    "temperature", effect="increases_objective"
)
print([(spec.pattern, spec.version) for spec in knowledge.specs])
```

Allow-list 以確定性的 PEP 503 distribution 名稱比對，因此大小寫與 `.`, `_`,
`-` 的差異視為相同。Entry-point 名稱只記錄 provenance，不能授權載入。若指定的 distribution 未安裝或 provider 格式錯誤，系統會在 registry 變動前拋出具型別的 `EngineError`。

## 從 v0.4 遷移

舊版建構方式仍可使用：

```python
import expdoe_dk as ed

legacy_space = ed.Space(
    [ed.Parameter("x", bounds=(0.0, 1.0))],
    objectives="yield",
    maximize=True,
)
legacy_constraint = ed.LinearConstraint(coeffs={"x": 1.0}, upper=0.8)
```

v0.5 建議改用明確 objectives、宣告式 constraints 與有版本的 knowledge specs：

```python
import expdoe_dk as ed
from expdoe_dk.domain import LinearConstraint

space = ed.Space(
    [ed.Parameter("x", bounds=(0.0, 1.0))],
    constraints=[LinearConstraint("x_limit", {"x": 1.0}, "<=", 0.8)],
    objectives=[ed.Objective("yield", "maximize")],
)
knowledge = ed.Knowledge().with_saturation(
    "x", direction="increasing", half_response=0.4
)
payload = space.to_dict()
restored = ed.Space.from_dict(payload)
assert restored.to_dict() == payload
```

Observer 與實驗室 device integration 是未來的外部工作，本 repository 尚未實作。

## 開發

```bash
cd expdoe-dk
pip install -e ".[dev]"
pytest -q
```

慢速整合測試只有在明確指定 `--run-slow` 時才執行。

## 授權

Apache License, Version 2.0。詳見 [`LICENSE`](./LICENSE) 與
[`NOTICE`](./NOTICE)。
