# 02-信贷违约预测 · 四模型对比 + 统计显著性检验

7 天 AI 工程师转型冲刺营 **Day 3**。

目标：不做「跑个模型看谁 AUC 高」，而是**用统计口径回答两个问题**——
哪个模型真的更好？这个结论能不能支撑上线？

---

## 环境

| 组件 | 版本 |
|---|---|
| Python | 3.11（`uv` 管理的项目级 `.venv`） |
| scikit-learn / xgboost | 1.9.1 / ≥3.2 |
| optuna | ≥5.0（TPE 采样，20 trials） |
| scipy / shap | 1.17.1 / 0.51.0 |
| 工具链 | uv / ruff / pytest |

## 快速开始

```
uv venv
uv pip install -e .
uv run pytest -q                  # 7 passed
uv run ruff check .               # 0 error
uv run python -m credit_model.tune      # Optuna 调参，20 条试验落 reports/experiments.csv
uv run python -m credit_model.persist   # 训练并持久化 models/best_model.joblib
uv run python -m credit_model.explain   # SHAP 解释
```

## 目录结构

```
src/credit_model/
├── data.py        切分 + WOE 变换（**先切分后编码**，防目标泄漏）
├── models.py      4 个模型 + AUC / KS 评估
├── stats_tests.py bootstrap AUC 置信区间 / McNemar 检验 / A-B 样本量
├── persist.py     训练 + joblib 持久化 + re-load 自检
├── explain.py     SHAP 解释
└── tune.py        Optuna TPE 超参搜索
tests/test_pipeline.py   7 个用例，含 2 条事故回归测试
```

## 实测结果（2026-09-22）

| 指标 | 数值 |
|---|---|
| 对比模型数 | **4**（LogisticRegression / L1-Logistic / XGBoost / XGBoost-tuned） |
| 基线 LR AUC + 95% CI | **0.7587**，[0.7458, 0.7709] |
| 基线 XGB AUC + 95% CI | **0.7540**，[0.7414, 0.7668] |
| 调参后 XGBoost | CV best **0.7756**；joblib re-load 后测试集 **0.7701** |
| **McNemar 检验** | **b=474, c=844, p = 1.4 × 10⁻²⁴**（固定阈值层面极显著） |
| 上线 A/B 所需样本量 | **5,719 / 组**（α=0.05, power=0.8, 检出 10% 相对提升） |
| pytest / ruff | **7 passed** / **0 error** |

### ⚠️ 一个必须同时说出口的限制

**AUC 置信区间大面积重叠，但 McNemar 的 p 值极显著——两者不矛盾。**

- 两个 95% CI 各自覆盖真值是边际命题；「两者之差是否为 0」要看**差值的分布**
- 本 p 值是**固定阈值（0.5）层面**的结论，不是排序能力（AUC）层面的结论
- 上线 A/B 需要 **5719/组 ≈ 11438 条**，而全量测试集只有 **9000 条** →
  **样本量不足以支撑上线结论**

> 正确判法是成对 bootstrap 构造差值 CI，或 DeLong 检验。本项目两者**均未做**（见下）。

## 事故复盘：WOE 索引错位（本项目最有价值的一段）

第一天 baseline 跑出 **AUC 0.5039、相关系数 0.004**——代码没报错、8 个单元测试全绿、日志一片正常。

病灶在上游：`woe_transform` 索引对齐失败，导致 **29.7%（train）/ 70%（test）的行全 NaN**，
剩下行的特征与标签彻底错配 → 模型只学到边际概率。

三条关键认识：

1. **IV 没有虚高**——`fit_woe` 的 crosstab 两边都是 0…n−1，按位置算，IV 表（pay_0=0.8736）完全正确。坏的只有变换输出。
2. 解法是 `woe_transform` 改为 **numpy 按位置分箱 + 列表按位置写回，全程不碰索引**。
3. **事故必须固化成测试**，否则会重演。补的两条回归测试：
   - 全 NaN 行数必须为 0
   - 单特征 AUC 体检（`pay_0` IV 0.87，实测单特征 AUC 0.284；本项目 WOE 定义为 ln(好/坏)，与违约负相关，**反向判别力 0.716**）

> **模型能跑 ≠ 数据正确。能跑只是最低标准。**

## ⚠️ 未做 / 待补

- **OOT（时间外）验证**：从 Day2 拖到 Day3，至今未做，`pay_0` 的泄漏嫌疑仍未实证
- **DeLong 检验**：与 McNemar 的适用场景区别仍未梳理
- **PSI 特征稳定性监控**：未测

## 踩过的坑

-  CV 里的 `l1_ratio=1` 弃用警告要清干净，否则日志里混着未来版本的报错
- `.gitignore` 必须先于数据下载，否则 23MiB 数据集会进 GitHub 历史
