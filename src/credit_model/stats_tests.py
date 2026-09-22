"""统计检验：AUC 置信区间 / McNemar / A-B 样本量。护城河模块。"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

SEED = 42


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """重抽样 n_boot 次，返回 (auc, ci_low, ci_high)。"""
    # TODO 1: rng = np.random.default_rng(seed)
    # TODO 2: 循环 n_boot 次——有放回抽索引，每次算 AUC，存进列表
    # TODO 3: np.percentile 取 2.5 和 97.5 分位，三元组返回
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], y_score[idx])
    aucs = aucs[~np.isnan(aucs)]
    auc = float(roc_auc_score(y_true, y_score))
    lo = float(np.percentile(aucs, 2.5))
    hi = float(np.percentile(aucs, 97.5))
    return auc, lo, hi



def mcnemar_test(y_true, pred_a, pred_b) -> tuple[int, int, float]:
    """两个模型在同一测试集上的配对检验，返回 (b, c, p_value)。
    b = A对B错 的样本数，c = A错B对 的样本数。"""
    # TODO 4: 两个布尔数组 a_ok = (pred_a == y_true)，b_ok 同理
    # TODO 5: b = int(np.sum(a_ok & ~b_ok))，c = int(np.sum(~a_ok & b_ok))
    # TODO 6: stats.binomtest(min(b, c), b + c, 0.5).pvalue——正是你在SPSS里做过的精确二项检验
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p



def ab_sample_size(p1: float, lift: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """两比例检验的每组样本量。p1=基线违约率，lift=相对提升。"""
    # TODO 7: p2 = p1 * (1 + lift)
    # TODO 8: z_a = stats.norm.ppf(1 - alpha/2)，z_b = stats.norm.ppf(power)
    # TODO 9: n = (z_a*sqrt(2*p̄*(1-p̄)) + z_b*sqrt(p1*(1-p1)+p2*(1-p2)))**2 / (p1-p2)**2，向上取整
    p2 = p1 * (1 + lift)
    p_bar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = ((z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
         / (p1 - p2) ** 2)
    return int(np.ceil(n))

