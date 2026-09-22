"""端到端与单元测试：数据对齐 / 统计函数 / 评估指标。"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from credit_model.data import load_train_test
from credit_model.models import evaluate
from credit_model.stats_tests import ab_sample_size, bootstrap_auc_ci, mcnemar_test

TARGET = "y"


def test_loader_shapes_and_no_target():
    Xtr, ytr, Xte, yte = load_train_test()
    assert len(Xtr) == len(ytr)
    assert len(Xte) == len(yte)
    assert TARGET not in Xtr.columns
    assert TARGET not in Xte.columns


def test_loader_has_no_all_nan_rows():
    """回归测试：2026-09-22 的 WOE 索引错位 bug 曾制造 30%/70% 全 NaN 行。"""
    Xtr, _, Xte, _ = load_train_test()
    assert int(Xtr.isna().all(axis=1).sum()) == 0
    assert int(Xte.isna().all(axis=1).sum()) == 0


def test_pay0_single_feature_is_predictive():
    """回归测试：错位 bug 当时让 IV=0.87 的特征单特征 AUC 只有 0.50。"""
    Xtr, ytr, _, _ = load_train_test()
    auc = roc_auc_score(ytr.to_numpy(), Xtr["pay_0_woe"].to_numpy())
    assert max(auc, 1 - auc) > 0.60



def test_bootstrap_ci_contains_point_estimate():
    Xtr, ytr, Xte, yte = load_train_test()
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(Xtr, ytr)
    score = model.predict_proba(Xte)[:, 1]
    auc, lo, hi = bootstrap_auc_ci(yte, score, n_boot=200)
    assert lo <= auc <= hi
    assert 0.0 < lo < hi < 1.0


def test_mcnemar_self_consistency():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    pred = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    b, c, p = mcnemar_test(y, pred, pred)
    assert b == 0 and c == 0
    assert p > 0.9


def test_ab_sample_size_monotonic():
    n_small = ab_sample_size(0.2212, 0.05)
    n_large = ab_sample_size(0.2212, 0.30)
    assert n_large < n_small
    assert n_small > 0


def test_evaluate_returns_five_metrics():
    Xtr, ytr, Xte, yte = load_train_test()
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(Xtr, ytr)
    m = evaluate("logreg", model, Xtr, ytr, Xte, yte)
    assert set(m) == {"roc_auc", "pr_auc", "f1", "ks", "brier"}
    assert 0.0 < float(m["roc_auc"]) < 1.0
