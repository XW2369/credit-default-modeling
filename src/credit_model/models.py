"""模型定义与评估。"""
from __future__ import annotations

import logging

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)
SEED = 42


def get_models() -> dict[str, object]:
    """返回待对比的模型字典。"""
    # TODO A: 三个模型 —— logreg / logreg_l1 / xgb
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "logreg_l1": LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, class_weight="balanced", random_state=SEED),
        "xgb": XGBClassifier(n_jobs=-1, eval_metric="auc", random_state=SEED),
    }



def evaluate(name: str, model, X_train, y_train, X_test, y_test) -> dict[str, float]:
    """fit 一个模型，返回五个指标。"""
    # TODO B: model.fit(X_train, y_train)
    # TODO C: y_score = model.predict_proba(X_test)[:, 1]
    # TODO D: 五个指标 —— roc_auc / pr_auc / f1(阈值0.5) / ks / brier
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "f1": float(f1_score(y_test, (y_score >= 0.5).astype(int))),
        "ks": float(np.max(tpr - fpr)),
        "brier": float(brier_score_loss(y_test, y_score)),
    }
    LOGGER.info("%s -> %s", name, metrics)
    return metrics

