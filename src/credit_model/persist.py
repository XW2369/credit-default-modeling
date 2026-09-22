"""用最优参数训练并持久化模型，re-load 后在测试集上出分。"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from credit_model.data import load_train_test

LOGGER = logging.getLogger(__name__)
SEED = 42
MODELS_DIR = Path("models")
BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.013136241620785074,
    "subsample": 0.9709314921616581,
    "colsample_bytree": 0.6226141121608478,
    "min_child_weight": 6,
    "reg_lambda": 0.1117062584526268,
}


def train_and_save() -> Path:
    Xtr, ytr, _, _ = load_train_test()
    model = XGBClassifier(**BEST_PARAMS, n_jobs=-1, eval_metric="auc", random_state=SEED)
    model.fit(Xtr, ytr)
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / "best_model.joblib"
    joblib.dump(model, path)
    LOGGER.info("saved %s", path)
    return path


if __name__ == "__main__":
    path = train_and_save()
    model = joblib.load(path)
    _, _, Xte, yte = load_train_test()
    score = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print("reloaded model test AUC:", round(float(score), 4))
