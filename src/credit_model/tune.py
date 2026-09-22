"""Optuna 调参：XGBoost 超参搜索 + 实验记录。"""
from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)
SEED = 42


def objective(trial, X, y) -> float:
    """Optuna 目标函数：返回 5 折分层 CV 的平均 AUC（越大越好）。"""
    # TODO 参数搜索空间：n_estimators / max_depth / learning_rate / subsample / colsample_bytree / min_child_weight / reg_lambda
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }
    model = XGBClassifier(**params, n_jobs=-1, eval_metric="auc", random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return float(scores.mean())



def run_study(X, y, n_trials: int = 20) -> tuple[optuna.Study, pd.DataFrame]:
    """跑 n_trials 次试验，返回 study 与试验记录表。"""
    # TODO study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    # TODO study.optimize(lambda t: objective(t, X, y), n_trials=n_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optua.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials)
    tbl = study.trials_dataframe()
    LOGGER.info("best trial #%s auc=%.4f params=%s", study.best_trial.number, study.best_value, study.best_params)
    return study, tbl
if __name__ == "__main__":
    from credit_model.data import load_train_test

    Xtr, ytr, _, _ = load_train_test()
    study, tbl = run_study(Xtr, ytr, n_trials=20)
    tbl.to_csv("reports/experiments.csv", index=False)
    print("best auc:", round(study.best_value, 4))
    print("best params:", study.best_params)

