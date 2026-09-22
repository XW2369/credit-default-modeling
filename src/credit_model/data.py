"""数据加载层：只负责搬运，不做任何 fit。"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

TARGET = "y"
LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(
    r"C:\Users\25466\Desktop\培训专业\代码仓库\01-信贷违约分析\data\processed"
)


def load_train_test(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """加载已经切分、已经 WOE 编码的建模宽表。

    Returns
    -------
    (X_train, y_train, X_test, y_test)
    """
    p = Path(data_dir)
    train = pd.read_csv(p / "features.csv")
    test = pd.read_csv(p / "features_test.csv")
    bad_train = int(train.isna().any(axis=1).sum())
    bad_test = int(test.isna().any(axis=1).sum())
    LOGGER.warning("drop all-NaN rows: train=%s test=%s", bad_train, bad_test)
    train = train.dropna()
    test = test.dropna()
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]
    LOGGER.info("train=%s test=%s", train.shape, test.shape)
    return X_train, y_train, X_test, y_test




