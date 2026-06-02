"""タスク定義 — このハーネスは **分類でも回帰でも使える汎用テンプレート**。

タスク非依存のループ (registry/記録/制度的記憶/診断) は一切変えず、
ここで定義する `Task` (metric/CV/model) を YAML の `task:` で選ぶだけで
二値分類 ↔ 回帰 (例: 天気予測=予報) を切り替えられる。

新しいタスク (分位点回帰・時系列予報・多クラス等) を足したいときは、
ここに `Task` を1つ追加して `TASKS` に登録するだけ。runner/registry は不変。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass
class Task:
    name: str
    metric_name: str
    metric: Callable                # (y_true, y_score) -> float (高いほど良いに正規化)
    higher_is_better: bool
    needs_stratify: bool            # CV で y による層化が必要か (分類=True)
    make_cv: Callable               # (n_splits, seed) -> splitter
    make_tree: Callable             # () -> 木モデル
    make_linear: Callable           # () -> 線形モデル
    predict: Callable               # (model, X) -> 連続スコア (分類=陽性確率 / 回帰=値)


def _neg_rmse(y, p):
    return -float(np.sqrt(mean_squared_error(y, p)))


TASKS: dict[str, Task] = {
    "classification": Task(
        name="classification",
        metric_name="AUC",
        metric=lambda y, p: float(roc_auc_score(y, p)),
        higher_is_better=True,
        needs_stratify=True,
        make_cv=lambda n, seed: StratifiedKFold(n_splits=n, shuffle=True, random_state=seed),
        make_tree=lambda: HistGradientBoostingClassifier(random_state=42),
        make_linear=lambda: LogisticRegression(max_iter=1000),
        predict=lambda model, X: model.predict_proba(X)[:, 1],
    ),
    "regression": Task(
        name="regression",
        metric_name="negRMSE",
        metric=_neg_rmse,
        higher_is_better=True,        # neg_rmse で高いほど良いに正規化済
        needs_stratify=False,
        # 時系列予報なら TimeSeriesSplit に差し替えるだけ (自己相関対策)
        make_cv=lambda n, seed: KFold(n_splits=n, shuffle=True, random_state=seed),
        make_tree=lambda: HistGradientBoostingRegressor(random_state=42),
        make_linear=lambda: Ridge(),
        predict=lambda model, X: model.predict(X),
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}. Available: {sorted(TASKS)}")
    return TASKS[name]
