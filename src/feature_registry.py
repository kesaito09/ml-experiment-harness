"""Feature Registry — 特徴量関数を名前で登録・取得する (NFL ハーネスから継承の中核パターン)。

YAML config から "name" だけで特徴量を呼び出せる。runner はこの registry 経由で
パイプラインを組み立てる。統一インターフェース:

    func(df, train_df=None, folds=None, target="y", **params) -> pd.DataFrame
"""
from __future__ import annotations

from typing import Callable

FEATURE_REGISTRY: dict[str, Callable] = {}


def register_feature(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        if name in FEATURE_REGISTRY:
            raise ValueError(f"Feature '{name}' is already registered.")
        FEATURE_REGISTRY[name] = func
        return func
    return decorator


def get_feature(name: str) -> Callable:
    if name not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature: '{name}'. Available: {sorted(FEATURE_REGISTRY)}")
    return FEATURE_REGISTRY[name]


def list_features() -> list[str]:
    return sorted(FEATURE_REGISTRY)
