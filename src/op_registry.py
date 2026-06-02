"""Op Registry — 「手法 (operation)」を名前で登録・取得する (feature_registry の双子)。

繰り返す手法 (blend / weight_search / ...) を src/ に1回だけ書いて登録し、YAML の `method:` で呼ぶ。
script に閉じ込めず library 化することで、同じ/派生実験を YAML 差し替えだけで再利用できる。

統一インターフェース:
    func(oofs: list[np.ndarray], y, **params) -> np.ndarray   # blended OOF を返す
    - oofs: inputs で参照した親実験の (seed平均) OOF ベクトル群
    - y:    target
    - **params: 手法固有のパラメータ (weights, search 等)
"""
from __future__ import annotations

from typing import Callable

OP_REGISTRY: dict[str, Callable] = {}


def register_op(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        if name in OP_REGISTRY:
            raise ValueError(f"Op '{name}' is already registered.")
        OP_REGISTRY[name] = func
        return func
    return decorator


def get_op(name: str) -> Callable:
    if name not in OP_REGISTRY:
        raise ValueError(f"Unknown op: '{name}'. Available: {sorted(OP_REGISTRY)}")
    return OP_REGISTRY[name]


def list_ops() -> list[str]:
    return sorted(OP_REGISTRY)
