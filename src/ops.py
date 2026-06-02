"""登録 op (再利用可能な手法)。新しい手法はここに1回書いて @register_op するだけ。

各 op は親実験の OOF ベクトル群 (oofs) を受け、blended OOF を返す。
YAML から `method: <op名>` + `inputs: [親EXP/child...]` + `params: {...}` で呼ばれる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.op_registry import register_op


def _ranks(o: np.ndarray) -> np.ndarray:
    """0-1 正規化した順位 (スケール差を吸収して混ぜる)。"""
    return (pd.Series(o).rank().to_numpy() - 1) / (len(o) - 1)


@register_op("rank_blend")
def rank_blend(oofs, y=None, weights=None, **params):
    """各 OOF を順位正規化して加重平均 (weights 省略時は等重み)。"""
    R = np.column_stack([_ranks(o) for o in oofs])
    w = np.asarray(weights, dtype=float) if weights is not None else np.ones(R.shape[1])
    w = w / w.sum()
    return R @ w


@register_op("weight_search")
def weight_search(oofs, y, grid=11, **params):
    """少数メンバーの重みをグリッド探索し、metric 最大の blend と best_weights を返す。

    返り値 = (blended_oof, {"best_weights": [...]})。runner が best_weights を記録に merge。
    """
    import itertools

    from src.tasks import get_task
    task = get_task(params.get("task", "classification"))
    R = [_ranks(o) for o in oofs]
    n = len(R)
    best_score, best_w = -np.inf, None
    axis = np.linspace(0, 1, grid)
    for combo in itertools.product(axis, repeat=n):
        s = sum(combo)
        if s == 0:
            continue
        w = np.array(combo) / s
        blend = sum(wi * Ri for wi, Ri in zip(w, R))
        score = task.metric(y, blend)
        if score > best_score:
            best_score, best_w = score, w
    blended = sum(wi * Ri for wi, Ri in zip(best_w, R))
    return blended, {"best_weights": [round(float(x), 3) for x in best_w]}
