"""検証規律 — repeated CV と per-segment 汎化診断 (タスク非依存)。

NFL ハーネスの「2段階スクリーニング」「per-position 過学習検出」を汎用化した版。
小Nセグメントで勝ち・大Nで負けるパターン (087 型 overfit) を per-segment で検出する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.tasks import Task

SMALL_N = 80      # 小Nセグメント閾値
LARGE_N = 150     # 大Nセグメント閾値


def per_segment_diagnostic(df: pd.DataFrame, oof: np.ndarray, task: Task,
                           segment_col: str = "segment", target: str = "y",
                           min_n: int = 20) -> dict:
    """セグメント別 metric を集計し、汎化の偏りを見る (per-position 診断の汎用版)。"""
    y = df[target].to_numpy()
    seg = df[segment_col].astype(str)
    out: dict[str, dict] = {}
    for s in sorted(seg.unique()):
        m = (seg == s).to_numpy()
        if m.sum() < min_n:
            continue
        try:
            score = task.metric(y[m], np.asarray(oof)[m])
        except ValueError:
            continue
        out[s] = {"n": int(m.sum()), task.metric_name: round(float(score), 5)}
    return {"segment_scores": out}


def repeated_cv(run_one_seed, seeds) -> dict:
    """seed ごとに run_one_seed(seed)->float を回し、mean±std を返す (楽観バイアス排除)。"""
    scores = [run_one_seed(s) for s in seeds]
    return {
        "per_seed": [round(float(x), 5) for x in scores],
        "mean": round(float(np.mean(scores)), 5),
        "std": round(float(np.std(scores)), 5),
        "n_seeds": len(seeds),
    }
