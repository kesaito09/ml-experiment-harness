"""サンプルデータ (sklearn 同梱、外部DL不要)。

テンプレートのデモ用。分類 (breast_cancer) と回帰 (diabetes) の両方を提供する。
実運用では、ここを自分のデータ読込に差し替える (返り値の規約: features + 'segment' + 'y')。
'segment' は categorical 列で target encoding と per-segment 診断の軸になる
(NFL ハーネスの Position に相当する汎用版)。
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from src.config import TARGET_COL


def _add_segment(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    df = df.copy()
    df["segment"] = pd.qcut(df[base_col], q=4, labels=["A", "B", "C", "D"]).astype(str)
    return df


def load_dataset(name: str) -> pd.DataFrame:
    """name -> train_df (features + segment + y)。"""
    if name == "breast_cancer":          # 二値分類サンプル
        ds = load_breast_cancer(as_frame=True)
        df = ds.frame.rename(columns={"target": TARGET_COL})
        df.columns = [c.replace(" ", "_") for c in df.columns]
        return _add_segment(df, base_col="mean_radius")
    if name == "diabetes":               # 回帰サンプル
        ds = load_diabetes(as_frame=True)
        df = ds.frame.rename(columns={"target": TARGET_COL})
        return _add_segment(df, base_col="bmi")
    raise ValueError(f"Unknown dataset: {name}. Use 'breast_cancer' or 'diabetes', "
                     f"or 実データ読込に差し替えてください。")
