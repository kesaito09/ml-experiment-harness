"""汎用の例示特徴量 (ドメイン非依存)。

NFL ハーネスの 77 個のドメイン特徴量に相当する部分の "雛形"。
各 kind を 1 つずつ代表させ、taxonomy / feature_map のデモになるようにしてある。
target encoding は回帰でも分類でも成立する (target の群平均)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_registry import register_feature

_NUM_EXCLUDE = {"y", "segment"}


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _NUM_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


@register_feature("ratio")
def _ratio(df, train_df=None, folds=None, target="y", a=None, b=None, **params):
    """kind=ratio: 2 列の比 a/b (0除算は微小値で回避)。"""
    df = df.copy()
    cols = _numeric_cols(df)
    a = a or cols[0]
    b = b or cols[1]
    df[f"ratio_{a}_{b}"] = df[a] / (df[b].replace(0, np.nan)).fillna(df[b].mean() + 1e-9)
    return df


@register_feature("polynomial")
def _polynomial(df, train_df=None, folds=None, target="y", col=None, degree=2, **params):
    """kind=nonlinear (linear/nn 向け, 要標準化): col^degree。木は単調変換不変なので主に線形/NN用。"""
    df = df.copy()
    col = col or _numeric_cols(df)[0]
    df[f"{col}_p{degree}"] = df[col] ** degree
    return df


@register_feature("interaction")
def _interaction(df, train_df=None, folds=None, target="y", a=None, b=None, **params):
    """kind=interaction: 2 列の積 (木は split で暗黙表現可、線形は明示展開が必要)。"""
    df = df.copy()
    cols = _numeric_cols(df)
    a = a or cols[0]
    b = b or cols[1]
    df[f"x_{a}_{b}"] = df[a] * df[b]
    return df


@register_feature("standardize")
def _standardize(df, train_df=None, folds=None, target="y", cols=None, **params):
    """kind=group_zscore: train 統計で z-score (線形/NN が利用しやすい尺度)。"""
    df = df.copy()
    src = train_df if train_df is not None else df
    cols = cols or _numeric_cols(df)
    for c in cols:
        mu, sd = src[c].mean(), src[c].std()
        df[f"{c}_z"] = (df[c] - mu) / (sd + 1e-9)
    return df


@register_feature("target_encode")
def _target_encode(df, train_df=None, folds=None, target="y", col="segment",
                   out_col=None, smoothing=10.0, **params):
    """kind=target_encoding (OOF必須): カテゴリ列の OOF 平均ターゲット符号化。

    分類なら drafted 率、回帰なら目的値の群平均。CV fold = TE fold でリーク防止。
    """
    if train_df is None or folds is None:
        raise ValueError("target_encode requires train_df and folds")
    out_col = out_col or f"{col}_te"
    is_train = target in df.columns
    df = df.copy()
    gmean = train_df[target].mean()

    def fit(sub):
        s = sub.groupby(col)[target].agg(["mean", "count"])
        return (s["mean"] * s["count"] + gmean * smoothing) / (s["count"] + smoothing)

    if is_train:
        oof = np.full(len(train_df), gmean, dtype=float)
        for tr, va in folds:
            smooth = fit(train_df.iloc[tr])
            oof[va] = train_df.iloc[va][col].map(smooth).fillna(gmean).to_numpy()
        df[out_col] = oof
    else:
        smooth = fit(train_df)
        df[out_col] = df[col].map(smooth).fillna(gmean).to_numpy()
    return df
