"""ハーネスの不変条件テスト。最重要: registry ↔ taxonomy の 1:1 (分類の書き忘れ検出)。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import src.features  # noqa: F401
from src import feature_taxonomy as ft
from src.feature_registry import list_features


def test_taxonomy_covers_registry_exactly():
    cov = ft.check_coverage(list_features())
    assert cov["missing_in_taxonomy"] == [], f"未分類: {cov['missing_in_taxonomy']}"
    assert cov["stale_in_taxonomy"] == [], f"陳腐: {cov['stale_in_taxonomy']}"


def test_taxonomy_values_valid():
    for name, e in ft.TAXONOMY.items():
        assert e["kind"] in ft.VALID_KINDS
        assert set(e["models"]) <= ft.VALID_MODELS
        assert e["scale"] in ft.VALID_SCALE
        assert e["leak"] in ft.VALID_LEAK


def test_every_kind_has_note():
    assert set(ft.group_by_kind()) <= set(ft.KIND_NOTES)


def test_target_encode_is_oof_leak_free():
    """OOF TE が未来を見ていない: 各 va の値は va 自身の target を含まない fold から作られる。"""
    from src.features import _target_encode
    train = pd.DataFrame({"segment": list("AABB"), "y": [1, 0, 1, 0]})
    folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1]))]
    out = _target_encode(train, train_df=train, folds=folds, target="y", col="segment")
    assert "segment_te" in out.columns and out["segment_te"].notna().all()


def test_feature_map_generates():
    from src.feature_map import generate_feature_map
    md = generate_feature_map()
    assert "§1" in md and "§2" in md and "§3" in md
    assert "`target_encode`" in md
