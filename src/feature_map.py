"""特徴量マップ MD 生成器 — feature_taxonomy(SSOT) → docs/FEATURE_MAP.md。

手書きしないのでドリフトしない。`python -m src.runner --feature-map` で再生成。
"""
from __future__ import annotations

import src.features  # noqa: F401  registry を埋める
from src import feature_taxonomy as ft
from src.config import FEATURE_MAP_PATH
from src.feature_registry import list_features


def generate_feature_map() -> str:
    feats = list_features()
    cov = ft.check_coverage(feats)
    by_kind = ft.group_by_kind()
    L: list[str] = []
    L.append("# 特徴量マップ (自動生成)")
    L.append("")
    L.append("> `python -m src.runner --feature-map` で再生成。分類の編集は `src/feature_taxonomy.py` (SSOT)。")
    L.append("")
    L.append(f"登録: **{len(feats)}** / 分類済: **{len(ft.TAXONOMY)}** / 族: **{len(by_kind)}**")
    if cov["missing_in_taxonomy"] or cov["stale_in_taxonomy"]:
        L.append(f"> 🔴 カバレッジ不一致: {cov}")
    L.append("")
    L.append("## §1 族 × モデル適性")
    L.append("")
    L.append("| 族 | tree | linear | nn | scale | leak | 例 |")
    L.append("|---|---|---|---|---|---|---|")
    for kind, names in by_kind.items():
        e = ft.TAXONOMY[names[0]]
        cell = lambda m: "✓" if (m in e["models"] or "all" in e["models"]) else "—"
        leak = {"none": "—", "oof_required": "OOF必須", "leaky": "🔴"}[e["leak"]]
        L.append(f"| `{kind}` | {cell('tree')} | {cell('linear')} | {cell('nn')} "
                 f"| {e['scale']} | {leak} | {', '.join(names)} |")
    L.append("")
    L.append("## §2 レジストリ表 (YAML の feature 名がキー)")
    L.append("")
    L.append("| feature | kind | models | scale | leak | メモ |")
    L.append("|---|---|---|---|---|---|")
    for n in feats:
        e = ft.TAXONOMY[n]
        L.append(f"| `{n}` | {e['kind']} | {'/'.join(e['models'])} | {e['scale']} | {e['leak']} | {e['note']} |")
    L.append("")
    L.append("## §3 族単位の失敗の機序")
    L.append("")
    for kind in by_kind:
        L.append(f"- **`{kind}`**: {ft.KIND_NOTES.get(kind, '(未記入)')}")
    L.append("")
    return "\n".join(L)


def write_feature_map() -> None:
    FEATURE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_MAP_PATH.write_text(generate_feature_map())
    print(f"  [feature_map] regenerated ({FEATURE_MAP_PATH})")
