"""ハーネス全体の定数 (Single Source of Truth)。

タスク非依存の規律 (seed / fold / 2段階screening) をここに集約する。
metric と CV 分割だけがタスク固有 → tasks.py で切り替える。
"""
from __future__ import annotations

from pathlib import Path

SEED = 42
N_FOLDS = 5
TARGET_COL = "y"

# 2段階スクリーニング: 単一seedの楽観バイアスを排除する規律 (NFL ハーネスから継承)
STAGE1_SEEDS = [42, 7, 123, 2024, 99]                       # 1次: 容疑者抽出
STAGE2_SEEDS = STAGE1_SEEDS + [1, 17, 2718, 31337, 8, 256]  # 2次: 認定

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SUMMARY_PATH = DOCS_DIR / "EXP_SUMMARY.md"
FEATURE_MAP_PATH = DOCS_DIR / "FEATURE_MAP.md"
