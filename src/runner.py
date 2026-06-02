"""実験ランナー — YAML から実験実行 + 自動記録 + 集約 (ハーネスの心臓)。

フロー:
    YAML → データ読込 → registry で特徴量構築 → repeated CV (OOF) → per-segment 診断
         → YAML に result 追記 → EXP_SUMMARY.md / FEATURE_MAP.md 再生成

使い方:
    python -m src.runner --config experiments/exp001_classification_baseline.yaml
    python -m src.runner --summary
    python -m src.runner --feature-map
    python -m src.runner --features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import src.features  # noqa: F401  feature registry を埋める
import src.ops        # noqa: F401  op registry を埋める
from src import feature_taxonomy as ft
from src.config import (N_FOLDS, PROJECT_ROOT, STAGE1_SEEDS, STAGE2_SEEDS,
                        SUMMARY_PATH, TARGET_COL)
from src.data import load_dataset
from src.diagnostics import per_segment_diagnostic, repeated_cv
from src.feature_registry import get_feature, list_features
from src.op_registry import get_op
from src.tasks import get_task


# --- 特徴量パイプライン ---
def build_features(df, train_df, folds, features):
    for feat in features:
        if isinstance(feat, str):
            name, params = feat, {}
        else:
            name, params = feat["name"], feat.get("params", {}) or {}
        df = get_feature(name)(df, train_df=train_df, folds=folds,
                               target=TARGET_COL, **params)
    return df


def _model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """数値列だけをモデル入力に (target と raw categorical は除外)。"""
    drop = {TARGET_COL, "segment"}
    num = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return df[num].fillna(df[num].median(numeric_only=True))


# --- 1 seed の OOF CV ---
def _make_model(task, model_kind: str):
    """model_kind = tree / linear。モデル種の変更は新 EXP の単位 (EXP境界ルール)。"""
    return task.make_linear() if model_kind == "linear" else task.make_tree()


def _oof_one_seed(full: pd.DataFrame, features, task, seed: int, model_kind: str = "tree"):
    """特徴量を full train で1回構築 (TE は folds で OOF=leak-free) → 同じ folds で model CV。"""
    y = full[TARGET_COL].to_numpy()
    cv = task.make_cv(N_FOLDS, seed)
    folds = list(cv.split(full, y))
    built = build_features(full.copy(), train_df=full, folds=folds, features=features)
    X = _model_matrix(built)
    oof = np.zeros(len(full), dtype=float)
    for tr, va in folds:
        model = _make_model(task, model_kind)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = task.predict(model, X.iloc[va])
    return oof


def _oof_ensemble_one_seed(full, members, task, seed):
    """アンサンブル: 各メンバーの OOF を rank-average で混合 (ensemble = 新 EXP の単位)。

    members = [{model: tree, features:[...]}, {model: linear, features:[...]}]
    メンバーごとに整形を変えられる (木=raw / 線形=standardize) = per-model FE の思想。
    """
    ranks = []
    for mem in members:
        oof = _oof_one_seed(full, mem.get("features", []) or [], task, seed, mem["model"])
        ranks.append(pd.Series(oof).rank().to_numpy() / len(oof))  # 順位を 0-1 に正規化
    return np.mean(ranks, axis=0)


def _oof_path(yaml_path: Path, child: str) -> Path:
    """この実験の OOF キャッシュ先 (experiments*/<exp>/outputs/<child>_oof.npy)。"""
    return Path(yaml_path).parent.parent / "outputs" / f"{child}_oof.npy"


def _load_parent_oof(ref: str) -> np.ndarray:
    """親実験の OOF を ref='EXP000/child-exp000_baseline' で再利用ロード。"""
    exp, child = ref.split("/")
    hits = list(PROJECT_ROOT.glob(f"experiments*/{exp}/outputs/{child}_oof.npy"))
    if not hits:
        raise FileNotFoundError(
            f"cached OOF が無い: {ref} → 親実験を先に実行してOOFを生成せよ")
    return np.load(hits[0])


def run_from_yaml(path: Path, rerun: bool = False) -> None:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("status") == "DONE" and not rerun:
        print(f"  already DONE: {path} (use --rerun)")
        return
    dataset = cfg["config"]["dataset"]
    full = load_dataset(dataset)
    task = get_task(cfg["config"]["task"])
    y = full[TARGET_COL].to_numpy()
    child = cfg["meta"]["child"]
    method = cfg["config"].get("method")          # op 名 (あれば再利用パス)

    # ── 再利用パス: 登録 op を inputs(親OOF) に適用 (blend / weight_search 等) ──
    if method:
        inputs = cfg["config"]["inputs"]
        params = dict(cfg["config"].get("params", {}) or {})
        params.setdefault("task", task.name)
        oofs = [_load_parent_oof(r) for r in inputs]
        print(f"  method={method} inputs={inputs}")
        out = get_op(method)(oofs, y=y, **params)
        blended, extras = out if isinstance(out, tuple) else (out, {})  # op は (oof, extras) 返却可
        metric = round(float(task.metric(y, blended)), 5)
        diag = per_segment_diagnostic(full, blended, task)
        op_path = _oof_path(path, child); op_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(op_path, blended)                 # 派生元としても再利用可
        parent = cfg["config"].get("parent_metric")
        result = {"task": task.name, "method": method, "inputs": inputs,
                  "metric_name": task.metric_name, "metric_mean": metric,
                  "delta_vs_parent": (round(metric - parent, 5) if parent is not None else None),
                  "segment_scores": diag["segment_scores"]}
        result.update(extras)                     # weight_search の best_weights 等を記録に merge
        cfg["status"] = "DONE"; cfg["result"] = result
        Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        print(f"  [{Path(path).name}] status=DONE ({method}), metric_mean={metric}")
        write_summary()
        return

    # ── 標準パス: features + model で repeated CV ──
    features = cfg["config"].get("features", []) or []
    model_kind = cfg["config"].get("model", "tree")
    members = cfg["config"].get("members")
    stage = cfg["config"].get("stage", 1)
    seeds = STAGE1_SEEDS if stage == 1 else STAGE2_SEEDS

    desc_feat = (f"ensemble{[m['model'] for m in members]}" if model_kind == "ensemble"
                 else [f if isinstance(f, str) else f["name"] for f in features])
    print(f"  dataset={dataset} task={task.name} model={model_kind} metric={task.metric_name} "
          f"{desc_feat}")

    oofs_accum = []
    def one(seed):
        if model_kind == "ensemble":
            oof = _oof_ensemble_one_seed(full, members, task, seed)
        else:
            oof = _oof_one_seed(full, features, task, seed, model_kind)
        oofs_accum.append(oof)
        return task.metric(y, oof)
    rcv = repeated_cv(one, seeds)
    seed_avg_oof = np.mean(oofs_accum, axis=0)     # ★ 派生実験が再利用する成果物
    diag = per_segment_diagnostic(full, seed_avg_oof, task)

    for s, sc in diag["segment_scores"].items():
        print(f"    segment {s}: {sc}")
    print(f"  [repeated CV/{rcv['n_seeds']}seed] {task.metric_name} {rcv['mean']} ± {rcv['std']}")

    # OOF をキャッシュ (これで後続の blend/weight_search が inputs で参照できる)
    oof_path = _oof_path(path, child); oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, seed_avg_oof)

    parent = cfg["config"].get("parent_metric")
    delta = round(rcv["mean"] - parent, 5) if parent is not None else None
    cfg["status"] = "DONE"
    cfg["result"] = {
        "task": task.name, "metric_name": task.metric_name,
        "metric_mean": rcv["mean"], "metric_std": rcv["std"],
        "per_seed": rcv["per_seed"], "n_seeds": rcv["n_seeds"],
        "delta_vs_parent": delta,
        "segment_scores": diag["segment_scores"],
    }
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print(f"  [{Path(path).name}] status=DONE, metric_mean={rcv['mean']} (OOF cached: {oof_path.name})")
    write_summary()


# --- 集約 (自動生成) ---
def generate_summary() -> str:
    rows = []
    # experiments-sample/ (同梱サンプル) と experiments/ (利用者が作る) の両方を走査
    yamls = sorted(PROJECT_ROOT.glob("experiments*/EXP*/configs/*.yaml"))
    for yf in yamls:
        cfg = yaml.safe_load(yf.read_text())
        if not isinstance(cfg, dict):
            continue
        r = cfg.get("result", {}) or {}
        exp_id = f"{yf.parent.parent.name}/{yf.stem}"   # 例: EXP000/child-exp000_baseline
        conf = cfg.get("config", {})
        method_or_model = conf.get("method") or conf.get("model", "tree")  # op なら method を表示
        rows.append((exp_id, conf.get("task", "?"),
                     method_or_model,
                     cfg.get("status", "?"), r.get("metric_name", ""),
                     r.get("metric_mean", ""), r.get("metric_std", ""),
                     r.get("delta_vs_parent", ""),
                     cfg.get("meta", {}).get("description", "")))
    L = ["# EXP_SUMMARY (自動生成)", "",
         "> `python -m src.runner --summary` で全 YAML から再生成。**手動編集禁止**。", "",
         "| exp / child | task | model | status | metric | mean | std | Δ vs parent | 説明 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for exp_id, tsk, mdl, st, mn, mean, std, d, desc in rows:
        dd = f"{d:+.5f}" if isinstance(d, (int, float)) else ""
        L.append(f"| {exp_id} | {tsk} | {mdl} | {st} | {mn} | {mean} | {std} | {dd} | {desc} |")
    return "\n".join(L) + "\n"


def write_summary() -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(generate_summary())
    print(f"  [EXP_SUMMARY.md] regenerated ({SUMMARY_PATH})")
    _warn_taxonomy_coverage()
    from src.feature_map import write_feature_map
    write_feature_map()  # 派生は常に再生成 → 宙ぶらりん防止


def _warn_taxonomy_coverage() -> None:
    cov = ft.check_coverage(list_features())
    if cov["missing_in_taxonomy"]:
        print(f"  🔴 [taxonomy] 未分類: {cov['missing_in_taxonomy']} → feature_taxonomy.py に追加")
    elif not cov["stale_in_taxonomy"]:
        print("  [taxonomy] coverage OK (registry ↔ taxonomy 一致)")


def main() -> None:
    ap = argparse.ArgumentParser(description="汎用MLハーネス runner")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--feature-map", action="store_true")
    ap.add_argument("--features", action="store_true")
    args = ap.parse_args()
    if args.features:
        for n in list_features():
            print(f"  - {n}  ({ft.get(n)['kind']})")
    elif args.feature_map:
        from src.feature_map import write_feature_map
        write_feature_map()
    elif args.summary:
        write_summary()
    elif args.config:
        run_from_yaml(args.config, rerun=args.rerun)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
