---
name: feature-experiment
description: |
  ML 実験ハーネス (このリポジトリ) で新しい特徴量・モデル・アンサンブルを実験するための
  オーケストレーション・スキル。「○○特徴量を追加したい」「○○を試したい」「アイディアを実装して」で起動。
  pre-flight (ANTI_PATTERNS / FINDINGS / EXP_SUMMARY / FEATURE_MAP) → AskUserQuestion → FE実装+taxonomy登録
  → YAML 生成 → python -m src.runner で実行 → 結果・再評価クロスチェック・次提案、を 1 セッションで完遂する。

  以下で使う:
  - 「特徴量を追加したい」「新しいFE試したい」「○○を比較したい」と言われた時
  - 「アイディアを実装して」と言われた時
  - /feature-experiment で明示的に呼ばれた時
  - スコア改善のための実験 (FE / モデル / アンサンブル) をしたい時
user_invocable: true
---

# feature-experiment スキル (汎用 ML ハーネス版)

ユーザーが「○○ を追加したい」「○○ を試したい」と言ったら、以下のプロトコルで 1 実験を最後まで完遂する。

## 前提 (プロジェクト構造)

```
{project_root}/
├── src/
│   ├── runner.py            # run_from_yaml(), CLI (--config/--summary/--feature-map/--features)
│   ├── feature_registry.py  # @register_feature
│   ├── features.py          # @register_feature 付き特徴量関数
│   ├── feature_taxonomy.py  # 機械可読 SSOT (kind/models/scale/leak) ← FE追加時に必ず分類追加
│   ├── tasks.py             # classification / regression の Task 定義 (指標/CV/モデル)
│   └── data.py              # データ読込 (規約: features + segment + y)
├── experiments-sample/      # 同梱サンプル (EXP000/001/002, 分類)
│   └── EXP00N/configs/      # child-exp YAML 群
├── experiments/             # 利用者が作る実験 (任意)
├── docs/
│   ├── ANTI_PATTERNS.md     # 失敗 (Claude が追記(自動生成でない), 🔓再評価トリガ)
│   ├── FINDINGS.md          # 成功・新事実 (Claude が追記(自動生成でない), ⚠️失効トリガ)
│   ├── EXP_SUMMARY.md       # 自動生成 (手動編集禁止)
│   └── FEATURE_MAP.md       # 自動生成 (手動編集禁止)
└── tests/test_harness.py
```

存在しない場合は「まずハーネスを構築してください」と伝える。

---

## プロトコル

### Step 0: pre-flight チェック (必須)

実験提案の前に必ず読む:

1. **`docs/ANTI_PATTERNS.md`** — 試す前に NG と判明している施策を避ける
2. **`docs/FINDINGS.md`** — 既に「効く/真と確認済み」を確認 → 蒸し返し防止・効く手を活かす
3. **`docs/EXP_SUMMARY.md`** — 過去の類似実験の結果
4. **`docs/FEATURE_MAP.md`** — 提案FEの**族 (kind)** を §2 で特定、§3 の失敗機序を確認

該当時:
- ANTI_PATTERNS に該当 → 提示して「それでも試す？」
- FEATURE_MAP §3 の族失敗機序に該当 → 機序を提示して確認
- **再評価トリガ照合**: 今の新事実が ANTI_PATTERNS の `🔓再評価トリガ` を満たすなら、その「再挑戦候補」を提案に含める

### Step 1: コンテキスト把握

- `docs/EXP_SUMMARY.md` (直近トレンド)、既存 `experiments*/EXP*/configs/*.yaml` の最近 2-3 個 (YAMLスタイル)

### Step 2: 意図確認 (AskUserQuestion で 1-4 問)

1. **タスク**: classification / regression (どの問題か)
2. **ベース構成**: 現best / baseline / 別 child-exp
3. **新規 vs 既存**: `src/features.py` に該当関数があるか (grep)
4. **パラメータ / バリエーション**: smoothing, col, 複数試すか

### Step 3: EXP 境界判定

| 依頼 | 振り分け |
|---|---|
| 特徴量追加・FE比較・組合せ | **同 EXP の child-exp** |
| ハイパラ振り (同一モデル内) | **同 EXP の child-exp** |
| モデル種類変更 (tree↔linear 等) | **新 EXP** |
| アンサンブル (rank avg) | **新 EXP** |

迷ったら AskUserQuestion で確認。新 EXP は `experiments/EXP00N/configs/` を作る (3桁)。

### Step 4: 新 FE 実装 (必要なら) — taxonomy 登録まで必須

`src/features.py` を grep して無ければ追加:

```python
@register_feature("new_feature")
def _new_feature(df, train_df=None, folds=None, target="y", **params):
    """説明."""
    df = df.copy()
    # 実装 (TE 等の target 依存は folds で OOF, リーク防止)
    return df
```

統一インターフェース: `df` / `train_df` (target 列を持つ) / `folds` `[(tr,va),...]` / `target="y"` / `**params`。

**そして必ず `src/feature_taxonomy.py` の `TAXONOMY` に分類を追加**:
```python
"new_feature": _f("<kind>", ["tree"/"linear"/"nn"/"all"], "<scale>", "<leak>", "メモ"),
```
追加し忘れると `tests/test_harness.py::test_taxonomy_covers_registry_exactly` が落ちる (=ガード)。

テスト追加 + 実行:
```bash
PYTHONPATH=. python -m pytest tests/ -q
```

### Step 5: YAML 生成

連番確認 (`ls experiments/EXP00N/configs/`)、`child-exp{NNN}_{short}.yaml` で作成:

```yaml
meta:
  exp: EXP00N
  child: child-exp{NNN}_{short}
  name: child-exp{NNN}_{short}
  description: "1行"
  parent_child: child-exp000_baseline
  prompt: "ユーザーの仮説/指示"
config:
  dataset: breast_cancer      # or 自分のデータ
  task: classification        # classification / regression
  model: tree                 # tree / linear / ensemble
  stage: 1                    # 1=5seed screening / 2=11seed 認定
  parent_metric: <baseの metric_mean>
  features:
  - standardize
  # - { name: target_encode, params: { col: segment, smoothing: 20 } }
status: TODO
```

ユーザーに YAML を見せて **「これで実行します。よろしいですか？」** と確認。

### Step 6: 実行

```bash
PYTHONPATH=. python -m src.runner --config experiments/EXP00N/configs/child-exp{NNN}_*.yaml
```

自動で: repeated CV → YAML に `status: DONE` + `result` 追記 → `EXP_SUMMARY.md` / `FEATURE_MAP.md` 再生成 → taxonomy カバレッジ警告。

### Step 6.5: FINDINGS / ANTI_PATTERNS 追記 + 再評価クロスチェック

横断的に効いた/効かなかった/真と分かった事を蓄積:
- 効いた施策・新事実 → `docs/FINDINGS.md` (`⚠️失効トリガ` を具体的に)
- 試す価値なしと判明 → `docs/ANTI_PATTERNS.md` (条件付きなら `🔓再評価トリガ`)

**追記したら再評価クロスチェック**: FINDINGS 追記 → ANTI_PATTERNS の `🔓再評価トリガ` と EXP_SUMMARY の failed/marginal を走査し、トリガを満たす過去失敗を「再挑戦候補」として提示。判断は LLM (自動キーワード判定は禁止)。

### Step 7: 結果と次提案

```
✅ 結果: {metric} {mean} (Δ {±} vs parent), std {std}, category worked/marginal/failed
考察: [2-3 行]
次の候補: 1. ... 2. ... 3. ... (Step6.5 の再挑戦候補があれば含める)
```

---

## 失敗時の対応

- pytest 失敗 → 修正・再テスト (3回失敗でエスカレート)
- runner クラッシュ → 原因共有
- 不明点 → AskUserQuestion (推測で進めない)

## 重要事項

| 項目 | ルール |
|---|---|
| **random_state** | `src/config.py:SEED=42` |
| **判定** | **単一seed不可**、`stage` の repeated CV (5→11 seed)。Δ閾値はタスク次第 (例: AUC +0.005=worked) |
| **CV fold と TE fold** | 同一 (runner が同 seed で再生成) |
| **新 FE** | `@register_feature` + **`feature_taxonomy.TAXONOMY` 追加** + `tests/` (テストが強制) |
| **EXP_SUMMARY / FEATURE_MAP** | 自動生成 (手動編集禁止) |
| **ANTI_PATTERNS / FINDINGS** | Claude が追記(自動生成でない)、pre-flight で必ず参照、追記時に再評価クロスチェック |
| **task** | classification / regression を YAML で選ぶ。時系列なら `tasks.py` の CV を TimeSeriesSplit に |
