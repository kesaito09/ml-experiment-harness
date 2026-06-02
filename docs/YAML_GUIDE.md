# 実験 YAML の読み方

1つの YAML = **1実験の完全な記録**。「何をして（identity）/どうだったか（outcome）/どう生成したか（provenance）」が
全部ここに残る。**runner で動かしても script で動かしても、記録はこの同じ schema で SSOT に残る**のが核心。

> 最重要: **使った手法は全て YAML に記録される。**
> 特徴量（どの `feature` を使ったか）も、合成手法（どの `op`/`method` か）も、その `params`・`inputs` も。
> だから YAML を読めば「この実験が何をしたか」が一意に分かり、そのまま再現・派生できる。

---

## 共通構造

```yaml
meta:        # この実験は何者か
  exp: EXP000
  child: child-exp001_standardize
  name: child-exp001_standardize
  description: "+ standardize"
  parent_child: child-exp000_baseline   # 何を base に派生したか
  source: yaml                          # yaml(runner) / script (任意のpythonから記録)

config:      # ★何をしたか（手法の完全な指定）← ここが記録の主役
  ...

result:      # どうだったか（runner が自動追記）
  ...

status: TODO | RUNNING | DONE | FAILED
```

---

## A) 特徴量 + モデルの実験（標準パス）

```yaml
config:
  dataset: breast_cancer
  task: classification          # classification / regression
  model: tree                   # tree / linear / ensemble
  stage: 1                      # 1=5seed / 2=11seed
  parent_metric: 0.9912         # Δ 計算用
  features:                     # ★使った特徴量を「登録名」で全列挙 → 何を入れたか一目
  - standardize
  - name: target_encode         # パラメータ付きはこの形
    params: { col: segment, smoothing: 20 }
```

**読み方**: `features` に並ぶ名前が、その実験で投入した特徴量の全て（`src/feature_registry.py` に登録された関数名）。
各特徴量の素性（族・モデル適性・正規化・leak）は `docs/FEATURE_MAP.md` で名前を引けば分かる。
→ 「どの特徴量を、どのパラメータで使ったか」が YAML だけで完結して読める。

`model: ensemble` の場合は members も全部記録される:
```yaml
  model: ensemble
  members:
  - { model: tree,   features: [] }              # 木は raw
  - { model: linear, features: [standardize] }   # 線形は標準化（メンバー別整形も記録）
```

---

## B) 手法（op）の実験 — 親の成果物を再利用（再利用パス）

```yaml
config:
  dataset: breast_cancer
  task: classification
  method: weight_search         # ★使った手法(op)を「登録名」で記録 (rank_blend / weight_search …)
  inputs:                       # ★何を入力にしたか（親実験の OOF）も記録
  - EXP000/child-exp000_baseline
  - EXP001/child-exp000_logreg_baseline
  params: { grid: 11 }          # 手法のパラメータも記録
  parent_metric: 0.9912
```

**読み方**: `method` が「どの手法で合成したか」（blend か weight 探索か等）、`inputs` が「どの実験の OOF を再利用したか」。
手法は `src/ops.py` に登録された op（`@register_op`）で、名前で一意に辿れる。
→ **blend だったのか重み探索だったのかが YAML に明記される**ので、後から見て手法が分かる／差し替えて派生できる。

---

## result の読み方（runner が自動追記）

```yaml
result:
  task: classification
  metric_name: AUC
  metric_mean: 0.99436          # repeated CV 平均（標準パス）or 合成後スコア（op パス）
  metric_std: 0.00234           # seed 間ブレ（op パスでは省略のことあり）
  per_seed: [...]               # 各 seed のスコア
  delta_vs_parent: 0.00316      # parent_metric との差
  segment_scores: {...}         # per-segment 診断（過学習の偏りを見る）
  best_weights: [0.4, 0.6]      # ★op が返した付帯情報も記録（weight_search の最適重み等）
```

---

## 全手法が記録される、とは（このハーネスの肝）

| 何をした | YAML のどこに残るか | 名前の辿り先 |
|---|---|---|
| 特徴量を入れた | `config.features`（登録名＋params） | `src/feature_registry.py` / `docs/FEATURE_MAP.md` |
| モデルを選んだ | `config.model` / `config.members` | `src/tasks.py` |
| 合成手法を使った | `config.method`（op登録名）＋`config.params` | `src/ops.py` (`@register_op`) |
| 親の成果物を再利用した | `config.inputs`（親 EXP/child） | 親の `outputs/*_oof.npy` |
| 結果・付帯情報 | `result.*`（metric / Δ / segment / best_weights …） | runner が自動追記 |

→ **特徴量も op も「登録された名前」で記録される**から、YAML は実験の完全な設計図になる。
同じ実験の再現も、`features`/`method`/`inputs` を差し替えるだけの派生も、YAML を読んで書くだけ。
これが「script で動かしても記録は YAML に残す」＝ SSOT を一様に保つ思想の実体。

---

### 派生のしかた（早見）

| やりたいこと | 変える場所 |
|---|---|
| 同じ実験を再現 | その YAML をそのまま `--config` で再実行 |
| 特徴量を1つ足す | `config.features` に名前を追加 |
| 手法を blend→重み探索に | `config.method` を差し替え |
| 別の親 OOF を混ぜる | `config.inputs` を変更 |
| 分類→回帰 | `config.task` を `regression` に（`tasks.py` の seam） |
