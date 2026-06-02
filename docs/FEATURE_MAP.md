# 特徴量マップ (自動生成)

> `python -m src.runner --feature-map` で再生成。分類の編集は `src/feature_taxonomy.py` (SSOT)。

登録: **5** / 分類済: **5** / 族: **5**

## §1 族 × モデル適性

| 族 | tree | linear | nn | scale | leak | 例 |
|---|---|---|---|---|---|---|
| `interaction` | ✓ | — | — | none | — | interaction |
| `nonlinear` | — | ✓ | ✓ | standardize | — | polynomial |
| `ratio` | ✓ | ✓ | ✓ | none | — | ratio |
| `group_zscore` | — | ✓ | ✓ | zscored | — | standardize |
| `target_encoding` | ✓ | ✓ | ✓ | bounded01 | OOF必須 | target_encode |

## §2 レジストリ表 (YAML の feature 名がキー)

| feature | kind | models | scale | leak | メモ |
|---|---|---|---|---|---|
| `interaction` | interaction | tree | none | none | 木はsplitで暗黙表現可、線形は明示展開が必要 |
| `polynomial` | nonlinear | linear/nn | standardize | none | 木は単調変換不変→主に線形/NN向け |
| `ratio` | ratio | all | none | none | 2列の比。冗長になりやすい |
| `standardize` | group_zscore | linear/nn | zscored | none | 木には不要、線形/NNが利用しやすい尺度に |
| `target_encode` | target_encoding | all | bounded01 | oof_required | OOF必須。キーを深くすると小Nで過学習 |

## §3 族単位の失敗の機序

- **`interaction`**: 測定値の積。木は split で暗黙表現できるが**線形は明示展開しないと表現不能**。
- **`nonlinear`**: log/累乗/poly。**木は単調変換に不変なので基本効かない**。価値が出るのは線形/NN。
- **`ratio`**: 「X あたり Y」。単体寄与は小さく冗長になりがち。
- **`group_zscore`**: 群内/全体の標準化。木には不要、線形/NN には重要。group_col の選択が肝。
- **`target_encoding`**: OOF必須 (CV fold = TE fold)。**キーを深くするほど小Nで過学習**。smoothing で緩和。
