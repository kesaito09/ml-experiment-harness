# EXP_SUMMARY (自動生成)

> `python -m src.runner --summary` で全 YAML から再生成。**手動編集禁止**。

| exp / child | task | model | status | metric | mean | std | Δ vs parent | 説明 |
|---|---|---|---|---|---|---|---|---|
| EXP000/child-exp000_baseline | classification | tree | DONE | AUC | 0.9912 | 0.00136 |  | baseline (tree, 生特徴のみ) |
| EXP000/child-exp001_standardize | classification | tree | DONE | AUC | 0.9912 | 0.00136 | +0.00000 | + standardize |
| EXP000/child-exp002_segment_te | classification | tree | DONE | AUC | 0.991 | 0.001 | -0.00020 | + segment target encoding |
| EXP001/child-exp000_logreg_baseline | classification | linear | DONE | AUC | 0.9901 | 0.00128 |  | baseline (linear/LogReg, 生特徴) |
| EXP001/child-exp001_logreg_standardize | classification | linear | DONE | AUC | 0.99273 | 0.00234 | +0.00263 | + standardize (線形には効く想定) |
| EXP002/child-exp000_tree_linear_rankavg | classification | ensemble | DONE | AUC | 0.99341 | 0.00168 | +0.00221 | tree(raw)+linear(standardize) の rank平均アンサンブル |
