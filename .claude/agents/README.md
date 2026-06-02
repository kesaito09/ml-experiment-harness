# Agent Team (雛形)

このハーネスに付属する**助言専任のエージェントチーム**。実装はしない（読み取り専任）。
Leader (メインの Claude) が両者を起動し、2人を **SendMessage で直接議論**させて
「この実験は方法論的に健全か × ドメイン的に説明がつくか」を二面で詰める。

| エージェント | 役割 | model | 転用性 |
|---|---|---|---|
| [`ml-methodology-expert`](ml-methodology-expert.md) | 検証方法論・過学習規律・2段階screening・per-segment診断・再評価 | opus | **そのまま転用可** |
| [`domain-expert`](domain-expert.md) | ドメイン的に説明がつくか・物理/構造的根拠・上振れの物語 | sonnet | **テンプレ**（自分のドメインに置換） |

## 思想（なぜ2人か）

CV 上振れには2種類ある: **本物のシグナル** と **過学習の偶然**。これを分けるには2つの問いが要る:
1. (ml-expert) 「方法論的に信頼できるか？」— repeated CV / 2段階 / リーク / per-segment 診断
2. (domain-expert) 「ドメインの物語があるか？」— 説明できない上振れは却下

両方 yes のときだけ「本物」と認定する。ml-expert が domain-expert に
「この per-segment の勝ちに物語はあるか？」を問い、取れなければ採用しない。

## 使い方

```
（Leader から）
- ml-methodology-expert と domain-expert をチームとして起動
- 議題を渡す（例: 「child-exp003 の +0.004 は本物か？」）
- 2人に SendMessage で相互議論させ、両者の verdict を統合する
```

## domain-expert を自分のドメインに適応する

`domain-expert.md` 末尾の手順参照。要点:
1. `name`/`description` を `<domain>-expert` に
2. 本文の【...】プレースホルダを実ドメインの言葉に置換
3. `docs/domain/<NNNN>_domain_expert_brief.md`（手編集の差し替え口）を作る
