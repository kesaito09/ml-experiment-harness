"""特徴量分類 taxonomy — 機械可読 SSOT (NFL ハーネスの中核資産を汎用化)。

各特徴量を kind/models/scale/leak で分類。相関分析を族単位で切る・FEATURE_MAP.md を生成する・
registry とのカバレッジを test で強制する、の元データ。
「なぜ失敗したか」の散文は KIND_NOTES (生成MD §3) に置く (enum は理由を持てない)。
"""
from __future__ import annotations

VALID_KINDS = {
    "raw_transform", "ratio", "nonlinear", "interaction",
    "group_zscore", "target_encoding", "missing", "meta",
}
VALID_MODELS = {"tree", "linear", "nn", "all"}
VALID_SCALE = {"none", "standardize", "zscored", "bounded01", "categorical"}
VALID_LEAK = {"none", "oof_required", "leaky"}


def _f(kind, models, scale, leak, note=""):
    return {"kind": kind, "models": models, "scale": scale, "leak": leak, "note": note}


TAXONOMY: dict[str, dict] = {
    "ratio": _f("ratio", ["all"], "none", "none", "2列の比。冗長になりやすい"),
    "polynomial": _f("nonlinear", ["linear", "nn"], "standardize", "none",
                     "木は単調変換不変→主に線形/NN向け"),
    "interaction": _f("interaction", ["tree"], "none", "none",
                      "木はsplitで暗黙表現可、線形は明示展開が必要"),
    "standardize": _f("group_zscore", ["linear", "nn"], "zscored", "none",
                      "木には不要、線形/NNが利用しやすい尺度に"),
    "target_encode": _f("target_encoding", ["all"], "bounded01", "oof_required",
                        "OOF必須。キーを深くすると小Nで過学習"),
}

KIND_NOTES: dict[str, str] = {
    "ratio": "「X あたり Y」。単体寄与は小さく冗長になりがち。",
    "nonlinear": "log/累乗/poly。**木は単調変換に不変なので基本効かない**。価値が出るのは線形/NN。",
    "interaction": "測定値の積。木は split で暗黙表現できるが**線形は明示展開しないと表現不能**。",
    "group_zscore": "群内/全体の標準化。木には不要、線形/NN には重要。group_col の選択が肝。",
    "target_encoding": "OOF必須 (CV fold = TE fold)。**キーを深くするほど小Nで過学習**。smoothing で緩和。",
    "raw_transform": "生列の素変換。baseline 構成要素。",
    "missing": "欠損フラグ/パターン。production 代理でなく観測慣習を捉えることが多い。",
    "meta": "ユーティリティ (列除外/型変換)。予測寄与を持たない。",
}


def _validate() -> None:
    for name, e in TAXONOMY.items():
        assert e["kind"] in VALID_KINDS, f"{name}: bad kind {e['kind']}"
        assert set(e["models"]) <= VALID_MODELS, f"{name}: bad models {e['models']}"
        assert e["scale"] in VALID_SCALE, f"{name}: bad scale {e['scale']}"
        assert e["leak"] in VALID_LEAK, f"{name}: bad leak {e['leak']}"


_validate()


def get(name: str) -> dict | None:
    return TAXONOMY.get(name)


def group_by_kind() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, e in sorted(TAXONOMY.items()):
        out.setdefault(e["kind"], []).append(name)
    return out


def names_for_model(model: str) -> list[str]:
    return sorted(n for n, e in TAXONOMY.items()
                  if model in e["models"] or "all" in e["models"])


def check_coverage(registered: list[str]) -> dict[str, list[str]]:
    reg, tax = set(registered), set(TAXONOMY)
    return {"missing_in_taxonomy": sorted(reg - tax),
            "stale_in_taxonomy": sorted(tax - reg)}
