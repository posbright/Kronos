"""因子分析引擎：基于 DataSet 真实数据计算每个因子的多空方向、评分、贡献、

中文说明：
    本模块对训练集做「快速统计画像」，为前端提供可解释的因子洞察（无需训练模型）：
      1. 单因子 IC（rank IC，Spearman 秩相关）：因子值与未来收益的相关性。
         - direction（多空方向）= IC 符号：+1 越高越涨(偏多)，-1 越高越跌(偏空)
         - score（强度评分 0~100）= |IC| 相对归一
         - confidence（置信度 0~1）= 由 t 统计量映射（样本越多、相关越强越可信）
         - contribution（贡献占比）= 在所选因子中 |IC|×用户权重 的归一份额
      2. 同类冗余：同一大类内因子两两相关，|corr| 高于阈值的对视为冗余。
      3. 类别聚合：把同类、方向对齐后的 z 分数取均值 → 类别综合信号及其 IC。
      4. 综合画像：IC 加权的预期收益(粗略)、综合评分、置信度、信号分歧、日波动百分比。

    注意：IC 仅为线性/秩相关的统计画像，不等于模型贡献；真正的模型贡献以训练后
    factor_importance（factor_emb 各列 L2）为准。两者结合可交叉验证。

用法：
    from finetune_csv.app.factor_analysis import analyze_factors
    res = analyze_factors(cfg, factor_cols=[...], weights={...})

    python -m finetune_csv.app.factor_analysis --smoke
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 允许以脚本 / 模块两种方式运行
try:
    from .factor_meta import get_meta, CATEGORY_INFO
except ImportError:  # 直接 python factor_analysis.py
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from factor_meta import get_meta, CATEGORY_INFO  # type: ignore

# 候选未来收益标签（按优先级），均为「未来 H 日收益」，禁止使用同期/过去列防泄漏。
_TARGET_CANDIDATES = ["label_fwd_ret_5d", "label_fwd_ret_1d", "ret_5d"]
_MIN_IC = 0.005          # |IC| 低于该值视为方向中性
_REDUNDANT_CORR = 0.85   # 同类两两 |corr| 超过该值视为冗余


def _pick_target(df: pd.DataFrame) -> Optional[str]:
    for c in _TARGET_CANDIDATES:
        if c in df.columns and df[c].notna().sum() > 10:
            return c
    # 回退：用 close 与 future_close 现算 5 日收益
    if "future_close" in df.columns and "close" in df.columns:
        df["_fwd_ret"] = df["future_close"] / df["close"] - 1.0
        if df["_fwd_ret"].notna().sum() > 10:
            return "_fwd_ret"
    return None


def _t_to_confidence(ic: float, n: int) -> float:
    """由 IC 与样本量算 t 统计量并映射到置信度 0~1（|t|>=3 视为高置信）。"""
    if n < 5 or abs(ic) >= 1.0:
        return 0.0
    t = abs(ic) * math.sqrt((n - 2) / max(1e-9, 1.0 - ic * ic))
    return float(min(1.0, t / 3.0))


def _rank_ic(factor: pd.Series, target: pd.Series) -> tuple[float, int]:
    """计算秩相关 IC（Spearman），返回 (ic, 有效样本数)。"""
    m = factor.notna() & target.notna()
    n = int(m.sum())
    if n < 5:
        return 0.0, n
    f, t = factor[m], target[m]
    if f.nunique() < 2 or t.nunique() < 2:
        return 0.0, n
    # 秩相关 = 对秩取 Pearson（避免依赖 scipy 的 method='spearman'）
    ic = f.rank().corr(t.rank())
    return (0.0 if pd.isna(ic) else float(ic)), n


def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if not sd or pd.isna(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def analyze_factors(cfg, factor_cols: Optional[List[str]] = None,
                    weights: Optional[Dict[str, float]] = None,
                    max_rows: int = 200_000) -> Dict[str, object]:
    """对训练集做因子统计画像。

    Args:
        cfg:         PipelineConfig（用于定位 DataSet/train/dataset.csv）。
        factor_cols: 参与分析的因子列；None 时分析全部 tech_/fin_ 列。
        weights:     每因子用户权重（用于贡献占比与综合信号加权）；缺省按 1。
        max_rows:    读取行数上限（大表抽样以保证响应速度）。
    """
    train_csv = Path(cfg.dataset_root) / "train" / "dataset.csv"
    if not train_csv.exists():
        return {"error": f"未找到训练集 {train_csv}"}

    header = pd.read_csv(train_csv, nrows=0).columns.tolist()
    if factor_cols is None:
        factor_cols = [c for c in header
                       if (c.startswith("tech_") or c.startswith("fin_"))
                       and not c.endswith("_isna")]
    # 只读需要的列 + 目标候选 + close/future_close + symbol
    need = set(factor_cols)
    need.update([c for c in _TARGET_CANDIDATES if c in header])
    for c in ("close", "future_close", "symbol", "date"):
        if c in header:
            need.add(c)
    usecols = [c for c in header if c in need]

    df = pd.read_csv(train_csv, usecols=usecols)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    target = _pick_target(df)
    if target is None:
        return {"error": "数据集缺少可用的未来收益标签列"}

    weights = weights or {}
    tgt = df[target]

    # ---- 1) 单因子 IC / 方向 / 评分 / 置信度 ----
    fac_out: Dict[str, Dict[str, object]] = {}
    max_abs_ic = 1e-9
    for col in factor_cols:
        if col not in df.columns:
            continue
        ic, n = _rank_ic(df[col], tgt)
        max_abs_ic = max(max_abs_ic, abs(ic))
    for col in factor_cols:
        if col not in df.columns:
            continue
        meta = get_meta(col) or {}
        ic, n = _rank_ic(df[col], tgt)
        direction = 0 if abs(ic) < _MIN_IC else (1 if ic > 0 else -1)
        coverage = float(df[col].notna().mean())
        w = float(weights.get(col, 1.0))
        fac_out[col] = {
            "name_cn": meta.get("name_cn", col),
            "category": meta.get("category", "other"),
            "category_cn": meta.get("category_cn", "其他"),
            "side": meta.get("side", "other"),
            "bias": meta.get("bias", 0),
            "desc_cn": meta.get("desc_cn", ""),
            "ic": round(ic, 4),
            "abs_ic": round(abs(ic), 4),
            "direction": direction,
            "score": round(100.0 * abs(ic) / max_abs_ic, 1),
            "confidence": round(_t_to_confidence(ic, n), 3),
            "coverage": round(coverage, 3),
            "weight": w,
            "n": n,
        }

    # 贡献占比：|IC|×权重 归一
    contrib_raw = {c: d["abs_ic"] * d["weight"] for c, d in fac_out.items()}
    contrib_sum = sum(contrib_raw.values()) or 1e-9
    for c, d in fac_out.items():
        d["contribution"] = round(contrib_raw[c] / contrib_sum, 4)

    # ---- 2) 类别聚合 + 同类冗余 ----
    cat_members: Dict[str, List[str]] = {}
    for c, d in fac_out.items():
        cat_members.setdefault(str(d["category"]), []).append(c)

    cat_out: Dict[str, Dict[str, object]] = {}
    # 方向对齐后的类别 z 合成信号（供综合信号与预期收益使用）
    cat_signal: Dict[str, pd.Series] = {}
    for cat, members in cat_members.items():
        # 方向对齐的 z 分数：z × sign(IC)（IC=0 的因子不参与，避免噪声）
        aligned = []
        for c in members:
            d = fac_out[c]
            if d["direction"] == 0:
                continue
            aligned.append(_zscore(df[c].fillna(df[c].median())) * d["direction"])
        if aligned:
            sig = pd.concat(aligned, axis=1).mean(axis=1)
        else:
            sig = pd.Series(np.zeros(len(df)), index=df.index)
        cat_signal[cat] = sig
        agg_ic, n = _rank_ic(sig, tgt)

        # 同类两两冗余
        red_pairs = []
        valid = [c for c in members if df[c].notna().sum() > 5]
        if len(valid) >= 2:
            corr = df[valid].rank().corr().abs()
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    cv = corr.iloc[i, j]
                    if pd.notna(cv) and cv >= _REDUNDANT_CORR:
                        red_pairs.append({"a": valid[i], "b": valid[j],
                                          "corr": round(float(cv), 3)})
            mean_abs_corr = float(corr.values[np.triu_indices(len(valid), 1)].mean())
        else:
            mean_abs_corr = 0.0

        info = CATEGORY_INFO.get(cat, {})
        cat_out[cat] = {
            "name_cn": info.get("name_cn", cat),
            "side": info.get("side", "other"),
            "members": members,
            "agg_ic": round(agg_ic, 4),
            "direction": 0 if abs(agg_ic) < _MIN_IC else (1 if agg_ic > 0 else -1),
            "score": round(100.0 * abs(agg_ic) / max_abs_ic, 1),
            "confidence": round(_t_to_confidence(agg_ic, n), 3),
            "redundancy": red_pairs,
            "mean_abs_corr": round(mean_abs_corr, 3),
        }

    # ---- 3) 综合画像：预期收益 / 综合评分 / 置信度 / 分歧 ----
    tgt_std = float(tgt.std() or 0.0)
    # 取最新快照：每个 symbol 的最后一行（无 symbol 则用全表末行）
    if "symbol" in df.columns and "date" in df.columns:
        snap = (df.sort_values("date").groupby("symbol").tail(1))
    else:
        snap = df.tail(max(1, min(50, len(df))))

    # 每个因子在快照上的方向票 = sign(IC) × sign(z)，按 |IC|×权重 加权
    votes, vote_w, fwd_terms = [], [], []
    for c, d in fac_out.items():
        if d["direction"] == 0 or c not in snap.columns:
            continue
        z = _zscore(df[c].fillna(df[c].median()))
        z_snap = float(z.loc[snap.index].mean())
        w = d["abs_ic"] * d["weight"]
        votes.append(np.sign(d["direction"] * z_snap))
        vote_w.append(w)
        # 线性因子预测：IC × z（std 单位）→ ×目标 std → 收益单位
        fwd_terms.append(d["ic"] * z_snap * d["weight"])

    if vote_w:
        vote_w_arr = np.array(vote_w)
        net_vote = float(np.sum(np.array(votes) * vote_w_arr) / (vote_w_arr.sum() or 1e-9))
        divergence = round(1.0 - abs(net_vote), 3)         # 0=高度一致，1=完全分歧
        expected_ret = float(np.mean(fwd_terms)) * tgt_std  # 粗略预期 H 日收益
        comp_dir = 1 if net_vote > 0 else (-1 if net_vote < 0 else 0)
        comp_score = round(100.0 * abs(net_vote), 1)
        comp_conf = round(float(np.mean([fac_out[c]["confidence"] for c in fac_out
                                         if fac_out[c]["direction"] != 0]) or 0.0), 3)
    else:
        net_vote = 0.0
        divergence = 1.0
        expected_ret = 0.0
        comp_dir = 0
        comp_score = 0.0
        comp_conf = 0.0

    # 日波动百分比：close 日收益绝对值均值 ×100（按 symbol 防跨标的）
    daily_vol_pct = 0.0
    if "close" in df.columns:
        if "symbol" in df.columns and "date" in df.columns:
            d2 = df.sort_values(["symbol", "date"])
            pct = d2.groupby("symbol")["close"].pct_change().abs()
        else:
            pct = df["close"].pct_change().abs()
        daily_vol_pct = round(float(pct.mean() * 100.0), 3) if pct.notna().any() else 0.0

    return {
        "target": target,
        "n_rows": int(len(df)),
        "daily_volatility_pct": daily_vol_pct,
        "factors": fac_out,
        "categories": cat_out,
        "composite": {
            "direction": comp_dir,
            "score": comp_score,
            "confidence": comp_conf,
            "divergence": divergence,
            "expected_return": round(expected_ret, 5),
            "expected_return_pct": round(expected_ret * 100.0, 3),
            "target": target,
        },
    }


# ----------------------------- 冒烟自测 -----------------------------
class _StubCfg:
    def __init__(self, root: Path):
        self.dataset_root = root


def _smoke() -> None:
    import tempfile
    rng = np.random.default_rng(0)
    n_sym, n_days = 6, 120
    rows = []
    for s in range(n_sym):
        base = 10.0
        for t in range(n_days):
            base *= (1 + rng.normal(0, 0.01))
            rows.append({"symbol": f"{s:06d}", "date": t, "close": base})
    df = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    # 未来 5 日收益标签
    df["label_fwd_ret_5d"] = (df.groupby("symbol")["close"].shift(-5) / df["close"] - 1.0)
    # 构造一个「真有预测力」的因子（与未来收益正相关）+ 一个纯噪声因子 + 一个负向因子
    df["tech_roc"] = df["label_fwd_ret_5d"].fillna(0) * 50 + rng.normal(0, 0.3, len(df))
    df["tech_rsi"] = rng.normal(0, 1, len(df))                       # 噪声
    df["fin_asset_liability_ratio"] = -df["label_fwd_ret_5d"].fillna(0) * 40 + rng.normal(0, 0.4, len(df))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train").mkdir(parents=True)
        df.to_csv(root / "train" / "dataset.csv", index=False)
        cfg = _StubCfg(root)
        res = analyze_factors(cfg, factor_cols=["tech_roc", "tech_rsi",
                                                "fin_asset_liability_ratio"],
                              weights={"tech_roc": 1.5, "tech_rsi": 1.0,
                                       "fin_asset_liability_ratio": 1.0})
        assert "error" not in res, res
        f = res["factors"]
        # 正向因子方向应为 +1，负向因子 -1，强度高于噪声
        assert f["tech_roc"]["direction"] == 1, f["tech_roc"]
        assert f["fin_asset_liability_ratio"]["direction"] == -1, f["fin_asset_liability_ratio"]
        assert f["tech_roc"]["score"] > f["tech_rsi"]["score"], "有效因子评分应高于噪声"
        assert 0.0 <= res["composite"]["divergence"] <= 1.0
        assert res["daily_volatility_pct"] > 0
        print("[smoke] factor_analysis 通过：",
              {k: (v["direction"], v["score"], v["confidence"]) for k, v in f.items()})
        print("        composite=", res["composite"], "vol%=", res["daily_volatility_pct"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="因子统计分析引擎")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        ap.print_help()
