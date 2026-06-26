"""方案 C 步骤 2：对齐 Kronos 特征 + 外部因子 + 标签 → 融合宽表，并按时间切分。

中文说明：
    输入三张表：
      - kronos_features：build_kronos_features.py 产出（date,symbol,k_pred_ret,k_up_prob,k_pred_vol）
      - factors        ：你的基本面 / 消息面因子（date,symbol,pe,pb,roe,news_sent,...）
      - price          ：原始日线（timestamps + close，用于算未来收益标签）
    输出：
      - fusion_all.csv 及按时间切分的 fusion_train / fusion_val / fusion_test
    标签：label_fwd_ret_{H}d = close.shift(-H)/close - 1（未来 H 日收益，禁止泄漏，按 symbol 分组算）。

用法：
    # 真实使用
    python build_fusion_dataset.py \
        --kronos data/kronos_features_000001.csv \
        --factors data/factors_000001.csv \
        --price finetune_csv/data/A_000001_daily.csv \
        --out-dir data --horizon 5 \
        --train-end 2024-01-01 --val-end 2025-01-01

    # 冒烟自测（无需任何外部文件，用合成三表跑通对齐 + 标签 + 切分）
    python build_fusion_dataset.py --smoke
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 慢变因子（前向填充）与新闻类因子（缺失填 0）的默认列名约定
_FFILL_FACTORS = ["pe", "pb", "roe", "north_hold"]
_ZERO_FACTORS = ["news_sent", "news_count", "event_flag"]


def build_fusion(kf: pd.DataFrame, ff: pd.DataFrame, px: pd.DataFrame,
                 horizon: int = 5) -> pd.DataFrame:
    """对齐三表并生成带未来收益标签的融合宽表。

    Args:
        kf: Kronos 特征表（含 date,symbol,k_*）。
        ff: 外部因子表（含 date,symbol,...）。
        px: 价格表（含 timestamps 或 date + close + 可选 symbol）。
        horizon: 未来收益标签的天数 H。
    """
    kf = kf.copy()
    ff = ff.copy()
    px = px.copy()
    kf["date"] = pd.to_datetime(kf["date"]).dt.normalize()
    ff["date"] = pd.to_datetime(ff["date"]).dt.normalize()

    if "date" not in px.columns:
        if "timestamps" not in px.columns:
            raise ValueError("price 表必须含 'date' 或 'timestamps' 列")
        px["date"] = pd.to_datetime(px["timestamps"]).dt.normalize()
    else:
        px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    if "close" not in px.columns:
        raise ValueError("price 表必须含 'close' 列以计算未来收益标签")

    label_col = f"label_fwd_ret_{horizon}d"
    # 标签：按 symbol 分组（若有），防止跨标的串期
    if "symbol" in px.columns:
        px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
        px[label_col] = (px.groupby("symbol")["close"].shift(-horizon)
                         / px["close"] - 1.0)
        label_keys = ["date", "symbol"]
    else:
        px = px.sort_values("date").reset_index(drop=True)
        px[label_col] = px["close"].shift(-horizon) / px["close"] - 1.0
        label_keys = ["date"]

    df = kf.merge(ff, on=["date", "symbol"], how="left")
    df = df.merge(px[label_keys + [label_col]], on=label_keys, how="left")

    # 缺失处理：慢变因子前向填充，新闻类填 0（仅对存在的列操作）
    #    有 symbol 列时按标的分组 ffill（防跨标的串值）；否则整体 ffill。
    has_symbol = "symbol" in df.columns
    for col in _FFILL_FACTORS:
        if col in df.columns:
            df[col] = (df.groupby("symbol")[col].ffill() if has_symbol
                       else df[col].ffill())
    for col in _ZERO_FACTORS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, train_end: str, val_end: str):
    """按时间先后切分（禁止随机打乱，防止跨期泄漏）。"""
    d = pd.to_datetime(df["date"])
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    tr = df[d < train_end]
    va = df[(d >= train_end) & (d < val_end)]
    te = df[d >= val_end]
    return tr, va, te


def _synth_tables(n: int = 120, seed: int = 0):
    """生成合成的三张表，仅用于冒烟自测。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-06-01", periods=n, freq="D")
    close = 10 + np.cumsum(rng.standard_normal(n) * 0.1)
    kf = pd.DataFrame({
        "date": dates, "symbol": "000001",
        "k_pred_ret": rng.standard_normal(n) * 0.01,
        "k_up_prob": rng.uniform(0, 1, n),
        "k_pred_vol": rng.uniform(0, 0.02, n),
    })
    ff = pd.DataFrame({
        "date": dates, "symbol": "000001",
        "pe": 15 + rng.standard_normal(n), "pb": 1.5 + rng.standard_normal(n) * 0.1,
        "roe": 0.1 + rng.standard_normal(n) * 0.01,
        "news_sent": rng.standard_normal(n), "news_count": rng.integers(0, 5, n),
    })
    px = pd.DataFrame({"timestamps": dates, "symbol": "000001", "close": close})
    return kf, ff, px


def _smoke_test() -> None:
    """无需外部文件的端到端冒烟测试。"""
    import tempfile

    kf, ff, px = _synth_tables(n=120)
    df = build_fusion(kf, ff, px, horizon=5)

    label_col = "label_fwd_ret_5d"
    assert label_col in df.columns, "缺少标签列"
    assert df[label_col].notnull().all(), "标签存在缺失（应已 dropna）"
    # 最后 5 行因 shift(-5) 无未来价 -> 被 dropna，剩 120-5=115
    assert len(df) == 120 - 5, f"行数异常: {len(df)}"
    for col in ["k_pred_ret", "k_up_prob", "k_pred_vol", "pe", "pb", "news_sent"]:
        assert col in df.columns, f"缺少列 {col}"

    tr, va, te = time_split(df, train_end="2023-07-15", val_end="2023-08-15")
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0, "切分后存在空集"
    assert len(tr) + len(va) + len(te) == len(df), "切分行数不守恒"
    # 时间不重叠校验
    assert tr["date"].max() < va["date"].min() < te["date"].min()

    with tempfile.TemporaryDirectory() as tmp:
        for name, part in [("train", tr), ("val", va), ("test", te)]:
            part.to_csv(os.path.join(tmp, f"fusion_{name}.csv"), index=False)
            assert os.path.exists(os.path.join(tmp, f"fusion_{name}.csv"))
    print(f"[smoke] build_fusion_dataset 通过：融合 {len(df)} 行，"
          f"切分 train/val/test = {len(tr)}/{len(va)}/{len(te)}，标签与时间切分均正常")
    print(df[["date", "symbol", "k_pred_ret", "pe", label_col]].head(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="方案 C：构建融合数据集并按时间切分")
    parser.add_argument("--kronos", help="Kronos 特征 CSV（build_kronos_features 产出）")
    parser.add_argument("--factors", help="外部因子 CSV（date,symbol,...）")
    parser.add_argument("--price", help="价格 CSV（timestamps/date + close）")
    parser.add_argument("--out-dir", help="输出目录")
    parser.add_argument("--horizon", type=int, default=5, help="未来收益标签天数 H")
    parser.add_argument("--train-end", default="2024-01-01", help="训练集截止日（不含）")
    parser.add_argument("--val-end", default="2025-01-01", help="验证集截止日（不含）")
    parser.add_argument("--smoke", action="store_true", help="运行无需文件的冒烟自测")
    args = parser.parse_args()

    if args.smoke:
        _smoke_test()
        return

    required = [args.kronos, args.factors, args.price, args.out_dir]
    if any(v is None for v in required):
        parser.error("非 --smoke 模式下必须提供 --kronos --factors --price --out-dir")

    kf = pd.read_csv(args.kronos, dtype={"symbol": str})
    ff = pd.read_csv(args.factors, dtype={"symbol": str})
    px = pd.read_csv(args.price, dtype={"symbol": str})
    df = build_fusion(kf, ff, px, horizon=args.horizon)
    tr, va, te = time_split(df, train_end=args.train_end, val_end=args.val_end)

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "fusion_all.csv"), index=False)
    tr.to_csv(os.path.join(args.out_dir, "fusion_train.csv"), index=False)
    va.to_csv(os.path.join(args.out_dir, "fusion_val.csv"), index=False)
    te.to_csv(os.path.join(args.out_dir, "fusion_test.csv"), index=False)
    print(f"[build_fusion_dataset] 融合 {len(df)} 行 -> "
          f"train/val/test = {len(tr)}/{len(va)}/{len(te)}，已保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
