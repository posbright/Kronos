"""方案 C 步骤 3（dataC 编排版）：对齐 Kronos 特征 + dataC 因子 + 标签 → 融合宽表并按时间切分。

中文说明
    step2 产出的 DataSet/dataC/kronos_features.csv 是**多标的**特征表，其日期窗口
    （最近 N 个交易日）通常**横跨 dataC 的 validation 尾部与 test**。因此本脚本会把
    --sources 指定的若干 split（默认 validation+test）的 factors.csv / price.csv **纵向合并**，
    再调用 build_fusion_dataset.build_fusion / time_split 完成对齐、打标签、时间切分。

    与单文件版 build_fusion_dataset.py 的区别：
      - 自动从 dataC 目录结构读取并合并 val+test 的因子与价格（无需手工拼接）。
      - 自动只保留 kronos_features 覆盖到的标的（缩小规模、对齐干净）。
      - 切分边界可显式给定（--train-end/--val-end），否则按唯一交易日 70/15/15 自动切分。

输出
    DataSet/dataC/fusion_all.csv 及 fusion_train/val/test.csv（列结构完全一致）
    DataSet/dataC/fusion_report.json（行数 / 切分边界 / 标签列 / 因子列 / 标的）

防泄漏要点
    - 标签 label_fwd_ret_{H}d 按 symbol 分组用未来价计算（build_fusion 内实现）。
    - 时间切分严格按日期先后，禁止随机打乱；val/test 不与 train 在时间上重叠。
    - 因子合并 how="left"（以 kronos 行为基准），dataC 因子表本身 NaN-free。

用法
    python finetune_csv/build_dataC_step3_fusion.py \
        --data-root C:/xapproject/Quantia/Kronos/DataSet/dataC \
        --horizon 5

    # 显式切分边界（生产/全年数据推荐显式指定）
    python finetune_csv/build_dataC_step3_fusion.py \
        --train-end 2026-04-01 --val-end 2026-05-15
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from build_fusion_dataset import build_fusion, time_split  # noqa: E402


def _load_split_tables(data_root: Path, sources: List[str], symbols: set) -> tuple:
    """读取并纵向合并多个 split 的 factors.csv / price.csv，按 symbol 过滤去重。"""
    factor_frames, price_frames = [], []
    for s in sources:
        fdir = data_root / s
        fpath, ppath = fdir / "factors.csv", fdir / "price.csv"
        if not fpath.exists() or not ppath.exists():
            print(f"[step3][warn] 跳过缺失 split: {s}（{fpath} / {ppath}）")
            continue
        ff = pd.read_csv(fpath, dtype={"symbol": str})
        px = pd.read_csv(ppath, dtype={"symbol": str})
        if symbols:
            ff = ff[ff["symbol"].isin(symbols)]
            px = px[px["symbol"].isin(symbols)]
        factor_frames.append(ff)
        price_frames.append(px)
    if not factor_frames:
        raise RuntimeError(f"未从 {sources} 读到任何 factors/price，请检查 --data-root/--sources")

    factors = (pd.concat(factor_frames, ignore_index=True)
               .drop_duplicates(["date", "symbol"]).reset_index(drop=True))
    price = (pd.concat(price_frames, ignore_index=True)
             .drop_duplicates(["date", "symbol"]).reset_index(drop=True))
    return factors, price


def _auto_split_dates(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple:
    """按唯一交易日的分位自动确定 train_end / val_end（不含边界）。"""
    dates = pd.to_datetime(df["date"]).sort_values().unique()
    n = len(dates)
    i_tr = max(1, int(round(n * train_frac)))
    i_va = max(i_tr + 1, int(round(n * (train_frac + val_frac))))
    i_tr = min(i_tr, n - 2)
    i_va = min(i_va, n - 1)
    train_end = pd.Timestamp(dates[i_tr])
    val_end = pd.Timestamp(dates[i_va])
    return train_end, val_end


def main() -> None:
    ap = argparse.ArgumentParser(description="方案C step3：dataC 融合宽表 + 时间切分")
    ap.add_argument("--data-root", default="DataSet/dataC")
    ap.add_argument("--kronos", default=None, help="默认 {data-root}/kronos_features.csv")
    ap.add_argument("--sources", default="validation,test",
                    help="读取 factors/price 的 split，逗号分隔（默认 validation,test）")
    ap.add_argument("--out-dir", default=None, help="默认 {data-root}")
    ap.add_argument("--horizon", type=int, default=5, help="未来收益标签天数 H")
    ap.add_argument("--train-end", default=None, help="训练截止日(不含)，留空则自动分位切分")
    ap.add_argument("--val-end", default=None, help="验证截止日(不含)，留空则自动分位切分")
    ap.add_argument("--train-frac", type=float, default=0.70, help="自动切分训练占比")
    ap.add_argument("--val-frac", type=float, default=0.15, help="自动切分验证占比")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    kronos_path = Path(args.kronos) if args.kronos else data_root / "kronos_features.csv"
    out_dir = Path(args.out_dir) if args.out_dir else data_root
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    print(f"[step3] 读取 Kronos 特征: {kronos_path}")
    kf = pd.read_csv(kronos_path, dtype={"symbol": str})
    symbols = set(kf["symbol"].unique())
    print(f"[step3] kronos 行 {len(kf)}，标的 {len(symbols)}，"
          f"日期 {kf['date'].min()} -> {kf['date'].max()}")

    print(f"[step3] 合并 factors/price 来源: {sources}")
    factors, price = _load_split_tables(data_root, sources, symbols)
    print(f"[step3] 合并后 factors 行 {len(factors)}，price 行 {len(price)}")

    df = build_fusion(kf, factors, price, horizon=args.horizon)
    print(f"[step3] 融合后 {len(df)} 行（已打标签 label_fwd_ret_{args.horizon}d 并 dropna）")

    if args.train_end and args.val_end:
        train_end, val_end = pd.to_datetime(args.train_end), pd.to_datetime(args.val_end)
        split_mode = "manual"
    else:
        train_end, val_end = _auto_split_dates(df, args.train_frac, args.val_frac)
        split_mode = "auto"
    print(f"[step3] 切分边界({split_mode}): train < {train_end.date()} "
          f"<= val < {val_end.date()} <= test")

    tr, va, te = time_split(df, train_end=train_end, val_end=val_end)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "fusion_all.csv", index=False)
    tr.to_csv(out_dir / "fusion_train.csv", index=False)
    va.to_csv(out_dir / "fusion_val.csv", index=False)
    te.to_csv(out_dir / "fusion_test.csv", index=False)

    label_col = f"label_fwd_ret_{args.horizon}d"
    feat_cols = [c for c in df.columns if c not in ("date", "symbol", label_col)]
    report = {
        "kronos": str(kronos_path),
        "sources": sources,
        "horizon": args.horizon,
        "split_mode": split_mode,
        "train_end": str(train_end.date()),
        "val_end": str(val_end.date()),
        "rows": {"all": len(df), "train": len(tr), "val": len(va), "test": len(te)},
        "symbols": sorted(symbols),
        "label_col": label_col,
        "feature_cols": feat_cols,
        "date_range": [str(df["date"].min()), str(df["date"].max())],
    }
    with open(out_dir / "fusion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[step3] 融合 {len(df)} 行 -> train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    print(f"[step3] 已保存到 {out_dir}（fusion_all/train/val/test.csv + fusion_report.json）")


if __name__ == "__main__":
    main()
