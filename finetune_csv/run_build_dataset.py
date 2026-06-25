"""构建「20 只股票小范围验证」数据集（统一配置 + 滚动切分 + 因子缺失处理 + 过程日志）。

流程：
    1. 读取统一配置（configs/config_smoke20.yaml）。
    2. 选股票池：配置显式列表，或自动从 Quantia 缓存按字典序挑选 max_symbols 只（需 >= min_history）。
    3. 缓存优先读取日线（DB 兜底），合并 DB 技术指标 / 基本面因子。
    4. 以「今天-1」为锚点做滚动日期切分（test/val/train 向过去回推）。
    5. 对因子列做防泄漏缺失填充（ffill + 训练区间中位数兜底 + 缺失掩码）。
    6. 写出 DataSet/{train,validation,test}/dataset.csv，并在 runs/<exp>/<ver>/build 下落盘
       run.log、split_plan.json、missing_report.json、build_report.json。

用法：
    # 真实构建（缓存 + DB 均可达）
    python finetune_csv/run_build_dataset.py --config finetune_csv/configs/config_smoke20.yaml

    # 冒烟自测（合成数据，无需缓存 / DB）
    python finetune_csv/run_build_dataset.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import (  # noqa: E402
    PipelineConfig,
    rolling_date_splits,
    collect_trading_dates,
    analyze_missing,
    handle_factor_nulls,
    compute_tech_indicators,
    setup_run_logger,
)
import build_full_market_dataset as bfm  # noqa: E402


# 不参与缺失填充的「原始/标识」列；其余数值列均按因子处理。
_KEEP_RAW = {
    "date", "future_date", "symbol", "split", "kline_source", "cache_path",
    "open", "high", "low", "close", "volume", "amount",
    "amplitude", "quote_change", "ups_downs", "turnover",
    "future_close",
}


def _detect_factor_cols(df: pd.DataFrame, label_col: str) -> List[str]:
    """识别需要做缺失处理的因子列：数值列中排除原始价量列与标签列。"""
    cols = []
    for c in df.columns:
        if c in _KEEP_RAW or c == label_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _select_universe(cfg: PipelineConfig, logger) -> List[str]:
    """选股票池：优先配置显式列表，否则自动从缓存挑选。"""
    if cfg.universe_symbols:
        logger.info("使用配置显式股票池：%d 只", len(cfg.universe_symbols))
        return cfg.universe_symbols[: cfg.max_symbols]

    cache_syms = bfm.scan_cache_symbols(cfg.cache_hist_root)
    logger.info("缓存可用标的总数：%d，按 min_history=%d 过滤后取前 %d 只",
                len(cache_syms), cfg.min_history, cfg.max_symbols)
    return cache_syms  # 实际过滤在加载时按 min_history 进行


def build(cfg: PipelineConfig) -> dict:
    run_dir = cfg.run_dir("build")
    logger = setup_run_logger(str(run_dir))
    logger.info("=== 构建 20 只股票小范围验证数据集 ===")
    logger.info("配置文件：%s", cfg.config_path)
    logger.info("缓存目录：%s", cfg.cache_hist_root)

    if not cfg.cache_hist_root.exists():
        raise FileNotFoundError(f"缓存目录不存在：{cfg.cache_hist_root}")

    provider = bfm.QuantiaDataProvider(
        quantia_root=cfg.quantia_root,
        db_sleep=cfg.db_sleep,
        db_retries=cfg.db_retries,
    )

    candidates = _select_universe(cfg, logger)

    # 第一遍：加载日线、按 min_history 过滤，凑满 max_symbols。
    sym_klines = {}
    for sym in candidates:
        if len(sym_klines) >= cfg.max_symbols:
            break
        kdf, src = bfm.load_kline_from_cache(sym, cfg.cache_hist_root)
        source = "cache"
        if kdf.empty and provider.available:
            kdf = provider.fetch_kline_from_spot(sym, cfg.start_date, cfg.end_date)
            source = "db_spot"
        if kdf.empty:
            continue
        kdf = kdf[(kdf["date"] >= pd.Timestamp(cfg.start_date)) &
                  (kdf["date"] <= pd.Timestamp(cfg.end_date))].copy()
        if len(kdf) < cfg.min_history:
            continue
        sym_klines[sym] = (kdf, source)
        logger.info("选入 %s：%d 行（%s ~ %s，来源=%s）", sym, len(kdf),
                    kdf["date"].min().date(), kdf["date"].max().date(), source)

    if not sym_klines:
        raise RuntimeError("没有符合条件的标的，请检查缓存路径 / min_history / 日期范围")

    # 滚动切分：用全池交易日并集，锚点=今天-1。
    plan = rolling_date_splits(
        trading_dates=collect_trading_dates([k["date"] for k, _ in sym_klines.values()]),
        test_days=cfg.test_days,
        val_days=cfg.val_days,
        train_days=cfg.train_days,
        anchor=pd.Timestamp(cfg.anchor) if cfg.anchor else None,
        label_horizon=cfg.label_horizon,
    )
    logger.info("滚动切分方案：%s", plan.summary())
    (run_dir / "split_plan.json").write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # 第二遍：逐标的合并因子并打 split 标签。
    merged_all = []
    label_col = f"label_fwd_ret_{cfg.label_horizon}d"
    n_with_tech = n_with_fin = 0
    for sym, (kdf, source) in sym_klines.items():
        tech_df = pd.DataFrame()
        fin_df = pd.DataFrame()
        # 技术指标：默认本地用 OHLCV 现算（历史完整）；可配置为远程库。
        if cfg.tech_source == "local":
            tech_df = compute_tech_indicators(kdf)
            if not tech_df.empty:
                n_with_tech += 1
        elif provider.available and cfg.use_db_features:
            tech_df = provider.fetch_indicators(sym, cfg.start_date, cfg.end_date)
            if not tech_df.empty:
                n_with_tech += 1

        if provider.available and cfg.use_db_features:
            fin_df = provider.fetch_financial(sym, cfg.end_date)
            if not fin_df.empty:
                n_with_fin += 1

        merged = bfm.merge_symbol_features(
            symbol=sym, kline_df=kdf, tech_df=tech_df, fin_df=fin_df,
            horizon=cfg.label_horizon, train_end=plan.train_end, val_end=plan.val_end,
        )
        if merged.empty:
            continue
        merged["kline_source"] = source
        merged_all.append(merged)

    if not merged_all:
        raise RuntimeError("合并后无任何样本，请检查切分边界与标的历史长度")

    full = pd.concat(merged_all, ignore_index=True)

    # 因子缺失处理（防泄漏）：训练区间统计兜底，按 symbol 分组 ffill。
    factor_cols = _detect_factor_cols(full, label_col)
    missing_before = analyze_missing(full, factor_cols)
    train_mask = full["split"] == "train"
    full = handle_factor_nulls(
        full, factor_cols,
        strategy_map=cfg.factor_strategy or None,
        group_col="symbol",
        train_mask=train_mask,
        add_mask=cfg.factor_add_mask,
        mask_threshold=cfg.factor_mask_threshold,
    )
    missing_report = {
        "factor_count": len(factor_cols),
        "missing_rate_before_fill": {k: round(v, 4) for k, v in missing_before.items()},
        "added_mask_columns": [c for c in full.columns if c.endswith("_isna")],
    }
    (run_dir / "missing_report.json").write_text(
        json.dumps(missing_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("因子列 %d 个，缺失掩码列 %d 个",
                len(factor_cols), len(missing_report["added_mask_columns"]))

    # 写出三份数据集。
    out_root = cfg.dataset_root
    split_paths = bfm.ensure_out_layout(out_root)
    split_stats = {}
    for split in ("train", "validation", "test"):
        part = full[full["split"] == split].copy()
        # 覆盖写（避免与历史追加混淆）。
        if part.empty:
            pd.DataFrame().to_csv(split_paths[split], index=False)
        else:
            part.to_csv(split_paths[split], index=False)
        split_stats[split] = {
            "rows": int(len(part)),
            "symbols": int(part["symbol"].nunique()) if not part.empty else 0,
            "min_date": (part["date"].min() if not part.empty else None),
            "max_date": (part["date"].max() if not part.empty else None),
            "file": str(split_paths[split]),
        }
        logger.info("写出 %s：%d 行，%d 只标的 -> %s",
                    split, split_stats[split]["rows"], split_stats[split]["symbols"],
                    split_paths[split])

    report = {
        "exp_name": cfg.exp_name,
        "version": cfg.version,
        "universe_size": len(sym_klines),
        "symbols": sorted(sym_klines.keys()),
        "split_plan": plan.to_dict(),
        "label_col": label_col,
        "symbols_with_tech": n_with_tech,
        "symbols_with_fin": n_with_fin,
        "factor_cols": factor_cols,
        "splits": split_stats,
        "dataset_root": str(out_root),
        "run_dir": str(run_dir),
    }
    (run_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("构建完成，报告：%s", run_dir / "build_report.json")
    return report


def run_smoke() -> None:
    """合成数据冒烟自测：不依赖缓存 / DB，验证切分 + 缺失处理 + 写盘全链路。"""
    import tempfile

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=300)  # 300 个工作日

    def make_symbol(sym: str, shift: float) -> pd.DataFrame:
        base = 10.0 + np.cumsum(rng.standard_normal(len(dates)) * 0.2 + shift)
        df = pd.DataFrame({
            "date": dates,
            "open": base * 0.995, "high": base * 1.01, "low": base * 0.99,
            "close": base,
            "volume": rng.integers(2e5, 1e6, len(dates)).astype(float),
            "amount": rng.integers(1e6, 1e7, len(dates)).astype(float),
            "amplitude": rng.random(len(dates)) * 8,
            "quote_change": rng.standard_normal(len(dates)),
            "ups_downs": rng.standard_normal(len(dates)) * 0.1,
            "turnover": rng.random(len(dates)) * 5,
        })
        return df

    # 技术因子带零散缺失；基本面按季度（大段缺失）。
    tech = pd.DataFrame({
        "date": dates, "code": "000001",
        "tech_macd": rng.standard_normal(len(dates)),
        "tech_rsi": rng.random(len(dates)) * 100,
    })
    tech.loc[tech.sample(frac=0.1, random_state=1).index, "tech_macd"] = np.nan
    fin_dates = pd.to_datetime(["2022-12-31", "2023-03-31", "2023-06-30",
                                "2023-09-30", "2023-12-31", "2024-03-31"])
    fin = pd.DataFrame({
        "date": fin_dates, "code": "000001",
        "fin_roe": [8.1, 8.3, 8.7, 9.0, 9.2, 9.4],
    })

    horizon = 5
    anchor = dates[-1]
    plan = rolling_date_splits(
        trading_dates=collect_trading_dates([dates.to_series()]),
        test_days=20, val_days=20, train_days=None, anchor=anchor, label_horizon=horizon,
    )
    assert plan.n_test == 20 and plan.n_val == 20 and plan.n_train > 0, "切分异常"
    assert plan.train_end < plan.val_end < plan.test_end, "边界顺序异常"

    s1 = bfm.merge_symbol_features("000001", make_symbol("000001", 0.0), tech, fin,
                                   horizon, plan.train_end, plan.val_end)
    s2 = bfm.merge_symbol_features("000002", make_symbol("000002", 0.02),
                                   pd.DataFrame(), pd.DataFrame(),
                                   horizon, plan.train_end, plan.val_end)
    full = pd.concat([s1, s2], ignore_index=True)
    label_col = f"label_fwd_ret_{horizon}d"

    factor_cols = _detect_factor_cols(full, label_col)
    miss_before = analyze_missing(full, factor_cols)
    assert any(v > 0 for v in miss_before.values()), "冒烟数据应当包含缺失因子"

    filled = handle_factor_nulls(full, factor_cols, group_col="symbol",
                                 train_mask=(full["split"] == "train"), add_mask=True)
    assert not filled[factor_cols].isna().any().any(), "缺失填充后仍有 NaN"
    assert any(c.endswith("_isna") for c in filled.columns), "应追加缺失掩码列"

    with tempfile.TemporaryDirectory() as tmp:
        for split in ("train", "validation", "test"):
            part = filled[filled["split"] == split]
            assert len(part) > 0, f"{split} 为空"
            part.to_csv(os.path.join(tmp, f"{split}.csv"), index=False)

    print("[smoke] run_build_dataset 通过：")
    print("  切分：", plan.summary())
    print(f"  样本 train/val/test = "
          f"{(filled['split']=='train').sum()}/"
          f"{(filled['split']=='validation').sum()}/"
          f"{(filled['split']=='test').sum()}")
    print(f"  因子列 {len(factor_cols)} 个，掩码列 "
          f"{sum(c.endswith('_isna') for c in filled.columns)} 个")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 20 只股票小范围验证数据集")
    parser.add_argument("--config", default=str(_THIS_DIR / "configs" / "config_smoke20.yaml"),
                        help="统一配置 YAML 路径")
    parser.add_argument("--smoke", action="store_true", help="合成数据冒烟自测，不依赖缓存/DB")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    cfg = PipelineConfig(args.config)
    report = build(cfg)
    print(json.dumps({k: report[k] for k in ("exp_name", "version", "universe_size", "splits")},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
