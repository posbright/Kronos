"""因子版本的预测对比可视化 generate_factor_comparison（Phase 8）。

为「某个已训练的因子版本」生成 K 线预测对比图：在测试集某只股票上同时跑
    - 基线 Kronos（KronosPredictor，仅 OHLCV）          -> Kronos 原始输出
    - 因子增强 Kronos（FactorPredictor，注入因子条件）    -> 调整后预测
并叠加真实未来 K 线，输出 runs/<exp>/<version>/viz/kline_comparison.png。

与 run_visualize 的差异：因子 App 的版本目录下只有 train/factor/，tokenizer 与基座
主模型来自「底座版本」或「配置预训练」，故底座路径需显式传入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_PRICE_COLS = ["open", "high", "low", "close"]


def _select_window(df: pd.DataFrame, lookback: int, pred_len: int,
                   symbol: Optional[str]) -> Optional[pd.DataFrame]:
    """取某只股票最后 lookback+pred_len 根连续 K 线；样本不足返回 None。"""
    if "symbol" in df.columns:
        sym = symbol or df["symbol"].iloc[-1]
        df = df[df["symbol"] == sym].copy()
    need = lookback + pred_len
    if len(df) < need:
        return None
    tcol = "timestamps" if "timestamps" in df.columns else "date"
    df = df.sort_values(tcol).tail(need).reset_index(drop=True)
    df.index = pd.to_datetime(df[tcol])
    return df


def generate_factor_comparison(
        cfg, version: str, tok_src: str, pred_src: str,
        factor_model_dir: str, specs: Sequence[dict],
        symbol: Optional[str] = None) -> Optional[Dict[str, object]]:
    """生成「Kronos 原始 vs 因子调整」预测对比图。失败/数据不足时返回 None。

    Args:
        cfg:              PipelineConfig。
        version:          因子版本号（决定输出 viz 目录）。
        tok_src:          tokenizer 来源（底座版本 best_model 或预训练名）。
        pred_src:         基座主模型来源（Kronos 主模型）。
        factor_model_dir: 该版本因子模型 best_model 目录。
        specs:            规范因子通道规格（含 mean/raw 聚合，用于构造测试窗因子）。
        symbol:           指定股票；None 取测试集最后一只。

    Returns:
        {"kline": png_path, "symbol": ..., "expected_return_pct": ...} 或 None。
    """
    from model import Kronos, KronosTokenizer, KronosPredictor
    from finetune_csv.factor_model import load_factor_model
    from .factor_predictor import FactorPredictor
    from .factor_dataset import build_factor_matrix
    from viz import plot_kline_comparison

    lookback, pred_len = cfg.lookback_window, cfg.predict_window
    test_csv = Path(cfg.dataset_root) / "test" / "dataset.csv"
    if not test_csv.exists():
        return None
    df = pd.read_csv(test_csv)
    window = _select_window(df, lookback, pred_len, symbol)
    if window is None:
        return None
    history = window.iloc[:lookback]
    actual = window.iloc[lookback:]

    tokenizer = KronosTokenizer.from_pretrained(tok_src)
    base_model = Kronos.from_pretrained(pred_src)
    base_pred = KronosPredictor(base_model, tokenizer, device=None,
                                max_context=cfg.max_context)

    has_vol = "volume" in history.columns
    price_in = history[_PRICE_COLS + ["volume", "amount"]] if has_vol else history[_PRICE_COLS]
    x_ts = pd.Series(history.index)
    y_ts = pd.Series(actual.index)

    base_df = base_pred.predict(price_in, x_ts, y_ts, pred_len, verbose=False)
    base_df.index = actual.index
    pred_dict = {"基线 Kronos(原始)": base_df}

    # 因子增强预测：用 specs 在窗口上构造因子通道（含同类均值聚合）。
    factor_dim = len(specs)
    fac_model = load_factor_model(factor_model_dir, factor_dim)
    fac_pred = FactorPredictor(fac_model, tokenizer, device=None,
                               max_context=cfg.max_context)
    factor_full = build_factor_matrix(window.reset_index(drop=True), specs)  # [W,k]
    # 乘以每通道用户权重（与训练一致）。
    w = np.array([sp.get("weight", 1.0) for sp in specs], dtype=np.float32)
    factor_full = factor_full * w[np.newaxis, :]
    factor_hist = factor_full[:lookback]

    fac_df = fac_pred.predict(price_in, x_ts, y_ts, pred_len, factor_hist=factor_hist)
    fac_df.index = actual.index
    pred_dict["因子调整 Kronos"] = fac_df

    # 预期收益对比（末日收盘相对最后历史收盘）。
    last_close = float(history["close"].iloc[-1])
    base_ret = float(base_df["close"].iloc[-1] / last_close - 1.0)
    fac_ret = float(fac_df["close"].iloc[-1] / last_close - 1.0)
    act_ret = float(actual["close"].iloc[-1] / last_close - 1.0)

    sym_label = symbol or (window["symbol"].iloc[0] if "symbol" in window.columns else "")
    out_dir = cfg.runs_root / cfg.exp_name / version / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    kline_png = str(out_dir / "kline_comparison.png")
    title = (f"{sym_label} 预测对比 | 原始 {base_ret*100:+.2f}% · "
             f"因子调整 {fac_ret*100:+.2f}% · 实际 {act_ret*100:+.2f}%")
    plot_kline_comparison(history[_PRICE_COLS], actual[_PRICE_COLS], pred_dict,
                          kline_png, title=title)

    return {
        "kline": kline_png,
        "symbol": sym_label,
        "base_return_pct": round(base_ret * 100, 3),
        "factor_return_pct": round(fac_ret * 100, 3),
        "actual_return_pct": round(act_ret * 100, 3),
    }
