"""日 K 线预测对比绘图。

把「历史 K 线 + 真实未来 K 线 + 一个或多个模型的预测」画在同一张图上，直观对比
基线模型与因子增强模型的预测效果。纯 matplotlib 实现（不依赖 mplfinance）。

约定输入：
    history_df : 历史窗口，索引为时间，含 open/high/low/close。
    actual_df  : 真实未来窗口（含 OHLC）；可为 None（仅看预测）。
    pred_dict  : {模型标签: 预测 DataFrame(含 OHLC，至少 close)}，多个模型叠加对比。

输出：保存 PNG 到 out_path，返回该路径。
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # 无显示环境下后端
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ._fonts import setup_cjk_fonts  # noqa: E402

setup_cjk_fonts()

# 预测线条配色（按加入顺序循环取用）。
_PRED_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _draw_candles(ax, idx, o, h, l, c, width=0.6,
                  up_color="#26a69a", down_color="#ef5350", alpha=1.0,
                  label: Optional[str] = None):
    """在 ax 上按整数横坐标 idx 画一组蜡烛（影线 + 实体）。"""
    for k, (xi, oo, hh, ll, cc) in enumerate(zip(idx, o, h, l, c)):
        color = up_color if cc >= oo else down_color
        ax.vlines(xi, ll, hh, color=color, linewidth=1.0, alpha=alpha)
        lo, hi = (oo, cc) if cc >= oo else (cc, oo)
        ax.add_patch(plt.Rectangle((xi - width / 2, lo), width, max(hi - lo, 1e-9),
                                   facecolor=color, edgecolor=color, alpha=alpha,
                                   label=label if k == 0 else None))


def plot_kline_comparison(history_df: pd.DataFrame,
                          actual_df: Optional[pd.DataFrame],
                          pred_dict: Dict[str, pd.DataFrame],
                          out_path: str,
                          title: str = "Kronos 预测对比（日K线）",
                          history_tail: int = 60) -> str:
    """绘制历史 K 线 + 真实未来 K 线 + 各模型预测 close 对比图。

    Args:
        history_df:  历史窗口（索引时间，含 OHLC）。
        actual_df:   真实未来窗口（含 OHLC）；None 表示无真值。
        pred_dict:   {标签: 预测 DataFrame}；以 close 折线 + 高低带叠加。
        out_path:    输出 PNG 路径。
        title:       图标题。
        history_tail:仅展示历史最后多少根，避免过密。

    Returns:
        out_path。
    """
    hist = history_df.tail(history_tail).copy()
    n_hist = len(hist)

    # 统一用整数横坐标，避免周末 / 停牌造成的日期空档。
    fig, ax = plt.subplots(figsize=(13, 6))

    hist_idx = np.arange(n_hist)
    _draw_candles(ax, hist_idx, hist["open"].values, hist["high"].values,
                  hist["low"].values, hist["close"].values, alpha=0.55)

    # 拼接横坐标标签所需的时间序列。
    labels = list(hist.index)

    if actual_df is not None and len(actual_df) > 0:
        n_act = len(actual_df)
        act_idx = np.arange(n_hist, n_hist + n_act)
        _draw_candles(ax, act_idx, actual_df["open"].values, actual_df["high"].values,
                      actual_df["low"].values, actual_df["close"].values, alpha=1.0)
        labels += list(actual_df.index)
        ax.axvline(n_hist - 0.5, color="gray", linestyle="--", linewidth=1.0)
        ax.text(n_hist - 0.5, ax.get_ylim()[1], "  预测起点", va="top", ha="left",
                color="gray", fontsize=9)

    # 叠加各模型预测的 close 折线 + 高低带。
    for i, (label, pdf) in enumerate(pred_dict.items()):
        color = _PRED_COLORS[i % len(_PRED_COLORS)]
        n_pred = len(pdf)
        pidx = np.arange(n_hist, n_hist + n_pred)
        ax.plot(pidx, pdf["close"].values, color=color, marker="o", markersize=3,
                linewidth=1.6, label=f"{label} (pred close)")
        if {"high", "low"}.issubset(pdf.columns):
            ax.fill_between(pidx, pdf["low"].values, pdf["high"].values,
                            color=color, alpha=0.12)

    # 横坐标稀疏标注日期。
    all_n = n_hist + (len(actual_df) if actual_df is not None else
                      (max((len(p) for p in pred_dict.values()), default=0)))
    step = max(1, all_n // 12)
    ticks = list(range(0, all_n, step))
    tick_labels = []
    for t in ticks:
        if t < len(labels):
            ts = pd.Timestamp(labels[t])
            tick_labels.append(ts.strftime("%Y-%m-%d"))
        else:
            tick_labels.append("")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("价格")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
