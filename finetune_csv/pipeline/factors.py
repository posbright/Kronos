"""因子缺失值处理策略（leak-safe null handling）。

背景：基本面 / 技术指标 / 消息面因子常有缺失——
    - 基本面（fin_*）：季度披露，日频上大段为空，应「前向填充」沿用最近一期。
    - 技术指标（tech_*）：上市初期窗口不足为空，少量零散缺失，宜「前向填充 + 中位数兜底」。
    - 消息面 / 事件（news_/event_/sent_）：天然稀疏，缺失即「无事件」，应「填 0」。

核心原则：
    1. 防泄漏：填充只能用「过去」信息。前向填充（ffill）天然满足；禁止全局 bfill（会把未来
       值带到过去）。序列开头 ffill 无值时，用「该列在训练区间的中位数」兜底（仍是历史统计），
       再不行才填 0。
    2. 保留缺失信息：对缺失率较高的因子，额外生成 `<col>_isna` 掩码列（1=原始缺失），
       让模型有机会学习「缺失本身」的含义，而不是被填充值误导。
    3. 可验证：analyze_missing 给出每列缺失率，便于人工核对策略是否合理。

取值 / 阈值建议：
    - mask_threshold（默认 0.05）：缺失率 > 5% 的因子才追加掩码列，避免列爆炸。
    - drop_threshold（默认 0.6）：缺失率 > 60% 的因子默认建议丢弃（仅提示，不强制）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# 列名前缀 -> 默认策略。可被显式 strategy_map 覆盖。
_PREFIX_STRATEGY = [
    ("fin_", "ffill"),
    ("tech_", "ffill"),
    ("news", "zero"),
    ("event", "zero"),
    ("sent", "zero"),
    ("north", "ffill"),
]


def analyze_missing(df: pd.DataFrame, factor_cols: Sequence[str]) -> Dict[str, float]:
    """统计每个因子列的缺失率（0~1）。

    Args:
        df: 数据表。
        factor_cols: 因子列名。

    Returns:
        {列名: 缺失率}，列不存在则记为 1.0（视为完全缺失）。
    """
    out: Dict[str, float] = {}
    n = max(1, len(df))
    for c in factor_cols:
        if c not in df.columns:
            out[c] = 1.0
        else:
            out[c] = float(df[c].isna().mean())
    return out


def infer_factor_strategy(
    factor_cols: Sequence[str],
    explicit: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """按列名前缀推断每个因子的缺失处理策略（'ffill' 或 'zero'）。

    Args:
        factor_cols: 因子列名。
        explicit:    显式指定的 {列名: 策略}，优先级最高。

    Returns:
        {列名: 'ffill' | 'zero'}，未命中前缀的默认 'ffill'（最通用、最稳）。
    """
    explicit = explicit or {}
    strategy: Dict[str, str] = {}
    for c in factor_cols:
        if c in explicit:
            strategy[c] = explicit[c]
            continue
        chosen = "ffill"
        low = c.lower()
        for prefix, strat in _PREFIX_STRATEGY:
            if low.startswith(prefix):
                chosen = strat
                break
        strategy[c] = chosen
    return strategy


def handle_factor_nulls(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    strategy_map: Optional[Dict[str, str]] = None,
    group_col: Optional[str] = "symbol",
    train_mask: Optional[pd.Series] = None,
    add_mask: bool = True,
    mask_threshold: float = 0.05,
) -> pd.DataFrame:
    """对因子列做防泄漏缺失填充，并按需追加缺失掩码列。

    步骤（每列、按 group_col 分组以避免跨标的串味）：
        1. 记录原始缺失位置（用于掩码与统计）。
        2. ffill：前向填充（仅用过去）。
        3. 兜底：
           - 'ffill' 策略：序列开头仍缺失 -> 用「训练区间中位数」填充；再不行填 0。
           - 'zero'  策略：剩余缺失 -> 直接填 0（缺失即无事件）。
        4. 若该列缺失率 > mask_threshold 且 add_mask=True，追加 `<col>_isna`（1=原缺失）。

    Args:
        df:           输入表（不就地修改，返回副本）。
        factor_cols:  因子列名。
        strategy_map: {列名: 'ffill'|'zero'}；None 则用 infer_factor_strategy 推断。
        group_col:    分组列（如 'symbol'）；None 表示不分组。
        train_mask:   布尔 Series，标识训练区间行；中位数仅用训练区间统计（防泄漏）。
                      None 时退化为用全列中位数（小范围验证可接受）。
        add_mask:     是否追加缺失掩码列。
        mask_threshold: 追加掩码列的缺失率阈值。

    Returns:
        填充后的 DataFrame；保证 factor_cols 无缺失（assert 校验）。
    """
    out = df.copy()
    strategy_map = strategy_map or infer_factor_strategy(factor_cols)
    present = [c for c in factor_cols if c in out.columns]

    # 预先计算每列「训练区间中位数」，作为开头缺失的兜底（仍是历史统计，无未来泄漏）。
    medians: Dict[str, float] = {}
    for c in present:
        if train_mask is not None and train_mask.any():
            med = pd.to_numeric(out.loc[train_mask, c], errors="coerce").median()
        else:
            med = pd.to_numeric(out[c], errors="coerce").median()
        medians[c] = float(med) if pd.notna(med) else 0.0

    for c in present:
        was_na = out[c].isna()
        out[c] = pd.to_numeric(out[c], errors="coerce")

        # 1) 前向填充（分组，防止跨标的串味）。
        if group_col and group_col in out.columns:
            out[c] = out.groupby(group_col)[c].ffill()
        else:
            out[c] = out[c].ffill()

        # 2) 兜底填充。
        strat = strategy_map.get(c, "ffill")
        if strat == "zero":
            out[c] = out[c].fillna(0.0)
        else:  # ffill 策略：中位数兜底，再不行填 0。
            out[c] = out[c].fillna(medians[c]).fillna(0.0)

        # 3) 缺失掩码列。
        if add_mask and float(was_na.mean()) > mask_threshold:
            out[f"{c}_isna"] = was_na.astype(np.int8)

    remaining = out[present].isna().any().any() if present else False
    assert not remaining, "因子缺失填充后仍有 NaN，请检查 strategy_map / 输入数据"
    return out
