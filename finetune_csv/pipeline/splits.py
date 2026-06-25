"""滚动日期切分（rolling date splits）。

需求来源：以「当前日期-1天」为锚点，把 **测试集 / 验证集 / 训练集** 依次向前（过去）回推，
尽量使用最新数据。即：

    时间轴 ──────────────────────────────────────────────►  anchor(=今天-1)
    [        train        ][   validation   ][     test     ]
                          ^train_end        ^val_end        ^anchor

切分以「交易日个数」为单位（而非自然日），因为 A 股周末 / 节假日不交易；用自然日会让
不同区间实际样本量漂移。返回的边界日期可直接喂给 build_full_market_dataset.merge_symbol_features：
    - train      : date <= train_end
    - validation : train_end < date <= val_end
    - test       : date > val_end (且 date <= anchor)

关键参数与取值建议：
    - anchor      : 锚点日期，默认 = 今天 - 1 天；实际会再夹到「数据中可得的最新交易日」。
    - test_days   : 测试集交易日数。小范围验证常用 20~60；过小则指标方差大，过大则挤占训练。
    - val_days    : 验证集交易日数。一般与 test 同量级（20~60），用于早停 / 选模。
    - train_days  : 训练集交易日数；None 表示「锚点之前剩余全部」。日频数据建议尽量多。
    - label_horizon: 标签前看步数（与数据集构建一致）。仅用于在报告里提示边界购买区，
                     真正的越界样本清洗由 merge_symbol_features 完成。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import pandas as pd


@dataclass
class SplitPlan:
    """一次滚动切分的完整边界描述（日期均为 pandas.Timestamp，已归一到 00:00）。"""

    anchor: pd.Timestamp          # 实际锚点（已夹到可得最新交易日）
    train_start: pd.Timestamp     # 训练集最早交易日
    train_end: pd.Timestamp       # 训练集最晚交易日（含）
    val_start: pd.Timestamp       # 验证集最早交易日
    val_end: pd.Timestamp         # 验证集最晚交易日（含）
    test_start: pd.Timestamp      # 测试集最早交易日
    test_end: pd.Timestamp        # 测试集最晚交易日（含，= anchor）
    n_train: int                  # 训练集交易日数
    n_val: int                    # 验证集交易日数
    n_test: int                   # 测试集交易日数

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, pd.Timestamp):
                d[k] = v.strftime("%Y-%m-%d")
        return d

    def summary(self) -> str:
        return (
            f"train [{self.train_start.date()} ~ {self.train_end.date()}] ({self.n_train}d) | "
            f"val [{self.val_start.date()} ~ {self.val_end.date()}] ({self.n_val}d) | "
            f"test [{self.test_start.date()} ~ {self.test_end.date()}] ({self.n_test}d)"
        )


def collect_trading_dates(date_series_list: Sequence[pd.Series]) -> List[pd.Timestamp]:
    """合并多只标的的交易日，得到全市场去重、升序的交易日列表。

    Args:
        date_series_list: 多个 pandas.Series（每只标的的 date 列），元素可为字符串或 datetime。

    Returns:
        升序、去重、已归一到 00:00 的交易日列表。
    """
    all_dates: set = set()
    for s in date_series_list:
        if s is None or len(s) == 0:
            continue
        ts = pd.to_datetime(pd.Series(s), errors="coerce").dropna().dt.normalize()
        all_dates.update(ts.tolist())
    return sorted(all_dates)


def _default_anchor() -> pd.Timestamp:
    """默认锚点 = 今天 - 1 天（归一到 00:00）。"""
    return pd.Timestamp(dt.date.today() - dt.timedelta(days=1)).normalize()


def rolling_date_splits(
    trading_dates: Sequence[pd.Timestamp],
    test_days: int,
    val_days: int,
    train_days: Optional[int] = None,
    anchor: Optional[pd.Timestamp] = None,
    label_horizon: int = 0,
) -> SplitPlan:
    """以锚点为终点，向过去回推得到 test/val/train 三段连续的日期边界。

    Args:
        trading_dates: 升序交易日列表（用 collect_trading_dates 生成）。
        test_days:     测试集交易日数（最靠近锚点的一段）。
        val_days:      验证集交易日数（test 之前的一段）。
        train_days:    训练集交易日数；None=取 val 之前的剩余全部。
        anchor:        锚点日期；None=今天-1。会被夹到 <=锚点 的最新交易日。
        label_horizon: 标签前看步数；此处仅透传到日志，越界清洗由数据集构建完成。

    Returns:
        SplitPlan，边界日期可直接用于 merge_symbol_features(train_end, val_end)。

    Raises:
        ValueError: 交易日不足以容纳 test_days + val_days（+ 至少 1 天训练）。
    """
    if test_days <= 0 or val_days <= 0:
        raise ValueError("test_days 与 val_days 必须为正整数")

    dates = [pd.Timestamp(d).normalize() for d in trading_dates]
    dates = sorted(set(dates))
    if not dates:
        raise ValueError("trading_dates 为空，无法切分")

    anchor = (anchor or _default_anchor())
    anchor = pd.Timestamp(anchor).normalize()

    # 夹到 <= anchor 的最新交易日（数据可能晚于 anchor，也可能早于 anchor）。
    usable = [d for d in dates if d <= anchor]
    if not usable:
        raise ValueError(f"没有 <= 锚点 {anchor.date()} 的交易日可用")
    anchor = usable[-1]

    n = len(usable)
    need_min = test_days + val_days + 1
    if n < need_min:
        raise ValueError(
            f"可用交易日 {n} 不足以容纳 test({test_days}) + val({val_days}) + 训练(>=1)；"
            f"至少需要 {need_min} 个交易日。请减小 test_days/val_days 或扩大数据范围。"
        )

    test_start_idx = n - test_days                 # test = usable[test_start_idx:]
    val_start_idx = test_start_idx - val_days       # val  = usable[val_start_idx:test_start_idx]
    if train_days is None:
        train_start_idx = 0
    else:
        train_start_idx = max(0, val_start_idx - train_days)

    train_start = usable[train_start_idx]
    train_end = usable[val_start_idx - 1]
    val_start = usable[val_start_idx]
    val_end = usable[test_start_idx - 1]
    test_start = usable[test_start_idx]
    test_end = usable[-1]

    return SplitPlan(
        anchor=anchor,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        n_train=val_start_idx - train_start_idx,
        n_val=val_days,
        n_test=test_days,
    )
