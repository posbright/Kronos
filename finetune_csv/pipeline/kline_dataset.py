"""预切分 K 线数据集 PreSplitKlineDataset（多股票、防跨标的）。

与 finetune_base_model.CustomKlineDataset 的区别：
    1. CustomKlineDataset 读入一个 CSV 后按 train/val/test 比例「内部再切分」；
       本类直接把「整份预切分 CSV」当作数据集使用，不再二次切分（切分已由
       滚动日期流水线按真实交易日严格回推完成，防泄漏）。
    2. 数据集可能把多只股票纵向堆叠（含 symbol 列）。本类按 symbol 分组，
       仅在「同一只股票内部」滑动窗口，绝不跨标的取窗，避免把两只股票的
       价格序列拼接成一个虚假窗口。

时间戳列兼容：优先用 'timestamps'，否则用 'date'（日频）。
归一化与既有训练一致：逐窗口 z-score（仅用窗口内统计量）再裁剪到 ±clip。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PreSplitKlineDataset(Dataset):
    """加载单份预切分 CSV 的 K 线数据集（按 symbol 分组取窗）。

    参数：
        data_path:       预切分 CSV（含 open/high/low/close/volume/amount 与 timestamps 或 date）。
        role:            'train' 随机窗口起点（配合 set_epoch_seed 轮内打散）；
                         'eval' 确定性窗口起点（验证 / 测试可复现）。
        lookback_window: 历史回看长度。
        predict_window:  预测步数。窗口长度 = lookback + predict + 1。
        clip:            z-score 后裁剪阈值（±clip）。
        seed:            随机种子（role='train' 用）。
        symbol_col:      股票标识列名；None 时自动探测 'symbol'，不存在则视为单股票。
    """

    FEATURES: List[str] = ["open", "high", "low", "close", "volume", "amount"]
    TIME_FEATURES: List[str] = ["minute", "hour", "weekday", "day", "month"]

    def __init__(self, data_path: str, role: str = "train",
                 lookback_window: int = 90, predict_window: int = 10,
                 clip: float = 5.0, seed: int = 100,
                 symbol_col: Optional[str] = None):
        assert role in ("train", "eval"), f"role 必须是 train/eval，收到 {role}"
        self.data_path = data_path
        self.role = role
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.current_epoch = 0

        df = pd.read_csv(data_path)

        # 时间戳列兼容：timestamps 优先，否则 date。
        ts_col = "timestamps" if "timestamps" in df.columns else (
            "date" if "date" in df.columns else None)
        if ts_col is None:
            raise ValueError(f"{data_path} 缺少 timestamps / date 列，无法构建时间特征。")
        df["timestamps"] = pd.to_datetime(df[ts_col])

        # 自动探测股票标识列。
        if symbol_col is None:
            symbol_col = "symbol" if "symbol" in df.columns else None
        self.symbol_col = symbol_col

        sort_cols = ([symbol_col] if symbol_col else []) + ["timestamps"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        df["minute"] = df["timestamps"].dt.minute
        df["hour"] = df["timestamps"].dt.hour
        df["weekday"] = df["timestamps"].dt.weekday
        df["day"] = df["timestamps"].dt.day
        df["month"] = df["timestamps"].dt.month

        missing_feat = [c for c in self.FEATURES if c not in df.columns]
        if missing_feat:
            raise ValueError(f"{data_path} 缺少价量列：{missing_feat}")

        self.data = df[self.FEATURES + self.TIME_FEATURES].copy()
        if self.data.isnull().any().any():
            self.data = self.data.ffill().bfill()

        # 计算「合法窗口起点」：仅落在同一只股票内部、且窗口完整的起点。
        self._valid_starts: List[int] = self._build_valid_starts(df)
        if not self._valid_starts:
            raise ValueError(
                f"{data_path} 中没有任何一只股票的连续行数 >= 窗口长度 {self.window}。"
                f"请减小 lookback/predict 或扩大该切分天数。"
            )
        self.n_samples = len(self._valid_starts)

    def _build_valid_starts(self, df: pd.DataFrame) -> List[int]:
        """按 symbol 分组，收集所有不跨标的的合法窗口起点（全局行号）。"""
        starts: List[int] = []
        if self.symbol_col:
            # df 已按 [symbol, timestamps] 排序，同组行号连续；按组求边界。
            groups = df.groupby(self.symbol_col, sort=False).indices
            spans: List[Tuple[int, int]] = []
            for idx_arr in groups.values():
                spans.append((int(idx_arr[0]), int(idx_arr[-1]) + 1))  # [start, end)
        else:
            spans = [(0, len(df))]
        for s, e in spans:
            last_start = e - self.window  # 含 [start, start+window)
            for st in range(s, last_start + 1):
                starts.append(st)
        return starts

    def set_epoch_seed(self, epoch: int) -> None:
        """设置轮次种子（仅训练用，使每轮窗口顺序不同以增强泛化）。"""
        self.current_epoch = epoch

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.role == "train":
            # 在合法起点列表内做确定性打散（随 epoch 改变映射）。
            pos = (idx * 9973 + (self.current_epoch + 1) * 104729) % self.n_samples
        else:
            pos = idx % self.n_samples
        start_idx = self._valid_starts[pos]
        end_idx = start_idx + self.window

        window_data = self.data.iloc[start_idx:end_idx]
        x = window_data[self.FEATURES].values.astype(np.float32)
        x_stamp = window_data[self.TIME_FEATURES].values.astype(np.float32)

        # 逐窗口 z-score（只用窗口内统计量，防止跨样本泄漏），再裁剪极端值。
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)
