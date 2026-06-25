"""因子条件的预切分 K 线数据集 FactorKlinePreSplitDataset（方案 B + 每因子缩放）。

在 ``PreSplitKlineDataset``（多股票、防跨标的、逐窗口归一化）基础上，额外读取若干
「因子列」，并对每个因子施加用户设定的「缩放系数（weight）」：

    归一化后的因子 = zscore(factor, 仅用 lookback 段) * weight

设计要点：
    - 因子 z-score 只用窗口的「lookback 段」统计量（与推理期 FactorPredictor 一致，防泄漏）。
    - 每因子 weight 由前端可调：weight=0 等价于关闭该因子，weight 越大该因子信号越强。
    - 价格/时间特征与父类完全一致，__getitem__ 多返回一个 factor 张量。

返回三元组：(price[W,6], stamp[W,5], factor[W,k])，W = lookback + predict + 1。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from .kline_dataset import PreSplitKlineDataset
except ImportError:  # 允许作为单文件脚本直接运行（python factor_dataset.py --smoke）
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kline_dataset import PreSplitKlineDataset


def _zscore_arr(v: np.ndarray) -> np.ndarray:
    """对一维数组做全局 z-score（用于聚合前把同类因子拉到可比尺度）。"""
    mu, sd = float(np.mean(v)), float(np.std(v))
    if sd < 1e-8:
        return np.zeros_like(v)
    return (v - mu) / sd


def _fill_col_leaksafe(df: pd.DataFrame, col: str,
                       group_col: Optional[str]) -> np.ndarray:
    """防泄漏地补齐单列缺失，返回 float32 数组。

    规则（与 pipeline.factors.handle_factor_nulls 一致，禁止 bfill 引入未来值）：
        1. 有 group_col 时按标的分组前向填充（ffill），避免跨标的串值；否则整体 ffill。
        2. 序列开头 ffill 无值时，用「该列中位数」兜底（历史统计，非未来泄漏）；再不行填 0。

    注：规范数据集（run_build_dataset 产出）已无缺失，此处为 no-op；仅对未填充输入兜底。
    """
    s = pd.to_numeric(df[col], errors="coerce")
    if group_col and group_col in df.columns:
        s = s.groupby(df[group_col], sort=False).ffill()
    else:
        s = s.ffill()
    med = s.median()
    s = s.fillna(med if pd.notna(med) else 0.0)
    return s.astype(np.float32).values


def normalize_specs(factor_cols: Optional[Sequence[str]] = None,
                    factor_weights: Optional[Dict[str, float]] = None,
                    factor_specs: Optional[Sequence[dict]] = None) -> List[dict]:
    """把「原始列+权重」或「聚合规格」统一为规范的因子通道规格列表。

    规范通道规格字段：
        name   通道名（训练后用于 factor_importance 标签）
        mode   "raw"（单列直通）| "mean"（同类方向对齐后取 z 均值）
        cols   参与的原始列
        signs  每个 col 的方向(+1/-1)，仅 mean 模式用于对齐多空
        weight 缩放系数（0=关闭）

    优先级：显式 factor_specs > (factor_cols + factor_weights)。
    """
    if factor_specs:
        out = []
        for sp in factor_specs:
            cols = list(sp.get("cols") or ([sp["col"]] if sp.get("col") else []))
            if not cols:
                continue
            mode = sp.get("mode", "raw")
            signs = sp.get("signs") or [1] * len(cols)
            name = sp.get("name") or (cols[0] if mode == "raw" else "+".join(cols))
            out.append({"name": name, "mode": mode, "cols": cols,
                        "signs": [int(s) for s in signs],
                        "weight": float(sp.get("weight", 1.0))})
        return out
    # 回退：每个原始列一个 raw 通道
    cols = list(factor_cols or [])
    fw = dict(factor_weights or {})
    return [{"name": c, "mode": "raw", "cols": [c], "signs": [1],
             "weight": float(fw.get(c, 1.0))} for c in cols]


def build_factor_matrix(df: pd.DataFrame, specs: Sequence[dict],
                        group_col: Optional[str] = None) -> np.ndarray:
    """按规范通道规格构造 [N, n_channels] 因子矩阵（聚合在此完成，不含逐窗 z-score）。

    Args:
        df:        含各因子原始列的表（已按 symbol+时间排序）。
        specs:     规范通道规格（见 normalize_specs）。
        group_col: 标的列名；提供时缺失填充按标的分组，避免跨标的串值。
    """
    chans = []
    for sp in specs:
        cols = sp["cols"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"通道 {sp['name']} 缺少列：{missing}")
        if sp["mode"] == "mean" and len(cols) > 1:
            # 同类方向对齐后取 z 均值 -> 单通道
            mat = np.stack([_zscore_arr(_fill_col_leaksafe(df, c, group_col)) * s
                            for c, s in zip(cols, sp["signs"])], axis=1)
            chan = mat.mean(axis=1)
        else:
            chan = _fill_col_leaksafe(df, cols[0], group_col)
            if sp["signs"][0] < 0:
                chan = -chan
        chans.append(chan.astype(np.float32))
    return np.stack(chans, axis=1) if chans else np.zeros((len(df), 0), np.float32)


class FactorKlinePreSplitDataset(PreSplitKlineDataset):
    """带因子列与每因子缩放系数的预切分数据集（按 symbol 分组取窗）。"""

    def __init__(self, data_path: str, factor_cols: Sequence[str] = (),
                 factor_weights: Optional[Dict[str, float]] = None,
                 role: str = "train", lookback_window: int = 90,
                 predict_window: int = 10, clip: float = 5.0, seed: int = 100,
                 symbol_col: Optional[str] = None,
                 factor_specs: Optional[Sequence[dict]] = None):
        # 统一为规范通道规格（支持「原始列+权重」与「同类聚合」两种入参）。
        self.specs: List[dict] = normalize_specs(factor_cols, factor_weights, factor_specs)
        self.factor_cols: List[str] = [sp["name"] for sp in self.specs]
        # 每通道缩放系数：weight=0 等价关闭该通道。
        self.factor_weights = np.array([sp["weight"] for sp in self.specs],
                                       dtype=np.float32)

        super().__init__(data_path, role=role, lookback_window=lookback_window,
                         predict_window=predict_window, clip=clip, seed=seed,
                         symbol_col=symbol_col)

        # 父类只保留了价量 + 时间列；这里另存因子通道（聚合后，按全局行号对齐）。
        df = pd.read_csv(data_path)
        ts_col = "timestamps" if "timestamps" in df.columns else "date"
        df["timestamps"] = pd.to_datetime(df[ts_col])
        sort_cols = ([self.symbol_col] if self.symbol_col else []) + ["timestamps"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # [N, k]，k = 通道数；聚合（mean/raw、方向对齐）在此完成。
        # 传入 symbol 列，缺失填充按标的分组（防跨标的串值 / 防 bfill 引未来）。
        self.factor_data = build_factor_matrix(df, self.specs, group_col=self.symbol_col)


    @property
    def factor_dim(self) -> int:
        return len(self.factor_cols)

    def __getitem__(self, idx: int):
        # 复用父类的窗口起点选择逻辑（含 train 打散 / eval 确定性）。
        if self.role == "train":
            pos = (idx * 9973 + (self.current_epoch + 1) * 104729) % self.n_samples
        else:
            pos = idx % self.n_samples
        start_idx = self._valid_starts[pos]
        end_idx = start_idx + self.window

        window_data = self.data.iloc[start_idx:end_idx]
        x = window_data[self.FEATURES].values.astype(np.float32)
        x_stamp = window_data[self.TIME_FEATURES].values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

        # 因子：只用 lookback 段统计量做 z-score（与推理一致，防泄漏），再乘以每因子权重。
        factor_win = self.factor_data[start_idx:end_idx]              # [W, k]
        look = self.lookback_window
        f_mean = np.mean(factor_win[:look], axis=0)
        f_std = np.std(factor_win[:look], axis=0)
        factor = np.clip((factor_win - f_mean) / (f_std + 1e-5), -self.clip, self.clip)
        factor = factor * self.factor_weights[np.newaxis, :]

        return (torch.from_numpy(x), torch.from_numpy(x_stamp),
                torch.from_numpy(factor.astype(np.float32)))


def _smoke_test() -> None:
    """合成多股票 CSV，验证因子三元组形状、防跨标的、weight=0 关闭因子。"""
    import tempfile
    import os

    rng = np.random.default_rng(0)
    rows = []
    for sym in ("000001", "600000"):
        n = 80
        base = 10 + np.cumsum(rng.normal(0, 0.1, n))
        rows.append(pd.DataFrame({
            "symbol": sym,
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": base, "high": base + 0.2, "low": base - 0.2, "close": base + 0.05,
            "volume": rng.uniform(1e5, 1e6, n), "amount": rng.uniform(1e6, 1e7, n),
            "tech_macd": rng.normal(0, 1, n), "tech_rsi": rng.normal(0, 1, n),
        }))
    df = pd.concat(rows, ignore_index=True)

    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "ds.csv")
        df.to_csv(csv, index=False)
        cols = ["tech_macd", "tech_rsi"]

        ds = FactorKlinePreSplitDataset(csv, cols, {"tech_macd": 1.0, "tech_rsi": 0.0},
                                        role="eval", lookback_window=30, predict_window=5)
        x, stamp, factor = ds[0]
        W = 30 + 5 + 1
        assert x.shape == (W, 6) and stamp.shape == (W, 5) and factor.shape == (W, 2), \
            f"形状异常: {x.shape},{stamp.shape},{factor.shape}"
        # weight=0 的 tech_rsi 列应全 0；tech_macd 非全 0。
        assert torch.allclose(factor[:, 1], torch.zeros(W)), "weight=0 应关闭该因子"
        assert not torch.allclose(factor[:, 0], torch.zeros(W)), "weight=1 因子不应全 0"
        # 每只股票 80 行 -> 合法起点 80-36+1=45，两只共 90，确保未跨标的。
        assert ds.n_samples == 2 * (80 - W + 1), f"样本数异常 {ds.n_samples}"

        # 聚合模式：把两个因子按方向对齐取均值 -> 单通道。
        specs = [{"name": "mom_mean", "mode": "mean",
                  "cols": ["tech_macd", "tech_rsi"], "signs": [1, -1], "weight": 1.0}]
        ds2 = FactorKlinePreSplitDataset(csv, role="eval", lookback_window=30,
                                         predict_window=5, factor_specs=specs)
        _, _, f2 = ds2[0]
        assert ds2.factor_dim == 1 and f2.shape == (W, 1), \
            f"聚合后应为单通道: dim={ds2.factor_dim}, shape={tuple(f2.shape)}"
        assert ds2.factor_cols == ["mom_mean"], ds2.factor_cols

        # 缺失填充防跨标的：给 600000 开头 3 行置 NaN，确保不会被 000001 的值串入。
        dfn = df.copy()
        m = dfn["symbol"] == "600000"
        head_idx = dfn[m].index[:3]
        dfn.loc[head_idx, "tech_macd"] = np.nan
        mat = build_factor_matrix(
            dfn.sort_values(["symbol", "date"]).reset_index(drop=True),
            [{"name": "tech_macd", "mode": "raw", "cols": ["tech_macd"],
              "signs": [1], "weight": 1.0}],
            group_col="symbol")
        assert not np.isnan(mat).any(), "缺失填充后不应有 NaN"
        # 开头缺失用「该列中位数」兜底，绝不会等于 000001 末行的值（防跨标的串味）。
        last_000001 = float(dfn[dfn["symbol"] == "000001"]["tech_macd"].iloc[-1])
        first_600000_filled = float(mat[(dfn.sort_values(["symbol", "date"])
                                         .reset_index(drop=True)["symbol"] == "600000")
                                        .values.argmax(), 0])
        assert abs(first_600000_filled - last_000001) > 1e-9, "跨标的串味！开头不应取上一标的值"

        print(f"[smoke] factor_dataset 通过：三元组形状正确，weight=0 关闭因子，"
              f"样本数={ds.n_samples}（防跨标的）；聚合通道维度={ds2.factor_dim}；"
              f"缺失填充防跨标的 OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="因子条件预切分数据集")
    parser.add_argument("--smoke", action="store_true", help="运行合成数据冒烟自测")
    args = parser.parse_args()
    if args.smoke:
        _smoke_test()
    else:
        parser.print_help()
