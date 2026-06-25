"""CPU 基准：测量 Kronos 单窗 predict 耗时，外推 20 只标的特征生成的总时间与可行性。

仅用于评估，不写任何数据集产物。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
PREDICTOR = "NeoQuasar/Kronos-base"

LOOKBACK = 90
PRED = 5
PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def make_window(n: int = LOOKBACK + PRED, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 10 + np.cumsum(rng.standard_normal(n) * 0.1)
    return pd.DataFrame(
        {
            "timestamps": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.05,
            "volume": rng.integers(1e5, 1e6, n).astype(float),
            "amount": 0.0,
        }
    )


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"torch={torch.__version__} cpu_threads={torch.get_num_threads()}")

    t0 = time.time()
    tok = KronosTokenizer.from_pretrained(TOKENIZER)
    mdl = Kronos.from_pretrained(PREDICTOR)
    predictor = KronosPredictor(mdl, tok, device="cpu", max_context=512)
    print(f"模型加载耗时: {time.time() - t0:.1f}s")

    px = make_window()
    hist = px.iloc[:LOOKBACK]
    x_df = hist[PRICE_COLS].reset_index(drop=True)
    x_ts = hist["timestamps"].reset_index(drop=True)
    y_ts = px["timestamps"].iloc[LOOKBACK:LOOKBACK + PRED].reset_index(drop=True)

    # 预热一次（首个调用包含图构建/缓存，单独计）。
    t = time.time()
    predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED,
                      T=1.0, top_p=0.9, sample_count=1, verbose=False)
    warm = time.time() - t
    print(f"预热单次 predict(sample_count=1): {warm:.3f}s")

    # 稳定态：测 5 次单样本 predict。
    N = 5
    t = time.time()
    for _ in range(N):
        predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED,
                          T=1.0, top_p=0.9, sample_count=1, verbose=False)
    per_call = (time.time() - t) / N
    print(f"稳定态单次 predict(sample_count=1) 平均: {per_call:.3f}s")

    # 用 sample_count=S 的批量采样是否更快（一次前向多路径）。
    for S in (10, 30):
        t = time.time()
        predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED,
                          T=1.0, top_p=0.9, sample_count=S, verbose=False)
        dt = time.time() - t
        print(f"predict(sample_count={S}) 一次调用: {dt:.3f}s  ->  等效每路径 {dt / S:.3f}s")

    # 外推：每只股票窗口数 × 每窗采样成本。
    print("\n=== 20 只标的总耗时外推 ===")
    for trading_days in (600, 1200, 2300):
        windows = trading_days - LOOKBACK - PRED + 1
        for samples in (10, 30):
            # 现脚本实现为逐样本循环 -> per_call * samples * windows。
            sec = per_call * samples * windows * 20
            print(f"  交易日≈{trading_days} 窗口≈{windows} samples={samples}: "
                  f"单只 {per_call * samples * windows / 60:.1f}min, 20只 {sec / 3600:.2f}h")


if __name__ == "__main__":
    main()
