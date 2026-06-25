"""基于 OHLCV 在本地计算技术指标（纯 pandas / numpy，无需 TA-Lib）。

动机：远程库 cn_stock_indicators 仅覆盖近期，对 2020~2026 的训练区间约 97% 为空。
改为「用本地 OHLCV 现算」可获得完整历史，且口径自控、可复现。

输出列名与 build_full_market_dataset.CANDIDATE_TECH_COLS 完全一致（统一加 `tech_` 前缀），
因此下游合并 / 因子识别 / 缺失处理逻辑都无需改动。

指标与常用参数（括号内为本实现采用的窗口，均为业界常用默认值）：
    - MACD(12,26,9)         : macd / macds(信号) / macdh(柱=2×(macd-macds))
    - KDJ(9,3,3)            : kdjk / kdjd / kdjj
    - RSI(6/12/14/24)       : rsi_6 / rsi_12 / rsi / rsi_24（Wilder 平滑）
    - BOLL(20,2)            : boll(中轨) / boll_ub(上轨) / boll_lb(下轨)
    - ATR(14)               : 真实波幅均值（Wilder）
    - OBV                   : 能量潮（累积带方向成交量）
    - CCI(20)               : 顺势指标
    - VR(26)                : 成交量比率
    - ROC(12) / ROCMA(6)    : 变动率及其均线
    - DMI(14)               : pdi / mdi / adx / adxr
    - WR(6/10/14)           : 威廉指标
    - MFI(14)               : 资金流量指标
    - PSY(12) / PSYMA(6)    : 心理线及其均线
    - DMA(10,50,10)         : 平行线差（短均线-长均线）
    - SAR(0.02,0.2)         : 抛物线转向
    - TRIX(12)              : 三重指数平滑变动率

注意：各指标前若干行因窗口不足为 NaN（暖机期），交由 pipeline.factors 做防泄漏填充。
所有计算只使用「当前及过去」的数据，不会引入未来信息。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder 平滑 = alpha=1/n 的指数加权（RSI/ATR/DMI 通用）。"""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = _wilder(up, n)
    roll_down = _wilder(down, n)
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # down=0（全涨）时 rs=inf -> rsi=100；用 100 填充这些位置。
    return rsi.fillna(100.0)


def _psar(high: pd.Series, low: pd.Series,
          af_step: float = 0.02, af_max: float = 0.2) -> np.ndarray:
    """抛物线 SAR（标准迭代实现）。"""
    n = len(high)
    psar = np.full(n, np.nan)
    if n == 0:
        return psar
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)

    bull = True
    af = af_step
    ep = h[0]
    sar = l[0]
    psar[0] = sar
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            # SAR 不得高于前两根的最低价。
            sar = min(sar, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if h[i] > ep:
                ep = h[i]
                af = min(af + af_step, af_max)
            if l[i] < sar:  # 转空
                bull = False
                sar = ep
                ep = l[i]
                af = af_step
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if l[i] < ep:
                ep = l[i]
                af = min(af + af_step, af_max)
            if h[i] > sar:  # 转多
                bull = True
                sar = ep
                ep = h[i]
                af = af_step
        psar[i] = sar
    return psar


def compute_tech_indicators(kline_df: pd.DataFrame, prefix: str = "tech_") -> pd.DataFrame:
    """对单只标的的日线 OHLCV 计算全套技术指标。

    Args:
        kline_df: 含 ['date','open','high','low','close','volume'] 的表，按日期升序。
        prefix:   输出列前缀（默认 'tech_'，与 DB 口径一致）。

    Returns:
        DataFrame：['date'] + [prefix+指标...]，行数与输入一致（暖机期为 NaN）。
    """
    df = kline_df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = pd.to_numeric(df["volume"], errors="coerce").astype(float).fillna(0.0)

    out = pd.DataFrame({"date": df["date"].values})

    # ---- MACD ----
    macd = _ema(close, 12) - _ema(close, 26)
    macds = _ema(macd, 9)
    out["macd"] = macd
    out["macds"] = macds
    out["macdh"] = (macd - macds) * 2.0

    # ---- KDJ(9,3,3) ----
    low_n = low.rolling(9).min()
    high_n = high.rolling(9).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0.0, np.nan) * 100.0
    k = rsv.ewm(com=2, adjust=False).mean()   # alpha = 1/3
    d = k.ewm(com=2, adjust=False).mean()
    out["kdjk"] = k
    out["kdjd"] = d
    out["kdjj"] = 3.0 * k - 2.0 * d

    # ---- RSI ----
    out["rsi_6"] = _rsi(close, 6)
    out["rsi_12"] = _rsi(close, 12)
    out["rsi"] = _rsi(close, 14)
    out["rsi_24"] = _rsi(close, 24)

    # ---- BOLL(20,2) ----
    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    out["boll"] = mid
    out["boll_ub"] = mid + 2.0 * std
    out["boll_lb"] = mid - 2.0 * std

    # ---- ATR(14) ----
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = _wilder(tr, 14)

    # ---- OBV ----
    direction = np.sign(close.diff().fillna(0.0))
    out["obv"] = (direction * vol).cumsum()

    # ---- CCI(20) ----
    tp = (high + low + close) / 3.0
    tp_ma = tp.rolling(20).mean()
    md = (tp - tp_ma).abs().rolling(20).mean()
    out["cci"] = (tp - tp_ma) / (0.015 * md.replace(0.0, np.nan))

    # ---- VR(26) ----
    chg = close.diff()
    up_vol = vol.where(chg > 0, 0.0)
    down_vol = vol.where(chg < 0, 0.0)
    eq_vol = vol.where(chg == 0, 0.0)
    num = up_vol.rolling(26).sum() + 0.5 * eq_vol.rolling(26).sum()
    den = down_vol.rolling(26).sum() + 0.5 * eq_vol.rolling(26).sum()
    out["vr"] = num / den.replace(0.0, np.nan) * 100.0

    # ---- ROC(12) / ROCMA(6) ----
    roc = close.pct_change(12) * 100.0
    out["roc"] = roc
    out["rocma"] = roc.rolling(6).mean()

    # ---- DMI(14): pdi/mdi/adx/adxr ----
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)
    atr14 = _wilder(tr, 14)
    pdi = 100.0 * _wilder(plus_dm, 14) / atr14.replace(0.0, np.nan)
    mdi = 100.0 * _wilder(minus_dm, 14) / atr14.replace(0.0, np.nan)
    out["pdi"] = pdi
    out["mdi"] = mdi
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    adx = _wilder(dx, 14)
    out["adx"] = adx
    out["adxr"] = (adx + adx.shift(14)) / 2.0

    # ---- WR(6/10/14) ----
    for n in (6, 10, 14):
        hh = high.rolling(n).max()
        ll = low.rolling(n).min()
        out[f"wr_{n}"] = (hh - close) / (hh - ll).replace(0.0, np.nan) * 100.0

    # ---- MFI(14) ----
    mf = tp * vol
    pos_mf = mf.where(tp.diff() > 0, 0.0)
    neg_mf = mf.where(tp.diff() < 0, 0.0)
    mr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0.0, np.nan)
    out["mfi"] = 100.0 - 100.0 / (1.0 + mr)

    # ---- PSY(12) / PSYMA(6) ----
    psy = (close.diff() > 0).rolling(12).sum() / 12.0 * 100.0
    out["psy"] = psy
    out["psyma"] = psy.rolling(6).mean()

    # ---- DMA(10,50) ----
    out["dma"] = close.rolling(10).mean() - close.rolling(50).mean()

    # ---- SAR ----
    out["sar"] = _psar(high, low)

    # ---- TRIX(12) ----
    e1 = _ema(close, 12)
    e2 = _ema(e1, 12)
    e3 = _ema(e2, 12)
    out["trix"] = e3.pct_change() * 100.0

    # 加前缀（date 除外）。
    rename = {c: f"{prefix}{c}" for c in out.columns if c != "date"}
    out = out.rename(columns=rename)
    # inf -> NaN，交由下游缺失处理。
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _smoke_test() -> None:
    rng = np.random.default_rng(0)
    n = 400
    base = 10.0 + np.cumsum(rng.standard_normal(n) * 0.2)
    df = pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=n),
        "open": base * 0.995, "high": base * 1.02, "low": base * 0.98,
        "close": base, "volume": rng.integers(1e5, 1e6, n).astype(float),
    })
    feat = compute_tech_indicators(df)

    expected = [
        "macd", "macds", "macdh", "kdjk", "kdjd", "kdjj",
        "rsi_6", "rsi_12", "rsi", "rsi_24", "boll_ub", "boll", "boll_lb",
        "atr", "obv", "cci", "vr", "roc", "rocma", "pdi", "mdi", "adx", "adxr",
        "wr_6", "wr_10", "wr_14", "mfi", "psy", "psyma", "dma", "sar", "trix",
    ]
    cols = set(feat.columns)
    for e in expected:
        assert f"tech_{e}" in cols, f"缺少指标列 tech_{e}"
    assert len(feat) == n, "行数应与输入一致"

    # 暖机期(前 60 行)之后不应再有 NaN/inf。
    tail = feat.iloc[60:].drop(columns=["date"])
    bad = tail.replace([np.inf, -np.inf], np.nan).isna().sum()
    assert bad.sum() == 0, f"暖机后仍有缺失/无穷：\n{bad[bad > 0]}"

    # 取值范围合理性抽查。
    assert feat["tech_rsi"].iloc[60:].between(0, 100).all(), "RSI 应在 0~100"
    assert feat["tech_wr_14"].iloc[60:].between(0, 100).all(), "WR 应在 0~100"
    assert feat["tech_psy"].iloc[60:].between(0, 100).all(), "PSY 应在 0~100"
    print(f"[smoke] indicators 通过：计算 {len(expected)} 类指标，暖机后无缺失，取值范围正常")


if __name__ == "__main__":
    _smoke_test()
