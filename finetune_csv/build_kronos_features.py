"""用 Kronos 批量生成「预测衍生特征」，供下游融合模型使用（方案 C）。

中文说明：
    把 Kronos 当作纯价量预测器，对每个时间点用历史窗口做多次采样预测，统计出三类衍生特征：
      - k_pred_ret ：预测的 N 步收益率均值（多条采样路径的均值）
      - k_up_prob  ：上涨概率（多条路径中收益 > 0 的比例）
      - k_pred_vol ：预测不确定性（多条路径末值收益率的标准差）
    输出一张「每个交易日一行」的特征表，后续可与基本面 / 消息面因子对齐成融合数据集。

用法：
    # 真实使用
    python build_kronos_features.py \
        --price-csv finetune_csv/data/A_000001_daily.csv \
        --tokenizer pretrained/Kronos-Tokenizer-base \
        --predictor pretrained/Kronos-base \
        --out data/kronos_features_000001.csv \
        --symbol 000001 --lookback 90 --pred 5 --samples 30

    # 冒烟自测（无需任何预训练权重，自动用微型模型 + 合成数据跑通全流程）
    python build_kronos_features.py --smoke
"""

import argparse
import os
import sys
import tempfile

# 确保可以 `from model import ...`（本文件位于 finetune_csv/ 下）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

_PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def build_features(predictor: KronosPredictor, px: pd.DataFrame, symbol: str,
                   lookback: int, pred_len: int, samples: int,
                   verbose: bool = False) -> pd.DataFrame:
    """对价格表逐窗口生成 Kronos 衍生特征，返回特征 DataFrame。

    Args:
        predictor: 已构造好的 KronosPredictor。
        px: 含 timestamps + OHLCV(amount) 的价格表，timestamps 为 datetime。
        symbol: 标的代码（写入输出列）。
        lookback / pred_len / samples: 历史窗口、预测步数、每窗采样次数。
    """
    if "timestamps" not in px.columns:
        raise ValueError("price 表必须包含 'timestamps' 列")
    missing = [c for c in _PRICE_COLS if c not in px.columns]
    if missing:
        raise ValueError(f"price 表缺少列: {missing}")

    px = px.copy()
    px["timestamps"] = pd.to_datetime(px["timestamps"])
    px = px.sort_values("timestamps").reset_index(drop=True)

    rows = []
    # 含最后一个可用窗口：end 取到 len(px) - pred_len（range 上界需 +1）。
    for end in range(lookback, len(px) - pred_len + 1):
        hist = px.iloc[end - lookback:end]
        x_df = hist[_PRICE_COLS].reset_index(drop=True)
        x_ts = hist["timestamps"].reset_index(drop=True)
        y_ts = px["timestamps"].iloc[end:end + pred_len].reset_index(drop=True)

        last_close = float(x_df["close"].iloc[-1])
        preds = []
        for _ in range(samples):
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False,
            )
            preds.append(pred_df["close"].values)
        preds = np.asarray(preds, dtype=np.float64)        # [samples, pred_len]

        end_ret = preds[:, -1] / last_close - 1.0          # 每条路径的 N 步收益
        rows.append({
            "date": px["timestamps"].iloc[end].normalize(),
            "symbol": symbol,
            "k_pred_ret": float(end_ret.mean()),
            "k_up_prob": float((end_ret > 0).mean()),
            "k_pred_vol": float(np.std(end_ret)),
        })
        if verbose and (len(rows) % 20 == 0):
            print(f"  processed {len(rows)} windows ...")

    return pd.DataFrame(rows)


def _build_tiny_predictor() -> KronosPredictor:
    """构造微型 tokenizer + 主模型组成的 predictor，仅用于冒烟自测。"""
    torch.manual_seed(0)
    tok = KronosTokenizer(
        d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=1.0, gamma0=1.0, gamma=1.0, zeta=1.0, group_size=4,
    ).eval()
    mdl = Kronos(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    ).eval()
    return KronosPredictor(mdl, tok, device="cpu", max_context=64)


def _synth_price(n: int = 90, seed: int = 0) -> pd.DataFrame:
    """生成一段合成价格序列，仅用于冒烟自测。"""
    rng = np.random.default_rng(seed)
    base = 10 + np.cumsum(rng.standard_normal(n) * 0.1)
    return pd.DataFrame({
        "timestamps": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": base, "high": base + 0.2, "low": base - 0.2, "close": base + 0.05,
        "volume": rng.integers(1e5, 1e6, n).astype(float), "amount": 0.0,
    })


def _smoke_test() -> None:
    """无需真实权重的端到端冒烟测试。"""
    predictor = _build_tiny_predictor()
    px = _synth_price(n=80)
    feats = build_features(predictor, px, symbol="TEST",
                           lookback=60, pred_len=5, samples=3)
    assert len(feats) == 80 - 60 - 5 + 1, f"窗口数异常: {len(feats)}"
    assert list(feats.columns) == ["date", "symbol", "k_pred_ret", "k_up_prob", "k_pred_vol"]
    assert feats[["k_pred_ret", "k_up_prob", "k_pred_vol"]].notnull().all().all()
    assert ((feats["k_up_prob"] >= 0) & (feats["k_up_prob"] <= 1)).all()

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "kronos_features_smoke.csv")
        feats.to_csv(out, index=False)
        assert os.path.exists(out)
    print(f"[smoke] build_kronos_features 通过：生成 {len(feats)} 行特征，列与取值范围均正常")
    print(feats.head(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Kronos 批量生成预测衍生特征（方案 C）")
    parser.add_argument("--price-csv", help="价格 CSV（timestamps + OHLCV + amount）")
    parser.add_argument("--tokenizer", help="预训练 / 微调后的 tokenizer 目录")
    parser.add_argument("--predictor", help="预训练 / 微调后的主模型目录")
    parser.add_argument("--out", help="输出特征 CSV 路径")
    parser.add_argument("--symbol", default="UNKNOWN", help="标的代码")
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--pred", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--device", default=None, help="cpu / cuda:0 等，默认自动选择")
    parser.add_argument("--smoke", action="store_true", help="运行无需权重的冒烟自测")
    args = parser.parse_args()

    if args.smoke:
        _smoke_test()
        return

    required = [args.price_csv, args.tokenizer, args.predictor, args.out]
    if any(v is None for v in required):
        parser.error("非 --smoke 模式下必须提供 --price-csv --tokenizer --predictor --out")

    tok = KronosTokenizer.from_pretrained(args.tokenizer)
    mdl = Kronos.from_pretrained(args.predictor)
    predictor = KronosPredictor(mdl, tok, device=args.device, max_context=512)

    px = pd.read_csv(args.price_csv)
    feats = build_features(predictor, px, symbol=args.symbol,
                           lookback=args.lookback, pred_len=args.pred,
                           samples=args.samples, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    feats.to_csv(args.out, index=False)
    print(f"[build_kronos_features] 已保存 {len(feats)} 行特征到 {args.out}")


if __name__ == "__main__":
    main()
