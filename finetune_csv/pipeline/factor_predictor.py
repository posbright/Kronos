"""因子增强预测器 FactorPredictor。

为「方案 B」的 KronosWithFactor 提供与 KronosPredictor 等价的高层预测接口，
但在每一步自回归时把因子向量经 factor_emb 注入主干（与训练时一致）。

关键约定：
    - 价格按列 z-score（仅用 lookback 段统计，防泄漏），与 KronosPredictor 完全一致。
    - 因子同样按列 z-score（仅用 lookback 段统计）。
    - 推理时未来因子未知：将「最后一个已知因子」沿预测步保持不变（hold-last）。
      这是常见且保守的处理；若调用方能提供未来因子，可经 future_factor 传入覆盖。
    - factor_emb 零初始化时，本预测器输出与不带因子的基线预测在相同随机种子下一致
      （由冒烟自测验证）。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.kronos import sample_from_logits  # noqa: E402


@torch.no_grad()
def factor_auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, factor_seq,
                                     max_context, pred_len, clip=5.0, T=1.0,
                                     top_k=0, top_p=0.9, sample_count=1):
    """带因子条件的自回归推理（对照 model.kronos.auto_regressive_inference）。

    Args:
        factor_seq: [B, initial_seq_len + pred_len, k] 已归一化的因子序列（含未来 hold-last）。
    其余参数含义同基线推理。
    """
    x = torch.clip(x, -clip, clip)
    device = x.device

    # 复制多路并行采样。
    x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
    x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
    y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)
    factor_seq = factor_seq.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, factor_seq.size(1), factor_seq.size(2)).to(device)

    x_token = tokenizer.encode(x, half=True)
    initial_seq_len = x.size(1)
    batch_size = x_token[0].size(0)
    total_seq_len = initial_seq_len + pred_len
    full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

    generated_pre = x_token[0].new_empty(batch_size, pred_len)
    generated_post = x_token[1].new_empty(batch_size, pred_len)

    pre_buffer = x_token[0].new_zeros(batch_size, max_context)
    post_buffer = x_token[1].new_zeros(batch_size, max_context)
    buffer_len = min(initial_seq_len, max_context)
    if buffer_len > 0:
        start_idx = max(0, initial_seq_len - max_context)
        pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
        post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

    for i in range(pred_len):
        current_seq_len = initial_seq_len + i
        window_len = min(current_seq_len, max_context)
        if current_seq_len <= max_context:
            input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
        else:
            input_tokens = [pre_buffer, post_buffer]

        context_end = current_seq_len
        context_start = max(0, context_end - max_context)
        current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
        current_factor = factor_seq[:, context_start:context_end, :].contiguous()

        s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1],
                                             current_stamp, factor=current_factor)
        s1_logits = s1_logits[:, -1, :]
        sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

        s2_logits = model.decode_s2(context, sample_pre)
        s2_logits = s2_logits[:, -1, :]
        sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

        generated_pre[:, i] = sample_pre.squeeze(-1)
        generated_post[:, i] = sample_post.squeeze(-1)

        if current_seq_len < max_context:
            pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
            post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
        else:
            pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
            post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
            pre_buffer[:, -1] = sample_pre.squeeze(-1)
            post_buffer[:, -1] = sample_post.squeeze(-1)

    full_pre = torch.cat([x_token[0], generated_pre], dim=1)
    full_post = torch.cat([x_token[1], generated_post], dim=1)
    context_start = max(0, total_seq_len - max_context)
    input_tokens = [
        full_pre[:, context_start:total_seq_len].contiguous(),
        full_post[:, context_start:total_seq_len].contiguous(),
    ]
    z = tokenizer.decode(input_tokens, half=True)
    z = z.reshape(-1, sample_count, z.size(1), z.size(2))
    preds = z.cpu().numpy()
    preds = np.mean(preds, axis=1)
    return preds


class FactorPredictor:
    """KronosWithFactor 的高层预测器（接口与 KronosPredictor 对齐，额外接收因子）。"""

    PRICE_COLS = ["open", "high", "low", "close"]
    VOL_COL = "volume"
    AMT_COL = "amount"

    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5.0):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer.to(device).eval()
        self.max_context = max_context
        self.clip = clip

    @staticmethod
    def _calc_time_stamps(ts: pd.Series) -> np.ndarray:
        ts = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
        out = pd.DataFrame({
            "minute": ts.dt.minute, "hour": ts.dt.hour, "weekday": ts.dt.weekday,
            "day": ts.dt.day, "month": ts.dt.month,
        })
        return out.values.astype(np.float32)

    def predict(self, df: pd.DataFrame, x_timestamp, y_timestamp, pred_len: int,
                factor_hist: np.ndarray, future_factor: Optional[np.ndarray] = None,
                T: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                sample_count: int = 1) -> pd.DataFrame:
        """生成 pred_len 步的 OHLCV 预测。

        Args:
            df:           历史窗口，含 open/high/low/close（volume/amount 可缺省补 0）。
            factor_hist:  [lookback, k] 历史因子（与 df 行对齐）。
            future_factor:[pred_len, k] 可选的未来因子；None 时按 hold-last 处理。
        """
        df = df.copy()
        if self.VOL_COL not in df.columns:
            df[self.VOL_COL] = 0.0
            df[self.AMT_COL] = 0.0
        if self.AMT_COL not in df.columns:
            df[self.AMT_COL] = df[self.VOL_COL] * df[self.PRICE_COLS].mean(axis=1)

        cols = self.PRICE_COLS + [self.VOL_COL, self.AMT_COL]
        x = df[cols].values.astype(np.float32)
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

        # 因子归一化（仅用历史段统计），并拼接未来段（hold-last 或传入）。
        factor_hist = np.asarray(factor_hist, dtype=np.float32)
        f_mean, f_std = np.mean(factor_hist, axis=0), np.std(factor_hist, axis=0)
        f_hist_norm = np.clip((factor_hist - f_mean) / (f_std + 1e-5), -self.clip, self.clip)
        if future_factor is None:
            f_future = np.repeat(f_hist_norm[-1:], pred_len, axis=0)
        else:
            future_factor = np.asarray(future_factor, dtype=np.float32)
            f_future = np.clip((future_factor - f_mean) / (f_std + 1e-5), -self.clip, self.clip)
        factor_seq = np.concatenate([f_hist_norm, f_future], axis=0)  # [T+pred_len, k]

        x_stamp = self._calc_time_stamps(x_timestamp)
        y_stamp = self._calc_time_stamps(y_timestamp)

        xt = torch.from_numpy(x_norm[np.newaxis]).to(self.device)
        xst = torch.from_numpy(x_stamp[np.newaxis]).to(self.device)
        yst = torch.from_numpy(y_stamp[np.newaxis]).to(self.device)
        ft = torch.from_numpy(factor_seq[np.newaxis].astype(np.float32)).to(self.device)

        preds = factor_auto_regressive_inference(
            self.tokenizer, self.model, xt, xst, yst, ft,
            self.max_context, pred_len, self.clip, T, top_k, top_p, sample_count)
        preds = preds[:, -pred_len:, :].squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean
        return pd.DataFrame(preds, columns=cols, index=pd.Index(y_timestamp, name="timestamps"))


def _smoke_test() -> None:
    """无需下载权重的冒烟测试：

    用微型 tokenizer + 微型 KronosWithFactor（factor_emb 零初始化）验证：
    在相同随机种子下，FactorPredictor 的预测与基线 KronosPredictor 完全一致
    —— 即零初始化时因子旁路不改变预测（等价性），且输出形状/列正确。
    """
    from model.kronos import Kronos, KronosPredictor, KronosTokenizer
    from finetune_csv.factor_model import KronosWithFactor

    k = 4
    lookback, pred_len = 20, 5

    torch.manual_seed(0)
    tok = KronosTokenizer(
        d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=1.0, gamma0=1.0, gamma=1.0, zeta=1.0, group_size=4,
    ).eval()

    # 基线模型与因子模型共享同一套随机权重，便于严格对比。
    torch.manual_seed(1)
    base = Kronos(s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
                  ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
                  token_dropout_p=0.0, learn_te=True).eval()
    torch.manual_seed(1)
    fac = KronosWithFactor(s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
                           ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
                           token_dropout_p=0.0, learn_te=True).eval()
    fac.init_factor(k)  # factor_emb 零初始化
    fac.load_state_dict(base.state_dict(), strict=False)  # 主干权重与 base 对齐

    rng = np.random.default_rng(7)
    prices = 10 + np.cumsum(rng.normal(0, 0.1, size=(lookback, 1)), axis=0)
    df = pd.DataFrame({
        "open": prices[:, 0], "high": prices[:, 0] + 0.2,
        "low": prices[:, 0] - 0.2, "close": prices[:, 0] + 0.05,
        "volume": rng.uniform(1e3, 2e3, lookback), "amount": rng.uniform(1e4, 2e4, lookback),
    })
    x_ts = pd.Series(pd.date_range("2024-01-01", periods=lookback, freq="D"))
    y_ts = pd.Series(pd.date_range("2024-01-01", periods=lookback + pred_len, freq="D")[lookback:])
    factor_hist = rng.normal(0, 1, size=(lookback, k)).astype(np.float32)

    base_pred = KronosPredictor(base, tok, device="cpu", max_context=512)
    fac_pred = FactorPredictor(fac, tok, device="cpu", max_context=512)

    torch.manual_seed(123)
    out_base = base_pred.predict(df, x_ts, y_ts, pred_len, T=1.0, top_p=0.9,
                                 sample_count=1, verbose=False)
    torch.manual_seed(123)
    out_fac = fac_pred.predict(df, x_ts, y_ts, pred_len, factor_hist=factor_hist,
                               T=1.0, top_p=0.9, sample_count=1)

    assert out_fac.shape == out_base.shape, f"形状不一致: {out_fac.shape} vs {out_base.shape}"
    assert list(out_fac.columns) == list(out_base.columns), "列不一致"
    assert np.allclose(out_fac.values, out_base.values, atol=1e-5), \
        "零初始化时 FactorPredictor 应与基线 KronosPredictor 完全一致"
    print("[smoke] factor_predictor 通过：零初始化等价性成立，输出形状",
          tuple(out_fac.shape))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="因子增强预测器 FactorPredictor")
    parser.add_argument("--smoke", action="store_true", help="运行无需权重的冒烟自测")
    args = parser.parse_args()
    if args.smoke:
        _smoke_test()
    else:
        parser.print_help()
