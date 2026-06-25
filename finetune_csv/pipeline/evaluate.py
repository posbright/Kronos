"""模型评估指标计算（验证 / 测试共用）。

在持出集（验证 / 测试）上前向计算两类指标，均与训练时的损失口径一致，便于直接对比：

    - tokenizer_recon_mse : tokenizer 重建 MSE（z-score 空间），衡量量化保真度。
    - predictor_loss      : 预测器下一 token 的交叉熵损失（s1+s2 合计），衡量自回归预测质量。

注意：评估全程 no_grad、模型 eval()，不更新任何参数，避免对持出集产生泄漏。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_model(tokenizer, model, loader, device) -> Dict[str, float]:
    """在给定 DataLoader 上计算 tokenizer 重建 MSE 与预测器下一 token 损失。

    Args:
        tokenizer: 已加载的 KronosTokenizer（finetuned 或预训练）。
        model:     已加载的 Kronos 预测器。
        loader:    确定性 eval DataLoader（PreSplitKlineDataset, role='eval'）。
        device:    计算设备。

    Returns:
        dict: tokenizer_recon_mse / predictor_loss / n_samples / n_batches。
    """
    tokenizer.eval()
    model.eval()

    recon_sum = 0.0
    pred_loss_sum = 0.0
    n_samples = 0
    n_batches = 0

    for batch_x, batch_x_stamp in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
        bsz = batch_x.size(0)

        # 1) tokenizer 重建 MSE（与 finetune_tokenizer 验证口径一致）。
        zs, _, _, _ = tokenizer(batch_x)
        _, z = zs
        recon_sum += F.mse_loss(z, batch_x).item() * bsz

        # 2) 预测器下一 token 损失（与 finetune_base_model 验证口径一致）。
        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
        token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
        token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
        logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
        loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
        pred_loss_sum += loss.item() * bsz

        n_samples += bsz
        n_batches += 1

    if n_samples == 0:
        return {"tokenizer_recon_mse": float("nan"), "predictor_loss": float("nan"),
                "n_samples": 0, "n_batches": 0}

    return {
        "tokenizer_recon_mse": recon_sum / n_samples,
        "predictor_loss": pred_loss_sum / n_samples,
        "n_samples": n_samples,
        "n_batches": n_batches,
    }
