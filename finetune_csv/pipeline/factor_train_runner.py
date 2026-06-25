"""因子模型版本化训练 run_factor_train（Phase 3 后端核心）。

职责：按「用户选定的因子列 + 每因子缩放权重」在版本化目录下微调 KronosWithFactor，
全程落盘（日志 / 指标 / 最优模型 / 因子配置 / 因子重要性），供独立 App 调权重重训、
记录版本并横向对比。

数据：复用 DataSet/{train,validation}（多股票预切分 CSV，含 tech_*/fin_* 因子列），
按 FactorKlinePreSplitDataset 做 symbol-aware 取窗 + 逐窗口价格 z-score + 因子 lookback
段 z-score 再乘以每因子权重。

底座：tokenizer 冻结只编码；KronosWithFactor 从「指定基线版本的 basemodel」或
「配置里的预训练主模型」加载，factor_emb 零初始化后开始学因子边际贡献。

产出目录：runs/<exp>/<version>/train/factor/
    run.log / metrics.csv / run_meta.json / summary.json / best_model/ / factor_config.json
    factor_importance.json（每因子 L2 重要性占比，训练后从 factor_emb 列范数算）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PipelineConfig
from .factor_dataset import FactorKlinePreSplitDataset
from .logging_utils import TrainingLogger, setup_run_logger
from .train_bridge import resolve_device

# 训练进度回调签名：progress_cb(stage, epoch, total_epochs, train_loss, val_loss)。
ProgressCb = Callable[[str, int, int, float, Optional[float]], None]


def _build_factor_loaders(cfg: PipelineConfig, factor_cols: Sequence[str],
                          factor_weights: Dict[str, float],
                          factor_specs: Optional[Sequence[dict]] = None):
    """构建 train/val 因子数据加载器（缺验证集时返回 (loader, None)）。"""
    train_csv = cfg.dataset_root / "train" / "dataset.csv"
    val_csv = cfg.dataset_root / "validation" / "dataset.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"未找到训练集 {train_csv}；请先构建 DataSet。")

    train_ds = FactorKlinePreSplitDataset(
        str(train_csv), factor_cols, factor_weights, role="train",
        lookback_window=cfg.lookback_window, predict_window=cfg.predict_window,
        clip=cfg.clip, seed=cfg.seed, factor_specs=factor_specs)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, num_workers=cfg.num_workers)

    val_loader = None
    if val_csv.exists():
        val_ds = FactorKlinePreSplitDataset(
            str(val_csv), factor_cols, factor_weights, role="eval",
            lookback_window=cfg.lookback_window, predict_window=cfg.predict_window,
            clip=cfg.clip, seed=cfg.seed, factor_specs=factor_specs)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                drop_last=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, train_ds


def _factor_importance(model, factor_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """训练后从 factor_emb 列范数计算每因子重要性占比（与 viz.factor_weights 一致）。"""
    w = model.factor_emb.weight.detach().cpu().numpy()  # [d_model, k]
    l2 = np.linalg.norm(w, axis=0)
    total = float(l2.sum()) or 1.0
    items = sorted(zip(factor_cols, l2), key=lambda t: t[1], reverse=True)
    return {n: {"l2": float(v), "share": float(v) / total} for n, v in items}


def run_factor_train(cfg: PipelineConfig, factor_cols: Sequence[str],
                     factor_weights: Optional[Dict[str, float]] = None,
                     version: Optional[str] = None,
                     base_version: Optional[str] = None,
                     epochs: Optional[int] = None,
                     progress_cb: Optional[ProgressCb] = None,
                     factor_specs: Optional[Sequence[dict]] = None) -> Dict[str, object]:
    """在版本化目录下微调因子模型并落盘。

    Args:
        cfg:            统一配置。
        factor_cols:    参与训练的因子列名（factor_specs 提供时可留空，由通道名推导）。
        factor_weights: {因子名: 缩放权重}；缺省全 1.0，0 等价关闭。
        version:        目标版本号；None 时用 cfg.version（新建进程取时间戳）。
        base_version:   作为底座的已训练版本（用其 tokenizer + basemodel）；
                        None 时用 cfg.pretrained_tokenizer / cfg.pretrained_predictor。
        epochs:         训练轮数；None 用 cfg.basemodel_epochs。
        progress_cb:    每个 epoch 结束后回调，供 App 上报进度。
        factor_specs:   可选的「同类聚合」通道规格（见 factor_dataset.normalize_specs）；
                        提供时每个通道作为一个因子参与训练，factor_cols 退化为通道名。

    Returns:
        汇总字典（同时写入 train/factor/summary.json）。
    """
    from .factor_dataset import normalize_specs

    factor_weights = dict(factor_weights or {})
    # 统一为规范通道：通道名即训练时的「因子」标签（importance/config 都用它）。
    specs = normalize_specs(factor_cols, factor_weights, factor_specs)
    factor_cols = [sp["name"] for sp in specs]
    spec_weights = {sp["name"]: sp["weight"] for sp in specs}
    epochs = int(epochs or cfg.basemodel_epochs)
    if version:
        cfg.set_version(version)

    run_dir = cfg.runs_root / cfg.exp_name / cfg.version / "train" / "factor"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_dir = run_dir / "best_model"
    logger = setup_run_logger(str(run_dir), "kronos_factor_train")
    device = resolve_device(cfg)

    # 延迟导入，避免无谓依赖。
    from model import KronosTokenizer
    from finetune_csv.factor_model import KronosWithFactor

    # 底座来源：基线版本 > 配置预训练。
    if base_version:
        base_train = cfg.runs_root / cfg.exp_name / base_version / "train"
        tok_src = str(base_train / "tokenizer" / "best_model")
        pred_src = str(base_train / "basemodel" / "best_model")
    else:
        tok_src = cfg.pretrained_tokenizer
        pred_src = cfg.pretrained_predictor

    logger.info("=== 因子模型训练 ===")
    logger.info("version=%s | device=%s | epochs=%d", cfg.version, device, epochs)
    logger.info("factor_cols(通道)=%s", factor_cols)
    logger.info("factor_weights=%s", spec_weights)
    logger.info("tokenizer=%s | predictor=%s", tok_src, pred_src)

    tokenizer = KronosTokenizer.from_pretrained(tok_src).to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    model = KronosWithFactor.from_pretrained(pred_src)
    model.init_factor(len(factor_cols))
    model = model.to(device)

    train_loader, val_loader, train_ds = _build_factor_loaders(
        cfg, factor_cols, spec_weights, factor_specs=specs)
    logger.info("train 样本=%d | val 样本=%s | lookback=%d predict=%d",
                len(train_ds), "无" if val_loader is None else len(val_loader.dataset),
                cfg.lookback_window, cfg.predict_window)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.predictor_learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.adam_weight_decay)

    tlog = TrainingLogger(str(run_dir), logger=logger, patience=cfg.patience,
                          higher_better=False)
    tlog.log_meta(stage="factor", version=cfg.version, device=str(device),
                  epochs=epochs, factor_cols=factor_cols,
                  factor_weights=spec_weights,
                  tokenizer_src=tok_src, predictor_src=pred_src,
                  lookback=cfg.lookback_window, predict=cfg.predict_window,
                  batch_size=cfg.batch_size, lr=cfg.predictor_learning_rate)

    def _run_batch(batch, train: bool) -> float:
        batch_x, batch_stamp, batch_factor = (t.to(device) for t in batch)
        with torch.no_grad():
            s0, s1 = tokenizer.encode(batch_x, half=True)
        token_in = [s0[:, :-1], s1[:, :-1]]
        token_out = [s0[:, 1:], s1[:, 1:]]
        logits = model(token_in[0], token_in[1],
                       stamp=batch_stamp[:, :-1, :],
                       factor=batch_factor[:, :-1, :])
        loss, _, _ = model.head.compute_loss(logits[0], logits[1],
                                             token_out[0], token_out[1])
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        return float(loss.item())

    for epoch in range(epochs):
        if hasattr(train_ds, "set_epoch_seed"):
            train_ds.set_epoch_seed(epoch)
        model.train()
        tr_losses = [_run_batch(b, True) for b in train_loader]
        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")

        va: Optional[float] = None
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            with torch.no_grad():
                va = float(np.mean([_run_batch(b, False) for b in val_loader]))

        improved = tlog.log_epoch(epoch + 1, tr, va)
        if improved:
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
        if progress_cb is not None:
            progress_cb("factor", epoch + 1, epochs, tr, va)
        if tlog.should_stop():
            logger.info("早停触发（连续 %d 轮无提升）。", cfg.patience)
            break

    # 若全程未触发 improved（极端情况），至少保存末轮权重。
    if not best_dir.exists():
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(best_dir))

    # 因子配置与重要性落盘（App 对比用）。
    importance = _factor_importance(model, factor_cols)
    factor_config = {
        "factor_cols": factor_cols,
        "factor_weights": spec_weights,
        "factor_dim": len(factor_cols),
        "factor_specs": specs,
    }
    with open(best_dir / "factor_config.json", "w", encoding="utf-8") as f:
        json.dump(factor_config, f, ensure_ascii=False, indent=2)
    with open(run_dir / "factor_importance.json", "w", encoding="utf-8") as f:
        json.dump(importance, f, ensure_ascii=False, indent=2)

    summary = tlog.finalize(status="completed", extra={
        "stage": "factor",
        "version": cfg.version,
        "best_model": str(best_dir),
        "factor_config": factor_config,
        "factor_importance": importance,
    })
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="因子模型版本化训练（Phase 3 后端）")
    parser.add_argument("--smoke", action="store_true", help="合成数据 + 微型模型冒烟自测")
    args = parser.parse_args()
    if not args.smoke:
        parser.print_help()
        raise SystemExit(0)

    # 冒烟：合成多股票 DataSet + 微型 tokenizer/模型，跑 1 epoch 验证版本化产出。
    import os
    import sys
    import tempfile

    import pandas as pd

    _THIS = Path(__file__).resolve()
    sys.path.insert(0, str(_THIS.parents[2]))  # repo root
    sys.path.insert(0, str(_THIS.parents[1]))  # finetune_csv

    from finetune_csv.factor_model import KronosWithFactor  # noqa: E402
    from model import KronosTokenizer  # noqa: E402

    tmp = Path(tempfile.mkdtemp(prefix="kronos_factor_train_smoke_"))
    ds_root = tmp / "DataSet"
    rng = np.random.default_rng(0)
    for split in ("train", "validation"):
        rows = []
        for sym in ("000001", "600000"):
            n = 90
            base = 10 + np.cumsum(rng.normal(0, 0.1, n))
            rows.append(pd.DataFrame({
                "symbol": sym,
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "open": base, "high": base + 0.2, "low": base - 0.2, "close": base + 0.05,
                "volume": rng.uniform(1e5, 1e6, n), "amount": rng.uniform(1e6, 1e7, n),
                "tech_macd": rng.normal(0, 1, n), "tech_rsi": rng.normal(0, 1, n),
            }))
        d = ds_root / split
        d.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(d / "dataset.csv", index=False)

    # 写一个最小 YAML 配置，指向微型底座（用本地构造的微型权重目录）。
    tok_dir = tmp / "tiny_tok"
    pred_dir = tmp / "tiny_pred"
    torch.manual_seed(0)
    KronosTokenizer(d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2,
                    n_dec_layers=2, ffn_dropout_p=0.0, attn_dropout_p=0.0,
                    resid_dropout_p=0.0, s1_bits=4, s2_bits=4, beta=1.0, gamma0=1.0,
                    gamma=1.0, zeta=1.0, group_size=4).save_pretrained(str(tok_dir))
    KronosWithFactor(s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
                     ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
                     token_dropout_p=0.0, learn_te=True).save_pretrained(str(pred_dir))

    yaml_text = f"""
model_paths:
  exp_name: kronos_factor_smoke
  pretrained_tokenizer: {tok_dir.as_posix()}
  pretrained_predictor: {pred_dir.as_posix()}
data:
  out_root: {ds_root.as_posix()}
  lookback_window: 30
  predict_window: 5
training:
  basemodel_epochs: 1
  batch_size: 4
  num_workers: 0
  patience: 5
device:
  use_cuda: false
runs:
  runs_root: {(tmp / 'runs').as_posix()}
"""
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")

    cfg = PipelineConfig(str(cfg_path))
    cfg.set_version("smoke_v1")
    res = run_factor_train(cfg, ["tech_macd", "tech_rsi"],
                           {"tech_macd": 1.0, "tech_rsi": 0.0}, version="smoke_v1")
    best = Path(res["best_model"])
    assert (best / "factor_config.json").exists(), "缺 factor_config.json"
    assert (best.parent / "factor_importance.json").exists(), "缺 factor_importance.json"
    print("[smoke] factor_train_runner 通过：")
    print("  best_value=", res["best_value"], "best_epoch=", res["best_epoch"])
    print("  importance=", {k: round(v["share"], 3)
                            for k, v in res["factor_importance"].items()})
