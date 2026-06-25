"""训练脚本（tokenizer + 预测器），与验证 / 测试脚本分离。

职责：仅做「训练」。读取统一配置，使用 Phase 1 产出的预切分训练 / 验证集
（DataSet/train、DataSet/validation），先微调 tokenizer 再微调预测器，全程把
「模型目录、微调输出目录、每个 epoch 的训练 / 验证损失、收敛 / 早停」落盘到
runs/<exp>/<version>/train/ 下，便于后期评估与版本对比。

与既有 train_sequential.py 的区别：
    - 训练 / 验证 / 测试三个阶段拆成三个独立脚本（本文件只负责训练）。
    - 训练用 DataSet/train，验证用 DataSet/validation（来自滚动日期切分，
      不再按比例二次切分），通过 external_loaders 注入既有训练函数。
    - 用 TrainingLogger 结构化记录 metrics.csv / summary.json（收敛曲线 + 早停判定）。

用法：
    # 真实训练（需可加载预训练权重；CPU/GPU 均可，CPU 较慢）
    python finetune_csv/run_train.py --config finetune_csv/configs/config_smoke20.yaml

    # 仅训练预测器（复用已存在的微调 tokenizer）
    python finetune_csv/run_train.py --config ... --skip-tokenizer --version 20260624_xxxx

    # 冒烟自测（合成数据，校验数据管线与按 symbol 取窗，不下载模型）
    python finetune_csv/run_train.py --smoke
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import (  # noqa: E402
    PipelineConfig,
    build_trainer_config,
    build_loaders,
    resolve_device,
    setup_run_logger,
    TrainingLogger,
)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_epoch_callback(tlog: TrainingLogger):
    """把每个 epoch 的 (train_loss, val_loss) 记入 TrainingLogger，并返回是否早停。"""
    def _cb(epoch: int, train_loss: float, val_loss: float) -> bool:
        tlog.log_epoch(epoch, train_loss, val_metric=val_loss)
        return tlog.should_stop()
    return _cb


def train(cfg: PipelineConfig, skip_tokenizer: bool = False,
          skip_basemodel: bool = False) -> dict:
    """执行训练：tokenizer 阶段 -> 预测器阶段。返回关键路径与最优指标汇总。"""
    # 延迟导入：仅真实训练时才需要（避免冒烟自测触发模型 / 权重加载）。
    from model import Kronos, KronosTokenizer
    from finetune_tokenizer import train_tokenizer
    from finetune_base_model import train_model

    run_dir = cfg.run_dir("train")
    logger = setup_run_logger(str(run_dir), "kronos_train")
    device = resolve_device(cfg)
    _set_all_seeds(cfg.seed)

    train_csv = cfg.dataset_root / "train" / "dataset.csv"
    val_csv = cfg.dataset_root / "validation" / "dataset.csv"
    for p in (train_csv, val_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"未找到数据集 {p}；请先运行 run_build_dataset.py 构建数据。")

    tok_dir = run_dir / "tokenizer"
    bm_dir = run_dir / "basemodel"
    tcfg = build_trainer_config(cfg)

    logger.info("=== 训练配置 ===")
    logger.info("device=%s | version=%s | exp=%s", device, cfg.version, cfg.exp_name)
    logger.info("lookback=%d predict=%d batch=%d | tok_epochs=%d bm_epochs=%d",
                cfg.lookback_window, cfg.predict_window, cfg.batch_size,
                cfg.tokenizer_epochs, cfg.basemodel_epochs)
    logger.info("train_csv=%s | val_csv=%s", train_csv, val_csv)

    summary = {"version": cfg.version, "device": str(device),
               "tokenizer_best_model": None, "basemodel_best_model": None}

    # ---------- 阶段 1：tokenizer 微调 ----------
    if not skip_tokenizer:
        logger.info("---- 阶段 1/2：tokenizer 微调 ----")
        tokenizer = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer).to(device)
        loaders = build_loaders(cfg, str(train_csv), str(val_csv))
        tlog = TrainingLogger(str(tok_dir), logger, patience=cfg.patience, higher_better=False)
        tlog.log_meta(stage="tokenizer", pretrained=cfg.pretrained_tokenizer,
                      out_dir=str(tok_dir / "best_model"), epochs=cfg.tokenizer_epochs,
                      batch_size=cfg.batch_size, lr=cfg.tokenizer_learning_rate,
                      train_csv=str(train_csv), val_csv=str(val_csv),
                      train_samples=len(loaders[2]), val_samples=len(loaders[3]))
        best = train_tokenizer(tokenizer, device, tcfg, str(tok_dir), logger,
                               external_loaders=loaders,
                               epoch_callback=_make_epoch_callback(tlog))
        tlog.finalize(extra={"best_val_loss_returned": best})
        summary["tokenizer_best_model"] = str(tok_dir / "best_model")
    else:
        logger.info("跳过 tokenizer 微调（--skip-tokenizer）。")

    finetuned_tok = tok_dir / "best_model"

    # ---------- 阶段 2：预测器微调 ----------
    if not skip_basemodel:
        logger.info("---- 阶段 2/2：预测器微调 ----")
        if not finetuned_tok.exists():
            raise FileNotFoundError(
                f"未找到微调后的 tokenizer：{finetuned_tok}；请先训练 tokenizer 或去掉 --skip-tokenizer。")
        tokenizer = KronosTokenizer.from_pretrained(str(finetuned_tok)).to(device)
        model = Kronos.from_pretrained(cfg.pretrained_predictor).to(device)
        loaders = build_loaders(cfg, str(train_csv), str(val_csv))
        blog = TrainingLogger(str(bm_dir), logger, patience=cfg.patience, higher_better=False)
        blog.log_meta(stage="basemodel", pretrained_predictor=cfg.pretrained_predictor,
                      finetuned_tokenizer=str(finetuned_tok),
                      out_dir=str(bm_dir / "best_model"), epochs=cfg.basemodel_epochs,
                      batch_size=cfg.batch_size, lr=cfg.predictor_learning_rate,
                      train_csv=str(train_csv), val_csv=str(val_csv),
                      train_samples=len(loaders[2]), val_samples=len(loaders[3]))
        best = train_model(model, tokenizer, device, tcfg, str(bm_dir), logger,
                           external_loaders=loaders,
                           epoch_callback=_make_epoch_callback(blog))
        blog.finalize(extra={"best_val_loss_returned": best})
        summary["basemodel_best_model"] = str(bm_dir / "best_model")
    else:
        logger.info("跳过预测器微调（--skip-basemodel）。")

    logger.info("训练完成。产出根目录：%s", run_dir)
    return summary


def run_smoke() -> None:
    """冒烟自测：用合成多股票数据校验数据管线与「按 symbol 取窗、不跨标的」。不下载模型。"""
    from pipeline import PreSplitKlineDataset

    lookback, predict = 16, 4
    window = lookback + predict + 1
    rng = np.random.default_rng(0)

    def _mk(sym: str, n: int, base: float) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = base + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame({
            "date": dates, "symbol": sym,
            "open": close + rng.normal(0, 0.1, n),
            "high": close + abs(rng.normal(0, 0.3, n)),
            "low": close - abs(rng.normal(0, 0.3, n)),
            "close": close,
            "volume": rng.integers(1e5, 1e6, n).astype(float),
            "amount": rng.integers(1e7, 1e8, n).astype(float),
        })

    # 两只股票：一只 60 行（>=window），一只 30 行；验证 valid_starts 数量正确。
    df = pd.concat([_mk("000001", 60, 10.0), _mk("000002", 30, 20.0)], ignore_index=True)
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "dataset.csv")
        df.to_csv(csv, index=False)
        ds = PreSplitKlineDataset(csv, role="train", lookback_window=lookback,
                                  predict_window=predict, clip=5.0, seed=0)
        # 期望合法起点数 = (60-window+1) + (30-window+1)。
        exp = (60 - window + 1) + (30 - window + 1)
        assert ds.n_samples == exp, f"valid_starts 期望 {exp}，实际 {ds.n_samples}"
        x, x_stamp = ds[0]
        assert tuple(x.shape) == (window, 6), f"x 形状应为 ({window},6)，实际 {tuple(x.shape)}"
        assert tuple(x_stamp.shape) == (window, 5)
        assert torch.isfinite(x).all(), "归一化后存在非有限值"
        # eval role 确定性。
        ds_eval = PreSplitKlineDataset(csv, role="eval", lookback_window=lookback,
                                       predict_window=predict, clip=5.0, seed=0)
        assert torch.equal(ds_eval[0][0], ds_eval[0][0])
    print(f"[smoke] run_train 通过：2 只股票合法窗口数={exp}，按 symbol 取窗、形状/有限性正常")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos 训练（tokenizer + 预测器），与验证/测试分离")
    parser.add_argument("--config", type=str,
                        default=str(_THIS_DIR / "configs" / "config_smoke20.yaml"),
                        help="统一配置 YAML 路径")
    parser.add_argument("--version", type=str, default="",
                        help="运行版本号（留空=配置或时间戳）；指定后产出落入该版本目录")
    parser.add_argument("--skip-tokenizer", action="store_true", help="跳过 tokenizer 微调")
    parser.add_argument("--skip-basemodel", action="store_true", help="跳过预测器微调")
    parser.add_argument("--smoke", action="store_true", help="合成数据冒烟自测（不下载模型）")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    cfg = PipelineConfig(args.config)
    if args.version:
        cfg.set_version(args.version)
    summary = train(cfg, skip_tokenizer=args.skip_tokenizer, skip_basemodel=args.skip_basemodel)
    print("训练汇总：", summary)


if __name__ == "__main__":
    main()
