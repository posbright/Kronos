"""训练桥接层 train_bridge。

把统一配置 PipelineConfig 适配到既有训练函数
（finetune_tokenizer.train_tokenizer / finetune_base_model.train_model）所需的接口：

    1. build_trainer_config(cfg): 生成一个「带训练超参属性」的轻量配置对象，
       供训练函数直接读取（train_*.py 通过 external_loaders 注入数据，因此
       不需要 data_path / 比例切分等字段）。

    2. build_loaders(cfg, train_csv, val_csv): 用 PreSplitKlineDataset 从两份
       预切分 CSV 分别构建 train / val 的 DataLoader，返回训练函数 external_loaders
       所需的四元组 (train_loader, val_loader, train_dataset, val_dataset)。

    3. resolve_device(cfg): 统一设备选择（cuda -> cpu）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .config import PipelineConfig
from .kline_dataset import PreSplitKlineDataset


def build_trainer_config(cfg: PipelineConfig) -> SimpleNamespace:
    """从 PipelineConfig 抽出训练函数需要的超参，打包为轻量配置对象。

    注：train_tokenizer / train_model 在 external_loaders 模式下只读取超参，
    不读取 data_path / train_ratio 等切分字段，故此处无需提供。
    """
    return SimpleNamespace(
        tokenizer_learning_rate=cfg.tokenizer_learning_rate,
        predictor_learning_rate=cfg.predictor_learning_rate,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_weight_decay=cfg.adam_weight_decay,
        tokenizer_epochs=cfg.tokenizer_epochs,
        basemodel_epochs=cfg.basemodel_epochs,
        log_interval=cfg.log_interval,
        accumulation_steps=cfg.accumulation_steps,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )


def build_loaders(cfg: PipelineConfig, train_csv: str, val_csv: str
                  ) -> Tuple[DataLoader, DataLoader, PreSplitKlineDataset, PreSplitKlineDataset]:
    """从预切分的 train / val CSV 构建 DataLoader（供 external_loaders 注入）。"""
    train_ds = PreSplitKlineDataset(
        data_path=train_csv, role="train",
        lookback_window=cfg.lookback_window, predict_window=cfg.predict_window,
        clip=cfg.clip, seed=cfg.seed,
    )
    val_ds = PreSplitKlineDataset(
        data_path=val_csv, role="eval",
        lookback_window=cfg.lookback_window, predict_window=cfg.predict_window,
        clip=cfg.clip, seed=cfg.seed + 1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds


def build_eval_loader(cfg: PipelineConfig, csv_path: str) -> Tuple[DataLoader, PreSplitKlineDataset]:
    """从单份 CSV 构建确定性 eval DataLoader（验证 / 测试用）。"""
    ds = PreSplitKlineDataset(
        data_path=csv_path, role="eval",
        lookback_window=cfg.lookback_window, predict_window=cfg.predict_window,
        clip=cfg.clip, seed=cfg.seed,
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=False,
    )
    return loader, ds


def resolve_device(cfg: PipelineConfig) -> torch.device:
    """统一设备选择：优先 cuda（按 device_id），否则 cpu。"""
    if cfg.use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{cfg.device_id}")
    return torch.device("cpu")
