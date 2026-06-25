"""验证 / 测试阶段的共用执行逻辑。

run_validate.py 与 run_test.py 仅是「阶段名 + 数据集」不同的薄封装，核心评估流程
集中在此，避免重复：定位某个训练版本产出的微调模型，在对应持出集上前向计算指标，
并把结果落盘到 runs/<exp>/<version>/<stage>/。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .config import PipelineConfig
from .evaluate import evaluate_model
from .train_bridge import build_eval_loader, resolve_device
from .logging_utils import setup_run_logger

# 阶段名 -> 数据集子目录。
_STAGE_DATASET = {"validate": "validation", "test": "test"}


def run_eval_stage(cfg: PipelineConfig, stage: str,
                   version: Optional[str] = None) -> Dict[str, object]:
    """在指定阶段（validate/test）上评估某训练版本的模型。

    Args:
        cfg:     统一配置。
        stage:   'validate' 或 'test'。
        version: 训练版本号；None 时使用配置版本或最近一次训练版本。

    Returns:
        指标与路径汇总字典（同时写入 <stage>/summary.json）。
    """
    assert stage in _STAGE_DATASET, f"stage 必须是 validate/test，收到 {stage}"

    # 对齐训练版本：显式 version > 配置 runs.version > 最近一次训练目录。
    if version:
        cfg.set_version(version)
    elif not cfg.get("runs.version", ""):
        latest = cfg.latest_version()
        if latest:
            cfg.set_version(latest)

    run_dir = cfg.run_dir(stage)
    logger = setup_run_logger(str(run_dir), f"kronos_{stage}")
    device = resolve_device(cfg)

    # 延迟导入，避免无谓的权重加载依赖。
    from model import Kronos, KronosTokenizer

    train_dir = cfg.runs_root / cfg.exp_name / cfg.version / "train"
    tok_path = train_dir / "tokenizer" / "best_model"
    model_path = train_dir / "basemodel" / "best_model"
    for p in (tok_path, model_path):
        if not p.exists():
            raise FileNotFoundError(
                f"未找到训练产出 {p}（version={cfg.version}）；请先运行 run_train.py。")

    csv_path = cfg.dataset_root / _STAGE_DATASET[stage] / "dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到数据集 {csv_path}；请先运行 run_build_dataset.py。")

    logger.info("=== %s 评估 ===", stage)
    logger.info("version=%s | device=%s", cfg.version, device)
    logger.info("tokenizer=%s", tok_path)
    logger.info("model=%s", model_path)
    logger.info("dataset=%s", csv_path)

    tokenizer = KronosTokenizer.from_pretrained(str(tok_path)).to(device)
    model = Kronos.from_pretrained(str(model_path)).to(device)
    loader, ds = build_eval_loader(cfg, str(csv_path))
    logger.info("eval 样本数=%d（lookback=%d predict=%d）",
                len(ds), cfg.lookback_window, cfg.predict_window)

    metrics = evaluate_model(tokenizer, model, loader, device)
    logger.info("指标：tokenizer_recon_mse=%.6f | predictor_loss=%.6f | n=%d",
                metrics["tokenizer_recon_mse"], metrics["predictor_loss"],
                metrics["n_samples"])

    result = {
        "stage": stage,
        "version": cfg.version,
        "device": str(device),
        "dataset": str(csv_path),
        "tokenizer_model": str(tok_path),
        "predictor_model": str(model_path),
        "metrics": metrics,
    }
    out = run_dir / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("已写出 %s", out)
    return result
