"""Kronos 微调统一流水线工具包（Phase 1 基础设施）。

模块划分：
    - config        : 统一配置（PipelineConfig），所有 run_*.py 共用一份 YAML。
    - splits        : 滚动日期切分（以「当前日期-1」为锚点向前回推 test/val/train）。
    - factors       : 因子缺失值处理策略（防泄漏前向填充 + 中位数兜底 + 缺失掩码）。
    - logging_utils : 训练 / 构建过程的文件日志（模型目录、批次、收敛信息落盘）。

设计目标：配置统一、参数可解释、缺失值可控、过程可追溯，方便后期维护升级。
"""

from .config import PipelineConfig
from .splits import SplitPlan, rolling_date_splits, collect_trading_dates
from .factors import (
    analyze_missing,
    infer_factor_strategy,
    handle_factor_nulls,
)
from .indicators import compute_tech_indicators
from .kline_dataset import PreSplitKlineDataset
from .train_bridge import (
    build_trainer_config,
    build_loaders,
    build_eval_loader,
    resolve_device,
)
from .evaluate import evaluate_model
from .eval_runner import run_eval_stage
from .factor_predictor import FactorPredictor, factor_auto_regressive_inference
from .factor_dataset import FactorKlinePreSplitDataset
from .factor_train_runner import run_factor_train
from .logging_utils import setup_run_logger, TrainingLogger

__all__ = [
    "PipelineConfig",
    "SplitPlan",
    "rolling_date_splits",
    "collect_trading_dates",
    "analyze_missing",
    "infer_factor_strategy",
    "handle_factor_nulls",
    "compute_tech_indicators",
    "PreSplitKlineDataset",
    "build_trainer_config",
    "build_loaders",
    "build_eval_loader",
    "resolve_device",
    "evaluate_model",
    "run_eval_stage",
    "FactorPredictor",
    "factor_auto_regressive_inference",
    "FactorKlinePreSplitDataset",
    "run_factor_train",
    "setup_run_logger",
    "TrainingLogger",
]
