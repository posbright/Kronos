"""训练 / 构建过程的文件日志工具。

需求：把「模型目录、微调后模型目录、训练批次、收敛程度」等过程信息落盘成日志文件，
方便后期评估与排查。提供两类工具：

    1. setup_run_logger(run_dir)
       同时输出到控制台和 `run_dir/run.log` 的标准 logger。

    2. TrainingLogger
       结构化记录训练过程：
         - 每个 epoch 的 train/val 指标 -> 追加写入 `metrics.csv`（便于画收敛曲线）。
         - 收敛追踪：维护 best_val、best_epoch、距上次提升的 epoch 数（early-stop 信号）。
         - 关键路径与超参 -> 写入 `run_meta.json`（含模型目录、微调输出目录等）。
         - 训练结束 -> 写 `summary.json`（最优指标、是否收敛、总耗时）。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional


def setup_run_logger(run_dir: str, name: str = "kronos_pipeline",
                     level: int = logging.INFO) -> logging.Logger:
    """创建同时输出到控制台与 run.log 的 logger。

    Args:
        run_dir: 运行输出目录（不存在则创建），日志写入 run_dir/run.log。
        name:    logger 名称。
        level:   日志级别。

    Returns:
        配置好的 logging.Logger（重复调用不会重复添加 handler）。
    """
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    log_path = os.path.join(run_dir, "run.log")
    # 避免重复添加同一文件 / 控制台 handler。
    existing_files = {
        getattr(h, "baseFilename", None) for h in logger.handlers
        if isinstance(h, logging.FileHandler)
    }
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if os.path.abspath(log_path) not in {os.path.abspath(p) for p in existing_files if p}:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


class TrainingLogger:
    """结构化训练日志：指标、收敛、关键路径落盘。

    参数：
        run_dir:        运行输出目录。
        logger:         可选的 logging.Logger；None 时自动创建。
        patience:       连续多少个 epoch 验证指标无提升即判定「已收敛/可早停」。
                        常用 5~10；过小易误判，过大浪费算力。
        higher_better:  指标是否越大越好（loss=False，IC/准确率=True）。
    """

    def __init__(self, run_dir: str, logger: Optional[logging.Logger] = None,
                 patience: int = 8, higher_better: bool = False):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.logger = logger or setup_run_logger(run_dir)
        self.patience = patience
        self.higher_better = higher_better

        self.metrics_path = os.path.join(run_dir, "metrics.csv")
        self.meta_path = os.path.join(run_dir, "run_meta.json")
        self.summary_path = os.path.join(run_dir, "summary.json")

        self.history: List[Dict[str, Any]] = []
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.epochs_since_improve: int = 0
        self._t0 = time.time()
        self._meta: Dict[str, Any] = {}

        # 初始化 metrics.csv 头部（若不存在）。
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,val_metric,best_value,best_epoch,seconds\n")

    def log_meta(self, **kwargs: Any) -> None:
        """记录关键路径 / 超参（模型目录、微调输出目录、batch_size、lr 等）到 run_meta.json。"""
        self._meta.update(kwargs)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        # 同时打印到日志，便于快速查看。
        for k, v in kwargs.items():
            self.logger.info("meta | %s = %s", k, v)

    def log_epoch(self, epoch: int, train_loss: float,
                  val_metric: Optional[float] = None,
                  extra: Optional[Dict[str, Any]] = None) -> bool:
        """记录一个 epoch 的指标，更新收敛状态。

        Returns:
            improved: 本 epoch 验证指标是否较历史最优有提升（可据此保存 best_model）。
        """
        improved = False
        cmp_value = val_metric if val_metric is not None else train_loss
        if cmp_value is not None and not _is_nan(cmp_value):
            if self.best_value is None or _better(cmp_value, self.best_value, self.higher_better):
                self.best_value = float(cmp_value)
                self.best_epoch = epoch
                self.epochs_since_improve = 0
                improved = True
            else:
                self.epochs_since_improve += 1

        elapsed = round(time.time() - self._t0, 2)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metric": val_metric,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_since_improve": self.epochs_since_improve,
            "seconds": elapsed,
        }
        if extra:
            rec.update(extra)
        self.history.append(rec)

        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss},{val_metric},{self.best_value},"
                    f"{self.best_epoch},{elapsed}\n")

        self.logger.info(
            "epoch %d | train_loss=%.6f | val=%s | best=%s@%s | no_improve=%d",
            epoch, train_loss,
            "nan" if val_metric is None else f"{val_metric:.6f}",
            "nan" if self.best_value is None else f"{self.best_value:.6f}",
            self.best_epoch, self.epochs_since_improve,
        )
        return improved

    def should_stop(self) -> bool:
        """是否触发早停（连续 patience 个 epoch 无提升）。"""
        return self.epochs_since_improve >= self.patience

    def finalize(self, status: str = "completed",
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """写出 summary.json，返回汇总字典。"""
        converged = self.best_epoch is not None and (
            self.epochs_since_improve >= self.patience
            or (self.history and self.history[-1]["epoch"] == self.best_epoch)
        )
        summary = {
            "status": status,
            "epochs_run": len(self.history),
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_since_improve": self.epochs_since_improve,
            "patience": self.patience,
            "converged_or_early_stopped": bool(converged),
            "total_seconds": round(time.time() - self._t0, 2),
            "meta": self._meta,
        }
        if extra:
            summary.update(extra)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.logger.info("summary | %s", json.dumps(summary, ensure_ascii=False))
        return summary


def _is_nan(x: float) -> bool:
    try:
        return x != x
    except Exception:
        return False


def _better(new: float, best: float, higher_better: bool) -> bool:
    return new > best if higher_better else new < best
