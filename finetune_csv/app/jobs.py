"""后台训练任务管理器 JobManager（Phase 3）。

因子重训是「完整、长耗时」的训练，必须放后台执行、可查询进度、不阻塞前端。本模块用
单工作线程 + 队列串行执行（避免多份重训争抢 CPU/显存），并按版本号落盘产出：

    - submit(...) 立即返回 job_id 与新版本号；任务进入队列。
    - 工作线程逐个执行 run_factor_train，逐 epoch 更新进度。
    - 训练完成后顺带生成「因子权重分配图」，写入 runs/<exp>/<version>/viz/。
    - 状态：queued -> running -> completed / failed，全程线程安全。

注意：每个任务用「独立的 PipelineConfig 实例」并显式 set_version，确保版本目录互不串扰。
"""

from __future__ import annotations

import datetime as dt
import threading
import traceback
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, List, Optional


@dataclass
class Job:
    """一次因子重训任务的状态快照。"""
    job_id: str
    version: str
    factor_cols: List[str]
    factor_weights: Dict[str, float]
    base_version: Optional[str]
    epochs: Optional[int]
    factor_specs: Optional[List[dict]] = None    # 同类聚合通道规格（可选）
    status: str = "queued"                 # queued / running / completed / failed
    progress: Dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id, "version": self.version,
            "factor_cols": self.factor_cols, "factor_weights": self.factor_weights,
            "base_version": self.base_version, "epochs": self.epochs,
            "factor_specs": self.factor_specs,
            "status": self.status, "progress": self.progress, "error": self.error,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    """串行后台训练任务管理（单工作线程）。"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._queue: "Queue[str]" = Queue()
        self._counter = 0
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    # ---- 提交 / 查询 ----
    def submit(self, factor_cols: List[str], factor_weights: Dict[str, float],
               base_version: Optional[str] = None,
               epochs: Optional[int] = None,
               factor_specs: Optional[List[dict]] = None) -> Job:
        """创建并入队一个重训任务，立即返回（不阻塞）。"""
        if not factor_cols and not factor_specs:
            raise ValueError("factor_cols 与 factor_specs 不能同时为空")
        with self._lock:
            self._counter += 1
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"{ts}_{self._counter:03d}"
            job_id = version
            job = Job(job_id=job_id, version=version, factor_cols=list(factor_cols or []),
                      factor_weights=dict(factor_weights or {}),
                      base_version=base_version, epochs=epochs,
                      factor_specs=factor_specs)
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> List[Dict[str, object]]:
        with self._lock:
            return [self._jobs[j].to_dict() for j in reversed(self._order)]

    # ---- 工作线程 ----
    def _run_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                self._queue.task_done()
                continue
            try:
                self._execute(job)
            except Exception as exc:  # noqa: BLE001 — 后台线程需吞掉异常并记录
                with self._lock:
                    job.status = "failed"
                    job.error = f"{exc}\n{traceback.format_exc()}"
                    job.finished_at = dt.datetime.now().isoformat(timespec="seconds")
            finally:
                self._queue.task_done()

    def _execute(self, job: Job) -> None:
        # 延迟导入，避免 App 启动即加载重型依赖。
        from pipeline import PipelineConfig, run_factor_train
        from viz import plot_factor_weights

        with self._lock:
            job.status = "running"
            job.started_at = dt.datetime.now().isoformat(timespec="seconds")
            job.progress = {"epoch": 0, "total": job.epochs, "train_loss": None,
                            "val_loss": None}

        cfg = PipelineConfig(self.config_path)
        cfg.set_version(job.version)

        def _progress(stage: str, epoch: int, total: int,
                      train_loss: float, val_loss: Optional[float]) -> None:
            with self._lock:
                job.progress = {"stage": stage, "epoch": epoch, "total": total,
                                "train_loss": round(float(train_loss), 6),
                                "val_loss": (None if val_loss is None
                                             else round(float(val_loss), 6))}

        result = run_factor_train(
            cfg, job.factor_cols, job.factor_weights, version=job.version,
            base_version=job.base_version, epochs=job.epochs, progress_cb=_progress,
            factor_specs=job.factor_specs)

        # 生成因子权重分配图，落到该版本 viz 目录。
        importance = result.get("factor_importance", {})
        viz_dir = cfg.runs_root / cfg.exp_name / job.version / "viz"
        if importance:
            viz_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_factor_weights(importance, str(viz_dir / "factor_weights.png"),
                                    title=f"因子权重分配 {job.version}")
            except Exception:  # 出图失败不应判任务失败
                pass

        # 生成「Kronos 原始 vs 因子调整」K 线预测对比图（非致命，失败忽略）。
        try:
            from pipeline.factor_viz import generate_factor_comparison
            fcfg = result.get("factor_config", {})
            meta = result.get("meta", {})
            specs = fcfg.get("factor_specs") or []
            best_model = result.get("best_model")
            if specs and best_model and meta.get("tokenizer_src") and meta.get("predictor_src"):
                cmp_out = generate_factor_comparison(
                    cfg, job.version, meta["tokenizer_src"], meta["predictor_src"],
                    best_model, specs)
                if cmp_out:
                    with self._lock:
                        job.progress["comparison"] = {
                            k: cmp_out[k] for k in cmp_out if k != "kline"}
        except Exception:
            pass

        with self._lock:
            job.status = "completed"
            job.finished_at = dt.datetime.now().isoformat(timespec="seconds")
            job.progress["best_value"] = result.get("best_value")
            job.progress["best_epoch"] = result.get("best_epoch")
