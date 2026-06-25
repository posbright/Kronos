"""版本注册表：发现并汇总各因子训练版本，供 App 列表与对比使用。

从 ``runs/<exp>/<version>/`` 目录扫描所有因子训练产出，读取：
    - train/factor/summary.json     训练状态 / 最优指标 / 收敛信息
    - train/factor/best_model/factor_config.json   因子列与每因子权重
    - train/factor/factor_importance.json          训练后各因子重要性占比
    - validate/test 的 summary.json（若已评估）      持出集指标
    - viz/*.png                                      可视化图（若已生成）

并提供：
    - discover_factor_columns(cfg): 从 DataSet 表头识别可选因子列（tech_*/fin_*）。
    - list_versions(cfg):           列出全部因子版本及其汇总信息。
    - get_version(cfg, version):    单版本详情。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pipeline import PipelineConfig


def _read_json(path: Path) -> Optional[dict]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def discover_factor_columns(cfg: PipelineConfig) -> List[str]:
    """从 DataSet/train/dataset.csv 表头识别可选因子列（tech_*/fin_*，排除 *_isna 掩码）。"""
    train_csv = cfg.dataset_root / "train" / "dataset.csv"
    if not train_csv.exists():
        return []
    header = pd.read_csv(train_csv, nrows=0).columns.tolist()
    cols = [c for c in header
            if (c.startswith("tech_") or c.startswith("fin_")) and not c.endswith("_isna")]
    return cols


def _factor_dir(cfg: PipelineConfig, version: str) -> Path:
    return cfg.runs_root / cfg.exp_name / version / "train" / "factor"


def get_version(cfg: PipelineConfig, version: str) -> Optional[Dict[str, object]]:
    """读取单个版本的因子训练汇总；非因子版本返回 None。"""
    fdir = _factor_dir(cfg, version)
    summary = _read_json(fdir / "summary.json")
    if summary is None:
        return None  # 该版本不是因子训练版本

    fcfg = _read_json(fdir / "best_model" / "factor_config.json") or {}
    importance = _read_json(fdir / "factor_importance.json") or {}

    vdir = cfg.runs_root / cfg.exp_name / version
    validate = _read_json(vdir / "validate" / "summary.json")
    test = _read_json(vdir / "test" / "summary.json")

    viz_dir = vdir / "viz"
    viz_files = ([p.name for p in viz_dir.glob("*.png")] if viz_dir.is_dir() else [])

    return {
        "version": version,
        "status": summary.get("status"),
        "best_value": summary.get("best_value"),
        "best_epoch": summary.get("best_epoch"),
        "epochs_run": summary.get("epochs_run"),
        "converged": summary.get("converged_or_early_stopped"),
        "total_seconds": summary.get("total_seconds"),
        "factor_cols": fcfg.get("factor_cols", []),
        "factor_weights": fcfg.get("factor_weights", {}),
        "factor_dim": fcfg.get("factor_dim"),
        "factor_importance": importance,
        "validate_metrics": (validate or {}).get("metrics") if validate else None,
        "test_metrics": (test or {}).get("metrics") if test else None,
        "viz_files": viz_files,
    }


def list_versions(cfg: PipelineConfig) -> List[Dict[str, object]]:
    """列出 runs/<exp> 下全部因子训练版本（按版本名倒序，最新在前）。"""
    exp_dir = cfg.runs_root / cfg.exp_name
    if not exp_dir.is_dir():
        return []
    out: List[Dict[str, object]] = []
    for vdir in sorted((p for p in exp_dir.iterdir() if p.is_dir()),
                       key=lambda p: p.name, reverse=True):
        info = get_version(cfg, vdir.name)
        if info is not None:
            out.append(info)
    return out
