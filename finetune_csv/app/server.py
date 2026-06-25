"""独立因子调参 App 的 Flask 后端（Phase 3）。

提供 REST 接口：列出可选因子、提交重训、查询任务进度、列出/对比版本、读取可视化图。
训练在后台串行执行（见 jobs.JobManager），前端轮询进度。

启动见 finetune_csv/run_app.py。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory, render_template
from flask.json.provider import DefaultJSONProvider

from pipeline import PipelineConfig
from .jobs import JobManager
from .registry import discover_factor_columns, get_version, list_versions
from .factor_meta import all_meta, group_by_category, CATEGORY_INFO
from .factor_analysis import analyze_factors


class _NumpyJSONProvider(DefaultJSONProvider):
    """让 jsonify 能序列化 numpy 标量 / 数组（统计分析结果常含 numpy 类型）。"""

    @staticmethod
    def default(o):
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)


def create_app(config_path: str) -> Flask:
    """构建 Flask 应用。config_path 为统一训练配置 YAML（所有重训共用）。"""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.json = _NumpyJSONProvider(app)
    manager = JobManager(config_path)

    def _cfg() -> PipelineConfig:
        # 读类接口用临时 cfg（不缓存版本），避免与后台任务版本串扰。
        return PipelineConfig(config_path)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/factors")
    def api_factors():
        cfg = _cfg()
        cols = discover_factor_columns(cfg)
        return jsonify({
            "exp_name": cfg.exp_name,
            "factor_cols": cols,
            "factor_meta": all_meta(cols),
            "categories": CATEGORY_INFO,
            "groups": group_by_category(cols),
            "default_epochs": cfg.basemodel_epochs,
            "lookback": cfg.lookback_window,
            "predict": cfg.predict_window,
        })

    @app.get("/api/factor_meta")
    def api_factor_meta():
        cfg = _cfg()
        cols = discover_factor_columns(cfg)
        return jsonify({
            "categories": CATEGORY_INFO,
            "groups": group_by_category(cols),
            "factor_meta": all_meta(cols),
        })

    @app.post("/api/analysis")
    def api_analysis():
        """基于训练集的因子统计画像：多空方向 / 评分 / 贡献 / 综合画像。"""
        body = request.get_json(force=True, silent=True) or {}
        cols = body.get("factor_cols") or None
        weights = body.get("factor_weights") or None
        try:
            res = analyze_factors(_cfg(), factor_cols=cols, weights=weights)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        code = 400 if isinstance(res, dict) and res.get("error") else 200
        return jsonify(res), code

    @app.get("/api/versions")
    def api_versions():
        return jsonify(list_versions(_cfg()))

    @app.get("/api/versions/<version>")
    def api_version(version: str):
        info = get_version(_cfg(), version)
        if info is None:
            return jsonify({"error": f"版本 {version} 不存在或非因子版本"}), 404
        return jsonify(info)

    @app.post("/api/retrain")
    def api_retrain():
        body = request.get_json(force=True, silent=True) or {}
        factor_cols = body.get("factor_cols") or []
        factor_weights = body.get("factor_weights") or {}
        factor_specs = body.get("factor_specs") or None
        base_version = body.get("base_version") or None
        epochs = body.get("epochs")
        if not factor_cols and not factor_specs:
            return jsonify({"error": "factor_cols 与 factor_specs 不能同时为空"}), 400
        try:
            job = manager.submit(factor_cols, factor_weights, base_version,
                                 int(epochs) if epochs else None,
                                 factor_specs=factor_specs)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(job.to_dict()), 202

    @app.get("/api/jobs")
    def api_jobs():
        return jsonify(manager.all_jobs())

    @app.get("/api/jobs/<job_id>")
    def api_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            return jsonify({"error": f"任务 {job_id} 不存在"}), 404
        return jsonify(job.to_dict())

    @app.get("/api/compare")
    def api_compare():
        versions = [v for v in (request.args.get("versions", "").split(",")) if v]
        cfg = _cfg()
        out = []
        for v in versions:
            info = get_version(cfg, v)
            if info is not None:
                out.append(info)
        return jsonify(out)

    @app.get("/viz/<version>/<path:filename>")
    def viz_file(version: str, filename: str):
        cfg = _cfg()
        viz_dir = cfg.runs_root / cfg.exp_name / version / "viz"
        return send_from_directory(str(viz_dir), filename)

    return app
