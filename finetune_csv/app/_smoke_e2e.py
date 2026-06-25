"""App 端到端冒烟自测（无需下载权重）。

合成微型 DataSet + 微型 tokenizer/预测器 + 最小 YAML 配置，用 Flask test_client 走通：
    /api/factors -> /api/retrain -> 轮询 /api/jobs/<id> 至 completed
    -> /api/versions -> /api/compare -> /viz/<v>/factor_weights.png

用法：python finetune_csv/app/_smoke_e2e.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))  # repo root
sys.path.insert(0, str(_THIS.parents[1]))  # finetune_csv

import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model import KronosTokenizer  # noqa: E402
from factor_model import KronosWithFactor  # noqa: E402
from app import create_app  # noqa: E402


def _build_env(tmp: Path) -> Path:
    ds_root = tmp / "DataSet"
    rng = np.random.default_rng(0)
    for split in ("train", "validation", "test"):
        rows = []
        for sym in ("000001", "600000"):
            n = 90
            base = 10 + np.cumsum(rng.normal(0, 0.1, n))
            fwd = pd.Series(base).shift(-5) / pd.Series(base) - 1.0
            rows.append(pd.DataFrame({
                "symbol": sym,
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "open": base, "high": base + 0.2, "low": base - 0.2, "close": base + 0.05,
                "volume": rng.uniform(1e5, 1e6, n), "amount": rng.uniform(1e6, 1e7, n),
                "tech_macd": rng.normal(0, 1, n), "tech_rsi": rng.normal(0, 1, n),
                "tech_rsi_6": rng.normal(0, 1, n), "fin_roe": rng.normal(0, 1, n),
                "label_fwd_ret_5d": fwd.values,
            }))
        d = ds_root / split
        d.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(d / "dataset.csv", index=False)

    tok_dir, pred_dir = tmp / "tiny_tok", tmp / "tiny_pred"
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
  exp_name: kronos_app_smoke
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
    return cfg_path


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="kronos_app_e2e_"))
    cfg_path = _build_env(tmp)
    app = create_app(str(cfg_path))
    c = app.test_client()

    factors = c.get("/api/factors").get_json()
    assert set(["tech_macd", "tech_rsi", "fin_roe"]).issubset(set(factors["factor_cols"])), \
        f"因子发现异常: {factors['factor_cols']}"
    # 元数据与分组
    assert factors.get("factor_meta") and factors["factor_meta"][0].get("name_cn"), "缺因子中文元数据"
    assert factors.get("groups"), "缺因子分组"

    # 因子统计分析（多空/评分/综合画像）
    ana = c.post("/api/analysis", json={
        "factor_cols": ["tech_macd", "tech_rsi", "tech_rsi_6", "fin_roe"]}).get_json()
    assert "factors" in ana and "composite" in ana, f"分析返回异常: {ana}"
    assert "divergence" in ana["composite"] and "expected_return_pct" in ana["composite"]
    assert ana["daily_volatility_pct"] > 0, "日波动应为正"

    # 重训：技术动量类用「同类均值」聚合（tech_rsi+tech_rsi_6）+ 其余独立
    resp = c.post("/api/retrain", json={
        "factor_cols": ["tech_macd", "动量类·均值", "fin_roe"],
        "factor_specs": [
            {"name": "tech_macd", "mode": "raw", "cols": ["tech_macd"],
             "signs": [1], "weight": 1.0},
            {"name": "动量类·均值", "mode": "mean",
             "cols": ["tech_rsi", "tech_rsi_6"], "signs": [-1, -1], "weight": 1.0},
            {"name": "fin_roe", "mode": "raw", "cols": ["fin_roe"],
             "signs": [1], "weight": 1.5},
        ],
        "epochs": 1,
    })
    assert resp.status_code == 202, f"retrain 应返回 202，实际 {resp.status_code}"
    job = resp.get_json()
    jid = job["job_id"]

    # 轮询任务直至完成（后台线程，进程内）。
    deadline = time.time() + 180
    status = job["status"]
    while status in ("queued", "running") and time.time() < deadline:
        time.sleep(1.0)
        status = c.get(f"/api/jobs/{jid}").get_json()["status"]
    assert status == "completed", f"任务未完成: status={status}"

    vers = c.get("/api/versions").get_json()
    assert any(v["version"] == job["version"] for v in vers), "版本列表缺少新版本"

    cmp = c.get(f"/api/compare?versions={job['version']}").get_json()
    assert cmp and cmp[0]["factor_importance"], "对比缺少因子重要性"
    imp = cmp[0]["factor_importance"]
    # 3 个通道（含 1 个聚合均值通道）
    assert len(imp) == 3, f"通道数应为 3，实际 {list(imp.keys())}"
    assert "动量类·均值" in imp, f"缺少聚合通道: {list(imp.keys())}"

    png = c.get(f"/viz/{job['version']}/factor_weights.png")
    assert png.status_code == 200 and png.data[:4] == b"\x89PNG", "因子权重图未生成"

    # K 线预测对比图（Kronos 原始 vs 因子调整）
    kline = c.get(f"/viz/{job['version']}/kline_comparison.png")
    has_kline = kline.status_code == 200 and kline.data[:4] == b"\x89PNG"

    print("[smoke] app 端到端通过：")
    print("  version =", job["version"])
    print("  composite =", ana["composite"]["direction"], ana["composite"]["score"],
          "divergence", ana["composite"]["divergence"])
    print("  importance =", {k: round(v["share"], 3) for k, v in imp.items()})
    print("  viz factor_weights.png =", len(png.data), "bytes")
    print("  viz kline_comparison.png =",
          (str(len(kline.data)) + " bytes") if has_kline else "未生成(可接受)")


if __name__ == "__main__":
    main()
