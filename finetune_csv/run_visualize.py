"""可视化编排脚本（Phase 2）。

把「某个训练版本」的模型在测试集某只股票上的预测画成日 K 线对比图，并（若提供因子
模型）绘制因子权重分配图。产物落盘到 runs/<exp>/<version>/viz/。

两类预测对比：
    - 基线模型（KronosPredictor）：仅用 OHLCV 自回归预测。
    - 因子增强模型（FactorPredictor，方案 B）：在每步注入因子条件后预测。

用法：
    # 仅基线预测的 K 线对比图（需已有训练版本）
    python finetune_csv/run_visualize.py --config finetune_csv/configs/config_smoke20.yaml

    # 叠加因子增强模型 + 因子权重图
    python finetune_csv/run_visualize.py --config ... \
        --factor-model runs/<exp>/<ver>/train/factor/best_model \
        --factor-cols tech_macd,tech_rsi,tech_kdjk

    # 无需任何权重的冒烟自测（合成数据，验证绘图与因子重要性计算）
    python finetune_csv/run_visualize.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from viz import (  # noqa: E402
    plot_kline_comparison,
    factor_importance,
    load_factor_emb_weight,
    plot_factor_weights,
)

_PRICE_COLS = ["open", "high", "low", "close"]


def _time_col(df: pd.DataFrame) -> str:
    return "timestamps" if "timestamps" in df.columns else "date"


def _select_window(df: pd.DataFrame, lookback: int, pred_len: int,
                   symbol: Optional[str]) -> pd.DataFrame:
    """从（多股票堆叠的）数据集中取出某只股票最后 lookback+pred_len 根连续 K 线。"""
    if "symbol" in df.columns:
        sym = symbol or df["symbol"].iloc[-1]
        df = df[df["symbol"] == sym].copy()
    need = lookback + pred_len
    if len(df) < need:
        raise ValueError(f"该股票样本数 {len(df)} < lookback+pred_len={need}")
    tcol = _time_col(df)
    df = df.sort_values(tcol).tail(need).reset_index(drop=True)
    df.index = pd.to_datetime(df[tcol])
    return df


def run_visualize(cfg, version: Optional[str], symbol: Optional[str],
                  factor_model_dir: Optional[str],
                  factor_cols: Optional[List[str]]) -> Dict[str, str]:
    """真实模式：加载训练版本模型，生成 K 线对比 + 因子权重图。"""
    from model import Kronos, KronosTokenizer, KronosPredictor

    if version:
        cfg.set_version(version)
    elif not cfg.get("runs.version", ""):
        latest = cfg.latest_version()
        if latest:
            cfg.set_version(latest)

    lookback, pred_len = cfg.lookback_window, cfg.predict_window
    train_dir = cfg.runs_root / cfg.exp_name / cfg.version / "train"
    tok_path = train_dir / "tokenizer" / "best_model"
    model_path = train_dir / "basemodel" / "best_model"
    for p in (tok_path, model_path):
        if not p.exists():
            raise FileNotFoundError(f"未找到训练产出 {p}；请先运行 run_train.py。")

    csv_path = cfg.dataset_root / "test" / "dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到测试集 {csv_path}；请先运行 run_build_dataset.py。")

    out_dir = cfg.runs_root / cfg.exp_name / cfg.version / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    window = _select_window(df, lookback, pred_len, symbol)
    history = window.iloc[:lookback]
    actual = window.iloc[lookback:]

    tokenizer = KronosTokenizer.from_pretrained(str(tok_path))
    model = Kronos.from_pretrained(str(model_path))
    predictor = KronosPredictor(model, tokenizer, device=None, max_context=cfg.max_context)

    x_ts = pd.Series(history.index)
    y_ts = pd.Series(actual.index)
    base_pred = predictor.predict(history[_PRICE_COLS + ["volume", "amount"]]
                                  if "volume" in history.columns else history[_PRICE_COLS],
                                  x_ts, y_ts, pred_len, verbose=False)
    base_pred.index = actual.index
    pred_dict = {"基线 Kronos": base_pred}

    # 可选：因子增强预测。
    outputs: Dict[str, str] = {}
    if factor_model_dir and factor_cols:
        from finetune_csv.factor_model import load_factor_model
        from pipeline import FactorPredictor

        # 用 load_factor_model 正确恢复训练好的 factor_emb（避免被 init_factor 清零）。
        fac_model = load_factor_model(factor_model_dir, len(factor_cols))
        fac_predictor = FactorPredictor(fac_model, tokenizer, device=None,
                                        max_context=cfg.max_context)
        factor_hist = history[factor_cols].values.astype(np.float32)
        fac_pred = fac_predictor.predict(
            history[_PRICE_COLS + ["volume", "amount"]] if "volume" in history.columns
            else history[_PRICE_COLS],
            x_ts, y_ts, pred_len, factor_hist=factor_hist)
        fac_pred.index = actual.index
        pred_dict["因子增强 Kronos"] = fac_pred

    sym_label = symbol or (window["symbol"].iloc[0] if "symbol" in window.columns else "")
    kline_png = str(out_dir / "kline_comparison.png")
    plot_kline_comparison(history[_PRICE_COLS], actual[_PRICE_COLS], pred_dict,
                          kline_png, title=f"Kronos 预测对比 {sym_label}（日K线）")
    outputs["kline"] = kline_png

    # 因子权重分配图（若有因子模型）。
    if factor_model_dir:
        weight = load_factor_emb_weight(factor_model_dir)
        imp = factor_importance(weight, factor_cols)
        fw_png = str(out_dir / "factor_weights.png")
        plot_factor_weights(imp, fw_png)
        outputs["factor_weights"] = fw_png

    return outputs


def _smoke_test() -> None:
    """无需任何权重的冒烟测试：合成数据验证 K 线绘图 + 因子重要性计算 + 出图。"""
    import tempfile

    rng = np.random.default_rng(0)
    n_hist, n_pred = 40, 6
    base = 10 + np.cumsum(rng.normal(0, 0.15, n_hist + n_pred))

    def _ohlc(vals, idx):
        return pd.DataFrame({
            "open": vals, "high": vals + 0.25, "low": vals - 0.25,
            "close": vals + rng.normal(0, 0.05, len(vals)),
        }, index=idx)

    full_idx = pd.date_range("2024-01-01", periods=n_hist + n_pred, freq="D")
    history = _ohlc(base[:n_hist], full_idx[:n_hist])
    actual = _ohlc(base[n_hist:], full_idx[n_hist:])
    pred_a = _ohlc(base[n_hist:] + rng.normal(0, 0.2, n_pred), full_idx[n_hist:])
    pred_b = _ohlc(base[n_hist:] + rng.normal(0, 0.1, n_pred), full_idx[n_hist:])

    out_dir = Path(tempfile.mkdtemp(prefix="kronos_viz_smoke_"))
    kline_png = plot_kline_comparison(
        history, actual, {"基线 Kronos": pred_a, "因子增强 Kronos": pred_b},
        str(out_dir / "kline_comparison.png"))
    assert Path(kline_png).exists() and Path(kline_png).stat().st_size > 0

    # 因子重要性：构造 [d_model, k] 权重，列范数应可正确排序。
    names = ["tech_macd", "tech_rsi", "tech_kdjk", "fund_pe"]
    weight = np.zeros((8, 4), dtype=np.float32)
    weight[:, 0] = 1.0   # 最重要
    weight[:, 1] = 0.5
    weight[:, 2] = 0.1
    weight[:, 3] = 0.0
    imp = factor_importance(weight, names)
    assert list(imp.keys())[0] == "tech_macd", "L2 范数最大的因子应排第一"
    fw_png = plot_factor_weights(imp, str(out_dir / "factor_weights.png"))
    assert Path(fw_png).exists() and Path(fw_png).with_suffix(".json").exists()

    print("[smoke] run_visualize 通过：")
    print("  K线对比图 ->", kline_png)
    print("  因子权重图 ->", fw_png)
    print("  因子排序   ->", {k: round(v["share"], 3) for k, v in imp.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos 预测可视化（K线对比 + 因子权重）")
    parser.add_argument("--config", type=str,
                        default=str(_THIS_DIR / "configs" / "config_smoke20.yaml"))
    parser.add_argument("--version", type=str, default="", help="训练版本号（留空=最近一次）")
    parser.add_argument("--symbol", type=str, default="", help="股票代码（留空=取最后一只）")
    parser.add_argument("--factor-model", type=str, default="",
                        help="因子模型 best_model 目录（留空=跳过因子增强对比）")
    parser.add_argument("--factor-cols", type=str, default="",
                        help="因子列名，逗号分隔（与因子模型 factor_dim 对应）")
    parser.add_argument("--smoke", action="store_true", help="合成数据冒烟自测")
    args = parser.parse_args()

    if args.smoke:
        _smoke_test()
        return

    from pipeline import PipelineConfig

    cfg = PipelineConfig(args.config)
    factor_cols = [c.strip() for c in args.factor_cols.split(",") if c.strip()] or None
    outputs = run_visualize(cfg, args.version or None, args.symbol or None,
                            args.factor_model or None, factor_cols)
    print("已生成可视化产物：")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
