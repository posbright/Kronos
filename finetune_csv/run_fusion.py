"""方案 C-C1 生产编排入口：把「Kronos 衍生特征 → 因子对齐 → C1 下游融合模型」串成
可训练、可持久化、可服务的一条主线（train / predict / smoke 三个子命令）。

中文说明（遵循方案 C 工程最佳实践：以 C1 特征融合为主线）：
    本脚本把三段离线步骤封装为一个端到端服务入口：
      1. 用 Kronos 对历史价量逐窗口生成衍生特征（k_pred_ret / k_up_prob / k_pred_vol）。
      2. 与外部因子、未来收益标签对齐成融合宽表，并按时间切分（防泄漏）。
      3. 训练 C1 下游融合模型（LightGBM 优先，未安装回退 numpy Ridge），
         连同元信息打包成「模型 bundle」落盘。
      4. 线上服务时加载 bundle，对最新窗口生成特征并打分，输出对未来 H 日收益的预测。

    与 compare_fusion_strategies.py 的分工：
      - run_fusion.py  = 生产主线（只训练 / 服务 C1，产出可部署 bundle）。
      - compare_*.py   = 离线对照 / 选型（C1 vs C2，验证集选型 + 兜底切换）。

用法：
    # 训练并打包 C1 模型 bundle
    python finetune_csv/run_fusion.py train \
        --price-csv finetune_csv/data/A_000001_daily.csv \
        --factors data/factors_000001.csv \
        --tokenizer pretrained/Kronos-Tokenizer-base \
        --predictor pretrained/Kronos-base \
        --out-bundle runs/fusion_000001 \
        --symbol 000001 --lookback 90 --pred 5 --samples 30 --horizon 5 \
        --train-end 2024-01-01 --val-end 2025-01-01

    # 用训练好的 bundle 预测最新一天对未来 H 日收益的走势
    python finetune_csv/run_fusion.py predict \
        --bundle runs/fusion_000001 \
        --price-csv finetune_csv/data/A_000001_daily.csv \
        --factors data/factors_000001.csv \
        --out-json runs/fusion_000001/latest_prediction.json

    # 冒烟自测（无需任何预训练权重 / 外部文件，端到端 train→save→load→predict）
    python finetune_csv/run_fusion.py smoke
"""

import argparse
import datetime as _dt
import json
import os
import sys
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_kronos_features import build_features, _PRICE_COLS  # noqa: E402
from build_fusion_dataset import (build_fusion, time_split,  # noqa: E402
                                  _FFILL_FACTORS, _ZERO_FACTORS)
from compare_fusion_strategies import _metrics, _HAS_LGB  # noqa: E402
from kronos_loader import (  # noqa: E402
    DEFAULT_PREDICTOR_MS,
    DEFAULT_TOKENIZER_MS,
    load_kronos_predictor,
)

if _HAS_LGB:
    import lightgbm as lgb  # noqa: E402


# ----------------------------- C1 下游模型（可持久化） -----------------------------

def _ridge_fit(x, y, alpha=1.0):
    """numpy 闭式 Ridge 拟合（带标准化），返回可序列化的参数字典。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = x.mean(0), x.std(0) + 1e-8
    xs = (x - mu) / sd
    n, d = xs.shape
    xb = np.hstack([xs, np.ones((n, 1))])
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0                       # 不正则化偏置项
    w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    return {"mu": mu, "sd": sd, "w": w}


def resolve_backend(backend: str) -> str:
    """把 auto/lightgbm/ridge 解析为实际可用后端。"""
    if backend == "ridge":
        return "ridge"
    if backend == "lightgbm":
        if not _HAS_LGB:
            raise RuntimeError("指定 --backend lightgbm 但未安装 lightgbm，请改用 auto/ridge")
        return "lightgbm"
    return "lightgbm" if _HAS_LGB else "ridge"      # auto


class C1Model:
    """C1 特征融合下游模型（LightGBM 优先，Ridge 回退），支持 save/load 用于线上服务。"""

    def __init__(self, backend: str, feat_cols):
        self.backend = backend
        self.feat_cols = list(feat_cols)
        self._booster = None        # lightgbm Booster
        self._ridge = None          # {"mu","sd","w"}

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.backend == "lightgbm":
            dtrain = lgb.Dataset(x, label=y)
            params = {"objective": "regression", "learning_rate": 0.05,
                      "num_leaves": 31, "feature_fraction": 0.8,
                      "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1}
            self._booster = lgb.train(params, dtrain, num_boost_round=300)
        else:
            self._ridge = _ridge_fit(x, y, alpha=1.0)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.backend == "lightgbm":
            return self._booster.predict(x)
        p = self._ridge
        xs = (x - p["mu"]) / p["sd"]
        xb = np.hstack([xs, np.ones((len(xs), 1))])
        return xb @ p["w"]

    def save(self, bundle_dir: str):
        os.makedirs(bundle_dir, exist_ok=True)
        if self.backend == "lightgbm":
            self._booster.save_model(os.path.join(bundle_dir, "c1_lgb.txt"))
        else:
            np.savez(os.path.join(bundle_dir, "c1_ridge.npz"), **self._ridge)

    @classmethod
    def load(cls, bundle_dir: str, backend: str, feat_cols):
        m = cls(backend, feat_cols)
        if backend == "lightgbm":
            m._booster = lgb.Booster(model_file=os.path.join(bundle_dir, "c1_lgb.txt"))
        else:
            z = np.load(os.path.join(bundle_dir, "c1_ridge.npz"))
            m._ridge = {k: z[k] for k in z.files}
        return m


# ----------------------------- 训练流水线 -----------------------------

_KRONOS_COLS = ["k_pred_ret", "k_up_prob", "k_pred_vol"]


def train_pipeline(predictor, price_df, factors_df, *, symbol, lookback, pred,
                   samples, horizon, train_end, val_end, backend="auto",
                   tokenizer_src=None, predictor_src=None):
    """端到端训练 C1：Kronos 特征 → 融合宽表 → 时间切分 → 训练 → 返回 (模型, 元信息, 指标, 切分规模)。"""
    backend = resolve_backend(backend)
    label = f"label_fwd_ret_{horizon}d"

    kf = build_features(predictor, price_df, symbol=symbol, lookback=lookback,
                        pred_len=pred, samples=samples)
    df = build_fusion(kf, factors_df, price_df, horizon=horizon)

    reserved = set(["date", "symbol", label] + _KRONOS_COLS)
    factor_cols = [c for c in df.columns if c not in reserved]
    feat_cols = _KRONOS_COLS + factor_cols

    tr, va, te = time_split(df, train_end=train_end, val_end=val_end)
    if len(tr) == 0:
        raise ValueError("训练集为空：请检查 --train-end 与数据日期范围是否匹配")

    model = C1Model(backend, feat_cols).fit(tr[feat_cols].values, tr[label].values)

    metrics = {}
    if len(va):
        metrics["val"] = _metrics(va[label].values, model.predict(va[feat_cols].values))
    if len(te):
        metrics["test"] = _metrics(te[label].values, model.predict(te[feat_cols].values))

    manifest = {
        "strategy": "C1",
        "backend": backend,
        "symbol": symbol,
        "feat_cols": feat_cols,
        "kronos_cols": _KRONOS_COLS,
        "factor_cols": factor_cols,
        "label": label,
        "horizon": horizon,
        "lookback": lookback,
        "pred": pred,
        "samples": samples,
        "tokenizer_src": tokenizer_src,
        "predictor_src": predictor_src,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    return model, manifest, metrics, (len(tr), len(va), len(te))


# ----------------------------- 服务 / 预测 -----------------------------

def _future_timestamps(ts: pd.Series, n: int) -> pd.Series:
    """根据历史时间步长（频率无关）外推 n 个未来时间戳，用于真正预测「下一段」。"""
    ts = pd.to_datetime(pd.Series(ts)).sort_values().reset_index(drop=True)
    step = ts.diff().dropna().median()
    if pd.isna(step) or step == pd.Timedelta(0):
        step = pd.Timedelta(days=1)
    last = ts.iloc[-1]
    return pd.Series([last + step * (i + 1) for i in range(n)])


def _kronos_feature_one_window(predictor, hist: pd.DataFrame, future_ts: pd.Series,
                               samples: int) -> dict:
    """对单个历史窗口做多次采样预测，统计 Kronos 衍生特征（用于最新点预测）。"""
    x_df = hist[_PRICE_COLS].reset_index(drop=True)
    x_ts = hist["timestamps"].reset_index(drop=True)
    last_close = float(x_df["close"].iloc[-1])
    preds = []
    for _ in range(samples):
        pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=future_ts,
                                    pred_len=len(future_ts), T=1.0, top_p=0.9,
                                    sample_count=1, verbose=False)
        preds.append(pred_df["close"].values)
    preds = np.asarray(preds, dtype=np.float64)
    end_ret = preds[:, -1] / last_close - 1.0
    return {"k_pred_ret": float(end_ret.mean()),
            "k_up_prob": float((end_ret > 0).mean()),
            "k_pred_vol": float(np.std(end_ret))}


def _latest_factor_row(factors_df, symbol, date, factor_cols) -> dict:
    """取 <= 预测日的最近因子值（慢变因子前向填充，新闻/事件类缺失填 0）。"""
    out = {c: 0.0 for c in factor_cols}
    if factors_df is None or not factor_cols:
        return out
    f = factors_df.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    if "symbol" in f.columns:
        f = f[f["symbol"] == symbol]
    f = f[f["date"] <= date].sort_values("date")
    for c in factor_cols:
        if c in f.columns and len(f):
            if c in _ZERO_FACTORS:
                val = f[c].iloc[-1]
            else:                       # 慢变因子（含 _FFILL_FACTORS）前向填充取最近非空
                s = f[c].ffill()
                val = s.iloc[-1] if len(s) else np.nan
            out[c] = float(val) if pd.notna(val) else 0.0
    return out


def predict_latest(predictor, model: C1Model, manifest: dict,
                   price_df: pd.DataFrame, factors_df=None) -> dict:
    """加载好的 bundle 对「最新一根 K 线」打分，预测未来 H 期收益走势。"""
    px = price_df.copy()
    px["timestamps"] = pd.to_datetime(px["timestamps"])
    px = px.sort_values("timestamps").reset_index(drop=True)

    lookback, pred, samples = manifest["lookback"], manifest["pred"], manifest["samples"]
    if len(px) < lookback:
        raise ValueError(f"价格行数 {len(px)} < lookback {lookback}，历史不足以预测")

    hist = px.iloc[-lookback:]
    pred_date = pd.to_datetime(hist["timestamps"].iloc[-1]).normalize()
    future_ts = _future_timestamps(px["timestamps"], pred)

    feat = _kronos_feature_one_window(predictor, hist, future_ts, samples)
    feat.update(_latest_factor_row(factors_df, manifest["symbol"], pred_date,
                                   manifest["factor_cols"]))
    x = np.array([[feat[c] for c in manifest["feat_cols"]]], dtype=np.float64)
    score = float(model.predict(x)[0])

    return {
        "as_of_date": str(pred_date.date()),
        "symbol": manifest["symbol"],
        "horizon_days": manifest["horizon"],
        "pred_fwd_ret": score,
        "direction": "up" if score > 0 else "down",
        "k_up_prob": feat["k_up_prob"],
        "k_pred_vol": feat["k_pred_vol"],
        "backend": manifest["backend"],
    }


# ----------------------------- bundle 持久化 -----------------------------

def save_bundle(bundle_dir: str, model: C1Model, manifest: dict):
    os.makedirs(bundle_dir, exist_ok=True)
    model.save(bundle_dir)
    with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_bundle(bundle_dir: str):
    with open(os.path.join(bundle_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    model = C1Model.load(bundle_dir, manifest["backend"], manifest["feat_cols"])
    return model, manifest


# ----------------------------- CLI -----------------------------

def _load_predictor(tokenizer_src, predictor_src, device, model_source="modelscope"):
    predictor, load_meta = load_kronos_predictor(
        tokenizer_src=tokenizer_src,
        predictor_src=predictor_src,
        device=device,
        max_context=512,
        prefer_source=model_source,
        verbose=True,
    )
    return predictor, load_meta


def cmd_train(args):
    predictor, load_meta = _load_predictor(args.tokenizer, args.predictor, args.device, args.model_source)
    px = pd.read_csv(args.price_csv)
    ff = pd.read_csv(args.factors)
    model, manifest, metrics, sizes = train_pipeline(
        predictor, px, ff, symbol=args.symbol, lookback=args.lookback,
        pred=args.pred, samples=args.samples, horizon=args.horizon,
        train_end=args.train_end, val_end=args.val_end, backend=args.backend,
        tokenizer_src=load_meta["tokenizer"]["source"],
        predictor_src=load_meta["predictor"]["source"])
    manifest["tokenizer_provider"] = load_meta["tokenizer"]["provider"]
    manifest["predictor_provider"] = load_meta["predictor"]["provider"]
    save_bundle(args.out_bundle, model, manifest)
    print(f"[train] 后端={manifest['backend']}  切分 train/val/test={sizes}")
    for split, m in metrics.items():
        print(f"  {split}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    print(f"[train] C1 模型 bundle 已保存到 {args.out_bundle}")


def cmd_predict(args):
    model, manifest = load_bundle(args.bundle)
    tok_src = args.tokenizer or manifest.get("tokenizer_src") or DEFAULT_TOKENIZER_MS
    mdl_src = args.predictor or manifest.get("predictor_src") or DEFAULT_PREDICTOR_MS
    if not tok_src or not mdl_src:
        raise SystemExit("缺少 tokenizer/predictor 路径：bundle 未记录且未通过 --tokenizer/--predictor 提供")
    predictor, _ = _load_predictor(tok_src, mdl_src, args.device, args.model_source)
    px = pd.read_csv(args.price_csv)
    ff = pd.read_csv(args.factors) if args.factors else None
    out = predict_latest(predictor, model, manifest, px, ff)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[predict] 预测结果已写入 {args.out_json}")


def _smoke_test():
    """无需权重 / 文件的端到端冒烟：train → save → load → predict。"""
    from build_kronos_features import _build_tiny_predictor, _synth_price
    predictor = _build_tiny_predictor()
    px = _synth_price(n=160)
    rng = np.random.default_rng(0)
    n = len(px)
    ff = pd.DataFrame({
        "date": px["timestamps"].values, "symbol": "TEST",
        "pe": 15 + rng.standard_normal(n), "pb": 1.5 + rng.standard_normal(n) * 0.1,
        "roe": 0.1 + rng.standard_normal(n) * 0.01,
        "news_sent": rng.standard_normal(n),
        "news_count": rng.integers(0, 5, n).astype(float), "event_flag": 0.0,
    })
    train_end = str(pd.to_datetime(px["timestamps"].iloc[int(n * 0.6)]).date())
    val_end = str(pd.to_datetime(px["timestamps"].iloc[int(n * 0.8)]).date())

    model, manifest, metrics, sizes = train_pipeline(
        predictor, px, ff, symbol="TEST", lookback=60, pred=5, samples=3,
        horizon=5, train_end=train_end, val_end=val_end, backend="auto")
    assert sizes[0] > 0, "训练集为空"
    assert set(_KRONOS_COLS).issubset(manifest["feat_cols"]), "缺少 Kronos 特征列"

    with tempfile.TemporaryDirectory() as tmp:
        bundle = os.path.join(tmp, "bundle")
        save_bundle(bundle, model, manifest)
        assert os.path.exists(os.path.join(bundle, "manifest.json"))
        m2, man2 = load_bundle(bundle)
        out = predict_latest(predictor, m2, man2, px, ff)

    assert np.isfinite(out["pred_fwd_ret"]), "预测值非有限"
    assert out["direction"] in ("up", "down")
    assert 0.0 <= out["k_up_prob"] <= 1.0
    print(f"[smoke] run_fusion 通过：后端={manifest['backend']}，切分={sizes}，"
          f"最新预测={out['as_of_date']} {out['symbol']} "
          f"未来{out['horizon_days']}日 {out['direction']} ({out['pred_fwd_ret']:+.4f})")


def main():
    parser = argparse.ArgumentParser(description="方案 C-C1 生产编排：训练 / 服务可部署的融合模型")
    sub = parser.add_subparsers(dest="cmd")

    pt = sub.add_parser("train", help="训练并打包 C1 模型 bundle")
    pt.add_argument("--price-csv", required=True)
    pt.add_argument("--factors", required=True)
    pt.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_MS,
                    help="tokenizer 来源（默认 ModelScope ID，可传本地目录）")
    pt.add_argument("--predictor", default=DEFAULT_PREDICTOR_MS,
                    help="predictor 来源（默认 ModelScope ID，可传本地目录）")
    pt.add_argument("--out-bundle", required=True)
    pt.add_argument("--symbol", default="UNKNOWN")
    pt.add_argument("--lookback", type=int, default=90)
    pt.add_argument("--pred", type=int, default=5)
    pt.add_argument("--samples", type=int, default=30)
    pt.add_argument("--horizon", type=int, default=5)
    pt.add_argument("--train-end", default="2024-01-01")
    pt.add_argument("--val-end", default="2025-01-01")
    pt.add_argument("--backend", choices=["auto", "lightgbm", "ridge"], default="auto")
    pt.add_argument("--device", default=None)
    pt.add_argument("--model-source", choices=["modelscope", "hf"], default="modelscope",
                    help="模型源优先级（默认 modelscope，失败自动回退 hf）")

    pp = sub.add_parser("predict", help="用 bundle 预测最新一天的未来走势")
    pp.add_argument("--bundle", required=True)
    pp.add_argument("--price-csv", required=True)
    pp.add_argument("--factors", default=None)
    pp.add_argument("--tokenizer", default=None,
                    help="覆盖 bundle 记录的 tokenizer 路径（若两处都无则默认 ModelScope）")
    pp.add_argument("--predictor", default=None,
                    help="覆盖 bundle 记录的主模型路径（若两处都无则默认 ModelScope）")
    pp.add_argument("--device", default=None)
    pp.add_argument("--model-source", choices=["modelscope", "hf"], default="modelscope",
                    help="模型源优先级（默认 modelscope，失败自动回退 hf）")
    pp.add_argument("--out-json", default=None)

    sub.add_parser("smoke", help="无需权重/文件的端到端冒烟自测")

    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "smoke":
        _smoke_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
