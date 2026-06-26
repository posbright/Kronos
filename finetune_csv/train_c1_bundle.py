"""方案 C 缺口补全：多标的 C1 可部署 bundle 训练器（消费预构建的 fusion_*.csv）。

背景
    run_fusion.py train 是**单标的**端到端入口：它从一份 price-csv 自己重算 Kronos 特征，
    **不消费** build_dataC_step3_fusion.py 产出的多标的 fusion_{train,val,test}.csv。
    compare_fusion_strategies.py 只做离线选型、不落可部署 bundle。
    本脚本填补该缺口：直接吃多标的 fusion_*.csv，训练 C1 下游融合模型并打包成与
    run_fusion.py 完全一致的 bundle（manifest.json + c1_lgb.txt / c1_ridge.npz），可被
    run_fusion.py 的服务侧（C1Model.load）直接加载。

与其它脚本的分工
    - build_dataC_step3_fusion.py = 造数据（多标的融合宽表 + 时间切分）。
    - compare_fusion_strategies.py = 离线选型（C1 vs C2，验证集选型 + 兜底）。
    - run_fusion.py train          = 单标的端到端（自带特征生成 → bundle）。
    - 本脚本 train_c1_bundle.py    = 多标的，从已切分的 fusion_*.csv 训练 → 可部署 bundle。

评估口径
    - 池化指标（pooled）：把 val/test 全部样本拉平算 RMSE/IC/RankIC/Hit（与 compare 一致）。
    - 截面指标（by_date）：每个交易日内对该日全部标的算 IC/RankIC 再按日平均——多标的更合理，
      规避「池化 IC 受跨标的量纲与漂移污染」的问题。

防泄漏
    - 只在 train 上拟合下游模型；val 仅用于报告 / 选参，test 只看最终一次。
    - 时间切分由 step3 保证（train < val < test 不重叠）；本脚本不重新切分。

用法
    # 从 dataC 目录自动找 fusion_{train,val,test}.csv
    python finetune_csv/train_c1_bundle.py \
        --data-root C:/xapproject/Quantia/Kronos/DataSet/dataC \
        --out-bundle runs/dataC_c1 --horizon 5

    # 显式给三份文件 + 指定后端
    python finetune_csv/train_c1_bundle.py \
        --train DataSet/dataC/fusion_train.csv \
        --val   DataSet/dataC/fusion_val.csv \
        --test  DataSet/dataC/fusion_test.csv \
        --backend lightgbm --out-bundle runs/dataC_c1

    # 冒烟自测（无需任何文件 / 权重，合成多标的 fusion 数据跑通 train→save→load）
    python finetune_csv/train_c1_bundle.py --smoke

    # 上线打分（步骤8）：用训练好的多标的 bundle 对最新截面打分选股
    python finetune_csv/train_c1_bundle.py --predict \
        --data-root C:/xapproject/Quantia/Kronos/DataSet/dataC \
        --out-bundle runs/dataC_c1 --top 10 \
        --out-json runs/dataC_c1/latest_ranking.json
"""

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compare_fusion_strategies import _metrics, _ic, _rank_ic  # noqa: E402
from run_fusion import C1Model, resolve_backend  # noqa: E402

_KRONOS_COLS = ["k_pred_ret", "k_up_prob", "k_pred_vol"]


def _grouped_metrics(df: pd.DataFrame, label: str, pred: np.ndarray) -> dict:
    """按交易日做截面 IC/RankIC 再平均（多标的更合理的评估口径）。"""
    tmp = pd.DataFrame({"date": df["date"].values, "y": df[label].values, "p": pred})
    ics, rics, n_days = [], [], 0
    for _, g in tmp.groupby("date"):
        if len(g) < 3:                       # 当日标的太少，截面相关无意义
            continue
        ics.append(_ic(g["y"].values, g["p"].values))
        rics.append(_rank_ic(g["y"].values, g["p"].values))
        n_days += 1
    if n_days == 0:
        return {"IC_by_date": 0.0, "RankIC_by_date": 0.0, "n_days": 0}
    return {"IC_by_date": float(np.nanmean(ics)),
            "RankIC_by_date": float(np.nanmean(rics)), "n_days": n_days}


def _resolve_inputs(args) -> tuple:
    """解析三份 fusion 文件路径：显式 --train/--val/--test 优先，否则从 --data-root 推断。"""
    if args.train and args.val and args.test:
        return Path(args.train), Path(args.val), Path(args.test)
    if args.data_root:
        root = Path(args.data_root)
        return (root / "fusion_train.csv", root / "fusion_val.csv", root / "fusion_test.csv")
    raise SystemExit("请提供 --train/--val/--test 或 --data-root")


def _detect_feat_cols(df: pd.DataFrame, label: str, kronos_cols: List[str]) -> tuple:
    """自动识别特征列：除 date/symbol/label 外全部视为特征，并把 kronos 列排在前面。"""
    reserved = {"date", "symbol", label}
    all_feats = [c for c in df.columns if c not in reserved]
    k_present = [c for c in kronos_cols if c in all_feats]
    factor_cols = [c for c in all_feats if c not in k_present]
    return k_present + factor_cols, k_present, factor_cols


def train_bundle(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, *,
                 label: str, backend: str, kronos_cols: List[str],
                 out_bundle: str, sources=None) -> dict:
    """在已切分的多标的 fusion 数据上训练 C1 并打包 bundle，返回 manifest。"""
    backend = resolve_backend(backend)
    feat_cols, k_present, factor_cols = _detect_feat_cols(tr, label, kronos_cols)
    if not feat_cols:
        raise ValueError("未识别到任何特征列，请检查 fusion 文件列名")
    if len(tr) == 0:
        raise ValueError("训练集为空")

    model = C1Model(backend, feat_cols).fit(tr[feat_cols].values, tr[label].values)

    metrics = {}
    for name, part in [("val", va), ("test", te)]:
        if len(part) == 0:
            continue
        pred = model.predict(part[feat_cols].values)
        m = _metrics(part[label].values, pred)
        m.update(_grouped_metrics(part, label, pred))
        metrics[name] = m

    symbols = sorted(set(tr.get("symbol", pd.Series(dtype=str)).unique().tolist()))
    manifest = {
        "strategy": "C1",
        "multi_symbol": True,
        "backend": backend,
        "n_symbols": len(symbols),
        "symbols": symbols,
        "feat_cols": feat_cols,
        "kronos_cols": k_present,
        "factor_cols": factor_cols,
        "label": label,
        "sources": sources,
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }

    os.makedirs(out_bundle, exist_ok=True)
    model.save(out_bundle)
    with open(os.path.join(out_bundle, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def predict_bundle(bundle_dir: str, features_csv: str, *, as_of=None,
                   top: int = 10, out_json: str = None) -> dict:
    """上线打分：加载多标的 C1 bundle，对某一交易日的全市场截面打分并排序选股。

    与 run_fusion.py predict 的区别：run_fusion 是**单标的**、从 price-csv 现算 Kronos 特征；
    本函数针对**多标的 dataC bundle**，直接消费已算好的特征宽表（fusion_*.csv 同结构），
    对指定日（默认最新一天）的全部标的横截面打分 → 输出按预测收益排序的选股清单。
    """
    with open(os.path.join(bundle_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    feat_cols = manifest["feat_cols"]
    label = manifest.get("label")
    model = C1Model.load(bundle_dir, manifest["backend"], feat_cols)

    df = pd.read_csv(features_csv, dtype={"symbol": str}, parse_dates=["date"])
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"特征文件缺少 bundle 所需列: {missing}")

    as_of_ts = pd.to_datetime(as_of).normalize() if as_of else df["date"].max()
    cross = df[df["date"] == as_of_ts].copy()
    if cross.empty:
        raise SystemExit(f"特征文件中无 {as_of_ts.date()} 这一天的截面数据")

    cross["pred_fwd_ret"] = model.predict(cross[feat_cols].values)
    cross = cross.sort_values("pred_fwd_ret", ascending=False).reset_index(drop=True)
    cross["rank"] = np.arange(1, len(cross) + 1)

    def _row(r):
        d = {"rank": int(r["rank"]), "symbol": str(r["symbol"]),
             "pred_fwd_ret": float(r["pred_fwd_ret"]),
             "direction": "up" if r["pred_fwd_ret"] > 0 else "down"}
        for c in ("k_up_prob", "k_pred_vol"):
            if c in cross.columns:
                d[c] = float(r[c])
        if label and label in cross.columns and pd.notna(r[label]):
            d["realized_fwd_ret"] = float(r[label])      # 仅当截面带标签时给出（回看参考）
        return d

    ranked = [_row(r) for _, r in cross.iterrows()]
    horizon_days = None
    if label and label.startswith("label_fwd_ret_") and label.endswith("d"):
        try:
            horizon_days = int(label[len("label_fwd_ret_"):-1])
        except ValueError:
            horizon_days = None

    out = {
        "as_of_date": str(as_of_ts.date()),
        "horizon_days": horizon_days,
        "backend": manifest["backend"],
        "multi_symbol": manifest.get("multi_symbol", True),
        "n_symbols": int(len(cross)),
        "long_candidates": ranked[:top],            # 预测最强（做多候选）
        "short_candidates": ranked[-top:][::-1],    # 预测最弱（规避/做空候选）
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _print_metrics(metrics: dict) -> None:
    for split in ("val", "test"):
        m = metrics.get(split)
        if not m:
            continue
        tag = "验证集 (val)" if split == "val" else "测试集 (test)"
        print(f"  {tag}: RMSE={m['RMSE']:.4f} IC={m['IC']:.4f} RankIC={m['RankIC']:.4f} "
              f"Hit={m['Hit']:.3f} | IC_by_date={m['IC_by_date']:.4f} "
              f"RankIC_by_date={m['RankIC_by_date']:.4f}（{m['n_days']}天）")


def _synth_fusion(n_sym=8, n_days=80, seed=0):
    """合成多标的 fusion 三表，仅用于冒烟自测（含真实信号便于验证模型能学到）。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for s in range(n_sym):
        k = rng.standard_normal(n_days) * 0.01
        f = rng.standard_normal(n_days)
        label = 0.6 * k + 0.4 * 0.01 * f + rng.standard_normal(n_days) * 0.005
        rows.append(pd.DataFrame({
            "date": dates, "symbol": f"{s:06d}",
            "k_pred_ret": k, "k_up_prob": rng.uniform(0, 1, n_days),
            "k_pred_vol": rng.uniform(0, 0.02, n_days),
            "fin_roe": f, "local_ret_5d": rng.standard_normal(n_days) * 0.02,
            "label_fwd_ret_5d": label,
        }))
    df = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])
    i1, i2 = int(n_days * 0.7), int(n_days * 0.85)
    d = pd.to_datetime(df["date"])
    tr = df[d < dates[i1]]
    va = df[(d >= dates[i1]) & (d < dates[i2])]
    te = df[d >= dates[i2]]
    return tr, va, te


def _smoke_test() -> None:
    import tempfile
    tr, va, te = _synth_fusion()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = os.path.join(tmp, "c1")
        manifest = train_bundle(tr, va, te, label="label_fwd_ret_5d", backend="auto",
                                kronos_cols=_KRONOS_COLS, out_bundle=bundle,
                                sources=["smoke"])
        assert os.path.exists(os.path.join(bundle, "manifest.json"))
        assert manifest["multi_symbol"] and manifest["n_symbols"] == 8
        assert os.path.exists(os.path.join(bundle, "c1_lgb.txt")) or \
            os.path.exists(os.path.join(bundle, "c1_ridge.npz"))
        # 加载回来并确认可预测
        loaded = C1Model.load(bundle, manifest["backend"], manifest["feat_cols"])
        p = loaded.predict(te[manifest["feat_cols"]].values)
        assert len(p) == len(te)
        print(f"[smoke] train_c1_bundle 通过：后端={manifest['backend']} "
              f"标的={manifest['n_symbols']} 切分={manifest['rows']}")
        _print_metrics(manifest["metrics"])


def main() -> None:
    ap = argparse.ArgumentParser(description="方案C：多标的 C1 可部署 bundle 训练器")
    ap.add_argument("--data-root", default=None, help="dataC 目录（自动找 fusion_{train,val,test}.csv）")
    ap.add_argument("--train", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--label", default="label_fwd_ret_5d", help="标签列（与 step3 --horizon 对齐）")
    ap.add_argument("--horizon", type=int, default=None,
                    help="若给定则用 label_fwd_ret_{H}d 覆盖 --label")
    ap.add_argument("--kronos-cols", default=",".join(_KRONOS_COLS),
                    help="Kronos 特征列（逗号分隔，排在特征前部）")
    ap.add_argument("--backend", default="auto", choices=["auto", "lightgbm", "ridge"])
    ap.add_argument("--out-bundle", default="runs/dataC_c1", help="bundle 输出目录")
    ap.add_argument("--smoke", action="store_true", help="运行无需文件的冒烟自测")
    ap.add_argument("--predict", action="store_true",
                    help="上线打分模式：加载 --out-bundle 的 bundle 对截面特征打分选股")
    ap.add_argument("--features", default=None,
                    help="打分用特征宽表 CSV（默认 {data-root}/fusion_all.csv）")
    ap.add_argument("--as-of", default=None, help="打分的交易日（YYYY-MM-DD，默认取最新一天）")
    ap.add_argument("--top", type=int, default=10, help="多空候选各取前 N 只")
    ap.add_argument("--out-json", default=None, help="打分结果落盘路径")
    args = ap.parse_args()

    if args.smoke:
        _smoke_test()
        return

    if args.predict:
        features = args.features or (
            str(Path(args.data_root) / "fusion_all.csv") if args.data_root else None)
        if not features:
            raise SystemExit("请用 --features 或 --data-root 指定打分用特征宽表")
        out = predict_bundle(args.out_bundle, features, as_of=args.as_of,
                             top=args.top, out_json=args.out_json)
        print(f"[predict] 截面打分 {out['as_of_date']}（{out['n_symbols']} 只，"
              f"后端={out['backend']}，H={out['horizon_days']}日）")
        print("  做多候选（预测最强）:")
        for r in out["long_candidates"]:
            extra = f" 实际={r['realized_fwd_ret']:+.4f}" if "realized_fwd_ret" in r else ""
            print(f"    #{r['rank']:>2} {r['symbol']} 预测={r['pred_fwd_ret']:+.4f} "
                  f"{r['direction']} 上涨概率={r.get('k_up_prob', float('nan')):.2f}{extra}")
        print("  规避/做空候选（预测最弱）:")
        for r in out["short_candidates"]:
            extra = f" 实际={r['realized_fwd_ret']:+.4f}" if "realized_fwd_ret" in r else ""
            print(f"    #{r['rank']:>2} {r['symbol']} 预测={r['pred_fwd_ret']:+.4f} "
                  f"{r['direction']}{extra}")
        if args.out_json:
            print(f"[predict] 结果已写入 {args.out_json}")
        return

    label = f"label_fwd_ret_{args.horizon}d" if args.horizon else args.label
    kronos_cols = [c.strip() for c in args.kronos_cols.split(",") if c.strip()]
    train_p, val_p, test_p = _resolve_inputs(args)
    print(f"[c1] 读取 fusion: train={train_p} val={val_p} test={test_p}")
    tr = pd.read_csv(train_p, dtype={"symbol": str}, parse_dates=["date"])
    va = pd.read_csv(val_p, dtype={"symbol": str}, parse_dates=["date"])
    te = pd.read_csv(test_p, dtype={"symbol": str}, parse_dates=["date"])
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        if label not in part.columns:
            raise SystemExit(f"{name} 缺少标签列 {label}")

    sources = [str(train_p), str(val_p), str(test_p)]
    manifest = train_bundle(tr, va, te, label=label, backend=args.backend,
                            kronos_cols=kronos_cols, out_bundle=args.out_bundle,
                            sources=sources)
    print(f"[c1] 后端={manifest['backend']} 标的={manifest['n_symbols']} "
          f"特征={len(manifest['feat_cols'])} 切分={manifest['rows']}")
    _print_metrics(manifest["metrics"])
    print(f"[c1] bundle 已保存 -> {args.out_bundle}（manifest.json + "
          f"{'c1_lgb.txt' if manifest['backend']=='lightgbm' else 'c1_ridge.npz'}）")


if __name__ == "__main__":
    main()
