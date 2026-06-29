"""方案C 第2步（批并行加速版）：用 Kronos.predict_batch 为 dataC 标的批量生成衍生特征。

与逐窗串行版 build_dataC_step2_kronos_features.py 完全等价的产物，但把「同一标的的多个
窗口 × 多次采样」打包成一个 batch 一次前向，**GPU 利用率显著提升**，适合 6G/8G 显存全市场提速。

核心思路
- 每个交易日一个窗口（lookback 历史 -> 预测 pred 步）；每窗需 samples 次采样估不确定性。
- 把 (窗口数 × samples) 个序列堆叠成 batch，分块调用 KronosPredictor.predict_batch，
  按 --batch-size 控制单次并行序列数（即一次进显存的序列条数）。
- 所有序列共享同一 lookback / pred，满足 predict_batch 的等长约束。

输出与逐窗版一致：
- DataSet/dataC/kronos_features.csv：date,symbol,k_pred_ret,k_up_prob,k_pred_vol
- DataSet/dataC/_kronos_parts/{symbol}.csv：逐只增量产物（断点续跑）。
- DataSet/dataC/kronos_features_report.json

显存与 batch 经验（Kronos-base, lookback≈90, pred=5）：
- 6G 显存：--batch-size 96  起步（约 32 窗 × samples3 或 3 窗 × samples30）；OOM 则减半。
- 8G 显存：--batch-size 192 起步；显存富余可上探到 256。
- batch-size 指「并行序列条数」= 同时处理的 (窗口×samples)。samples 越大，单窗占的份额越多。

用法
    python finetune_csv/build_dataC_step2_kronos_features_batch.py \
        --device cuda:0 --max-symbols 300 --recent-days 250 \
        --lookback 90 --pred 5 --samples 30 --batch-size 192 --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kronos_loader import (  # noqa: E402
    DEFAULT_PREDICTOR_LOCAL,
    DEFAULT_TOKENIZER_LOCAL,
    load_kronos_predictor,
)

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _load_recent_price(data_root: Path) -> pd.DataFrame:
    frames = []
    for split in ("validation", "test"):
        f = data_root / split / "price.csv"
        if not f.exists():
            raise FileNotFoundError(f"缺少 {f}")
        df = pd.read_csv(f, dtype={"symbol": str})
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def _pick_symbols(price: pd.DataFrame, need: int, n: int, seed: int, offset: int = 0) -> List[str]:
    """选标的：按 seed 固定打乱全部合格标的，再取 [offset, offset+n) 一段。

    - ``n<=0``：取全市场全部（仅由 offset 偏移）。
    - ``offset``：分批训练用。同一 seed 下 0/300/600... 递进可不重叠覆盖全市场。
    """
    counts = price.groupby("symbol").size()
    candidates = sorted(counts[counts >= need].index.tolist())
    if not candidates:
        raise RuntimeError(f"无标的满足最少 {need} 行的历史长度要求")
    order = np.random.default_rng(seed).permutation(candidates)
    k = len(order) if n <= 0 else min(n, len(order))
    chosen = order[offset:offset + k] if n > 0 else order[offset:]
    return sorted(map(str, chosen))


def _resolve_device(device: str) -> str:
    dev = (device or "auto").strip().lower()
    if dev == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print(f"[step2-batch][warn] 指定了 {dev} 但当前环境无可用 CUDA，已回退 cpu。")
        return "cpu"
    return dev


def _build_symbol_features(predictor, px: pd.DataFrame, symbol: str,
                           lookback: int, pred_len: int, samples: int,
                           batch_size: int) -> pd.DataFrame:
    """对单只标的逐窗采集 close 预测，用 predict_batch 批并行；返回与逐窗版同列特征。"""
    px = px.sort_values("timestamps").reset_index(drop=True)
    ends = list(range(lookback, len(px) - pred_len + 1))
    if not ends:
        return pd.DataFrame(columns=["date", "symbol", "k_pred_ret", "k_up_prob", "k_pred_vol"])

    # 为每个窗口准备 (x_df, x_ts, y_ts, last_close)；每窗复制 samples 次进 batch。
    jobs = []
    meta = []
    for end in ends:
        hist = px.iloc[end - lookback:end]
        x_df = hist[PRICE_COLS].reset_index(drop=True)
        x_ts = hist["timestamps"].reset_index(drop=True)
        y_ts = px["timestamps"].iloc[end:end + pred_len].reset_index(drop=True)
        last_close = float(x_df["close"].iloc[-1])
        meta.append((px["timestamps"].iloc[end].normalize(), last_close))
        jobs.append((x_df, x_ts, y_ts, samples))

    # 展开成扁平 batch 输入。
    df_list, xts_list, yts_list, owner = [], [], [], []
    for wi, (x_df, x_ts, y_ts, s) in enumerate(jobs):
        for _ in range(s):
            df_list.append(x_df); xts_list.append(x_ts); yts_list.append(y_ts); owner.append(wi)

    closes_by_win = [[] for _ in jobs]
    for i in range(0, len(df_list), batch_size):
        sl = slice(i, i + batch_size)
        preds = predictor.predict_batch(
            df_list[sl], xts_list[sl], yts_list[sl], pred_len=pred_len,
            T=1.0, top_p=0.9, sample_count=1, verbose=False,
        )
        for j, p in enumerate(preds):
            closes_by_win[owner[i + j]].append(p["close"].values)

    rows = []
    for wi, (date, last_close) in enumerate(meta):
        preds = np.asarray(closes_by_win[wi], dtype=np.float64)  # [samples, pred_len]
        end_ret = preds[:, -1] / last_close - 1.0
        rows.append({
            "date": date, "symbol": symbol,
            "k_pred_ret": float(end_ret.mean()),
            "k_up_prob": float((end_ret > 0).mean()),
            "k_pred_vol": float(np.std(end_ret)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="方案C 第2步：Kronos 特征（predict_batch 批并行版）")
    ap.add_argument("--data-root", default="DataSet/dataC")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_LOCAL)
    ap.add_argument("--predictor", default=DEFAULT_PREDICTOR_LOCAL)
    ap.add_argument("--model-source", choices=["modelscope", "hf"], default="modelscope")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--max-symbols", type=int, default=10,
                    help="要生成特征的标的数（0 或负数=全市场全量）")
    ap.add_argument("--symbol-offset", type=int, default=0,
                    help="分批起点偏移；同一 seed 下 0/300/600... 递进不重叠覆盖全市场")
    ap.add_argument("--recent-days", type=int, default=120)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--pred", type=int, default=5)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=192,
                    help="单次 predict_batch 的并行序列数；6G建议96、8G建议192，OOM则减半")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto / cuda:0 / mps / cpu")
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_path = Path(args.out) if args.out else data_root / "kronos_features.csv"
    report_path = Path(args.report) if args.report else data_root / "kronos_features_report.json"
    parts_dir = data_root / "_kronos_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"[step2-batch] 运行设备: {device}（请求 --device {args.device}），batch-size={args.batch_size}")

    need = args.recent_days + args.lookback + args.pred
    price = _load_recent_price(data_root)
    symbols = _pick_symbols(price, need=need, n=args.max_symbols, seed=args.seed, offset=args.symbol_offset)
    if not symbols:
        raise SystemExit(f"[step2-batch] 未选中任何标的（--symbol-offset {args.symbol_offset} 可能超出合格标的总数）")
    print(f"[step2-batch] 选中 {len(symbols)} 只标的")

    predictor, load_meta = load_kronos_predictor(
        tokenizer_src=args.tokenizer, predictor_src=args.predictor, device=device,
        max_context=args.max_context, prefer_source=args.model_source, verbose=True,
    )

    price_by_sym = {s: g for s, g in price[price["symbol"].isin(symbols)].groupby("symbol")}
    all_feats, per_symbol = [], []
    t_start = time.time()
    for i, sym in enumerate(symbols, start=1):
        part_file = parts_dir / f"{sym}.csv"
        if args.skip_existing and part_file.exists():
            feats = pd.read_csv(part_file, dtype={"symbol": str})
            feats["date"] = pd.to_datetime(feats["date"])
            all_feats.append(feats)
            per_symbol.append({"symbol": sym, "rows": int(len(feats)), "seconds": 0.0, "resumed": True})
            print(f"[step2-batch] ({i}/{len(symbols)}) {sym}: 复用 part {len(feats)} 行")
            continue
        g = price_by_sym[sym].tail(need).rename(columns={"date": "timestamps"})
        px = g[["timestamps"] + PRICE_COLS].copy()
        t_sym = time.time()
        feats = _build_symbol_features(predictor, px, sym, args.lookback, args.pred,
                                       args.samples, args.batch_size)
        dt = time.time() - t_sym
        feats.to_csv(part_file, index=False)
        all_feats.append(feats)
        per_symbol.append({"symbol": sym, "rows": int(len(feats)), "seconds": round(dt, 1), "resumed": False})
        done = time.time() - t_start
        eta = done / i * (len(symbols) - i)
        print(f"[step2-batch] ({i}/{len(symbols)}) {sym}: {len(feats)} 行, {dt:.1f}s | ETA {eta/60:.1f}min")

    feats_all = pd.concat(all_feats, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats_all.to_csv(out_path, index=False)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(data_root),
        "params": {
            "max_symbols": args.max_symbols, "recent_days": args.recent_days,
            "lookback": args.lookback, "pred": args.pred, "samples": args.samples,
            "batch_size": args.batch_size, "seed": args.seed,
            "device": args.device, "device_resolved": device, "model_source": args.model_source,
        },
        "loaded": load_meta, "symbols": symbols, "per_symbol": per_symbol,
        "total_rows": int(len(feats_all)),
        "elapsed_seconds": round(time.time() - t_start, 1), "out_file": str(out_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step2-batch] 完成：{len(feats_all)} 行 -> {out_path}；总耗时 {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
