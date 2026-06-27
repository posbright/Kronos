"""方案C 第2步：用 Kronos 为 dataC 子集标的批量生成衍生特征（CPU 可行版）。

背景
- 全历史全市场逐窗推理在 CPU 上不可行（单次 predict≈0.75s，20只全历史≈11天）。
- 本脚本做「可行子集」：随机选 N 只标的，仅为每只**最近 recent_days 个交易日**生成特征，
  并用较小 samples 控制总时长。用于跑通/验证方案C 第2→6步流程。

输入
- DataSet/dataC/{validation,test}/price.csv（最近窗口足够覆盖 recent_days+lookback+pred）。
- 缓存的 Kronos 权重（默认 ModelScope 优先：AI-ModelScope/*，HF 兜底：NeoQuasar/*）。

输出
- DataSet/dataC/kronos_features.csv：date,symbol,k_pred_ret,k_up_prob,k_pred_vol
- DataSet/dataC/_kronos_parts/{symbol}.csv：逐只增量产物（断点续跑用）。
- DataSet/dataC/kronos_features_report.json：参数、选中标的、设备、耗时、行数。

健壮性
- 逐只增量落盘：每只标的算完立即写 part，进程中断不丢已完成标的。
- 断点续跑：--skip-existing（默认开）复用已有 part，重跑自动从断点继续。

设备选择（CPU / GPU）
- --device auto（默认）：自动优先 CUDA(GPU) -> Apple MPS -> CPU。
- --device cuda:0：强制用 GPU（不可用时回退 CPU 并告警）。
- --device cpu：强制 CPU（自动 set_num_threads 用满核心）。
- GPU 通常比 CPU 快 8~25 倍；脚本会按设备打印耗时外推。
- 注意：本脚本仍是逐只逐窗推理；GPU 下若要进一步提速全市场全历史，
  建议改用 KronosPredictor.predict_batch 做多标的/多窗并行（见文档第3节 GPU 说明）。

用法
    # CPU 子集（默认 auto 在无 GPU 时即 CPU）
    python finetune_csv/build_dataC_step2_kronos_features.py \
        --data-root C:/xapproject/Quantia/Kronos/DataSet/dataC \
        --max-symbols 10 --recent-days 120 --lookback 90 --pred 5 --samples 10 --seed 42

    # GPU（显存够时可放大规模：更多标的 / 更长时间窗 / 更高 samples）
    python finetune_csv/build_dataC_step2_kronos_features.py \
        --device cuda:0 --max-symbols 100 --recent-days 500 --samples 30
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

from build_kronos_features import build_features  # noqa: E402
from kronos_loader import (  # noqa: E402
    DEFAULT_PREDICTOR_LOCAL,
    DEFAULT_TOKENIZER_LOCAL,
    load_kronos_predictor,
)

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _load_recent_price(data_root: Path) -> pd.DataFrame:
    """读取 validation+test 价量（最近两段），按 symbol 拼接，足以覆盖 recent 窗口。"""
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


def _pick_symbols(price: pd.DataFrame, need: int, n: int, seed: int) -> List[str]:
    counts = price.groupby("symbol").size()
    candidates = sorted(counts[counts >= need].index.tolist())
    if not candidates:
        raise RuntimeError(f"无标的满足最少 {need} 行的历史长度要求")
    rng = np.random.default_rng(seed)
    k = min(n, len(candidates))
    chosen = rng.choice(candidates, size=k, replace=False)
    return sorted(map(str, chosen))


def _resolve_device(device: str) -> str:
    """解析设备字符串。

    - ``auto``：优先 CUDA(GPU) -> Apple MPS -> CPU。
    - 显式 ``cuda:0`` / ``mps`` / ``cpu`` 原样返回（cuda 不可用时回退 cpu 并告警）。
    """
    dev = (device or "auto").strip().lower()
    if dev == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print(f"[step2][warn] 指定了 {dev} 但当前环境无可用 CUDA，已回退 cpu。")
        return "cpu"
    return dev


# 单次 predict(sample_count=1) 的经验耗时（秒），用于耗时外推。
# CPU 实测 ~0.75s；GPU 取保守经验值（实际取决于显卡，通常快 8~25 倍）。
_PER_CALL_SEC = {"cpu": 0.75, "cuda": 0.05, "mps": 0.15}



def main() -> None:
    ap = argparse.ArgumentParser(description="方案C 第2步：Kronos 衍生特征（CPU 子集版）")
    ap.add_argument("--data-root", default="C:/xapproject/Quantia/Kronos/DataSet/dataC")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_LOCAL,
                    help="tokenizer 默认目录（model/pretrained/Kronos-Tokenizer-base；缺失则走远端）")
    ap.add_argument("--predictor", default=DEFAULT_PREDICTOR_LOCAL,
                    help="predictor 默认目录（model/pretrained/Kronos-base；缺失则走远端）")
    ap.add_argument("--model-source", choices=["modelscope", "hf"], default="modelscope",
                    help="模型源优先级（默认 modelscope，失败自动回退 hf）")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--max-symbols", type=int, default=10)
    ap.add_argument("--recent-days", type=int, default=120, help="仅为每只标的最近 N 个交易日生成特征")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--pred", type=int, default=5)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto",
                    help="auto(默认: GPU->MPS->CPU) / cuda:0 / mps / cpu")
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                    help="跳过已生成 part 的标的（断点续跑，默认开启）")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_path = Path(args.out) if args.out else data_root / "kronos_features.csv"
    report_path = Path(args.report) if args.report else data_root / "kronos_features_report.json"
    parts_dir = data_root / "_kronos_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"[step2] 运行设备: {device}（请求 --device {args.device}）")

    need = args.recent_days + args.lookback + args.pred
    print(f"[step2] 读取最近价量 (validation+test) ...")
    price = _load_recent_price(data_root)
    symbols = _pick_symbols(price, need=need, n=args.max_symbols, seed=args.seed)
    print(f"[step2] 选中 {len(symbols)} 只标的: {symbols}")

    print(f"[step2] 加载 Kronos 权重（优先 {args.model_source}）...")
    t0 = time.time()
    predictor, load_meta = load_kronos_predictor(
        tokenizer_src=args.tokenizer,
        predictor_src=args.predictor,
        device=device,
        max_context=args.max_context,
        prefer_source=args.model_source,
        verbose=True,
    )
    print(f"[step2] 模型加载耗时 {time.time() - t0:.1f}s")

    per_call = _PER_CALL_SEC.get(device.split(":")[0], 0.75)
    est = len(symbols) * args.recent_days * args.samples * per_call
    print(f"[step2] 预计耗时约 {est / 3600:.2f}h（{len(symbols)}只 × {args.recent_days}窗 × "
          f"{args.samples}样本 × {per_call}s/次@{device.split(':')[0]}）")

    price_by_sym = {s: g for s, g in price[price["symbol"].isin(symbols)].groupby("symbol")}

    all_feats = []
    per_symbol = []
    t_start = time.time()
    for i, sym in enumerate(symbols, start=1):
        part_file = parts_dir / f"{sym}.csv"
        # 断点续跑：已存在的 part 直接复用，避免重复推理。
        if args.skip_existing and part_file.exists():
            feats = pd.read_csv(part_file, dtype={"symbol": str})
            feats["date"] = pd.to_datetime(feats["date"])
            all_feats.append(feats)
            per_symbol.append({"symbol": sym, "rows": int(len(feats)), "seconds": 0.0, "resumed": True})
            print(f"[step2] ({i}/{len(symbols)}) {sym}: 复用已有 part {len(feats)} 行（跳过推理）")
            continue

        g = price_by_sym[sym].sort_values("date").tail(need).reset_index(drop=True)
        px = g.rename(columns={"date": "timestamps"})[["timestamps"] + PRICE_COLS].copy()

        t_sym = time.time()
        feats = build_features(
            predictor, px, symbol=sym,
            lookback=args.lookback, pred_len=args.pred, samples=args.samples,
            verbose=False,
        )
        dt = time.time() - t_sym
        # 逐只增量落盘：即使后续中断，已完成标的不会丢失，可断点续跑。
        feats.to_csv(part_file, index=False)
        all_feats.append(feats)
        per_symbol.append({"symbol": sym, "rows": int(len(feats)), "seconds": round(dt, 1), "resumed": False})
        done = time.time() - t_start
        eta = done / i * (len(symbols) - i)
        print(f"[step2] ({i}/{len(symbols)}) {sym}: {len(feats)} 行, {dt:.1f}s -> {part_file.name} | 已用 {done/60:.1f}min, ETA {eta/60:.1f}min")

    feats_all = pd.concat(all_feats, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats_all.to_csv(out_path, index=False)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(data_root),
        "tokenizer": args.tokenizer,
        "predictor": args.predictor,
        "params": {
            "max_symbols": args.max_symbols,
            "recent_days": args.recent_days,
            "lookback": args.lookback,
            "pred": args.pred,
            "samples": args.samples,
            "seed": args.seed,
            "device": args.device,
            "device_resolved": device,
            "model_source": args.model_source,
        },
        "loaded": load_meta,
        "symbols": symbols,
        "per_symbol": per_symbol,
        "total_rows": int(len(feats_all)),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "out_file": str(out_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[step2] 完成：{len(feats_all)} 行特征 -> {out_path}")
    print(f"[step2] 总耗时 {(time.time() - t_start)/60:.1f}min；报告 -> {report_path}")


if __name__ == "__main__":
    main()
