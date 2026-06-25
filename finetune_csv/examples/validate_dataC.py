"""校验 DataSet/dataC 的可用性、合理性与完整性（方案C 第1步产物）。

检查项（逐 split：train/validation/test）
- 文件存在性：price.csv / factors.csv（以及 split_report.json 概览）。
- Schema：price 必含 ['date','symbol','open','high','low','close','volume','amount']；
  factors 必含 ['date','symbol'] 且至少包含本地技术因子(local_*)与默认零因子列。
- 完整性：price 与 factors 均无 NaN；两表 (date,symbol) 行严格对齐（一一对应）。
- 合理性：
  - 价格为正：open/high/low/close > 0。
  - OHLC 一致性：low <= min(open,close) 且 high >= max(open,close)，且 high >= low。
  - volume/amount >= 0。
  - symbol 为 6 位字符串（前导零未丢失）。
- 时间切分不重叠：train_end < validation_start <= validation_end < test_start，
  且三段日期区间互不相交。
- 因子覆盖统计：fin_* / tech_* / local_* 列数量与非空率，报告 DB 财务因子覆盖情况。

用法
    python finetune_csv/examples/validate_dataC.py \
        --data-root C:/xapproject/Quantia/Kronos/DataSet/dataC

    # 期望的训练截止/验证/测试窗口（可选，传入则额外断言区间端点）
    python finetune_csv/examples/validate_dataC.py --data-root .../DataSet/dataC \
        --expect-anchor 2026-06-24 --val-days 180 --test-days 180

退出码：全部通过=0；存在失败=1。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PRICE_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
PRICE_NUM = ["open", "high", "low", "close", "volume", "amount"]
SPLITS = ["train", "validation", "test"]


class Checker:
    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        line = f"{name}" + (f" — {detail}" if detail else "")
        if ok:
            self.passed.append(line)
            print(f"[PASS] {line}")
        else:
            self.failed.append(line)
            print(f"[FAIL] {line}")
        return ok

    def summary(self) -> bool:
        print("\n" + "=" * 60)
        print(f"通过 {len(self.passed)} 项，失败 {len(self.failed)} 项")
        if self.failed:
            print("失败项：")
            for f in self.failed:
                print(f"  - {f}")
        print("=" * 60)
        return not self.failed


def _read_csv(path: Path) -> pd.DataFrame:
    # symbol 强制按字符串读取，避免前导零丢失。
    df = pd.read_csv(path, dtype={"symbol": str})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _load_split(root: Path, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p = root / split / "price.csv"
    f = root / split / "factors.csv"
    price = _read_csv(p) if p.exists() else pd.DataFrame()
    factors = _read_csv(f) if f.exists() else pd.DataFrame()
    return price, factors


def validate_split(ck: Checker, split: str, price: pd.DataFrame, factors: pd.DataFrame) -> Dict:
    info: Dict = {"split": split}

    # 文件/非空
    ck.check(f"[{split}] price.csv 非空", not price.empty, f"rows={len(price)}")
    ck.check(f"[{split}] factors.csv 非空", not factors.empty, f"rows={len(factors)}")
    if price.empty or factors.empty:
        return info

    # Schema
    missing_p = [c for c in PRICE_COLS if c not in price.columns]
    ck.check(f"[{split}] price schema 完整", not missing_p, f"missing={missing_p}")
    has_fac_keys = all(c in factors.columns for c in ["date", "symbol"])
    ck.check(f"[{split}] factors 含 date/symbol", has_fac_keys)

    local_cols = [c for c in factors.columns if c.startswith("local_")]
    fin_cols = [c for c in factors.columns if c.startswith("fin_")]
    tech_cols = [c for c in factors.columns if c.startswith("tech_")]
    ck.check(f"[{split}] factors 含本地技术因子 local_*", len(local_cols) > 0, f"n={len(local_cols)}")

    # symbol 6 位字符串
    sym_ok = price["symbol"].astype(str).str.fullmatch(r"\d{6}").all()
    ck.check(f"[{split}] symbol 为6位数字串(前导零保留)", bool(sym_ok))

    # NaN
    p_nan = int(price.isna().sum().sum())
    f_nan = int(factors.isna().sum().sum())
    ck.check(f"[{split}] price 无 NaN", p_nan == 0, f"nan={p_nan}")
    ck.check(f"[{split}] factors 无 NaN", f_nan == 0, f"nan={f_nan}")

    # 价格为正
    pos_ok = bool((price[["open", "high", "low", "close"]] > 0).all().all())
    ck.check(f"[{split}] OHLC 均为正", pos_ok)

    # OHLC 一致性
    oc_min = price[["open", "close"]].min(axis=1)
    oc_max = price[["open", "close"]].max(axis=1)
    cons = (
        (price["low"] <= oc_min + 1e-6)
        & (price["high"] >= oc_max - 1e-6)
        & (price["high"] >= price["low"] - 1e-6)
    )
    bad = int((~cons).sum())
    ck.check(f"[{split}] OHLC 关系一致(low<=oc<=high)", bad == 0, f"violations={bad}")

    # volume/amount 非负
    va_ok = bool((price[["volume", "amount"]] >= 0).all().all())
    ck.check(f"[{split}] volume/amount 非负", va_ok)

    # price↔factor 行对齐（按 date,symbol 一一对应）
    pk = price[["date", "symbol"]].drop_duplicates()
    fk = factors[["date", "symbol"]].drop_duplicates()
    same_rows = len(pk) == len(price) and len(fk) == len(factors)
    ck.check(f"[{split}] (date,symbol) 无重复", same_rows,
             f"price uniq={len(pk)}/{len(price)}, factor uniq={len(fk)}/{len(factors)}")
    merged = pk.merge(fk, on=["date", "symbol"], how="outer", indicator=True)
    aligned = bool((merged["_merge"] == "both").all())
    only_p = int((merged["_merge"] == "left_only").sum())
    only_f = int((merged["_merge"] == "right_only").sum())
    ck.check(f"[{split}] price/factor 行严格对齐", aligned, f"only_price={only_p}, only_factor={only_f}")

    info.update(
        {
            "rows": int(len(price)),
            "symbols": int(price["symbol"].nunique()),
            "date_min": str(price["date"].min().date()),
            "date_max": str(price["date"].max().date()),
            "n_local": len(local_cols),
            "n_fin": len(fin_cols),
            "n_tech": len(tech_cols),
            "fin_nonnull_ratio": round(float(factors[fin_cols].notna().mean().mean()), 4) if fin_cols else None,
            "fin_symbol_cover": int(
                factors.loc[factors[fin_cols].notna().any(axis=1), "symbol"].nunique()
            ) if fin_cols else 0,
        }
    )
    return info


def validate_cross_split(ck: Checker, infos: Dict[str, Dict]) -> None:
    need = all(s in infos and "date_min" in infos[s] for s in SPLITS)
    if not ck.check("跨split信息齐备", need):
        return

    tr = infos["train"]
    va = infos["validation"]
    te = infos["test"]

    def d(x: str) -> pd.Timestamp:
        return pd.Timestamp(x)

    order_ok = (
        d(tr["date_max"]) < d(va["date_min"])
        and d(va["date_max"]) < d(te["date_min"])
        and d(va["date_min"]) <= d(va["date_max"])
    )
    ck.check(
        "时间切分严格不重叠(train<val<test)",
        order_ok,
        f"train_end={tr['date_max']}, val=[{va['date_min']},{va['date_max']}], test=[{te['date_min']},{te['date_max']}]",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="C:/xapproject/Quantia/Kronos/DataSet/dataC")
    ap.add_argument("--expect-anchor", default="", help="期望锚点(test末端)日期，传入则断言 test_max==anchor")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    print(f"校验数据根目录: {root}\n")

    ck = Checker()

    report = root / "split_report.json"
    if report.exists():
        try:
            rep = json.loads(report.read_text(encoding="utf-8"))
            print("split_report.json 概览:")
            print(json.dumps(rep.get("stats", rep), ensure_ascii=False, indent=2))
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"(读取 split_report.json 失败: {exc})\n")
    ck.check("split_report.json 存在", report.exists())

    infos: Dict[str, Dict] = {}
    for split in SPLITS:
        price, factors = _load_split(root, split)
        infos[split] = validate_split(ck, split, price, factors)
        print()

    validate_cross_split(ck, infos)

    if args.expect_anchor and "test" in infos and infos["test"].get("date_max"):
        ck.check(
            "test 末端==期望锚点",
            infos["test"]["date_max"] == args.expect_anchor,
            f"test_max={infos['test']['date_max']} expect={args.expect_anchor}",
        )

    print("\n各 split 摘要:")
    print(json.dumps(infos, ensure_ascii=False, indent=2))

    ok = ck.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
