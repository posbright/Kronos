"""Build full-market A-share train/validation/test datasets for Kronos.

Design goals:
- Cache first: read local Quantia kline cache under cache/hist.
- DB fallback: if cache is missing/corrupted/unreadable, fallback to cn_stock_spot.
- Enrich from remote DB with technical indicators and fundamentals.
- Low DB concurrency by design: serial queries with configurable sleep/retry.
- Leak-safe split: purge boundary samples whose labels cross split boundaries.

Output layout (under --out-root):
- train/dataset.csv
- validation/dataset.csv
- test/dataset.csv
- build_report.json

Examples:
    python finetune_csv/build_full_market_dataset.py \
        --quantia-root C:/xapproject/Quantia/Quantia \
        --out-root C:/xapproject/Quantia/Kronos/DataSet

    python finetune_csv/build_full_market_dataset.py --smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOG = logging.getLogger("build_full_market_dataset")

SYMBOL_RE = re.compile(r"^(\d{6})(?:qfq)?\.gzip\.pickle$")

# Candidate technical columns in cn_stock_indicators.
CANDIDATE_TECH_COLS = [
    "macd", "macds", "macdh",
    "kdjk", "kdjd", "kdjj",
    "rsi_6", "rsi_12", "rsi", "rsi_24",
    "boll_ub", "boll", "boll_lb",
    "atr", "obv", "cci", "vr", "roc", "rocma",
    "pdi", "mdi", "adx", "adxr",
    "wr_6", "wr_10", "wr_14",
    "mfi", "psy", "psyma", "dma", "sar", "trix",
]

# Candidate fundamental columns in cn_stock_financial.
CANDIDATE_FIN_COLS = [
    "eps", "bps", "ocfps",
    "revenue", "net_profit",
    "revenue_yoy", "net_profit_yoy",
    "roe", "roa", "gross_margin", "net_profit_margin",
    "asset_liability_ratio", "current_ratio", "quick_ratio",
    "total_asset_turnover", "inventory_turnover", "receivable_turnover",
    "rd_ratio", "rd_expense", "admin_expense", "selling_expense", "financial_expense",
]

# Base columns expected for Kronos-compatible kline data.
BASE_KLINE_COLS = [
    "date", "open", "high", "low", "close", "volume",
    "amount", "amplitude", "quote_change", "ups_downs", "turnover",
]


@dataclass
class BuildStats:
    symbols_total: int = 0
    symbols_processed: int = 0
    symbols_skipped: int = 0
    cache_success: int = 0
    cache_failed: int = 0
    db_kline_fallback: int = 0
    symbols_with_tech: int = 0
    symbols_with_fin: int = 0
    split_rows: Dict[str, int] = field(default_factory=lambda: {"train": 0, "validation": 0, "test": 0})
    split_symbols: Dict[str, set] = field(default_factory=lambda: {"train": set(), "validation": set(), "test": set()})
    split_min_date: Dict[str, Optional[str]] = field(default_factory=lambda: {"train": None, "validation": None, "test": None})
    split_max_date: Dict[str, Optional[str]] = field(default_factory=lambda: {"train": None, "validation": None, "test": None})


class QuantiaDataProvider:
    """Light wrapper around Quantia DB access for low-concurrency reads."""

    def __init__(self, quantia_root: Path, db_sleep: float, db_retries: int):
        self.quantia_root = quantia_root
        self.db_sleep = max(0.0, db_sleep)
        self.db_retries = max(1, db_retries)
        self.available = False
        self._table_cols_cache: Dict[str, set] = {}

        quantia_pkg_parent = str(quantia_root)
        if quantia_pkg_parent not in sys.path:
            sys.path.insert(0, quantia_pkg_parent)

        try:
            import quantia.lib.database as mdb  # type: ignore

            self.mdb = mdb
            self.available = True
            LOG.info("Quantia DB provider is available.")
        except Exception as exc:
            self.available = False
            self.mdb = None
            LOG.warning("Quantia DB provider import failed: %s", exc)

    def _query_df(self, sql: str, params: Sequence | None = None) -> pd.DataFrame:
        if not self.available:
            return pd.DataFrame()

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.db_retries + 1):
            try:
                with self.mdb.get_connection() as conn:
                    df = pd.read_sql(sql, conn, params=params)
                if self.db_sleep > 0:
                    time.sleep(self.db_sleep)
                return df
            except Exception as exc:
                last_exc = exc
                LOG.warning("DB query failed (attempt %d/%d): %s", attempt, self.db_retries, exc)
                if self.db_sleep > 0:
                    time.sleep(self.db_sleep)

        LOG.error("DB query failed after retries: %s", last_exc)
        return pd.DataFrame()

    def table_exists(self, table_name: str) -> bool:
        if not self.available:
            return False
        try:
            return bool(self.mdb.checkTableIsExist(table_name))
        except Exception:
            return False

    def table_columns(self, table_name: str) -> set:
        if table_name in self._table_cols_cache:
            return self._table_cols_cache[table_name]

        if not self.table_exists(table_name):
            self._table_cols_cache[table_name] = set()
            return set()

        sql = (
            "SELECT COLUMN_NAME "
            "FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s"
        )
        schema = getattr(self.mdb, "db_database", None)
        df = self._query_df(sql, params=(schema, table_name))
        cols = set(df["COLUMN_NAME"].astype(str).tolist()) if not df.empty else set()
        self._table_cols_cache[table_name] = cols
        return cols

    def fetch_symbols_from_spot(self) -> List[str]:
        if not self.table_exists("cn_stock_spot"):
            return []
        sql = (
            "SELECT DISTINCT `code` FROM `cn_stock_spot` "
            "WHERE `code` REGEXP '^[0-9]{6}$'"
        )
        df = self._query_df(sql)
        if df.empty or "code" not in df.columns:
            return []
        return sorted(df["code"].astype(str).str.zfill(6).tolist())

    def fetch_kline_from_spot(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.table_exists("cn_stock_spot"):
            return pd.DataFrame()

        sql = (
            "SELECT `date`, "
            "`open_price` AS `open`, `high_price` AS `high`, `low_price` AS `low`, "
            "`new_price` AS `close`, `volume`, `deal_amount` AS `amount`, "
            "`amplitude`, `change_rate` AS `quote_change`, `ups_downs`, `turnoverrate` AS `turnover` "
            "FROM `cn_stock_spot` "
            "WHERE `code` = %s AND `date` >= %s AND `date` <= %s "
            "ORDER BY `date`"
        )
        df = self._query_df(sql, params=(symbol, start_date, end_date))
        return normalize_kline_df(df)

    def fetch_indicators(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        table = "cn_stock_indicators"
        cols = self.table_columns(table)
        if not cols:
            return pd.DataFrame()

        selected = [c for c in CANDIDATE_TECH_COLS if c in cols]
        if not selected:
            return pd.DataFrame()

        select_sql = ", ".join(["`date`", "`code`"] + [f"`{c}`" for c in selected])
        sql = (
            f"SELECT {select_sql} FROM `{table}` "
            "WHERE `code` = %s AND `date` >= %s AND `date` <= %s "
            "ORDER BY `date`"
        )
        df = self._query_df(sql, params=(symbol, start_date, end_date))
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        rename_map = {c: f"tech_{c}" for c in selected}
        return df.rename(columns=rename_map)

    def fetch_financial(self, symbol: str, end_date: str) -> pd.DataFrame:
        table = "cn_stock_financial"
        cols = self.table_columns(table)
        if not cols:
            return pd.DataFrame()

        selected = [c for c in CANDIDATE_FIN_COLS if c in cols]
        if not selected or "report_date" not in cols:
            return pd.DataFrame()

        select_sql = ", ".join(["`report_date`", "`code`"] + [f"`{c}`" for c in selected])
        sql = (
            f"SELECT {select_sql} FROM `{table}` "
            "WHERE `code` = %s AND `report_date` <= %s "
            "ORDER BY `report_date`"
        )
        df = self._query_df(sql, params=(symbol, end_date))
        if df.empty:
            return df

        df = df.rename(columns={"report_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        rename_map = {c: f"fin_{c}" for c in selected}
        return df.rename(columns=rename_map)


def ensure_out_layout(out_root: Path) -> Dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    split_paths = {}
    for split in ("train", "validation", "test"):
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_paths[split] = split_dir / "dataset.csv"
    return split_paths


def scan_cache_symbols(cache_hist_root: Path) -> List[str]:
    symbols = set()

    # Root-level files.
    for p in cache_hist_root.glob("*.gzip.pickle"):
        m = SYMBOL_RE.match(p.name)
        if m:
            symbols.add(m.group(1))

    # Prefix folders: 000, 001, 300, 600, etc. Exclude index.
    for child in cache_hist_root.iterdir():
        if not child.is_dir() or child.name == "index":
            continue
        for p in child.glob("*.gzip.pickle"):
            m = SYMBOL_RE.match(p.name)
            if m:
                symbols.add(m.group(1))

    return sorted(symbols)


def normalize_volume_to_shares(df: pd.DataFrame) -> pd.DataFrame:
    if "volume" not in df.columns:
        return df

    v = pd.to_numeric(df["volume"], errors="coerce")
    positive = v[v > 0]
    if positive.empty:
        df["volume"] = v.fillna(0.0)
        return df

    # Heuristic: cache data may store volume in lots (100 shares).
    med = float(positive.median())
    if med < 2_000_000:
        v = v * 100.0
    df["volume"] = v
    return df


def normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=BASE_KLINE_COLS)

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=BASE_KLINE_COLS)

    out = df.copy()

    rename_map = {
        "timestamps": "date",
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "new_price": "close",
        "deal_amount": "amount",
        "pct_chg": "quote_change",
        "change": "ups_downs",
        "turnoverrate": "turnover",
    }
    out = out.rename(columns=rename_map)

    if "date" not in out.columns:
        if out.index.name in ("date", "timestamps"):
            out = out.reset_index().rename(columns={out.index.name: "date"})
        elif isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "date"})

    for col in BASE_KLINE_COLS:
        if col not in out.columns:
            out[col] = np.nan

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    for c in ["open", "high", "low", "close", "volume", "amount", "amplitude", "quote_change", "ups_downs", "turnover"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = normalize_volume_to_shares(out)
    out = out.dropna(subset=["date", "close"]) 
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out[BASE_KLINE_COLS]


def load_kline_from_cache(symbol: str, cache_hist_root: Path) -> Tuple[pd.DataFrame, str]:
    prefix = symbol[:3]
    candidates = [
        cache_hist_root / prefix / f"{symbol}qfq.gzip.pickle",
        cache_hist_root / prefix / f"{symbol}.gzip.pickle",
        cache_hist_root / f"{symbol}qfq.gzip.pickle",
        cache_hist_root / f"{symbol}.gzip.pickle",
    ]

    for path in candidates:
        if not path.exists():
            continue
        corrupt_marker = Path(str(path) + ".corrupt")
        if corrupt_marker.exists():
            continue
        try:
            raw = pd.read_pickle(path, compression="gzip")
            return normalize_kline_df(raw), str(path)
        except Exception as exc:
            LOG.debug("Failed to read cache %s: %s", path, exc)
            continue

    return pd.DataFrame(columns=BASE_KLINE_COLS), ""


def add_local_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change(1)
    out["ret_5d"] = out["close"].pct_change(5)
    out["ma_5"] = out["close"].rolling(5).mean()
    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_20"] = out["close"].rolling(20).mean()
    out["vol_ma_5"] = out["volume"].rolling(5).mean()
    out["amt_ma_5"] = out["amount"].rolling(5).mean()
    out["hl_spread"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_change"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    return out


def merge_symbol_features(
    symbol: str,
    kline_df: pd.DataFrame,
    tech_df: pd.DataFrame,
    fin_df: pd.DataFrame,
    horizon: int,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> pd.DataFrame:
    if kline_df.empty:
        return pd.DataFrame()

    base = kline_df.copy().sort_values("date").reset_index(drop=True)
    base["symbol"] = symbol

    if not tech_df.empty:
        tech_cols = [c for c in tech_df.columns if c not in ("date", "code")]
        tech_merge = tech_df[["date"] + tech_cols].drop_duplicates(subset=["date"], keep="last")
        base = base.merge(tech_merge, on="date", how="left")

    if not fin_df.empty:
        fin_cols = [c for c in fin_df.columns if c not in ("date", "code")]
        fin_merge = fin_df[["date"] + fin_cols].drop_duplicates(subset=["date"], keep="last")
        base = pd.merge_asof(
            base.sort_values("date"),
            fin_merge.sort_values("date"),
            on="date",
            direction="backward",
        )

    base = add_local_features(base)

    base["future_close"] = base["close"].shift(-horizon)
    base["future_date"] = base["date"].shift(-horizon)
    label_col = f"label_fwd_ret_{horizon}d"
    base[label_col] = (base["future_close"] / base["close"]) - 1.0

    fill_cols = [c for c in base.columns if c.startswith("tech_") or c.startswith("fin_")]
    if fill_cols:
        base[fill_cols] = base[fill_cols].ffill()

    # Leak-safe split assignment with boundary purge.
    date_s = base["date"]
    future_s = base["future_date"]

    train_mask = (date_s <= train_end) & (future_s <= train_end)
    val_mask = (date_s > train_end) & (date_s <= val_end) & (future_s <= val_end)
    test_mask = date_s > val_end

    base["split"] = np.where(train_mask, "train", np.where(val_mask, "validation", np.where(test_mask, "test", "drop")))

    base = base[base["split"] != "drop"].copy()
    base = base.dropna(subset=[label_col, "future_date"]).reset_index(drop=True)

    base["date"] = base["date"].dt.strftime("%Y-%m-%d")
    base["future_date"] = pd.to_datetime(base["future_date"]).dt.strftime("%Y-%m-%d")

    return base


def append_split_csv(split_file: Path, df: pd.DataFrame) -> None:
    if df.empty:
        # Materialize an empty file for downstream workflow consistency.
        if not split_file.exists():
            pd.DataFrame().to_csv(split_file, index=False)
        return
    write_header = not split_file.exists()
    df.to_csv(split_file, mode="a", header=write_header, index=False)


def update_stats_dates(stats: BuildStats, split: str, min_date: str, max_date: str) -> None:
    if stats.split_min_date[split] is None or min_date < stats.split_min_date[split]:
        stats.split_min_date[split] = min_date
    if stats.split_max_date[split] is None or max_date > stats.split_max_date[split]:
        stats.split_max_date[split] = max_date


def build_dataset(args: argparse.Namespace) -> Dict:
    out_root = Path(args.out_root).resolve()
    split_paths = ensure_out_layout(out_root)

    provider = QuantiaDataProvider(
        quantia_root=Path(args.quantia_root).resolve(),
        db_sleep=args.db_sleep,
        db_retries=args.db_retries,
    )

    cache_hist_root = Path(args.cache_hist_root).resolve()
    if not cache_hist_root.exists():
        raise FileNotFoundError(f"Cache root not found: {cache_hist_root}")

    cache_symbols = scan_cache_symbols(cache_hist_root)
    db_symbols = provider.fetch_symbols_from_spot() if (provider.available and args.scan_db_symbols) else []

    symbols = sorted(set(cache_symbols).union(db_symbols))
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    # Split boundaries come from CLI args (inclusive end dates).
    # train: date <= train_end; validation: train_end < date <= val_end; test: date > val_end.
    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.validation_end)

    stats = BuildStats(symbols_total=len(symbols))
    t0 = time.time()

    LOG.info("Total symbols to process: %d", len(symbols))
    LOG.info("Split boundaries: train_end=%s, validation_end=%s", args.train_end, args.validation_end)

    for idx, symbol in enumerate(symbols, start=1):
        kline_df, cache_path = load_kline_from_cache(symbol, cache_hist_root)
        source = "cache"

        if not kline_df.empty:
            stats.cache_success += 1
        else:
            stats.cache_failed += 1
            if provider.available and not args.disable_db_fallback_kline:
                kline_df = provider.fetch_kline_from_spot(symbol, args.start_date, args.end_date)
                if not kline_df.empty:
                    source = "db_spot"
                    stats.db_kline_fallback += 1

        if kline_df.empty:
            stats.symbols_skipped += 1
            if idx % args.progress_every == 0:
                LOG.info("Progress %d/%d, processed=%d, skipped=%d", idx, len(symbols), stats.symbols_processed, stats.symbols_skipped)
            continue

        kline_df = kline_df[(kline_df["date"] >= pd.Timestamp(args.start_date)) & (kline_df["date"] <= pd.Timestamp(args.end_date))].copy()
        if len(kline_df) < args.label_horizon + 20:
            stats.symbols_skipped += 1
            continue

        tech_df = pd.DataFrame()
        fin_df = pd.DataFrame()

        if provider.available and not args.disable_db_features:
            tech_df = provider.fetch_indicators(symbol, args.start_date, args.end_date)
            if not tech_df.empty:
                stats.symbols_with_tech += 1

            fin_df = provider.fetch_financial(symbol, args.end_date)
            if not fin_df.empty:
                stats.symbols_with_fin += 1

        merged = merge_symbol_features(
            symbol=symbol,
            kline_df=kline_df,
            tech_df=tech_df,
            fin_df=fin_df,
            horizon=args.label_horizon,
            train_end=train_end,
            val_end=val_end,
        )

        if merged.empty:
            stats.symbols_skipped += 1
            continue

        merged["kline_source"] = source
        merged["cache_path"] = cache_path

        for split in ("train", "validation", "test"):
            part = merged[merged["split"] == split].copy()
            if part.empty:
                continue
            append_split_csv(split_paths[split], part)
            stats.split_rows[split] += len(part)
            stats.split_symbols[split].add(symbol)
            update_stats_dates(stats, split, part["date"].min(), part["date"].max())

        stats.symbols_processed += 1

        if idx % args.progress_every == 0:
            LOG.info(
                "Progress %d/%d, processed=%d, skipped=%d, train/val/test=%d/%d/%d",
                idx,
                len(symbols),
                stats.symbols_processed,
                stats.symbols_skipped,
                stats.split_rows["train"],
                stats.split_rows["validation"],
                stats.split_rows["test"],
            )

    elapsed = round(time.time() - t0, 2)

    report = {
        "run_time_seconds": elapsed,
        "quantia_root": str(Path(args.quantia_root).resolve()),
        "cache_hist_root": str(cache_hist_root),
        "output_root": str(out_root),
        "date_range": {"start_date": args.start_date, "end_date": args.end_date},
        "split_boundaries": {
            "train_end": args.train_end,
            "validation_end": args.validation_end,
            "test_start": str((pd.Timestamp(args.validation_end) + pd.Timedelta(days=1)).date()),
        },
        "label_horizon": args.label_horizon,
        "symbols": {
            "total": stats.symbols_total,
            "processed": stats.symbols_processed,
            "skipped": stats.symbols_skipped,
            "cache_success": stats.cache_success,
            "cache_failed": stats.cache_failed,
            "db_kline_fallback": stats.db_kline_fallback,
            "with_tech_features": stats.symbols_with_tech,
            "with_fin_features": stats.symbols_with_fin,
        },
        "splits": {
            split: {
                "rows": stats.split_rows[split],
                "symbols": len(stats.split_symbols[split]),
                "min_date": stats.split_min_date[split],
                "max_date": stats.split_max_date[split],
                "file": str(split_paths[split]),
            }
            for split in ("train", "validation", "test")
        },
    }

    report_path = out_root / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOG.info("Build completed in %.2fs", elapsed)
    LOG.info("Rows train/validation/test: %d / %d / %d", stats.split_rows["train"], stats.split_rows["validation"], stats.split_rows["test"])
    LOG.info("Report written: %s", report_path)

    return report


def run_smoke(args: argparse.Namespace) -> None:
    np.random.seed(7)

    # Synthetic daily data for 2 symbols.
    dates = pd.date_range("2022-01-01", periods=160, freq="D")

    def make_symbol_df(seed_shift: float) -> pd.DataFrame:
        base = 10.0 + np.cumsum(np.random.randn(len(dates)) * 0.2 + seed_shift)
        df = pd.DataFrame(
            {
                "date": dates,
                "open": base * (1 - 0.005),
                "high": base * (1 + 0.01),
                "low": base * (1 - 0.01),
                "close": base,
                "volume": np.random.randint(200000, 1000000, len(dates)).astype(float),
                "amount": np.random.randint(1_000_000, 10_000_000, len(dates)).astype(float),
                "amplitude": np.random.rand(len(dates)) * 8,
                "quote_change": np.random.randn(len(dates)),
                "ups_downs": np.random.randn(len(dates)) * 0.1,
                "turnover": np.random.rand(len(dates)) * 5,
            }
        )
        return df

    tech = pd.DataFrame(
        {
            "date": dates,
            "code": "000001",
            "tech_macd": np.random.randn(len(dates)),
            "tech_rsi": np.random.rand(len(dates)) * 100,
        }
    )
    fin_dates = pd.to_datetime(["2021-12-31", "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"])
    fin = pd.DataFrame(
        {
            "date": fin_dates,
            "code": "000001",
            "fin_roe": [8.1, 8.3, 8.7, 9.0, 9.2],
            "fin_revenue_yoy": [12.0, 12.3, 12.6, 12.8, 13.1],
        }
    )

    # Keep smoke independent from CLI defaults and force all three splits to appear.
    train_end = dates[79]
    val_end = dates[119]

    s1 = merge_symbol_features("000001", make_symbol_df(0.0), tech, fin, args.label_horizon, train_end, val_end)
    s2 = merge_symbol_features("000002", make_symbol_df(0.03), pd.DataFrame(), pd.DataFrame(), args.label_horizon, train_end, val_end)
    all_df = pd.concat([s1, s2], ignore_index=True)

    assert not all_df.empty, "smoke: merged dataset is empty"
    label_col = f"label_fwd_ret_{args.label_horizon}d"
    assert label_col in all_df.columns, "smoke: label column missing"
    assert all_df[label_col].notna().all(), "smoke: label has NaN"
    assert set(all_df["split"].unique()) <= {"train", "validation", "test"}, "smoke: invalid split"
    assert (all_df["split"] == "train").any(), "smoke: train split empty"
    assert (all_df["split"] == "validation").any(), "smoke: validation split empty"
    assert (all_df["split"] == "test").any(), "smoke: test split empty"

    tmp_out = Path(args.out_root).resolve() / "_smoke"
    split_files = ensure_out_layout(tmp_out)
    for split in ("train", "validation", "test"):
        append_split_csv(split_files[split], all_df[all_df["split"] == split])

    for split in ("train", "validation", "test"):
        assert split_files[split].exists(), f"smoke: missing file for {split}"

    print(
        "[smoke] build_full_market_dataset passed: "
        f"rows={len(all_df)}, train/validation/test="
        f"{len(all_df[all_df['split']=='train'])}/"
        f"{len(all_df[all_df['split']=='validation'])}/"
        f"{len(all_df[all_df['split']=='test'])}, out={tmp_out}"
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_quantia_root = Path("C:/xapproject/Quantia/Quantia")
    default_cache_root = default_quantia_root / "quantia" / "cache" / "hist"

    parser = argparse.ArgumentParser(description="Build full-market A-share datasets for Kronos")
    parser.add_argument("--quantia-root", default=str(default_quantia_root), help="Quantia repository root")
    parser.add_argument("--cache-hist-root", default=str(default_cache_root), help="Quantia cache/hist root")
    parser.add_argument("--out-root", default=str(repo_root / "DataSet"), help="Output root for train/validation/test")

    parser.add_argument("--start-date", default="2017-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=str(dt.date.today()), help="End date (YYYY-MM-DD)")

    parser.add_argument("--train-end", default="2022-12-31", help="Train split end date (inclusive)")
    parser.add_argument("--validation-end", default="2024-12-31", help="Validation split end date (inclusive)")
    parser.add_argument("--label-horizon", type=int, default=5, help="Forward return horizon in trading days")

    parser.add_argument("--max-symbols", type=int, default=0, help="Limit number of symbols (0 means all)")
    parser.add_argument("--scan-db-symbols", action="store_true", help="Union symbols from cn_stock_spot")
    parser.add_argument("--disable-db-fallback-kline", action="store_true", help="Disable kline fallback from cn_stock_spot")
    parser.add_argument("--disable-db-features", action="store_true", help="Disable DB feature enrichment")

    parser.add_argument("--db-sleep", type=float, default=0.05, help="Sleep seconds after each DB query")
    parser.add_argument("--db-retries", type=int, default=3, help="Retry attempts per DB query")

    parser.add_argument("--progress-every", type=int, default=100, help="Log progress every N symbols")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.label_horizon <= 0:
        raise ValueError("--label-horizon must be > 0")

    if pd.Timestamp(args.train_end) >= pd.Timestamp(args.validation_end):
        raise ValueError("--train-end must be earlier than --validation-end")

    if args.smoke:
        run_smoke(args)
        return

    report = build_dataset(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
