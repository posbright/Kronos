"""方案C 第1步：从 Quantia cache/DB 构建价量CSV与因子CSV（按最新日期向后回推切分）。

目标
- 从 Quantia 缓存优先获取全市场价量数据（cache/hist）。
- 缓存缺失/损坏时回退到远程数据库 cn_stock_spot。
- 因子优先从可用来源获取：
  1) cache 可直接提供的列（amplitude/quote_change/ups_downs/turnover）
  2) 远程 DB 的 cn_stock_indicators（技术因子）
  3) 远程 DB 的 cn_stock_financial（财务因子，按披露日向后对齐）
- 缺失值策略：能重算则重算（本地技术因子），无法获取则按稳定策略填充，确保后续训练不因 NaN 中断。
- 输出到 DataSet/dataC/{train,validation,test}/，每个 split 含：
  - price.csv   : date,symbol,open,high,low,close,volume,amount
  - factors.csv : date,symbol,<factor...>
  - split_report.json

切分逻辑（按最新日期向后回推）
- 先确定锚点日期 anchor_date（默认自动取全量数据最大 date）。
- test 集：anchor_date 向前 test_days 天（含锚点）
- validation 集：test 之前 val_days 天
- train 集：validation 之前的全部历史（可受 start_date 限制）

示例
    python finetune_csv/build_dataC_step1_from_quantia.py \
        --quantia-root C:/xapproject/Quantia/Quantia \
        --out-root C:/xapproject/Quantia/Kronos/DataSet/dataC \
        --test-days 120 --val-days 120

    # 冒烟（不依赖外部项目）
    python finetune_csv/build_dataC_step1_from_quantia.py --smoke
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# 复用已验证的 Quantia 访问与 K 线标准化能力，避免重复实现。
from build_full_market_dataset import (  # noqa: E402
    BASE_KLINE_COLS,
    CANDIDATE_FIN_COLS,
    QuantiaDataProvider,
    add_local_features,
    load_kline_from_cache,
    normalize_kline_df,
    scan_cache_symbols,
)

LOG = logging.getLogger("build_dataC_step1")

PRICE_OUT_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
CACHE_FACTOR_COLS = ["amplitude", "quote_change", "ups_downs", "turnover"]
DEFAULT_ZERO_FACTORS = ["news_sent", "news_count", "event_flag"]


def _strip_quotes(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
        return s[1:-1]
    return s


def _read_simple_dotenv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = _strip_quotes(v.strip())
    return out


def _is_local_host(host: str) -> bool:
    h = (host or "").strip().lower()
    return h in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _apply_db_env_from_quantia(args: argparse.Namespace) -> Dict[str, str]:
    """从 Quantia 项目 .env 注入 DB 连接配置，默认优先使用远程 DB。"""
    quantia_root = Path(args.quantia_root).resolve()
    env_file = quantia_root / ".env"
    env_kv = _read_simple_dotenv(env_file)

    chosen: Dict[str, str] = {}
    keys = [
        "QUANTIA_DB_HOST",
        "QUANTIA_DB_USER",
        "QUANTIA_DB_PASSWORD",
        "QUANTIA_DB_DATABASE",
        "QUANTIA_DB_PORT",
        "QUANTIA_DB_CHARSET",
    ]
    for k in keys:
        if k in env_kv:
            chosen[k] = env_kv[k]

    # CLI 显式覆盖优先级最高。
    if args.db_host:
        chosen["QUANTIA_DB_HOST"] = args.db_host
    if args.db_user:
        chosen["QUANTIA_DB_USER"] = args.db_user
    if args.db_password:
        chosen["QUANTIA_DB_PASSWORD"] = args.db_password
    if args.db_database:
        chosen["QUANTIA_DB_DATABASE"] = args.db_database
    if args.db_port:
        chosen["QUANTIA_DB_PORT"] = str(args.db_port)
    if args.db_charset:
        chosen["QUANTIA_DB_CHARSET"] = args.db_charset

    host = chosen.get("QUANTIA_DB_HOST", "")
    # 默认策略：优先远程。若 .env 仅给了本地地址，且没有显式覆盖，则不强制注入。
    if args.prefer_remote_db and host and _is_local_host(host) and not args.db_host:
        LOG.warning("检测到 .env 的 DB_HOST 为本地地址(%s)，未启用远程优先覆盖", host)
        return {}

    # 强制写入当前进程环境，覆盖已有系统环境，避免被本地配置抢占。
    for k, v in chosen.items():
        os.environ[k] = str(v)

    masked = dict(chosen)
    if "QUANTIA_DB_PASSWORD" in masked:
        masked["QUANTIA_DB_PASSWORD"] = "***"
    LOG.info("DB 配置来源: %s", env_file)
    LOG.info("DB 生效参数: %s", json.dumps(masked, ensure_ascii=False))
    return chosen


def _check_db_connectivity(timeout_sec: int = 5) -> Dict[str, str]:
    """快速验证数据库连通性与认证（SELECT 1）。"""
    host = os.environ.get("QUANTIA_DB_HOST", "")
    user = os.environ.get("QUANTIA_DB_USER", "")
    password = os.environ.get("QUANTIA_DB_PASSWORD", "")
    database = os.environ.get("QUANTIA_DB_DATABASE", "")
    port = int(os.environ.get("QUANTIA_DB_PORT", "3306"))
    charset = os.environ.get("QUANTIA_DB_CHARSET", "utf8mb4")

    report = {
        "host": host,
        "port": str(port),
        "database": database,
        "user": user,
        "socket_connect": "fail",
        "mysql_auth": "fail",
        "select_1": "fail",
    }

    # 1) TCP 可达性。
    sock = socket.create_connection((host, port), timeout=timeout_sec)
    sock.close()
    report["socket_connect"] = "ok"

    # 2) MySQL 认证 + 查询。
    import pymysql  # 局部导入，未安装时给出明确错误。

    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
        charset=charset,
        connect_timeout=timeout_sec,
        read_timeout=timeout_sec,
        write_timeout=timeout_sec,
        autocommit=True,
    )
    report["mysql_auth"] = "ok"
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        _ = cur.fetchone()
    report["select_1"] = "ok"
    conn.close()
    return report


@dataclass
class SplitRange:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def ensure_layout(out_root: Path) -> Dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "validation", "test"):
        d = out_root / split
        d.mkdir(parents=True, exist_ok=True)
        out[split] = d
    return out


def pick_anchor_date(rows: List[pd.DataFrame], configured: Optional[str]) -> pd.Timestamp:
    if configured:
        return pd.Timestamp(configured).normalize()
    mx = None
    for df in rows:
        if df.empty:
            continue
        cur = pd.to_datetime(df["date"]).max()
        if pd.notna(cur):
            mx = cur if mx is None else max(mx, cur)
    if mx is None:
        raise ValueError("无法确定 anchor_date：没有任何有效数据")
    return pd.Timestamp(mx).normalize()


def make_ranges(anchor: pd.Timestamp, val_days: int, test_days: int, start_date: Optional[str]) -> List[SplitRange]:
    anchor = pd.Timestamp(anchor).normalize()
    val_days = int(val_days)
    test_days = int(test_days)
    if val_days <= 0 or test_days <= 0:
        raise ValueError("--val-days 和 --test-days 必须 > 0")

    test_end = anchor
    test_start = anchor - dt.timedelta(days=test_days - 1)

    val_end = test_start - dt.timedelta(days=1)
    val_start = val_end - dt.timedelta(days=val_days - 1)

    train_end = val_start - dt.timedelta(days=1)
    train_start = pd.Timestamp(start_date).normalize() if start_date else pd.Timestamp("1900-01-01")

    if train_end < train_start:
        raise ValueError("切分窗口过大导致 train 区间为空，请减小 val/test 天数或放宽 start_date")

    return [
        SplitRange("train", train_start, train_end),
        SplitRange("validation", val_start, val_end),
        SplitRange("test", test_start, test_end),
    ]


def split_by_ranges(df: pd.DataFrame, ranges: Sequence[SplitRange]) -> Dict[str, pd.DataFrame]:
    d = pd.to_datetime(df["date"]).dt.normalize()
    out = {}
    for r in ranges:
        m = (d >= r.start) & (d <= r.end)
        part = df[m].copy().sort_values(["date", "symbol"]).reset_index(drop=True)
        out[r.name] = part
    return out


def _fill_factor_na(factors: pd.DataFrame, zero_factor_cols: Sequence[str]) -> pd.DataFrame:
    if factors.empty:
        return factors

    out = factors.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    all_cols = [c for c in out.columns if c not in ("date", "symbol")]

    for col in all_cols:
        # 先按标的前向填充，避免跨标的串值。
        out[col] = out.groupby("symbol")[col].ffill()

        if col in zero_factor_cols:
            out[col] = out[col].fillna(0.0)
            continue

        # 对仍缺失的列，优先用全局中位数（稳健），再兜底 0。
        med = pd.to_numeric(out[col], errors="coerce").median()
        if pd.isna(med):
            med = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(float(med))

    return out


def build_local_factors_from_price(px: pd.DataFrame) -> pd.DataFrame:
    # 基于价量可重算的技术因子：尽量避免依赖外部库。
    feat = add_local_features(px)
    cols = [
        "date",
        "ret_1d",
        "ret_5d",
        "ma_5",
        "ma_10",
        "ma_20",
        "vol_ma_5",
        "amt_ma_5",
        "hl_spread",
        "oc_change",
    ]
    out = feat[[c for c in cols if c in feat.columns]].copy()
    rename = {c: f"local_{c}" for c in out.columns if c != "date"}
    out = out.rename(columns=rename)
    return out


def fetch_financial_bulk(
    provider: QuantiaDataProvider,
    end_date: str,
    symbols: Optional[Sequence[str]] = None,
    chunk: int = 400,
) -> pd.DataFrame:
    """分批拉取全市场财务因子（按 code+report_date），避免逐股串行慢查询。

    远程服务器对一次性 32 万行的巨查询会触发 "Lost connection during query"，
    因此按 code 分批（每批 ``chunk`` 只，使用 ``WHERE code IN (...)``）发起小查询，
    单条查询很小、稳健可重试。当未提供 ``symbols`` 时回退为单条全量查询。

    返回列：symbol, date(=report_date), fin_*。无数据时返回空表。
    """
    if not getattr(provider, "available", False):
        return pd.DataFrame()
    table = "cn_stock_financial"
    cols = provider.table_columns(table)
    if not cols or "report_date" not in cols or "code" not in cols:
        return pd.DataFrame()

    selected = [c for c in CANDIDATE_FIN_COLS if c in cols]
    if not selected:
        return pd.DataFrame()

    select_cols = ", ".join(["`code`", "`report_date`"] + [f"`{c}`" for c in selected])

    parts: List[pd.DataFrame] = []
    if symbols:
        uniq = sorted({str(s).zfill(6) for s in symbols})
        n_batches = (len(uniq) + chunk - 1) // chunk
        for bi in range(n_batches):
            batch = uniq[bi * chunk : (bi + 1) * chunk]
            placeholders = ", ".join(["%s"] * len(batch))
            sql = (
                f"SELECT {select_cols} FROM `{table}` "
                f"WHERE `code` IN ({placeholders}) AND `report_date` <= %s "
                "ORDER BY `code`, `report_date`"
            )
            params = tuple(batch) + (end_date,)
            dfb = provider._query_df(sql, params=params)
            if not dfb.empty:
                parts.append(dfb)
            if (bi + 1) % 5 == 0 or bi == n_batches - 1:
                LOG.info("财务因子分批拉取 %d/%d 批完成。", bi + 1, n_batches)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        sql = (
            f"SELECT {select_cols} FROM `{table}` "
            "WHERE `report_date` <= %s "
            "ORDER BY `code`, `report_date`"
        )
        df = provider._query_df(sql, params=(end_date,))

    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"report_date": "date", "code": "symbol"})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    rename_map = {c: f"fin_{c}" for c in selected}
    return df.rename(columns=rename_map)


def attach_financial_asof(price_df: pd.DataFrame, fin_sym: pd.DataFrame) -> pd.DataFrame:
    """把单只标的的财务因子按披露日 backward-asof 对齐到交易日。"""
    if fin_sym is None or fin_sym.empty:
        return price_df
    left = price_df.sort_values("date").copy()
    right = fin_sym.drop_duplicates("date", keep="last").sort_values("date").copy()
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    right["date"] = pd.to_datetime(right["date"]).astype("datetime64[ns]")
    fin_cols = [c for c in right.columns if c.startswith("fin_")]
    merged = pd.merge_asof(left, right[["date"] + fin_cols], on="date", direction="backward")
    return merged


def merge_factor_sources(
    symbol: str,
    kline_df: pd.DataFrame,
    provider: QuantiaDataProvider,
    start_date: str,
    end_date: str,
    disable_db_features: bool,
    include_local_recompute: bool,
    db_tech: bool = False,
    db_financial: bool = True,
    fin_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    base = pd.DataFrame({"date": kline_df["date"].copy()}).drop_duplicates().sort_values("date")
    base["symbol"] = symbol

    # 1) cache 原生可用因子。
    cache_cols = [c for c in CACHE_FACTOR_COLS if c in kline_df.columns]
    if cache_cols:
        cache_factor = kline_df[["date"] + cache_cols].drop_duplicates("date", keep="last")
        base = base.merge(cache_factor, on="date", how="left")

    # 2) 本地重算因子（来自价量）。
    if include_local_recompute:
        local_factor = build_local_factors_from_price(kline_df)
        base = base.merge(local_factor, on="date", how="left")

    # 3) DB 技术因子（仅在显式开启 db_tech 时；远程该表通常只覆盖近月，默认关闭）。
    if provider.available and not disable_db_features and db_tech:
        tech_df = provider.fetch_indicators(symbol, start_date, end_date)
        if not tech_df.empty:
            tech_cols = [c for c in tech_df.columns if c not in ("date", "code")]
            base = base.merge(tech_df[["date"] + tech_cols].drop_duplicates("date", keep="last"), on="date", how="left")

    # 4) DB 财务因子（全历史有效）：优先用批量预取的 fin_lookup，避免逐股串行查询。
    if provider.available and not disable_db_features and db_financial:
        fin_sym = None
        if fin_lookup is not None:
            fin_sym = fin_lookup.get(symbol)
        else:
            fin_sym = provider.fetch_financial(symbol, end_date)
        if fin_sym is not None and not fin_sym.empty:
            base = attach_financial_asof(base, fin_sym)

    # 5) 若消息面因子不存在，补默认列，保证后续方案C脚本接口一致。
    for z in DEFAULT_ZERO_FACTORS:
        if z not in base.columns:
            base[z] = 0.0

    return base.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_all(args: argparse.Namespace) -> Dict:
    quantia_root = Path(args.quantia_root).resolve()
    cache_root = Path(args.cache_hist_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not cache_root.exists():
        raise FileNotFoundError(f"cache 目录不存在: {cache_root}")

    split_dirs = ensure_layout(out_root)

    provider = QuantiaDataProvider(
        quantia_root=quantia_root,
        db_sleep=args.db_sleep,
        db_retries=args.db_retries,
    )

    cache_symbols = scan_cache_symbols(cache_root)
    db_symbols: List[str] = []
    if provider.available and args.scan_db_symbols:
        db_symbols = provider.fetch_symbols_from_spot()

    symbols = sorted(set(cache_symbols).union(db_symbols))
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    LOG.info("symbols total: %d", len(symbols))

    all_price_parts: List[pd.DataFrame] = []
    all_factor_parts: List[pd.DataFrame] = []

    stats = {
        "symbols_total": len(symbols),
        "symbols_processed": 0,
        "symbols_skipped": 0,
        "cache_price_ok": 0,
        "db_price_fallback": 0,
        "factor_from_db_symbols": 0,
        "factor_financial_symbols": 0,
        "factor_local_recompute_symbols": 0,
        "price_rows_dropped_invalid": 0,
        "symbols_with_invalid_price": 0,
    }

    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize() if args.end_date else pd.Timestamp(dt.date.today())

    # 批量预取全市场财务因子（全历史有效），构造 symbol->DataFrame 查找表，
    # 避免逐股 4900+ 次串行 DB 查询（这是远程库逐股拉取卡顿的根因）。
    fin_lookup: Optional[Dict[str, pd.DataFrame]] = None
    db_financial = provider.available and (not args.disable_db_features) and args.db_financial
    db_tech = provider.available and (not args.disable_db_features) and args.db_tech
    if db_financial:
        LOG.info("批量预取全市场财务因子 (report_date <= %s) ...", str(end_date.date()))
        fin_all = fetch_financial_bulk(provider, str(end_date.date()), symbols=symbols, chunk=args.fin_chunk)
        if fin_all.empty:
            LOG.warning("财务因子批量预取为空，将跳过 DB 财务因子。")
            db_financial = False
        else:
            fin_lookup = {
                sym: g.drop(columns=["symbol"]).reset_index(drop=True)
                for sym, g in fin_all.groupby("symbol", sort=False)
            }
            LOG.info("财务因子预取完成：%d 行，覆盖 %d 只标的。", len(fin_all), len(fin_lookup))

    for i, symbol in enumerate(symbols, start=1):
        px, cache_path = load_kline_from_cache(symbol, cache_root)
        used_db_price = False

        if not px.empty:
            stats["cache_price_ok"] += 1
        else:
            if provider.available and not args.disable_db_fallback_kline:
                px = provider.fetch_kline_from_spot(symbol, str(start_date.date()), str(end_date.date()))
                used_db_price = not px.empty
            if used_db_price:
                stats["db_price_fallback"] += 1

        if px.empty:
            stats["symbols_skipped"] += 1
            continue

        px = normalize_kline_df(px)
        px = px[(px["date"] >= start_date) & (px["date"] <= end_date)].copy()
        if px.empty:
            stats["symbols_skipped"] += 1
            continue

        # 价格净化（默认开启）：在计算本地因子之前剔除非法价量行，避免坏价污染相邻 ma/ret。
        # qfq 复权对部分高分红个股早期/中段回算会出现负价（amount 随之为负），无法用于训练；
        # 零成交量(停牌日 volume=0)合法，予以保留。
        if getattr(args, "sanitize_prices", True):
            ohlc = px[["open", "high", "low", "close"]]
            valid = (ohlc > 0).all(axis=1) & (px["volume"] >= 0) & (px["amount"] >= 0)
            n_bad = int((~valid).sum())
            if n_bad > 0:
                stats["price_rows_dropped_invalid"] += n_bad
                stats["symbols_with_invalid_price"] += 1
                px = px[valid].copy()
            if px.empty:
                stats["symbols_skipped"] += 1
                continue

        px["symbol"] = symbol
        price_df = px[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]].copy()

        factor_df = merge_factor_sources(
            symbol=symbol,
            kline_df=px,
            provider=provider,
            start_date=str(start_date.date()),
            end_date=str(end_date.date()),
            disable_db_features=args.disable_db_features,
            include_local_recompute=not args.disable_local_recompute,
            db_tech=db_tech,
            db_financial=db_financial,
            fin_lookup=fin_lookup,
        )

        if provider.available and not args.disable_db_features:
            tech_cols = [c for c in factor_df.columns if c.startswith("tech_")]
            fin_cols = [c for c in factor_df.columns if c.startswith("fin_")]
            if tech_cols or fin_cols:
                stats["factor_from_db_symbols"] += 1
            # 仅统计真正取到非空财务因子的标的。
            if fin_cols and factor_df[fin_cols].notna().any().any():
                stats["factor_financial_symbols"] += 1

        local_factor_cols = [c for c in factor_df.columns if c.startswith("local_")]
        if local_factor_cols:
            stats["factor_local_recompute_symbols"] += 1

        price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
        factor_df["date"] = pd.to_datetime(factor_df["date"]).dt.normalize()

        all_price_parts.append(price_df)
        all_factor_parts.append(factor_df)
        stats["symbols_processed"] += 1

        if i % args.progress_every == 0:
            LOG.info("progress %d/%d processed=%d skipped=%d", i, len(symbols), stats["symbols_processed"], stats["symbols_skipped"])

    if not all_price_parts:
        raise RuntimeError("未构建出任何价量数据，请检查 cache_root / DB 可用性")

    all_price = pd.concat(all_price_parts, ignore_index=True)
    all_price = all_price.sort_values(["date", "symbol"]).reset_index(drop=True)

    all_factors = pd.concat(all_factor_parts, ignore_index=True)
    all_factors = all_factors.sort_values(["date", "symbol"]).reset_index(drop=True)

    # 统一 symbol 为 6 位零填充字符串，避免写盘后被读回为整数丢失前导零（如 000001 -> 1）。
    all_price["symbol"] = all_price["symbol"].astype(str).str.zfill(6)
    all_factors["symbol"] = all_factors["symbol"].astype(str).str.zfill(6)

    # 价格净化兜底（默认开启）：逐股阶段已净化，这里再做一次全局兜底，确保无漏网非法价量行。
    if getattr(args, "sanitize_prices", True):
        ohlc = all_price[["open", "high", "low", "close"]]
        valid_mask = (
            (ohlc > 0).all(axis=1)
            & (all_price["volume"] >= 0)
            & (all_price["amount"] >= 0)
        )
        dropped = int((~valid_mask).sum())
        if dropped > 0:
            valid_keys = all_price.loc[valid_mask, ["date", "symbol"]]
            all_price = all_price.loc[valid_mask].reset_index(drop=True)
            # 因子表按有效 (date,symbol) 内连接，保持与价量严格对齐。
            all_factors = all_factors.merge(valid_keys, on=["date", "symbol"], how="inner").reset_index(drop=True)
            stats["price_rows_dropped_invalid"] += dropped
            LOG.info("价格净化兜底：额外剔除 %d 行非法价量。", dropped)
        LOG.info(
            "价格净化汇总：累计剔除 %d 行（涉及 %d 只标的）。",
            stats.get("price_rows_dropped_invalid", 0),
            stats.get("symbols_with_invalid_price", 0),
        )

    all_factors = _fill_factor_na(all_factors, zero_factor_cols=DEFAULT_ZERO_FACTORS)

    anchor = pick_anchor_date([all_price], args.anchor_date)
    ranges = make_ranges(anchor=anchor, val_days=args.val_days, test_days=args.test_days, start_date=args.start_date)

    split_price = split_by_ranges(all_price, ranges)
    split_factors = split_by_ranges(all_factors, ranges)

    # 写入分目录。
    split_summary = {}
    for r in ranges:
        split = r.name
        price_part = split_price[split]
        factor_part = split_factors[split]

        p_out = split_dirs[split] / "price.csv"
        f_out = split_dirs[split] / "factors.csv"
        price_part.to_csv(p_out, index=False)
        factor_part.to_csv(f_out, index=False)

        split_summary[split] = {
            "date_start": str(r.start.date()),
            "date_end": str(r.end.date()),
            "price_rows": int(len(price_part)),
            "factor_rows": int(len(factor_part)),
            "symbols": int(price_part["symbol"].nunique()) if not price_part.empty else 0,
            "price_file": str(p_out),
            "factor_file": str(f_out),
        }

    report = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quantia_root": str(quantia_root),
        "cache_hist_root": str(cache_root),
        "output_root": str(out_root),
        "range": {
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "anchor_date": str(anchor.date()),
            "val_days": args.val_days,
            "test_days": args.test_days,
        },
        "params": {
            "max_symbols": args.max_symbols,
            "scan_db_symbols": bool(args.scan_db_symbols),
            "disable_db_fallback_kline": bool(args.disable_db_fallback_kline),
            "disable_db_features": bool(args.disable_db_features),
            "disable_local_recompute": bool(args.disable_local_recompute),
            "db_sleep": args.db_sleep,
            "db_retries": args.db_retries,
        },
        "stats": stats,
        "splits": split_summary,
    }

    report_path = out_root / "split_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def run_smoke(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=180, freq="D")

    def mk(symbol: str, drift: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        close = 10 + np.cumsum(rng.normal(drift, 0.2, len(dates)))
        px = pd.DataFrame({
            "date": dates,
            "symbol": symbol,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e5, 1e6, len(dates)).astype(float),
            "amount": rng.integers(1e6, 1e7, len(dates)).astype(float),
        })
        fac = pd.DataFrame({
            "date": dates,
            "symbol": symbol,
            "turnover": rng.uniform(0, 8, len(dates)),
            "local_ret_1d": pd.Series(close).pct_change(1),
            "news_sent": np.nan,
            "event_flag": np.nan,
        })
        return px, fac

    p1, f1 = mk("000001", 0.02)
    p2, f2 = mk("000002", 0.01)
    all_price = pd.concat([p1, p2], ignore_index=True)
    all_fac = pd.concat([f1, f2], ignore_index=True)
    all_fac = _fill_factor_na(all_fac, DEFAULT_ZERO_FACTORS)

    anchor = pick_anchor_date([all_price], configured="2025-06-24")
    ranges = make_ranges(anchor=anchor, val_days=30, test_days=30, start_date="2025-01-01")
    sp = split_by_ranges(all_price, ranges)
    sf = split_by_ranges(all_fac, ranges)

    for k in ("train", "validation", "test"):
        assert len(sp[k]) > 0, f"smoke: {k} price empty"
        assert len(sf[k]) > 0, f"smoke: {k} factor empty"
        assert sp[k]["date"].min() <= sp[k]["date"].max(), f"smoke: {k} date invalid"

    assert sf["test"]["news_sent"].isna().sum() == 0, "smoke: zero fill failed"

    print(
        "[smoke] build_dataC_step1 passed: "
        f"train/val/test price rows={len(sp['train'])}/{len(sp['validation'])}/{len(sp['test'])}, "
        f"factor rows={len(sf['train'])}/{len(sf['validation'])}/{len(sf['test'])}"
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    quantia_root = Path("C:/xapproject/Quantia/Quantia")
    cache_root = quantia_root / "quantia" / "cache" / "hist"

    p = argparse.ArgumentParser(description="Build scheme-C step1 price/factor CSVs from Quantia cache + DB")

    p.add_argument("--quantia-root", default=str(quantia_root), help="Quantia 项目根目录")
    p.add_argument("--cache-hist-root", default=str(cache_root), help="Quantia cache/hist 根目录")
    p.add_argument("--out-root", default=str(repo_root / "DataSet" / "dataC"), help="输出根目录")

    p.add_argument("--start-date", default="2017-01-01", help="最早起始日期（YYYY-MM-DD）")
    p.add_argument("--end-date", default="", help="结束日期（YYYY-MM-DD），为空则今天")
    p.add_argument("--anchor-date", default="", help="切分锚点日期（YYYY-MM-DD），为空自动取数据最大日期")
    p.add_argument("--val-days", type=int, default=180, help="validation 回推天数")
    p.add_argument("--test-days", type=int, default=180, help="test 回推天数")

    p.add_argument("--max-symbols", type=int, default=0, help="最多处理多少标的，0=全部")
    p.add_argument("--scan-db-symbols", action="store_true", help="额外扫描 cn_stock_spot 的 symbol 并并集")

    p.add_argument("--disable-db-fallback-kline", action="store_true", help="禁用价量DB回退")
    p.add_argument("--disable-db-features", action="store_true", help="禁用因子DB拉取（仅 cache+本地重算）")
    p.add_argument("--disable-local-recompute", action="store_true", help="禁用本地技术因子重算")

    # 细粒度 DB 因子开关：
    # - 财务因子 cn_stock_financial 覆盖全历史（1988→2026），默认开启。
    # - 技术指标 cn_stock_indicators 远程仅覆盖近月（约 2026-02 起），无法覆盖历史训练区间，默认关闭，
    #   历史技术因子统一由本地重算（local_*）提供，避免逐股串行慢查询且无收益。
    p.add_argument("--db-financial", action=argparse.BooleanOptionalAction, default=True,
                   help="是否拉取 DB 财务因子 fin_*（全历史有效，默认开启）")
    p.add_argument("--db-tech", action=argparse.BooleanOptionalAction, default=False,
                   help="是否拉取 DB 技术指标 tech_*（远程仅近月，默认关闭）")

    p.add_argument("--sanitize-prices", action=argparse.BooleanOptionalAction, default=True,
                   help="是否剔除非法价量行（OHLC<=0 或 amount<0 或 volume<0，默认开启）")

    p.add_argument("--fin-chunk", type=int, default=400,
                   help="财务因子分批拉取每批 code 数量（避免远程大查询断连，默认400）")

    p.add_argument("--db-sleep", type=float, default=0.05, help="每次DB查询后的休眠秒数")
    p.add_argument("--db-retries", type=int, default=3, help="DB查询重试次数")
    p.add_argument("--progress-every", type=int, default=100, help="每处理 N 个symbol打印进度")

    p.add_argument("--prefer-remote-db", action=argparse.BooleanOptionalAction, default=True,
                   help="是否优先使用 Quantia 项目 .env 中的远程 DB 配置（默认开启）")
    p.add_argument("--db-host", default="", help="显式覆盖 DB host")
    p.add_argument("--db-user", default="", help="显式覆盖 DB user")
    p.add_argument("--db-password", default="", help="显式覆盖 DB password")
    p.add_argument("--db-database", default="", help="显式覆盖 DB database")
    p.add_argument("--db-port", type=int, default=0, help="显式覆盖 DB port")
    p.add_argument("--db-charset", default="", help="显式覆盖 DB charset")

    p.add_argument("--check-db", action="store_true", help="只检查 DB 连通性并退出")
    p.add_argument("--db-check-timeout", type=int, default=5, help="DB 连通性检测超时时间（秒）")

    p.add_argument("--smoke", action="store_true", help="运行冒烟测试")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.smoke:
        run_smoke(args)
        return

    # 在导入 Quantia DB 模块前，优先注入项目 .env 的远程 DB 配置。
    applied = _apply_db_env_from_quantia(args)

    if args.check_db:
        try:
            rep = _check_db_connectivity(timeout_sec=max(1, int(args.db_check_timeout)))
            print(json.dumps({"db_check": "ok", "effective": rep, "applied_env_keys": sorted(applied.keys())}, ensure_ascii=False, indent=2))
            return
        except Exception as e:
            print(json.dumps({"db_check": "fail", "error": str(e), "applied_env_keys": sorted(applied.keys())}, ensure_ascii=False, indent=2))
            raise

    if not args.end_date:
        args.end_date = str(dt.date.today())

    t0 = time.time()
    report = build_all(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    LOG.info("done in %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()
