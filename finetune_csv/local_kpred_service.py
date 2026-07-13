#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local Kronos HTTP provider for Quantia K-line prediction."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kronos_loader import (
    DEFAULT_PREDICTOR_LOCAL,
    DEFAULT_TOKENIZER_LOCAL,
    load_kronos_predictor,
)

_DEFAULT_BUNDLE = _REPO_ROOT / "runs" / "dataC_c1"
_DEFAULT_FEATURES = _REPO_ROOT / "DataSet" / "dataC" / "fusion_all.csv"
_DEFAULT_CONFIG = _REPO_ROOT / "finetune_csv" / "configs" / "local_kpred.yaml"
_REQUIRED_PRICE_COLS = ["open", "high", "low", "close"]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    if not config_path.exists():
        raise ValueError(f"config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return config


def _setting(config: dict[str, Any], section: str, key: str, env_name: str,
             default: Any, cast):
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value.strip() != "":
        return cast(env_value)
    section_values = config.get(section, {})
    if isinstance(section_values, dict) and key in section_values:
        return cast(section_values[key])
    return default


def _path_setting(config: dict[str, Any], section: str, key: str,
                  env_name: str, default: str) -> str:
    value = Path(_setting(config, section, key, env_name, default, str)).expanduser()
    if not value.is_absolute():
        value = _REPO_ROOT / value
    return str(value.resolve())


def _bool_setting(config: dict[str, Any], section: str, key: str,
                  env_name: str, default: bool = False) -> bool:
    if env_name in os.environ:
        return _env_bool(env_name, default)
    value = config.get(section, {}).get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _horizon_setting(config: dict[str, Any], max_pred_days: int) -> tuple[int, ...]:
    value: Any = os.environ.get("KRONOS_EVALUATION_HORIZONS")
    if value is None or value.strip() == "":
        value = config.get("inference", {}).get(
            "evaluation_horizons", [1, 3, 5, 10, 15, 30]
        )
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError("evaluation_horizons must be a list or comma-separated string")
    try:
        horizons = tuple(sorted({int(item) for item in value}))
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation_horizons must contain integers") from exc
    if not horizons or horizons[0] < 1 or horizons[-1] > max_pred_days:
        raise ValueError(
            f"evaluation_horizons must be within 1..{max_pred_days} and cannot be empty"
        )
    return horizons


def _finite_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _history_frame(rows: Any, lookback: int) -> pd.DataFrame:
    if not isinstance(rows, list) or len(rows) < lookback:
        raise ValueError(f"history requires at least {lookback} rows")
    frame = pd.DataFrame(rows)
    required = ["date", *_REQUIRED_PRICE_COLS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"history missing columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    frame = frame.sort_values("date").tail(lookback).reset_index(drop=True)
    if len(frame) < lookback:
        raise ValueError(f"history requires {lookback} unique dated rows")
    for column in _REQUIRED_PRICE_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["volume"] = pd.to_numeric(frame.get("volume", 0), errors="coerce").fillna(0)
    if "amount" in frame.columns:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    else:
        frame["amount"] = frame["volume"] * frame[_REQUIRED_PRICE_COLS].mean(axis=1)
    columns = [*_REQUIRED_PRICE_COLS, "volume", "amount"]
    if frame[columns].isnull().any().any() or not np.isfinite(frame[columns]).all().all():
        raise ValueError("history contains invalid numeric values")
    if (frame[_REQUIRED_PRICE_COLS] <= 0).any().any():
        raise ValueError("history OHLC must be positive")
    return frame


def _future_index(values: Any, days: int, last_date: pd.Timestamp) -> pd.DatetimeIndex:
    if not isinstance(values, list) or len(values) != days:
        raise ValueError("future_timestamps length must equal days")
    future = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if future.hasnans or not future.is_monotonic_increasing or future.has_duplicates:
        raise ValueError("future_timestamps must be unique ascending dates")
    if any(value <= last_date for value in future):
        raise ValueError("future_timestamps must be after history")
    return future


def _sanitize_predictions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    predictions = []
    for timestamp, row in frame.iterrows():
        open_price = max(_finite_float(row["open"], "open"), 0.0001)
        close_price = max(_finite_float(row["close"], "close"), 0.0001)
        high = max(_finite_float(row["high"], "high"), open_price, close_price)
        low = max(min(_finite_float(row["low"], "low"), open_price, close_price), 0.0001)
        predictions.append({
            "date": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
            "open": round(open_price, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(close_price, 4),
            "volume": max(0, int(round(_finite_float(row.get("volume", 0), "volume")))),
            "amount": max(0.0, round(_finite_float(row.get("amount", 0), "amount"), 2)),
        })
    return predictions


class C1Enhancer:
    """Optional C1 scorer guarded by model quality, universe, and feature freshness."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.enabled = _bool_setting(config, "c1", "enabled", "KRONOS_C1_ENABLED")
        self.allow_unvalidated = _bool_setting(
            config, "c1", "allow_unvalidated", "KRONOS_C1_ALLOW_UNVALIDATED"
        )
        self.bundle_dir = Path(_path_setting(
            config, "c1", "bundle", "KRONOS_C1_BUNDLE", str(_DEFAULT_BUNDLE)
        ))
        self.features_csv = Path(_path_setting(
            config, "c1", "features_csv", "KRONOS_C1_FEATURES_CSV", str(_DEFAULT_FEATURES)
        ))
        self.min_test_ic = _setting(
            config, "c1", "min_test_ic", "KRONOS_C1_MIN_TEST_IC", 0.0, float
        )
        self.max_feature_age_days = _setting(
            config, "c1", "max_feature_age_days", "KRONOS_C1_MAX_FEATURE_AGE_DAYS", 7, int
        )
        self._model = None
        self._manifest: dict[str, Any] | None = None
        self._features: pd.DataFrame | None = None
        self.blocked_reason = "disabled"
        if self.enabled:
            self._load()

    def _load(self) -> None:
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.exists() or not self.features_csv.exists():
            self.blocked_reason = "bundle_or_features_missing"
            return
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        test_ic = float(manifest.get("metrics", {}).get("test", {}).get("IC_by_date", float("-inf")))
        if test_ic < self.min_test_ic and not self.allow_unvalidated:
            self._manifest = manifest
            self.blocked_reason = f"quality_gate_failed:test_ic_by_date={test_ic:.4f}"
            return
        from run_fusion import C1Model

        self._model = C1Model.load(
            str(self.bundle_dir), manifest["backend"], manifest["feat_cols"]
        )
        features = pd.read_csv(self.features_csv, dtype={"symbol": str}, parse_dates=["date"])
        missing = [column for column in manifest["feat_cols"] if column not in features.columns]
        if missing:
            self.blocked_reason = f"features_missing:{','.join(missing)}"
            return
        self._manifest = manifest
        self._features = features
        self.blocked_reason = ""

    def status(self) -> dict[str, Any]:
        manifest = self._manifest or {}
        return {
            "enabled": self.enabled,
            "ready": self._model is not None and self._features is not None,
            "reason": self.blocked_reason or None,
            "allow_unvalidated": self.allow_unvalidated,
            "test_metrics": manifest.get("metrics", {}).get("test"),
            "n_symbols": manifest.get("n_symbols"),
        }

    def score(self, symbol: str, as_of: pd.Timestamp, horizon_days: int, kronos_return: float,
              predictions: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        status = self.status()
        if not status["ready"]:
            return None, status
        assert self._manifest is not None and self._features is not None and self._model is not None
        label = str(self._manifest.get("label") or "")
        expected_horizon = None
        if label.startswith("label_fwd_ret_") and label.endswith("d"):
            try:
                expected_horizon = int(label[len("label_fwd_ret_"):-1])
            except ValueError:
                pass
        if expected_horizon is not None and horizon_days != expected_horizon:
            status.update(applied=False, reason=f"horizon_mismatch:expected_{expected_horizon}d")
            return None, status
        if symbol not in set(self._manifest.get("symbols", [])):
            status.update(applied=False, reason="symbol_not_in_bundle")
            return None, status
        candidates = self._features[
            (self._features["symbol"] == symbol) & (self._features["date"] <= as_of)
        ].sort_values("date")
        if candidates.empty:
            status.update(applied=False, reason="feature_row_not_found")
            return None, status
        row = candidates.iloc[-1]
        feature_date = pd.Timestamp(row["date"])
        age = (as_of.normalize() - feature_date.normalize()).days
        if age > self.max_feature_age_days and not self.allow_unvalidated:
            status.update(applied=False, reason=f"features_stale:{age}d",
                          feature_as_of=str(feature_date.date()))
            return None, status
        values = row[self._manifest["feat_cols"]].astype(float).to_numpy()[None, :]
        predicted_return = float(self._model.predict(values)[0])
        score = float(np.clip(predicted_return / 0.10, -1.0, 1.0))
        daily_returns = pd.Series([item["close"] for item in predictions]).pct_change().dropna()
        sigma = float(daily_returns.std(ddof=0) * 100) if len(daily_returns) else 0.0
        status.update(applied=True, reason=None, feature_as_of=str(feature_date.date()),
                      feature_age_days=age, predicted_return=predicted_return)
        pro = {
            "composite_score": score,
            "rating": "偏多" if predicted_return > 0.01 else "偏空" if predicted_return < -0.01 else "中性",
            "confidence": "实验" if self.allow_unvalidated else "中",
            "conflict_level": "高" if predicted_return * kronos_return < 0 else "低",
            "adj_return_pct": predicted_return * 100,
            "factor_return_pct": predicted_return * 100,
            "kronos_raw_return_pct": kronos_return * 100,
            "sigma_daily_pct": sigma,
            "factors": [{
                "key": "scheme_c1",
                "label": "方案C(C1)收益增强",
                "score": score,
                "weight": 1.0,
                "contribution": score,
            }],
        }
        return pro, status


class LocalKpredEngine:
    def __init__(self, predictor=None, model_meta: dict[str, Any] | None = None,
                 config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.max_context = max(32, min(512, _setting(
            self.config, "model", "max_context", "KRONOS_MAX_CONTEXT", 512, int
        )))
        self.lookback = max(32, min(self.max_context, _setting(
            self.config, "inference", "lookback", "KRONOS_LOOKBACK", 256, int
        )))
        self.max_pred_days = max(1, min(120, _setting(
            self.config, "inference", "max_pred_days", "KRONOS_MAX_PRED_DAYS", 30, int
        )))
        self.evaluation_horizons = _horizon_setting(self.config, self.max_pred_days)
        self.sample_count = max(1, min(64, _setting(
            self.config, "inference", "sample_count", "KRONOS_SAMPLE_COUNT", 1, int
        )))
        self.temperature = max(0.05, min(5.0, _setting(
            self.config, "inference", "temperature", "KRONOS_TEMPERATURE", 1.0, float
        )))
        self.top_k = max(0, min(1024, _setting(
            self.config, "inference", "top_k", "KRONOS_TOP_K", 1, int
        )))
        self.top_p = max(0.01, min(1.0, _setting(
            self.config, "inference", "top_p", "KRONOS_TOP_P", 1.0, float
        )))
        self.clip = max(1.0, min(20.0, _setting(
            self.config, "inference", "clip", "KRONOS_CLIP", 5.0, float
        )))
        self.device = _setting(
            self.config, "model", "device", "KRONOS_DEVICE", None, str
        ) or None
        self._lock = Lock()
        started = time.perf_counter()
        if predictor is None:
            predictor, model_meta = load_kronos_predictor(
                tokenizer_src=_path_setting(
                    self.config, "model", "tokenizer", "KRONOS_TOKENIZER",
                    DEFAULT_TOKENIZER_LOCAL
                ),
                predictor_src=_path_setting(
                    self.config, "model", "predictor", "KRONOS_PREDICTOR",
                    DEFAULT_PREDICTOR_LOCAL
                ),
                device=self.device,
                max_context=self.max_context,
                verbose=True,
            )
        predictor.clip = self.clip
        self.predictor = predictor
        self.model_meta = model_meta or {"predictor": {"provider": "injected", "source": "test"}}
        self.load_seconds = time.perf_counter() - started
        self.c1 = C1Enhancer(self.config)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_ready": self.predictor is not None,
            "model": self.model_meta,
            "load_seconds": round(self.load_seconds, 3),
            "max_context": self.max_context,
            "lookback": self.lookback,
            "max_pred_days": self.max_pred_days,
            "evaluation_horizons": list(self.evaluation_horizons),
            "sample_count": self.sample_count,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "clip": self.clip,
            "c1": self.c1.status(),
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        symbol = str(payload.get("code") or payload.get("symbol") or "").strip()
        days = int(payload.get("days", 5))
        if len(symbol) != 6 or not symbol.isdigit():
            raise ValueError("code must be six digits")
        if days not in self.evaluation_horizons:
            raise ValueError(
                f"days must be one of {list(self.evaluation_horizons)}"
            )
        history = _history_frame(payload.get("history"), self.lookback)
        last_date = pd.Timestamp(history["date"].iloc[-1])
        future = _future_index(payload.get("future_timestamps"), days, last_date)
        model_input = history[[*_REQUIRED_PRICE_COLS, "volume", "amount"]]
        with self._lock:
            predicted = self.predictor.predict(
                model_input,
                history["date"].reset_index(drop=True),
                pd.Series(future),
                days,
                T=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                sample_count=self.sample_count,
                verbose=False,
            )
        predictions = _sanitize_predictions(predicted)
        last_close = float(history["close"].iloc[-1])
        kronos_return = predictions[-1]["close"] / last_close - 1.0
        pro, c1_status = self.c1.score(symbol, last_date, days, kronos_return, predictions)
        source = self.model_meta.get("predictor", {}).get("source", "Kronos")
        return {
            "symbol": symbol,
            "name": payload.get("name") or symbol,
            "last_close": last_close,
            "last_date": str(last_date.date()),
            "predictions": predictions,
            "pro": pro,
            "provider": "local",
            "model_version": f"kronos:{Path(str(source)).name}",
            "history_last_date": str(last_date.date()),
            "prediction_start_date": predictions[0]["date"],
            "evaluation_horizons": [
                horizon for horizon in self.evaluation_horizons if horizon <= days
            ],
            "stale": bool(payload.get("history_stale", False)),
            "c1": c1_status,
            "latencyMs": round((time.perf_counter() - started) * 1000),
        }


class _Handler(BaseHTTPRequestHandler):
    engine: LocalKpredEngine

    def _write_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(404, {"code": -1, "msg": "not found"})
            return
        self._write_json(200, self.engine.health())

    def do_POST(self) -> None:
        if self.path not in {"/v1/open-api/kpred", "/v1/kline/predict"}:
            self._write_json(404, {"code": -1, "msg": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._write_json(200, self.engine.predict(payload))
        except ValueError as exc:
            self._write_json(400, {"code": -1, "msg": str(exc), "error_code": "INVALID_ARGUMENT"})
        except Exception as exc:  # noqa: BLE001
            self._write_json(500, {"code": -1, "msg": str(exc), "error_code": "INFERENCE_FAILED"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Kronos provider for Quantia")
    parser.add_argument("--config", default=os.environ.get("KRONOS_CONFIG", str(_DEFAULT_CONFIG)))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = _load_config(args.config)
    host = args.host or _setting(
        config, "service", "host", "KRONOS_SERVICE_HOST", "127.0.0.1", str
    )
    port = args.port or _setting(
        config, "service", "port", "KRONOS_SERVICE_PORT", 18081, int
    )
    _Handler.engine = LocalKpredEngine(config=config)
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"[kpred] config={args.config}")
    print(f"[kpred] listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
