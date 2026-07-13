"""Kronos 预测路径的多 horizon 准确率评测。"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

DEFAULT_EVALUATION_HORIZONS = (1, 3, 5, 10, 15, 30)
_REQUIRED_OHLC = ("open", "high", "low", "close")


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _normalize_rows(rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
                    label: str) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    required = {"date", *_REQUIRED_OHLC}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError(f"{label} contains invalid dates")
    if frame["date"].duplicated().any():
        raise ValueError(f"{label} contains duplicate dates")
    for column in _REQUIRED_OHLC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not frame[column].map(math.isfinite).all():
            raise ValueError(f"{label}.{column} contains non-finite values")
    return frame.sort_values("date").reset_index(drop=True)


def _smape(predicted: float, actual: float) -> float:
    denominator = abs(predicted) + abs(actual)
    return 0.0 if denominator == 0 else 2.0 * abs(predicted - actual) / denominator


def evaluate_forecast_path(
    predictions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    actuals: Sequence[Mapping[str, Any]] | pd.DataFrame,
    last_actual_close: float,
    horizons: Iterable[int] = DEFAULT_EVALUATION_HORIZONS,
) -> list[dict[str, Any]]:
    """按预测路径第 h 步对齐真实交易日，返回逐 horizon 评测记录。"""
    predicted_frame = _normalize_rows(predictions, "predictions")
    actual_frame = _normalize_rows(actuals, "actuals").set_index("date")
    last_close = _finite_float(last_actual_close, "last_actual_close")
    if last_close <= 0:
        raise ValueError("last_actual_close must be positive")

    normalized_horizons = tuple(sorted(set(int(value) for value in horizons)))
    if not normalized_horizons or normalized_horizons[0] < 1:
        raise ValueError("horizons must contain positive integers")

    records: list[dict[str, Any]] = []
    for horizon in normalized_horizons:
        if horizon > len(predicted_frame):
            records.append({
                "horizon": horizon,
                "status": "not_predicted",
                "target_date": None,
            })
            continue

        predicted = predicted_frame.iloc[horizon - 1]
        target_date = pd.Timestamp(predicted["date"])
        base = {
            "horizon": horizon,
            "target_date": target_date.date().isoformat(),
        }
        if target_date not in actual_frame.index:
            records.append({**base, "status": "pending"})
            continue

        actual = actual_frame.loc[target_date]
        predicted_close = float(predicted["close"])
        actual_close = float(actual["close"])
        predicted_return = predicted_close / last_close - 1.0
        actual_return = actual_close / last_close - 1.0
        direction_correct = int(
            (predicted_return > 0) == (actual_return > 0)
            if predicted_return != 0 and actual_return != 0
            else predicted_return == actual_return
        )
        ohlc_valid = bool(
            predicted["low"] <= min(predicted["open"], predicted["close"])
            and predicted["high"] >= max(predicted["open"], predicted["close"])
            and predicted["low"] <= predicted["high"]
        )
        records.append({
            **base,
            "status": "observed",
            "predicted_close": predicted_close,
            "actual_close": actual_close,
            "predicted_return": predicted_return,
            "actual_return": actual_return,
            "close_abs_error": abs(predicted_close - actual_close),
            "close_smape": _smape(predicted_close, actual_close),
            "return_abs_error": abs(predicted_return - actual_return),
            "direction_correct": direction_correct,
            "ohlc_valid": ohlc_valid,
        })
    return records


def aggregate_forecast_evaluations(
    records: Sequence[Mapping[str, Any]],
    horizons: Iterable[int] = DEFAULT_EVALUATION_HORIZONS,
) -> dict[str, Any]:
    """聚合多股票/多批次记录，输出可直接持久化和可视化的 horizon 指标。"""
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["horizon"])].append(record)

    result: dict[str, Any] = {"by_horizon": {}}
    for horizon in sorted(set(int(value) for value in horizons)):
        horizon_records = grouped.get(horizon, [])
        observed = [row for row in horizon_records if row.get("status") == "observed"]
        metrics: dict[str, Any] = {
            "horizon": horizon,
            "n_expected": len(horizon_records),
            "n_observed": len(observed),
            "coverage": len(observed) / len(horizon_records) if horizon_records else 0.0,
        }
        if observed:
            metrics.update({
                "close_mae": sum(float(row["close_abs_error"]) for row in observed) / len(observed),
                "close_smape": sum(float(row["close_smape"]) for row in observed) / len(observed),
                "return_mae": sum(float(row["return_abs_error"]) for row in observed) / len(observed),
                "directional_accuracy": sum(int(row["direction_correct"]) for row in observed) / len(observed),
                "ohlc_valid_rate": sum(bool(row["ohlc_valid"]) for row in observed) / len(observed),
            })
        result["by_horizon"][str(horizon)] = metrics
    return result
