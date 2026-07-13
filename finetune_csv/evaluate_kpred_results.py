#!/usr/bin/env python3
"""用真实 K 线 CSV 评测本地服务保存的预测 JSON。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.forecast_evaluation import (
    DEFAULT_EVALUATION_HORIZONS,
    aggregate_forecast_evaluations,
    evaluate_forecast_path,
)


def _load_prediction(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("data"), dict):
        payload = payload["data"]
    if not isinstance(payload.get("predictions"), list):
        raise ValueError("prediction JSON must contain predictions[]")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Kronos prediction horizons")
    parser.add_argument("--prediction-json", required=True, type=Path)
    parser.add_argument("--actual-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_EVALUATION_HORIZONS),
        help="comma-separated path steps, default: 1,3,5,10,15,30",
    )
    args = parser.parse_args()

    prediction = _load_prediction(args.prediction_json)
    actuals = pd.read_csv(args.actual_csv)
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    records = evaluate_forecast_path(
        prediction["predictions"],
        actuals,
        prediction["last_close"],
        horizons,
    )
    output = {
        "symbol": prediction.get("symbol"),
        "model_version": prediction.get("model_version"),
        "last_date": prediction.get("last_date"),
        "records": records,
        "summary": aggregate_forecast_evaluations(records, horizons),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
