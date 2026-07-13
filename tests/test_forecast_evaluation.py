#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune_csv"))

from pipeline.forecast_evaluation import (  # noqa: E402
    aggregate_forecast_evaluations,
    evaluate_forecast_path,
)


def _ohlc(date, close):
    return {
        "date": date,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
    }


class ForecastEvaluationTest(unittest.TestCase):
    def test_selected_horizons_use_steps_from_one_prediction_path(self):
        predictions = [_ohlc(f"2026-07-{day:02d}", 10 + index / 10)
                       for index, day in enumerate(range(14, 24), start=1)]
        actuals = [_ohlc("2026-07-14", 10.2), _ohlc("2026-07-16", 10.1)]

        records = evaluate_forecast_path(predictions, actuals, 10.0, (1, 3, 5, 10, 15))

        self.assertEqual([row["horizon"] for row in records], [1, 3, 5, 10, 15])
        self.assertEqual(records[0]["status"], "observed")
        self.assertEqual(records[1]["target_date"], "2026-07-16")
        self.assertEqual(records[2]["status"], "pending")
        self.assertEqual(records[-1]["status"], "not_predicted")

    def test_aggregate_reports_coverage_and_accuracy_per_horizon(self):
        records = [
            {"horizon": 1, "status": "observed", "close_abs_error": 0.1,
             "close_smape": 0.01, "return_abs_error": 0.02,
             "direction_correct": 1, "ohlc_valid": True},
            {"horizon": 1, "status": "pending"},
            {"horizon": 3, "status": "not_predicted"},
        ]

        summary = aggregate_forecast_evaluations(records, (1, 3))

        self.assertEqual(summary["by_horizon"]["1"]["coverage"], 0.5)
        self.assertEqual(summary["by_horizon"]["1"]["directional_accuracy"], 1.0)
        self.assertEqual(summary["by_horizon"]["3"]["n_expected"], 1)
        self.assertEqual(summary["by_horizon"]["3"]["coverage"], 0.0)

    def test_duplicate_actual_dates_are_rejected(self):
        predictions = [_ohlc("2026-07-14", 10.1)]
        actuals = [_ohlc("2026-07-14", 10.2), _ohlc("2026-07-14", 10.3)]

        with self.assertRaisesRegex(ValueError, "duplicate dates"):
            evaluate_forecast_path(predictions, actuals, 10.0, (1,))


if __name__ == "__main__":
    unittest.main()
