#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune_csv"))
sys.path.insert(0, str(ROOT))

from local_kpred_service import C1Enhancer, LocalKpredEngine, _load_config  # noqa: E402


class _FakePredictor:
    def predict(self, frame, history_dates, future_dates, pred_len, **kwargs):
        return pd.DataFrame([
            {"open": 10.0, "high": 9.0, "low": 11.0, "close": 10.2,
             "volume": -1, "amount": 100.0}
            for _ in range(pred_len)
        ], index=future_dates)


def _payload(days=3):
    history_dates = pd.bdate_range("2026-01-01", periods=90)
    future_dates = pd.bdate_range(history_dates[-1] + pd.Timedelta(days=1), periods=days)
    return {
        "code": "300308",
        "days": days,
        "history": [
            {"date": str(date.date()), "open": 10.0, "high": 10.3,
             "low": 9.8, "close": 10.1, "volume": 1000, "amount": 10100}
            for date in history_dates
        ],
        "future_timestamps": [str(date.date()) for date in future_dates],
    }


class LocalKpredServiceTest(unittest.TestCase):
    def test_yaml_config_controls_inference_limits(self):
        config = _load_config(ROOT / "finetune_csv" / "configs" / "local_kpred.yaml")
        config["model"]["max_context"] = 128
        config["inference"].update({
            "lookback": 256,
            "max_pred_days": 999,
            "sample_count": 100,
        })
        with mock.patch.dict(os.environ, {
            "KRONOS_LOOKBACK": "",
            "KRONOS_MAX_PRED_DAYS": "",
            "KRONOS_SAMPLE_COUNT": "",
        }, clear=False):
            engine = LocalKpredEngine(_FakePredictor(), config=config)

        self.assertEqual(engine.max_context, 128)
        self.assertEqual(engine.lookback, 128)
        self.assertEqual(engine.max_pred_days, 120)
        self.assertEqual(engine.sample_count, 64)

    def test_relative_paths_resolve_from_repository_root(self):
        config = _load_config("finetune_csv/configs/local_kpred.yaml")
        with mock.patch.dict(os.environ, {"KRONOS_C1_ENABLED": "0"}, clear=False):
            engine = LocalKpredEngine(_FakePredictor(), config=config)

        self.assertEqual(engine.lookback, 256)
        self.assertEqual(engine.max_pred_days, 30)
        self.assertEqual(engine.evaluation_horizons, (1, 3, 5, 10, 15, 30))
        self.assertEqual(engine.top_k, 1)
        self.assertEqual(engine.top_p, 1.0)

    def test_engine_sanitizes_predictions_and_keeps_c1_optional(self):
        with mock.patch.dict(os.environ, {
            "KRONOS_C1_ENABLED": "0",
            "KRONOS_LOOKBACK": "",
        }, clear=False):
            result = LocalKpredEngine(
                _FakePredictor(), config={"inference": {"lookback": 90}}
            ).predict(_payload())

        self.assertEqual(len(result["predictions"]), 3)
        self.assertIsNone(result["pro"])
        self.assertEqual(result["c1"]["reason"], "disabled")
        for item in result["predictions"]:
            self.assertGreaterEqual(item["high"], max(item["open"], item["close"]))
            self.assertLessEqual(item["low"], min(item["open"], item["close"]))
            self.assertGreaterEqual(item["volume"], 0)

    def test_engine_rejects_horizon_outside_evaluation_options(self):
        with mock.patch.dict(os.environ, {
            "KRONOS_C1_ENABLED": "0",
            "KRONOS_LOOKBACK": "",
        }, clear=False):
            engine = LocalKpredEngine(
                _FakePredictor(), config={"inference": {"lookback": 90}}
            )

        with self.assertRaisesRegex(ValueError, "days must be one of"):
            engine.predict(_payload(days=2))

    def test_evaluation_horizons_support_environment_override(self):
        with mock.patch.dict(os.environ, {
            "KRONOS_C1_ENABLED": "0",
            "KRONOS_LOOKBACK": "",
            "KRONOS_EVALUATION_HORIZONS": "1,5,30",
        }, clear=False):
            engine = LocalKpredEngine(
                _FakePredictor(), config={"inference": {"lookback": 90}}
            )

        self.assertEqual(engine.evaluation_horizons, (1, 5, 30))

    def test_c1_negative_ic_bundle_is_blocked_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "manifest.json").write_text(json.dumps({
                "metrics": {"test": {"IC_by_date": -0.0961}},
                "n_symbols": 10,
            }), encoding="utf-8")
            features = root / "features.csv"
            features.write_text("date,symbol\n2026-01-01,300528\n", encoding="utf-8")
            env = {
                "KRONOS_C1_ENABLED": "1",
                "KRONOS_C1_ALLOW_UNVALIDATED": "0",
                "KRONOS_C1_BUNDLE": str(bundle),
                "KRONOS_C1_FEATURES_CSV": str(features),
                "KRONOS_C1_MIN_TEST_IC": "0",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                enhancer = C1Enhancer()

        self.assertFalse(enhancer.status()["ready"])
        self.assertEqual(
            enhancer.status()["reason"],
            "quality_gate_failed:test_ic_by_date=-0.0961",
        )


if __name__ == "__main__":
    unittest.main()
