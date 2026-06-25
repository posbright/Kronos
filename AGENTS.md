# Kronos — AI Agent Instructions

Kronos is a decoder-only foundation model for financial K-line (OHLCV) sequences. It uses a two-stage design: a **tokenizer** (BSQ) quantizes continuous K-lines into hierarchical discrete tokens (coarse `s1` + fine `s2`), and an **autoregressive Transformer** predicts future tokens. This fork adds Quantia-specific CSV finetuning pipelines under [finetune_csv/](finetune_csv/).

For deep dives, read — don't duplicate — the Chinese docs in [document/](document/README.md): [overview](document/01_项目概述.md), [quick start](document/02_快速开始.md), [architecture](document/03_模型架构.md), [API reference](document/04_核心API参考.md), [finetuning](document/05_微调指南.md), [examples & WebUI](document/06_示例与WebUI.md). LoRA/factor finetuning strategies are in [document/LoRa/](document/LoRa/).

## Environment

- Python 3.10+. A virtualenv exists at `.venv` (activate before running anything).
- Install deps: `pip install -r requirements.txt`. Core stack: PyTorch ≥2.0, `huggingface_hub`, `modelscope` (CN mirror), `safetensors`, `einops`, `akshare`/`lightgbm` (examples).
- Models load from the Hugging Face Hub under the `NeoQuasar/*` org (e.g. `NeoQuasar/Kronos-Tokenizer-base`, `NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-base`). Use `from_modelscope(...)` when HF is unreachable from China.

## Inference (the core API)

Always go through `KronosPredictor` ([model/kronos.py](model/kronos.py)); it handles validation, z-score normalization, autoregressive sampling, and denormalization.

```python
from model import Kronos, KronosTokenizer, KronosPredictor
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
pred_df = predictor.predict(df, x_timestamp, y_timestamp, pred_len, T=1.0, top_p=0.9, sample_count=1)
```

Hard constraints (do not violate):
- **`max_context=512`** is the maximum context window for `Kronos-small`/`Kronos-base` — exceeding it overflows the sliding buffer. Keep `lookback` ≤ 512.
- Input `df` must contain `['open', 'high', 'low', 'close']`; `volume`/`amount` are optional (auto-filled with zeros).
- `x_timestamp`/`y_timestamp` are mandatory — time features `[minute, hour, weekday, day, month]` are derived from them.
- `predict_batch(...)` requires **every** series to share identical `lookback` and `pred_len` (no ragged batches).
- `sample_count > 1` generates parallel paths that are averaged — slower but yields uncertainty bands.
- Device must be consistent (`cuda:0` → `mps` → `cpu`, auto-selected when `device=None`).

See runnable examples in [examples/](examples/) (e.g. [prediction_example.py](examples/prediction_example.py), [prediction_batch_example.py](examples/prediction_batch_example.py), akshare/CN-market scripts).

## Tests

```bash
pytest tests/test_kronos_regression.py -v
```

[tests/test_kronos_regression.py](tests/test_kronos_regression.py) runs on **CPU** and is deterministic: it pins exact model revisions (`TOKENIZER_REVISION`/`MODEL_REVISION` git hashes) and compares against golden files in [tests/data/](tests/data/) (`regression_output_512.csv`, `regression_output_256.csv`) plus an MSE check. If you change model/inference code, regenerate goldens with [tests/data/generate_regression_output.py](tests/data/generate_regression_output.py) — but only intentionally, since changing the pinned revisions breaks reproducibility.

## Finetuning — two separate routes

**1. Qlib route** ([finetune/](finetune/)) — original CSI300 pipeline configured via a Python class in [finetune/config.py](finetune/config.py) (not YAML). Order: `qlib_data_preprocess.py` (pickles train/val/test) → `train_tokenizer.py` → `train_predictor.py`. Launch with `torchrun --standalone --nproc_per_node=N finetune/train_tokenizer.py` (single-GPU: plain `python`).

**2. CSV route** ([finetune_csv/](finetune_csv/)) — Quantia-specific, YAML-configured via [config_loader.py](finetune_csv/config_loader.py) (dot-notation getters, `{exp_name}` path templating). Preferred for quick iteration:

```bash
python finetune_csv/train_sequential.py --config finetune_csv/configs/config_ali09988_candle-5min.yaml
```

`train_sequential.py` runs tokenizer → base model in one shot (`--skip-existing`, `--skip-tokenizer`, `--skip-basemodel` flags). Checkpoints land at `{base_path}/{exp_name}/{tokenizer|basemodel}/best_model/`. See [finetune_csv/README.md](finetune_csv/README.md) and the config example in [finetune_csv/configs/](finetune_csv/configs/).

CSV input columns: `timestamps, open, high, low, close, volume, amount`. Normalization is z-score over the lookback window only (prevents leakage), clipped to ±5.0.

## Quantia integration

The integration is **data-driven, not import-based**. [finetune_csv/build_full_market_dataset.py](finetune_csv/build_full_market_dataset.py) reads the sibling Quantia project (`C:\xapproject\Quantia\Quantia`): cache-first from `cache/hist` (pickled A-share klines), DB fallback to `cn_stock_spot`, enriched with `cn_stock_indicators` (technical) and `cn_stock_financial` (fundamentals), then writes leak-safe `DataSet/{train,validation,test}/dataset.csv`. Factor-fusion strategies live in [factor_model.py](finetune_csv/factor_model.py), [expand_tokenizer.py](finetune_csv/expand_tokenizer.py), and [build_fusion_dataset.py](finetune_csv/build_fusion_dataset.py) — see [document/LoRa/](document/LoRa/) for the design rationale (方案A/B/C).

## Conventions & pitfalls

- Hierarchical sampling is **sequential**: `s1` (coarse) is sampled first, then `s2` (fine) conditioned on `s1`. Never parallelize them.
- Distributed training uses NCCL (GPU) / gloo (CPU); rank/world_size come from `torchrun` env vars. DDP helpers are in [finetune/utils/training_utils.py](finetune/utils/training_utils.py).
- This is research/quant code: paths in scripts and docs are sometimes hardcoded to `C:/xapproject/Quantia/...`. Prefer config/CLI args over editing hardcoded paths.
- The WebUI Flask app is under [webui/](webui/) with its own [requirements.txt](webui/requirements.txt).
