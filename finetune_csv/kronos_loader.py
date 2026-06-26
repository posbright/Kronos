"""Shared Kronos model loading helpers.

This module centralizes model source resolution so scripts can use:
- local path first
- ModelScope first (default)
- Hugging Face as fallback
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_TOKENIZER_MS = "AI-ModelScope/Kronos-Tokenizer-base"
DEFAULT_PREDICTOR_MS = "AI-ModelScope/Kronos-base"

DEFAULT_TOKENIZER_HF = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_PREDICTOR_HF = "NeoQuasar/Kronos-base"


def _provider_order(prefer_source: str) -> list[str]:
    prefer = (prefer_source or "modelscope").strip().lower()
    if prefer not in ("modelscope", "hf"):
        raise ValueError("prefer_source must be 'modelscope' or 'hf'")
    return ["modelscope", "hf"] if prefer == "modelscope" else ["hf", "modelscope"]


def _map_default_source(src: str, asset: str, provider: str) -> str:
    if asset == "tokenizer":
        ms, hf = DEFAULT_TOKENIZER_MS, DEFAULT_TOKENIZER_HF
    else:
        ms, hf = DEFAULT_PREDICTOR_MS, DEFAULT_PREDICTOR_HF

    if not src or src in (ms, hf):
        return ms if provider == "modelscope" else hf
    return src


def _load_one(loader_cls, src: str, asset: str, prefer_source: str, verbose: bool):
    src = (src or "").strip()
    local = Path(src)
    if src and local.exists():
        if verbose:
            print(f"[model] {asset}: local -> {local}")
        return loader_cls.from_pretrained(str(local)), {"provider": "local", "source": str(local)}

    errors = []
    seen = set()
    for provider in _provider_order(prefer_source):
        resolved = _map_default_source(src, asset, provider)
        key = (provider, resolved)
        if key in seen:
            continue
        seen.add(key)
        try:
            if verbose:
                print(f"[model] {asset}: trying {provider} -> {resolved}")
            if provider == "modelscope":
                obj = loader_cls.from_modelscope(resolved)
            else:
                obj = loader_cls.from_pretrained(resolved)
            if verbose:
                print(f"[model] {asset}: loaded from {provider} -> {resolved}")
            return obj, {"provider": provider, "source": resolved}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider}:{resolved} => {type(exc).__name__}: {exc}")

    msg = "\n  - ".join(errors) if errors else "<no attempts>"
    raise RuntimeError(f"Failed to load {asset} from both ModelScope and HF.\n  - {msg}")


def load_kronos_predictor(*, tokenizer_src: str, predictor_src: str,
                          device=None, max_context: int = 512,
                          prefer_source: str = "modelscope", verbose: bool = True):
    """Load tokenizer/model with source fallback and build KronosPredictor.

    Returns:
        predictor, meta
    """
    from model import Kronos, KronosTokenizer, KronosPredictor

    tok, tok_meta = _load_one(
        KronosTokenizer, tokenizer_src, asset="tokenizer",
        prefer_source=prefer_source, verbose=verbose,
    )
    mdl, mdl_meta = _load_one(
        Kronos, predictor_src, asset="predictor",
        prefer_source=prefer_source, verbose=verbose,
    )
    predictor = KronosPredictor(mdl, tok, device=device, max_context=max_context)
    meta = {
        "prefer_source": (prefer_source or "modelscope").strip().lower(),
        "tokenizer": tok_meta,
        "predictor": mdl_meta,
    }
    return predictor, meta
