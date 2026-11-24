"""
Phase 2.2 — Cross-encoder reranker
-----------------------------------
Why a reranker?
  Hybrid retrieval (BM25 + dense + RRF) is fast and recall-friendly, but
  both signals score a query and a passage in isolation. A cross-encoder
  jointly attends over the (query, passage) pair, giving a far more
  accurate relevance score at the cost of a few hundred milliseconds.

  The standard production pattern is two-stage retrieval:
    1) Cheap recall: get ~20 candidates with hybrid search
    2) Costly precision: rerank those 20 with a cross-encoder, keep top 5

We use `cross-encoder/ms-marco-MiniLM-L-6-v2`:
  - ~80 MB, runs comfortably on CPU
  - Trained on the MS MARCO passage-ranking task → strong out-of-domain
    generalisation for short QA-style queries against text chunks

Failure modes are handled defensively:
  - Model download fails (no internet) → `rerank()` returns the input
    chunks unchanged and logs a single warning. The pipeline still works.
  - Empty input → empty output, no crash.
  - Thread-safe singleton: model loaded once across requests.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy singleton — model is heavy (~80 MB) and slow to load (~1-2 s on CPU).
# We don't want to pay that cost at import time or per request.
# ---------------------------------------------------------------------------
_model = None
_model_load_failed = False
_lock = threading.Lock()


def _get_model():
    """
    Load the cross-encoder model lazily. Returns None if loading failed
    (so callers can short-circuit to a no-op rerank instead of crashing).
    """
    global _model, _model_load_failed
    if _model is not None:
        return _model
    if _model_load_failed:
        return None

    with _lock:
        if _model is not None:
            return _model
        if _model_load_failed:
            return None
        try:
            # Imported lazily so that environments without
            # sentence-transformers can still import this module (e.g.
            # for unit-testing the no-op fallback path).
            from sentence_transformers import CrossEncoder
            model_name = getattr(
                settings, "reranker_model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
            _model = CrossEncoder(model_name)
            logger.info("Cross-encoder reranker loaded: %s", model_name)
        except Exception as exc:
            # Network failure, missing dependency, corrupted cache — any of
            # these should not break retrieval. Mark as failed so we don't
            # retry on every request.
            logger.warning(
                "Cross-encoder reranker unavailable (%s); falling back to "
                "hybrid-only ranking.", exc,
            )
            _model_load_failed = True
            _model = None
    return _model


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Re-order `chunks` by cross-encoder relevance to `query`.

    Args:
        query : the user's question
        chunks: list of retrieval result dicts (must each have a "text" key)
        top_n : truncate to this many after reranking (None = keep all)

    Returns:
        A new list, sorted by `rerank_score` desc. Each dict gains a
        `rerank_score` (float) field. If the model is unavailable the
        input order is preserved and `rerank_score` is set to None.
    """
    if not chunks or not query or not query.strip():
        return list(chunks)[: top_n] if top_n is not None else list(chunks)

    model = _get_model()

    if model is None:
        # Graceful fallback: tag the items so callers know reranking
        # was attempted but skipped, and preserve hybrid order.
        out = [dict(c, rerank_score=None) for c in chunks]
        return out[: top_n] if top_n is not None else out

    try:
        pairs = [(query, c.get("text", "")) for c in chunks]
        # `predict` returns one logit per pair; higher = more relevant.
        # We do not apply sigmoid — relative ordering is what matters.
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:
        logger.warning(
            "Cross-encoder predict failed (%s); preserving hybrid order.", exc,
        )
        out = [dict(c, rerank_score=None) for c in chunks]
        return out[: top_n] if top_n is not None else out

    scored = [
        dict(chunk, rerank_score=float(s))
        for chunk, s in zip(chunks, scores)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    if top_n is not None:
        scored = scored[: top_n]
    return scored


def is_available() -> bool:
    """True iff the cross-encoder model is loaded and usable."""
    return _get_model() is not None
