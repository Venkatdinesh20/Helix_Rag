"""
Step 6: Retrieval Service
--------------------------
Given a user's question, find the most relevant chunks from the vector store.

How it works (hybrid retrieval — Phase 2.1):
  1. Embed the query and search FAISS for the top-N dense (semantic) hits.
  2. Tokenize the query and search a BM25 index for the top-N lexical hits.
  3. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF).
  4. Return the top-K fused results, each annotated with both individual
     scores and the fused RRF score.

Why hybrid?
  - Dense embeddings excel at semantic similarity ("how do I cancel?" ≈
    "refund policy") but miss rare terms, acronyms, names, and IDs.
  - BM25 is great at exact / rare-token matches but blind to synonyms.
  - Combined via RRF, the two signals consistently outperform either alone
    on heterogeneous corpora — and RRF needs no score-scale tuning.

Reciprocal Rank Fusion (Cormack et al., 2009):
  For each document d, RRF score = Σ over rankers r of 1 / (k + rank_r(d))
  where rank is 1-based. We use k = 60 (the canonical default).
"""

import re
import threading
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from pipeline.build_index import load_index
from pipeline.generate_embeddings import embed_query


# ---------------------------------------------------------------------------
# Module-level cache — index and metadata are loaded once at startup.
# Loading from disk on every query would add ~100ms latency per request.
# ---------------------------------------------------------------------------
_index = None
_metadata_records = None
_bm25: BM25Okapi | None = None
_bm25_corpus_size: int = 0
_lock = threading.Lock()

# Tokenizer: lowercased alphanumeric tokens of length ≥ 2. Deliberately
# language-agnostic and stopword-free — BM25's IDF already discounts
# common terms, and dropping stopwords here would hurt phrase recall.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase + alphanumeric token split used for the BM25 index."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


def _build_bm25(metadata_records: List[Dict[str, Any]]) -> BM25Okapi | None:
    """Build a BM25 index from chunk texts. Returns None for empty corpora."""
    if not metadata_records:
        return None
    tokenized_corpus = [_tokenize(rec["text"]) for rec in metadata_records]
    # rank_bm25 raises on an all-empty corpus; guard against that explicitly.
    if not any(tokenized_corpus):
        return None
    return BM25Okapi(tokenized_corpus)


def _get_index_and_metadata():
    """Lazy-load the FAISS index, metadata, and BM25 index. Thread-safe."""
    global _index, _metadata_records, _bm25, _bm25_corpus_size
    if _index is None or _metadata_records is None:
        with _lock:
            if _index is None or _metadata_records is None:
                _index, _metadata_records = load_index()
                _bm25 = _build_bm25(_metadata_records)
                _bm25_corpus_size = len(_metadata_records) if _metadata_records else 0
    return _index, _metadata_records


def _get_bm25() -> BM25Okapi | None:
    """Return the cached BM25 index, building it if needed."""
    global _bm25, _bm25_corpus_size
    _get_index_and_metadata()  # ensure metadata is loaded
    if _bm25 is None or _bm25_corpus_size != len(_metadata_records or []):
        with _lock:
            _bm25 = _build_bm25(_metadata_records or [])
            _bm25_corpus_size = len(_metadata_records or [])
    return _bm25


def reload_index() -> None:
    """
    Force the in-memory FAISS index, metadata, and BM25 index to reload.

    Called after a successful document upload so newly ingested chunks
    are visible to subsequent /ask calls without restarting the process.
    """
    global _index, _metadata_records, _bm25, _bm25_corpus_size
    with _lock:
        _index, _metadata_records = load_index()
        _bm25 = _build_bm25(_metadata_records)
        _bm25_corpus_size = len(_metadata_records) if _metadata_records else 0


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Fuse multiple ranked lists of document indices into a single ordering.

    Args:
        ranked_lists : list of ranked lists; each inner list is the doc
                       indices in best-to-worst order from one ranker.
        k            : RRF damping constant (default 60, per the paper).

    Returns:
        list of (doc_idx, rrf_score) sorted by rrf_score descending.
    """
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _dense_search(
    query_vector: np.ndarray,
    top_k: int,
) -> List[Tuple[int, float]]:
    """Run a FAISS dense search; return (idx, cosine_score) pairs."""
    index, _ = _get_index_and_metadata()
    if index is None or index.ntotal == 0 or top_k <= 0:
        return []
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, k)
    out: List[Tuple[int, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        out.append((int(idx), float(score)))
    return out


def _bm25_search(query: str, top_k: int) -> List[Tuple[int, float]]:
    """Run BM25 lexical search; return (idx, bm25_score) pairs."""
    bm25 = _get_bm25()
    if bm25 is None or top_k <= 0:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    if scores.size == 0:
        return []
    # argsort descending, keep top_k with strictly positive score (zero
    # score means no shared terms — no signal worth fusing).
    order = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant chunks for a given query.

    By default this runs hybrid retrieval: BM25 + FAISS fused with RRF.
    Toggle via the `hybrid_search_enabled` setting (or pass-through env
    var HYBRID_SEARCH_ENABLED=false) to fall back to dense-only.

    Returned dicts include:
      - chunk_id, text, metadata
      - score      : cosine similarity (dense). Used by confidence buckets.
      - dense_score: same as score (explicit, for transparency)
      - bm25_score : raw BM25 score (0.0 if not matched lexically)
      - rrf_score  : fused Reciprocal Rank Fusion score
      - retrieval_mode : "hybrid" or "dense"
    """
    index, metadata_records = _get_index_and_metadata()

    if top_k <= 0 or index is None or index.ntotal == 0:
        return []

    # Embed the query — must use the same model as chunk embedding.
    query_vector = embed_query(query)
    query_vector = query_vector.reshape(1, -1).astype("float32")

    hybrid_enabled = getattr(settings, "hybrid_search_enabled", True)
    rrf_k = getattr(settings, "rrf_k", 60)
    rerank_enabled = getattr(settings, "rerank_enabled", True)
    rerank_pool = max(getattr(settings, "rerank_candidate_pool", 20), top_k)

    # Cast a wider net on each ranker so the fusion has room to reorder.
    # A 4× multiplier is a common rule of thumb that keeps latency low
    # while giving RRF enough candidates to surface true positives that
    # one ranker alone would have missed. When the reranker is enabled
    # we widen further so it has a meaningful pool to choose from.
    base_k = rerank_pool if rerank_enabled else top_k
    candidate_k = max(base_k * 4, base_k)

    dense_hits = _dense_search(query_vector, candidate_k)
    dense_score_by_idx: Dict[int, float] = {idx: s for idx, s in dense_hits}

    if hybrid_enabled:
        bm25_hits = _bm25_search(query, candidate_k)
    else:
        bm25_hits = []
    bm25_score_by_idx: Dict[int, float] = {idx: s for idx, s in bm25_hits}

    if bm25_hits:
        fused = reciprocal_rank_fusion(
            [[i for i, _ in dense_hits], [i for i, _ in bm25_hits]],
            k=rrf_k,
        )
        mode = "hybrid"
    else:
        # Dense-only fallback (hybrid disabled, BM25 unbuilt, or no
        # lexical overlap). Preserve dense order and synthesise a
        # single-list RRF score so downstream callers see a consistent shape.
        fused = [
            (idx, 1.0 / (rrf_k + rank))
            for rank, (idx, _) in enumerate(dense_hits, start=1)
        ]
        mode = "dense"

    # If we'll rerank, keep a larger pool through the build step and then
    # truncate after the cross-encoder rescores. Otherwise truncate now.
    pool_size = rerank_pool if rerank_enabled else top_k
    fused = fused[:pool_size]

    results: List[Dict[str, Any]] = []
    for idx, rrf_score in fused:
        # Items found only by BM25 won't have a dense score yet — recover
        # it cheaply via FAISS's reconstruct() rather than re-running
        # the full search. This keeps the `score` field meaningful for
        # the confidence badge even on lexical-only matches.
        if idx in dense_score_by_idx:
            cosine = dense_score_by_idx[idx]
        else:
            try:
                vec = index.reconstruct(int(idx))
                cosine = float(np.dot(query_vector[0], vec))
            except Exception:
                cosine = 0.0

        chunk = metadata_records[idx]
        results.append({
            "chunk_id":       chunk["chunk_id"],
            "text":           chunk["text"],
            "metadata":       chunk["metadata"],
            "score":          float(cosine),
            "dense_score":    float(cosine),
            "bm25_score":     float(bm25_score_by_idx.get(idx, 0.0)),
            "rrf_score":      float(rrf_score),
            "retrieval_mode": mode,
            "rerank_score":   None,
        })

    # ---- Cross-encoder rerank (Phase 2.2) ----------------------------
    # Joint (query, passage) scoring trades a small latency hit for a
    # large precision win. The reranker module degrades gracefully if
    # the model can't be loaded — it just returns the input unchanged
    # with rerank_score=None, so the pipeline never breaks.
    if rerank_enabled and len(results) > 1:
        # Imported lazily to keep the retrieval module importable in
        # environments without sentence-transformers (e.g. lightweight
        # CI workers running pure-function tests only).
        from app.reranker import rerank as _rerank  # noqa: WPS433
        results = _rerank(query, results, top_n=top_k)
    else:
        results = results[:top_k]

    return results


if __name__ == "__main__":
    """
    Quick smoke test — run with:
      python -m app.retrieval

    Try different questions to see which chunks are retrieved.
    """
    test_queries = [
        "What is the refund policy?",
        "How can I contact support?",
        "What documents are needed for onboarding?",
        "How many days of annual leave do I get?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query!r}")
        results = retrieve(query, top_k=2)
        for r in results:
            print(f"  [{r['chunk_id']}]  score={r['score']:.4f}")
            print(f"    {r['text'][:120]!r}")
