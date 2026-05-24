"""
Tests for hybrid retrieval (Phase 2.1) — BM25 + FAISS via Reciprocal Rank Fusion.

We verify:
  - RRF math: the fusion function returns the expected ordering and scores.
  - Tokenizer: lowercases, drops short/punctuation tokens, handles empty input.
  - Hybrid path on the real index returns the expected enriched dict shape
    (score, dense_score, bm25_score, rrf_score, retrieval_mode).
  - A query with a distinctive exact-match token surfaces that token in BM25
    hits, demonstrating lexical signal is active.
  - Reload rebuilds the BM25 index in lock-step with metadata.
  - retrieve(top_k=0) and empty-query edge cases.
  - When hybrid is disabled, results carry retrieval_mode == "dense" and
    bm25_score == 0 for every hit.
"""

import pytest

from app import retrieval as retrieval_module
from app.retrieval import (
    _bm25_search,
    _get_index_and_metadata,
    _tokenize,
    reciprocal_rank_fusion,
    reload_index,
    retrieve,
)


# ---------------------------------------------------------------------------
# Pure-function tests (no index required)
# ---------------------------------------------------------------------------

class TestTokenize:

    def test_lowercases(self):
        assert _tokenize("Hello WORLD") == ["hello", "world"]

    def test_drops_short_tokens(self):
        # "a" and "I" are dropped (len < 2), "ai" and "is" kept.
        assert _tokenize("a I ai is") == ["ai", "is"]

    def test_alphanumeric_only(self):
        assert _tokenize("foo-bar_baz!! 42") == ["foo", "bar", "baz", "42"]

    def test_empty(self):
        assert _tokenize("") == []
        assert _tokenize(None) == []  # type: ignore[arg-type]


class TestRRF:

    def test_single_list_passthrough_order(self):
        fused = reciprocal_rank_fusion([[10, 20, 30]], k=60)
        # Order preserved, scores strictly decreasing.
        ids = [i for i, _ in fused]
        assert ids == [10, 20, 30]
        scores = [s for _, s in fused]
        assert scores[0] > scores[1] > scores[2]

    def test_two_lists_boosts_common_doc(self):
        # Doc 7 appears at rank 1 in both lists → highest fused score.
        fused = reciprocal_rank_fusion([[7, 8, 9], [7, 1, 2]], k=60)
        assert fused[0][0] == 7
        # Verify the exact RRF math: 1/(60+1) + 1/(60+1) = 2/61
        assert fused[0][1] == pytest.approx(2 / 61)

    def test_disjoint_lists_top_ranks_tie(self):
        # Two completely disjoint ranked lists — the two rank-1 docs tie
        # at 1/(60+1), and rank-2 docs tie at 1/(60+2). RRF math.
        fused = reciprocal_rank_fusion([[1, 2], [3, 4]], k=60)
        scores = dict(fused)
        assert scores[1] == pytest.approx(1 / 61)
        assert scores[3] == pytest.approx(1 / 61)
        assert scores[2] == pytest.approx(1 / 62)
        assert scores[4] == pytest.approx(1 / 62)

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []


# ---------------------------------------------------------------------------
# Tests that hit the real index. Skip cleanly if the workspace has no index
# built — keeps the suite green on a fresh checkout.
# ---------------------------------------------------------------------------

def _has_index() -> bool:
    try:
        idx, meta = _get_index_and_metadata()
        return idx is not None and idx.ntotal > 0 and bool(meta)
    except Exception:
        return False


def _distinctive_token() -> str | None:
    """Pull a long, distinctive token from the first record for BM25 probing."""
    _, meta = _get_index_and_metadata()
    if not meta:
        return None
    stopwords = {"this", "that", "with", "from", "have", "been", "they",
                 "which", "their", "would", "could", "about", "there",
                 "where", "while", "should"}
    for rec in meta[:5]:
        for tok in _tokenize(rec["text"]):
            if len(tok) >= 7 and tok.isalpha() and tok not in stopwords:
                return tok
    return None


@pytest.mark.skipif(not _has_index(), reason="no FAISS index built")
class TestHybridIntegration:

    def test_result_shape_includes_hybrid_fields(self):
        results = retrieve("summary of the document", top_k=3)
        assert results, "expected non-empty results from the real index"
        for r in results:
            assert {"chunk_id", "text", "metadata", "score",
                    "dense_score", "bm25_score", "rrf_score",
                    "retrieval_mode"} <= set(r.keys())
            assert isinstance(r["score"], float)
            assert isinstance(r["bm25_score"], float)
            assert isinstance(r["rrf_score"], float)
            assert r["retrieval_mode"] in ("hybrid", "dense")

    def test_top_k_respected(self):
        results = retrieve("anything", top_k=2)
        assert len(results) <= 2

    def test_zero_top_k_returns_empty(self):
        assert retrieve("anything", top_k=0) == []

    def test_bm25_search_finds_distinctive_token(self):
        tok = _distinctive_token()
        if tok is None:
            pytest.skip("no distinctive token in corpus")
        hits = _bm25_search(tok, top_k=5)
        assert hits, f"BM25 found no hits for distinctive token {tok!r}"
        # The top hit's chunk text should contain the token.
        _, meta = _get_index_and_metadata()
        top_idx = hits[0][0]
        assert tok in meta[top_idx]["text"].lower()

    def test_hybrid_marks_mode_when_lexical_overlap(self):
        tok = _distinctive_token()
        if tok is None:
            pytest.skip("no distinctive token in corpus")
        results = retrieve(tok, top_k=3)
        assert results
        # With a real lexical token, BM25 must contribute → mode == "hybrid".
        assert results[0]["retrieval_mode"] == "hybrid"

    def test_disable_hybrid_falls_back_to_dense(self, monkeypatch):
        monkeypatch.setattr(retrieval_module.settings,
                            "hybrid_search_enabled", False)
        results = retrieve("summary of the document", top_k=3)
        assert results
        for r in results:
            assert r["retrieval_mode"] == "dense"
            assert r["bm25_score"] == 0.0

    def test_reload_rebuilds_bm25(self):
        reload_index()
        # After reload, internal state is consistent: bm25 corpus size
        # matches the metadata length.
        _, meta = _get_index_and_metadata()
        assert retrieval_module._bm25_corpus_size == len(meta)
        if meta:
            assert retrieval_module._bm25 is not None
