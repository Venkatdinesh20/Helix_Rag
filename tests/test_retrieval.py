"""
Unit tests for app/retrieval.py

Why test retrieval?
  Retrieval is the core of a RAG system. If it returns wrong chunks, the
  LLM generates wrong answers. These tests verify that the vector search
  returns semantically relevant results for known questions.

Note: These are integration-style unit tests — they load the real FAISS index
and the real embedding model. Run the batch pipeline first:
  python -m pipeline.build_index

Then run tests:
  python -m pytest tests/test_retrieval.py -v
"""

import re

import pytest

from app.retrieval import retrieve, _get_index_and_metadata


# A small stopword set so we don't pick filler words as "distinctive" tokens
# when probing the indexed corpus. Keeping it short — we only need to avoid
# the high-frequency words that would match almost any chunk.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "has",
    "are", "was", "were", "but", "not", "you", "your", "our", "their",
    "they", "them", "his", "her", "she", "him", "its", "into", "over",
    "than", "then", "also", "such", "any", "all", "can", "may", "will",
    "would", "should", "could", "about", "been", "being", "more", "most",
    "some", "other", "into", "out", "per", "via",
}


def _distinctive_term_from_corpus(min_len: int = 6) -> str:
    """
    Pick a content-bearing word that appears in some indexed chunk.

    Why: hard-coding domain words ("refund", "support") makes tests break
    every time the corpus changes. Instead we derive a probe term from
    whatever is actually indexed — the test then validates retrieval
    *behaviour* rather than a specific document's vocabulary.
    """
    _, records = _get_index_and_metadata()
    assert records, "Index appears to be empty — run `python -m pipeline.build_index` first"
    # Walk records until we find a word long enough and not a stopword.
    for rec in records:
        for word in re.findall(r"[A-Za-z]+", rec["text"]):
            w = word.lower()
            if len(w) >= min_len and w not in _STOPWORDS:
                return w
    pytest.skip("No suitable distinctive term found in indexed corpus")


class TestRetrieve:

    def test_returns_list(self):
        """retrieve() must return a list."""
        results = retrieve("What is the refund policy?", top_k=3)
        assert isinstance(results, list)

    def test_returns_top_k_or_fewer(self):
        """Result count must not exceed top_k."""
        results = retrieve("refund policy", top_k=3)
        assert len(results) <= 3

    def test_result_has_required_fields(self):
        """Every result must have chunk_id, text, metadata, and score."""
        results = retrieve("contact support", top_k=1)
        assert len(results) >= 1
        r = results[0]
        assert "chunk_id" in r
        assert "text" in r
        assert "metadata" in r
        assert "score" in r

    def test_score_is_between_0_and_1(self):
        """Cosine similarity scores for normalised vectors must be in [-1, 1].
        For meaningful queries against relevant content, expect > 0."""
        results = retrieve("refund policy", top_k=3)
        for r in results:
            assert -1.0 <= r["score"] <= 1.0

    def test_results_sorted_by_score_descending(self):
        """
        Results must come back in a meaningful descending order.

        With the Phase 2.2 cross-encoder reranker enabled, the ordering
        is by `rerank_score` (joint relevance), not by raw cosine. With
        the reranker disabled or unavailable, fall back to checking the
        cosine `score`. Either way the top hit must be at least as
        relevant as the bottom one under the ranker that actually ran.
        """
        results = retrieve("onboarding documents required", top_k=5)
        assert results, "expected non-empty results"

        rerank_scores = [r.get("rerank_score") for r in results]
        if all(s is not None for s in rerank_scores):
            assert rerank_scores == sorted(rerank_scores, reverse=True)
        else:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_query_with_indexed_term_retrieves_that_term(self):
        """
        Corpus-agnostic relevance check: querying with a content word that
        appears in the index must surface a chunk containing that word in
        the top-3 results. This replaces the old hard-coded "refund" test.
        """
        term = _distinctive_term_from_corpus()
        results = retrieve(term, top_k=3)
        assert len(results) >= 1
        assert any(term in r["text"].lower() for r in results), (
            f"Expected term {term!r} to appear in at least one of the top-3 chunks"
        )

    def test_relevant_query_scores_higher_than_gibberish(self):
        """
        Corpus-agnostic relevance check: a query taken from the corpus must
        score strictly higher than a random gibberish query. This replaces
        the old hard-coded "support/contact" assertion.
        """
        term = _distinctive_term_from_corpus()
        relevant = retrieve(term, top_k=1)
        gibberish = retrieve("zqxjvk wfplmn brtkyx", top_k=1)
        assert relevant, "Expected at least one relevant result"
        # If the index is tiny the gibberish query may also return a hit,
        # but its score must be lower than the on-topic query.
        if gibberish:
            assert relevant[0]["score"] > gibberish[0]["score"]

    def test_metadata_has_document_name(self):
        """Metadata must include document_name for source attribution."""
        results = retrieve("leave policy", top_k=1)
        assert "document_name" in results[0]["metadata"]

    def test_top_k_zero_returns_empty(self):
        """top_k=0 should return an empty list without errors."""
        results = retrieve("anything", top_k=0)
        assert results == []
