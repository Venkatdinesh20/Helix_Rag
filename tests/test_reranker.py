"""
Tests for the cross-encoder reranker (Phase 2.2).

We verify both the pure module behaviour (mockable, fast) and that the
end-to-end `retrieve()` path benefits from rerank ordering when enabled.

Strategy:
  - For unit tests we monkeypatch `_get_model` with a tiny fake model so
    we never download weights in CI and the tests run in milliseconds.
  - We test the graceful-failure path explicitly: if model loading fails
    or `predict` raises, the input is returned with `rerank_score=None`
    and no exception escapes.
"""

from __future__ import annotations

import pytest

from app import reranker as reranker_module
from app.reranker import rerank


# ---------------------------------------------------------------------------
# Fake cross-encoder models for deterministic tests
# ---------------------------------------------------------------------------

class _FakeModel:
    """Returns a predetermined score per (query, passage) pair."""

    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def predict(self, pairs, show_progress_bar=False):  # noqa: ARG002
        self.calls += 1
        # The reranker passes one pair per candidate, so we just return
        # `len(pairs)` scores from our predetermined list.
        return list(self._scores[: len(pairs)])


class _BrokenModel:
    def predict(self, pairs, show_progress_bar=False):  # noqa: ARG002
        raise RuntimeError("simulated cross-encoder failure")


def _chunks(*texts):
    return [{"chunk_id": f"c{i}", "text": t, "score": 0.5}
            for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Module-level unit tests
# ---------------------------------------------------------------------------

class TestRerankUnit:

    def test_empty_chunks_returns_empty(self):
        assert rerank("query", []) == []

    def test_blank_query_short_circuits(self, monkeypatch):
        # Should not even attempt to load the model on a blank query.
        called = {"n": 0}

        def _spy():
            called["n"] += 1
            return None
        monkeypatch.setattr(reranker_module, "_get_model", _spy)

        chunks = _chunks("a", "b", "c")
        out = rerank("   ", chunks)
        assert out == chunks
        assert called["n"] == 0

    def test_reorders_by_score(self, monkeypatch):
        # Three chunks; rerank scores 0.1, 0.9, 0.4 → expected order c1, c2, c0.
        fake = _FakeModel(scores=[0.1, 0.9, 0.4])
        monkeypatch.setattr(reranker_module, "_get_model", lambda: fake)

        chunks = _chunks("first", "second", "third")
        out = rerank("q", chunks)
        assert [c["chunk_id"] for c in out] == ["c1", "c2", "c0"]
        # Scores are attached and floats.
        assert [c["rerank_score"] for c in out] == [pytest.approx(0.9),
                                                     pytest.approx(0.4),
                                                     pytest.approx(0.1)]

    def test_top_n_truncates_after_rerank(self, monkeypatch):
        fake = _FakeModel(scores=[0.2, 0.8, 0.6, 0.1])
        monkeypatch.setattr(reranker_module, "_get_model", lambda: fake)

        chunks = _chunks("a", "b", "c", "d")
        out = rerank("q", chunks, top_n=2)
        assert len(out) == 2
        assert [c["chunk_id"] for c in out] == ["c1", "c2"]

    def test_model_unavailable_preserves_order(self, monkeypatch):
        # Simulate model load failure: rerank must return chunks in
        # input order with rerank_score=None — never raise.
        monkeypatch.setattr(reranker_module, "_get_model", lambda: None)

        chunks = _chunks("a", "b", "c")
        out = rerank("q", chunks)
        assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2"]
        assert all(c["rerank_score"] is None for c in out)

    def test_predict_failure_preserves_order(self, monkeypatch):
        # If predict() raises mid-call, fall back to hybrid order.
        monkeypatch.setattr(reranker_module, "_get_model",
                            lambda: _BrokenModel())

        chunks = _chunks("a", "b", "c")
        out = rerank("q", chunks)
        assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2"]
        assert all(c["rerank_score"] is None for c in out)

    def test_load_failure_caches(self, monkeypatch):
        # _get_model must not retry on every call after a load failure —
        # otherwise we'd hammer HuggingFace on every request in a
        # network-flaky environment.
        attempts = {"n": 0}

        class _ImportBoom:
            def __init__(self, *a, **kw):
                attempts["n"] += 1
                raise RuntimeError("network down")

        # Reset module state and replace the CrossEncoder import target.
        reranker_module._model = None
        reranker_module._model_load_failed = False

        import sentence_transformers
        monkeypatch.setattr(sentence_transformers, "CrossEncoder", _ImportBoom)

        assert reranker_module._get_model() is None
        assert reranker_module._get_model() is None
        assert attempts["n"] == 1  # only tried once

        # Reset for other tests.
        reranker_module._model_load_failed = False


# ---------------------------------------------------------------------------
# Integration with retrieve() — uses a stubbed reranker so we don't depend
# on the real model being downloaded.
# ---------------------------------------------------------------------------

class TestRetrieveIntegration:

    def test_retrieve_applies_rerank_scores(self, monkeypatch):
        # Patch the reranker used inside retrieve() to a deterministic stub.
        from app import retrieval as retrieval_module

        def _stub_rerank(query, chunks, top_n=None):
            # Reverse order, assigning decreasing scores so the result is
            # sorted by rerank_score descending and we can prove the step ran.
            reversed_chunks = list(reversed(chunks))
            n = len(reversed_chunks)
            for i, c in enumerate(reversed_chunks):
                c["rerank_score"] = float(n - i)  # n, n-1, ..., 1
            return reversed_chunks[: top_n] if top_n is not None else reversed_chunks

        # Patch via the imported symbol used at call site (lazy import
        # inside retrieve()), so we patch the source module.
        monkeypatch.setattr("app.reranker.rerank", _stub_rerank)

        from app.retrieval import retrieve
        results = retrieve("any query", top_k=3)
        if not results:
            pytest.skip("empty index")
        # Every result has a rerank_score after the stub ran.
        assert all(r.get("rerank_score") is not None for r in results)
        # And they are sorted descending by rerank_score.
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_disable_rerank(self, monkeypatch):
        from app import retrieval as retrieval_module
        from app.retrieval import retrieve

        monkeypatch.setattr(retrieval_module.settings,
                            "rerank_enabled", False)

        results = retrieve("any query", top_k=3)
        if not results:
            pytest.skip("empty index")
        # When disabled, every chunk has rerank_score=None.
        assert all(r["rerank_score"] is None for r in results)
