"""
Tests for app.eval_harness (Phase 3.2).

Strategy:
  - Unit-test each metric on hand-crafted inputs with known answers.
  - Integration-test `evaluate()` with a fake `ask_fn` so we don't
    need the real vector store / LLM.
  - Smoke-test the CLI runner indirectly by validating its imports.
"""
from __future__ import annotations

import json
import os

import pytest

from app.eval_harness import (
    answer_relevance,
    context_utilization,
    evaluate,
    hit_at_k,
    keyword_coverage,
    llm_judge_faithfulness,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

def test_precision_at_k_all_relevant():
    assert precision_at_k(["A", "A", "A"], "A", k=3) == 1.0


def test_precision_at_k_partial():
    # 1 of 3 retrieved match
    assert precision_at_k(["A", "B", "C"], "A", k=3) == pytest.approx(1 / 3)


def test_precision_at_k_none_relevant():
    assert precision_at_k(["B", "C", "D"], "A", k=3) == 0.0


def test_precision_at_k_k_larger_than_results():
    # Only 2 retrieved, k=5 — denominator should be 2, not 5
    assert precision_at_k(["A", "A"], "A", k=5) == 1.0


def test_precision_at_k_empty():
    assert precision_at_k([], "A", k=3) == 0.0


def test_precision_at_k_zero_k():
    assert precision_at_k(["A"], "A", k=0) == 0.0


# ---------------------------------------------------------------------------
# recall_at_k / hit_at_k / reciprocal_rank
# ---------------------------------------------------------------------------

def test_recall_and_hit_are_aliased():
    docs = ["X", "Y", "A", "B"]
    assert recall_at_k(docs, "A", k=5) == 1.0
    assert hit_at_k(docs, "A", k=5) == 1.0
    assert recall_at_k(docs, "Z", k=5) == 0.0


def test_recall_respects_k_window():
    # Gold doc at position 4 — out of top-3
    assert recall_at_k(["X", "Y", "Z", "A"], "A", k=3) == 0.0
    assert recall_at_k(["X", "Y", "Z", "A"], "A", k=4) == 1.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["A", "B"], "A") == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["X", "Y", "A"], "A") == pytest.approx(1 / 3)


def test_reciprocal_rank_missing():
    assert reciprocal_rank(["X", "Y"], "A") == 0.0


# ---------------------------------------------------------------------------
# keyword_coverage
# ---------------------------------------------------------------------------

def test_keyword_coverage_all_present():
    ans = "Refunds are available within 30 days of purchase."
    kws = ["30 days", "refund", "purchase"]
    assert keyword_coverage(ans, kws) == 1.0


def test_keyword_coverage_partial():
    ans = "We accept refunds."
    kws = ["30 days", "refund", "purchase"]
    # Only "refund" appears
    assert keyword_coverage(ans, kws) == pytest.approx(1 / 3)


def test_keyword_coverage_case_insensitive():
    assert keyword_coverage("REFUND POLICY", ["refund"]) == 1.0


def test_keyword_coverage_empty_keywords_is_vacuously_full():
    # No keywords to satisfy → trivially 1.0; avoids penalising free-form QA
    assert keyword_coverage("anything", []) == 1.0


def test_keyword_coverage_empty_answer():
    assert keyword_coverage("", ["a"]) == 0.0


# ---------------------------------------------------------------------------
# answer_relevance
# ---------------------------------------------------------------------------

def test_answer_relevance_exact_topic():
    q = "What is the refund policy?"
    a = "The refund policy allows full refunds."
    # Overlap on {refund, policy} — score should be reasonably high
    assert answer_relevance(q, a) > 0.3


def test_answer_relevance_off_topic_is_zero():
    q = "What is the refund policy?"
    a = "Penguins are flightless birds native to Antarctica."
    assert answer_relevance(q, a) == 0.0


def test_answer_relevance_empty_inputs():
    assert answer_relevance("", "anything") == 0.0
    assert answer_relevance("anything", "") == 0.0


def test_answer_relevance_stopwords_ignored():
    # The only shared token (refund) is content; "the" / "is" filtered.
    q = "What is the refund?"
    a = "Refund."
    assert answer_relevance(q, a) == 1.0


# ---------------------------------------------------------------------------
# context_utilization
# ---------------------------------------------------------------------------

def test_context_utilization_full_use():
    chunks = [
        {"text": "Refunds are available within thirty days of original purchase."},
        {"text": "Customer support email is support@company.com."},
    ]
    answer = "Refunds within thirty days; contact support@company.com for help."
    # Both chunks contribute distinctive tokens
    assert context_utilization(answer, chunks) == 1.0


def test_context_utilization_partial():
    chunks = [
        {"text": "Refunds are available within thirty days of purchase."},
        {"text": "The corporate headquarters is located in Mumbai India."},
    ]
    answer = "Refunds are available within thirty days."
    # Only chunk 1 is referenced
    assert context_utilization(answer, chunks) == 0.5


def test_context_utilization_no_chunks():
    assert context_utilization("anything", []) == 0.0


def test_context_utilization_empty_answer():
    assert context_utilization("", [{"text": "anything"}]) == 0.0


# ---------------------------------------------------------------------------
# evaluate() — end-to-end with fake ask_fn
# ---------------------------------------------------------------------------

def _fake_ask_perfect(question: str):
    """Always returns the correct doc and an answer covering keywords."""
    return {
        "answer":     "Refunds are available within 30 days of purchase. Contact support@company.com.",
        "sources":    [
            {"document_name": "company_policies.txt", "chunk_id": "c1", "score": 0.9, "text": "Refunds within 30 days of purchase."},
            {"document_name": "company_policies.txt", "chunk_id": "c2", "score": 0.7, "text": "Support email support@company.com."},
        ],
        "latency_ms": 100,
        "cost_usd":   0.000123,
    }


def _fake_ask_bad(question: str):
    """Wrong doc, no keywords."""
    return {
        "answer":     "I don't know.",
        "sources":    [{"document_name": "wrong.txt", "chunk_id": "x", "score": 0.1, "text": "irrelevant text"}],
        "latency_ms": 200,
        "cost_usd":   0.0,
    }


_GOLD_QUESTIONS = [
    {
        "id": "q1",
        "question": "What is the refund policy?",
        "expected_document": "company_policies.txt",
        "expected_keywords": ["30 days", "refund", "purchase"],
    },
    {
        "id": "q2",
        "question": "How can I contact support?",
        "expected_document": "company_policies.txt",
        "expected_keywords": ["support@company.com"],
    },
]


def test_evaluate_perfect_run_scores_one():
    report = evaluate(_GOLD_QUESTIONS, _fake_ask_perfect, k=5)
    s = report["summary"]
    assert s["n_questions"] == 2
    assert s["precision_at_k"] == 1.0
    assert s["recall_at_k"] == 1.0
    assert s["mrr"] == 1.0
    assert s["keyword_coverage"] == 1.0
    assert s["latency_ms"]["p50"] == 100
    assert s["cost_usd_total"] == pytest.approx(0.000246)


def test_evaluate_bad_run_scores_zero():
    report = evaluate(_GOLD_QUESTIONS, _fake_ask_bad, k=5)
    s = report["summary"]
    assert s["precision_at_k"] == 0.0
    assert s["recall_at_k"] == 0.0
    assert s["mrr"] == 0.0
    assert s["keyword_coverage"] == 0.0


def test_evaluate_per_question_shape():
    report = evaluate(_GOLD_QUESTIONS, _fake_ask_perfect, k=5)
    assert len(report["per_question"]) == 2
    pq = report["per_question"][0]
    for key in (
        "id", "question", "expected_document", "retrieved_docs",
        "precision_at_k", "recall_at_k", "mrr",
        "keyword_coverage", "answer_relevance", "context_utilization",
        "latency_ms", "cost_usd", "answer",
    ):
        assert key in pq


def test_evaluate_with_real_questions_file():
    """Sanity: the harness can consume the repo's own gold file shape."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "evaluation", "test_questions.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    # Use only the first 3 questions to keep this fast and deterministic.
    report = evaluate(gold[:3], _fake_ask_bad, k=3)
    assert report["summary"]["n_questions"] == 3


# ---------------------------------------------------------------------------
# llm_judge_faithfulness — guard rails (we don't call the live API)
# ---------------------------------------------------------------------------

def test_llm_judge_returns_none_without_client():
    assert llm_judge_faithfulness("q", "a", [{"text": "ctx"}], client=None) is None


def test_llm_judge_returns_none_on_empty_inputs():
    class _FakeClient: ...
    assert llm_judge_faithfulness("q", "",  [{"text": "x"}], client=_FakeClient()) is None
    assert llm_judge_faithfulness("q", "a", [],              client=_FakeClient()) is None


def test_llm_judge_parses_numeric_score():
    """Stubbed OpenAI client returns a string; harness clamps to [0,1]."""

    class _Resp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]

    class _FakeChatCompletions:
        def create(self, **kwargs):  # noqa: ARG002
            return _Resp("0.83\n")

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        chat = _FakeChat()

    score = llm_judge_faithfulness(
        "Q?", "A.", [{"text": "context"}], client=_FakeClient(), model="x"
    )
    assert score == pytest.approx(0.83)


def test_llm_judge_handles_exceptions():
    class _Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):  # noqa: ARG004
                    raise RuntimeError("api down")

    assert llm_judge_faithfulness("q", "a", [{"text": "x"}], client=_Boom()) is None
