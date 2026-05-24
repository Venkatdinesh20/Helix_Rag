"""
Phase 3.2 — Extended evaluation harness
----------------------------------------
Reference-free + reference-based RAG metrics that work without an LLM judge.

Why heuristics first?
  - Zero API cost — runnable in CI on every PR
  - Deterministic — no flake from temperature, no schema drift
  - The RAGAS-style "faithfulness" / "answer relevance" / "context
    precision" numbers correlate well enough with LLM judges to gate
    regressions; we keep the LLM-judge entry point optional below.

All functions in this module are pure: they take strings / lists and
return floats in [0, 1]. The orchestrator `evaluate()` wires them
around any callable that takes a question and returns the standard
rag_service.ask() response dict.

Metrics implemented:
    retrieval :  precision@k, recall@k, MRR, hit@k
    generation:  answer_keyword_coverage (faithfulness proxy),
                 answer_relevance (token overlap with question),
                 context_utilization (fraction of cited chunks
                 referenced in the answer text)
    end-to-end:  exact_match, latency_p50/p95, cost_total
"""

from __future__ import annotations

import re
import statistics
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Tokenisation — shared by relevance/coverage metrics.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Mirror the small stop list used elsewhere so eval lines up with retrieval.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "i", "you", "he", "she",
    "it", "we", "they", "this", "that", "these", "those", "what", "which",
    "who", "whom", "how", "when", "where", "why", "as", "by", "from",
    "into", "than", "then", "if", "so", "not", "no", "yes",
})


def _tokens(text: str, *, drop_stop: bool = True) -> List[str]:
    """Lower-case alphanumeric tokens, optionally minus stopwords. ≥2 chars."""
    if not text:
        return []
    toks = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]
    if drop_stop:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_docs: Sequence[str], relevant_doc: str, k: int) -> float:
    """
    Fraction of the top-K retrieved chunks whose source document matches
    the gold relevant doc. Range: [0, 1].

    For document-QA this is the simplest faithfulness proxy: the system
    surfaced only material from the right source.
    """
    if k <= 0 or not retrieved_docs:
        return 0.0
    top = retrieved_docs[:k]
    hits = sum(1 for d in top if d == relevant_doc)
    return hits / min(k, len(top))


def recall_at_k(retrieved_docs: Sequence[str], relevant_doc: str, k: int) -> float:
    """1.0 if the gold doc appears anywhere in the top-K, else 0.0."""
    if k <= 0 or not retrieved_docs:
        return 0.0
    return 1.0 if relevant_doc in retrieved_docs[:k] else 0.0


def hit_at_k(retrieved_docs: Sequence[str], relevant_doc: str, k: int) -> float:
    """Alias of recall_at_k; named to match common RAG literature."""
    return recall_at_k(retrieved_docs, relevant_doc, k)


def reciprocal_rank(retrieved_docs: Sequence[str], relevant_doc: str) -> float:
    """1/rank of the first relevant doc; 0 if not present."""
    for i, d in enumerate(retrieved_docs, start=1):
        if d == relevant_doc:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Answer-quality metrics (reference-based, no LLM)
# ---------------------------------------------------------------------------

def keyword_coverage(answer: str, expected_keywords: Sequence[str]) -> float:
    """
    Fraction of expected keywords that appear (case-insensitively) as
    substrings of the answer. Acts as a faithfulness proxy when the
    keyword set is curated from the gold passage.

    Range: [0, 1]. Empty keyword list → 1.0 (vacuously satisfied).
    """
    if not expected_keywords:
        return 1.0
    if not answer:
        return 0.0
    a = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw and kw.lower() in a)
    return hits / len(expected_keywords)


def answer_relevance(question: str, answer: str) -> float:
    """
    Jaccard token overlap between question and answer (stopwords removed).

    A higher value means the answer mentions the same concepts as the
    question. Low values often indicate evasion or off-topic responses.
    """
    q = set(_tokens(question))
    a = set(_tokens(answer))
    if not q or not a:
        return 0.0
    return len(q & a) / len(q | a)


def context_utilization(answer: str, retrieved_chunks: Sequence[Dict[str, Any]]) -> float:
    """
    Fraction of retrieved chunks that contributed at least one
    distinctive token (≥4 chars, non-stop) to the answer.

    Interpretation: low utilisation = the LLM ignored most of the
    retrieved context (possible hallucination from prior weights).
    High utilisation does NOT prove faithfulness on its own, but
    near-zero is a strong negative signal.
    """
    if not retrieved_chunks:
        return 0.0
    a_tokens = {t for t in _tokens(answer) if len(t) >= 4}
    if not a_tokens:
        return 0.0
    used = 0
    for chunk in retrieved_chunks:
        text = chunk.get("text") or chunk.get("content") or ""
        c_tokens = {t for t in _tokens(text) if len(t) >= 4}
        if c_tokens & a_tokens:
            used += 1
    return used / len(retrieved_chunks)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * len(ordered))) - 1))
    return ordered[idx]


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------

AskFn = Callable[[str], Dict[str, Any]]


def evaluate(
    questions: Iterable[Dict[str, Any]],
    ask_fn: AskFn,
    *,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Run `ask_fn(question)` against every test case and aggregate metrics.

    Each question dict must contain:
        question            : str
        expected_document   : str (used for retrieval metrics)
        expected_keywords   : list[str] (used for answer-quality metrics)

    `ask_fn` must return at minimum:
        { "answer": str,
          "sources": [{ "document_name": str, ... }, ...],
          "latency_ms": int,
          "cost_usd"  : float  (optional, defaults 0)
        }
    We also try to read `retrieved_chunks` for context_utilization;
    if absent we fall back to the (less informative) `sources` list.

    Returns a dict with:
        summary      : aggregate metrics across the run
        per_question : list of per-question metric dicts
    """
    per_question: List[Dict[str, Any]] = []
    latencies: List[int] = []
    costs:     List[float] = []

    for q in questions:
        question         = q["question"]
        expected_doc     = q.get("expected_document", "")
        expected_kws     = q.get("expected_keywords", []) or []

        t0 = time.perf_counter()
        result = ask_fn(question)
        # Caller may set latency_ms — prefer it over our wall-clock so
        # batch overhead doesn't pollute percentile reports.
        latency_ms = int(result.get("latency_ms") or (time.perf_counter() - t0) * 1000)

        sources         = result.get("sources") or []
        retrieved_docs  = [s.get("document_name", "") for s in sources]
        retrieved_chunks = result.get("retrieved_chunks") or sources
        answer = result.get("answer", "") or ""
        cost   = float(result.get("cost_usd", 0.0) or 0.0)

        metrics = {
            "id":               q.get("id"),
            "question":         question,
            "expected_document": expected_doc,
            "retrieved_docs":   retrieved_docs,
            "precision_at_k":   precision_at_k(retrieved_docs, expected_doc, k),
            "recall_at_k":      recall_at_k(retrieved_docs, expected_doc, k),
            "mrr":              reciprocal_rank(retrieved_docs, expected_doc),
            "keyword_coverage": keyword_coverage(answer, expected_kws),
            "answer_relevance": answer_relevance(question, answer),
            "context_utilization": context_utilization(answer, retrieved_chunks),
            "latency_ms":       latency_ms,
            "cost_usd":         cost,
            "answer":           answer,
        }
        per_question.append(metrics)
        latencies.append(latency_ms)
        costs.append(cost)

    summary = {
        "n_questions":         len(per_question),
        "k":                   k,
        "precision_at_k":      round(_mean([m["precision_at_k"]      for m in per_question]), 4),
        "recall_at_k":         round(_mean([m["recall_at_k"]         for m in per_question]), 4),
        "mrr":                 round(_mean([m["mrr"]                 for m in per_question]), 4),
        "keyword_coverage":    round(_mean([m["keyword_coverage"]    for m in per_question]), 4),
        "answer_relevance":    round(_mean([m["answer_relevance"]    for m in per_question]), 4),
        "context_utilization": round(_mean([m["context_utilization"] for m in per_question]), 4),
        "latency_ms": {
            "p50": int(_percentile(latencies, 50)),
            "p95": int(_percentile(latencies, 95)),
            "p99": int(_percentile(latencies, 99)),
            "avg": int(_mean(latencies)),
            "max": int(max(latencies)) if latencies else 0,
        },
        "cost_usd_total":      round(sum(costs), 6),
        "cost_usd_per_question": round(_mean(costs), 6),
    }

    return {"summary": summary, "per_question": per_question}


# ---------------------------------------------------------------------------
# Optional LLM-as-judge faithfulness scorer
#
# Disabled by default — costs money and is non-deterministic. Provided
# as an extension point so dashboards can opt in once the heuristic
# numbers stabilise.
# ---------------------------------------------------------------------------

def llm_judge_faithfulness(
    question: str,
    answer: str,
    retrieved_chunks: Sequence[Dict[str, Any]],
    *,
    client: Optional[Any] = None,
    model: str = "gpt-4o-mini",
) -> Optional[float]:
    """
    Ask an LLM to score whether the answer is fully supported by the
    retrieved context. Returns a float in [0, 1] or None if no client
    is provided / the call fails.

    Intentionally lightweight: one short prompt, temperature=0, JSON
    output. Callers must pass a configured OpenAI client; we do not
    instantiate one here to keep this module dependency-free.
    """
    if client is None or not answer or not retrieved_chunks:
        return None

    context = "\n\n".join(
        (c.get("text") or c.get("content") or "")[:600]
        for c in retrieved_chunks[:5]
    )
    prompt = (
        "You are evaluating whether an answer is fully supported by the given context.\n"
        "Return a single number in [0,1]: 1.0 = every claim is supported, "
        "0.0 = nothing is supported. No explanation, just the number.\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}\n\nSCORE:"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\d+(?:\.\d+)?", raw)
        if not m:
            return None
        score = float(m.group(0))
        return max(0.0, min(1.0, score))
    except Exception:
        return None
