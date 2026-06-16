"""
RAG Observability — Core Tracer
Tracks latency, token cost, retrieval quality, and confidence per query.
Stores traces in SQLite for dashboard and CI regression gating.
"""

import time
import sqlite3
import json
import uuid
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "traces.db"
DB_PATH.parent.mkdir(exist_ok=True)

# Pricing (per 1K tokens) — update as needed
COST_TABLE = {
    "gpt-4o":              {"input": 0.005,  "output": 0.015},
    "gpt-4o-mini":         {"input": 0.00015,"output": 0.0006},
    "gpt-4.1":             {"input": 0.002,  "output": 0.008},
    "claude-sonnet-4-5":   {"input": 0.003,  "output": 0.015},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}


@dataclass
class RAGTrace:
    trace_id: str
    query: str
    model: str

    # Latency breakdown (ms)
    latency_retrieval_ms: float
    latency_rerank_ms: float
    latency_llm_ms: float
    latency_total_ms: float

    # Token usage
    tokens_input: int
    tokens_output: int
    tokens_embedding: int
    cost_usd: float

    # Retrieval quality
    num_docs_retrieved: int
    num_docs_after_rerank: int
    top_score: float           # highest retrieval score
    mean_score: float          # mean retrieval score
    filter_score_threshold: float

    # LLM output quality
    confidence_score: float    # 0-1, from LLM self-assessment or heuristic
    has_citation: bool         # did response include source citation?
    faithfulness_score: float  # 0-1, RAGAS-style (set post-eval or -1 if not computed)
    answer_relevance: float    # 0-1 (set post-eval or -1 if not computed)

    # Outcome
    response_length_chars: int
    was_fallback: bool         # did it hit the "no context found" fallback?
    error: Optional[str]

    timestamp: float


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY,
            query TEXT,
            model TEXT,
            latency_retrieval_ms REAL,
            latency_rerank_ms REAL,
            latency_llm_ms REAL,
            latency_total_ms REAL,
            tokens_input INTEGER,
            tokens_output INTEGER,
            tokens_embedding INTEGER,
            cost_usd REAL,
            num_docs_retrieved INTEGER,
            num_docs_after_rerank INTEGER,
            top_score REAL,
            mean_score REAL,
            filter_score_threshold REAL,
            confidence_score REAL,
            has_citation INTEGER,
            faithfulness_score REAL,
            answer_relevance REAL,
            response_length_chars INTEGER,
            was_fallback INTEGER,
            error TEXT,
            timestamp REAL
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def compute_cost(model: str, tokens_input: int, tokens_output: int, tokens_embedding: int = 0) -> float:
    rates = COST_TABLE.get(model, {"input": 0.001, "output": 0.003})
    cost = (tokens_input / 1000 * rates["input"]) + (tokens_output / 1000 * rates["output"])
    emb_rates = COST_TABLE.get("text-embedding-3-small", {"input": 0.00002})
    cost += tokens_embedding / 1000 * emb_rates["input"]
    return round(cost, 6)


def save_trace(trace: RAGTrace):
    conn = sqlite3.connect(DB_PATH)
    d = asdict(trace)
    d["has_citation"] = int(d["has_citation"])
    d["was_fallback"] = int(d["was_fallback"])
    cols = ", ".join(d.keys())
    placeholders = ", ".join("?" * len(d))
    conn.execute(f"INSERT OR REPLACE INTO traces ({cols}) VALUES ({placeholders})", list(d.values()))
    conn.commit()
    conn.close()


def get_traces(limit: int = 500) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM traces ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_percentiles(metric: str, window: int = 100) -> dict:
    """Return p50/p95/p99 for any numeric metric over last N traces."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        f"SELECT {metric} FROM traces WHERE {metric} IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
        (window,)
    ).fetchall()
    conn.close()
    values = sorted([r[0] for r in rows if r[0] is not None])
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "count": 0}
    n = len(values)
    return {
        "p50":   round(values[int(n * 0.50)], 2),
        "p95":   round(values[int(n * 0.95)], 2),
        "p99":   round(values[min(int(n * 0.99), n-1)], 2),
        "mean":  round(statistics.mean(values), 2),
        "count": n
    }


def get_summary_stats(window: int = 100) -> dict:
    traces = get_traces(window)
    if not traces:
        return {}

    def _pct(key):
        return get_percentiles(key, window)

    total_cost = sum(t["cost_usd"] for t in traces)
    fallback_rate = sum(1 for t in traces if t["was_fallback"]) / len(traces)
    citation_rate = sum(1 for t in traces if t["has_citation"]) / len(traces)
    error_rate    = sum(1 for t in traces if t["error"]) / len(traces)
    avg_faith     = [t["faithfulness_score"] for t in traces if t["faithfulness_score"] >= 0]

    return {
        "total_traces": len(traces),
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_per_query_usd": round(total_cost / len(traces), 6),
        "fallback_rate": round(fallback_rate, 3),
        "citation_rate": round(citation_rate, 3),
        "error_rate": round(error_rate, 3),
        "avg_faithfulness": round(statistics.mean(avg_faith), 3) if avg_faith else -1,
        "latency_total": _pct("latency_total_ms"),
        "latency_retrieval": _pct("latency_retrieval_ms"),
        "latency_llm": _pct("latency_llm_ms"),
        "confidence": _pct("confidence_score"),
        "top_score": _pct("top_score"),
    }


class RAGObserver:
    """
    Context-manager based observer. Wrap your RAG pipeline stages with this.

    Usage:
        obs = RAGObserver(query="What is X?", model="gpt-4o-mini")
        with obs.track("retrieval"):
            docs = retriever.get(query)
        with obs.track("rerank"):
            docs = reranker.rerank(docs)
        with obs.track("llm"):
            response = llm.generate(query, docs)
        obs.finish(
            docs_retrieved=len(docs),
            docs_after_rerank=top_k,
            top_score=docs[0].score,
            mean_score=mean([d.score for d in docs]),
            tokens_input=usage.input,
            tokens_output=usage.output,
            tokens_embedding=emb_tokens,
            confidence=0.92,
            has_citation=True,
            faithfulness=-1,   # set later by eval pipeline
            answer_relevance=-1,
            response=response_text,
            was_fallback=False
        )
    """

    def __init__(self, query: str, model: str, filter_score_threshold: float = 0.7):
        self.query = query
        self.model = model
        self.filter_score_threshold = filter_score_threshold
        self.trace_id = str(uuid.uuid4())
        self._timings: dict[str, float] = {}
        self._t_start = time.perf_counter()
        self._stage_start: Optional[float] = None
        self._current_stage: Optional[str] = None

    @contextmanager
    def track(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[stage] = (time.perf_counter() - t0) * 1000  # ms

    def finish(
        self,
        docs_retrieved: int,
        docs_after_rerank: int,
        top_score: float,
        mean_score: float,
        tokens_input: int,
        tokens_output: int,
        tokens_embedding: int,
        confidence: float,
        has_citation: bool,
        faithfulness: float,
        answer_relevance: float,
        response: str,
        was_fallback: bool,
        error: Optional[str] = None
    ) -> RAGTrace:
        total_ms = (time.perf_counter() - self._t_start) * 1000
        cost = compute_cost(self.model, tokens_input, tokens_output, tokens_embedding)

        trace = RAGTrace(
            trace_id=self.trace_id,
            query=self.query,
            model=self.model,
            latency_retrieval_ms=self._timings.get("retrieval", 0),
            latency_rerank_ms=self._timings.get("rerank", 0),
            latency_llm_ms=self._timings.get("llm", 0),
            latency_total_ms=total_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_embedding=tokens_embedding,
            cost_usd=cost,
            num_docs_retrieved=docs_retrieved,
            num_docs_after_rerank=docs_after_rerank,
            top_score=top_score,
            mean_score=mean_score,
            filter_score_threshold=self.filter_score_threshold,
            confidence_score=confidence,
            has_citation=has_citation,
            faithfulness_score=faithfulness,
            answer_relevance=answer_relevance,
            response_length_chars=len(response),
            was_fallback=was_fallback,
            error=error,
            timestamp=time.time()
        )
        save_trace(trace)
        return trace
