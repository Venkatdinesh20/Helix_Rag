"""
Phase 2.4 — Cost & token tracking
----------------------------------
Cheap, thread-safe in-memory accounting for LLM spend.

Why in-memory?
  - Single-instance Cloud Run friendly. For multi-instance the same
    interface can later be backed by Redis / SQLite — no caller changes
    required.
  - No external dependency, no IO on the hot path: each `record()` is a
    few field updates under a lock.

What it tracks (per call AND aggregated globally + per-session):
  - prompt_tokens / completion_tokens / total_tokens
  - cost_usd (derived from a pricing table keyed by model)
  - request count, error count, latency samples (rolling)

What it does NOT do:
  - Per-user billing, hard quotas, persistence across restarts. Those
    require a real store and authn — out of scope for the recruiter demo.

Pricing table is conservative and easy to update; unknown models fall
back to gpt-4o-mini rates with a logged warning rather than crashing.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens) — keep in sync with provider docs.
# Source of truth as of 2025: https://openai.com/api/pricing/
# ---------------------------------------------------------------------------

# (input_per_1m, output_per_1m)
_PRICING: Dict[str, tuple[float, float]] = {
    "gpt-4o-mini":      (0.150, 0.600),
    "gpt-4o":           (2.500, 10.000),
    "gpt-4-turbo":      (10.000, 30.000),
    "gpt-3.5-turbo":    (0.500, 1.500),
}

# Model names that are free / not billable.
_FREE_MODELS = frozenset({"demo-mode", "guardrail", "conversational"})


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Return cost in USD for a single LLM call.

    Free / sentinel models (demo, guardrail short-circuits, hard-coded
    small-talk) return 0 — they never hit the provider.

    Unknown billable models fall back to gpt-4o-mini rates rather than
    raising; we'd rather under-bill in a dashboard than crash a request.
    """
    if not model or model in _FREE_MODELS:
        return 0.0
    if prompt_tokens < 0 or completion_tokens < 0:
        return 0.0
    rates = _PRICING.get(model) or _PRICING["gpt-4o-mini"]
    in_rate, out_rate = rates
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

# Cap the rolling latency buffer so memory stays bounded even under load.
_LATENCY_BUFFER_SIZE = 1000


@dataclass
class _SessionStats:
    """Per-session running totals. Created lazily on first record."""
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0
    cost_usd:          float = 0.0
    requests:          int = 0
    errors:            int = 0


@dataclass
class CostTracker:
    """
    Thread-safe singleton-style aggregator.

    Use `get_tracker()` from callers — direct instantiation is reserved
    for tests that need an isolated counter.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Global totals
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    cost_usd:          float = 0.0
    requests:          int   = 0
    errors:            int   = 0
    started_at:        float = field(default_factory=time.time)

    # Per-session totals
    _sessions: Dict[str, _SessionStats] = field(default_factory=lambda: defaultdict(_SessionStats))

    # Rolling latency samples (ms); used for P50/P95/P99 in /stats.
    _latencies: Deque[int] = field(default_factory=lambda: deque(maxlen=_LATENCY_BUFFER_SIZE))

    # Per-model call counts for breakdown in dashboards.
    _model_calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ----- mutation --------------------------------------------------------
    def record(
        self,
        *,
        session_id: Optional[str],
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        error: bool = False,
    ) -> float:
        """
        Record one LLM (or short-circuit) call. Returns the cost in USD.

        Called from rag_service.ask() and ask_stream() at the end of
        every request, regardless of success/failure, so the dashboard
        sees error rates too.
        """
        prompt_tokens     = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        cost = estimate_cost(model, prompt_tokens, completion_tokens)

        with self._lock:
            self.prompt_tokens     += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens      += prompt_tokens + completion_tokens
            self.cost_usd          += cost
            self.requests          += 1
            if error:
                self.errors += 1
            self._latencies.append(max(0, int(latency_ms or 0)))
            if model:
                self._model_calls[model] += 1

            if session_id:
                s = self._sessions[session_id]
                s.prompt_tokens     += prompt_tokens
                s.completion_tokens += completion_tokens
                s.total_tokens      += prompt_tokens + completion_tokens
                s.cost_usd          += cost
                s.requests          += 1
                if error:
                    s.errors += 1

        return cost

    # ----- read ------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        """Atomic copy of global stats — safe to serialise to JSON."""
        with self._lock:
            latencies = list(self._latencies)
            model_calls = dict(self._model_calls)
            return {
                "uptime_seconds":    int(time.time() - self.started_at),
                "requests":          self.requests,
                "errors":            self.errors,
                "error_rate":        (self.errors / self.requests) if self.requests else 0.0,
                "prompt_tokens":     self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens":      self.total_tokens,
                "cost_usd":          round(self.cost_usd, 6),
                "latency_ms":        _latency_summary(latencies),
                "model_calls":       model_calls,
                "sessions":          len(self._sessions),
            }

    def session_snapshot(self, session_id: str) -> Optional[Dict[str, object]]:
        """Per-session totals, or None if the session is unknown."""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return None
            return {
                "session_id":        session_id,
                "requests":          s.requests,
                "errors":            s.errors,
                "prompt_tokens":     s.prompt_tokens,
                "completion_tokens": s.completion_tokens,
                "total_tokens":      s.total_tokens,
                "cost_usd":          round(s.cost_usd, 6),
            }

    def reset(self) -> None:
        """Test helper — wipe all counters."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.cost_usd = 0.0
            self.requests = 0
            self.errors = 0
            self.started_at = time.time()
            self._sessions.clear()
            self._latencies.clear()
            self._model_calls.clear()


# ---------------------------------------------------------------------------
# Latency percentile helper
# ---------------------------------------------------------------------------

def _latency_summary(samples: List[int]) -> Dict[str, int]:
    """Return P50/P95/P99/avg/max from a list of latencies. Zeroes when empty."""
    if not samples:
        return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "avg": 0, "max": 0}
    ordered = sorted(samples)
    n = len(ordered)

    def _pct(p: float) -> int:
        # Nearest-rank percentile — simple, well-defined, no scipy dep.
        idx = min(n - 1, max(0, int(round(p / 100 * n)) - 1))
        return ordered[idx]

    return {
        "count": n,
        "p50":   _pct(50),
        "p95":   _pct(95),
        "p99":   _pct(99),
        "avg":   sum(ordered) // n,
        "max":   ordered[-1],
    }


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_tracker_lock = threading.Lock()
_tracker: Optional[CostTracker] = None


def get_tracker() -> CostTracker:
    """Lazy thread-safe singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = CostTracker()
    return _tracker
