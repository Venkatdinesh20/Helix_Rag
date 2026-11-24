"""
Conversation Store — in-memory session-scoped chat history.

Design rationale
----------------
A production RAG system needs multi-turn awareness so the LLM can resolve
references like "what about its cost?" or "tell me more". To keep this
deployable on a single Cloud Run instance with $0 infrastructure, we use
an in-memory TTL-bounded store rather than Redis.

Properties:
  - Thread-safe (asyncio + sync compatible: a simple Lock is enough at
    Cloud Run's per-instance scale).
  - Bounded memory: each session keeps at most MAX_TURNS pairs, the whole
    store evicts sessions inactive for more than SESSION_TTL_SECONDS.
  - Stateless API: a stale or unknown session_id silently becomes a new
    session — the client never has to reset anything.

Trade-offs (documented for honesty):
  - Memory is lost on container restart. For a portfolio/demo this is
    acceptable; production deployments would back this with Redis.
  - Cloud Run instances are independent — sticky sessions are not
    guaranteed. The client passes the same session_id back, which is
    enough for single-instance Cloud Run free tier deployments.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List

# Tunable limits — exposed at module level so tests can monkey-patch
MAX_TURNS_PER_SESSION = 8          # last 8 user+assistant pairs are kept
SESSION_TTL_SECONDS = 60 * 60      # 1 hour of inactivity → session evicted
MAX_TOTAL_SESSIONS = 1_000         # hard cap to prevent unbounded growth


@dataclass
class Turn:
    """A single user/assistant exchange."""
    role: str          # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_accessed = time.time()

    def append(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        # Trim to the last N turn-pairs (= 2*N entries)
        max_entries = MAX_TURNS_PER_SESSION * 2
        if len(self.turns) > max_entries:
            del self.turns[: len(self.turns) - max_entries]
        self.touch()

    def history_for_llm(self) -> List[Dict[str, str]]:
        """Serialize turns into the OpenAI chat-completions message shape."""
        return [{"role": t.role, "content": t.content} for t in self.turns]


class ConversationStore:
    """Thread-safe LRU-ish session store with TTL eviction."""

    def __init__(self) -> None:
        # OrderedDict gives us O(1) move-to-end so we can implement LRU cheaply
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def get_or_create(self, session_id: str | None) -> Session:
        """
        Return the session for `session_id`, creating one if it does not
        exist (or has been evicted). A blank or None id always allocates
        a fresh session.
        """
        with self._lock:
            self._evict_expired_locked()

            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                # Move to end of LRU
                self._sessions.move_to_end(session_id)
                return session

            # Create new session
            new_id = session_id or self._new_id()
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            self._enforce_capacity_locked()
            return session

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a turn. Creates session if missing — never raises."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(session_id=session_id)
                self._sessions[session_id] = session
                self._enforce_capacity_locked()
            session.append(role, content)
            self._sessions.move_to_end(session_id)

    def reset(self, session_id: str) -> bool:
        """Clear history for a session. Returns True if it existed."""
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            return existed

    def stats(self) -> Dict[str, int]:
        """Lightweight introspection for the dashboard."""
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "total_turns": sum(len(s.turns) for s in self._sessions.values()),
            }

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex

    def _evict_expired_locked(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        expired = [sid for sid, s in self._sessions.items() if s.last_accessed < cutoff]
        for sid in expired:
            self._sessions.pop(sid, None)

    def _enforce_capacity_locked(self) -> None:
        while len(self._sessions) > MAX_TOTAL_SESSIONS:
            # popitem(last=False) evicts the least recently used
            self._sessions.popitem(last=False)


# Module-level singleton — the entire process shares one store
_store = ConversationStore()


def get_store() -> ConversationStore:
    """Public accessor — keeps the singleton swappable in tests."""
    return _store
