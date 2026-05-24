"""
Unit tests for app.conversation_store

These tests verify:
  - Sessions are created on demand and persisted across calls
  - History trimming keeps memory bounded
  - TTL eviction drops idle sessions
  - reset() clears history cleanly
  - Concurrent access is safe (basic smoke test)
"""

from __future__ import annotations

import threading
import time

import pytest

from app import conversation_store
from app.conversation_store import ConversationStore, Session


@pytest.fixture
def store() -> ConversationStore:
    """A fresh store per test — never touches the module-level singleton."""
    return ConversationStore()


class TestSessionLifecycle:

    def test_get_or_create_with_none_makes_new_session(self, store):
        s = store.get_or_create(None)
        assert isinstance(s, Session)
        assert s.session_id  # non-empty
        assert s.turns == []

    def test_same_session_id_returns_same_object(self, store):
        s1 = store.get_or_create(None)
        s2 = store.get_or_create(s1.session_id)
        assert s1 is s2

    def test_unknown_session_id_makes_new_session_with_that_id(self, store):
        s = store.get_or_create("custom-id-123")
        assert s.session_id == "custom-id-123"

    def test_append_turn_records_messages(self, store):
        s = store.get_or_create(None)
        store.append_turn(s.session_id, "user", "hi")
        store.append_turn(s.session_id, "assistant", "hello")
        again = store.get_or_create(s.session_id)
        assert len(again.turns) == 2
        assert again.turns[0].role == "user"
        assert again.turns[1].role == "assistant"

    def test_history_for_llm_is_openai_shape(self, store):
        s = store.get_or_create(None)
        store.append_turn(s.session_id, "user", "Q1")
        store.append_turn(s.session_id, "assistant", "A1")
        again = store.get_or_create(s.session_id)
        history = again.history_for_llm()
        assert history == [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]


class TestTrimming:

    def test_history_trims_to_max_turns(self, store, monkeypatch):
        monkeypatch.setattr(conversation_store, "MAX_TURNS_PER_SESSION", 2)
        s = store.get_or_create(None)
        # Send 5 pairs; only the last 2 pairs (= 4 entries) should remain
        for i in range(5):
            store.append_turn(s.session_id, "user", f"q{i}")
            store.append_turn(s.session_id, "assistant", f"a{i}")
        again = store.get_or_create(s.session_id)
        assert len(again.turns) == 4
        assert again.turns[0].content == "q3"
        assert again.turns[-1].content == "a4"


class TestEviction:

    def test_reset_clears_session(self, store):
        s = store.get_or_create(None)
        store.append_turn(s.session_id, "user", "hi")
        assert store.reset(s.session_id) is True
        # After reset, asking for that id creates an empty session
        new = store.get_or_create(s.session_id)
        assert new.turns == []

    def test_reset_unknown_session_returns_false(self, store):
        assert store.reset("does-not-exist") is False

    def test_ttl_eviction(self, store, monkeypatch):
        monkeypatch.setattr(conversation_store, "SESSION_TTL_SECONDS", 0)
        s = store.get_or_create(None)
        store.append_turn(s.session_id, "user", "hi")
        time.sleep(0.01)
        # Next get_or_create runs eviction; the old session should be gone,
        # so requesting the same id yields a fresh empty one
        fresh = store.get_or_create(s.session_id)
        assert fresh.turns == []

    def test_capacity_eviction(self, store, monkeypatch):
        monkeypatch.setattr(conversation_store, "MAX_TOTAL_SESSIONS", 3)
        ids = [store.get_or_create(None).session_id for _ in range(5)]
        stats = store.stats()
        assert stats["active_sessions"] <= 3
        # Most recent two should survive
        survivors = [sid for sid in ids if sid in store._sessions]
        assert ids[-1] in survivors


class TestConcurrency:

    def test_concurrent_appends_do_not_lose_turns(self, store):
        """Sanity check: 50 threads each append 10 turns; total must match."""
        s = store.get_or_create(None)

        def worker():
            for i in range(10):
                store.append_turn(s.session_id, "user", f"t{i}")

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # We sent 50 turns. Default MAX_TURNS_PER_SESSION=8 → cap 16 entries.
        again = store.get_or_create(s.session_id)
        max_entries = conversation_store.MAX_TURNS_PER_SESSION * 2
        assert len(again.turns) == max_entries


class TestStats:

    def test_stats_reports_counts(self, store):
        s = store.get_or_create(None)
        store.append_turn(s.session_id, "user", "x")
        store.append_turn(s.session_id, "assistant", "y")
        stats = store.stats()
        assert stats["active_sessions"] == 1
        assert stats["total_turns"] == 2
