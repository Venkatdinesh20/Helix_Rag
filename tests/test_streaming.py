"""
Streaming (Server-Sent Events) tests for /ask/stream.

What we verify:
  - The endpoint advertises the SSE content-type.
  - Events arrive in the expected order: meta → token(s) → done.
  - The wire format follows the SSE spec: "event: <t>\\ndata: <json>\\n\\n".
  - Conversation turns are persisted after a clean stream.
  - Empty-retrieval and small-talk paths still produce a valid event sequence.

We do not call the real OpenAI API. The generation layer is bypassed by
monkeypatching `stream_answer` to yield a deterministic fake stream.
"""

import json
import re

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import rag_service
from app.conversation_store import get_store

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(body: str):
    """Parse raw SSE bytes into a list of {type, payload} dicts in order."""
    events = []
    # Split into frames separated by a blank line
    for frame in re.split(r"\n\n", body):
        frame = frame.strip("\r\n")
        if not frame:
            continue
        ev_type = "message"
        data_str = ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                ev_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str += line[len("data:"):].strip()
        if not data_str:
            continue
        events.append({"event": ev_type, "data": json.loads(data_str)})
    return events


def _fake_stream(*_args, **_kwargs):
    """Yield a deterministic stream that mimics generation.stream_answer."""
    yield ("token", "Hello ")
    yield ("token", "world")
    yield ("done", {
        "answer":            "Hello world",
        "model":             "fake-model",
        "prompt_tokens":     12,
        "completion_tokens": 3,
        "total_tokens":      15,
    })


def _fake_retrieve_one_chunk(*_args, **_kwargs):
    return [{
        "chunk_id": "doc1_p1_c0",
        "text":     "Some context.",
        "score":    0.9,
        "metadata": {"document_name": "doc1.pdf", "page": 1},
    }]


def _fake_retrieve_empty(*_args, **_kwargs):
    return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamHappyPath:

    def test_content_type_is_event_stream(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_one_chunk)
        monkeypatch.setattr(rag_service, "stream_answer", _fake_stream)
        with client.stream("POST", "/ask/stream", json={"question": "hi there about docs"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())
        # Sanity: at least some frames
        assert "event:" in body
        assert "data:" in body

    def test_event_sequence_meta_tokens_done(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_one_chunk)
        monkeypatch.setattr(rag_service, "stream_answer", _fake_stream)
        with client.stream("POST", "/ask/stream", json={"question": "what is in the document?"}) as r:
            body = "".join(r.iter_text())

        events = _parse_sse(body)
        types = [e["event"] for e in events]
        # Must start with meta, end with done, with at least one token in between
        assert types[0] == "meta"
        assert types[-1] == "done"
        assert "token" in types
        token_texts = [e["data"]["text"] for e in events if e["event"] == "token"]
        assert "".join(token_texts) == "Hello world"

    def test_meta_contains_sources_and_session(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_one_chunk)
        monkeypatch.setattr(rag_service, "stream_answer", _fake_stream)
        with client.stream("POST", "/ask/stream", json={"question": "tell me about the doc"}) as r:
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        meta = events[0]["data"]
        assert meta["type"] == "meta"
        assert isinstance(meta["session_id"], str) and meta["session_id"]
        assert isinstance(meta["sources"], list) and len(meta["sources"]) == 1
        assert meta["sources"][0]["document_name"] == "doc1.pdf"
        assert meta["retrieval_score"] == pytest.approx(0.9, abs=1e-4)

    def test_done_contains_usage_and_latency(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_one_chunk)
        monkeypatch.setattr(rag_service, "stream_answer", _fake_stream)
        with client.stream("POST", "/ask/stream", json={"question": "what does the doc say?"}) as r:
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        done = events[-1]["data"]
        assert done["type"] == "done"
        assert done["model"] == "fake-model"
        assert done["total_tokens"] == 15
        assert done["prompt_tokens"] == 12
        assert done["completion_tokens"] == 3
        assert isinstance(done["latency_ms"], int)
        assert done["latency_ms"] >= 0
        assert done["error"] is None


class TestStreamPersistsHistory:

    def test_session_turns_are_appended_after_done(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_one_chunk)
        monkeypatch.setattr(rag_service, "stream_answer", _fake_stream)
        with client.stream("POST", "/ask/stream", json={"question": "first streamed question about docs"}) as r:
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        session_id = events[0]["data"]["session_id"]

        session = get_store().get_or_create(session_id)
        roles = [t.role for t in session.turns]
        contents = [t.content for t in session.turns]
        assert roles[-2:] == ["user", "assistant"]
        assert contents[-1] == "Hello world"


class TestStreamEdgeCases:

    def test_empty_retrieval_returns_fallback_token(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_empty)
        # stream_answer must not be called when there are no chunks
        def _boom(*a, **kw):
            raise AssertionError("stream_answer should not be called on empty retrieval")
        monkeypatch.setattr(rag_service, "stream_answer", _boom)

        with client.stream("POST", "/ask/stream", json={"question": "obscure unrelated question xyz"}) as r:
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        token_texts = [e["data"]["text"] for e in events if e["event"] == "token"]
        joined = "".join(token_texts)
        assert "could not find" in joined.lower()
        assert events[-1]["event"] == "done"

    def test_small_talk_short_circuits_without_llm(self, monkeypatch):
        # Greetings must not touch retrieval or generation
        def _boom_retrieve(*a, **kw):
            raise AssertionError("retrieve should not be called on greeting")
        def _boom_stream(*a, **kw):
            raise AssertionError("stream_answer should not be called on greeting")
        monkeypatch.setattr(rag_service, "retrieve", _boom_retrieve)
        monkeypatch.setattr(rag_service, "stream_answer", _boom_stream)

        with client.stream("POST", "/ask/stream", json={"question": "hi"}) as r:
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        types = [e["event"] for e in events]
        assert types[0] == "meta"
        assert types[-1] == "done"
        token_texts = [e["data"]["text"] for e in events if e["event"] == "token"]
        assert any("Hello" in t or "👋" in t for t in token_texts)
        assert events[-1]["data"]["model"] == "conversational"

    def test_rejects_blank_question(self):
        r = client.post("/ask/stream", json={"question": ""})
        assert r.status_code == 422   # pydantic validation

    def test_sse_frame_format_is_well_formed(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_one_chunk)
        monkeypatch.setattr(rag_service, "stream_answer", _fake_stream)
        with client.stream("POST", "/ask/stream", json={"question": "frame format check"}) as r:
            body = "".join(r.iter_text())
        # Every frame must contain both an event: line and a data: line
        for frame in body.split("\n\n"):
            frame = frame.strip()
            if not frame:
                continue
            assert frame.startswith("event:"), f"Bad frame head: {frame!r}"
            assert "\ndata:" in frame, f"Frame missing data: {frame!r}"


class TestStreamErrorPath:

    def test_generation_error_emits_error_event(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_one_chunk)

        def _erroring_stream(*a, **kw):
            yield ("token", "partial ")
            yield ("error", "LLM streaming failed: simulated")

        monkeypatch.setattr(rag_service, "stream_answer", _erroring_stream)

        with client.stream("POST", "/ask/stream", json={"question": "force a generation error"}) as r:
            assert r.status_code == 200  # SSE always starts 200
            body = "".join(r.iter_text())

        events = _parse_sse(body)
        types = [e["event"] for e in events]
        assert "error" in types
        # Must NOT have a "done" after an error
        assert types[-1] == "error"
        # And no partial assistant turn must be persisted
        meta = events[0]["data"]
        session = get_store().get_or_create(meta["session_id"])
        # The last role should not be "assistant" with "partial " (we never appended)
        if session.turns:
            assert not (session.turns[-1].role == "assistant"
                        and session.turns[-1].content == "partial ")
