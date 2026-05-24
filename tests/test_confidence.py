"""
Tests for the confidence bucketing.

Why a dedicated test file?
  The confidence rubric is a contract surfaced in the API (and the UI badge).
  Changing the thresholds is a *user-visible* behaviour change, so we pin
  both the boundary semantics and the field plumbing through /ask and
  /ask/stream.
"""

import json
import re

import pytest
from fastapi.testclient import TestClient

from app import rag_service
from app.main import app
from app.rag_service import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    confidence_from_score,
)

client = TestClient(app)


class TestConfidenceFromScore:

    def test_high_at_boundary(self):
        assert confidence_from_score(CONFIDENCE_HIGH_THRESHOLD) == "high"

    def test_high_above(self):
        assert confidence_from_score(0.9) == "high"

    def test_medium_at_boundary(self):
        assert confidence_from_score(CONFIDENCE_MEDIUM_THRESHOLD) == "medium"

    def test_medium_just_below_high(self):
        assert confidence_from_score(CONFIDENCE_HIGH_THRESHOLD - 1e-6) == "medium"

    def test_low_just_below_medium(self):
        assert confidence_from_score(CONFIDENCE_MEDIUM_THRESHOLD - 1e-6) == "low"

    def test_zero_is_low(self):
        assert confidence_from_score(0.0) == "low"

    def test_negative_is_low(self):
        assert confidence_from_score(-0.5) == "low"

    def test_none_is_low(self):
        assert confidence_from_score(None) == "low"


# ---------------------------------------------------------------------------
# Wire-level: /ask returns the confidence field
# ---------------------------------------------------------------------------

def _fake_retrieve_score(score: float):
    def _r(*_a, **_k):
        return [{
            "chunk_id": "doc1_p1_c0",
            "text":     "ctx",
            "score":    score,
            "metadata": {"document_name": "doc1.pdf", "page": 1},
        }]
    return _r


def _fake_generate(*_a, **_k):
    return {
        "answer":            "fake answer",
        "model":             "fake-model",
        "prompt_tokens":     1,
        "completion_tokens": 1,
        "total_tokens":      2,
        "error":             None,
    }


class TestAskExposesConfidence:

    def test_high_score_returns_high(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_score(0.9))
        monkeypatch.setattr(rag_service, "generate_answer", _fake_generate)
        r = client.post("/ask", json={"question": "anything about the doc"})
        assert r.status_code == 200
        assert r.json()["confidence"] == "high"

    def test_medium_score_returns_medium(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_score(0.6))
        monkeypatch.setattr(rag_service, "generate_answer", _fake_generate)
        r = client.post("/ask", json={"question": "another query"})
        assert r.json()["confidence"] == "medium"

    def test_low_score_returns_low(self, monkeypatch):
        # Above MIN_RETRIEVAL_SCORE (0.2) but below the medium threshold
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_score(0.3))
        monkeypatch.setattr(rag_service, "generate_answer", _fake_generate)
        r = client.post("/ask", json={"question": "borderline query"})
        assert r.json()["confidence"] == "low"

    def test_empty_retrieval_is_low(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", lambda *a, **k: [])
        r = client.post("/ask", json={"question": "totally unrelated question"})
        body = r.json()
        assert body["confidence"] == "low"
        assert body["retrieval_score"] == 0.0


# ---------------------------------------------------------------------------
# Wire-level: /ask/stream meta event carries confidence
# ---------------------------------------------------------------------------

def _parse_sse(body: str):
    events = []
    for frame in re.split(r"\n\n", body):
        frame = frame.strip("\r\n")
        if not frame:
            continue
        etype = "message"
        data = ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                etype = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data += line[len("data:"):].strip()
        if data:
            events.append({"event": etype, "data": json.loads(data)})
    return events


def _fake_stream(*_a, **_k):
    yield ("token", "Hello")
    yield ("done", {
        "answer":            "Hello",
        "model":             "fake-model",
        "prompt_tokens":     1,
        "completion_tokens": 1,
        "total_tokens":      2,
    })


class TestStreamExposesConfidence:

    def test_meta_event_includes_confidence_high(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_score(0.85))
        monkeypatch.setattr(rag_service, "stream_answer", _fake_stream)
        with client.stream("POST", "/ask/stream", json={"question": "strong match query"}) as r:
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        assert events[0]["event"] == "meta"
        assert events[0]["data"]["confidence"] == "high"

    def test_meta_event_includes_confidence_medium(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", _fake_retrieve_score(0.55))
        monkeypatch.setattr(rag_service, "stream_answer", _fake_stream)
        with client.stream("POST", "/ask/stream", json={"question": "medium match query"}) as r:
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        assert events[0]["data"]["confidence"] == "medium"

    def test_empty_retrieval_meta_is_low(self, monkeypatch):
        monkeypatch.setattr(rag_service, "retrieve", lambda *a, **k: [])
        with client.stream("POST", "/ask/stream", json={"question": "no-match query xyz"}) as r:
            body = "".join(r.iter_text())
        events = _parse_sse(body)
        assert events[0]["data"]["confidence"] == "low"
