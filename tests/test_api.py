"""
API integration tests for app/main.py

Why test the API?
  The API is the user-facing contract. These tests verify that:
  - HTTP status codes are correct
  - Response schemas match what was promised
  - Input validation works (rejects bad requests)
  - The /health endpoint always works

These use FastAPI's TestClient (built on httpx) — no real server needed.

Run with:
  python -m pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


class TestRootEndpoint:

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_serves_html_ui(self):
        """Root path now serves the static single-page UI (HTML)."""
        response = client.get("/")
        assert response.headers["content-type"].startswith("text/html")
        assert "<html" in response.text.lower()


class TestAskEndpoint:

    def test_ask_returns_200_for_valid_question(self):
        response = client.post("/ask", json={"question": "What is the refund policy?"})
        assert response.status_code == 200

    def test_ask_response_has_required_fields(self):
        response = client.post("/ask", json={"question": "How can I contact support?"})
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "retrieval_score" in data
        assert "latency_ms" in data
        assert "model" in data
        assert "total_tokens" in data

    def test_ask_returns_string_answer(self):
        response = client.post("/ask", json={"question": "What is the refund policy?"})
        data = response.json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_ask_returns_sources_list(self):
        response = client.post("/ask", json={"question": "refund policy"})
        data = response.json()
        assert isinstance(data["sources"], list)

    def test_ask_source_has_required_fields(self):
        response = client.post("/ask", json={"question": "What is the refund policy?"})
        data = response.json()
        if data["sources"]:
            source = data["sources"][0]
            assert "chunk_id" in source
            assert "score" in source

    def test_ask_latency_ms_is_positive(self):
        response = client.post("/ask", json={"question": "refund"})
        data = response.json()
        assert data["latency_ms"] >= 0

    def test_ask_rejects_empty_question(self):
        """Empty question should return HTTP 422 (validation error)."""
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_ask_rejects_missing_question_field(self):
        """Missing question field should return HTTP 422."""
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_ask_rejects_question_over_1000_chars(self):
        """Questions over 1000 chars should be rejected with HTTP 422."""
        long_question = "x" * 1001
        response = client.post("/ask", json={"question": long_question})
        assert response.status_code == 422

    def test_ask_accepts_question_at_max_length(self):
        """A 1000-character question should be accepted."""
        max_question = "what is the policy " * 52  # ~1000 chars
        response = client.post("/ask", json={"question": max_question[:1000]})
        assert response.status_code == 200


class TestChatMemory:
    """Phase 1.1 — session-scoped conversation memory."""

    def test_ask_returns_session_id(self):
        response = client.post("/ask", json={"question": "hi"})
        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0

    def test_ask_echoes_provided_session_id(self):
        sid = "test-session-abc-123"
        response = client.post("/ask", json={"question": "hi", "session_id": sid})
        assert response.json()["session_id"] == sid

    def test_session_persists_across_calls(self):
        first = client.post("/ask", json={"question": "hi"}).json()
        sid = first["session_id"]
        second = client.post(
            "/ask", json={"question": "thanks", "session_id": sid}
        ).json()
        assert second["session_id"] == sid

    def test_session_reset_endpoint(self):
        # Create session
        first = client.post("/ask", json={"question": "hi"}).json()
        sid = first["session_id"]
        # Reset it
        resp = client.post("/session/reset", json={"session_id": sid})
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert body["cleared"] is True

    def test_session_reset_unknown_id_returns_false(self):
        resp = client.post(
            "/session/reset", json={"session_id": "never-existed-xyz"}
        )
        assert resp.status_code == 200
        assert resp.json()["cleared"] is False
