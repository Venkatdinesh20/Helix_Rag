"""
Tests for the /dashboard live metrics page (Phase 3.1).

The dashboard is a static HTML page that polls /stats every few seconds.
We don't run a browser here — these tests verify:
  - the route exists and serves HTML
  - the page references the /stats endpoint
  - it contains the expected metric placeholders
  - /stats stays JSON and contains all keys the dashboard reads
"""
from __future__ import annotations

import re

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_dashboard_route_returns_html():
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    # FastAPI's FileResponse sets text/html for .html files.
    assert "text/html" in resp.headers.get("content-type", "")
    body = resp.text
    assert "<html" in body.lower()
    assert "</html>" in body.lower()


def test_dashboard_polls_stats_endpoint():
    """The page must actually fetch /stats — guards against URL drift."""
    body = client.get("/dashboard").text
    assert 'fetch("/stats"' in body or "fetch('/stats'" in body


def test_dashboard_contains_all_kpi_placeholders():
    """If we rename a KPI id in the HTML without updating the JS it breaks
    silently. This catches that during refactors."""
    body = client.get("/dashboard").text
    required_ids = [
        "kpi-uptime", "kpi-requests", "kpi-errors",
        "kpi-tokens",  "kpi-cost",     "kpi-sessions",
        "latency-bars", "model-bars",  "tokens-bars",
    ]
    for el_id in required_ids:
        assert f'id="{el_id}"' in body, f"missing dashboard element: {el_id}"


def test_stats_payload_has_keys_the_dashboard_reads():
    """The dashboard renders the following fields; if /stats stops emitting
    them the UI silently shows '—'. This test is the contract between the
    two halves of the feature."""
    # Make at least one call so we have non-zero data
    client.post("/ask", json={"question": "hi"})

    body = client.get("/stats").json()
    g = body["global"]

    for key in (
        "uptime_seconds", "requests", "errors", "error_rate",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cost_usd", "model_calls", "sessions", "latency_ms",
    ):
        assert key in g, f"/stats missing key required by dashboard: {key}"

    for key in ("count", "p50", "p95", "p99", "avg", "max"):
        assert key in g["latency_ms"], f"latency_ms missing {key}"


def test_dashboard_link_present_on_main_page():
    """Discoverability — main page header must link to /dashboard."""
    body = client.get("/").text
    # tolerate either single or double quoted href
    assert re.search(r'href=["\']/dashboard["\']', body), "no /dashboard link in index"


def test_dashboard_does_not_leak_secrets():
    """Sanity: the static HTML must never reference real API keys / models
    via env injection. We simply check for sk- style tokens."""
    body = client.get("/dashboard").text
    assert "sk-" not in body
    assert "OPENAI_API_KEY" not in body
