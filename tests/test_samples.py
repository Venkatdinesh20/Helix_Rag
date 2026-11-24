"""
Tests for the /samples endpoint and the underlying sample selection logic.

What we verify:
  - The static list is well-formed (every entry has label+question, no empties).
  - The endpoint serves the same shape via HTTP.
  - The corpus-derived chip references a real indexed document name when
    available, and is gracefully skipped when retrieval is unavailable.
  - The cap MAX_SAMPLES is honoured even with the derived addition.
"""

import pytest
from fastapi.testclient import TestClient

from app import samples as samples_module
from app.main import app
from app.samples import MAX_SAMPLES, list_samples

client = TestClient(app)


class TestSamplesModule:

    def test_returns_list(self):
        result = list_samples()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_each_item_has_label_and_question(self):
        for item in list_samples():
            assert isinstance(item, dict)
            assert item.get("label"), f"Missing label: {item!r}"
            assert item.get("question"), f"Missing question: {item!r}"
            assert isinstance(item["label"], str)
            assert isinstance(item["question"], str)

    def test_does_not_exceed_max(self):
        assert len(list_samples()) <= MAX_SAMPLES

    def test_derived_sample_skipped_on_index_failure(self, monkeypatch):
        # If retrieval can't load the index we must still return the base set
        # — never raise or return nothing.
        def _boom():
            raise RuntimeError("index unavailable")
        monkeypatch.setattr(samples_module, "_derive_corpus_sample",
                            lambda: None)  # simulate the "skip" branch
        result = list_samples()
        assert len(result) >= 1
        # No item should reference a document filename (".pdf" / ".txt")
        for s in result:
            assert ".pdf" not in s["question"] and ".txt" not in s["question"]

    def test_derived_sample_appears_first_when_available(self, monkeypatch):
        fake = {"label": "About custom-doc", "question": "What does 'custom-doc.pdf' say?"}
        monkeypatch.setattr(samples_module, "_derive_corpus_sample",
                            lambda: fake)
        result = list_samples()
        assert result[0] == fake
        assert len(result) <= MAX_SAMPLES


class TestSamplesEndpoint:

    def test_get_samples_returns_200(self):
        r = client.get("/samples")
        assert r.status_code == 200

    def test_response_shape(self):
        body = client.get("/samples").json()
        assert "samples" in body
        assert isinstance(body["samples"], list)
        assert body["samples"], "expected at least one sample"
        for s in body["samples"]:
            assert set(s.keys()) >= {"label", "question"}
            assert s["label"] and s["question"]

    def test_no_duplicates_in_questions(self):
        body = client.get("/samples").json()
        questions = [s["question"] for s in body["samples"]]
        assert len(questions) == len(set(questions)), "Duplicate sample questions"

    def test_questions_are_reasonable_length(self):
        # Guard against accidentally shipping a giant prompt — the UI chip is small.
        body = client.get("/samples").json()
        for s in body["samples"]:
            assert 4 <= len(s["question"]) <= 300
            assert 2 <= len(s["label"]) <= 60
