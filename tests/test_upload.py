"""
Tests for the document upload endpoint and ingestion service.

These tests run end-to-end through FastAPI's TestClient and the real
ingestion pipeline. To keep them fast and isolated:

  - Each test uses tmp directories for the FAISS index and raw upload folder
    (monkeypatched into both pipeline.build_index and app.ingestion_service)
  - A tiny stub embedder replaces sentence-transformers so we don't pay model
    load time. We only assert structural behaviour (chunks_added, dim mismatch,
    cache reload, validation), not embedding quality.
"""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Test fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_vector_store(tmp_path, monkeypatch):
    """
    Redirect the FAISS index files and the raw uploads folder into a tmp dir
    for every test. This guarantees tests do not mutate the real index.
    """
    vec_dir = tmp_path / "vector_store"
    vec_dir.mkdir()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    index_file = str(vec_dir / "faiss_index.index")
    metadata_file = str(vec_dir / "chunks_metadata.json")

    from pipeline import build_index as build_mod
    from app import ingestion_service as ingest_mod

    monkeypatch.setattr(build_mod, "VECTOR_STORE_DIR", str(vec_dir))
    monkeypatch.setattr(build_mod, "INDEX_FILE", index_file)
    monkeypatch.setattr(build_mod, "METADATA_FILE", metadata_file)
    monkeypatch.setattr(ingest_mod, "VECTOR_STORE_DIR", str(vec_dir))
    monkeypatch.setattr(ingest_mod, "INDEX_FILE", index_file)
    monkeypatch.setattr(ingest_mod, "METADATA_FILE", metadata_file)
    monkeypatch.setattr(ingest_mod, "RAW_DIR", raw_dir)

    yield {
        "index_file": index_file,
        "metadata_file": metadata_file,
        "raw_dir": raw_dir,
    }


@pytest.fixture(autouse=True)
def stub_embedder(monkeypatch):
    """
    Replace embed_chunks with a deterministic 8-dim stub so the test suite
    does not load the 80MB sentence-transformer model. The dimension is small
    but consistent — that's all the index plumbing needs.
    """
    from app import ingestion_service as ingest_mod

    def fake_embed(chunks):
        out = []
        for c in chunks:
            # Hash text → deterministic 8-d vector, L2-normalised
            h = abs(hash(c["text"])) % (2 ** 31)
            rng = np.random.default_rng(h)
            v = rng.standard_normal(8).astype("float32")
            v /= np.linalg.norm(v) + 1e-9
            out.append({**c, "embedding": v})
        return out

    monkeypatch.setattr(ingest_mod, "embed_chunks", fake_embed)
    yield


@pytest.fixture
def client():
    # Import after monkeypatches so the app picks up the test config
    from app.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_rate_limit_buckets():
    """Reset the per-IP upload rate limiter between tests."""
    from app import api as api_mod
    api_mod._upload_buckets.clear()
    yield
    api_mod._upload_buckets.clear()


# ── Sample payloads ──────────────────────────────────────────────────────────


def _txt_bytes(text: str = "This is a small uploaded document about refunds. " * 20) -> bytes:
    return text.encode("utf-8")


# A minimal one-page PDF generated with reportlab would be heavy. Instead we
# craft a tiny syntactically valid PDF containing a single text object. pypdf
# can parse this. If pypdf cannot extract text from such a tiny file the test
# falls back to txt.
_MINIMAL_PDF = (
    b"%PDF-1.1\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/"
    b"Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 44>>stream\n"
    b"BT /F1 12 Tf 50 80 Td (hello refunds policy) Tj ET\n"
    b"endstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 6\n0000000000 65535 f \n"
    b"trailer<</Size 6/Root 1 0 R>>\n"
    b"startxref\n0\n%%EOF\n"
)


# ── Validation ───────────────────────────────────────────────────────────────


class TestUploadValidation:

    def test_rejects_missing_file(self, client):
        resp = client.post("/upload")
        assert resp.status_code == 422  # FastAPI's missing-field validation

    def test_rejects_unsupported_extension(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("evil.exe", b"MZ\x90\x00binary", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()

    def test_rejects_empty_file(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_rejects_non_utf8_text(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("bad.txt", b"\xff\xfe\xfd", "text/plain")},
        )
        assert resp.status_code == 400
        assert "utf-8" in resp.json()["detail"].lower()

    def test_rejects_fake_pdf_without_magic(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("fake.pdf", b"not really a pdf", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "pdf" in resp.json()["detail"].lower()

    def test_rejects_oversize_via_streaming_cap(self, client, monkeypatch):
        # Force the streaming cap to a tiny value so we don't have to allocate
        # 10 MB in the test process.
        from app import api as api_mod
        monkeypatch.setattr(api_mod, "MAX_UPLOAD_BYTES", 128)
        payload = b"A" * 500
        resp = client.post(
            "/upload",
            files={"file": ("big.txt", payload, "text/plain")},
        )
        assert resp.status_code == 413


# ── Successful ingestion ─────────────────────────────────────────────────────


class TestUploadSuccess:

    def test_txt_upload_indexes_and_returns_counts(self, client, isolated_vector_store):
        resp = client.post(
            "/upload",
            files={"file": ("policy.txt", _txt_bytes(), "text/plain")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["document_name"].endswith(".txt")
        assert body["original_filename"] == "policy.txt"
        assert body["chunks_added"] >= 1
        assert body["total_chunks"] == body["chunks_added"]

        # The file was actually written under the test raw dir
        saved = isolated_vector_store["raw_dir"] / body["document_name"]
        assert saved.exists()
        assert saved.read_bytes() == _txt_bytes()

        # Metadata file exists and matches total_chunks
        with open(isolated_vector_store["metadata_file"], encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) == body["total_chunks"]

    def test_second_upload_appends_to_existing_index(self, client):
        first = client.post(
            "/upload",
            files={"file": ("a.txt", _txt_bytes("alpha document about refunds. " * 20), "text/plain")},
        ).json()
        second = client.post(
            "/upload",
            files={"file": ("b.txt", _txt_bytes("beta document about support. " * 20), "text/plain")},
        ).json()
        assert second["total_chunks"] > first["total_chunks"]
        assert second["chunks_added"] >= 1

    def test_filename_path_traversal_is_sanitised(self, client, isolated_vector_store):
        resp = client.post(
            "/upload",
            files={"file": ("../../etc/passwd.txt", _txt_bytes(), "text/plain")},
        )
        assert resp.status_code == 200
        body = resp.json()
        # No path components survived
        assert "/" not in body["document_name"]
        assert "\\" not in body["document_name"]
        assert ".." not in body["document_name"]
        # File landed inside the test raw dir, not somewhere outside
        saved = isolated_vector_store["raw_dir"] / body["document_name"]
        assert saved.exists()

    def test_duplicate_filename_does_not_overwrite(self, client, isolated_vector_store):
        first = client.post(
            "/upload",
            files={"file": ("dup.txt", _txt_bytes("first content " * 30), "text/plain")},
        ).json()
        second = client.post(
            "/upload",
            files={"file": ("dup.txt", _txt_bytes("second content " * 30), "text/plain")},
        ).json()
        assert first["document_name"] != second["document_name"]
        assert (isolated_vector_store["raw_dir"] / first["document_name"]).exists()
        assert (isolated_vector_store["raw_dir"] / second["document_name"]).exists()


# ── Rate limiting ────────────────────────────────────────────────────────────


class TestUploadRateLimit:

    def test_too_many_uploads_returns_429(self, client, monkeypatch):
        from app import api as api_mod
        monkeypatch.setattr(api_mod, "UPLOAD_RATE_MAX_REQUESTS", 2)

        for _ in range(2):
            r = client.post(
                "/upload",
                files={"file": ("rl.txt", _txt_bytes(), "text/plain")},
            )
            assert r.status_code == 200, r.text

        r = client.post(
            "/upload",
            files={"file": ("rl.txt", _txt_bytes(), "text/plain")},
        )
        assert r.status_code == 429
        assert "Retry-After" in r.headers


# ── Direct service tests (no HTTP) ───────────────────────────────────────────


class TestIngestionServiceUnit:

    def test_rolls_back_file_on_pipeline_failure(self, isolated_vector_store, monkeypatch):
        """If embedding blows up, the saved file must be removed."""
        from app import ingestion_service as ingest_mod

        def boom(_chunks):
            raise RuntimeError("simulated embed failure")

        monkeypatch.setattr(ingest_mod, "embed_chunks", boom)
        with pytest.raises(RuntimeError):
            ingest_mod.ingest_upload(_txt_bytes(), "willfail.txt")

        # No file should be left behind in the raw dir
        leftovers = list(isolated_vector_store["raw_dir"].iterdir())
        assert leftovers == []

    def test_dim_mismatch_raises(self, isolated_vector_store, monkeypatch):
        """An existing index built with dim=8 must reject 16-d uploads."""
        from app import ingestion_service as ingest_mod

        # First upload (dim 8, from default fixture)
        ingest_mod.ingest_upload(_txt_bytes("first " * 30), "first.txt")

        # Now patch the embedder to produce dim 16 and try again
        def embed_16(chunks):
            out = []
            for c in chunks:
                v = np.ones(16, dtype="float32")
                v /= np.linalg.norm(v)
                out.append({**c, "embedding": v})
            return out

        monkeypatch.setattr(ingest_mod, "embed_chunks", embed_16)
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            ingest_mod.ingest_upload(_txt_bytes("second " * 30), "second.txt")
