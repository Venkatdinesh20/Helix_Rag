"""
Document ingestion service
---------------------------
Validates uploaded files and incrementally adds them to the live FAISS index
without restarting the process.

Pipeline per upload:
  1. Validate: extension allow-list, MIME sniff via magic bytes, size limit
  2. Sanitize the filename and save under data/raw/ with a unique stem
  3. Run ingest → clean → chunk → embed on JUST the new file
  4. Append the new vectors to the existing FAISS index on disk
  5. Append metadata records to chunks_metadata.json
  6. Refresh the in-memory retrieval cache so /ask sees the new doc

Concurrency:
  All mutating operations are serialised by a single module-level lock
  (`_index_lock`). Ingestion is I/O-heavy and embedding is CPU-heavy, but
  uploads are rare relative to /ask traffic, so a coarse lock is fine and
  far simpler than per-shard locking.

Safety:
  - Extension allow-list (.pdf, .txt) blocks scripts, archives, executables
  - Magic-byte sniffing ensures the bytes match the claimed type
  - Size cap (default 10 MB) prevents OOM and runaway embedding cost
  - Filenames are sanitised — no path traversal, no shell metacharacters

This module never imports FastAPI types; it can be unit-tested in isolation.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from pipeline.build_index import (
    INDEX_FILE,
    METADATA_FILE,
    VECTOR_STORE_DIR,
)
from pipeline.chunk_documents import chunk_documents
from pipeline.clean_text import clean_documents
from pipeline.generate_embeddings import embed_chunks
from pipeline.ingest_documents import _load_pdf_bytes, _load_txt_bytes

logger = logging.getLogger("rag_ingestion")


# ── Limits & policy ──────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
RAW_DIR = Path("data/raw")

# Magic-byte prefixes for content sniffing. Defence-in-depth: an attacker
# can rename a binary to .txt, but our text path also enforces UTF-8 decode.
_PDF_MAGIC = b"%PDF-"


# ── Errors ───────────────────────────────────────────────────────────────────

class UploadValidationError(ValueError):
    """Raised for any client-correctable validation failure (bad type, oversize)."""


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IngestResult:
    document_name: str          # final on-disk filename (sanitised)
    original_filename: str      # what the client uploaded
    bytes_written: int
    chunks_added: int
    total_chunks: int           # size of the index after this upload


# ── Serialise all mutating work ──────────────────────────────────────────────

_index_lock = threading.Lock()


# ── Public API ───────────────────────────────────────────────────────────────

def ingest_upload(file_bytes: bytes, filename: str) -> IngestResult:
    """
    Validate, persist, and incrementally index a single uploaded document.

    Args:
        file_bytes: raw file content (already buffered in memory — caller is
                    responsible for not buffering files larger than MAX_UPLOAD_BYTES).
        filename:   original filename from the client (untrusted).

    Returns: IngestResult with the index size after the upload.

    Raises:
        UploadValidationError: bad extension, oversize, content/type mismatch,
                               empty document, or zero chunks extracted.
        RuntimeError:          unrecoverable I/O or index error.
    """
    safe_name = _validate_and_sanitize(file_bytes, filename)

    with _index_lock:
        saved_path = _save_unique(file_bytes, safe_name)
        try:
            pages = _extract_pages(file_bytes, saved_path)
            if not pages:
                # Nothing readable inside (e.g. image-only scanned PDF).
                # Remove the file so a retry with the real text version works.
                _unlink_quiet(saved_path)
                raise UploadValidationError(
                    "Could not extract any text from the document. "
                    "Image-only PDFs are not supported."
                )

            chunks = chunk_documents(clean_documents(pages), chunk_size=500, overlap=80)
            if not chunks:
                _unlink_quiet(saved_path)
                raise UploadValidationError("Document contains no usable text after cleaning.")

            embedded = embed_chunks(chunks)
            total = _append_to_index(embedded)

            logger.info(
                f"ingest ok name={saved_path.name} pages={len(pages)} "
                f"chunks_added={len(embedded)} total_chunks={total}"
            )
            return IngestResult(
                document_name=saved_path.name,
                original_filename=filename,
                bytes_written=len(file_bytes),
                chunks_added=len(embedded),
                total_chunks=total,
            )
        except UploadValidationError:
            raise
        except Exception:
            # Any failure after the file is on disk → roll back the disk write
            # so retries are idempotent.
            _unlink_quiet(saved_path)
            logger.exception(f"ingest failed name={saved_path.name}")
            raise


# ── Validation & sanitization ────────────────────────────────────────────────

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _validate_and_sanitize(file_bytes: bytes, filename: str) -> str:
    if not filename:
        raise UploadValidationError("Missing filename.")

    size = len(file_bytes)
    if size == 0:
        raise UploadValidationError("Uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise UploadValidationError(
            f"File too large: {size} bytes (max {MAX_UPLOAD_BYTES})."
        )

    # Take only the base name to defeat path traversal (e.g. "../../etc/passwd").
    base = os.path.basename(filename)
    suffix = Path(base).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}."
        )

    # Magic-byte / content sniffing.
    if suffix == ".pdf":
        if not file_bytes.startswith(_PDF_MAGIC):
            raise UploadValidationError("File does not look like a valid PDF.")
    elif suffix == ".txt":
        try:
            file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise UploadValidationError("Text file is not valid UTF-8.") from e

    # Sanitise filename: keep alnum, dot, dash, underscore only.
    stem = Path(base).stem
    safe_stem = _UNSAFE_CHARS.sub("_", stem).strip("._-") or "upload"
    return f"{safe_stem}{suffix}"


def _save_unique(file_bytes: bytes, safe_name: str) -> Path:
    """Save bytes to data/raw/, appending a short uuid if the name already exists."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / safe_name
    if target.exists():
        stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
        target = RAW_DIR / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    target.write_bytes(file_bytes)
    return target


def _extract_pages(file_bytes: bytes, saved_path: Path) -> List[Dict[str, Any]]:
    """Dispatch to the right byte-level loader based on file extension."""
    suffix = saved_path.suffix.lower()
    doc_name = saved_path.name
    source = str(saved_path)
    if suffix == ".pdf":
        return _load_pdf_bytes(file_bytes, doc_name, source)
    if suffix == ".txt":
        return _load_txt_bytes(file_bytes, doc_name, source)
    # _validate_and_sanitize already rejects other types — this is unreachable.
    raise UploadValidationError(f"Unsupported extension: {suffix}")


# ── Incremental index update ─────────────────────────────────────────────────

def _append_to_index(embedded: List[Dict[str, Any]]) -> int:
    """
    Add new embeddings to the existing FAISS index and metadata files.

    If the index doesn't exist yet (cold start with no prior docs) it is
    created. Returns the total chunk count after the update.

    Also refreshes the in-memory retrieval cache.
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    vectors = np.stack([c["embedding"] for c in embedded]).astype("float32")
    new_records = [
        {"chunk_id": c["chunk_id"], "text": c["text"], "metadata": c["metadata"]}
        for c in embedded
    ]

    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        if index.d != vectors.shape[1]:
            raise RuntimeError(
                f"Embedding dimension mismatch: index is {index.d}-d "
                f"but new vectors are {vectors.shape[1]}-d. Rebuild required."
            )
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        index = faiss.IndexFlatIP(vectors.shape[1])
        records = []

    index.add(vectors)
    records.extend(new_records)

    # Write atomically: temp file then rename, so a crash mid-write doesn't
    # leave the metadata file in a half-written state.
    faiss.write_index(index, INDEX_FILE)
    _atomic_write_json(METADATA_FILE, records)

    # Invalidate the retrieval cache so the next /ask sees the new chunks.
    # Imported lazily to avoid a circular import at module load.
    from app.retrieval import reload_index
    reload_index()

    return index.ntotal


def _atomic_write_json(path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _unlink_quiet(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except OSError:
        logger.warning(f"failed to remove file during rollback: {p}")
