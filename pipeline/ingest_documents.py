"""
Step 1: Document Ingestion
--------------------------
Reads raw PDF and TXT files from a local folder OR a GCS bucket prefix.

Accepts two URI forms:
  - Local path  :  data/raw/          (or any relative/absolute path)
  - GCS prefix  :  gs://my-bucket/docs/

Why metadata?
  Every answer the RAG system gives must be traceable back to a
  specific document and page. Metadata is what makes that possible.

Output shape (list of dicts):
  [
    {
      "text": "The actual extracted text...",
      "metadata": {
        "document_name": "refund_policy.pdf",
        "page": 2,
        "source": "gs://my-bucket/docs/refund_policy.pdf",
        "file_type": "pdf"
      }
    },
    ...
  ]
"""

import io
import os
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader


def load_documents_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Dispatch to GCS or local loader based on the path prefix.
    Pass a gs://bucket/prefix URI to read from Cloud Storage.
    """
    if folder_path.startswith("gs://"):
        return _load_from_gcs(folder_path)
    return _load_from_local(folder_path)


def _load_from_gcs(gcs_uri: str) -> List[Dict[str, Any]]:
    """
    Read PDF and TXT files from a GCS bucket prefix.

    Requires:  pip install google-cloud-storage
    Auth:      Application Default Credentials (ADC) — works automatically
               on Cloud Run, Cloud Build, and after `gcloud auth application-default login`

    gcs_uri format:  gs://bucket-name/optional/prefix/
    """
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        raise RuntimeError(
            "google-cloud-storage is required for GCS ingestion. "
            "Install it with: pip install google-cloud-storage"
        )

    # Parse  gs://bucket/prefix  →  bucket="bucket", prefix="prefix/"
    without_scheme = gcs_uri[5:]  # strip "gs://"
    bucket_name, _, prefix = without_scheme.partition("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print(f"  [WARN] No files found at {gcs_uri}")
        return []

    documents = []
    for blob in blobs:
        name = blob.name
        suffix = Path(name).suffix.lower()
        file_bytes = blob.download_as_bytes()
        source_uri = f"gs://{bucket_name}/{name}"
        doc_name = Path(name).name

        if suffix == ".pdf":
            try:
                pages = _load_pdf_bytes(file_bytes, doc_name, source_uri)
                documents.extend(pages)
                print(f"  [OK] {doc_name}: {len(pages)} page(s) loaded from GCS")
            except Exception as e:
                print(f"  [SKIP] {doc_name}: could not read — {e}")

        elif suffix == ".txt":
            try:
                pages = _load_txt_bytes(file_bytes, doc_name, source_uri)
                documents.extend(pages)
                print(f"  [OK] {doc_name}: loaded from GCS")
            except Exception as e:
                print(f"  [SKIP] {doc_name}: could not read — {e}")

    return documents


def _load_from_local(folder_path: str) -> List[Dict[str, Any]]:
    """
    Read all PDF and TXT files in a folder.
    Returns a flat list of page-level documents with metadata.
    Skips files that cannot be read and prints a warning.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    documents = []

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() == ".pdf":
            try:
                pages = _load_pdf(file_path)
                documents.extend(pages)
                print(f"  [OK] {file_path.name}: {len(pages)} page(s) loaded")
            except Exception as e:
                print(f"  [SKIP] {file_path.name}: could not read — {e}")

        elif file_path.suffix.lower() == ".txt":
            try:
                pages = _load_txt(file_path)
                documents.extend(pages)
                print(f"  [OK] {file_path.name}: loaded as 1 document")
            except Exception as e:
                print(f"  [SKIP] {file_path.name}: could not read — {e}")

        else:
            # Unsupported file type — skip silently
            pass

    return documents


def _load_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file, one entry per page.

    Why page-by-page?
      Splitting at the page level preserves natural document boundaries
      and lets us tell the user exactly which page an answer came from.
    """
    reader = PdfReader(str(file_path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        # Skip pages with no readable text (e.g. image-only scans)
        if not text or not text.strip():
            continue

        pages.append({
            "text": text,
            "metadata": {
                "document_name": file_path.name,
                "page": page_num,
                "source": str(file_path),
                "file_type": "pdf",
            }
        })

    return pages


def _load_txt(file_path: Path) -> List[Dict[str, Any]]:
    """
    Read a plain text file as a single document entry.

    Text files don't have pages, so page is recorded as 1.
    """
    text = file_path.read_text(encoding="utf-8")

    if not text.strip():
        return []

    return [{
        "text": text,
        "metadata": {
            "document_name": file_path.name,
            "page": 1,
            "source": str(file_path),
            "file_type": "txt",
        }
    }]


if __name__ == "__main__":
    """
    Quick smoke test — run this directly to verify ingestion works:
      python -m pipeline.ingest_documents
      python -m pipeline.ingest_documents gs://my-bucket/docs/
    """
    import sys
    raw_folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    print(f"Loading documents from: {raw_folder}")
    docs = load_documents_from_folder(raw_folder)

    print(f"\nTotal pages/documents loaded: {len(docs)}")
    for doc in docs[:5]:
        m = doc["metadata"]
        print(f"  {m['document_name']} | page {m['page']} | {len(doc['text'])} chars")


# ── Byte-level helpers (used by GCS loader) ──────────────────────────────────

def _load_pdf_bytes(data: bytes, doc_name: str, source: str) -> List[Dict[str, Any]]:
    """Extract text from PDF bytes (used when file comes from GCS)."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        pages.append({
            "text": text,
            "metadata": {
                "document_name": doc_name,
                "page": page_num,
                "source": source,
                "file_type": "pdf",
            }
        })
    return pages


def _load_txt_bytes(data: bytes, doc_name: str, source: str) -> List[Dict[str, Any]]:
    """Decode TXT bytes to string (used when file comes from GCS)."""
    text = data.decode("utf-8")
    if not text.strip():
        return []
    return [{
        "text": text,
        "metadata": {
            "document_name": doc_name,
            "page": 1,
            "source": source,
            "file_type": "txt",
        }
    }]
