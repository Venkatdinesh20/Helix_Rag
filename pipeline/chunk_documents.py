"""
Step 3: Text Chunking
----------------------
Splits cleaned documents into smaller, overlapping chunks.

Why chunk?
  LLMs and embedding models have input size limits. A long document
  cannot be passed as one unit. More importantly, embedding a 50-page
  document into a single vector loses precision — it becomes an average
  of everything and matches nothing well in a search.

Why overlap?
  If a sentence falls exactly at a chunk boundary, it gets split across
  two chunks and may not match any query well. Overlap ensures that
  context near a boundary appears in both the previous and next chunk.

  Example (overlap = 80 chars):
    Chunk 1: "...request a full refund within 30 days of purchase. To initiate..."
    Chunk 2: "...within 30 days of purchase. To initiate a refund, contact support..."
    The shared text bridges the boundary.

Parameters (from architecture doc):
  chunk_size : int  — target chunk length in characters (default 500)
  overlap    : int  — number of characters to repeat between chunks (default 80)

Output shape:
  Each chunk is a dict:
  {
    "chunk_id"   : "company_policies_001",
    "text"       : "...chunk text...",
    "metadata"   : {
      "document_name" : "company_policies.txt",
      "page"          : 1,
      "source"        : "data/raw/company_policies.txt",
      "file_type"     : "txt",
      "chunk_index"   : 0
    }
  }
"""

from pathlib import Path
from typing import Any, Dict, List


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 80,
) -> List[Dict[str, Any]]:
    """
    Chunk all documents in the list.
    Returns a flat list of chunk dicts across all documents.
    """
    all_chunks = []
    for doc in documents:
        chunks = _chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks


def _chunk_document(
    doc: Dict[str, Any],
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """
    Split a single document's text into overlapping chunks.

    Strategy: character-level sliding window.
      - Start at position 0
      - Take chunk_size characters
      - Move forward by (chunk_size - overlap) characters
      - Repeat until the end of the text

    Why character-level instead of token-level?
      Token counts vary by model. Characters are universal and predictable.
      For a production system you would switch to a tokenizer-aware splitter,
      but character-based chunking is a solid starting point.
    """
    text = doc["text"]
    metadata = doc["metadata"]
    doc_stem = Path(metadata["document_name"]).stem  # e.g. "company_policies"

    step = chunk_size - overlap  # how far we advance each iteration
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        # Skip chunks that are only whitespace (can happen at the end)
        if chunk_text:
            chunk_id = f"{doc_stem}_{chunk_index + 1:03d}"

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    **metadata,               # preserve all original metadata
                    "chunk_index": chunk_index,  # position of this chunk in the doc
                }
            })
            chunk_index += 1

        start += step

    return chunks


if __name__ == "__main__":
    """
    Quick smoke test — run with:
      python -m pipeline.chunk_documents
    """
    from pipeline.clean_text import clean_documents
    from pipeline.ingest_documents import load_documents_from_folder

    raw_docs = load_documents_from_folder("data/raw")
    cleaned_docs = clean_documents(raw_docs)
    chunks = chunk_documents(cleaned_docs, chunk_size=500, overlap=80)

    print(f"Total chunks created: {len(chunks)}\n")
    for chunk in chunks:
        print(f"  [{chunk['chunk_id']}]  {len(chunk['text'])} chars")
        print(f"    {chunk['text'][:100]!r}")
        print()
