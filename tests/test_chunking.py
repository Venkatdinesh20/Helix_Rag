"""
Unit tests for pipeline/chunk_documents.py

Why test chunking?
  Chunking is the step with the most tunable logic (size, overlap, edge cases).
  Bugs here affect every downstream step: embeddings, retrieval, and answers.

Run with:
  python -m pytest tests/test_chunking.py -v
"""

import pytest
from pipeline.chunk_documents import chunk_documents, _chunk_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, name: str = "test.txt") -> dict:
    return {
        "text": text,
        "metadata": {
            "document_name": name,
            "page": 1,
            "source": f"data/raw/{name}",
            "file_type": "txt",
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkDocuments:

    def test_short_text_produces_one_chunk(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        doc = _make_doc("Hello world. This is a short document.")
        chunks = chunk_documents([doc], chunk_size=500, overlap=80)
        assert len(chunks) == 1

    def test_chunk_id_format(self):
        """Chunk IDs should follow the pattern: {stem}_{index:03d}"""
        doc = _make_doc("x" * 1000, name="my_doc.txt")
        chunks = chunk_documents([doc], chunk_size=500, overlap=80)
        assert chunks[0]["chunk_id"] == "my_doc_001"
        assert chunks[1]["chunk_id"] == "my_doc_002"

    def test_overlap_creates_shared_content(self):
        """Consecutive chunks should share overlap characters at their boundary."""
        text = "A" * 500 + "B" * 500
        doc = _make_doc(text)
        chunks = chunk_documents([doc], chunk_size=500, overlap=100)

        # There should be at least 2 chunks
        assert len(chunks) >= 2

        # The end of chunk 1 and start of chunk 2 should share content
        end_of_chunk_1 = chunks[0]["text"][-100:]
        start_of_chunk_2 = chunks[1]["text"][:100]
        assert end_of_chunk_1 == start_of_chunk_2

    def test_metadata_is_preserved(self):
        """Original metadata fields must survive chunking unchanged."""
        doc = _make_doc("Some text.", name="policy.pdf")
        doc["metadata"]["page"] = 7
        chunks = chunk_documents([doc], chunk_size=500, overlap=80)
        assert chunks[0]["metadata"]["document_name"] == "policy.pdf"
        assert chunks[0]["metadata"]["page"] == 7
        assert chunks[0]["metadata"]["file_type"] == "txt"

    def test_chunk_index_is_added_to_metadata(self):
        """Each chunk should have chunk_index added to metadata."""
        doc = _make_doc("x" * 1000)
        chunks = chunk_documents([doc], chunk_size=500, overlap=80)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_empty_text_produces_no_chunks(self):
        """A document with only whitespace should produce no chunks."""
        doc = _make_doc("   \n\n\t  ")
        chunks = chunk_documents([doc], chunk_size=500, overlap=80)
        assert len(chunks) == 0

    def test_multiple_documents(self):
        """Chunks from multiple documents should all be in a flat list."""
        docs = [
            _make_doc("x" * 600, name="doc1.txt"),
            _make_doc("y" * 600, name="doc2.txt"),
        ]
        chunks = chunk_documents(docs, chunk_size=500, overlap=80)
        doc_names = [c["metadata"]["document_name"] for c in chunks]
        assert "doc1.txt" in doc_names
        assert "doc2.txt" in doc_names

    def test_chunk_text_is_not_empty(self):
        """No chunk should have empty or whitespace-only text."""
        doc = _make_doc("Hello " * 200)
        chunks = chunk_documents([doc], chunk_size=500, overlap=80)
        for chunk in chunks:
            assert chunk["text"].strip() != ""
