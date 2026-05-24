"""
Step 4: Embedding Generation
-----------------------------
Converts text chunks into numerical vectors (embeddings).

Why embeddings?
  Keyword search finds documents that contain the exact words used.
  Semantic search finds documents that MEAN the same thing, even if
  different words are used. Embeddings make semantic search possible.

  Example:
    Query : "How do I get my money back?"
    Chunk : "Customers may request a refund within 30 days."
    A keyword search finds nothing. A vector search finds this chunk
    because both sentences mean the same thing.

How it works:
  A pre-trained sentence-transformer model reads a text string and
  produces a fixed-length vector (e.g. 384 numbers for MiniLM-L6-v2).
  Texts with similar meaning produce similar vectors.
  Similarity is measured by cosine distance in vector space.

Model: all-MiniLM-L6-v2
  - Small (80MB), fast, good quality for English text
  - Produces 384-dimensional vectors
  - Already listed in requirements.txt (sentence-transformers)

Performance note:
  The model is loaded once and reused for all chunks.
  Loading it inside a loop would be extremely slow.
"""

from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Module-level model — loaded once when this module is first imported.
# All functions in this module share the same model instance.
# ---------------------------------------------------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model.
    The model is only downloaded/loaded on the first call.
    Subsequent calls reuse the already-loaded model in memory.
    """
    global _model
    if _model is None:
        print(f"  Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        print(f"  Model loaded. Embedding dimension: {_model.get_embedding_dimension()}")
    return _model


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add an 'embedding' key to each chunk dict.

    The embedding is a numpy array of float32 values.
    Shape: (embedding_dim,) — e.g. (384,) for MiniLM-L6-v2

    Returns a new list — original chunks are not mutated.
    """
    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]

    print(f"  Generating embeddings for {len(texts)} chunk(s)...")

    # encode() handles batching internally and returns a numpy array
    # shape: (num_chunks, embedding_dim)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize so cosine sim = dot product
    )

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            **chunk,
            "embedding": embedding  # numpy array, shape (384,)
        })

    return embedded_chunks


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string for use at retrieval time.

    Returns a 1D numpy array, shape (embedding_dim,).
    Uses the same model and normalization as embed_chunks so that
    dot product similarity is comparable.
    """
    model = _get_model()
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding[0]  # shape (embedding_dim,)


if __name__ == "__main__":
    """
    Quick smoke test — run with:
      python -m pipeline.generate_embeddings
    """
    from pipeline.chunk_documents import chunk_documents
    from pipeline.clean_text import clean_documents
    from pipeline.ingest_documents import load_documents_from_folder

    raw_docs = load_documents_from_folder("data/raw")
    cleaned = clean_documents(raw_docs)
    chunks = chunk_documents(cleaned, chunk_size=500, overlap=80)
    embedded = embed_chunks(chunks)

    print(f"\nEmbedded {len(embedded)} chunk(s)")
    for ec in embedded:
        emb = ec["embedding"]
        print(f"  [{ec['chunk_id']}]  vector shape: {emb.shape}  norm: {np.linalg.norm(emb):.4f}")
