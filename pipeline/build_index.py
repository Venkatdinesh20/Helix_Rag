"""
Step 5: Build the FAISS Index
------------------------------
Takes embedded chunks and saves them into a FAISS vector index on disk.

What is FAISS?
  Facebook AI Similarity Search — a library for fast nearest-neighbor
  search over vectors. Given a query vector, it finds the most similar
  stored vectors in milliseconds, even with millions of entries.

Why FAISS for local/demo use?
  - No server required, runs entirely in memory / local files
  - Fast enough for thousands to hundreds-of-thousands of chunks
  - In production this would be replaced by Vertex AI Vector Search,
    Pinecone, Weaviate, or Qdrant — but the interface stays the same.

What is saved to disk:
  vector_store/faiss_index.index   — binary file with the raw vectors
  vector_store/chunks_metadata.json — text + metadata for each chunk

How they are linked:
  Both files use the same ordering. Row i in the FAISS index corresponds
  to chunks_metadata[i]. This is how we turn "index position 3" back into
  the original chunk text and document name.

Index type: IndexFlatIP
  - "Flat" means exact search (no approximation)
  - "IP" means Inner Product (dot product)
  - Because our embeddings are L2-normalized, inner product == cosine similarity
  - For larger indexes, IndexIVFFlat or IndexHNSWFlat would be faster
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np


VECTOR_STORE_DIR = "vector_store"
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
METADATA_FILE = os.path.join(VECTOR_STORE_DIR, "chunks_metadata.json")


def build_index(embedded_chunks: List[Dict[str, Any]]) -> None:
    """
    Build a FAISS index from embedded chunks and save it to disk.

    Steps:
      1. Extract the embedding vectors into a numpy matrix
      2. Create a FAISS IndexFlatIP (inner product = cosine for L2-normed vecs)
      3. Add all vectors to the index
      4. Save the index binary to disk
      5. Save chunk text + metadata to JSON (linked by position)
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # Step 1: Stack embeddings into a 2D numpy array
    # Shape: (num_chunks, embedding_dim) e.g. (5, 384)
    embeddings = np.stack([chunk["embedding"] for chunk in embedded_chunks]).astype("float32")
    num_chunks, embedding_dim = embeddings.shape
    print(f"  Building index: {num_chunks} chunks, {embedding_dim}-dim vectors")

    # Step 2: Create the FAISS index
    # IndexFlatIP = exact search using inner product (cosine sim for L2-normed vectors)
    index = faiss.IndexFlatIP(embedding_dim)

    # Step 3: Add all vectors
    index.add(embeddings)
    print(f"  FAISS index total vectors: {index.ntotal}")

    # Step 4: Save the index to disk
    faiss.write_index(index, INDEX_FILE)
    print(f"  Index saved to: {INDEX_FILE}")

    # Step 5: Save metadata (text + metadata dict) as JSON
    # We strip the 'embedding' key — vectors are already in the FAISS file
    metadata_records = [
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        }
        for chunk in embedded_chunks
    ]
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved to: {METADATA_FILE}")


def load_index() -> tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Load the FAISS index and metadata from disk.
    Returns (index, metadata_records).

    Called at API startup — loaded once and reused for every query.
    """
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_FILE}. "
            "Run the pipeline first: python -m pipeline.build_index"
        )
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_FILE}. "
            "Run the pipeline first: python -m pipeline.build_index"
        )

    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata_records = json.load(f)

    return index, metadata_records


if __name__ == "__main__":
    """
    Run the full batch pipeline end-to-end:
      python -m pipeline.build_index

    This is the single command you run whenever you add/update documents.
    It will:
      1. Ingest  → 2. Clean  → 3. Chunk  → 4. Embed  → 5. Index
    """
    from pipeline.chunk_documents import chunk_documents
    from pipeline.clean_text import clean_documents
    from pipeline.generate_embeddings import embed_chunks
    from pipeline.ingest_documents import load_documents_from_folder

    raw_folder = "data/raw"
    print("=" * 50)
    print("RAG Batch Pipeline")
    print("=" * 50)

    print("\n[1/5] Ingesting documents...")
    raw_docs = load_documents_from_folder(raw_folder)
    print(f"  {len(raw_docs)} page(s) loaded")

    print("\n[2/5] Cleaning text...")
    cleaned = clean_documents(raw_docs)
    print(f"  {len(cleaned)} document(s) cleaned")

    print("\n[3/5] Chunking...")
    chunks = chunk_documents(cleaned, chunk_size=500, overlap=80)
    print(f"  {len(chunks)} chunk(s) created")

    print("\n[4/5] Generating embeddings...")
    embedded = embed_chunks(chunks)
    print(f"  {len(embedded)} chunk(s) embedded")

    print("\n[5/5] Building FAISS index...")
    build_index(embedded)

    print("\nPipeline complete. Vector store is ready.")
    print(f"  Index : {INDEX_FILE}")
    print(f"  Metadata : {METADATA_FILE}")
