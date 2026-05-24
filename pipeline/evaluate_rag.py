"""
Experiment Tracking: RAG Hyperparameter Search
------------------------------------------------
Architecture Section 5.10 — Experiment Tracking.

The quality of a RAG system depends heavily on three tunable parameters:
  - chunk_size    : how many characters per chunk
  - chunk_overlap : how many characters overlap between adjacent chunks
  - top_k         : how many chunks are retrieved per query

This script runs the full pipeline (chunk → embed → index → retrieve → evaluate)
for each combination and produces a comparison table. This answers questions like:
  "Does a larger chunk size improve or hurt retrieval accuracy?"
  "Is top_k=3 good enough or do I need top_k=7?"

Why this matters:
  Changing a single parameter can shift MRR from 0.5 to 1.0.
  You cannot know the best setting without measuring it empirically.
  This script makes that measurement systematic and repeatable.

Usage:
  python -m pipeline.evaluate_rag

Output:
  evaluation/experiment_results.json  — machine-readable results for all runs
  Comparison table printed to stdout

In production: hook this into CI/CD. Run it when you change chunking logic.
Log results to Vertex AI Experiments or MLflow for long-term tracking.
"""

import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

# Add project root so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import faiss

from pipeline.chunk_documents import chunk_document
from pipeline.generate_embeddings import embed_chunks, embed_query
from pipeline.ingest_documents import ingest_folder

# ---------------------------------------------------------------------------
# Experiment grid — edit this to test different hyperparameters
# ---------------------------------------------------------------------------
EXPERIMENT_GRID = [
    {"chunk_size": 300, "chunk_overlap": 50,  "top_k": 5},
    {"chunk_size": 500, "chunk_overlap": 80,  "top_k": 5},   # current default
    {"chunk_size": 500, "chunk_overlap": 80,  "top_k": 3},
    {"chunk_size": 800, "chunk_overlap": 100, "top_k": 5},
    {"chunk_size": 800, "chunk_overlap": 100, "top_k": 7},
]

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TEST_QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "test_questions.json"
)


# ---------------------------------------------------------------------------
# Inline retrieval (no FAISS file on disk — builds an in-memory index)
# This isolates the experiment from whatever index is currently deployed.
# ---------------------------------------------------------------------------

def build_in_memory_index(
    raw_data_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple:
    """
    Ingest documents, chunk them, embed them, and return an in-memory FAISS
    index together with the metadata records.  Nothing is written to disk.
    """
    documents = ingest_folder(raw_data_dir)

    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(f"No chunks produced from {raw_data_dir}")

    embeddings = embed_chunks(all_chunks)          # shape: (n_chunks, dim)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)                 # inner product = cosine for L2-normed vecs
    index.add(embeddings.astype(np.float32))

    return index, all_chunks


def retrieve_from_index(
    query: str,
    index,
    metadata_records: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run a vector search against the in-memory index."""
    query_vec = embed_query(query).reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata_records[idx]
        results.append({
            "document_name": chunk["metadata"]["document_name"],
            "score": float(score),
        })
    return results


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_docs: List[str], expected_doc: str) -> float:
    return 1.0 if expected_doc in retrieved_docs else 0.0


def reciprocal_rank(retrieved_docs: List[str], expected_doc: str) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc == expected_doc:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_experiment(
    config: Dict[str, Any],
    test_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    chunk_size    = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]
    top_k         = config["top_k"]

    t0 = time.perf_counter()
    index, metadata_records = build_in_memory_index(RAW_DATA_DIR, chunk_size, chunk_overlap)
    build_time_s = round(time.perf_counter() - t0, 2)

    recall_scores = []
    rr_scores     = []

    for q in test_questions:
        retrieved = retrieve_from_index(
            q["question"], index, metadata_records, top_k
        )
        retrieved_docs = [r["document_name"] for r in retrieved]
        expected_doc   = q["expected_document"]

        recall_scores.append(recall_at_k(retrieved_docs, expected_doc))
        rr_scores.append(reciprocal_rank(retrieved_docs, expected_doc))

    n = len(test_questions)
    return {
        "chunk_size":    chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k":         top_k,
        "n_chunks":      len(metadata_records),
        "mean_recall":   round(sum(recall_scores) / n, 4),
        "mrr":           round(sum(rr_scores) / n, 4),
        "build_time_s":  build_time_s,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    print(f"\nRunning {len(EXPERIMENT_GRID)} experiments on {len(test_questions)} questions...")
    print("=" * 75)

    all_results = []
    for i, config in enumerate(EXPERIMENT_GRID, start=1):
        label = f"[{i}/{len(EXPERIMENT_GRID)}] chunk={config['chunk_size']}/{config['chunk_overlap']} top_k={config['top_k']}"
        print(f"  {label} ...", end=" ", flush=True)
        result = run_experiment(config, test_questions)
        all_results.append(result)
        print(f"MRR={result['mrr']:.4f}  Recall={result['mean_recall']:.4f}  chunks={result['n_chunks']}")

    # Sort by MRR descending so the best config is on top
    all_results.sort(key=lambda r: (r["mrr"], r["mean_recall"]), reverse=True)

    print("\n" + "=" * 75)
    print(f"{'RANK':<5} {'CHUNK':>6} {'OVLP':>5} {'TOP_K':>6} {'CHUNKS':>7} {'MRR':>6} {'RECALL':>7} {'BUILD_S':>8}")
    print("-" * 75)
    for rank, r in enumerate(all_results, start=1):
        marker = " ← best" if rank == 1 else ""
        print(
            f"{rank:<5} {r['chunk_size']:>6} {r['chunk_overlap']:>5} {r['top_k']:>6} "
            f"{r['n_chunks']:>7} {r['mrr']:>6.4f} {r['mean_recall']:>7.4f} "
            f"{r['build_time_s']:>7.2f}s{marker}"
        )

    output_path = os.path.join(
        os.path.dirname(__file__), "..", "evaluation", "experiment_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"experiments": all_results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")
