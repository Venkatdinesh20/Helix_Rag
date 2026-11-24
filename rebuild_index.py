"""
Run this script whenever you add, remove, or replace files in data/raw/.
It rebuilds the FAISS vector store so the API serves answers from your latest documents.

Usage:
    python rebuild_index.py
"""
import os
import sys

# Ensure the project root is on the path regardless of where this script is called from
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from pipeline.ingest_documents import load_documents_from_folder
from pipeline.clean_text import clean_documents
from pipeline.chunk_documents import chunk_documents
from pipeline.generate_embeddings import embed_chunks
from pipeline.build_index import build_index, INDEX_FILE, METADATA_FILE

RAW_FOLDER = os.path.join(ROOT, "data", "raw")

print("=" * 50)
print("RAG — Rebuilding Vector Store")
print("=" * 50)

print(f"\n[1/5] Ingesting documents from {RAW_FOLDER} ...")
raw_docs = load_documents_from_folder(RAW_FOLDER)
print(f"  {len(raw_docs)} page(s) loaded")

print("\n[2/5] Cleaning text...")
cleaned = clean_documents(raw_docs)

print("\n[3/5] Chunking...")
chunks = chunk_documents(cleaned, chunk_size=500, overlap=80)
print(f"  {len(chunks)} chunks created")

print("\n[4/5] Generating embeddings (may take a moment on first run)...")
embedded = embed_chunks(chunks)
print(f"  {len(embedded)} chunks embedded")

print("\n[5/5] Saving FAISS index...")
build_index(embedded)

print("\nDone! Restart the server for changes to take effect.")
print(f"  Index    : {INDEX_FILE}")
print(f"  Metadata : {METADATA_FILE}")
