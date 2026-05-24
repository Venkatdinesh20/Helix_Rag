"""
Step 2: Text Cleaning
---------------------
Cleans raw extracted text before it is chunked and embedded.

Why clean before chunking?
  Noisy text (extra spaces, broken hyphens, garbage characters) creates
  noisy embeddings. Noisy embeddings reduce retrieval accuracy — the
  vector search returns wrong or irrelevant chunks.

What we clean:
  1. Normalize line endings  (Windows \r\n → \n)
  2. Repair hyphenated line breaks  (re-\nfund → refund)
  3. Collapse multiple blank lines  (3 empty lines → 1)
  4. Strip leading/trailing whitespace from each line
  5. Collapse multiple spaces inside a line
  6. Strip leading/trailing whitespace from the whole text

What we do NOT touch:
  - Metadata (document_name, page, source) — never modified here
  - Intentional paragraph breaks — we keep single blank lines
"""

import re
from typing import Any, Dict, List


def clean_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply text cleaning to every document in the list.
    Returns a new list — original documents are not mutated.
    """
    return [_clean_document(doc) for doc in documents]


def _clean_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Clean the text of a single document, preserving its metadata."""
    cleaned_text = clean_text(doc["text"])
    return {
        "text": cleaned_text,
        "metadata": doc["metadata"]  # metadata is never modified
    }


def clean_text(text: str) -> str:
    """
    Apply all cleaning steps to a raw text string.

    Steps in order — each step builds on the previous one:
      1. Normalize line endings first, so later steps only deal with \n
      2. Fix hyphenated breaks before collapsing lines
      3. Collapse blank lines before stripping individual lines
      4. Strip each line's whitespace
      5. Collapse runs of spaces within a line
      6. Final strip of the whole text
    """
    # Step 1: Normalize Windows line endings to Unix
    # \r\n (Windows) and \r (old Mac) both become \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 2: Repair hyphenated line breaks
    # PDF extraction often splits a word across lines with a hyphen:
    #   "re-\nfund" → "refund"
    # The \s* handles any trailing space before the newline.
    text = re.sub(r"-\s*\n\s*", "", text)

    # Step 3: Collapse 3+ consecutive blank lines into 2
    # Preserves paragraph structure without excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 4: Strip whitespace from the start and end of each line
    # Removes tabs, trailing spaces, and indentation artifacts
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Step 5: Collapse multiple spaces within a line into one
    # e.g. "hello   world" → "hello world"
    text = re.sub(r" {2,}", " ", text)

    # Step 6: Strip the whole text
    text = text.strip()

    return text


if __name__ == "__main__":
    """
    Quick smoke test — run with:
      python -m pipeline.clean_text
    """
    from pipeline.ingest_documents import load_documents_from_folder

    raw_docs = load_documents_from_folder("data/raw")
    cleaned_docs = clean_documents(raw_docs)

    print(f"Cleaned {len(cleaned_docs)} document(s)\n")
    for original, cleaned in zip(raw_docs, cleaned_docs):
        original_len = len(original["text"])
        cleaned_len = len(cleaned["text"])
        name = cleaned["metadata"]["document_name"]
        print(f"  {name}: {original_len} chars → {cleaned_len} chars")
        print(f"  Preview: {cleaned['text'][:200]!r}")
        print()
