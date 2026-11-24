"""
Curated sample questions surfaced in the empty-state UI.

Why a dedicated module?
  - The list is part of the product surface (first impressions). Keeping it
    in code (versioned, reviewed) rather than scattered in the frontend
    means the API can serve the same list to docs, tests, and the UI.
  - One auto-derived sample is appended so the chip set always references
    a real indexed document — proof to a first-time visitor that the
    knowledge base is non-empty and that retrieval works.

Contract:
    list_samples() -> list[{"label": str, "question": str}]
  - "label"    : short text shown on the chip
  - "question" : the exact text submitted to /ask when the chip is clicked
  - At most MAX_SAMPLES items returned.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Static, hand-curated prompts. These work for almost any corpus and double
# as smoke-tests of the four behaviours that matter most:
#   summarisation, fact extraction, comparison, and grounding-on-absence.
_BASE_SAMPLES: List[Dict[str, str]] = [
    {
        "label":    "Summarize the document",
        "question": "Give me a concise summary of the main points covered in this document.",
    },
    {
        "label":    "Key facts",
        "question": "What are the most important facts or claims stated in this document?",
    },
    {
        "label":    "Names & dates",
        "question": "List the proper names, organisations and dates that appear in the document.",
    },
    {
        "label":    "Out-of-scope check",
        "question": "What is the population of Mars?",
    },
]

MAX_SAMPLES = 5


def _derive_corpus_sample() -> Dict[str, str] | None:
    """
    Build one sample that references an actual indexed document name. This
    makes the chip set feel "live" even when the corpus changes. Returns
    None if the index is empty or cannot be loaded.
    """
    try:
        from app.retrieval import _get_index_and_metadata  # lazy: avoid import cycles
        _, records = _get_index_and_metadata()
    except Exception:
        return None

    if not records:
        return None

    # Pick the first record's document name. We don't need anything clever
    # here — just a real filename so the user sees their own corpus reflected.
    doc_name = (records[0].get("metadata") or {}).get("document_name")
    if not doc_name or not isinstance(doc_name, str):
        return None

    # Strip the extension for a cleaner chip label
    pretty = doc_name.rsplit(".", 1)[0] if "." in doc_name else doc_name
    pretty = pretty.replace("_", " ").replace("-", " ").strip()
    if not pretty:
        return None

    return {
        "label":    f"About {pretty}"[:48],
        "question": f"What does the document '{doc_name}' say in the most important sections?",
    }


def list_samples() -> List[Dict[str, Any]]:
    """Return the curated starter prompts, with one corpus-derived addition."""
    samples = list(_BASE_SAMPLES)
    derived = _derive_corpus_sample()
    if derived is not None:
        # Put the corpus-specific chip first so it's the most prominent.
        samples.insert(0, derived)
    return samples[:MAX_SAMPLES]
