"""
Phase 3: Evaluation Metrics
-----------------------------
Measures whether the RAG system is actually working.

This is one of the most important parts of ML Engineering.
Building a system is easy. Knowing if it works — and by how much —
is what separates a production ML system from a demo.

Two evaluation areas:

1. RETRIEVAL EVALUATION
   Checks whether the correct document was retrieved.

   Metrics:
     Precision@K  — of the top-K retrieved chunks, what fraction are relevant?
                    "Are we returning mostly good results?"
     Recall@K     — was the expected document present in the top-K results?
                    "Did we find what we needed?"
     MRR          — Mean Reciprocal Rank. How high was the first correct result
                    ranked? Score of 1.0 = always first. 0.5 = first relevant
                    result was at position 2 on average.

2. ANSWER EVALUATION
   Checks whether the generated answer is grounded in the retrieved context.

   Metrics:
     Keyword match rate — do expected keywords appear in the answer?
                          A lightweight proxy for answer correctness.
     Avg retrieval score — mean cosine similarity of top retrieved chunk.
                           Higher = more confident retrieval.

How to use this:
  python -m evaluation.metrics

This script:
  1. Loads the test questions from test_questions.json
  2. Runs each question through the full RAG pipeline
  3. Computes retrieval and answer metrics
  4. Prints a summary report
  5. Saves results to evaluation/results.json

In production: run this in CI/CD after every code change.
If metrics drop below a threshold, block the deployment.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.generation import build_context_block
from app.rag_service import ask
from app.retrieval import retrieve


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_docs: List[str], expected_doc: str, k: int) -> float:
    """
    Precision@K: fraction of top-K retrieved docs that are relevant.
    In a single-document setup, this is 1/k if the expected doc appears
    in top-K, else 0. For multi-document setups, count all relevant docs.

    Formula: |relevant ∩ retrieved@K| / K
    """
    top_k_docs = retrieved_docs[:k]
    relevant_count = sum(1 for d in top_k_docs if d == expected_doc)
    return relevant_count / k if k > 0 else 0.0


def recall_at_k(retrieved_docs: List[str], expected_doc: str, k: int) -> float:
    """
    Recall@K: was the expected document found in the top-K results?
    Binary for single-expected-document setups: 1.0 if found, 0.0 if not.

    Formula: |relevant ∩ retrieved@K| / |relevant|
    Here |relevant| = 1 (one expected document per question).
    """
    top_k_docs = retrieved_docs[:k]
    return 1.0 if expected_doc in top_k_docs else 0.0


def reciprocal_rank(retrieved_docs: List[str], expected_doc: str) -> float:
    """
    Reciprocal Rank: 1 / rank_of_first_relevant_result.

    Examples:
      First result is correct   → RR = 1/1 = 1.0
      Second result is correct  → RR = 1/2 = 0.5
      Third result is correct   → RR = 1/3 = 0.33
      Not found                 → RR = 0.0

    MRR (Mean Reciprocal Rank) = average of RR across all questions.
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc == expected_doc:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------

def keyword_match_rate(answer: str, expected_keywords: List[str]) -> float:
    """
    Fraction of expected keywords found in the answer (case-insensitive).

    This is a lightweight proxy for answer correctness when you don't have
    an LLM-as-judge or a reference answer. It's not perfect — a keyword
    can appear without the answer being correct — but it's fast, cheap,
    and catches obvious failures.

    In production: use an LLM to judge faithfulness and answer relevance
    (e.g. RAGAS framework). That requires an LLM API call per evaluation.
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matched / len(expected_keywords)


# ---------------------------------------------------------------------------
# Faithfulness metric — LLM-as-judge (Architecture Section 5.9B)
# ---------------------------------------------------------------------------

def faithfulness_score(
    question: str,
    answer: str,
    context_chunks: List[Dict[str, Any]],
) -> Optional[float]:
    """
    Ask the LLM to judge whether the answer is grounded in the retrieved context.

    Returns:
      1.0  — answer is fully supported by the context
      0.5  — answer is partially supported
      0.0  — answer is not supported / hallucinated
      None — LLM not available (no API key), skipped

    Why LLM-as-judge?
      Keyword matching can pass even when the answer is misleading.
      An LLM can assess semantic faithfulness: does the answer make claims
      beyond what the context actually says?

    Cost: one LLM call per evaluated question. Keep the test set small or
    run only nightly in CI to manage cost.
    """
    try:
        from openai import OpenAI, OpenAIError
    except ImportError:
        return None

    api_key = settings.openai_api_key.get_secret_value()
    if not api_key or api_key == "your-api-key-here":
        return None

    context_block = build_context_block(context_chunks)

    judge_prompt = f"""You are a strict factual auditor for a RAG system.

Context provided to the RAG system:
{context_block}

Question asked:
{question}

Answer produced by the RAG system:
{answer}

Task: Decide whether the answer is supported by the context.

Reply with ONLY one of these three tokens — nothing else:
  FULLY     — every claim in the answer is directly supported by the context
  PARTIALLY — some claims are supported but the answer adds or exaggerates information
  NOT       — the answer makes claims not found in the context (hallucination)
"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=5,
        )
        verdict = response.choices[0].message.content.strip().upper()
        mapping = {"FULLY": 1.0, "PARTIALLY": 0.5, "NOT": 0.0}
        return mapping.get(verdict, None)
    except OpenAIError:
        return None

def evaluate(test_questions_path: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Run the full evaluation on the test question set.
    Returns a dict with per-question results and aggregate metrics.
    """
    with open(test_questions_path, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    results = []
    precision_scores = []
    recall_scores = []
    rr_scores = []
    keyword_scores = []
    faithfulness_scores = []
    retrieval_score_list = []

    print(f"\nRunning evaluation on {len(test_questions)} question(s)...")
    print("=" * 60)

    for q in test_questions:
        question = q["question"]
        expected_doc = q["expected_document"]
        expected_keywords = q.get("expected_keywords", [])

        # Retrieve chunks (without calling LLM — cheaper for retrieval eval)
        retrieved = retrieve(question, top_k=top_k)
        retrieved_docs = [r["metadata"]["document_name"] for r in retrieved]
        top_score = retrieved[0]["score"] if retrieved else 0.0

        # Compute retrieval metrics
        p_at_k = precision_at_k(retrieved_docs, expected_doc, top_k)
        r_at_k = recall_at_k(retrieved_docs, expected_doc, top_k)
        rr      = reciprocal_rank(retrieved_docs, expected_doc)

        # Get full RAG answer (includes demo mode if no API key)
        rag_result = ask(question)
        answer = rag_result["answer"]

        # Compute answer metrics
        kw_rate = keyword_match_rate(answer, expected_keywords)
        faith = faithfulness_score(question, answer, retrieved)

        precision_scores.append(p_at_k)
        recall_scores.append(r_at_k)
        rr_scores.append(rr)
        keyword_scores.append(kw_rate)
        if faith is not None:
            faithfulness_scores.append(faith)
        retrieval_score_list.append(top_score)

        result = {
            "id":                q["id"],
            "question":          question,
            "expected_document": expected_doc,
            "top_retrieved_doc": retrieved_docs[0] if retrieved_docs else None,
            "precision_at_k":    round(p_at_k, 4),
            "recall_at_k":       round(r_at_k, 4),
            "reciprocal_rank":   round(rr, 4),
            "keyword_match_rate": round(kw_rate, 4),
            "faithfulness_score": faith,
            "top_retrieval_score": round(top_score, 4),
            "answer_preview":    answer[:200],
        }
        results.append(result)

        # Per-question output
        status = "PASS" if r_at_k >= 1.0 else "FAIL"
        faith_str = f"{faith:.2f}" if faith is not None else "n/a (no key)"
        print(f"\n[{status}] {q['id']}: {question}")
        print(f"  Expected doc : {expected_doc}")
        print(f"  Top retrieved: {retrieved_docs[0] if retrieved_docs else 'none'} (score={top_score:.4f})")
        print(f"  Precision@{top_k}={p_at_k:.2f}  Recall@{top_k}={r_at_k:.2f}  RR={rr:.2f}  Keywords={kw_rate:.2f}  Faithfulness={faith_str}")

    # Aggregate metrics
    n = len(test_questions)
    mean_faithfulness = (
        round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
        if faithfulness_scores else None
    )
    summary = {
        "num_questions":        n,
        "top_k":                top_k,
        "mean_precision_at_k":  round(sum(precision_scores) / n, 4),
        "mean_recall_at_k":     round(sum(recall_scores) / n, 4),
        "mrr":                  round(sum(rr_scores) / n, 4),
        "mean_keyword_match":   round(sum(keyword_scores) / n, 4),
        "mean_faithfulness":    mean_faithfulness,
        "mean_retrieval_score": round(sum(retrieval_score_list) / n, 4),
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Questions evaluated : {summary['num_questions']}")
    print(f"  Top-K               : {summary['top_k']}")
    print(f"  Mean Precision@K    : {summary['mean_precision_at_k']:.4f}")
    print(f"  Mean Recall@K       : {summary['mean_recall_at_k']:.4f}  ← 1.0 = perfect")
    print(f"  MRR                 : {summary['mrr']:.4f}            ← 1.0 = always ranked first")
    print(f"  Mean Keyword Match  : {summary['mean_keyword_match']:.4f}  ← proxy for answer quality")
    faith_display = f"{mean_faithfulness:.4f}" if mean_faithfulness is not None else "n/a (no API key)"
    print(f"  Mean Faithfulness   : {faith_display}  ← LLM-as-judge (1.0=grounded, 0=hallucinated)")
    print(f"  Mean Retrieval Score: {summary['mean_retrieval_score']:.4f}  ← avg cosine similarity")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(__file__), "test_questions.json")
    eval_output = evaluate(test_path, top_k=5)

    # Save results for tracking over time / experiment comparison
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
