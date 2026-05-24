"""
Step 7: LLM Answer Generation
-------------------------------
Takes the user's question and retrieved context chunks, builds a prompt,
and calls an LLM to generate a grounded natural language answer.

Key design decisions:
  1. Grounding instruction — the prompt tells the model to answer ONLY from
     the provided context. This is the primary defense against hallucination.

  2. Explicit fallback — if the answer is not in the context, the model is
     instructed to say so. This is better than fabricating an answer.

  3. Source citation — the model is asked to mention source documents.
     This makes answers auditable and trustworthy.

  4. No hardcoded secrets — the API key comes from environment variables
     via config.py. Hardcoding API keys is a serious security vulnerability
     (OWASP A02: Cryptographic Failures).

  5. Graceful degradation — if the LLM call fails (timeout, quota, etc.)
     we return a clear error response rather than crashing the API.

LLM choice: OpenAI gpt-3.5-turbo (configurable via OPENAI_MODEL in .env)
  For production: add retries with exponential backoff, token counting,
  and cost tracking.
"""

import random
import time
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI, OpenAIError

from app.config import settings

# ---------------------------------------------------------------------------
# Module-level OpenAI client singleton
# Creating a new OpenAI() on every request would spin up a new HTTP connection
# pool on every call — wasteful and slower under load. A singleton reuses the
# underlying httpx session across all requests in the process.
# ---------------------------------------------------------------------------
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> Optional[OpenAI]:
    """Return the shared OpenAI client, creating it once on first use."""
    global _openai_client
    if _openai_client is None:
        api_key = settings.openai_api_key.get_secret_value()
        if not api_key or api_key == "your-api-key-here":
            return None
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

# ---------------------------------------------------------------------------
# Prompt template
# This is the most important part of the generation step.
# The system prompt defines the model's behaviour for every request.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Rules you must follow:
1. Answer using information present in the context, including information that is related or relevant to the question even if the exact words don't match.
   Example: if the question asks about "ML Engineer" skills and the context mentions machine learning, Python ML pipelines, or data science work — use that information.
2. Only respond with "I could not find this information in the provided documents." if the context contains absolutely nothing relevant to the question topic.
3. Always cite the source document name(s) when you use information from them.
4. Be concise and accurate. Do not add information not present in the context.
"""


def build_context_block(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a readable context block for the prompt.

    Each chunk is labelled with its source document and page so the model
    can cite them in its answer.
    """
    if not retrieved_chunks:
        return "No context retrieved."

    parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        doc_name = chunk["metadata"].get("document_name", "unknown")
        page = chunk["metadata"].get("page", "?")
        parts.append(
            f"[Source {i}: {doc_name}, page {page}]\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)


def _build_messages(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """
    Compose the OpenAI message list: system prompt → prior turns → current user turn.
    Shared by both the synchronous and streaming generation paths so the
    grounding rules and prompt format stay identical.
    """
    context_block = build_context_block(retrieved_chunks)
    user_message = f"""Context:
{context_block}

Question: {question}

Answer:"""

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return messages


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Generate a grounded answer using the LLM.

    Args:
        question         : the user's question
        retrieved_chunks : output from retrieval.retrieve()
        history          : optional prior conversation turns in OpenAI message
                           format (role/content). Used for multi-turn follow-ups.

    Returns a dict:
      {
        "answer"         : "The refund policy allows...",
        "model"          : "gpt-3.5-turbo",
        "prompt_tokens"  : 312,
        "completion_tokens": 87,
        "total_tokens"   : 399,
        "error"          : None   ← or an error message string if LLM failed
      }
    """
    client = _get_openai_client()
    if client is None:
        # No API key configured — return a clear message instead of crashing
        return {
            "answer": _demo_answer(question, retrieved_chunks),
            "model": "demo-mode",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": None,
        }

    messages = _build_messages(question, retrieved_chunks, history)

    max_retries = 3
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0,      # 0 = deterministic, more consistent for factual QA
                max_tokens=512,     # keep responses concise
            )
            last_error = None
            break  # success — exit retry loop
        except OpenAIError as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s + small random jitter
                wait = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(wait)

    try:
        if last_error is not None:
            raise last_error

        answer = response.choices[0].message.content.strip()
        usage = response.usage

        return {
            "answer":             answer,
            "model":              settings.openai_model,
            "prompt_tokens":      usage.prompt_tokens,
            "completion_tokens":  usage.completion_tokens,
            "total_tokens":       usage.total_tokens,
            "error":              None,
        }

    except OpenAIError as e:
        # LLM call failed — return graceful error, do not crash the API
        return {
            "answer":             "I was unable to generate an answer at this time. Please try again.",
            "model":              settings.openai_model,
            "prompt_tokens":      0,
            "completion_tokens":  0,
            "total_tokens":       0,
            "error":              str(e),
        }


def _demo_answer(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Fallback when no API key is configured.
    Returns the raw retrieved context so the system is still testable
    without an LLM key. Shows exactly what the LLM would receive.
    """
    if not retrieved_chunks:
        return "I could not find this information in the provided documents."

    context = build_context_block(retrieved_chunks)
    top_chunk = retrieved_chunks[0]["text"][:300]

    return (
        f"[DEMO MODE — no OpenAI API key configured]\n\n"
        f"Question: {question}\n\n"
        f"Top retrieved chunk:\n{top_chunk}...\n\n"
        f"In production, the LLM would answer from:\n{context[:400]}..."
    )


# ---------------------------------------------------------------------------
# Streaming generation
#
# stream_answer() yields token deltas as the model produces them, then a
# single final dict with the assembled answer and (best-effort) token usage.
# The caller distinguishes deltas from the final marker by type:
#   ("token", "...")          ← incremental text fragment
#   ("done",  {answer, ...})  ← stream finished cleanly
#   ("error", "message")      ← stream aborted; caller should surface failure
#
# Design notes:
#   - We do not retry mid-stream. A failure after bytes have been delivered
#     would corrupt the answer; instead we yield an "error" event so the API
#     layer can flush an SSE error frame to the client.
#   - usage stats are requested via stream_options when supported by the
#     model; older models may not return usage on stream and we degrade
#     gracefully to zeros.
#   - Demo mode (no API key) still streams — we chunk the static demo
#     answer so the UI typewriter effect works without an LLM key.
# ---------------------------------------------------------------------------

def stream_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
) -> Iterator[tuple]:
    """
    Stream a grounded answer token-by-token. See module docstring above for
    the event protocol. Always finishes with exactly one ("done", {...}) or
    ("error", "msg") event so callers can deterministically close the stream.
    """
    client = _get_openai_client()
    if client is None:
        # Demo mode: chunk the static answer so the UI still shows streaming.
        demo = _demo_answer(question, retrieved_chunks)
        for piece in _chunked(demo, size=24):
            yield ("token", piece)
        yield ("done", {
            "answer":            demo,
            "model":             "demo-mode",
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
        })
        return

    messages = _build_messages(question, retrieved_chunks, history)
    pieces: List[str] = []
    usage_total = 0
    usage_prompt = 0
    usage_completion = 0

    try:
        # stream_options.include_usage works on current OpenAI Chat Completions
        # API and gives us token counts in the final chunk. If the model/SDK
        # doesn't support it, the call still succeeds — we just get no usage.
        stream = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0,
            max_tokens=512,
            stream=True,
            stream_options={"include_usage": True},
        )
        for event in stream:
            # The final chunk with usage info has no choices.
            usage = getattr(event, "usage", None)
            if usage is not None:
                usage_total = getattr(usage, "total_tokens", 0) or 0
                usage_prompt = getattr(usage, "prompt_tokens", 0) or 0
                usage_completion = getattr(usage, "completion_tokens", 0) or 0

            if not getattr(event, "choices", None):
                continue
            delta = event.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                pieces.append(piece)
                yield ("token", piece)

    except OpenAIError as e:
        yield ("error", f"LLM streaming failed: {e}")
        return
    except Exception as e:  # network drop, etc.
        yield ("error", f"Unexpected streaming failure: {e}")
        return

    full_answer = "".join(pieces).strip()
    yield ("done", {
        "answer":            full_answer,
        "model":             settings.openai_model,
        "prompt_tokens":     usage_prompt,
        "completion_tokens": usage_completion,
        "total_tokens":      usage_total,
    })


def _chunked(text: str, size: int) -> Iterator[str]:
    """Yield fixed-size text slices so demo-mode also streams."""
    for i in range(0, len(text), size):
        yield text[i:i + size]
