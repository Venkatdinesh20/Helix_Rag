"""
Step 8a: RAG Service — Orchestrator
-------------------------------------
Ties retrieval and generation together into a single callable.

Why have this layer?
  - The API (api.py) should only handle HTTP concerns (request/response)
  - The RAG logic (what to retrieve, how to generate) lives here
  - This makes the service independently testable without running HTTP
  - The same service could be called from a CLI, a batch job, or an API

Flow:
  ask(question)
    → retrieve top-K chunks from vector store
    → if no chunks found: return fallback response
    → generate answer from LLM using chunks as context
    → assemble and return response dict with sources + metrics
"""

import re
import time
from typing import Any, Dict, Iterator, Optional

from app.config import settings
from app.conversation_store import get_store
from app.cost_tracker import estimate_cost, get_tracker
from app.generation import generate_answer, stream_answer
from app.guardrails import check_input, redact_pii
from app.retrieval import retrieve


# Score threshold — chunks below this score are considered not relevant.
# 0.2 works well for short/resume-style docs; raise to 0.3-0.4 for stricter grounding.
MIN_RETRIEVAL_SCORE = 0.2

# ---------------------------------------------------------------------------
# Confidence buckets
#
# We expose a single qualitative label alongside the raw retrieval_score so
# the UI can render a badge without re-implementing the thresholds, and so
# every caller (HTTP, streaming, future CLI) agrees on the same rubric.
#
# Thresholds are tuned for cosine similarity from a normalised MiniLM index:
#   ≥ 0.75 → strong semantic match
#   ≥ 0.50 → plausible match, may need verification
#   <  0.50 → likely off-topic; the LLM may struggle to ground its answer
# ---------------------------------------------------------------------------
CONFIDENCE_HIGH_THRESHOLD   = 0.75
CONFIDENCE_MEDIUM_THRESHOLD = 0.50


def confidence_from_score(score: float | None) -> str:
    """Map a top-retrieval cosine score to a HIGH/MEDIUM/LOW label."""
    if score is None or score <= 0:
        return "low"
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Conversational intent detection
# These patterns match greetings, thanks, and other small-talk that has
# nothing to do with documents. We handle them instantly without touching
# the vector store or LLM — no latency, no token cost.
# ---------------------------------------------------------------------------
_GREETING_PATTERNS = re.compile(
    r"^\s*(hi+|hey+|hello+|howdy|yo+|sup|what'?s up|greetings|good\s?(morning|afternoon|evening|day))\s*[!?.]*\s*$",
    re.IGNORECASE,
)
_THANKS_PATTERNS = re.compile(
    r"^\s*(thanks?|thank\s?you|ty|thx|cheers|much\s?appreciated)\s*[!?.]*\s*$",
    re.IGNORECASE,
)
_HOWRU_PATTERNS = re.compile(
    r"^\s*(how are you|how r u|how do you do|you good|how'?s it going)\s*[!?.]*\s*$",
    re.IGNORECASE,
)
_BYE_PATTERNS = re.compile(
    r"^\s*(bye+|goodbye|see\s?ya|take care|cya|later|good\s?night)\s*[!?.]*\s*$",
    re.IGNORECASE,
)


def _conversational_response(question: str, elapsed_ms: int) -> Dict[str, Any] | None:
    """
    Return an instant response for small-talk/greetings, or None if the
    question looks like a real document query.
    """
    q = question.strip()
    if _GREETING_PATTERNS.match(q):
        answer = "Hello! 👋 I'm your document assistant. Ask me anything about the documents in this knowledge base and I'll find the answer for you."
    elif _HOWRU_PATTERNS.match(q):
        answer = "I'm doing great, thanks for asking! Ready to help you find information from the documents. What would you like to know?"
    elif _THANKS_PATTERNS.match(q):
        answer = "You're welcome! Feel free to ask more questions about the documents."
    elif _BYE_PATTERNS.match(q):
        answer = "Goodbye! Come back anytime you have questions about the documents. 👋"
    else:
        return None  # not small-talk — proceed with RAG

    return {
        "answer":          answer,
        "sources":         [],
        "retrieval_score": 0.0,
        "confidence":      "high",  # small-talk is intentionally off-RAG; mark as confident
        "latency_ms":      elapsed_ms,
        "model":           "conversational",
        "total_tokens":    0,
        "error":           None,
    }


def ask(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point: answer a user's question using RAG.

    Args:
        question   : the user's question as a plain string
        session_id : optional opaque id for multi-turn context. If omitted
                     or unknown, a brand new session is created and its id
                     is returned in the response.

    Returns:
      {
        "answer"           : "The refund policy allows...",
        "sources"          : [ { "document_name": ..., "page": ..., "chunk_id": ... } ],
        "retrieval_score"  : 0.82,         ← top chunk similarity score
        "latency_ms"       : 1450,
        "model"            : "gpt-3.5-turbo",
        "total_tokens"     : 399,
        "session_id"       : "abc123...",
        "error"            : None
      }
    """
    start_time = time.time()
    store = get_store()
    session = store.get_or_create(session_id)

    # Step -1: Input guardrails — reject too-long, empty, or prompt-injection
    # attempts before they touch retrieval/generation. This is cheap, has
    # no LLM cost, and gives the UI a clear reason code for telemetry.
    if getattr(settings, "guardrails_enabled", True):
        gr = check_input(question)
        if not gr.allowed:
            elapsed_ms = int((time.time() - start_time) * 1000)
            get_tracker().record(
                session_id=session.session_id,
                model="guardrail",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                error=False,
            )
            return {
                "answer":              gr.user_message or "Request blocked.",
                "sources":             [],
                "retrieval_score":     0.0,
                "confidence":          "low",
                "latency_ms":          elapsed_ms,
                "model":               "guardrail",
                "prompt_tokens":       0,
                "completion_tokens":   0,
                "total_tokens":        0,
                "cost_usd":            0.0,
                "session_id":          session.session_id,
                "error":               None,
                "guardrail_blocked":   True,
                "guardrail_reason":    gr.reason_code,
            }
        question = gr.sanitized

    # Step 0: Handle conversational messages before touching the vector store
    elapsed_ms = int((time.time() - start_time) * 1000)
    convo = _conversational_response(question, elapsed_ms)
    if convo is not None:
        # Even small-talk goes into memory so the LLM sees the rapport later
        store.append_turn(session.session_id, "user", question)
        store.append_turn(session.session_id, "assistant", convo["answer"])
        convo["session_id"]        = session.session_id
        convo["prompt_tokens"]     = 0
        convo["completion_tokens"] = 0
        convo["cost_usd"]          = 0.0
        get_tracker().record(
            session_id=session.session_id,
            model="conversational",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=elapsed_ms,
            error=False,
        )
        return convo

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve(question, top_k=settings.retrieval_top_k)

    # Step 2: Filter chunks below the minimum score threshold
    # Low-score chunks add noise to the prompt and can mislead the LLM
    relevant_chunks = [c for c in retrieved_chunks if c["score"] >= MIN_RETRIEVAL_SCORE]

    # Step 3: Handle empty retrieval
    if not relevant_chunks:
        elapsed_ms = int((time.time() - start_time) * 1000)
        fallback_answer = "I could not find this information in the provided documents."
        store.append_turn(session.session_id, "user", question)
        store.append_turn(session.session_id, "assistant", fallback_answer)
        get_tracker().record(
            session_id=session.session_id,
            model="no-retrieval",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=elapsed_ms,
            error=False,
        )
        return {
            "answer":            fallback_answer,
            "sources":           [],
            "retrieval_score":   0.0,
            "confidence":        "low",
            "latency_ms":        elapsed_ms,
            "model":             settings.openai_model,
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
            "cost_usd":          0.0,
            "session_id":        session.session_id,
            "error":             None,
        }

    # Step 4: Generate answer using LLM with prior conversation history
    history = session.history_for_llm()
    generation_result = generate_answer(question, relevant_chunks, history=history)

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Step 5: Assemble sources list for the response
    # Sources tell the user exactly which documents the answer came from
    sources = [
        {
            "document_name": chunk["metadata"].get("document_name"),
            "page":          chunk["metadata"].get("page"),
            "chunk_id":      chunk["chunk_id"],
            "score":         round(chunk["score"], 4),
        }
        for chunk in relevant_chunks
    ]

    # Persist the turn pair AFTER generation so a generation failure does
    # not pollute history with an empty/incorrect assistant message.
    if generation_result.get("error") is None:
        store.append_turn(session.session_id, "user", question)
        store.append_turn(session.session_id, "assistant", generation_result["answer"])

    # Output guardrail: optional PII redaction. Applied AFTER persisting
    # so the conversation memory keeps the original wording for context,
    # while the user-facing response is scrubbed.
    final_answer = generation_result["answer"]
    if getattr(settings, "pii_redaction_enabled", False):
        final_answer = redact_pii(final_answer)

    top_score = round(relevant_chunks[0]["score"], 4)
    prompt_tokens     = generation_result.get("prompt_tokens", 0)
    completion_tokens = generation_result.get("completion_tokens", 0)
    cost_usd = get_tracker().record(
        session_id=session.session_id,
        model=generation_result["model"],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=elapsed_ms,
        error=generation_result.get("error") is not None,
    )
    return {
        "answer":            final_answer,
        "sources":           sources,
        "retrieval_score":   top_score,
        "confidence":        confidence_from_score(top_score),
        "latency_ms":        elapsed_ms,
        "model":             generation_result["model"],
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      generation_result["total_tokens"],
        "cost_usd":          round(cost_usd, 6),
        "session_id":        session.session_id,
        "error":             generation_result["error"],
    }


# ---------------------------------------------------------------------------
# Streaming variant
#
# ask_stream() returns an iterator of structured events. The API layer
# serializes each event as an SSE frame. We keep the *exact same* retrieval
# + small-talk + fallback semantics as ask(), so the only behavioural
# difference visible to the client is the delivery mechanism.
#
# Event protocol (each event is a dict with a "type" field):
#   {"type": "meta",  "session_id", "sources", "retrieval_score"}
#   {"type": "token", "text": "..."}                         (repeated)
#   {"type": "done",  "latency_ms", "model", "total_tokens",
#                     "prompt_tokens", "completion_tokens", "error": None}
#   {"type": "error", "message"}                             (terminal)
#
# Exactly one of "done" or "error" is emitted last.
# ---------------------------------------------------------------------------

def ask_stream(
    question: str,
    session_id: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Streaming counterpart of ask(). See module-level docstring for protocol."""
    start_time = time.time()
    store = get_store()
    session = store.get_or_create(session_id)

    # ---- Input guardrails (Phase 2.3) ----
    # Same checks as ask(). On block we emit a single meta+token+done
    # triplet so the SSE client lifecycle terminates cleanly.
    if getattr(settings, "guardrails_enabled", True):
        gr = check_input(question)
        if not gr.allowed:
            block_msg = gr.user_message or "Request blocked."
            elapsed_ms = int((time.time() - start_time) * 1000)
            get_tracker().record(
                session_id=session.session_id,
                model="guardrail",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                error=False,
            )
            yield {
                "type":              "meta",
                "session_id":        session.session_id,
                "sources":           [],
                "retrieval_score":   0.0,
                "confidence":        "low",
                "guardrail_blocked": True,
                "guardrail_reason":  gr.reason_code,
            }
            yield {"type": "token", "text": block_msg}
            yield {
                "type":              "done",
                "latency_ms":        elapsed_ms,
                "model":             "guardrail",
                "total_tokens":      0,
                "prompt_tokens":     0,
                "completion_tokens": 0,
                "cost_usd":          0.0,
                "error":             None,
            }
            return
        question = gr.sanitized

    # ---- Small-talk fast path: emit meta + the canned answer as one token ----
    elapsed_ms = int((time.time() - start_time) * 1000)
    convo = _conversational_response(question, elapsed_ms)
    if convo is not None:
        store.append_turn(session.session_id, "user", question)
        store.append_turn(session.session_id, "assistant", convo["answer"])
        elapsed_ms = int((time.time() - start_time) * 1000)
        get_tracker().record(
            session_id=session.session_id,
            model="conversational",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=elapsed_ms,
            error=False,
        )
        yield {
            "type":            "meta",
            "session_id":      session.session_id,
            "sources":         [],
            "retrieval_score": 0.0,
            "confidence":      "high",  # small-talk: not a RAG miss
        }
        yield {"type": "token", "text": convo["answer"]}
        yield {
            "type":              "done",
            "latency_ms":        elapsed_ms,
            "model":             "conversational",
            "total_tokens":      0,
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "cost_usd":          0.0,
            "error":             None,
        }
        return

    # ---- Retrieve ----
    retrieved_chunks = retrieve(question, top_k=settings.retrieval_top_k)
    relevant_chunks = [c for c in retrieved_chunks if c["score"] >= MIN_RETRIEVAL_SCORE]

    if not relevant_chunks:
        fallback = "I could not find this information in the provided documents."
        store.append_turn(session.session_id, "user", question)
        store.append_turn(session.session_id, "assistant", fallback)
        elapsed_ms = int((time.time() - start_time) * 1000)
        get_tracker().record(
            session_id=session.session_id,
            model="no-retrieval",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=elapsed_ms,
            error=False,
        )
        yield {
            "type":            "meta",
            "session_id":      session.session_id,
            "sources":         [],
            "retrieval_score": 0.0,
            "confidence":      "low",
        }
        yield {"type": "token", "text": fallback}
        yield {
            "type":              "done",
            "latency_ms":        elapsed_ms,
            "model":             settings.openai_model,
            "total_tokens":      0,
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "cost_usd":          0.0,
            "error":             None,
        }
        return

    sources = [
        {
            "document_name": chunk["metadata"].get("document_name"),
            "page":          chunk["metadata"].get("page"),
            "chunk_id":      chunk["chunk_id"],
            "score":         round(chunk["score"], 4),
        }
        for chunk in relevant_chunks
    ]

    # Emit meta event immediately so the UI can render sources / spinner
    # while tokens are still arriving from the LLM.
    top_score = round(relevant_chunks[0]["score"], 4)
    yield {
        "type":            "meta",
        "session_id":      session.session_id,
        "sources":         sources,
        "retrieval_score": top_score,
        "confidence":      confidence_from_score(top_score),
    }

    # ---- Stream LLM tokens ----
    history = session.history_for_llm()
    final_meta: Dict[str, Any] = {}
    stream_error: Optional[str] = None
    full_answer = ""

    for kind, payload in stream_answer(question, relevant_chunks, history=history):
        if kind == "token":
            yield {"type": "token", "text": payload}
        elif kind == "done":
            final_meta = payload
            full_answer = payload.get("answer", "")
        elif kind == "error":
            stream_error = payload
            break

    if stream_error is not None:
        # Do NOT persist a half-finished assistant message — mirrors ask()'s
        # guarantee that history stays clean on failure.
        get_tracker().record(
            session_id=session.session_id,
            model=settings.openai_model,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=int((time.time() - start_time) * 1000),
            error=True,
        )
        yield {"type": "error", "message": stream_error}
        return

    # Persist turn pair only after a clean stream completion
    if full_answer:
        store.append_turn(session.session_id, "user", question)
        store.append_turn(session.session_id, "assistant", full_answer)

    elapsed_ms = int((time.time() - start_time) * 1000)
    final_model     = final_meta.get("model", settings.openai_model)
    prompt_tokens   = final_meta.get("prompt_tokens", 0)
    completion_tokens = final_meta.get("completion_tokens", 0)
    cost_usd = get_tracker().record(
        session_id=session.session_id,
        model=final_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=elapsed_ms,
        error=False,
    )

    yield {
        "type":              "done",
        "latency_ms":        elapsed_ms,
        "model":             final_model,
        "total_tokens":      final_meta.get("total_tokens", 0),
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd":          round(cost_usd, 6),
        "error":             None,
    }
