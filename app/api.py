"""
Step 8b: FastAPI Endpoints
----------------------------
Defines the HTTP API layer. This file only handles:
  - Request validation (Pydantic models)
  - Calling the RAG service
  - Formatting HTTP responses
  - HTTP error handling

It does NOT contain any retrieval or generation logic — that lives in
rag_service.py. This separation makes each layer independently testable.

Endpoints:
  GET  /          → root info
  GET  /health    → liveness probe (used by Cloud Run, load balancers)
  POST /ask       → main RAG endpoint

Security notes:
  - Input length is capped to prevent prompt injection and cost abuse
  - Error details are not leaked to callers in production
  - Sensitive fields (API keys) never appear in responses
"""

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Iterator

import json
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.conversation_store import get_store
from app.cost_tracker import get_tracker
from app.ingestion_service import (
    MAX_UPLOAD_BYTES,
    UploadValidationError,
    ingest_upload,
)
from app.rag_service import ask, ask_stream
from app.samples import list_samples

router = APIRouter()
logger = logging.getLogger("rag_api")

# ---------------------------------------------------------------------------
# Request / Response schemas
# Pydantic validates all incoming data automatically.
# If the request doesn't match the schema, FastAPI returns HTTP 422.
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,    # cap input to prevent abuse and runaway token costs
        description="The question to answer from the document knowledge base.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Opaque conversation id for multi-turn context. "
            "Omit on the first call — the server will return a new id "
            "which the client must echo back on subsequent calls."
        ),
    )


class SourceInfo(BaseModel):
    document_name: str | None
    page: int | None
    chunk_id: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    retrieval_score: float
    confidence: str  # "high" | "medium" | "low" — derived from retrieval_score
    latency_ms: int
    model: str
    total_tokens: int
    # Phase 2.4 — cost & token breakdown surfaced per-request so the UI
    # can show running spend without an extra /stats round-trip.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    session_id: str
    # Guardrails (Phase 2.3) — present and True when input was rejected
    # by the safety layer. UI can highlight refusals distinctly from
    # low-confidence answers.
    guardrail_blocked: bool = False
    guardrail_reason: str | None = None


class ResetSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)


class ResetSessionResponse(BaseModel):
    session_id: str
    cleared: bool


class SampleQuestion(BaseModel):
    label: str
    question: str


class SamplesResponse(BaseModel):
    samples: list[SampleQuestion]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse)
def ask_question(request_body: AskRequest, http_request: Request) -> AskResponse:
    """
    Answer a question from the document knowledge base.

    The system retrieves the most relevant document chunks and generates
    a grounded answer. Sources are included in every response so answers
    are traceable and auditable.
    """
    # Retrieve the request_id set by the logging middleware so both log lines
    # (HTTP-level and application-level) share the same correlation ID
    request_id = getattr(http_request.state, "request_id", "unknown")

    result = ask(request_body.question, session_id=request_body.session_id)

    # If the RAG service returned an error, log it but return gracefully
    if result.get("error"):
        # Log the actual error internally — never expose raw errors to callers
        logger.error(
            f"[{request_id}] generation_error={result['error']} "
            f"question_length={len(request_body.question)}"
        )
        raise HTTPException(
            status_code=503,
            detail="The answer generation service is temporarily unavailable. Please try again.",
        )

    # Section 10: log all application-level fields the HTTP middleware cannot see
    chunk_ids = [s["chunk_id"] for s in result["sources"]]
    scores    = [round(s["score"], 4) for s in result["sources"]]
    logger.info(
        f"[{request_id}] /ask "
        f"session={result['session_id']} "
        f"question_length={len(request_body.question)} "
        f"top_score={result['retrieval_score']} "
        f"chunk_ids={chunk_ids} "
        f"scores={scores} "
        f"tokens={result['total_tokens']} "
        f"model={result['model']}"
    )

    return AskResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        retrieval_score=result["retrieval_score"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        model=result["model"],
        total_tokens=result["total_tokens"],
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
        cost_usd=result.get("cost_usd", 0.0),
        session_id=result["session_id"],
        guardrail_blocked=result.get("guardrail_blocked", False),
        guardrail_reason=result.get("guardrail_reason"),
    )


# ---------------------------------------------------------------------------
# Streaming endpoint (Server-Sent Events)
#
# Why SSE and not WebSockets?
#   - One-way server→client streaming is exactly the SSE use case.
#   - Plays nicely with Cloud Run / standard HTTP infrastructure.
#   - No extra dependencies; the browser native EventSource (or fetch +
#     ReadableStream) parses the wire format for free.
#
# Why POST instead of GET (the EventSource default)?
#   - We need a JSON request body (question + session_id) and we want
#     parity with /ask. The client uses fetch() + body.getReader() and
#     parses the SSE frames manually.
#
# Frame format (one frame per event):
#   event: <type>\n
#   data:  <json>\n
#   \n
# Terminal event is always either "done" or "error".
# ---------------------------------------------------------------------------

@router.post("/ask/stream")
def ask_question_stream(request_body: AskRequest, http_request: Request) -> StreamingResponse:
    """
    Stream a grounded answer using Server-Sent Events.

    Emits a `meta` event with sources, repeated `token` events with text
    deltas, and a terminal `done` (or `error`) event with timing/usage.
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    question = request_body.question
    session_id = request_body.session_id

    def event_source() -> Iterator[bytes]:
        # Track the final event so we can write a single log line summarising
        # the request (mirrors the /ask logger).
        last_done: Dict = {}
        last_error: str | None = None
        try:
            for ev in ask_stream(question, session_id=session_id):
                etype = ev.get("type", "message")
                if etype == "done":
                    last_done = ev
                elif etype == "error":
                    last_error = ev.get("message", "unknown")
                # SSE wire format. json.dumps keeps non-ASCII safe.
                payload = json.dumps(ev, ensure_ascii=False)
                yield f"event: {etype}\ndata: {payload}\n\n".encode("utf-8")
        except Exception as e:  # pragma: no cover — defensive last resort
            logger.exception(f"[{request_id}] /ask/stream crashed")
            err = json.dumps({"type": "error", "message": "internal_error"})
            yield f"event: error\ndata: {err}\n\n".encode("utf-8")
            last_error = str(e)
        finally:
            if last_error:
                logger.error(f"[{request_id}] /ask/stream error={last_error}")
            elif last_done:
                logger.info(
                    f"[{request_id}] /ask/stream "
                    f"session={session_id or 'new'} "
                    f"question_length={len(question)} "
                    f"latency_ms={last_done.get('latency_ms')} "
                    f"tokens={last_done.get('total_tokens')} "
                    f"model={last_done.get('model')}"
                )

    headers = {
        # Disable proxy/CDN buffering so tokens reach the browser immediately.
        "Cache-Control":     "no-cache",
        "X-Accel-Buffering": "no",
        "Connection":        "keep-alive",
    }
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/session/reset", response_model=ResetSessionResponse)
def reset_session(payload: ResetSessionRequest, http_request: Request) -> ResetSessionResponse:
    """
    Clear conversation history for a session.
    The next /ask call with this id will start a fresh thread.
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    cleared = get_store().reset(payload.session_id)
    logger.info(f"[{request_id}] /session/reset session={payload.session_id} cleared={cleared}")
    return ResetSessionResponse(session_id=payload.session_id, cleared=cleared)


@router.get("/samples", response_model=SamplesResponse)
def get_samples() -> SamplesResponse:
    """
    Return curated starter questions for the UI empty state.

    The list combines hand-picked prompts with one auto-derived question that
    references a real indexed document, so the chips always feel "live"
    against whatever corpus is loaded.
    """
    return SamplesResponse(samples=[SampleQuestion(**s) for s in list_samples()])


# ---------------------------------------------------------------------------
# Cost & usage statistics (Phase 2.4)
#
# Returns process-wide aggregates plus an optional per-session breakdown.
# Read-only and cheap: a single lock acquisition + dict copy.
# ---------------------------------------------------------------------------

@router.get("/stats")
def get_stats(session_id: str | None = None) -> dict:
    """
    Process-wide token / cost / latency aggregates.

    Query params:
        session_id : if provided, also returns that session's totals
                     under a `session` key (None if unknown).
    """
    tracker = get_tracker()
    payload: dict = {"global": tracker.snapshot()}
    if session_id:
        payload["session"] = tracker.session_snapshot(session_id)
    return payload


# ---------------------------------------------------------------------------
# Observability — v2
#
# `/stats` (above) reports coarse, process-wide token/cost/latency totals
# from the in-memory CostTracker. It says nothing about *quality*: how much
# of the latency is retrieval vs. rerank vs. generation, how confident the
# model was, whether an answer cited its sources, or how often we fell back
# to "no relevant context found".
#
# app/tracer.py has recorded exactly that — a full RAGTrace per /ask call,
# persisted to SQLite — since it was added, but nothing ever served it back
# out. This endpoint is that missing read path.
# ---------------------------------------------------------------------------

@router.get("/observability")
def get_observability(window: int = 100, recent: int = 20) -> dict:
    """
    Query-level RAG quality metrics, sourced from the trace store.

    Query params:
        window : number of most-recent traces to aggregate for percentiles
                 and rates (default 100)
        recent : number of most-recent individual traces to return verbatim
                 for a "recent queries" table (default 20, capped at 100)
    """
    from app.tracer import get_summary_stats, get_traces

    recent = max(0, min(recent, 100))
    summary = get_summary_stats(window=window)
    traces = get_traces(limit=recent)

    trimmed = [
        {
            "trace_id": t.get("trace_id"),
            "query": (t.get("query") or "")[:160],
            "model": t.get("model"),
            "latency_total_ms": t.get("latency_total_ms"),
            "latency_retrieval_ms": t.get("latency_retrieval_ms"),
            "latency_rerank_ms": t.get("latency_rerank_ms"),
            "latency_llm_ms": t.get("latency_llm_ms"),
            "top_score": t.get("top_score"),
            "confidence_score": t.get("confidence_score"),
            "has_citation": bool(t.get("has_citation")),
            "was_fallback": bool(t.get("was_fallback")),
            "cost_usd": t.get("cost_usd"),
            "error": t.get("error"),
            "timestamp": t.get("timestamp"),
        }
        for t in traces
    ]

    return {"summary": summary, "recent": trimmed}


# ---------------------------------------------------------------------------
# Document upload
#
# Why a small in-process rate limiter (not Redis)?
#   On the free Cloud Run tier we have one container most of the time. A naive
#   per-IP sliding window in memory is enough to stop accidental bursts and
#   trivial abuse. Real production would push this to an API gateway / Redis.
# ---------------------------------------------------------------------------

UPLOAD_RATE_WINDOW_SECONDS = 60
UPLOAD_RATE_MAX_REQUESTS = 5
_upload_buckets: Dict[str, Deque[float]] = {}
_upload_buckets_lock = threading.Lock()


def _rate_limit_upload(client_key: str) -> None:
    """Raise HTTP 429 if `client_key` has exceeded the upload rate budget."""
    now = time.monotonic()
    cutoff = now - UPLOAD_RATE_WINDOW_SECONDS
    with _upload_buckets_lock:
        bucket = _upload_buckets.setdefault(client_key, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= UPLOAD_RATE_MAX_REQUESTS:
            retry_in = int(bucket[0] + UPLOAD_RATE_WINDOW_SECONDS - now) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Too many uploads. Try again in {retry_in}s.",
                headers={"Retry-After": str(retry_in)},
            )
        bucket.append(now)


class UploadResponse(BaseModel):
    document_name: str
    original_filename: str
    bytes_written: int
    chunks_added: int
    total_chunks: int


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    http_request: Request,
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload a PDF or TXT document. The file is validated, saved under
    `data/raw/`, chunked, embedded, and appended to the live FAISS index.
    Subsequent `/ask` calls can immediately retrieve from the new content.

    Limits:
      - Max size: 10 MB
      - Allowed types: .pdf, .txt
      - Rate limit: 5 uploads / minute / client IP
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    client_ip = http_request.client.host if http_request.client else "unknown"
    _rate_limit_upload(client_ip)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    # Stream-read with a hard cap so a malicious client can't exhaust memory
    # by lying about Content-Length.
    chunks: list[bytes] = []
    bytes_read = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        bytes_read += len(chunk)
        if bytes_read > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte limit.",
            )
        chunks.append(chunk)
    payload = b"".join(chunks)

    try:
        result = ingest_upload(payload, file.filename)
    except UploadValidationError as e:
        logger.info(f"[{request_id}] /upload rejected name={file.filename} reason={e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception(f"[{request_id}] /upload failed name={file.filename}")
        raise HTTPException(
            status_code=500,
            detail="Could not process the uploaded document. Please try again.",
        )

    logger.info(
        f"[{request_id}] /upload ok name={result.document_name} "
        f"bytes={result.bytes_written} chunks_added={result.chunks_added} "
        f"total_chunks={result.total_chunks}"
    )
    return UploadResponse(
        document_name=result.document_name,
        original_filename=result.original_filename,
        bytes_written=result.bytes_written,
        chunks_added=result.chunks_added,
        total_chunks=result.total_chunks,
    )
"""
Step 8b: FastAPI Endpoints
---------------------------
Defines the HTTP API layer. This file only handles:
  - Request validation (Pydantic models)
  - Calling the RAG service
  - Formatting HTTP responses
  - HTTP error handling

It does NOT contain any retrieval or generation logic — that lives in
rag_service.py. This separation makes each layer independently testable.

Endpoints:
  GET  /          → root info
  GET  /health    → liveness probe (used by Cloud Run, load balancers)
  POST /ask       → main RAG endpoint

Security notes:
  - Input length is capped to prevent prompt injection and cost abuse
  - Error details are not leaked to callers in production
  - Sensitive fields (API keys) never appear in responses
"""

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Iterator

import json
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.conversation_store import get_store
from app.cost_tracker import get_tracker
from app.ingestion_service import (
    MAX_UPLOAD_BYTES,
    UploadValidationError,
    ingest_upload,
)
from app.rag_service import ask, ask_stream
from app.samples import list_samples

router = APIRouter()
logger = logging.getLogger("rag_api")

# ---------------------------------------------------------------------------
# Request / Response schemas
# Pydantic validates all incoming data automatically.
# If the request doesn't match the schema, FastAPI returns HTTP 422.
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,    # cap input to prevent abuse and runaway token costs
        description="The question to answer from the document knowledge base.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Opaque conversation id for multi-turn context. "
            "Omit on the first call — the server will return a new id "
            "which the client must echo back on subsequent calls."
        ),
    )


class SourceInfo(BaseModel):
    document_name: str | None
    page: int | None
    chunk_id: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    retrieval_score: float
    confidence: str  # "high" | "medium" | "low" — derived from retrieval_score
    latency_ms: int
    model: str
    total_tokens: int
    # Phase 2.4 — cost & token breakdown surfaced per-request so the UI
    # can show running spend without an extra /stats round-trip.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    session_id: str
    # Guardrails (Phase 2.3) — present and True when input was rejected
    # by the safety layer. UI can highlight refusals distinctly from
    # low-confidence answers.
    guardrail_blocked: bool = False
    guardrail_reason: str | None = None


class ResetSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)


class ResetSessionResponse(BaseModel):
    session_id: str
    cleared: bool


class SampleQuestion(BaseModel):
    label: str
    question: str


class SamplesResponse(BaseModel):
    samples: list[SampleQuestion]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse)
def ask_question(request_body: AskRequest, http_request: Request) -> AskResponse:
    """
    Answer a question from the document knowledge base.

    The system retrieves the most relevant document chunks and generates
    a grounded answer. Sources are included in every response so answers
    are traceable and auditable.
    """
    # Retrieve the request_id set by the logging middleware so both log lines
    # (HTTP-level and application-level) share the same correlation ID
    request_id = getattr(http_request.state, "request_id", "unknown")

    result = ask(request_body.question, session_id=request_body.session_id)

    # If the RAG service returned an error, log it but return gracefully
    if result.get("error"):
        # Log the actual error internally — never expose raw errors to callers
        logger.error(
            f"[{request_id}] generation_error={result['error']} "
            f"question_length={len(request_body.question)}"
        )
        raise HTTPException(
            status_code=503,
            detail="The answer generation service is temporarily unavailable. Please try again.",
        )

    # Section 10: log all application-level fields the HTTP middleware cannot see
    chunk_ids = [s["chunk_id"] for s in result["sources"]]
    scores    = [round(s["score"], 4) for s in result["sources"]]
    logger.info(
        f"[{request_id}] /ask "
        f"session={result['session_id']} "
        f"question_length={len(request_body.question)} "
        f"top_score={result['retrieval_score']} "
        f"chunk_ids={chunk_ids} "
        f"scores={scores} "
        f"tokens={result['total_tokens']} "
        f"model={result['model']}"
    )

    return AskResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        retrieval_score=result["retrieval_score"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        model=result["model"],
        total_tokens=result["total_tokens"],
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
        cost_usd=result.get("cost_usd", 0.0),
        session_id=result["session_id"],
        guardrail_blocked=result.get("guardrail_blocked", False),
        guardrail_reason=result.get("guardrail_reason"),
    )


# ---------------------------------------------------------------------------
# Streaming endpoint (Server-Sent Events)
#
# Why SSE and not WebSockets?
#   - One-way server→client streaming is exactly the SSE use case.
#   - Plays nicely with Cloud Run / standard HTTP infrastructure.
#   - No extra dependencies; the browser native EventSource (or fetch +
#     ReadableStream) parses the wire format for free.
#
# Why POST instead of GET (the EventSource default)?
#   - We need a JSON request body (question + session_id) and we want
#     parity with /ask. The client uses fetch() + body.getReader() and
#     parses the SSE frames manually.
#
# Frame format (one frame per event):
#   event: <type>\n
#   data:  <json>\n
#   \n
# Terminal event is always either "done" or "error".
# ---------------------------------------------------------------------------

@router.post("/ask/stream")
def ask_question_stream(request_body: AskRequest, http_request: Request) -> StreamingResponse:
    """
    Stream a grounded answer using Server-Sent Events.

    Emits a `meta` event with sources, repeated `token` events with text
    deltas, and a terminal `done` (or `error`) event with timing/usage.
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    question = request_body.question
    session_id = request_body.session_id

    def event_source() -> Iterator[bytes]:
        # Track the final event so we can write a single log line summarising
        # the request (mirrors the /ask logger).
        last_done: Dict = {}
        last_error: str | None = None
        try:
            for ev in ask_stream(question, session_id=session_id):
                etype = ev.get("type", "message")
                if etype == "done":
                    last_done = ev
                elif etype == "error":
                    last_error = ev.get("message", "unknown")
                # SSE wire format. json.dumps keeps non-ASCII safe.
                payload = json.dumps(ev, ensure_ascii=False)
                yield f"event: {etype}\ndata: {payload}\n\n".encode("utf-8")
        except Exception as e:  # pragma: no cover — defensive last resort
            logger.exception(f"[{request_id}] /ask/stream crashed")
            err = json.dumps({"type": "error", "message": "internal_error"})
            yield f"event: error\ndata: {err}\n\n".encode("utf-8")
            last_error = str(e)
        finally:
            if last_error:
                logger.error(f"[{request_id}] /ask/stream error={last_error}")
            elif last_done:
                logger.info(
                    f"[{request_id}] /ask/stream "
                    f"session={session_id or 'new'} "
                    f"question_length={len(question)} "
                    f"latency_ms={last_done.get('latency_ms')} "
                    f"tokens={last_done.get('total_tokens')} "
                    f"model={last_done.get('model')}"
                )

    headers = {
        # Disable proxy/CDN buffering so tokens reach the browser immediately.
        "Cache-Control":     "no-cache",
        "X-Accel-Buffering": "no",
        "Connection":        "keep-alive",
    }
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/session/reset", response_model=ResetSessionResponse)
def reset_session(payload: ResetSessionRequest, http_request: Request) -> ResetSessionResponse:
    """
    Clear conversation history for a session.
    The next /ask call with this id will start a fresh thread.
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    cleared = get_store().reset(payload.session_id)
    logger.info(f"[{request_id}] /session/reset session={payload.session_id} cleared={cleared}")
    return ResetSessionResponse(session_id=payload.session_id, cleared=cleared)


@router.get("/samples", response_model=SamplesResponse)
def get_samples() -> SamplesResponse:
    """
    Return curated starter questions for the UI empty state.

    The list combines hand-picked prompts with one auto-derived question that
    references a real indexed document, so the chips always feel "live"
    against whatever corpus is loaded.
    """
    return SamplesResponse(samples=[SampleQuestion(**s) for s in list_samples()])


# ---------------------------------------------------------------------------
# Cost & usage statistics (Phase 2.4)
#
# Returns process-wide aggregates plus an optional per-session breakdown.
# Read-only and cheap: a single lock acquisition + dict copy.
# ---------------------------------------------------------------------------

@router.get("/stats")
def get_stats(session_id: str | None = None) -> dict:
    """
    Process-wide token / cost / latency aggregates.

    Query params:
        session_id : if provided, also returns that session's totals
                     under a `session` key (None if unknown).
    """
    tracker = get_tracker()
    payload: dict = {"global": tracker.snapshot()}
    if session_id:
        payload["session"] = tracker.session_snapshot(session_id)
    return payload


# ---------------------------------------------------------------------------
# Document upload
#
# Why a small in-process rate limiter (not Redis)?
#   On the free Cloud Run tier we have one container most of the time. A naive
#   per-IP sliding window in memory is enough to stop accidental bursts and
#   trivial abuse. Real production would push this to an API gateway / Redis.
# ---------------------------------------------------------------------------

UPLOAD_RATE_WINDOW_SECONDS = 60
UPLOAD_RATE_MAX_REQUESTS = 5
_upload_buckets: Dict[str, Deque[float]] = {}
_upload_buckets_lock = threading.Lock()


def _rate_limit_upload(client_key: str) -> None:
    """Raise HTTP 429 if `client_key` has exceeded the upload rate budget."""
    now = time.monotonic()
    cutoff = now - UPLOAD_RATE_WINDOW_SECONDS
    with _upload_buckets_lock:
        bucket = _upload_buckets.setdefault(client_key, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= UPLOAD_RATE_MAX_REQUESTS:
            retry_in = int(bucket[0] + UPLOAD_RATE_WINDOW_SECONDS - now) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Too many uploads. Try again in {retry_in}s.",
                headers={"Retry-After": str(retry_in)},
            )
        bucket.append(now)


class UploadResponse(BaseModel):
    document_name: str
    original_filename: str
    bytes_written: int
    chunks_added: int
    total_chunks: int


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    http_request: Request,
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload a PDF or TXT document. The file is validated, saved under
    `data/raw/`, chunked, embedded, and appended to the live FAISS index.
    Subsequent `/ask` calls can immediately retrieve from the new content.

    Limits:
      - Max size: 10 MB
      - Allowed types: .pdf, .txt
      - Rate limit: 5 uploads / minute / client IP
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    client_ip = http_request.client.host if http_request.client else "unknown"
    _rate_limit_upload(client_ip)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    # Stream-read with a hard cap so a malicious client can't exhaust memory
    # by lying about Content-Length.
    chunks: list[bytes] = []
    bytes_read = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        bytes_read += len(chunk)
        if bytes_read > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte limit.",
            )
        chunks.append(chunk)
    payload = b"".join(chunks)

    try:
        result = ingest_upload(payload, file.filename)
    except UploadValidationError as e:
        logger.info(f"[{request_id}] /upload rejected name={file.filename} reason={e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception(f"[{request_id}] /upload failed name={file.filename}")
        raise HTTPException(
            status_code=500,
            detail="Could not process the uploaded document. Please try again.",
        )

    logger.info(
        f"[{request_id}] /upload ok name={result.document_name} "
        f"bytes={result.bytes_written} chunks_added={result.chunks_added} "
        f"total_chunks={result.total_chunks}"
    )
    return UploadResponse(
        document_name=result.document_name,
        original_filename=result.original_filename,
        bytes_written=result.bytes_written,
        chunks_added=result.chunks_added,
        total_chunks=result.total_chunks,
    )
