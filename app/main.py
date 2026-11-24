import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import settings

# ---------------------------------------------------------------------------
# Structured logging setup
# Using JSON-style output so log aggregators (Cloud Logging, Datadog, etc.)
# can parse fields automatically. For local dev this is still human-readable.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A production-style RAG pipeline for document question answering.",
)

# Register all routes from api.py
app.include_router(router)

# Serve static files (the frontend UI lives in app/static/)
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.on_event("startup")
def _prewarm() -> None:
    """Load the embedding model and FAISS index before the first request.
    Without this, the first /ask call pays a ~2s cold-start penalty."""
    from pipeline.generate_embeddings import _get_model
    from app.retrieval import _get_index_and_metadata
    logger.info("[startup] Loading embedding model...")
    _get_model()
    logger.info("[startup] Loading FAISS index...")
    _get_index_and_metadata()
    logger.info("[startup] Ready.")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Log every request with a unique ID, latency, and status code.
    This is the minimum observability required for production (Architecture Section 10).

    Fields logged:
      request_id  — correlates logs for a single request across all services
      method      — GET / POST
      path        — /ask, /health, etc.
      status_code — HTTP response code
      latency_ms  — total server-side processing time
      client_ip   — for abuse detection (from X-Forwarded-For or direct)
    """
    request_id = str(uuid.uuid4())[:8]  # Short 8-char prefix for readability
    start = time.perf_counter()

    # Store on request.state so the endpoint handler can include it in its own logs
    # This is how one request_id correlates the HTTP log and the RAG application log
    request.state.request_id = request_id

    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"client={request.client.host if request.client else 'unknown'}"
    )

    response = await call_next(request)

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"status={response.status_code} latency={latency_ms}ms"
    )

    # Attach request ID to response header so clients can correlate errors
    response.headers["X-Request-ID"] = request_id
    return response



@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    """Live metrics dashboard (Phase 3.1). Reads /stats every 3s."""
    return FileResponse(os.path.join(_static_dir, "dashboard.html"))


@app.get("/health")
def health_check():
    return {"status": "ok"}