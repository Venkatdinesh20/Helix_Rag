# ── Stage 1: Builder ────────────────────────────────────────────────────────
# Install all Python dependencies in a separate stage.
# This keeps the final image lean — build tools are not shipped to production.
FROM python:3.12-slim AS builder

WORKDIR /app

# Install dependencies first (separate layer = cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pre-download the embedding model so cold-start does NOT block on a
# HuggingFace download (Cloud Run startup probe times out otherwise).
# We pin the cache under /app/.cache so the non-root appuser can read it.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0
RUN python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('all-MiniLM-L6-v2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
" && chmod -R a+rX /app/.cache

# Copy application source code
COPY app/ ./app/
COPY pipeline/ ./pipeline/
# Bake the prebuilt vector store into the image so the container is
# self-contained. For larger corpora override at deploy time via the
# GCS_VECTOR_STORE_URI env var (see startup.sh).
COPY vector_store/ ./vector_store/

# Install google-cloud-storage so the startup script can pull the vector store from GCS.
# We do this in the runtime stage (not builder) because it needs to run as the app user.
RUN pip install --no-cache-dir google-cloud-storage

# startup.sh: downloads vector store from GCS then starts uvicorn.
# GCS_VECTOR_STORE_URI is set at deploy time, e.g.:
#   gs://my-bucket/vector_store/
# If not set, the container expects a local vector_store/ (useful for local docker run).
COPY startup.sh ./startup.sh

# Security: run as non-root user
# Create data/raw dir and ensure appuser owns all writable paths
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --no-create-home appuser \
 && chmod +x startup.sh \
 && mkdir -p data/raw data/processed \
 && chown -R appuser:appuser data/ vector_store/
USER appuser

# Cloud Run sets PORT env variable; default to 8000 for local use
ENV PORT=8000
EXPOSE ${PORT}

# Health check — Cloud Run and load balancers use this to know if the container is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health').raise_for_status()"

# Start the FastAPI server
# --workers 1: single worker for Cloud Run (each instance handles one request at a time)
# Increase workers for VM-based deployments
# Start via startup.sh which pulls vector store from GCS then launches uvicorn
CMD ["sh", "startup.sh"]
