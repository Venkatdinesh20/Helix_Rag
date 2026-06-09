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
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" && chmod -R 755 /app/.cache

# Copy application source code
COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY vector_store/ ./vector_store/
COPY startup.sh ./startup.sh

# Security: run as non-root user
# Create data/raw dir and ensure appuser owns all writable paths
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --no-create-home appuser \
 && chmod +x startup.sh \
 && mkdir -p data/raw data/processed \
 && chown -R appuser:appuser data/ vector_store/ /app/.cache
USER appuser

# HuggingFace Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "startup.sh"]
