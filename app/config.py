import os

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

load_dotenv()


class Settings(BaseModel):
    app_name: str = "Production RAG Pipeline"
    app_version: str = "0.1.0"
    environment: str = os.getenv("ENVIRONMENT", "local")

    # LLM settings — read from .env so secrets are never hardcoded
    # SecretStr prevents the key from leaking into repr(), logs, or stack traces.
    # We explicitly wrap the default in SecretStr() so pydantic coerces it
    # regardless of whether validate_default is True or False.
    openai_api_key: SecretStr = SecretStr(os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Retrieval settings — tunable experiment parameters
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # Hybrid search (Phase 2.1) — BM25 + FAISS fused with Reciprocal Rank
    # Fusion. Disable to fall back to dense-only retrieval (useful for
    # A/B comparison in evals).
    hybrid_search_enabled: bool = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() in ("1", "true", "yes")
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # Cross-encoder reranker (Phase 2.2) — re-orders the top hybrid
    # candidates with a joint (query, passage) scoring model for higher
    # precision. Adds ~50-200 ms on CPU; toggle off for latency-sensitive
    # paths or for ablation studies in evals.
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidate_pool: int = int(os.getenv("RERANK_CANDIDATE_POOL", "20"))

    # Guardrails (Phase 2.3) — input validation and output safety.
    # `guardrails_enabled` is the master switch (length + prompt-injection
    # detection). `pii_redaction_enabled` is separate because some corpora
    # (e.g. a candidate's own resume) legitimately need to surface contact
    # info — off by default, opt-in for multi-tenant deployments.
    guardrails_enabled: bool = os.getenv("GUARDRAILS_ENABLED", "true").lower() in ("1", "true", "yes")
    pii_redaction_enabled: bool = os.getenv("PII_REDACTION_ENABLED", "false").lower() in ("1", "true", "yes")


settings = Settings()