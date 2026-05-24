"""
Phase 2.3 — Guardrails
-----------------------
Three layers of defence around the RAG pipeline:

  1. Input guardrails — applied before retrieval:
       - Length limits (DoS protection, prompt budget control)
       - Prompt-injection detection (jailbreaks, role hijacking, exfiltration)
       - Empty / whitespace-only rejection

  2. Topic/grounding guardrails — applied between retrieval and generation:
       - If retrieval scores are uniformly low we already short-circuit in
         rag_service. The guardrail layer surfaces *why* via a structured
         reason code so the UI can show a clearer message.

  3. Output guardrails — applied after generation:
       - Redact obvious PII patterns (emails, phone numbers, SSNs) that
         could be leaked through the LLM's response.

Design principles:
  - Pure functions, no external services, no model calls — these run on
    every request and must be fast and cost-free.
  - Conservative defaults: when in doubt, allow. We surface a flag/reason
    so observability + future evals can tune thresholds, rather than
    silently blocking users.
  - Every block has a stable reason code (machine-readable) plus a
    user-facing message. The API can return both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Configurable limits — kept as module constants so they are trivially
# overridable in tests and clearly visible in code review.
# ---------------------------------------------------------------------------

MAX_QUESTION_LENGTH = 2000
MIN_QUESTION_LENGTH = 1  # after .strip()


# ---------------------------------------------------------------------------
# Prompt-injection signatures
#
# These are coarse pattern matches drawn from documented jailbreak corpora
# (DAN, "ignore previous instructions", role-hijacking, etc.). They catch
# the bulk of low-effort attacks without an ML classifier.
#
# We deliberately do NOT block all occurrences of "system" or "prompt" —
# legitimate users frequently ask about system architecture, prompt
# templates, etc. The patterns target imperative attack phrasing.
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    # "Ignore previous instructions" family
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context|messages?)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|prior|above|earlier|the)\s+(instructions?|prompts?|rules?|system)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(everything|all|previous|prior|above)\b", re.IGNORECASE),
    # Role / persona hijacking
    re.compile(r"\byou\s+are\s+now\s+(a|an|the)\s+\w+", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+\w+", re.IGNORECASE),
    re.compile(r"\bpretend\s+(to\s+be|you\s+are|that\s+you\s+are)\b", re.IGNORECASE),
    re.compile(r"\broleplay\s+as\b", re.IGNORECASE),
    # System-prompt exfiltration
    re.compile(r"\b(reveal|show|print|repeat|leak|disclose|output)\s+(me\s+)?(your\s+|the\s+)?(system\s+|initial\s+|original\s+|hidden\s+)?(prompt|instructions?|rules?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(are\s+|were\s+)?your\s+(original\s+|initial\s+|system\s+|hidden\s+)?(instructions?|prompts?|rules?)\b", re.IGNORECASE),
    # Well-known jailbreak handles
    re.compile(r"\bDAN\s+(mode|prompt)\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    # Bypass/override phrasing
    re.compile(r"\b(override|bypass|circumvent)\s+(your\s+|the\s+)?(safety|filter|guardrails?|restrictions?|policies)\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+(any\s+)?(restrictions?|filters?|safety|limitations?)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Output PII redaction
#
# Light-touch regex redaction. NOT a substitute for a real PII engine, but
# stops the most common accidental leaks (email, phone, SSN-shaped IDs).
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Phone: matches common US/INT formats; intentionally lenient.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
)
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Outcome of an input guardrail check."""
    allowed: bool
    reason_code: Optional[str] = None     # stable machine-readable id
    user_message: Optional[str] = None    # safe-to-display message
    sanitized: str = ""                   # normalised question (always set)


# ---------------------------------------------------------------------------
# Input guardrails
# ---------------------------------------------------------------------------

def check_input(question: str) -> GuardrailResult:
    """
    Validate and sanitize a user-submitted question.

    Returns a GuardrailResult. On `allowed=False` the API should refuse
    the request with a friendly message; the `reason_code` is suitable
    for logging and dashboards.
    """
    # Treat None defensively even though the API layer validates types.
    if question is None:
        return GuardrailResult(
            allowed=False,
            reason_code="empty_question",
            user_message="Please enter a question.",
            sanitized="",
        )

    sanitized = question.strip()

    # Normalise repeated whitespace inside the question to a single space.
    # This prevents trivial bypasses like "ignore\u00a0\u00a0\u00a0previous".
    sanitized = re.sub(r"\s+", " ", sanitized)

    if len(sanitized) < MIN_QUESTION_LENGTH:
        return GuardrailResult(
            allowed=False,
            reason_code="empty_question",
            user_message="Please enter a question.",
            sanitized=sanitized,
        )

    if len(sanitized) > MAX_QUESTION_LENGTH:
        return GuardrailResult(
            allowed=False,
            reason_code="too_long",
            user_message=(
                f"Your question is too long ({len(sanitized)} characters). "
                f"Please keep it under {MAX_QUESTION_LENGTH} characters."
            ),
            sanitized=sanitized[:MAX_QUESTION_LENGTH],
        )

    matched = detect_prompt_injection(sanitized)
    if matched is not None:
        return GuardrailResult(
            allowed=False,
            reason_code="prompt_injection",
            user_message=(
                "Your message looks like an attempt to override the "
                "assistant's instructions. Please rephrase as a normal "
                "question about the documents."
            ),
            sanitized=sanitized,
        )

    return GuardrailResult(
        allowed=True,
        reason_code=None,
        user_message=None,
        sanitized=sanitized,
    )


def detect_prompt_injection(text: str) -> Optional[str]:
    """
    Return the matched pattern source on the first injection hit, or None.

    Exposed so observability / unit tests can identify *which* pattern
    triggered, without parsing the user message.
    """
    if not text:
        return None
    for pattern in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


# ---------------------------------------------------------------------------
# Output guardrails
# ---------------------------------------------------------------------------

def redact_pii(text: str) -> str:
    """
    Replace email/phone/SSN-shaped substrings with redacted placeholders.

    This is intentionally over-broad: in a document-QA setting, surfacing
    a user's email back to a third party is a leak even if the text came
    from a retrieved chunk. If a downstream caller needs the raw text it
    should request it from the source chunk directly.
    """
    if not text:
        return text
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _SSN_RE.sub("[REDACTED_SSN]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def contains_pii(text: str) -> bool:
    """True iff `text` matches any PII pattern. Used by tests and metrics."""
    if not text:
        return False
    return bool(_EMAIL_RE.search(text) or _SSN_RE.search(text) or _PHONE_RE.search(text))
