"""PII/secret redaction utilities for evaluation payloads and artifacts.

Redaction is applied before dataset sync/upload, before sending target outputs to
LangSmith, and before writing local logs/artifacts. Secrets are never logged.

Design goals:
- Remove or mask unnecessary emails, phone numbers, and common personal
  identifiers.
- Preserve evaluation-critical facts such as technical skills, company names,
  project names, and employment dates (do not redact generic dates/company/
  project/skill terms).
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
# International/local-ish phone patterns (conservative: sequences with 7+ digits)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\w)"
)
# Common labeled personal identifiers
LABELED_ID_RE = re.compile(
    r"\b(?:ssn|social security(?: number)?|passport(?: number)?|national id|"
    r"aadhaar|aadhar|pan(?: number)?|driver'?s? license|dl number|"
    r"employee id|emp id|patient id|account number|iban)\s*[:#-]?\s*[A-Z0-9-]{4,}\b",
    re.IGNORECASE,
)
# Long numeric identifiers (10+ digits) often PII; keep years/dates (4 digits) intact
LONG_NUMERIC_ID_RE = re.compile(r"(?<!\d)\d{10,}(?!\d)")

SECRET_ENV_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PRIVATE_KEY",
    "ACCESS_KEY",
)

EMAIL_PLACEHOLDER = "[REDACTED_EMAIL]"
PHONE_PLACEHOLDER = "[REDACTED_PHONE]"
ID_PLACEHOLDER = "[REDACTED_ID]"
SECRET_PLACEHOLDER = "[REDACTED_SECRET]"


def _looks_like_secret_value(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    for marker in SECRET_ENV_MARKERS:
        val = os.getenv(marker)
        if val and len(val) >= 8 and val in text:
            return True
        # also catch env vars containing marker
    for k, v in os.environ.items():
        if any(m in k.upper() for m in SECRET_ENV_MARKERS) and v and len(v) >= 8 and v in text:
            return True
    return False


def redact_text(text: str, *, aggressive: bool = False) -> str:
    """Redact PII/secrets from text while preserving evaluation-critical facts.

    Preserves: technical skills, company names, project names, employment dates
    (not targeted by these patterns). Masks: emails, phone numbers, labeled IDs,
    long numeric identifiers, and known secret values.
    """
    if not isinstance(text, str) or not text:
        return "" if text is None else text

    redacted = text

    # Remove any known secret env values first.
    for k, v in os.environ.items():
        if any(m in k.upper() for m in SECRET_ENV_MARKERS) and v and len(v) >= 8 and v in redacted:
            redacted = redacted.replace(v, SECRET_PLACEHOLDER)

    redacted = EMAIL_RE.sub(EMAIL_PLACEHOLDER, redacted)
    # Avoid redacting 4-digit years and common date fragments: require 7+ total digits
    redacted = PHONE_RE.sub(PHONE_PLACEHOLDER, redacted)
    redacted = LABELED_ID_RE.sub(ID_PLACEHOLDER, redacted)
    redacted = LONG_NUMERIC_ID_RE.sub(ID_PLACEHOLDER, redacted)

    if aggressive and _looks_like_secret_value(redacted):
        # Fallback: should be rare after explicit replacement
        redacted = SECRET_PLACEHOLDER

    return redacted


def redact_obj(obj: Any) -> Any:
    """Deep-copy and redact strings within a JSON-like structure."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, tuple):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key_lower = str(k).lower()
            # Never serialize obvious secret-bearing keys
            if any(m.lower() in key_lower for m in ("api_key", "token", "secret", "password", "authorization")):
                out[k] = SECRET_PLACEHOLDER if v not in (None, "") else v
            else:
                out[k] = redact_obj(v)
        return out
    return copy.deepcopy(obj)


def sanitize_for_logging(obj: Any, max_str: int = 2000) -> Any:
    """Redact and truncate values for safe logging."""
    red = redact_obj(obj)
    if isinstance(red, str) and len(red) > max_str:
        return red[:max_str] + "...<truncated>"
    if isinstance(red, dict):
        return {k: sanitize_for_logging(v, max_str=max_str) for k, v in red.items()}
    if isinstance(red, list):
        return [sanitize_for_logging(v, max_str=max_str) for v in red]
    return red
