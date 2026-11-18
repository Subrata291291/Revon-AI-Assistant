import json
import os
import random
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768

# Defaults chosen to stay below Gemini free-tier embed_content quota (100 RPM/model/project/user by default).
DEFAULT_EMBEDDING_BATCH_SIZE = 20
DEFAULT_EMBEDDING_RPM = 90
DEFAULT_EMBEDDING_MAX_RETRIES = 5
DEFAULT_EMBEDDING_BASE_BACKOFF_SECONDS = 1.0
DEFAULT_EMBEDDING_MAX_BACKOFF_SECONDS = 60.0
DEFAULT_EMBEDDING_REQUEST_TIMEOUT_SECONDS = 60.0

# Local daily request-budget protection (estimate only; Google quota/usage is authoritative).
DEFAULT_EMBEDDING_DAILY_BUDGET_ENABLED = True
DEFAULT_EMBEDDING_DAILY_REQUEST_BUDGET = 1000
DEFAULT_EMBEDDING_DAILY_BUDGET_WINDOW_ID = "default"


class EmbeddingRateLimitError(Exception):
    """Retryable Gemini quota/rate-limit exhaustion for embedding requests."""


class EmbeddingRequestError(Exception):
    """Non-retryable or exhausted embedding request failure."""


class EmbeddingDailyBudgetExhaustedError(EmbeddingRequestError):
    """Non-retryable local daily embedding request-budget exhaustion (local estimate only)."""


class EmbeddingDailyBudgetStateError(EmbeddingRequestError):
    """Non-retryable local daily budget state persistence/recovery error (fail-closed)."""


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except Exception:
            value = default
    if value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        value = default
    else:
        try:
            value = float(str(raw).strip())
        except Exception:
            value = default
    if value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _get_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key is None:
        return None
    key = str(key).strip()
    return key or None


def get_embedding_batch_size() -> int:
    # Keep batches efficient while allowing operators to tune for API/payload limits.
    return _env_int("GEMINI_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE, minimum=1, maximum=100)


def get_embedding_rpm_limit() -> int:
    # Configurable request-rate limit; default stays below common 100 RPM free-tier quota.
    return _env_int("GEMINI_EMBEDDING_RPM", DEFAULT_EMBEDDING_RPM, minimum=1, maximum=10000)


def get_embedding_max_retries() -> int:
    return _env_int("GEMINI_EMBEDDING_MAX_RETRIES", DEFAULT_EMBEDDING_MAX_RETRIES, minimum=0, maximum=20)


def get_embedding_daily_budget_enabled() -> bool:
    return _env_bool("GEMINI_EMBEDDING_DAILY_BUDGET_ENABLED", DEFAULT_EMBEDDING_DAILY_BUDGET_ENABLED)


def get_embedding_daily_request_budget() -> int:
    return _env_int("GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET", DEFAULT_EMBEDDING_DAILY_REQUEST_BUDGET, minimum=1, maximum=10_000_000)


def get_embedding_daily_budget_window_id() -> str:
    raw = os.getenv("GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID")
    if raw is None:
        return DEFAULT_EMBEDDING_DAILY_BUDGET_WINDOW_ID
    value = str(raw).strip()
    return value or DEFAULT_EMBEDDING_DAILY_BUDGET_WINDOW_ID


def get_embedding_daily_budget_state_path() -> str:
    raw = os.getenv("GEMINI_EMBEDDING_DAILY_BUDGET_STATE_PATH")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    # Default is a local filesystem path; for production accounting use a durable persistent volume/path
    # (ephemeral/temp storage can lose state across restarts/deployments). For distributed deployments,
    # a shared transactional store (e.g. PostgreSQL/Supabase) is required instead of process-local file state.
    return os.path.join(tempfile.gettempdir(), "revon-gemini-embedding-daily-quota.json")


def get_embedding_daily_budget_recovery_mode() -> str:
    raw = os.getenv("GEMINI_EMBEDDING_DAILY_BUDGET_RECOVERY_MODE")
    if raw is None:
        return ""
    return str(raw).strip().lower()


_client: genai.Client | None = None
_client_api_key: str | None = None
_rate_limit_lock = threading.Lock()
_next_request_at = 0.0
_daily_budget_lock = threading.Lock()


def _get_client(api_key: str | None = None) -> genai.Client:
    global _client, _client_api_key
    resolved = (api_key.strip() if isinstance(api_key, str) and api_key.strip() else _get_api_key())
    if not resolved:
        raise ValueError("GEMINI_API_KEY is missing")
    if _client is None or _client_api_key != resolved:
        _client = genai.Client(api_key=resolved)
        _client_api_key = resolved
    return _client


def _reset_rate_limiter_for_tests() -> None:
    global _next_request_at
    with _rate_limit_lock:
        _next_request_at = 0.0


def _acquire_request_slot(
    rpm: int | None = None,
    *,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Enforce a minimum interval between embedding API requests (RPM limiter)."""
    global _next_request_at
    effective_rpm = rpm if rpm is not None else get_embedding_rpm_limit()
    effective_rpm = max(1, int(effective_rpm))
    min_interval = 60.0 / float(effective_rpm)
    wait = 0.0
    with _rate_limit_lock:
        now = now_fn()
        scheduled = max(now, _next_request_at)
        _next_request_at = scheduled + min_interval
        wait = max(0.0, scheduled - now)
    if wait > 0:
        sleep_fn(wait)


def _fresh_daily_budget_state(window_id: str, *, recovered: bool = False) -> dict[str, Any]:
    state: dict[str, Any] = {
        "window_id": window_id,
        "used_requests": 0,
        "updated_at_epoch": time.time(),
        "note": "local_estimate_google_authoritative",
    }
    if recovered:
        state["recovered_from_state_error"] = True
        state["recovered_at_epoch"] = time.time()
    return state


def _read_daily_budget_state(path: str, window_id: str) -> dict[str, Any]:
    """Load persisted local daily-budget state.

    Fail-closed: if an existing state file is corrupted or unreadable, do not reset used_requests
    and do not allow embedding requests until an operator explicitly recovers the state.
    Missing state file is initialized as a fresh local estimate for the configured window.
    """
    if not os.path.exists(path):
        return _fresh_daily_budget_state(window_id)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as exc:
        raise EmbeddingDailyBudgetStateError(
            "Local Gemini embedding daily-budget state file exists but is unreadable; refusing to issue embedding requests "
            "to avoid unsafe quota-accounting reset (local estimate only; Google quota is authoritative). "
            f"state_path={path}. Operator recovery required: restore/fix file permissions or replace with a valid state file, "
            "or explicitly reinitialize by setting GEMINI_EMBEDDING_DAILY_BUDGET_RECOVERY_MODE=reset for a controlled recovery run "
            "after verifying local accounting impact. "
            f"Cause: {exc}"
        ) from exc

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("state root must be a JSON object")
        if "window_id" not in data or "used_requests" not in data:
            raise ValueError("state must contain window_id and used_requests")
        stored_window = str(data.get("window_id", ""))
        used_raw = data.get("used_requests", 0)
        if isinstance(used_raw, bool) or not isinstance(used_raw, int):
            # Be strict to avoid ambiguity/corruption masking.
            raise ValueError("used_requests must be an integer")
        if used_raw < 0:
            raise ValueError("used_requests must be non-negative")
        used = used_raw

        # Operator-controlled window reset mechanism: if configured window differs, start fresh (no time-based reset).
        if stored_window != window_id:
            return _fresh_daily_budget_state(window_id)
        return {
            "window_id": window_id,
            "used_requests": used,
            "updated_at_epoch": data.get("updated_at_epoch"),
            "note": "local_estimate_google_authoritative",
        }
    except EmbeddingDailyBudgetStateError:
        raise
    except Exception as exc:
        recovery_mode = get_embedding_daily_budget_recovery_mode()
        if recovery_mode == "reset":
            state = _fresh_daily_budget_state(window_id, recovered=True)
            state["recovery_reason"] = "corrupt_or_invalid_state"
            _write_daily_budget_state(path, state)
            return state
        raise EmbeddingDailyBudgetStateError(
            "Local Gemini embedding daily-budget state file exists but is corrupted or invalid; refusing to issue embedding requests "
            "to avoid unsafe quota-accounting reset (local estimate only; Google quota is authoritative). "
            f"state_path={path}. Operator recovery required: repair the state file contents, or explicitly reinitialize by setting "
            "GEMINI_EMBEDDING_DAILY_BUDGET_RECOVERY_MODE=reset for a controlled recovery run after verifying local accounting impact "
            "(this explicit recovery resets the local counter for the configured window). "
            f"Cause: {exc}"
        ) from exc


def _write_daily_budget_state(path: str, state: dict[str, Any]) -> None:
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    payload = json.dumps(state, separators=(",", ":"), sort_keys=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp_path, path)


def _reserve_daily_budget_request_attempt(
    *,
    enabled: bool | None = None,
    budget: int | None = None,
    window_id: str | None = None,
    state_path: str | None = None,
) -> None:
    """Count and persist one embedding API request attempt before it is issued.

    Local estimate only; Google quota/usage remains authoritative. Never auto-resets by assumed quota reset time.
    Operator-controlled reset: change GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID (or configured window) to start a new window.
    """
    resolved_enabled = get_embedding_daily_budget_enabled() if enabled is None else bool(enabled)
    if not resolved_enabled:
        return

    resolved_budget = get_embedding_daily_request_budget() if budget is None else max(1, int(budget))
    resolved_window = get_embedding_daily_budget_window_id() if window_id is None else (str(window_id).strip() or DEFAULT_EMBEDDING_DAILY_BUDGET_WINDOW_ID)
    resolved_path = get_embedding_daily_budget_state_path() if state_path is None else str(state_path)

    with _daily_budget_lock:
        state = _read_daily_budget_state(resolved_path, resolved_window)
        used = int(state.get("used_requests", 0) or 0)
        if used + 1 > resolved_budget:
            raise EmbeddingDailyBudgetExhaustedError(
                "Local Gemini embedding daily request budget exhausted (local estimate only; Google quota is authoritative). "
                f"used_requests={used}, configured_budget={resolved_budget}, window_id={resolved_window}. "
                "No embedding API request was issued. "
                "To continue after confirming quota/reset with Google, use the operator-controlled reset mechanism by changing "
                "GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID (or increasing the configured local budget) and retry; "
                "ingestion is resumable and no document is marked completed/active on this stop."
            )

        state["used_requests"] = used + 1
        state["window_id"] = resolved_window
        state["updated_at_epoch"] = time.time()
        state["note"] = "local_estimate_google_authoritative"
        _write_daily_budget_state(resolved_path, state)


def _get_status_code(error: Exception) -> int | None:
    for attr in ("code", "status_code", "status"):
        val = getattr(error, attr, None)
        if isinstance(val, bool):
            continue
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            try:
                return int(val.strip())
            except Exception:
                pass
    resp = getattr(error, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            val = getattr(resp, attr, None)
            if isinstance(val, int):
                return val
    return None


def _iter_error_details(error: Exception) -> Iterator[Any]:
    details = getattr(error, "details", None)
    if details is None:
        return iter(())
    if isinstance(details, (list, tuple)):
        return iter(details)
    return iter((details,))


def _parse_retry_delay_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Common forms: "27s", "27.5s", "0.027s"
        m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*s", s, flags=re.IGNORECASE)
        if m:
            try:
                return max(0.0, float(m.group(1)))
            except Exception:
                return None
        try:
            return max(0.0, float(s))
        except Exception:
            return None
    if isinstance(value, dict):
        # google.rpc.RetryInfo / protobuf-json variants
        for k in ("retryDelay", "retry_delay", "delay"):
            if k in value:
                parsed = _parse_retry_delay_seconds(value.get(k))
                if parsed is not None:
                    return parsed
        seconds = value.get("seconds")
        nanos = value.get("nanos", 0)
        if seconds is not None:
            try:
                sec = float(seconds)
                nano = float(nanos or 0)
                return max(0.0, sec + (nano / 1_000_000_000.0))
            except Exception:
                return None
        return None
    # Protobuf/object forms
    for attr in ("retry_delay", "retryDelay", "delay"):
        if hasattr(value, attr):
            parsed = _parse_retry_delay_seconds(getattr(value, attr))
            if parsed is not None:
                return parsed
    if hasattr(value, "seconds") or hasattr(value, "nanos"):
        try:
            sec = float(getattr(value, "seconds", 0) or 0)
            nano = float(getattr(value, "nanos", 0) or 0)
            if sec or nano:
                return max(0.0, sec + (nano / 1_000_000_000.0))
        except Exception:
            return None
    return None


def extract_retry_delay_seconds(error: Exception) -> float | None:
    # Direct/common attributes
    for attr in ("retry_delay", "retryDelay", "retry_after", "retryAfter"):
        if hasattr(error, attr):
            parsed = _parse_retry_delay_seconds(getattr(error, attr))
            if parsed is not None:
                return parsed

    # google.rpc.RetryInfo in error.details
    for detail in _iter_error_details(error):
        if isinstance(detail, dict):
            t = str(detail.get("@type") or detail.get("type") or "")
            if "RetryInfo" in t or "retryDelay" in detail or "retry_delay" in detail:
                parsed = _parse_retry_delay_seconds(detail.get("retryDelay") or detail.get("retry_delay") or detail)
                if parsed is not None:
                    return parsed
        else:
            type_name = type(detail).__name__
            if "RetryInfo" in type_name or hasattr(detail, "retry_delay") or hasattr(detail, "retryDelay"):
                parsed = _parse_retry_delay_seconds(detail)
                if parsed is not None:
                    return parsed

    # Response headers fallback
    resp = getattr(error, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if headers:
        try:
            ra = headers.get("retry-after") or headers.get("Retry-After")
            parsed = _parse_retry_delay_seconds(ra)
            if parsed is not None:
                return parsed
        except Exception:
            pass

    # Last resort: parse "retryDelay" / "retry_delay" token from message text
    msg = str(error)
    m = re.search(r"retry[_ ]?delay[\"']?\s*[:=]\s*[\"']?([0-9]+(?:\.[0-9]+)?s?)", msg, flags=re.IGNORECASE)
    if m:
        return _parse_retry_delay_seconds(m.group(1))
    return None


def _is_resource_exhausted(error: Exception) -> bool:
    status = _get_status_code(error)
    msg = str(error).upper()
    if status == 429:
        return True
    if "RESOURCE_EXHAUSTED" in msg:
        return True
    # Some SDKs embed status in text
    if "429" in msg and ("RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg or "RATE" in msg):
        return True
    # Inspect details for canonical status/reason
    for detail in _iter_error_details(error):
        if isinstance(detail, dict):
            if str(detail.get("status", "")).upper() == "RESOURCE_EXHAUSTED":
                return True
            if str(detail.get("reason", "")).upper() in {"RATE_LIMIT_EXCEEDED", "RESOURCE_EXHAUSTED"}:
                return True
            txt = json_safe_str(detail).upper()
            if "RESOURCE_EXHAUSTED" in txt:
                return True
        else:
            txt = str(detail).upper()
            if "RESOURCE_EXHAUSTED" in txt:
                return True
    return False


def json_safe_str(value: Any) -> str:
    try:
        import json as _json

        return _json.dumps(value, default=str)
    except Exception:
        return str(value)


def _is_retryable_server_error(error: Exception) -> bool:
    status = _get_status_code(error)
    if status in {500, 502, 503, 504}:
        return True
    # Transport/timeouts may surface as generic exceptions in SDKs; be conservative and only retry obvious transient cases.
    name = type(error).__name__.lower()
    msg = str(error).lower()
    transient_markers = (
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "connection error",
        "server disconnected",
        "deadline exceeded",
        "unavailable",
    )
    if status is None and any(m in msg for m in transient_markers):
        # Avoid misclassifying explicit non-retryable client semantics.
        if "invalid argument" in msg or "permission denied" in msg or "unauthenticated" in msg:
            return False
        return True
    if "timeout" in name and status is None:
        return True
    return False


def _is_non_retryable_client_error(error: Exception) -> bool:
    if isinstance(error, (EmbeddingDailyBudgetExhaustedError, EmbeddingDailyBudgetStateError)):
        return True
    status = _get_status_code(error)
    if status in {400, 401, 403, 404, 409, 422}:
        # 409 conflict is typically not a transient quota issue for embeddings.
        return True
    msg = str(error).upper()
    non_retryable_markers = (
        "INVALID_ARGUMENT",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "FAILED_PRECONDITION",
        "NOT_FOUND",
        "API_KEY",
    )
    if any(m in msg for m in non_retryable_markers) and not _is_resource_exhausted(error):
        return True
    return False


def _classify_embedding_error(error: Exception) -> tuple[str, float | None]:
    """Return (kind, retry_delay_seconds_if_known). kind in {'rate_limit','retryable','non_retryable'}."""
    if isinstance(error, (EmbeddingDailyBudgetExhaustedError, EmbeddingDailyBudgetStateError)):
        return "non_retryable", None
    if _is_resource_exhausted(error):
        return "rate_limit", extract_retry_delay_seconds(error)
    if _is_non_retryable_client_error(error):
        return "non_retryable", None
    if _is_retryable_server_error(error):
        return "retryable", extract_retry_delay_seconds(error)
    # Default: do not retry unknown errors to avoid infinite/non-safe retries.
    return "non_retryable", None


def _compute_backoff_delay(
    *,
    attempt: int,
    kind: str,
    retry_delay_seconds: float | None,
    base: float,
    cap: float,
    rng: random.Random,
) -> float:
    # Exponential backoff with jitter; honor RetryInfo delay when provided (especially for quota exhaustion).
    exp = min(cap, base * (2 ** max(0, attempt)))
    jitter = rng.uniform(0.0, min(exp, 1.0) if exp > 0 else 0.0)
    backoff = min(cap, exp + jitter)
    if retry_delay_seconds is not None:
        # Add small jitter to spread synchronized retries while respecting server guidance.
        suggested = min(cap, max(0.0, float(retry_delay_seconds)) + rng.uniform(0.0, 1.0))
        return max(backoff, suggested) if kind == "rate_limit" else suggested
    return backoff


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for a single text (with rate limiting and bounded retry)."""
    vectors = _generate_embedding_batch([text])
    return vectors[0]


def _generate_embedding_batch(
    texts: list[str],
    *,
    client: genai.Client | None = None,
    rpm: int | None = None,
    max_retries: int | None = None,
    base_backoff_seconds: float | None = None,
    max_backoff_seconds: float | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
) -> list[list[float]]:
    """
    Generate separate embeddings for multiple texts in one API call (batchEmbedContents).

    Handles 429 RESOURCE_EXHAUSTED separately with RetryInfo-aware bounded backoff+jitter,
    retries transient server errors, and avoids retrying non-retryable errors.
    Enforces configurable local daily request budget (persistent estimate; Google quota authoritative)
    before each API request attempt, including retries.
    """
    if not texts:
        return []

    resolved_client = client or _get_client()
    resolved_rpm = get_embedding_rpm_limit() if rpm is None else max(1, int(rpm))
    resolved_max_retries = get_embedding_max_retries() if max_retries is None else max(0, int(max_retries))
    resolved_base = _env_float("GEMINI_EMBEDDING_BASE_BACKOFF_SECONDS", DEFAULT_EMBEDDING_BASE_BACKOFF_SECONDS, minimum=0.1)
    if base_backoff_seconds is not None:
        resolved_base = max(0.1, float(base_backoff_seconds))
    resolved_cap = _env_float("GEMINI_EMBEDDING_MAX_BACKOFF_SECONDS", DEFAULT_EMBEDDING_MAX_BACKOFF_SECONDS, minimum=1.0)
    if max_backoff_seconds is not None:
        resolved_cap = max(1.0, float(max_backoff_seconds))
    resolved_rng = rng or random.Random()

    contents = [types.Content(parts=[types.Part(text=text)]) for text in texts]

    for attempt in range(resolved_max_retries + 1):
        _acquire_request_slot(resolved_rpm, now_fn=now_fn, sleep_fn=sleep_fn)
        # Count every Gemini embedding API request attempt, including retries, before issuing the request.
        # Stops before the next request would exceed the configured local budget (estimate; Google authoritative).
        _reserve_daily_budget_request_attempt()
        try:
            response = resolved_client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
            )

            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch. Expected {len(texts)}, received {len(embeddings)}."
                )

            vectors: list[list[float]] = []
            for emb in embeddings:
                values = getattr(emb, "values", None)
                if values is None:
                    raise RuntimeError("Embedding response missing vector values.")
                vectors.append(list(values))
            return vectors

        except Exception as error:
            kind, retry_delay_seconds = _classify_embedding_error(error)
            is_last = attempt >= resolved_max_retries

            if kind == "non_retryable" or is_last:
                if kind == "rate_limit":
                    raise EmbeddingRateLimitError(
                        f"Gemini embedding rate limit/quota exhausted after {attempt + 1} attempt(s): {error}"
                    ) from error
                raise

            delay = _compute_backoff_delay(
                attempt=attempt,
                kind=kind,
                retry_delay_seconds=retry_delay_seconds,
                base=resolved_base,
                cap=resolved_cap,
                rng=resolved_rng,
            )
            label = "rate limit/quota (RESOURCE_EXHAUSTED)" if kind == "rate_limit" else "transient error"
            print(
                f"Gemini embedding {label}. Retry {attempt + 1}/{resolved_max_retries} "
                f"after {delay:.2f}s. Cause: {error}"
            )
            sleep_fn(delay)

    raise EmbeddingRateLimitError("Gemini embedding failed after retries.")


def iter_embeddings(
    chunks: list[dict],
    *,
    batch_size: int | None = None,
    rpm: int | None = None,
    max_retries: int | None = None,
    client: genai.Client | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
) -> Iterator[list[dict]]:
    """Yield embedded chunk batches incrementally (for resumable ingest/upsert flows)."""
    if not chunks:
        return
    resolved_batch_size = get_embedding_batch_size() if batch_size is None else max(1, int(batch_size))
    total_chunks = len(chunks)
    for start in range(0, total_chunks, resolved_batch_size):
        batch = chunks[start : start + resolved_batch_size]
        batch_number = (start // resolved_batch_size) + 1
        total_batches = (total_chunks + resolved_batch_size - 1) // resolved_batch_size
        print(
            f"Generating embeddings: Batch {batch_number}/{total_batches} ({len(batch)} chunks)"
        )
        texts = [chunk["text"] for chunk in batch]
        vectors = _generate_embedding_batch(
            texts,
            client=client,
            rpm=rpm,
            max_retries=max_retries,
            sleep_fn=sleep_fn,
            now_fn=now_fn,
            rng=rng,
        )
        embedded_batch: list[dict] = []
        for chunk, vector in zip(batch, vectors, strict=True):
            embedded_batch.append(
                {
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "embedding": vector,
                }
            )
        yield embedded_batch


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """Generate embeddings for document chunks using batched requests with rate limiting/retries."""
    if not chunks:
        return []
    embedded_chunks: list[dict] = []
    total_chunks = len(chunks)
    for embedded_batch in iter_embeddings(chunks):
        embedded_chunks.extend(embedded_batch)

    if len(embedded_chunks) != total_chunks:
        raise RuntimeError(
            f"Final embedding count mismatch. Expected {total_chunks}, received {len(embedded_chunks)}."
        )
    return embedded_chunks


# Backward-compatible module-level aliases (configurable defaults resolve at call time where applicable).
EMBEDDING_BATCH_SIZE = DEFAULT_EMBEDDING_BATCH_SIZE
MAX_RETRIES = DEFAULT_EMBEDDING_MAX_RETRIES
api_key = _get_api_key()

try:
    # Preserve eager initialization behavior when key is configured; allows tests to patch client attribute.
    client = _get_client(api_key) if api_key else None
except Exception:
    client = None
