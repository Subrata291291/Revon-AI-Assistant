import os
import re
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, Callable, Iterator

try:
    from langsmith.run_helpers import get_current_run_tree, trace as ls_trace, tracing_context
except Exception:  # pragma: no cover
    get_current_run_tree = None  # type: ignore
    ls_trace = None  # type: ignore
    tracing_context = None  # type: ignore

_request_root_run: ContextVar[Any] = ContextVar("request_root_run", default=None)

try:
    from langsmith_config import get_langsmith_project, is_langsmith_enabled
except Exception:  # pragma: no cover
    def is_langsmith_enabled() -> bool:  # type: ignore
        return False

    def get_langsmith_project() -> str:  # type: ignore
        return "revon-policy-chatbot"

SYSTEM_INSTRUCTION_VERSION = "pdf-grounded-v1"

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-().]{7,}\d)(?!\d)")
_PROFILE_URL_RE = re.compile(r"https?://(?:www\.)?(?:linkedin\.com|github\.com)/\S+", re.IGNORECASE)
_SECRET_KV_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization|bearer)\b\s*[:=]\s*([^\s,;\"']+)"
)


def redact_text(text: Any, max_len: int | None = None) -> str:
    if text is None:
        return ""
    s = str(text)
    s = _EMAIL_RE.sub("[REDACTED_EMAIL]", s)
    s = _PHONE_RE.sub("[REDACTED_PHONE]", s)
    s = _PROFILE_URL_RE.sub("[REDACTED_PROFILE_URL]", s)
    s = _SECRET_KV_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", s)
    if max_len is not None and len(s) > max_len:
        return s[: max(0, max_len - 3)] + "..."
    return s


def redact_obj(obj: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "[TRUNCATED]"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, list):
        return [redact_obj(x, depth + 1) for x in obj[:50]]
    if isinstance(obj, tuple):
        return [redact_obj(x, depth + 1) for x in obj[:50]]
    if isinstance(obj, dict):
        out = {}
        for k, v in list(obj.items())[:100]:
            key = str(k)
            if re.search(r"(?i)(api[_-]?key|token|password|secret|authorization|credential)", key):
                out[key] = "[REDACTED]"
            else:
                out[key] = redact_obj(v, depth + 1)
        return out
    return redact_text(obj)


def safe_excerpt(text: Any, limit: int = 240) -> str:
    s = redact_text(text or "")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)].rstrip() + "..."


def _apply_run_updates(run: Any, kwargs: dict[str, Any]) -> None:
    if not run:
        return
    inputs = kwargs.get("inputs")
    outputs = kwargs.get("outputs")
    metadata = kwargs.get("metadata")
    tags = kwargs.get("tags")
    name = kwargs.get("name")
    error = kwargs.get("error")
    if inputs is not None and hasattr(run, "add_inputs"):
        try:
            run.add_inputs(redact_obj(inputs))
        except Exception:
            pass
    if outputs is not None:
        # Do not end runs here; let trace context managers own lifecycle. Accumulate outputs/metadata instead.
        if hasattr(run, "add_outputs"):
            try:
                run.add_outputs(redact_obj(outputs))
            except Exception:
                try:
                    if hasattr(run, "outputs") and isinstance(getattr(run, "outputs", None), dict):
                        run.outputs.update(redact_obj(outputs))
                except Exception:
                    pass
        elif hasattr(run, "outputs"):
            try:
                if isinstance(getattr(run, "outputs", None), dict):
                    run.outputs.update(redact_obj(outputs))
                else:
                    run.outputs = redact_obj(outputs)
            except Exception:
                pass
    if metadata is not None and hasattr(run, "add_metadata"):
        try:
            run.add_metadata(redact_obj(metadata))
        except Exception:
            pass
    if tags is not None and hasattr(run, "add_tags"):
        try:
            run.add_tags(list(tags))
        except Exception:
            pass
    if name is not None and hasattr(run, "name"):
        try:
            run.name = str(name)
        except Exception:
            pass
    if error is not None and hasattr(run, "error"):
        try:
            run.error = redact_text(error, max_len=500)
        except Exception:
            pass
    extra = kwargs.get("extra")
    if extra is not None and hasattr(run, "extra") and isinstance(getattr(run, "extra", None), dict):
        try:
            run.extra.update(redact_obj(extra))
        except Exception:
            pass


def safe_set_current_run(**kwargs: Any) -> None:
    try:
        if get_current_run_tree is None:
            return
        run = get_current_run_tree()
        if not run:
            run = _request_root_run.get()
        _apply_run_updates(run, kwargs)
    except Exception:
        return


def safe_set_request_root_run(**kwargs: Any) -> None:
    """Update the active chat_request root run when available (useful for finalization)."""
    try:
        run = _request_root_run.get()
        if run is None and get_current_run_tree is not None:
            run = get_current_run_tree()
        _apply_run_updates(run, kwargs)
    except Exception:
        return


@contextmanager
def timed_operation() -> Iterator[Callable[[], float]]:
    start = time.perf_counter()
    try:
        yield lambda: round((time.perf_counter() - start) * 1000.0, 2)
    finally:
        pass


@contextmanager
def trace_operation(
    name: str,
    run_type: str = "chain",
    *,
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Create a child/root trace span when enabled; never raise tracing failures.

    Propagates the entered run as parent context for @traceable-decorated calls via
    langsmith.tracing_context so child operations nest correctly (including during SSE).
    """
    if not is_langsmith_enabled() or ls_trace is None:
        yield None
        return
    run = None
    token = None
    try:
        ctx = ls_trace(
            name=name,
            run_type=run_type,  # type: ignore[arg-type]
            inputs=redact_obj(inputs or {}),
            metadata=redact_obj(metadata or {}),
            tags=tags or [],
            project_name=get_langsmith_project(),
        )
        with ctx as entered_run:
            run = entered_run
            if name == "chat_request" and run is not None:
                token = _request_root_run.set(run)
            try:
                if tracing_context is not None and run is not None:
                    with tracing_context(parent=run, project_name=get_langsmith_project()):
                        yield run
                else:
                    yield run
            finally:
                if token is not None:
                    try:
                        _request_root_run.reset(token)
                    except Exception:
                        pass
    except Exception:
        # tracing context failures must not break request execution
        try:
            if token is not None:
                _request_root_run.reset(token)
        except Exception:
            pass
        yield run


@contextmanager
def request_trace_context(
    *,
    endpoint: str,
    streaming: bool,
    question: str,
    conversation_id: str | None = None,
    top_k: int = 5,
    extra_inputs: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    inputs = {
        "endpoint": endpoint,
        "streaming": streaming,
        "question": safe_excerpt(question, 500),
        "conversation_id": conversation_id,
        "top_k": top_k,
    }
    if extra_inputs:
        inputs.update(redact_obj(extra_inputs))
    metadata = {
        "operation": "chat_request",
        "endpoint": endpoint,
        "streaming": streaming,
        "system_instruction_version": SYSTEM_INSTRUCTION_VERSION,
    }
    if extra_metadata:
        metadata.update(redact_obj(extra_metadata))
    with trace_operation("chat_request", run_type="chain", inputs=inputs, metadata=metadata) as run:
        yield run


def build_trace_sources(sources: list[dict[str, Any]] | None, reason: str) -> dict[str, Any]:
    srcs = sources or []
    return {
        "sources": [redact_obj(s) for s in srcs],
        "count": len(srcs),
        "reason": reason,
    }


def source_state(
    *,
    retrieval_performed: bool = True,
    pinecone_match_count: int = 0,
    kept_chunk_count: int = 0,
    cited_source_count: int = 0,
    retrieval_error: str | None = None,
    retrieval_not_required_reason: str | None = None,
) -> tuple[str, str]:
    if retrieval_error:
        return "retrieval_error", "Retrieval failed before usable chunks were obtained."
    if not retrieval_performed:
        return "retrieval_not_required", retrieval_not_required_reason or "Document retrieval was not required for this request."
    if pinecone_match_count <= 0:
        return "no_chunks_retrieved", "Pinecone returned no matching chunks."
    if kept_chunk_count <= 0:
        return "no_chunks_after_processing_or_gating", "Retrieved chunks were excluded by processing or relevance gating."
    if cited_source_count <= 0:
        return "retrieved_but_uncited", "Chunks were retrieved and provided as context, but the final answer cited no sources."
    return "retrieved_and_cited", "Retrieved chunks were cited in the final answer."


def grounding_state(*, has_context: bool, cited_source_count: int, abstained: bool, retrieval_performed: bool = True) -> str:
    if abstained or (retrieval_performed and not has_context):
        return "abstained"
    if cited_source_count > 0:
        return "grounded"
    if has_context:
        return "partially_grounded"
    # conversational/non-retrieval answers without document context
    return "grounded" if not retrieval_performed else "abstained"


def classify_trace_exception(exc: Exception) -> str:
    return redact_text(f"{type(exc).__name__}: {exc}", max_len=300)
