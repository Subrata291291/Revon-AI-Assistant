"""Context-local retrieval capture for the real RAG workflow.

Installs a temporary, safely-restored hook around
`retrieval_service.retrieve_context` that records retrieval results only for the
active context-local sink. Default evaluation concurrency must remain 1.
"""

from __future__ import annotations

import contextlib
import contextvars
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

_active_sink: contextvars.ContextVar["RetrievalCaptureSink | None"] = contextvars.ContextVar(
    "revon_eval_retrieval_capture_sink",
    default=None,
)


@dataclass
class RetrievalCallRecord:
    question: str
    top_k: int
    result: dict[str, Any]
    chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalCaptureSink:
    calls: list[RetrievalCallRecord] = field(default_factory=list)

    @property
    def last(self) -> RetrievalCallRecord | None:
        return self.calls[-1] if self.calls else None


_SOURCE_BLOCK_RE = re.compile(
    r"\[Source\s+(?P<num>\d+)\]\s*"
    r"Filename:\s*(?P<filename>.*?)\n"
    r"Page:\s*(?P<page>.*?)\n"
    r"Content:\s*(?P<content>.*?)(?=\n\n\[Source\s+\d+\]|\Z)",
    re.DOTALL,
)


def parse_context_chunks(context: str, sources: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Parse formatted retrieval context into structured chunk records."""
    if not context:
        return []
    chunks: list[dict[str, Any]] = []
    sources = sources or []
    for match in _SOURCE_BLOCK_RE.finditer(context):
        idx = int(match.group("num"))
        filename = (match.group("filename") or "").strip() or "Unknown document"
        page_raw = (match.group("page") or "").strip()
        try:
            page_number = int(page_raw)
        except Exception:
            page_number = page_raw if page_raw else "Unknown"
        text = (match.group("content") or "").strip()
        src = sources[idx - 1] if 0 < idx <= len(sources) else {}
        score = src.get("score") if isinstance(src, dict) else None
        chunks.append(
            {
                "source_index": idx,
                "evidence_id": f"source_{idx}",
                "filename": filename,
                "page_number": page_number,
                "score": score,
                "text": text,
            }
        )
    # Fallback: if parsing failed but sources exist, create minimal records
    if not chunks and sources:
        for i, src in enumerate(sources, start=1):
            if not isinstance(src, dict):
                continue
            chunks.append(
                {
                    "source_index": i,
                    "evidence_id": f"source_{i}",
                    "filename": src.get("filename", "Unknown document"),
                    "page_number": src.get("page_number", "Unknown"),
                    "score": src.get("score"),
                    "text": "",
                }
            )
    return chunks


@contextlib.contextmanager
def retrieval_capture() -> Iterator[RetrievalCaptureSink]:
    """Capture retrieval calls in a context-local sink with safe hook restore.

    NOTE: This temporarily patches `retrieval_service.retrieve_context` (a module
    attribute). The evaluation runner must use max_concurrency=1 and this context
    manager always restores the original in a finally block.
    """
    import retrieval_service

    sink = RetrievalCaptureSink()
    token = _active_sink.set(sink)
    original = retrieval_service.retrieve_context

    def _wrapped(question: str, top_k: int = 5):
        result = original(question, top_k)
        active = _active_sink.get()
        if active is sink:
            try:
                chunks = parse_context_chunks(
                    (result or {}).get("context", "") or "",
                    (result or {}).get("sources") or [],
                )
                active.calls.append(
                    RetrievalCallRecord(
                        question=question,
                        top_k=top_k,
                        result=result or {},
                        chunks=chunks,
                    )
                )
            except Exception:
                # Never break the real workflow due to capture issues
                active.calls.append(
                    RetrievalCallRecord(question=question, top_k=top_k, result=result or {}, chunks=[])
                )
        return result

    retrieval_service.retrieve_context = _wrapped  # type: ignore[assignment]
    try:
        yield sink
    finally:
        retrieval_service.retrieve_context = original  # type: ignore[assignment]
        _active_sink.reset(token)
