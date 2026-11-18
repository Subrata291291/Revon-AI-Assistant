"""Real RevOn RAG workflow evaluation target (additive, evaluation-only).

Runs the existing `answer_conversational_question` workflow without modifying
production modules. Ensures per-example isolation:
- unique conversation_id per example
- dedicated evaluation-only conversation scope (never touches production users'
  conversation history)
- context-local retrieval capture with safe restore
- child traces preserve SDK experiment/project context for correct parent-child
  association when provided, with a safe non-production fallback when not
- explicit guard against unsafe in-process global state swaps

NOTE: This target temporarily patches module-level references in
`retrieval_service` (conversation manager, retrieval capture hook). It must run
in a dedicated evaluation process/context. `max_concurrency=1` alone does not
prevent overlap with production traffic in the same process.
"""

from __future__ import annotations

import os
import uuid
from contextlib import nullcontext
from typing import Any

from conversation_service import ConversationManager
from langsmith.run_helpers import get_current_run_tree, get_tracing_context, tracing_context

from .capture import retrieval_capture
from .redaction import redact_obj

PRODUCTION_PROJECT = "revon-policy-chatbot"
DEFAULT_EVAL_TRACE_PROJECT = "revon-policy-chatbot-eval"
EVAL_PROJECT_PREFIX = "revon-policy-chatbot-eval"


def _allow_in_process_risk(allow_override: bool | None = None) -> bool:
    if allow_override is True:
        return True
    return os.getenv("REVON_EVAL_ALLOW_IN_PROCESS_RISK") == "1"


def _assert_safe_execution_context(allow_in_process_risk: bool | None = None) -> None:
    """Explicit guard for global module-state swaps used by evaluation capture.

    The target swaps `retrieval_service.conversation_manager` for the duration of
    a call. This is only safe in a dedicated evaluation process (or explicitly
    acknowledged isolated test context). It must not run inside a process that
    may serve production traffic.
    """
    if _allow_in_process_risk(allow_in_process_risk):
        return
    raise RuntimeError(
        "Unsafe evaluation execution context: target performs global module-state "
        "swaps (conversation manager/capture hooks) and must run in a dedicated "
        "evaluation process. Run via the evaluation CLI in a separate process, or "
        "set REVON_EVAL_ALLOW_IN_PROCESS_RISK=1 only for explicitly isolated "
        "non-production contexts (e.g. unit tests)."
    )


def _normalize_cited_sources(sources: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = []
    for s in sources or []:
        if not isinstance(s, dict):
            continue
        out.append(
            {
                "filename": s.get("filename"),
                "page_number": s.get("page_number", s.get("page")),
                "score": s.get("score"),
            }
        )
    return out


def _is_safe_eval_project(project_name: str | None) -> bool:
    """Return True if project_name is an allowed non-production evaluation project."""
    if not project_name:
        return False
    if project_name == PRODUCTION_PROJECT:
        return False
    # Allow experiment projects (prefix-<uuid8>) and the base eval/fallback project.
    return project_name.startswith(EVAL_PROJECT_PREFIX)


def _resolve_trace_project(trace_project_name: str | None) -> tuple[str, bool]:
    """Resolve effective tracing project without breaking SDK experiment association.

    Returns (project_name, force_context):
    - If an SDK/eval tracing context is already active (e.g. LangSmith evaluate
      target run in experiment project `revon-policy-chatbot-eval-<uuid>`), preserve
      it and do not override the tracing context (force_context=False).
    - Otherwise, return a safe non-production fallback project and require setting it
      in tracing context (force_context=True) to prevent leakage to production.
    """
    current = None
    try:
        current = get_current_run_tree()
    except Exception:
        current = None
    try:
        ctx = get_tracing_context() or {}
    except Exception:
        ctx = {}

    sdk_project = (getattr(current, "session_name", None) if current else None) or ctx.get("project_name")

    if _is_safe_eval_project(sdk_project):
        # Preserve SDK-provided experiment/parent context for correct association.
        return str(sdk_project), False

    fallback = trace_project_name or os.getenv("REVON_EVAL_TRACE_PROJECT") or DEFAULT_EVAL_TRACE_PROJECT
    if not _is_safe_eval_project(fallback):
        fallback = DEFAULT_EVAL_TRACE_PROJECT
    # No safe SDK experiment context available (or context points to production):
    # enforce isolated fallback. Association requires SDK experiment context.
    return fallback, True


def _fallback_tracing_context(project_name: str):
    """Create a tracing_context that sets only project routing, preserving other fields.

    Used only when no safe SDK experiment context is available. `tracing_context`
    replaces provided fields, so pass through the current context to avoid dropping
    unrelated state while enforcing a non-production project.
    """
    try:
        ctx = get_tracing_context() or {}
    except Exception:
        ctx = {}
    return tracing_context(
        project_name=project_name,
        tags=ctx.get("tags"),
        metadata=ctx.get("metadata"),
        parent=ctx.get("parent"),
        enabled=ctx.get("enabled"),
        client=ctx.get("client"),
        replicas=ctx.get("replicas"),
        distributed_parent_id=ctx.get("distributed_parent_id"),
    )


def run_rag_target(
    inputs: dict[str, Any],
    *,
    top_k: int = 5,
    redact: bool = True,
    trace_project_name: str | None = None,
    allow_in_process_risk: bool | None = None,
) -> dict[str, Any]:
    """Execute the real RAG workflow and return normalized evaluation outputs.

    Args:
        inputs: expects `question` (str) and optional `example_id`.
        top_k: retrieval depth (default matches production chat path).
        redact: redact PII/secrets in returned payloads (recommended).
        trace_project_name: optional fallback LangSmith tracing project for child
            traces when no SDK experiment context is available. Defaults to
            REVON_EVAL_TRACE_PROJECT or revon-policy-chatbot-eval. Must remain
            isolated from production. When running under LangSmith evaluate, the
            SDK-provided experiment project/context is preserved for correct
            parent-child/example association and this fallback is not forced.
        allow_in_process_risk: explicit override for dedicated-process guard.
            If None, checks REVON_EVAL_ALLOW_IN_PROCESS_RISK env var.
    """
    import retrieval_service

    _assert_safe_execution_context(allow_in_process_risk=allow_in_process_risk)

    question = str((inputs or {}).get("question") or "").strip()
    if not question:
        raise ValueError("inputs.question is required")

    example_id = str((inputs or {}).get("example_id") or "example")
    conversation_id = f"eval-{example_id}-{uuid.uuid4()}"

    effective_trace_project, force_fallback_context = _resolve_trace_project(trace_project_name)
    # Safety invariant: evaluation traces must never target the production project.
    if not _is_safe_eval_project(effective_trace_project):
        raise RuntimeError(
            f"Unsafe evaluation trace project: {effective_trace_project!r}. "
            "Production isolation cannot be guaranteed."
        )

    # Dedicated evaluation-only conversation scope; never modifies/clears/deletes
    # production users' conversation history. We only swap the module reference
    # used by retrieval_service and restore it in finally.
    eval_manager = ConversationManager()
    original_manager = retrieval_service.conversation_manager
    retrieval_service.conversation_manager = eval_manager

    try:
        trace_ctx = (
            _fallback_tracing_context(effective_trace_project)
            if force_fallback_context
            else nullcontext()
        )
        with trace_ctx:
            with retrieval_capture() as sink:
                result = retrieval_service.answer_conversational_question(
                    question=question,
                    conversation_id=conversation_id,
                    client_history=None,
                    top_k=top_k,
                )

        last_retrieval = sink.last
        retrieved_chunks = list(last_retrieval.chunks) if last_retrieval else []
        retrieval_result = (last_retrieval.result if last_retrieval else {}) or {}
        retrieved_sources = retrieval_result.get("sources") or []
        retrieved_context = retrieval_result.get("context") or ""
        total_retrieved_matches = retrieval_result.get("total_matches", len(retrieved_sources))

        output = {
            "question": question,
            "example_id": example_id,
            "conversation_id": conversation_id,
            "final_answer": (result or {}).get("answer", ""),
            "intent": (result or {}).get("intent"),
            "is_policy_query": (result or {}).get("is_policy_query", False),
            "cited_sources": _normalize_cited_sources((result or {}).get("sources") or []),
            "cited_total_matches": (result or {}).get("total_matches", 0),
            "retrieved_chunks": retrieved_chunks,
            "retrieved_sources": retrieved_sources,
            "retrieval_scores": [s.get("score") for s in retrieved_sources if isinstance(s, dict)],
            "retrieved_context": retrieved_context,
            "total_retrieved_matches": total_retrieved_matches,
            "workflow": "answer_conversational_question",
            "top_k": top_k,
            "trace_project_name": effective_trace_project,
        }
        return redact_obj(output) if redact else output
    finally:
        retrieval_service.conversation_manager = original_manager


def langsmith_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """LangSmith-compatible target wrapper (SDK-verified simplified signature).

    The LangSmith SDK (0.12.6) invokes the wrapped/traceable target with
    `langsmith_extra` containing the experiment project, `reference_example_id`,
    client, metadata, and `on_end` callback. Those are handled by the traceable
    wrapper; this function preserves that context by not overriding the active
    tracing project when a safe SDK experiment context is present. For non-
    experiment contexts it falls back to REVON_EVAL_TRACE_PROJECT (isolated from
    production). Does not mutate global env.
    """
    top_k = int((inputs or {}).get("top_k") or 5)
    return run_rag_target(inputs, top_k=top_k, redact=True)
