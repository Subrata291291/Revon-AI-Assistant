import os
import logging

from langsmith import traceable

logger = logging.getLogger("revon-policy-chatbot.langsmith")


def is_langsmith_enabled() -> bool:
    return (
        os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


def get_langsmith_project() -> str:
    return os.getenv(
        "LANGSMITH_PROJECT",
        "revon-policy-chatbot"
    )


def trace_chat(func):
    """Reusable tracing decorator for chat pipeline functions."""
    return traceable(
        name=func.__name__,
        project_name=get_langsmith_project(),
        enabled=is_langsmith_enabled(),
    )(func)


def trace_retrieval(func):
    """Reusable tracing decorator for retrieval functions."""
    return traceable(
        name=func.__name__,
        run_type="retriever",
        project_name=get_langsmith_project(),
        enabled=is_langsmith_enabled(),
    )(func)


def trace_llm(func):
    """Reusable tracing decorator for LLM functions."""
    return traceable(
        name=func.__name__,
        run_type="llm",
        project_name=get_langsmith_project(),
        enabled=is_langsmith_enabled(),
    )(func)
