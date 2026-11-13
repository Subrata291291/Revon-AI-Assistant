import os
import logging
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv

from llm_router import (
    get_router,
    LLMProviderRouter,
    LLMResult,
    LLMProviderError,
    AllProvidersFailedError,
    MidStreamProviderError,
)

logger = logging.getLogger("revon-policy-chatbot.groq")

# Ensure .env is loaded reliably regardless of working directory
_dotenv_path = Path(__file__).resolve().parent / ".env"
if _dotenv_path.is_file():
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def generate_chat_response(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1500,
    model: str | None = None,
) -> str:
    """
    Generate an LLM response using the centralized Multi-LLM Provider Router:
    Priority: Groq (Primary) -> OpenRouter -> Gemini -> OpenAI
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")

    return get_router().generate_chat_response(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_chat_stream(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1500,
    model: str | None = None,
) -> Generator[str, None, None]:
    """
    Generate an LLM streaming response using the centralized Multi-LLM Provider Router:
    Priority: Groq (Primary) -> OpenRouter -> Gemini -> OpenAI
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")

    return get_router().generate_chat_stream(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_response(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Generate an LLM response given a single system and user prompt pair.
    Routes sequentially through the Multi-LLM Provider Router.
    """
    if not system_prompt.strip():
        raise ValueError("System prompt cannot be empty")

    if not user_prompt.strip():
        raise ValueError("User prompt cannot be empty")

    return get_router().generate_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
