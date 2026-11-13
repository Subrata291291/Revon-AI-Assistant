import os
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv
from groq import Groq

# Ensure .env is loaded reliably regardless of working directory
_dotenv_path = Path(__file__).resolve().parent / ".env"
if _dotenv_path.is_file():
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing")

client = Groq(
    api_key=GROQ_API_KEY,
    timeout=60.0,
    max_retries=2
)


def generate_chat_response(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> str:
    """
    Generate an LLM response from a full conversational message list.

    Each message should follow the schema: {"role": "system"|"user"|"assistant", "content": str}.
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        reasoning_effort="medium"
    )

    response = completion.choices[0].message.content

    if not response:
        raise RuntimeError("Groq returned an empty response")

    return response.strip()


def generate_chat_stream(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> Generator[str, None, None]:
    """
    Generate an LLM streaming response yielding text tokens progressively.

    Each message should follow the schema: {"role": "system"|"user"|"assistant", "content": str}.
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        reasoning_effort="medium",
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate_response(
    system_prompt: str,
    user_prompt: str
) -> str:
    """
    Generate an LLM response given a single system and user prompt pair.
    Maintained for backward compatibility.
    """
    if not system_prompt.strip():
        raise ValueError("System prompt cannot be empty")

    if not user_prompt.strip():
        raise ValueError("User prompt cannot be empty")

    return generate_chat_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
