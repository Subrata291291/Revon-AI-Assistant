
"""Shared LangSmith client configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


def create_langsmith_client() -> Client:
    """Load backend environment and create an authenticated LangSmith client."""

    backend_dir = Path(__file__).resolve().parents[2]
    env_file = backend_dir / ".env"

    # Preserve environment variables already explicitly configured.
    load_dotenv(dotenv_path=env_file, override=False)

    api_key = os.getenv("LANGSMITH_API_KEY")
    endpoint = os.getenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com",
    )

    if not api_key:
        raise RuntimeError(
            "LANGSMITH_API_KEY is missing. Check backend/.env"
        )

    return Client(
        api_key=api_key,
        api_url=endpoint,
    )
