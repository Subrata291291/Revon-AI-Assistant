"""
Centralized Multi-LLM Provider Router for RevOn AI Assistant.

Provides sequential, production-grade fallback across configured LLM providers:
Priority: Groq (Primary) -> OpenRouter (1st fallback) -> Gemini (2nd fallback) -> OpenAI (Final fallback)

Features:
- Configurable provider priority (LLM_PROVIDER_PRIORITY)
- Bounded timeouts and resilient error classification (transient vs auth/config)
- Streaming compatibility with pre-token fallback and mid-stream error safety
- Normalized response schema and source citation preservation
- Structured observability and logging without leaking secrets or credentials
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Any

import httpx
from dotenv import load_dotenv

from langsmith_config import trace_llm

# Ensure .env is loaded reliably
_dotenv_path = Path(__file__).resolve().parent / ".env"
if _dotenv_path.is_file():
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()

logger = logging.getLogger("revon-policy-chatbot.router")


# ============================================================================
# 1. Data Models and Exceptions
# ============================================================================

@dataclass
class LLMResult:
    """Normalized response container from any LLM provider."""
    content: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    latency_seconds: float = 0.0


class LLMProviderError(Exception):
    """Base exception for provider failures."""
    def __init__(
        self,
        provider: str,
        message: str,
        status_code: int | None = None,
        is_transient: bool = True,
        retry_after: float | None = None,
    ):
        super().__init__(f"[{provider}] {message} (status: {status_code})")
        self.provider = provider
        self.message = message
        self.status_code = status_code
        self.is_transient = is_transient
        self.retry_after = retry_after


class AllProvidersFailedError(Exception):
    """Raised when all configured providers in the priority chain fail."""
    def __init__(self, failure_history: list[dict[str, Any]]):
        super().__init__("All configured LLM providers failed.")
        self.failure_history = failure_history


class MidStreamProviderError(Exception):
    """Raised when a provider disconnects or fails mid-stream after emitting content."""
    def __init__(self, provider: str, message: str):
        super().__init__(f"Provider '{provider}' failed mid-stream: {message}")
        self.provider = provider
        self.message = message


# ============================================================================
# 2. Base Provider Adapter Interface
# ============================================================================

class BaseLLMProvider(ABC):
    """Abstract interface implemented by all LLM provider adapters."""

    name: str

    def __init__(
        self,
        model: str,
        api_key: str | None,
        timeout: float = 30.0,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url

    def is_configured(self) -> bool:
        """Check whether the provider has required credentials configured."""
        return bool(self.api_key and str(self.api_key).strip())

    @trace_llm
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMResult:
        """Generate non-streaming response."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Generator[str, None, None]:
        """Generate streaming token chunks."""
        pass


# ============================================================================
# 3. Provider Implementations
# ============================================================================

class GroqProvider(BaseLLMProvider):
    """Primary Provider: Groq API with automatic standby model failover."""

    name = "groq"
    STANDBY_MODELS = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        super().__init__(model=model, api_key=api_key, timeout=timeout)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            if not self.api_key:
                raise LLMProviderError(
                    provider=self.name,
                    message="GROQ_API_KEY is not configured",
                    status_code=401,
                    is_transient=False,
                )
            self._client = Groq(api_key=self.api_key, timeout=self.timeout, max_retries=1)
        return self._client

    def _get_candidate_models(self) -> list[str]:
        models = [self.model]
        for m in self.STANDBY_MODELS:
            if m not in models:
                models.append(m)
        return models

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMResult:
        client = self._get_client()
        candidates = self._get_candidate_models()
        last_err = None

        for cand_model in candidates:
            start_time = time.time()
            try:
                kwargs: dict[str, Any] = {
                    "model": cand_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                }
                if cand_model.startswith("openai/"):
                    kwargs["reasoning_effort"] = "medium"

                completion = client.chat.completions.create(**kwargs)
                duration = time.time() - start_time
                content = completion.choices[0].message.content or ""
                finish_reason = getattr(completion.choices[0], "finish_reason", None)
                usage = getattr(completion, "usage", None)

                return LLMResult(
                    content=content.strip(),
                    provider=self.name,
                    model=cand_model,
                    finish_reason=str(finish_reason) if finish_reason else None,
                    usage=dict(usage) if usage else None,
                    latency_seconds=duration,
                )
            except Exception as e:
                last_err = e
                err_str = str(e)
                status = getattr(e, "status_code", None)
                if not status:
                    if "429" in err_str:
                        status = 429
                    elif "503" in err_str:
                        status = 503
                    elif "500" in err_str:
                        status = 500
                    elif "401" in err_str:
                        status = 401

                logger.warning(
                    "[LLM Router] Groq model '%s' failed (status: %s): %s",
                    cand_model,
                    status,
                    err_str,
                )

        raise LLMProviderError(
            provider=self.name,
            message=str(last_err),
            status_code=getattr(last_err, "status_code", 429 if "429" in str(last_err) else 500),
            is_transient=True,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Generator[str, None, None]:
        client = self._get_client()
        candidates = self._get_candidate_models()
        last_err = None

        for cand_model in candidates:
            try:
                kwargs: dict[str, Any] = {
                    "model": cand_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "stream": True,
                }
                if cand_model.startswith("openai/"):
                    kwargs["reasoning_effort"] = "medium"

                stream_resp = client.chat.completions.create(**kwargs)
                yielded = False
                for chunk in stream_resp:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        yielded = True
                if yielded:
                    return
            except Exception as e:
                last_err = e
                logger.warning(
                    "[LLM Router] Groq streaming model '%s' failed: %s",
                    cand_model,
                    e,
                )

        raise LLMProviderError(
            provider=self.name,
            message=str(last_err),
            status_code=getattr(last_err, "status_code", 429 if "429" in str(last_err) else 500),
            is_transient=True,
        )


class OpenRouterProvider(BaseLLMProvider):
    """First Fallback: OpenRouter OpenAI-compatible API."""

    name = "openrouter"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        base_url: str | None = None,
    ):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        model = model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        super().__init__(model=model, api_key=api_key, timeout=timeout, base_url=base_url)

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LLMProviderError(
                provider=self.name,
                message="OPENROUTER_API_KEY is not configured",
                status_code=401,
                is_transient=False,
            )
        return {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "HTTP-Referer": "https://revon.in",
            "X-Title": "RevOn AI Chatbot",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMResult:
        headers = self._get_headers()
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                duration = time.time() - start_time

                if resp.status_code != 200:
                    status = resp.status_code
                    is_transient = status in (429, 500, 502, 503, 504, 402, 408)
                    raise LLMProviderError(
                        provider=self.name,
                        message=f"HTTP {status}: {resp.text[:200]}",
                        status_code=status,
                        is_transient=is_transient,
                    )

                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LLMProviderError(
                        provider=self.name,
                        message="OpenRouter returned empty choices list",
                        status_code=502,
                        is_transient=True,
                    )

                content = choices[0].get("message", {}).get("content", "")
                finish_reason = choices[0].get("finish_reason")
                usage = data.get("usage")

                return LLMResult(
                    content=content.strip(),
                    provider=self.name,
                    model=self.model,
                    finish_reason=finish_reason,
                    usage=usage,
                    latency_seconds=duration,
                )
        except httpx.TimeoutException as te:
            raise LLMProviderError(
                provider=self.name,
                message=f"Request timed out after {self.timeout}s: {te}",
                status_code=504,
                is_transient=True,
            ) from te
        except httpx.RequestError as re:
            raise LLMProviderError(
                provider=self.name,
                message=f"Network connection failed: {re}",
                status_code=503,
                is_transient=True,
            ) from re

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Generator[str, None, None]:
        headers = self._get_headers()
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        status = resp.status_code
                        raise LLMProviderError(
                            provider=self.name,
                            message=f"HTTP {status} on streaming initiation",
                            status_code=status,
                            is_transient=status in (429, 500, 502, 503, 504, 402, 408),
                        )

                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line_str = line.strip()
                        if line_str.startswith("data:"):
                            data_part = line_str[5:].strip()
                            if data_part == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_part)
                                choices = chunk_json.get("choices") or []
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
        except httpx.TimeoutException as te:
            raise LLMProviderError(
                provider=self.name,
                message=f"Streaming timed out: {te}",
                status_code=504,
                is_transient=True,
            ) from te
        except httpx.RequestError as re:
            raise LLMProviderError(
                provider=self.name,
                message=f"Streaming connection failed: {re}",
                status_code=503,
                is_transient=True,
            ) from re


class GeminiProvider(BaseLLMProvider):
    """Second Fallback: Google Gemini via official google-genai SDK."""

    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        super().__init__(model=model, api_key=api_key, timeout=timeout)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            if not self.api_key:
                raise LLMProviderError(
                    provider=self.name,
                    message="GEMINI_API_KEY is not configured",
                    status_code=401,
                    is_transient=False,
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _format_gemini_payload(self, messages: list[dict[str, str]]):
        from google.genai import types

        system_instruction_parts = []
        contents = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_instruction_parts.append(content)
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                ))
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=content)]
                ))

        system_instruction = "\n\n".join(system_instruction_parts).strip() or None
        return system_instruction, contents

    def _extract_text_from_response(self, resp: Any) -> str:
        if getattr(resp, "text", None):
            return resp.text.strip()
        if getattr(resp, "candidates", None) and resp.candidates:
            cand = resp.candidates[0]
            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                texts = [p.text for p in cand.content.parts if getattr(p, "text", None)]
                if texts:
                    return "".join(texts).strip()
        return ""

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMResult:
        from google.genai import types

        client = self._get_client()
        system_instruction, contents = self._format_gemini_payload(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        start_time = time.time()
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            duration = time.time() - start_time
            text = self._extract_text_from_response(resp)

            finish_reason = None
            if getattr(resp, "candidates", None) and resp.candidates:
                finish_reason = str(getattr(resp.candidates[0], "finish_reason", None))

            return LLMResult(
                content=text,
                provider=self.name,
                model=self.model,
                finish_reason=finish_reason,
                latency_seconds=duration,
            )
        except Exception as e:
            err_str = str(e)
            status = 503
            if "429" in err_str:
                status = 429
            elif "404" in err_str:
                status = 404
            elif "401" in err_str:
                status = 401
            elif "500" in err_str:
                status = 500

            raise LLMProviderError(
                provider=self.name,
                message=err_str,
                status_code=status,
                is_transient=status in (429, 503, 500, 502, 504),
            ) from e

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Generator[str, None, None]:
        from google.genai import types

        client = self._get_client()
        system_instruction, contents = self._format_gemini_payload(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            stream_resp = client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            for chunk in stream_resp:
                token_text = self._extract_text_from_response(chunk)
                if token_text:
                    yield token_text
        except Exception as e:
            err_str = str(e)
            status = 503
            if "429" in err_str:
                status = 429
            elif "500" in err_str:
                status = 500

            raise LLMProviderError(
                provider=self.name,
                message=err_str,
                status_code=status,
                is_transient=status in (429, 503, 500, 502, 504),
            ) from e


class OpenAIProvider(BaseLLMProvider):
    """Final Fallback: Official OpenAI REST API."""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        base_url: str | None = None,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        super().__init__(model=model, api_key=api_key, timeout=timeout, base_url=base_url)

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LLMProviderError(
                provider=self.name,
                message="OPENAI_API_KEY is not configured",
                status_code=401,
                is_transient=False,
            )
        return {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMResult:
        headers = self._get_headers()
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                duration = time.time() - start_time

                if resp.status_code != 200:
                    status = resp.status_code
                    is_transient = status in (429, 500, 502, 503, 504, 408)
                    raise LLMProviderError(
                        provider=self.name,
                        message=f"HTTP {status}: {resp.text[:200]}",
                        status_code=status,
                        is_transient=is_transient,
                    )

                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LLMProviderError(
                        provider=self.name,
                        message="OpenAI returned empty choices list",
                        status_code=502,
                        is_transient=True,
                    )

                content = choices[0].get("message", {}).get("content", "")
                finish_reason = choices[0].get("finish_reason")
                usage = data.get("usage")

                return LLMResult(
                    content=content.strip(),
                    provider=self.name,
                    model=self.model,
                    finish_reason=finish_reason,
                    usage=usage,
                    latency_seconds=duration,
                )
        except httpx.TimeoutException as te:
            raise LLMProviderError(
                provider=self.name,
                message=f"Request timed out: {te}",
                status_code=504,
                is_transient=True,
            ) from te
        except httpx.RequestError as re:
            raise LLMProviderError(
                provider=self.name,
                message=f"Connection failed: {re}",
                status_code=503,
                is_transient=True,
            ) from re

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Generator[str, None, None]:
        headers = self._get_headers()
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        status = resp.status_code
                        raise LLMProviderError(
                            provider=self.name,
                            message=f"HTTP {status} on stream initiation",
                            status_code=status,
                            is_transient=status in (429, 500, 502, 503, 504),
                        )

                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line_str = line.strip()
                        if line_str.startswith("data:"):
                            data_part = line_str[5:].strip()
                            if data_part == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_part)
                                choices = chunk_json.get("choices") or []
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
        except httpx.TimeoutException as te:
            raise LLMProviderError(
                provider=self.name,
                message=f"Streaming timed out: {te}",
                status_code=504,
                is_transient=True,
            ) from te
        except httpx.RequestError as re:
            raise LLMProviderError(
                provider=self.name,
                message=f"Streaming connection failed: {re}",
                status_code=503,
                is_transient=True,
            ) from re


# ============================================================================
# 4. Central Multi-Provider Router
# ============================================================================

class LLMProviderRouter:
    """
    Central sequential router that executes providers according to priority order.
    Priority default: groq -> openrouter -> gemini -> openai.
    """

    DEFAULT_PRIORITY = ["groq", "openrouter", "gemini", "openai"]

    def __init__(
        self,
        priority: list[str] | None = None,
        timeout: float = 30.0,
        providers: dict[str, BaseLLMProvider] | None = None,
    ):
        raw_prio = os.getenv("LLM_PROVIDER_PRIORITY")
        if priority:
            self.priority = [p.strip().lower() for p in priority if p.strip()]
        elif raw_prio:
            self.priority = [p.strip().lower() for p in raw_prio.split(",") if p.strip()]
        else:
            self.priority = list(self.DEFAULT_PRIORITY)

        self.timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", str(timeout)))

        if providers:
            self.providers = providers
        else:
            self.providers = {
                "groq": GroqProvider(timeout=self.timeout),
                "openrouter": OpenRouterProvider(timeout=self.timeout),
                "gemini": GeminiProvider(timeout=self.timeout),
                "openai": OpenAIProvider(timeout=self.timeout),
            }

    def get_configured_providers(self) -> list[BaseLLMProvider]:
        """Return configured providers in strict priority sequence."""
        ordered = []
        for name in self.priority:
            p = self.providers.get(name)
            if p and p.is_configured():
                ordered.append(p)
            elif p:
                logger.debug("[LLM Router] Provider '%s' is not configured (missing key); skipping.", name)
        return ordered

    def generate_chat_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """
        Execute providers sequentially until one succeeds.
        Returns the generated text string.
        """
        result = self.execute_generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result.content

    def execute_generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMResult:
        """Execute non-streaming generation with full metadata observability."""
        providers = self.get_configured_providers()
        if not providers:
            raise AllProvidersFailedError([{
                "provider": "none",
                "error": "No LLM providers are configured with valid API keys."
            }])

        failures: list[dict[str, Any]] = []

        for attempt_idx, provider in enumerate(providers, start=1):
            logger.info(
                "[LLM Router] Attempting provider: %s (model: %s, attempt: %d/%d)",
                provider.name,
                provider.model,
                attempt_idx,
                len(providers),
            )
            start_time = time.time()
            try:
                res = provider.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                duration = time.time() - start_time
                logger.info(
                    "[LLM Router] %s generation successful in %.2fs. Final provider: %s",
                    provider.name.capitalize(),
                    duration,
                    provider.name,
                )
                return res
            except Exception as exc:
                duration = time.time() - start_time
                status = getattr(exc, "status_code", None)
                msg = str(exc)
                logger.warning(
                    "[LLM Router] %s failed: status=%s, error=%s (duration: %.2fs)",
                    provider.name.capitalize(),
                    status,
                    msg,
                    duration,
                )
                failures.append({
                    "provider": provider.name,
                    "model": provider.model,
                    "status_code": status,
                    "error": msg,
                    "duration_seconds": duration,
                })
                if attempt_idx < len(providers):
                    next_p = providers[attempt_idx]
                    logger.info("[LLM Router] Switching provider: %s", next_p.name)

        logger.error(
            "[LLM Router] All %d providers failed. Failures: %s",
            len(providers),
            json.dumps(failures),
        )
        raise AllProvidersFailedError(failures)

    @trace_llm
    def generate_chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Generator[str, None, None]:
        """
        Generate streaming tokens with pre-token fallback safety:
        - Before the first content chunk is yielded, a provider failure automatically triggers the next provider.
        - Once output has started emitting to the client, failures raise MidStreamProviderError without restarting.
        """
        providers = self.get_configured_providers()
        if not providers:
            raise AllProvidersFailedError([{
                "provider": "none",
                "error": "No LLM providers are configured with valid API keys."
            }])

        failures: list[dict[str, Any]] = []

        for attempt_idx, provider in enumerate(providers, start=1):
            logger.info(
                "[LLM Router] Attempting streaming with provider: %s (model: %s, attempt: %d/%d)",
                provider.name,
                provider.model,
                attempt_idx,
                len(providers),
            )
            start_time = time.time()

            try:
                stream_gen = provider.stream(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Peek first token before yielding to ensure connection succeeds
                try:
                    first_chunk = next(stream_gen)
                except StopIteration:
                    # Provider produced zero tokens, treat as failure
                    raise LLMProviderError(
                        provider=provider.name,
                        message="Provider stream closed without emitting any tokens",
                        status_code=502,
                        is_transient=True,
                    )

                # First token arrived! Yield it and stream the rest
                yield first_chunk

                # Stream remainder. If failure happens here, it is mid-stream.
                try:
                    for chunk in stream_gen:
                        yield chunk
                    duration = time.time() - start_time
                    logger.info(
                        "[LLM Router] %s streaming completed in %.2fs. Final provider: %s",
                        provider.name.capitalize(),
                        duration,
                        provider.name,
                    )
                    return
                except Exception as mid_err:
                    logger.error(
                        "[LLM Router] %s failed mid-stream: %s",
                        provider.name.capitalize(),
                        mid_err,
                    )
                    raise MidStreamProviderError(
                        provider=provider.name,
                        message=str(mid_err),
                    ) from mid_err

            except MidStreamProviderError:
                # Do NOT attempt another provider mid-stream to prevent corrupted/duplicate content
                raise
            except Exception as pre_token_err:
                duration = time.time() - start_time
                status = getattr(pre_token_err, "status_code", None)
                msg = str(pre_token_err)
                logger.warning(
                    "[LLM Router] %s failed before first token: status=%s, error=%s",
                    provider.name.capitalize(),
                    status,
                    msg,
                )
                failures.append({
                    "provider": provider.name,
                    "model": provider.model,
                    "status_code": status,
                    "error": msg,
                    "duration_seconds": duration,
                })
                if attempt_idx < len(providers):
                    next_p = providers[attempt_idx]
                    logger.info("[LLM Router] Switching provider: %s", next_p.name)

        raise AllProvidersFailedError(failures)

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """Legacy helper matching the single system/user prompt interface."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate_chat_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# Global singleton instance
_router_instance: LLMProviderRouter | None = None


def get_router() -> LLMProviderRouter:
    """Retrieve or create the global LLMProviderRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMProviderRouter()
    return _router_instance


def set_router(router: LLMProviderRouter | None):
    """Set or reset global router (useful for testing)."""
    global _router_instance
    _router_instance = router
