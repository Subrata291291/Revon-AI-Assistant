"""Isolated LLM-as-a-judge for real evaluation runs.

This module is intentionally separate from the production LLM provider/fallback
router (`llm_router.py` / `groq_service.py`) and must not be used by production
runtime paths. Judge configuration is recorded in experiment metadata (without
secrets).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import httpx

JudgeFn = Callable[[str, str], dict[str, Any]]


@dataclass
class JudgeConfig:
    provider: str
    base_url: str | None
    model: str | None
    temperature: float = 0.0
    timeout_seconds: float = 30.0
    enabled: bool = False

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "isolated_from_production_router": True,
        }


def load_judge_config_from_env() -> JudgeConfig:
    provider = (os.getenv("EVAL_JUDGE_PROVIDER") or "none").strip().lower()
    base_url = os.getenv("EVAL_JUDGE_BASE_URL")
    model = os.getenv("EVAL_JUDGE_MODEL")
    enabled_flag = (os.getenv("EVAL_JUDGE_ENABLED") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    enabled = enabled_flag and provider not in ("", "none") and bool(model)
    return JudgeConfig(
        provider=provider or "none",
        base_url=base_url,
        model=model,
        temperature=float(os.getenv("EVAL_JUDGE_TEMPERATURE", "0")),
        timeout_seconds=float(os.getenv("EVAL_JUDGE_TIMEOUT_SECONDS", "30")),
        enabled=enabled,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("judge response is not text")
    text = text.strip()
    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # fenced or embedded object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("could not parse JSON object from judge response")


class OpenAICompatibleJudge:
    """Minimal isolated judge client using an OpenAI-compatible chat API.

    Requires env: EVAL_JUDGE_API_KEY (or provider-specific key if set),
    EVAL_JUDGE_BASE_URL, EVAL_JUDGE_MODEL. Never uses production llm_router.
    """

    def __init__(self, config: JudgeConfig | None = None):
        self.config = config or load_judge_config_from_env()
        self.api_key = os.getenv("EVAL_JUDGE_API_KEY") or ""

    def is_available(self) -> bool:
        return bool(self.config.enabled and self.api_key and self.config.model and self.config.base_url)

    def __call__(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("isolated judge is not configured/available")
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = (
            (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
            if isinstance(data, dict)
            else None
        )
        if not content:
            raise ValueError("judge returned empty content")
        return _extract_json_object(content)


def create_judge(config: JudgeConfig | None = None) -> JudgeFn | None:
    cfg = config or load_judge_config_from_env()
    if not cfg.enabled or cfg.provider in ("", "none"):
        return None
    judge = OpenAICompatibleJudge(cfg)
    if not judge.is_available():
        return None
    return judge
