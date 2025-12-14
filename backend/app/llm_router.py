from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import settings


@dataclass
class LLMChoice:
    text: str
    finish_reason: str | None = None


def _openai_client(*, provider: str):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(f"openai_sdk_missing: {e}")

    # Provider mapping:
    # - openai: OpenAI directly
    # - openrouter: OpenAI-compatible API via OpenRouter
    if provider == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY missing")
        return OpenAI(api_key=settings.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    # default: OpenAI
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")

    if settings.OPENAI_BASE_URL:
        return OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)

    return OpenAI(api_key=settings.OPENAI_API_KEY)


def chat(
    *,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1200,
) -> LLMChoice:
    client = _openai_client(provider=provider)

    # OpenAI SDK handles timeouts via client config; we keep it simple.
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    return LLMChoice(text=(choice.message.content or "").strip(), finish_reason=getattr(choice, "finish_reason", None))


def safe_provider_and_model(ai_provider: str | None, ai_model: str | None) -> Tuple[str, str]:
    """Normalize provider/model names and apply defaults."""

    provider = (ai_provider or "openai").strip().lower()
    model = (ai_model or settings.OPENAI_MODEL).strip()

    # Normalize common provider labels from client spec
    if provider in {"gpt", "openai"}:
        provider = "openai"
    if provider in {"openrouter", "openrouter.ai"}:
        provider = "openrouter"
    if provider in {"claude", "gemini"}:
        # Prefer OpenRouter for non-OpenAI models unless user configured a base_url.
        if settings.OPENROUTER_API_KEY:
            provider = "openrouter"

    return provider, model
