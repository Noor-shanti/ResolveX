"""Provider construction for the structured M5 language-model stages."""

from __future__ import annotations

import os
from typing import Mapping


GEMINI_MODEL = "gemini-3.5-flash-lite"


def configured_provider(environ: Mapping[str, str] | None = None) -> str:
    """Return the configured provider; this project currently supports Gemini."""
    values = os.environ if environ is None else environ
    provider = values.get("LLM_PROVIDER", "gemini").strip().lower()
    if provider != "gemini":
        raise ValueError("LLM_PROVIDER must be 'gemini'")
    return provider


def configured_model_name(provider: str, environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    if provider != "gemini":
        raise ValueError("LLM provider must be 'gemini'")
    return values.get("GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL


def create_chat_model(
    *, api_key: str | None = None, provider: str | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Create the chat model used by the single structured M5 generation chain."""
    values = os.environ if environ is None else environ
    selected = (provider or configured_provider(values)).strip().lower()
    if selected != "gemini":
        raise ValueError("LLM provider must be 'gemini'")
    key = api_key or values.get("GEMINI_API_KEY", "")
    if not key.strip():
        raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = configured_model_name("gemini", values)
    return ChatGoogleGenerativeAI(
        model=model_name, api_key=key.strip(), vertexai=False, temperature=None,
        max_output_tokens=2048, max_retries=1, timeout=60,
    )
