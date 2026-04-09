"""
llm_call.py — Unified LLM call helper.

All models go through LiteLLM for unified tool-call support.
gemini/* models are routed directly to Google (api_base forced to None).

Returns a lightweight response object with:
  .choices[0].message.content  — response text
  .usage.prompt_tokens
  .usage.completion_tokens
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _Message:
    content: str = ""


@dataclass
class _Choice:
    message: _Message = field(default_factory=_Message)


@dataclass
class _Response:
    choices: list[_Choice] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)


def completion(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs: Any,
) -> _Response:
    """Route all models through LiteLLM for unified tool-call support."""
    if model.startswith("gemini/"):
        # Gemini must go directly to Google — always use Google key, never the proxy or OpenAI key
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        api_base = None
    return _call_litellm(model, messages, temperature, api_key, api_base, **kwargs)



def _call_litellm(
    model: str,
    messages: list[dict],
    temperature: float,
    api_key: str | None,
    api_base: str | None,
    **kwargs: Any,
) -> _Response:
    """Call LiteLLM (handles OpenAI-compatible proxies and other providers)."""
    import litellm

    litellm.drop_params = True
    resp = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        api_key=api_key,
        api_base=api_base,
        **kwargs,
    )
    return resp  # LiteLLM response already has .choices[0].message.content and .usage
