"""
llm_call.py — OpenRouter-backed LLM call helper via the openai SDK.

Returns the raw openai response object, which exposes:
  .choices[0].message.content
  .choices[0].message.tool_calls
  .usage.prompt_tokens / .usage.completion_tokens
"""
from __future__ import annotations

import os
from typing import Any


def _make_client(api_key: str | None, api_base: str | None):
    from openai import OpenAI
    return OpenAI(
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        base_url=api_base or "https://openrouter.ai/api/v1",
    )


def _make_async_client(api_key: str | None, api_base: str | None):
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        base_url=api_base or "https://openrouter.ai/api/v1",
    )


def completion(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | None = None,
    **kwargs: Any,
):
    client = _make_client(api_key, api_base)
    call_kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    if timeout is not None:
        call_kwargs["timeout"] = timeout
    return client.chat.completions.create(**call_kwargs)


async def async_completion(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | None = None,
    **kwargs: Any,
):
    client = _make_async_client(api_key, api_base)
    call_kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    if timeout is not None:
        call_kwargs["timeout"] = timeout
    return await client.chat.completions.create(**call_kwargs)
