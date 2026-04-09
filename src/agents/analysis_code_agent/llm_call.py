"""
llm_call.py — Unified LLM call helper.

Routes gemini/* models directly through google-genai SDK.
All other models go through LiteLLM (proxy or native).

Returns a lightweight response object with:
  .choices[0].message.content  — response text
  .usage.prompt_tokens
  .usage.completion_tokens
"""
from __future__ import annotations

import os
import time
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
    """Call the appropriate backend based on model prefix."""
    if model.startswith("gemini/"):
        return _call_google_genai(model, messages, temperature)
    else:
        return _call_litellm(model, messages, temperature, api_key, api_base, **kwargs)


def _call_google_genai(model: str, messages: list[dict], temperature: float) -> _Response:
    """Call Google GenAI SDK directly."""
    from google import genai
    from google.genai import types

    gemini_model = model.split("/", 1)[1]  # "gemini/gemini-2.5-pro" → "gemini-2.5-pro"
    api_key = os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key)

    # Convert OpenAI-style messages to Google GenAI format
    system_instruction = None
    contents = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )

    resp = client.models.generate_content(
        model=gemini_model,
        contents=contents,
        config=config,
    )

    text = resp.text or ""
    usage = _Usage(
        prompt_tokens=getattr(resp.usage_metadata, "prompt_token_count", 0) or 0,
        completion_tokens=getattr(resp.usage_metadata, "candidates_token_count", 0) or 0,
    )
    return _Response(choices=[_Choice(message=_Message(content=text))], usage=usage)


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
