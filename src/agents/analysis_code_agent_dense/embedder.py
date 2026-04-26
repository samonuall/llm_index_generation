"""
embedder.py — Tokenizer-aware text embedder for the dense agent.

Hits an OpenAI-compatible /v1/embeddings endpoint (e.g. vLLM, TEI) using the
same payload shape as:

    POST {endpoint_url}/embeddings
    {"model": "Qwen/Qwen3-Embedding-0.6B", "input": "hello world"}

Two responsibilities:

1. **Truncation** — every input string is tokenized with the model's own
   tokenizer (HF AutoTokenizer) and clipped to ``max_input_tokens`` BEFORE the
   request is sent. Truncation is mandatory for change #2 in the plan: the
   embedding endpoint will refuse / silently truncate inputs longer than the
   model's context window, so we do it explicitly here.

2. **Batching** — multiple inputs are packed into a single POST whenever
   possible, respecting both an item-count limit and a total-token-count limit
   per request. Failures are retried with exponential backoff.

The embedder is created from a config dict (see config.yaml) so the same code
works whether you point at a local Qwen vLLM server or any other
OpenAI-compatible embeddings endpoint.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger("analysis_code_agent_dense")


@dataclass
class EmbedderConfig:
    endpoint_url: str
    model: str
    tokenizer_id: str
    max_input_tokens: int = 8192
    safety_margin_tokens: int = 8
    batch_max_items: int = 64
    batch_max_tokens: int = 250_000
    request_timeout: float = 120.0
    max_retries: int = 3
    backoff: float = 2.0
    api_key_env: str | None = None
    normalize: bool = True
    truncate_dim: int | None = None  # optional Matryoshka truncation; None = native dim

    @classmethod
    def from_dict(cls, cfg: dict) -> "EmbedderConfig":
        return cls(
            endpoint_url=cfg["embedding_endpoint_url"].rstrip("/"),
            model=cfg["embedding_model"],
            tokenizer_id=cfg.get("embedding_tokenizer_id", cfg["embedding_model"]),
            max_input_tokens=int(cfg.get("embedding_max_input_tokens", 8192)),
            safety_margin_tokens=int(cfg.get("embedding_safety_margin_tokens", 8)),
            batch_max_items=int(cfg.get("embedding_batch_max_items", 64)),
            batch_max_tokens=int(cfg.get("embedding_batch_max_tokens", 250_000)),
            request_timeout=float(cfg.get("embedding_request_timeout", 120.0)),
            max_retries=int(cfg.get("embedding_max_retries", 3)),
            backoff=float(cfg.get("embedding_backoff", 2.0)),
            api_key_env=cfg.get("embedding_api_key_env"),
            normalize=bool(cfg.get("embedding_normalize", True)),
            truncate_dim=cfg.get("embedding_truncate_dim"),
        )


class Embedder:
    """Truncates inputs to the model's max-token budget, then batches embed calls."""

    def __init__(self, config: EmbedderConfig) -> None:
        self._cfg = config
        self._tokenizer = self._load_tokenizer(config.tokenizer_id)

        api_key = ""
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env, "")
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        # Probe the endpoint once to discover the native embedding dimension.
        # This guarantees the LanceDB schema width matches the actual vectors.
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def max_input_tokens(self) -> int:
        return self._cfg.max_input_tokens

    @property
    def dim(self) -> int:
        """Embedding dimensionality. Probes the endpoint on first access."""
        if self._dim is None:
            sample = self._embed_batch(["dim probe"])
            self._dim = len(sample[0])
            if self._cfg.truncate_dim is not None:
                self._dim = min(self._dim, int(self._cfg.truncate_dim))
        return self._dim

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Embed a list of texts. Returns (vectors, n_truncated).

        Truncation is performed per-input using the model's tokenizer. Inputs
        are then packed into batches that respect ``batch_max_items`` and
        ``batch_max_tokens`` and POSTed to the endpoint.
        """
        if not texts:
            return [], 0
        prepped, n_truncated = self._truncate_all(texts)
        vectors = self._embed_in_batches(prepped)
        return vectors, n_truncated

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (also truncated)."""
        prepped, _ = self._truncate_all([text])
        return self._embed_batch(prepped)[0]

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def _load_tokenizer(self, tokenizer_id: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for tokenizer-aware truncation. "
                "Install with: uv add transformers"
            ) from e
        return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    def _truncate_all(self, texts: list[str]) -> tuple[list[str], int]:
        """Tokenize each text and clip to ``max_input_tokens - safety_margin``.

        Returns the truncated strings and the number of inputs that were
        actually clipped (logged so the agent can see how often truncation
        kicks in).
        """
        max_tokens = max(1, self._cfg.max_input_tokens - self._cfg.safety_margin_tokens)
        out: list[str] = []
        n_truncated = 0
        for text in texts:
            ids = self._tokenizer.encode(text or "", add_special_tokens=False)
            if len(ids) > max_tokens:
                ids = ids[:max_tokens]
                out.append(self._tokenizer.decode(ids, skip_special_tokens=True))
                n_truncated += 1
            else:
                out.append(text or "")
        if n_truncated:
            logger.debug(
                "Truncated %d / %d inputs to %d tokens (model=%s)",
                n_truncated, len(texts), max_tokens, self._cfg.model,
            )
        return out, n_truncated

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text or "", add_special_tokens=False))

    # ------------------------------------------------------------------
    # Batching + transport
    # ------------------------------------------------------------------

    def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """Pack texts into batches respecting item + token limits, then embed."""
        out: list[list[float]] = []
        batch: list[str] = []
        batch_tokens = 0
        for text in texts:
            t = self._count_tokens(text)
            would_overflow = (
                len(batch) >= self._cfg.batch_max_items
                or (batch and batch_tokens + t > self._cfg.batch_max_tokens)
            )
            if would_overflow:
                out.extend(self._embed_batch(batch))
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += t
        if batch:
            out.extend(self._embed_batch(batch))
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """POST a single batch to the embeddings endpoint with retries."""
        payload: dict = {"model": self._cfg.model, "input": texts}
        url = f"{self._cfg.endpoint_url}/embeddings"

        last_exc: Exception | None = None
        for attempt in range(self._cfg.max_retries):
            try:
                with httpx.Client(timeout=self._cfg.request_timeout, headers=self._headers) as client:
                    r = client.post(url, json=payload)
                    r.raise_for_status()
                    body = r.json()
                vectors = [item["embedding"] for item in body["data"]]
                if self._cfg.truncate_dim is not None:
                    d = int(self._cfg.truncate_dim)
                    vectors = [v[:d] for v in vectors]
                if self._cfg.normalize:
                    vectors = [_l2_normalize(v) for v in vectors]
                return vectors
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last_exc = e
                logger.warning(
                    "Embedding batch (size=%d) failed on attempt %d/%d: %s",
                    len(texts), attempt + 1, self._cfg.max_retries, e,
                )
                if attempt < self._cfg.max_retries - 1:
                    time.sleep(self._cfg.backoff * (2 ** attempt))
        assert last_exc is not None
        raise last_exc


def _l2_normalize(vec: list[float]) -> list[float]:
    s = 0.0
    for x in vec:
        s += x * x
    if s <= 0.0:
        return vec
    inv = 1.0 / (s ** 0.5)
    return [x * inv for x in vec]
