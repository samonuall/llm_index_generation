"""
dense_client.py — Thin httpx-based HTTP client for the dense vector server.

Mirrors ``analysis_code_agent.bm25_client.BM25Client`` so the rest of the agent
code is symbol-compatible: ``build_index``, ``retrieve``, ``batch_retrieve``,
``delete_index``, ``health``. Each method creates a fresh ``httpx.Client`` per
call to avoid stale connection issues during long agent loops.

The server (``dense_server.py``) does the heavy lifting (embedding, LanceDB
HNSW index creation, MaxP aggregation). This client just shuffles JSON.
"""
from __future__ import annotations

import functools
import time

import httpx


def _with_retry(max_attempts: int = 3, backoff: float = 2.0):
    """Decorator: on httpx errors, wait with exponential backoff and retry."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (httpx.RequestError, httpx.TimeoutException) as e:
                    last_exc = e
                except httpx.HTTPStatusError as e:
                    if e.response.status_code < 500:
                        raise
                    # Attach the response body so callers can see the server-side
                    # error message rather than just the HTTP status code.
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", e.response.text)
                    except Exception:
                        detail = e.response.text
                    last_exc = RuntimeError(
                        f"{e.request.method} {e.request.url} → {e.response.status_code}: {detail}"
                    )
                if attempt < max_attempts - 1:
                    time.sleep(backoff * (2 ** attempt))
            raise last_exc

        return wrapper

    return decorator


class DenseClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8766",
        timeout: float = 600.0,
        max_retries: int = 3,
        batch_size: int = 5_000,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._batch_size = batch_size

    def build_index(self, name: str, chunks: list, persist: bool = False) -> None:
        """Build a vector index on the server.

        For small chunk lists (<=batch_size) sends a single POST.
        For larger lists, streams chunks in batches via append/finalize so the
        server-side embedding loop sees them all without one giant payload.
        """
        if len(chunks) <= self._batch_size:
            self._build_index_single(name, chunks, persist)
        else:
            self._build_index_batched(name, chunks, persist)

    @_with_retry()
    def _build_index_single(self, name: str, chunks: list, persist: bool) -> None:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            payload = {
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
                "persist": persist,
            }
            r = client.post(f"/index/{name}/build", json=payload)
            r.raise_for_status()

    def _build_index_batched(self, name: str, chunks: list, persist: bool) -> None:
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            self._append_chunks(name, batch)
        self._finalize_index(name, persist)

    @_with_retry()
    def _append_chunks(self, name: str, chunks: list) -> None:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            payload = {
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
            }
            r = client.post(f"/index/{name}/append", json=payload)
            r.raise_for_status()

    @_with_retry()
    def _finalize_index(self, name: str, persist: bool) -> None:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            r = client.post(f"/index/{name}/finalize", json={"persist": persist})
            r.raise_for_status()

    @_with_retry()
    def retrieve(self, name: str, query: str, top_k: int = 100) -> list[dict]:
        """POST /index/{name}/retrieve -> list of {doc_id, score, rank}."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            r = client.post(
                f"/index/{name}/retrieve", json={"query": query, "top_k": top_k}
            )
            r.raise_for_status()
            return r.json()["results"]

    @_with_retry()
    def batch_retrieve(
        self, name: str, queries: list, top_k: int = 100
    ) -> list[dict]:
        """POST /index/{name}/batch_retrieve -> list of {query_id, ranked_docs}.
        queries: list of EvalQuery objects."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            payload = {
                "queries": [
                    {"query_id": q.query_id, "query_text": q.query_text}
                    for q in queries
                ],
                "top_k": top_k,
            }
            r = client.post(f"/index/{name}/batch_retrieve", json=payload)
            r.raise_for_status()
            return r.json()["results"]

    @_with_retry()
    def delete_index(self, name: str) -> None:
        """DELETE /index/{name}."""
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            r = client.delete(f"/index/{name}")
            r.raise_for_status()

    def delete_index_safe(self, name: str) -> None:
        """Delete an index, swallowing all errors (used in `finally` cleanup blocks)."""
        try:
            self.delete_index(name)
        except Exception:
            pass

    def health(self) -> bool:
        """GET /health -> True if server is up (no retry; used for polling)."""
        try:
            timeout = httpx.Timeout(connect=1.0, read=1.0, write=1.0, pool=1.0)
            with httpx.Client(
                base_url=self.base_url,
                timeout=timeout,
                trust_env=False,
            ) as client:
                r = client.get("/health")
                return r.status_code == 200
        except Exception:
            return False

    def probe_embedding(self, timeout: float = 120.0) -> int:
        """GET /embed_health -> embedding dim, or raise RuntimeError with details.

        Uses a longer timeout because a cold vLLM worker may take 30–60 s to
        load the model before it can respond to its first embed request.
        """
        t = httpx.Timeout(connect=timeout, read=timeout, write=30.0, pool=5.0)
        with httpx.Client(base_url=self.base_url, timeout=t, trust_env=False) as client:
            r = client.get("/embed_health")
            if r.status_code != 200:
                detail = ""
                try:
                    detail = r.json().get("detail", r.text)
                except Exception:
                    detail = r.text
                raise RuntimeError(
                    f"Embedding endpoint probe failed (HTTP {r.status_code}): {detail}"
                )
            return r.json()["embedding_dim"]
