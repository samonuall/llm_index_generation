"""
dense_server.py - FastAPI dense-retrieval index server backed by a Qwen3
embedding endpoint with persistent LanceDB storage.

Each named index is a LanceDB table with schema:

    chunk_id  : string   (globally unique chunk id)
    doc_id    : string   (parent document id)
    text      : string   (chunk text that was embedded)
    metadata  : string   (JSON-serialized chunk.metadata dict)
    vector    : fixed_size_list<float32, dim>  (L2-normalized embedding)

Tables are stored under the LanceDB URI given by --persist-dir, so every
hypothesis / iteration index (e.g. `current_iter0`, `hyp_iter3_H2`,
`harness_eval_iter5`) lives on disk and can be re-opened later.

Chunk and query embeddings are produced in batches by calling an upstream
embedding endpoint with ``embedding_batch_size`` texts per HTTP request.
Two wire formats are supported (``--embedding-format``):

  * ``openai`` — POST {endpoint}/embeddings, body {"model", "input": [...]},
    response ``data[i].embedding`` (OpenAI / vLLM compatible).
  * ``simple`` — POST {endpoint} directly, body {"texts": [...]},
    response ``{"embeddings": [[...], ...]}``.

Usage:
    python dense_server.py \
        --port 8765 \
        --persist-dir .dense_cache \
        --embedding-endpoint http://host:8001/v1 \
        --embedding-model Qwen/Qwen3-Embedding-8B \
        --embedding-api-key EMPTY \
        --embedding-format openai
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any

import httpx
import lancedb
import numpy as np
import pyarrow as pa
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Dense Index Server (Qwen3 + LanceDB)")

# ---- LanceDB connection (populated at startup) ----

_db: lancedb.DBConnection | None = None
_persist_dir: pathlib.Path | None = None

# Per-table in-memory set of chunk_ids seen during this process. Used to
# dedup append() calls so retried HTTP requests do not create duplicate rows.
_known_ids: dict[str, set[str]] = {}

# ---- Embedding client (populated at startup) ----

# Format of the upstream embedding endpoint:
#   "openai" — POST {base}/embeddings  body={"model", "input": [...]}   resp.data[i].embedding
#   "simple" — POST {base}             body={"texts": [...]}            resp["embeddings"]
_embed_format: str = "openai"
_embed_client: OpenAI | None = None      # used when _embed_format == "openai"
_embed_endpoint: str = ""                 # raw URL used when _embed_format == "simple"
_embed_api_key: str = "EMPTY"
_embed_timeout: float = 120.0
_embed_model: str = ""
_embed_batch_size: int = 32
_embed_max_chars: int = 20000
_embed_max_tokens: int = 0          # 0 = disabled; otherwise token-level truncation cap
_query_instruction: str = ""

# Lazily-initialized tiktoken encoder used when _embed_max_tokens > 0.
_token_encoder = None


def _get_token_encoder():
    """Return a tiktoken encoder, lazily initialized on first use.

    Tries the exact embedding model first (handles OpenAI ada/3-small/3-large),
    falls back to cl100k_base, which is a reasonable approximation for most
    OpenAI-compatible endpoints (incl. vLLM). If tiktoken itself is unavailable
    we return None and the caller falls back to char-level truncation.
    """
    global _token_encoder
    if _token_encoder is not None:
        return _token_encoder
    try:
        import tiktoken
    except Exception:
        return None
    try:
        _token_encoder = tiktoken.encoding_for_model(_embed_model)
    except Exception:
        try:
            _token_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _token_encoder = None
    return _token_encoder


# ---- Request / response models ----

class ChunkIn(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = {}


class BuildRequest(BaseModel):
    chunks: list[ChunkIn]
    persist: bool = True  # LanceDB is always persistent; flag retained for API compat


class AppendChunksRequest(BaseModel):
    chunks: list[ChunkIn]


class FinalizeRequest(BaseModel):
    persist: bool = True


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 100


class QueryIn(BaseModel):
    query_id: str
    query_text: str


class BatchRetrieveRequest(BaseModel):
    queries: list[QueryIn]
    top_k: int = 100


# ---- Embedding helpers ----

def _prep_text(text: str) -> str:
    """Truncate text before sending to the embedding endpoint.

    Order of operations:
      1. Char-level cap (fast; primarily to keep HTTP payloads reasonable).
      2. Token-level cap (only when _embed_max_tokens > 0). This is the one
         that matters for OpenAI-compatible endpoints with a hard input-token
         limit (e.g. 8192 for text-embedding-3-*, or some vLLM deployments of
         Qwen3-Embedding configured with a short max_model_len). If tiktoken
         cannot tokenize, we fall back to an aggressive char-based estimate
         (~3 chars/token) so we still stay under the model's limit.
    """
    if _embed_max_chars and len(text) > _embed_max_chars:
        text = text[:_embed_max_chars]
    if _embed_max_tokens and _embed_max_tokens > 0:
        enc = _get_token_encoder()
        if enc is not None:
            try:
                tokens = enc.encode(text, disallowed_special=())
            except TypeError:
                tokens = enc.encode(text)
            if len(tokens) > _embed_max_tokens:
                text = enc.decode(tokens[:_embed_max_tokens])
        else:
            approx_chars = _embed_max_tokens * 3
            if len(text) > approx_chars:
                text = text[:approx_chars]
    return text


def _format_query(text: str) -> str:
    """Prepend task instruction to queries per Qwen3-Embedding convention."""
    if _query_instruction:
        return f"Instruct: {_query_instruction}\nQuery: {text}"
    return text


def _embed_call_openai(batch: list[str]) -> list[list[float]]:
    """OpenAI-compatible call: POST {base}/embeddings with {"model", "input"}."""
    if _embed_client is None:
        raise RuntimeError("OpenAI embedding client not initialized")
    resp = _embed_client.embeddings.create(model=_embed_model, input=batch)
    return [d.embedding for d in resp.data]


def _embed_call_simple(batch: list[str]) -> list[list[float]]:
    """Simple endpoint: POST {url} with {"texts": batch}, expect {"embeddings": [...]}.

    Mirrors the reference client shape:
        requests.post(API_URL, json={"texts": batch}, timeout=120)
        response.json()["embeddings"]
    """
    if not _embed_endpoint:
        raise RuntimeError("Simple embedding endpoint URL not configured")
    headers: dict[str, str] = {}
    if _embed_api_key and _embed_api_key.upper() != "EMPTY":
        headers["Authorization"] = f"Bearer {_embed_api_key}"
    r = httpx.post(
        _embed_endpoint,
        json={"texts": batch},
        timeout=_embed_timeout,
        headers=headers or None,
    )
    r.raise_for_status()
    data = r.json()
    if "embeddings" not in data:
        raise RuntimeError(
            f"Simple embedding response missing 'embeddings' key (got keys={list(data.keys())[:8]})"
        )
    return data["embeddings"]


def _embed_batch(texts: list[str], is_query: bool, max_retries: int = 3) -> np.ndarray:
    """Embed a list of texts via the configured embedding endpoint, returning
    a (n, dim) float32 array of L2-normalized vectors.

    Format is controlled by ``_embed_format`` (``openai`` or ``simple``).
    Texts are always sent in batches of ``_embed_batch_size`` per HTTP request,
    which is the main source of throughput when embedding tens of thousands of
    chunks.
    """
    prepped = [
        _format_query(_prep_text(t)) if is_query else _prep_text(t)
        for t in texts
    ]

    out_vecs: list[list[float]] = []
    n = len(prepped)
    n_batches = (n + _embed_batch_size - 1) // _embed_batch_size if n else 0
    if n_batches:
        endpoint_desc = (
            f"{_embed_endpoint} (simple)"
            if _embed_format == "simple"
            else f"{_embed_client.base_url if _embed_client else '?'} (openai /embeddings)"
        )
        print(
            f"[server] Embedding {n} {'queries' if is_query else 'chunks'} "
            f"in {n_batches} batch(es) of up to {_embed_batch_size} — via {endpoint_desc}"
        )

    call_fn = _embed_call_simple if _embed_format == "simple" else _embed_call_openai

    for i in range(0, n, _embed_batch_size):
        batch = prepped[i : i + _embed_batch_size]
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                out_vecs.extend(call_fn(batch))
                break
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (2 ** attempt))
        else:
            endpoint_str = (
                _embed_endpoint
                if _embed_format == "simple"
                else (str(_embed_client.base_url) if _embed_client else "unknown")
            )
            raise RuntimeError(
                "Embedding call failed after "
                f"{max_retries} attempts (format={_embed_format}, endpoint={endpoint_str}, "
                f"model={_embed_model}): {last_exc}"
            )

    arr = np.asarray(out_vecs, dtype=np.float32)
    # L2-normalize so cosine similarity equals dot product. LanceDB's "cosine"
    # metric already normalizes internally, but we normalize here as well so
    # scores are comparable regardless of the chosen metric.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


# ---- LanceDB helpers ----

def _make_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("metadata", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
    ])


def _rows_from_chunks(chunks: list[dict], vecs: np.ndarray) -> list[dict]:
    return [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "text": c["text"],
            "metadata": json.dumps(c.get("metadata", {}) or {}),
            "vector": vecs[i].tolist(),
        }
        for i, c in enumerate(chunks)
    ]


def _table_names() -> list[str]:
    if _db is None:
        return []
    return list(_db.table_names())


def _table_row_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in _table_names():
        try:
            counts[name] = _db.open_table(name).count_rows()
        except Exception:
            counts[name] = -1
    return counts


def _maxp_aggregate(rows: list[dict], top_k: int) -> list[dict]:
    """Aggregate chunk-level results into doc-level (MaxP) ranking.

    Each row must carry keys: doc_id, _distance. Score = 1 - cosine_distance.
    """
    doc_scores: dict[str, float] = {}
    for r in rows:
        doc_id = r["doc_id"]
        dist = r.get("_distance", r.get("score", 0.0))
        # LanceDB returns cosine distance (1 - cos_sim); convert to similarity.
        score = float(1.0 - dist) if "_distance" in r else float(dist)
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [
        {"doc_id": doc_id, "score": round(score, 6), "rank": rank + 1}
        for rank, (doc_id, score) in enumerate(sorted_docs)
    ]


def _search_table(name: str, q_vec: np.ndarray, top_k: int) -> list[dict]:
    """Run a single cosine search on a LanceDB table and return MaxP-aggregated docs."""
    if _db is None:
        raise HTTPException(500, "LanceDB not initialized")
    if name not in _table_names():
        raise HTTPException(404, f"Index '{name}' not found")

    table = _db.open_table(name)
    n_rows = table.count_rows()
    if n_rows == 0:
        return []

    # Over-fetch chunks so that after MaxP aggregation we still have top_k docs.
    candidate_k = min(n_rows, max(top_k * 10, 1000))
    rows = (
        table.search(q_vec.astype(np.float32))
        .metric("cosine")
        .limit(candidate_k)
        .select(["chunk_id", "doc_id"])
        .to_list()
    )
    return _maxp_aggregate(rows, top_k)


def _to_embedding_http_error(exc: Exception) -> HTTPException:
    """Wrap embedding backend failures in an actionable HTTP 502 response."""
    if _embed_format == "simple":
        endpoint_str = _embed_endpoint or "unknown"
    else:
        endpoint_str = str(_embed_client.base_url) if _embed_client else "unknown"
    return HTTPException(
        502,
        (
            "Embedding backend request failed. "
            f"format={_embed_format}, endpoint={endpoint_str}, "
            f"model={_embed_model}, error={exc}"
        ),
    )


# ---- Endpoints ----

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pid": os.getpid(),
        "indexes": _table_names(),
        "embedding_model": _embed_model,
        "embedding_format": _embed_format,
        "embedding_endpoint": (
            _embed_endpoint
            if _embed_format == "simple"
            else (str(_embed_client.base_url) if _embed_client else None)
        ),
        "persist_dir": str(_persist_dir) if _persist_dir else None,
    }


@app.get("/indexes")
def list_indexes():
    return {
        "indexes": _table_names(),
        "n_chunks": _table_row_counts(),
    }


@app.post("/index/{name}/build")
def build_index(name: str, req: BuildRequest):
    """Embed all chunks in batch and create (or overwrite) a LanceDB table."""
    if _db is None:
        raise HTTPException(500, "LanceDB not initialized")
    if not req.chunks:
        raise HTTPException(400, "No chunks provided")

    chunk_dicts = [c.model_dump() for c in req.chunks]
    texts = [c["text"] for c in chunk_dicts]

    try:
        vecs = _embed_batch(texts, is_query=False)
    except Exception as exc:
        raise _to_embedding_http_error(exc) from exc
    dim = int(vecs.shape[1])

    rows = _rows_from_chunks(chunk_dicts, vecs)
    table = _db.create_table(name, schema=_make_schema(dim), mode="overwrite")
    table.add(rows)

    _known_ids[name] = {c["chunk_id"] for c in chunk_dicts}

    print(f"[server] Built LanceDB table '{name}' with {len(rows)} rows (dim={dim})")
    return {"status": "built", "n_chunks": len(rows), "dim": dim}


@app.post("/index/{name}/append")
def append_chunks(name: str, req: AppendChunksRequest):
    """Embed new chunks in batch and append them to a LanceDB table.

    Creates the table on the first append. Idempotent: duplicate chunk_ids
    (seen during this process) are silently skipped.
    """
    if _db is None:
        raise HTTPException(500, "LanceDB not initialized")

    seen = _known_ids.setdefault(name, set())
    # On first append in a new process, populate `seen` from any existing rows
    # so duplicates across restarts are still handled.
    if not seen and name in _table_names():
        try:
            existing = _db.open_table(name).to_pandas(columns=["chunk_id"])
            seen.update(existing["chunk_id"].tolist())
        except Exception:
            pass

    new_chunks = [c.model_dump() for c in req.chunks if c.chunk_id not in seen]
    n_skipped = len(req.chunks) - len(new_chunks)

    if new_chunks:
        texts = [c["text"] for c in new_chunks]
        try:
            vecs = _embed_batch(texts, is_query=False)
        except Exception as exc:
            raise _to_embedding_http_error(exc) from exc
        rows = _rows_from_chunks(new_chunks, vecs)

        if name in _table_names():
            table = _db.open_table(name)
            table.add(rows)
        else:
            dim = int(vecs.shape[1])
            table = _db.create_table(name, schema=_make_schema(dim), mode="overwrite")
            table.add(rows)

        seen.update(c["chunk_id"] for c in new_chunks)

    n_total = _db.open_table(name).count_rows() if name in _table_names() else 0
    return {
        "status": "appended",
        "n_staged": n_total,
        "n_skipped": n_skipped,
    }


@app.post("/index/{name}/finalize")
def finalize_index(name: str, req: FinalizeRequest):
    """No-op for the LanceDB backend — rows are durable as soon as they are added.

    Retained so the DenseClient's append/finalize flow keeps working unchanged.
    """
    if _db is None:
        raise HTTPException(500, "LanceDB not initialized")
    if name not in _table_names():
        raise HTTPException(400, f"No table '{name}' — nothing to finalize")

    n = _db.open_table(name).count_rows()
    print(f"[server] Finalized LanceDB table '{name}' ({n} rows, persisted to disk)")
    return {"status": "built", "n_chunks": n}


@app.post("/index/{name}/retrieve")
def retrieve(name: str, req: RetrieveRequest):
    if _db is None:
        raise HTTPException(500, "LanceDB not initialized")
    try:
        q_vec = _embed_batch([req.query], is_query=True)[0]
    except Exception as exc:
        raise _to_embedding_http_error(exc) from exc
    results = _search_table(name, q_vec, req.top_k)
    return {"results": results}


@app.post("/index/{name}/batch_retrieve")
def batch_retrieve(name: str, req: BatchRetrieveRequest):
    if _db is None:
        raise HTTPException(500, "LanceDB not initialized")
    if name not in _table_names():
        raise HTTPException(404, f"Index '{name}' not found")

    query_texts = [q.query_text for q in req.queries]
    # Batch-embed all queries in a single pass.
    try:
        q_mat = _embed_batch(query_texts, is_query=True)
    except Exception as exc:
        raise _to_embedding_http_error(exc) from exc

    all_results = []
    for i, q in enumerate(req.queries):
        ranked = _search_table(name, q_mat[i], req.top_k)
        all_results.append({"query_id": q.query_id, "ranked_docs": ranked})
    return {"results": all_results}


@app.delete("/index/{name}")
def delete_index(name: str):
    if _db is None:
        raise HTTPException(500, "LanceDB not initialized")
    if name not in _table_names():
        raise HTTPException(404, f"Index '{name}' not found")
    _db.drop_table(name)
    _known_ids.pop(name, None)
    return {"status": "deleted"}


# ---- CLI entry point ----

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Dense Index Server (Qwen3 + LanceDB)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--persist-dir", type=str, default=".dense_cache",
        help="LanceDB URI (directory) where all tables are stored",
    )
    parser.add_argument(
        "--embedding-endpoint",
        type=str,
        default=os.environ.get("EMBEDDING_ENDPOINT", "http://localhost:8001/v1"),
        help="Base URL for the OpenAI-compatible embeddings endpoint (Qwen3)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=os.environ.get("EMBEDDING_API_KEY", "EMPTY"),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-chars", type=int, default=20000)
    parser.add_argument(
        "--embedding-max-tokens",
        type=int,
        default=int(os.environ.get("EMBEDDING_MAX_TOKENS", "0")),
        help=(
            "If > 0, truncate each input to this many tokens before calling the "
            "embedding endpoint. Required for OpenAI-format endpoints whose model "
            "has a hard input-token cap (e.g. 8192)."
        ),
    )
    parser.add_argument(
        "--embedding-format",
        type=str,
        choices=["openai", "simple"],
        default=os.environ.get("EMBEDDING_FORMAT", "openai"),
        help=(
            "Wire format for the embedding endpoint. "
            "'openai' => POST {endpoint}/embeddings with {model,input}; "
            "'simple' => POST {endpoint} with {texts:[...]} expecting {embeddings:[...]}"
        ),
    )
    parser.add_argument("--embedding-timeout", type=float, default=120.0)
    parser.add_argument(
        "--query-instruction",
        type=str,
        default=os.environ.get(
            "QUERY_INSTRUCTION",
            "Given a web search query, retrieve relevant passages that answer the query.",
        ),
    )
    args = parser.parse_args()

    _persist_dir = pathlib.Path(args.persist_dir)
    _persist_dir.mkdir(parents=True, exist_ok=True)
    _db = lancedb.connect(str(_persist_dir))

    _embed_format = args.embedding_format
    _embed_endpoint = args.embedding_endpoint
    _embed_api_key = args.embedding_api_key
    _embed_timeout = args.embedding_timeout
    _embed_model = args.embedding_model
    _embed_batch_size = args.embedding_batch_size
    _embed_max_chars = args.embedding_max_chars
    _embed_max_tokens = args.embedding_max_tokens
    _query_instruction = args.query_instruction

    if _embed_format == "openai":
        _embed_client = OpenAI(base_url=args.embedding_endpoint, api_key=args.embedding_api_key)
    else:
        _embed_client = None

    existing = _table_names()
    print(
        f"[server] LanceDB at {_persist_dir} — {len(existing)} existing table(s): "
        f"{existing[:10]}{'...' if len(existing) > 10 else ''}"
    )
    print(
        f"[server] Dense server starting on port {args.port} — "
        f"format={_embed_format}, embedding endpoint={args.embedding_endpoint}, "
        f"model={_embed_model}, batch_size={_embed_batch_size}, "
        f"max_chars={_embed_max_chars}, max_tokens={_embed_max_tokens or 'off'}"
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
