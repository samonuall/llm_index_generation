"""
dense_server.py — FastAPI server hosting LanceDB vector indexes.

Mirrors the BM25 server's API surface (build / append / finalize / retrieve /
batch_retrieve / delete / health), but every named "index" is a LanceDB
*table* with a real **HNSW vector index** built on top:

    tbl.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_HNSW_SQ",
        ...
    )

Per-loop / per-hypothesis lifecycle (driven by the agent):
    1. POST /index/{name}/build      → drop any existing table named {name},
                                       embed all chunk texts (truncated to the
                                       model's max_input_tokens), insert into
                                       a fresh LanceDB table, then build the
                                       HNSW vector index with cosine metric.
    2. POST /index/{name}/retrieve   → embed the query, search the HNSW index,
                                       MaxP-aggregate over chunks → docs.
    3. DELETE /index/{name}          → drop the table (and any persisted files
                                       for it under lance_uri).

Usage:
    python dense_server.py --port 8766 \\
        --lance-uri .lance_cache \\
        --config-path src/agents/analysis_code_agent_dense/config.yaml
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import pathlib
import shutil
import sys
import time
from typing import Any

import lancedb
import pyarrow as pa
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Allow `from embedder import ...` when running this file directly via uv run.
_THIS_DIR = pathlib.Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from embedder import Embedder, EmbedderConfig  # noqa: E402

logger = logging.getLogger("dense_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

app = FastAPI(title="Dense Vector Index Server (LanceDB)")


# ---- In-memory state ----

_db: Any = None  # lancedb.DBConnection
_lance_uri: pathlib.Path | None = None
_embedder: Embedder | None = None
_index_cfg: dict = {}

_indexes: dict[str, dict] = {}
# entry: {"n_chunks", "n_truncated", "build_seconds", "index_type", "metric", "dim"}

_staging: dict[str, list[dict]] = {}
_staging_ids: dict[str, set[str]] = {}


def _lance_list_tables() -> list[str]:
    """Return LanceDB table names without relying on deprecated APIs when possible."""
    if _db is None:
        return []
    if hasattr(_db, "list_tables"):
        return list(_db.list_tables())  # type: ignore[no-untyped-call]
    return list(_db.table_names())


def _lance_table_exists(name: str) -> bool:
    return name in set(_lance_list_tables())


# ---- Request / response models ----

class ChunkIn(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = {}


class BuildRequest(BaseModel):
    chunks: list[ChunkIn]
    persist: bool = False


class AppendChunksRequest(BaseModel):
    chunks: list[ChunkIn]


class FinalizeRequest(BaseModel):
    persist: bool = False


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 100


class QueryIn(BaseModel):
    query_id: str
    query_text: str


class BatchRetrieveRequest(BaseModel):
    queries: list[QueryIn]
    top_k: int = 100


# ---- Helpers ----

def _schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("metadata", pa.string()),  # JSON-encoded
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


def _drop_table_if_exists(name: str) -> None:
    """Drop a table on disk and clear our metadata for it. Idempotent."""
    try:
        if _lance_table_exists(name):
            _db.drop_table(name)
            logger.info("Dropped existing table '%s'", name)
    except Exception as e:
        logger.warning("Failed dropping existing table '%s': %s", name, e)
    _indexes.pop(name, None)


def _build_index_from_chunks(name: str, chunk_dicts: list[dict], persist: bool) -> dict:
    """Embed chunks, create the table, and build the HNSW vector index."""
    assert _embedder is not None and _db is not None

    t0 = time.time()
    texts = [c["text"] for c in chunk_dicts]
    vectors, n_truncated = _embedder.embed_documents(texts)
    if not vectors:
        raise HTTPException(400, "No vectors produced (input list was empty after truncation)")

    dim = len(vectors[0])

    # Drop any existing table with this name first so the table is recreated cleanly.
    _drop_table_if_exists(name)

    rows = []
    for c, v in zip(chunk_dicts, vectors):
        rows.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "text": c["text"],
            "metadata": json.dumps(c.get("metadata", {}) or {}),
            "vector": v,
        })
    arrow_table = pa.Table.from_pylist(rows, schema=_schema(dim))

    # Use mode="overwrite" as a hard safety-net: if _drop_table_if_exists()
    # silently swallowed a drop failure (e.g. a stale table from a previous
    # crashed run), create_table would raise "Table 'X' already exists".
    # overwrite=True / mode="overwrite" avoids that entirely.
    try:
        tbl = _db.create_table(name, data=arrow_table, mode="overwrite")
    except TypeError:
        # Older lancedb versions use overwrite= instead of mode=
        tbl = _db.create_table(name, data=arrow_table, overwrite=True)

    # Vector index configuration.
    vec_cfg = _index_cfg or {}
    metric = vec_cfg.get("metric", "cosine")
    index_type = vec_cfg.get("index_type", "IVF_HNSW_SQ")

    n = len(chunk_dicts)
    requested_partitions = int(vec_cfg.get("num_partitions", 256))
    # IVF needs at least one row per partition; clamp gracefully on small corpora.
    num_partitions = max(1, min(requested_partitions, max(1, n // 64)))
    num_sub_vectors = int(vec_cfg.get("num_sub_vectors", 96))
    # num_sub_vectors must divide dim for PQ flavours; for IVF_HNSW_SQ it's only used
    # if SQ falls back, but we still keep it sensible.
    if dim % num_sub_vectors != 0:
        # pick the largest divisor of dim that is <= requested
        for cand in range(num_sub_vectors, 0, -1):
            if dim % cand == 0:
                num_sub_vectors = cand
                break

    # Build kwargs progressively — newer LanceDB versions dropped some params.
    # IVF_HNSW_SQ uses scalar quantisation, not PQ, so num_sub_vectors is not
    # meaningful for it and may be rejected.  We try increasingly minimal
    # combinations and fall back to a flat (exhaustive) scan on small corpora.
    create_index_kwargs = dict(
        metric=metric,
        vector_column_name="vector",
        index_type=index_type,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        m=int(vec_cfg.get("m", 32)),
        ef_construction=int(vec_cfg.get("ef_construction", 200)),
    )
    _index_built = False
    _attempts = [
        # (keys to remove on this attempt)
        [],                              # full kwargs
        ["m", "ef_construction"],        # drop HNSW-specific params
        ["num_sub_vectors"],             # also drop PQ param
        ["m", "ef_construction", "num_sub_vectors"],  # minimal IVF
    ]
    for extra_drops in _attempts:
        kwargs = {k: v for k, v in create_index_kwargs.items() if k not in extra_drops}
        try:
            tbl.create_index(**kwargs)
            _index_built = True
            break
        except TypeError as e:
            logger.debug("create_index attempt with drops=%s raised TypeError: %s", extra_drops, e)
        except Exception as e:
            logger.debug("create_index attempt with drops=%s raised %s: %s", extra_drops, type(e).__name__, e)
    if not _index_built:
        logger.warning(
            "All create_index attempts failed for '%s' — using flat (exhaustive) search. "
            "This is safe for corpora <=10 k chunks but slower at scale.",
            name,
        )

    try:
        tbl.create_scalar_index("doc_id")
    except Exception as e:
        logger.debug("create_scalar_index('doc_id') failed (non-fatal): %s", e)

    build_seconds = round(time.time() - t0, 2)
    info = {
        "n_chunks": n,
        "n_truncated": n_truncated,
        "build_seconds": build_seconds,
        "index_type": index_type,
        "metric": metric,
        "dim": dim,
    }
    _indexes[name] = info

    if not persist and _lance_uri is not None:
        # Always-on persistence is the LanceDB default; we just track that the
        # caller asked for an ephemeral index — it'll be cleaned by DELETE.
        pass

    logger.info(
        "Built index '%s' (%d chunks, %d truncated, dim=%d, %s/%s, %.1fs)",
        name, n, n_truncated, dim, index_type, metric, build_seconds,
    )
    return info


def _search_documents(name: str, query: str, top_k: int) -> list[dict]:
    """Embed a query, run vector search, MaxP-aggregate over doc_id."""
    assert _embedder is not None and _db is not None
    if name not in _indexes:
        raise HTTPException(404, f"Index '{name}' not found")

    qv = _embedder.embed_query(query)
    tbl = _db.open_table(name)

    candidate_k = int(_index_cfg.get("candidate_k", 1000))
    nprobes = int(_index_cfg.get("nprobes", 16))
    refine_factor = int(_index_cfg.get("refine_factor", 10))

    search = tbl.search(qv, vector_column_name="vector").metric(_indexes[name]["metric"]).limit(candidate_k)
    try:
        search = search.nprobes(nprobes).refine_factor(refine_factor)
    except AttributeError:
        # Older LanceDB versions expose nprobes via a different chain; fall back gracefully.
        pass
    rows = search.to_list()

    # rows is a list of dicts with chunk fields plus a `_distance` column.
    # For cosine distance LanceDB returns 1 - cosine_similarity, so we convert back.
    doc_scores: dict[str, float] = {}
    for row in rows:
        doc_id = row.get("doc_id")
        if doc_id is None:
            continue
        dist = float(row.get("_distance", 0.0))
        score = 1.0 - dist  # cosine similarity in [-1, 1]
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [
        {"doc_id": doc_id, "score": round(score, 6), "rank": rank + 1}
        for rank, (doc_id, score) in enumerate(sorted_docs)
    ]


# ---- Endpoints ----

@app.get("/health")
def health():
    # Do not force an embedding endpoint call during health checks.
    # Startup polling should stay local/fast even if the remote embedding
    # endpoint is slow or temporarily unavailable.
    embedding_dim = None
    if _embedder is not None:
        embedding_dim = getattr(_embedder, "_dim", None)
    return {
        "status": "ok",
        "indexes": list(_indexes.keys()),
        "embedding_model": _embedder.model if _embedder else None,
        "embedding_dim": embedding_dim,
    }


@app.get("/embed_health")
def embed_health():
    """Probe the embedding endpoint with a tiny request and return the dimension.

    Called once by the agent after server startup to verify the remote vLLM
    service is reachable *before* any index build is attempted.
    """
    assert _embedder is not None
    try:
        dim = _embedder.dim  # triggers a real /v1/embeddings call on first access
    except Exception as e:
        logger.exception("Embedding endpoint probe failed: %s", e)
        raise HTTPException(503, f"Embedding endpoint unreachable: {type(e).__name__}: {e}") from e
    return {"status": "ok", "embedding_dim": dim, "model": _embedder.model}


def list_indexes():
    return {
        "indexes": list(_indexes.keys()),
        "info": {name: info for name, info in _indexes.items()},
    }


@app.post("/index/{name}/build")
def build_index(name: str, req: BuildRequest):
    if not req.chunks:
        raise HTTPException(400, "No chunks provided")
    chunk_dicts = [c.model_dump() for c in req.chunks]
    try:
        info = _build_index_from_chunks(name, chunk_dicts, req.persist)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Build failed for index '%s': %s: %s", name, type(e).__name__, e)
        raise HTTPException(500, f"Build failed: {type(e).__name__}: {e}") from e
    return {"status": "built", **info}


@app.post("/index/{name}/append")
def append_chunks(name: str, req: AppendChunksRequest):
    """Append chunks to a staging buffer for batched index building.

    Idempotent: duplicate chunk_ids (e.g. from a retried request) are silently skipped.
    """
    if name not in _staging:
        _staging[name] = []
        _staging_ids[name] = set()
    new_chunks = [
        c.model_dump() for c in req.chunks
        if c.chunk_id not in _staging_ids[name]
    ]
    _staging_ids[name].update(c["chunk_id"] for c in new_chunks)
    _staging[name].extend(new_chunks)
    return {
        "status": "appended",
        "n_staged": len(_staging[name]),
        "n_skipped": len(req.chunks) - len(new_chunks),
    }


@app.post("/index/{name}/finalize")
def finalize_index(name: str, req: FinalizeRequest):
    """Build the index from staged chunks, then clear staging on success.

    This endpoint is idempotent under client retries:
    - If the index is already built and no staged chunks remain, return success.
    - If build fails, keep staged chunks so finalize can be retried.
    """
    has_staged = name in _staging and bool(_staging[name])
    if not has_staged:
        if name in _indexes:
            return {"status": "already_built", **_indexes[name]}
        raise HTTPException(400, f"No staged chunks for index '{name}'")

    chunk_dicts = _staging[name]
    try:
        info = _build_index_from_chunks(name, chunk_dicts, req.persist)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Finalize/build failed for index '%s': %s: %s", name, type(e).__name__, e)
        raise HTTPException(500, f"Build failed: {type(e).__name__}: {e}") from e

    # Only clear staging after a successful build.
    _staging.pop(name, None)
    _staging_ids.pop(name, None)
    return {"status": "built", **info}


@app.post("/index/{name}/retrieve")
def retrieve(name: str, req: RetrieveRequest):
    results = _search_documents(name, req.query, req.top_k)
    return {"results": results}


@app.post("/index/{name}/batch_retrieve")
def batch_retrieve(name: str, req: BatchRetrieveRequest):
    if name not in _indexes:
        raise HTTPException(404, f"Index '{name}' not found")
    out = []
    for q in req.queries:
        ranked = _search_documents(name, q.query_text, req.top_k)
        out.append({"query_id": q.query_id, "ranked_docs": ranked})
    return {"results": out}


@app.delete("/index/{name}")
def delete_index(name: str):
    try:
        table_exists = _lance_table_exists(name)
    except Exception as e:
        logger.warning("_lance_table_exists('%s') raised %s — assuming exists for safety", name, e)
        table_exists = True
    if name not in _indexes and not table_exists:
        raise HTTPException(404, f"Index '{name}' not found")
    _drop_table_if_exists(name)
    return {"status": "deleted"}


# ---- CLI entry point ----

def _atexit_cleanup() -> None:
    """Drop ephemeral tables (whose names match agent prefixes) on shutdown."""
    if _db is None:
        return
    try:
        # "current" is the exact name used by the loop-scoped index; it must be
        # listed explicitly because startswith("") would match everything.
        ephemeral_exact = {"current"}
        ephemeral_prefixes = ("current_", "hyp_", "baseline_", "synthesized_", "harness_eval", "final_eval")
        for tname in _lance_list_tables():
            if tname in ephemeral_exact or tname.startswith(ephemeral_prefixes):
                try:
                    _db.drop_table(tname)
                except Exception:
                    pass
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Dense Vector Index Server (LanceDB)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--lance-uri", type=str, default=".lance_cache")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the agent's config.yaml (provides embedding + vector_index settings)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe lance-uri before starting (drops all persisted tables)",
    )
    args = parser.parse_args()

    _lance_uri = pathlib.Path(args.lance_uri).resolve()
    if args.reset and _lance_uri.exists():
        shutil.rmtree(_lance_uri)
    _lance_uri.mkdir(parents=True, exist_ok=True)

    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _embedder = Embedder(EmbedderConfig.from_dict(cfg))
    _index_cfg = cfg.get("vector_index", {})

    _db = lancedb.connect(str(_lance_uri))

    # Pre-populate _indexes with any persisted tables. Do not touch the embedder
    # here – probing dims requires calling /v1/embeddings and can take minutes
    # on a cold vLLM worker, which would block FastAPI/uvicorn from binding.
    try:
        for tname in _lance_list_tables():
            try:
                tbl = _db.open_table(tname)
                _indexes[tname] = {
                    "n_chunks": tbl.count_rows(),
                    "n_truncated": -1,
                    "build_seconds": -1.0,
                    "index_type": _index_cfg.get("index_type", "IVF_HNSW_SQ"),
                    "metric": _index_cfg.get("metric", "cosine"),
                    "dim": None,
                }
            except Exception as e:
                logger.warning("Could not open persisted table '%s': %s", tname, e)
    except Exception as e:
        logger.warning("Could not list persisted tables: %s", e)

    atexit.register(_atexit_cleanup)

    logger.info(
        "Dense server starting on port %d (lance_uri=%s, model=%s)",
        args.port, _lance_uri, _embedder.model,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
