"""
Shared test fixtures for the LLM Index Generation test suite.

All fixtures use small, synthetic data so tests are fast, deterministic,
and don't require the real CRUMB corpus. The documents and queries are
designed so that BM25 retrieval produces predictable results:

  q1 ("furry pet animal ... retractable claws") -> doc_001 (The Cat)
  q2 ("planets orbiting the sun ... gas giants") -> doc_002 (Solar System)
  q3 ("interpreted language ... whitespace")     -> doc_003 (Python Programming)
  q4 ("green plants ... sunlight chlorophyll")   -> doc_005 (Photosynthesis)
  q5 ("tallest mountain ... Nepal ... Himalayas") -> doc_007 (Mount Everest)

Fixture data lives in tests/fixtures/ as JSONL files so the data format
matches production exactly.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Make src/ packages importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_EVAL_DIR = _SRC_DIR / "evaluation"
_SCRIPTS_DIR = _EVAL_DIR / "scripts"
_AGENTS_DIR = _SRC_DIR / "agents"

# Add project root first so 'src' module is importable
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Add specific directories for direct imports (backward compatibility)
for path in [_EVAL_DIR, _SCRIPTS_DIR, _AGENTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from schema import Document, Chunk, EvalQuery  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Core data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents() -> List[Document]:
    """8 short synthetic Wikipedia-style documents."""
    docs = []
    with (FIXTURES_DIR / "sample_documents.jsonl").open() as f:
        for line in f:
            docs.append(Document(**json.loads(line)))
    return docs


@pytest.fixture
def sample_queries() -> List[EvalQuery]:
    """5 queries with known ground-truth relevant doc IDs."""
    queries = []
    with (FIXTURES_DIR / "sample_queries.jsonl").open() as f:
        for line in f:
            q = json.loads(line)
            queries.append(EvalQuery(
                query_id=q["query_id"],
                query_text=q["query_content"],    # ✅ CORRECT
                relevant_doc_ids=q["relevant_doc_ids"],
            ))
    return queries


@pytest.fixture
def sample_chunks(sample_documents) -> List[Chunk]:
    """Baseline chunks: one chunk per document, raw text unchanged."""
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}_0",
            doc_id=doc.doc_id,
            text=doc.text,
        )
        for doc in sample_documents
    ]


# ---------------------------------------------------------------------------
# Multi-chunk fixture (for testing chunking strategies)
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_chunks(sample_documents) -> List[Chunk]:
    """
    Two chunks per document: split roughly in half by paragraphs.
    Useful for testing aggregation modes (max, sum, avg).
    """
    chunks = []
    for doc in sample_documents:
        paragraphs = doc.text.split("\n")
        mid = max(1, len(paragraphs) // 2)
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_0",
            doc_id=doc.doc_id,
            text="\n".join(paragraphs[:mid]),
        ))
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_1",
            doc_id=doc.doc_id,
            text="\n".join(paragraphs[mid:]),
        ))
    return chunks


# ---------------------------------------------------------------------------
# BM25 index fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def built_index(sample_chunks):
    """A BM25Index built from sample_chunks. Marked slow if needed."""
    from build_index import BM25Index
    return BM25Index(sample_chunks)


@pytest.fixture
def multi_chunk_index(multi_chunks):
    """A BM25Index built from multi_chunks for aggregation tests."""
    from build_index import BM25Index
    return BM25Index(multi_chunks)


# ---------------------------------------------------------------------------
# File-based fixtures (for testing JSONL I/O)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path, sample_documents, sample_queries) -> pathlib.Path:
    """
    Write sample documents and queries to a tmp directory in the same
    JSONL format that get_data.py produces. Returns the directory path.

    Layout:
      tmp_path/test_split/documents.jsonl
      tmp_path/test_split/queries.jsonl
    """
    split_dir = tmp_path / "test_split"
    split_dir.mkdir()

    with (split_dir / "documents.jsonl").open("w") as f:
        for doc in sample_documents:
            f.write(json.dumps({
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata,
            }) + "\n")

    with (split_dir / "queries.jsonl").open("w") as f:
        for q in sample_queries:
            f.write(json.dumps({
                "query_id": q.query_id,
                "query_content": q.query_text,
                "relevant_doc_ids": q.relevant_doc_ids,
            }) + "\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Eval results fixture (mock output from evaluate())
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_eval_results() -> dict:
    """
    A realistic eval results dict matching the shape returned by
    test_preprocessing_split.evaluate(). Useful for testing prompt
    construction and agent logic without running the actual pipeline.
    """
    return {
        "agent": "test_agent",
        "split": "test_split",
        "timestamp": "2026-03-31T12:00:00",
        "iteration": 0,
        "config": {
            "top_k": 100,
            "n_docs": 8,
            "n_queries": 5,
            "n_chunks": 8,
            "chunks_per_doc": 1.0,
        },
        "metrics": {
            "recall_at_10": 0.80,
            "recall_at_100": 0.80,
            "ndcg_at_10": 0.65,
        },
        "crumb_metrics": None,
    }


# ---------------------------------------------------------------------------
# Data layer fixtures (for testing data loading and storage)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_crumb_dataset():
    """Mock CRUMB dataset for testing data loading without network calls."""
    return [
        {
            "query_id": "q1",
            "query_content": "What is machine learning?",
            "passage_qrels": [{"id": "doc1", "label": 1}],
        },
        {
            "query_id": "q2",
            "query_content": "Python programming tutorial",
            "passage_qrels": [{"id": "doc2", "label": 1}],
        },
    ]


@pytest.fixture
def mock_crumb_corpus():
    """Mock CRUMB corpus for testing document loading."""
    return [
        {"document_id": "doc1", "document_content": "Machine learning is a subset of AI."},
        {"document_id": "doc2", "document_content": "Python is a programming language."},
        {"document_id": "doc3", "document_content": "Data science uses statistics."},
    ]


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory for testing data caching."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir