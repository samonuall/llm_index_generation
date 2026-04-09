"""
tests/data/test_data_transformation.py

Tests for "data transformation" logic in src.evaluation.scripts.get_data:
- Query extraction: qrels fallback, label filtering, limiting
- Corpus mapping: document_id/document_content -> doc_id/text, metadata defaulting
- Cache read/write behavior is avoided by forcing cache misses into tmp dirs

These tests DO NOT hit the network. They mock datasets.load_dataset and the
cache directory used by get_data.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pytest

import src.evaluation.scripts.get_data as get_data


# -----------------------------
# Helpers
# -----------------------------

class _StreamingDataset:
    """Simple iterable to mimic HF streaming dataset."""
    def __init__(self, items: List[Dict[str, Any]]):
        self._items = items

    def __iter__(self):
        yield from self._items


@pytest.fixture
def fake_cache_root(tmp_path: Path) -> Path:
    # We'll make get_cache_dir return <tmp>/data/<split>...
    return tmp_path / "data"


@pytest.fixture
def patch_cache_dir(monkeypatch, fake_cache_root: Path):
    def _fake_get_cache_dir(split: str) -> Path:
        d = fake_cache_root / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(get_data, "get_cache_dir", _fake_get_cache_dir)
    return _fake_get_cache_dir


# -----------------------------
# Query transformation tests
# -----------------------------

def test_load_queries_filters_to_positive_labels_and_uses_full_document_qrels(monkeypatch, patch_cache_dir):
    """
    When full_document_qrels exists, it should be used.
    Only qrels with label > 0 should be included.
    Queries with no positive relevant IDs should be dropped.
    """
    split = "tip_of_the_tongue"

    hf_items = [
        {
            "query_id": "q1",
            "query_content": "foo",
            "full_document_qrels": [{"id": "d1", "label": 1}, {"id": "d2", "label": 0}],
        },
        {
            "query_id": "q2",
            "query_content": "bar",
            "full_document_qrels": [{"id": "d3", "label": -1}],  # no positive -> drop
        },
        {
            "query_id": "q3",
            "query_content": "baz",
            "full_document_qrels": [{"id": "d4", "label": 2}, {"id": "d5", "label": 1}],
        },
    ]

    def fake_load_dataset(*args, **kwargs):
        # load_queries uses load_dataset(..., split=crumb_name)
        return hf_items

    monkeypatch.setattr(get_data, "load_dataset", fake_load_dataset)

    out = get_data.load_queries(split)

    assert out == [
        {"query_id": "q1", "query_content": "foo", "relevant_doc_ids": ["d1"]},
        {"query_id": "q3", "query_content": "baz", "relevant_doc_ids": ["d4", "d5"]},
    ]


def test_load_queries_falls_back_to_passage_qrels(monkeypatch, patch_cache_dir):
    """If full_document_qrels is missing/None, load_queries should use passage_qrels."""
    split = "paper_retrieval"

    hf_items = [
        {
            "query_id": "q1",
            "query_content": "x",
            "full_document_qrels": None,
            "passage_qrels": [{"id": "docA", "label": 1}],
        },
        {
            "query_id": "q2",
            "query_content": "y",
            # full_document_qrels missing entirely
            "passage_qrels": [{"id": "docB", "label": 0}, {"id": "docC", "label": 1}],
        },
    ]

    monkeypatch.setattr(get_data, "load_dataset", lambda *a, **k: hf_items)

    out = get_data.load_queries(split)

    assert out == [
        {"query_id": "q1", "query_content": "x", "relevant_doc_ids": ["docA"]},
        {"query_id": "q2", "query_content": "y", "relevant_doc_ids": ["docC"]},
    ]


def test_load_queries_handles_missing_qrels_fields(monkeypatch, patch_cache_dir):
    """If neither qrels field exists, query should be dropped (no relevant ids)."""
    split = "stack_exchange"

    hf_items = [
        {"query_id": "q1", "query_content": "x"},  # no qrels => drop
        {"query_id": "q2", "query_content": "y", "full_document_qrels": []},  # empty => drop
        {"query_id": "q3", "query_content": "z", "full_document_qrels": [{"id": "d1", "label": 1}]},
    ]

    monkeypatch.setattr(get_data, "load_dataset", lambda *a, **k: hf_items)

    out = get_data.load_queries(split)

    assert out == [
        {"query_id": "q3", "query_content": "z", "relevant_doc_ids": ["d1"]},
    ]


def test_load_queries_applies_n_queries_limit_after_filtering(monkeypatch, patch_cache_dir):
    """n_queries should slice the final filtered list, not the raw HF dataset."""
    split = "clinical_trial"

    hf_items = [
        {"query_id": "q1", "query_content": "a", "full_document_qrels": [{"id": "d1", "label": 1}]},
        {"query_id": "q2", "query_content": "b", "full_document_qrels": [{"id": "d2", "label": 1}]},
        {"query_id": "q3", "query_content": "c", "full_document_qrels": [{"id": "d3", "label": 1}]},
    ]
    monkeypatch.setattr(get_data, "load_dataset", lambda *a, **k: hf_items)

    out = get_data.load_queries(split, n_queries=2)

    assert len(out) == 2
    assert [q["query_id"] for q in out] == ["q1", "q2"]


def test_load_queries_writes_expected_jsonl_format(monkeypatch, patch_cache_dir):
    """On cache miss, load_queries should write JSONL with expected keys."""
    split = "legal_qa"

    hf_items = [
        {
            "query_id": "q1",
            "query_content": "foo",
            "full_document_qrels": [{"id": "d1", "label": 1}],
        }
    ]
    monkeypatch.setattr(get_data, "load_dataset", lambda *a, **k: hf_items)

    out = get_data.load_queries(split)

    cache_dir = get_data.get_cache_dir(split)
    cache_file = cache_dir / "queries.jsonl"
    assert cache_file.exists()

    lines = cache_file.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row == out[0]
    assert set(row.keys()) == {"query_id", "query_content", "relevant_doc_ids"}


# -----------------------------
# Corpus transformation tests
# -----------------------------

def test_load_full_corpus_streaming_maps_fields_and_stringifies_doc_id(monkeypatch, patch_cache_dir):
    """
    Field mapping:
      document_id -> doc_id (str)
      document_content -> text
      metadata -> {}
    """
    split = "tip_of_the_tongue"
    hf_items = [
        {"document_id": 123, "document_content": "hello"},
        {"document_id": "abc", "document_content": "world"},
    ]

    def fake_load_dataset(*args, **kwargs):
        assert kwargs.get("streaming") is True
        return _StreamingDataset(hf_items)

    monkeypatch.setattr(get_data, "load_dataset", fake_load_dataset)

    docs = get_data.load_full_corpus_streaming(split, max_docs=10)

    assert docs == [
        {"doc_id": "123", "text": "hello", "metadata": {}},
        {"doc_id": "abc", "text": "world", "metadata": {}},
    ]


def test_load_full_corpus_streaming_respects_max_docs(monkeypatch, patch_cache_dir):
    split = "paper_retrieval"
    hf_items = [{"document_id": i, "document_content": f"doc{i}"} for i in range(10)]

    monkeypatch.setattr(
        get_data,
        "load_dataset",
        lambda *a, **k: _StreamingDataset(hf_items),
    )

    docs = get_data.load_full_corpus_streaming(split, max_docs=3)

    assert len(docs) == 3
    assert docs[0]["doc_id"] == "0"
    assert docs[-1]["doc_id"] == "2"


def test_load_full_corpus_streaming_writes_expected_jsonl_format(monkeypatch, patch_cache_dir):
    split = "theorem_retrieval"
    hf_items = [{"document_id": "d1", "document_content": "content"}]

    monkeypatch.setattr(
        get_data,
        "load_dataset",
        lambda *a, **k: _StreamingDataset(hf_items),
    )

    docs = get_data.load_full_corpus_streaming(split, max_docs=1)

    cache_dir = get_data.get_cache_dir(split)
    cache_file = cache_dir / "documents.jsonl"
    assert cache_file.exists()

    lines = cache_file.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row == docs[0]
    assert set(row.keys()) == {"doc_id", "text", "metadata"}


def test_load_full_corpus_streaming_reads_from_cache_without_calling_load_dataset(monkeypatch, patch_cache_dir):
    """
    If cache file exists, function should return cached docs and never call load_dataset.
    """
    split = "code_retrieval"
    cache_dir = get_data.get_cache_dir(split)
    cache_file = cache_dir / "documents.jsonl"
    cached = [
        {"doc_id": "x", "text": "cached", "metadata": {}},
        {"doc_id": "y", "text": "cached2", "metadata": {"a": 1}},
    ]
    with cache_file.open("w") as f:
        for row in cached:
            f.write(json.dumps(row) + "\n")

    def boom(*a, **k):
        raise AssertionError("load_dataset should not be called on cache hit")

    monkeypatch.setattr(get_data, "load_dataset", boom)

    out = get_data.load_full_corpus_streaming(split, max_docs=5)
    assert out == cached


def test_load_queries_reads_from_cache_without_calling_load_dataset(monkeypatch, patch_cache_dir):
    """
    If cache file exists, function should return cached queries and never call load_dataset.
    """
    split = "set_operation_entity_retrieval"
    cache_dir = get_data.get_cache_dir(split)
    cache_file = cache_dir / "queries.jsonl"
    cached = [
        {"query_id": "q1", "query_content": "cached", "relevant_doc_ids": ["d1"]},
        {"query_id": "q2", "query_content": "cached2", "relevant_doc_ids": ["d2", "d3"]},
    ]
    with cache_file.open("w") as f:
        for row in cached:
            f.write(json.dumps(row) + "\n")

    def boom(*a, **k):
        raise AssertionError("load_dataset should not be called on cache hit")

    monkeypatch.setattr(get_data, "load_dataset", boom)

    out = get_data.load_queries(split)
    assert out == cached