"""
tests/data/test_data_pipeline.py

End-to-end workflow tests for the "data pipeline" represented by
src.evaluation.scripts.get_data:

  cache_dir = get_cache_dir(split, limit)
  queries   = load_queries(split, n_queries, limit)
  docs      = load_full_corpus_streaming(split, limit)

These tests:
- do NOT hit the network (mock datasets.load_dataset)
- verify JSONL cache artifacts are created correctly
- verify cache hits prevent additional "downloads"
- optionally validate that query relevant_doc_ids exist in the downloaded docs
  when the fake dataset is constructed that way
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import src.evaluation.scripts.get_data as get_data


class _StreamingDataset:
    def __init__(self, items: List[Dict[str, Any]]):
        self._items = items

    def __iter__(self):
        yield from self._items


@pytest.fixture
def fake_cache_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def patch_cache_dir(monkeypatch, fake_cache_root: Path):
    def _fake_get_cache_dir(split: str, n_docs=None) -> Path:
        if n_docs:
            d = fake_cache_root / f"{split}_{n_docs}docs"
        else:
            d = fake_cache_root / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(get_data, "get_cache_dir", _fake_get_cache_dir)
    return _fake_get_cache_dir


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_pipeline_cache_miss_creates_expected_files(monkeypatch, patch_cache_dir):
    """
    End-to-end on cache miss:
    - downloads (mocked) queries + docs
    - writes queries.jsonl and documents.jsonl to the right directory
    - returns transformed dicts
    """
    split = "tip_of_the_tongue"
    limit = 3
    n_queries = 2

    # --- Fake HF datasets ---
    hf_queries = [
        {
            "query_id": "q1",
            "query_content": "cat claws retractable",
            "passage_qrels": [{"id": "doc_001", "label": 1}],
        },
        {
            "query_id": "q2",
            "query_content": "solar system gas giants",
            "passage_qrels": [{"id": "doc_002", "label": 1}],
        },
        {
            # will be filtered out (no positive labels)
            "query_id": "q_bad",
            "query_content": "should drop",
            "passage_qrels": [{"id": "doc_999", "label": 0}],
        },
    ]

    hf_docs = [
        {"document_id": "doc_001", "document_content": "Cats are small mammals."},
        {"document_id": "doc_002", "document_content": "Jupiter and Saturn are gas giants."},
        {"document_id": "doc_003", "document_content": "Extra distractor doc."},
        {"document_id": "doc_004", "document_content": "Should not be downloaded due to limit."},
    ]

    def fake_load_dataset(name: str, config: str, split: str, streaming: bool = False):
        if config == "evaluation_queries":
            assert streaming is False
            return hf_queries
        if config == "passage_corpus":
            assert streaming is True
            return _StreamingDataset(hf_docs)
        raise AssertionError(f"Unexpected dataset config: {config}")

    monkeypatch.setattr(get_data, "load_dataset", fake_load_dataset)

    # --- Pipeline steps (mirrors get_data.main) ---
    cache_dir = get_data.get_cache_dir(split, limit)
    queries = get_data.load_queries(split, n_queries=n_queries, n_docs=limit)
    docs = get_data.load_full_corpus_streaming(split, max_docs=limit)

    # --- Verify returned transformed objects ---
    assert [q["query_id"] for q in queries] == ["q1", "q2"]
    assert queries[0]["relevant_doc_ids"] == ["doc_001"]
    assert len(docs) == 3
    assert docs[0] == {"doc_id": "doc_001", "text": "Cats are small mammals.", "metadata": {}}

    # --- Verify cache artifacts exist and contents match ---
    q_file = cache_dir / "queries.jsonl"
    d_file = cache_dir / "documents.jsonl"
    assert q_file.exists()
    assert d_file.exists()

    assert _read_jsonl(q_file) == queries
    assert _read_jsonl(d_file) == docs


def test_pipeline_cache_hit_skips_downloads(monkeypatch, patch_cache_dir):
    """
    Run pipeline twice:
    - first run writes caches
    - second run should read caches and not call load_dataset
    """
    split = "paper_retrieval"
    limit = 2

    hf_queries = [
        {
            "query_id": "q1",
            "query_content": "x",
            "passage_qrels": [{"id": "doc_1", "label": 1}],
        }
    ]
    hf_docs = [
        {"document_id": "doc_1", "document_content": "Doc 1"},
        {"document_id": "doc_2", "document_content": "Doc 2"},
    ]

    calls = {"evaluation_queries": 0, "passage_corpus": 0}

    def fake_load_dataset_first(name: str, config: str, split: str, streaming: bool = False):
        if config == "evaluation_queries":
            calls["evaluation_queries"] += 1
            return hf_queries
        if config == "passage_corpus":
            calls["passage_corpus"] += 1
            return _StreamingDataset(hf_docs)
        raise AssertionError("unexpected")

    monkeypatch.setattr(get_data, "load_dataset", fake_load_dataset_first)

    # First run (cache miss)
    _ = get_data.load_queries(split, n_queries=None, n_docs=limit)
    _ = get_data.load_full_corpus_streaming(split, max_docs=limit)

    assert calls["evaluation_queries"] == 1
    assert calls["passage_corpus"] == 1

    # Second run should not call load_dataset at all
    def boom(*a, **k):
        raise AssertionError("load_dataset should not be called on cache hit")

    monkeypatch.setattr(get_data, "load_dataset", boom)

    q2 = get_data.load_queries(split, n_queries=None, n_docs=limit)
    d2 = get_data.load_full_corpus_streaming(split, max_docs=limit)

    assert q2 == [{"query_id": "q1", "query_content": "x", "relevant_doc_ids": ["doc_1"]}]
    assert d2 == [
        {"doc_id": "doc_1", "text": "Doc 1", "metadata": {}},
        {"doc_id": "doc_2", "text": "Doc 2", "metadata": {}},
    ]


def test_pipeline_relevant_doc_ids_are_present_in_downloaded_docs(monkeypatch, patch_cache_dir):
    """
    This is a sanity check you typically want downstream:
    every query's relevant_doc_ids should exist in the docs you've downloaded.

    Note: in real life, if you limit docs, you may break this. Here we construct
    the fake dataset so it holds.
    """
    split = "stack_exchange"
    limit = 5

    hf_queries = [
        {
            "query_id": "q1",
            "query_content": "about docA",
            "passage_qrels": [{"id": "docA", "label": 1}, {"id": "docB", "label": 1}],
        },
        {
            "query_id": "q2",
            "query_content": "about docC",
            "passage_qrels": [{"id": "docC", "label": 1}],
        },
    ]
    hf_docs = [
        {"document_id": "docA", "document_content": "A content"},
        {"document_id": "docB", "document_content": "B content"},
        {"document_id": "docC", "document_content": "C content"},
        {"document_id": "docD", "document_content": "D content"},
        {"document_id": "docE", "document_content": "E content"},
    ]

    def fake_load_dataset(name: str, config: str, split: str, streaming: bool = False):
        if config == "evaluation_queries":
            return hf_queries
        if config == "passage_corpus":
            return _StreamingDataset(hf_docs)
        raise AssertionError("unexpected")

    monkeypatch.setattr(get_data, "load_dataset", fake_load_dataset)

    queries = get_data.load_queries(split, n_queries=None, n_docs=limit)
    docs = get_data.load_full_corpus_streaming(split, max_docs=limit)

    doc_ids = {d["doc_id"] for d in docs}
    for q in queries:
        for rid in q["relevant_doc_ids"]:
            assert rid in doc_ids


def test_pipeline_limit_can_break_relevance_and_that_is_expected(monkeypatch, patch_cache_dir):
    """
    Documents are often limited with --limit for dev/testing. That can make
    some relevant_doc_ids absent. This test documents that behavior: the loader
    does not attempt to enforce relevance coverage.
    """
    split = "clinical_trial"
    limit = 1  # intentionally too small

    hf_queries = [
        {
            "query_id": "q1",
            "query_content": "needs doc2",
            "passage_qrels": [{"id": "doc2", "label": 1}],
        }
    ]
    hf_docs = [
        {"document_id": "doc1", "document_content": "only doc1 downloaded"},
        {"document_id": "doc2", "document_content": "doc2 exists but won't be downloaded"},
    ]

    def fake_load_dataset(name: str, config: str, split: str, streaming: bool = False):
        if config == "evaluation_queries":
            return hf_queries
        if config == "passage_corpus":
            return _StreamingDataset(hf_docs)
        raise AssertionError("unexpected")

    monkeypatch.setattr(get_data, "load_dataset", fake_load_dataset)

    queries = get_data.load_queries(split, n_queries=None, n_docs=limit)
    docs = get_data.load_full_corpus_streaming(split, max_docs=limit)

    assert queries[0]["relevant_doc_ids"] == ["doc2"]
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "doc1"

    # Relevance is broken due to limit; that's expected.
    assert "doc2" not in {d["doc_id"] for d in docs}