"""
tests/data/test_data_cache.py

Caching-mechanism tests for src.evaluation.scripts.get_data.

In this repo, "caching" means:
- get_cache_dir(split, n_docs) creates/returns a deterministic directory
- load_queries() uses <cache_dir>/queries.jsonl if it exists (cache hit)
- load_full_corpus_streaming() uses <cache_dir>/documents.jsonl if it exists (cache hit)
- on cache miss, these functions write JSONL to the expected locations

All tests avoid network calls by mocking get_data.load_dataset.
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
    """
    Make get_data.get_cache_dir write into a temp folder so tests don't touch
    the real repo ./data directory.
    """
    def _fake_get_cache_dir(split: str, n_docs=None) -> Path:
        if n_docs:
            d = fake_cache_root / f"{split}_{n_docs}docs"
        else:
            d = fake_cache_root / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(get_data, "get_cache_dir", _fake_get_cache_dir)
    return _fake_get_cache_dir


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_get_cache_dir_creates_directory(patch_cache_dir, fake_cache_root: Path):
    d1 = get_data.get_cache_dir("tip_of_the_tongue", None)
    assert d1.exists()
    assert d1.is_dir()
    assert d1 == fake_cache_root / "tip_of_the_tongue"

    d2 = get_data.get_cache_dir("tip_of_the_tongue", 500)
    assert d2.exists()
    assert d2.is_dir()
    assert d2 == fake_cache_root / "tip_of_the_tongue_500docs"


def test_load_queries_cache_hit_does_not_call_load_dataset(monkeypatch, patch_cache_dir):
    split = "paper_retrieval"
    cache_file = get_data.get_cache_dir(split, None) / "queries.jsonl"

    cached = [
        {"query_id": "q1", "query_content": "cached", "relevant_doc_ids": ["d1"]},
        {"query_id": "q2", "query_content": "cached2", "relevant_doc_ids": ["d2", "d3"]},
    ]
    _write_jsonl(cache_file, cached)

    def boom(*a, **k):
        raise AssertionError("load_dataset should not be called on cache hit")

    monkeypatch.setattr(get_data, "load_dataset", boom)

    out = get_data.load_queries(split)
    assert out == cached


def test_load_queries_cache_miss_writes_cache(monkeypatch, patch_cache_dir):
    split = "stack_exchange"
    cache_file = get_data.get_cache_dir(split, None) / "queries.jsonl"
    assert not cache_file.exists()

    hf_items = [
        {
            "query_id": "q1",
            "query_content": "foo",
            "passage_qrels": [{"id": "d1", "label": 1}, {"id": "d2", "label": 0}],
        },
        {
            "query_id": "q2",
            "query_content": "bar",
            "passage_qrels": [{"id": "d3", "label": -1}],  # filtered out
        },
    ]
    monkeypatch.setattr(get_data, "load_dataset", lambda *a, **k: hf_items)

    out = get_data.load_queries(split)

    assert out == [
        {"query_id": "q1", "query_content": "foo", "relevant_doc_ids": ["d1"]},
    ]
    assert cache_file.exists()

    cached_lines = cache_file.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in cached_lines] == out


def test_load_queries_n_docs_suffix_is_used_for_cache_dir(monkeypatch, patch_cache_dir):
    """
    load_queries(split, n_docs=...) should use the same cache_dir suffix scheme
    as get_cache_dir(split, n_docs).
    """
    split = "clinical_trial"
    n_docs = 123
    cache_file = get_data.get_cache_dir(split, n_docs) / "queries.jsonl"
    assert "clinical_trial_123docs" in str(cache_file)

    hf_items = [
        {
            "query_id": "q1",
            "query_content": "foo",
            "passage_qrels": [{"id": "d1", "label": 1}],
        }
    ]
    monkeypatch.setattr(get_data, "load_dataset", lambda *a, **k: hf_items)

    out = get_data.load_queries(split, n_queries=None, n_docs=n_docs)
    assert cache_file.exists()
    assert out[0]["query_id"] == "q1"


def test_load_full_corpus_cache_hit_does_not_call_load_dataset(monkeypatch, patch_cache_dir):
    split = "code_retrieval"
    max_docs = 10
    cache_file = get_data.get_cache_dir(split, max_docs) / "documents.jsonl"

    cached = [
        {"doc_id": "1", "text": "cached", "metadata": {}},
        {"doc_id": "2", "text": "cached2", "metadata": {"x": 1}},
    ]
    _write_jsonl(cache_file, cached)

    def boom(*a, **k):
        raise AssertionError("load_dataset should not be called on cache hit")

    monkeypatch.setattr(get_data, "load_dataset", boom)

    out = get_data.load_full_corpus_streaming(split, max_docs=max_docs)
    assert out == cached


def test_load_full_corpus_cache_miss_writes_cache_and_respects_limit(monkeypatch, patch_cache_dir):
    split = "tip_of_the_tongue"
    max_docs = 2
    cache_file = get_data.get_cache_dir(split, max_docs) / "documents.jsonl"
    assert not cache_file.exists()

    hf_items = [
        {"document_id": 1, "document_content": "doc1"},
        {"document_id": 2, "document_content": "doc2"},
        {"document_id": 3, "document_content": "doc3"},
    ]

    def fake_load_dataset(*args, **kwargs):
        assert kwargs.get("streaming") is True
        return _StreamingDataset(hf_items)

    monkeypatch.setattr(get_data, "load_dataset", fake_load_dataset)

    out = get_data.load_full_corpus_streaming(split, max_docs=max_docs)

    assert out == [
        {"doc_id": "1", "text": "doc1", "metadata": {}},
        {"doc_id": "2", "text": "doc2", "metadata": {}},
    ]
    assert cache_file.exists()

    cached_lines = cache_file.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in cached_lines] == out


def test_cache_is_split_specific(monkeypatch, patch_cache_dir):
    """
    Ensure caches don't bleed across splits: same n_docs, different splits
    => different cache dirs/files.
    """
    split_a = "paper_retrieval"
    split_b = "legal_qa"
    n_docs = 5

    file_a = get_data.get_cache_dir(split_a, n_docs) / "documents.jsonl"
    file_b = get_data.get_cache_dir(split_b, n_docs) / "documents.jsonl"

    assert file_a != file_b

    _write_jsonl(file_a, [{"doc_id": "a", "text": "A", "metadata": {}}])
    _write_jsonl(file_b, [{"doc_id": "b", "text": "B", "metadata": {}}])

    monkeypatch.setattr(get_data, "load_dataset", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no net")))

    out_a = get_data.load_full_corpus_streaming(split_a, max_docs=n_docs)
    out_b = get_data.load_full_corpus_streaming(split_b, max_docs=n_docs)

    assert out_a[0]["doc_id"] == "a"
    assert out_b[0]["doc_id"] == "b"


def test_cache_is_limit_specific_for_corpus(monkeypatch, patch_cache_dir):
    """
    The corpus cache path includes the max_docs suffix; different limits should
    not collide.
    """
    split = "theorem_retrieval"
    file_5 = get_data.get_cache_dir(split, 5) / "documents.jsonl"
    file_10 = get_data.get_cache_dir(split, 10) / "documents.jsonl"
    assert file_5 != file_10

    _write_jsonl(file_5, [{"doc_id": "5", "text": "five", "metadata": {}}])
    _write_jsonl(file_10, [{"doc_id": "10", "text": "ten", "metadata": {}}])

    monkeypatch.setattr(get_data, "load_dataset", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no net")))

    out_5 = get_data.load_full_corpus_streaming(split, max_docs=5)
    out_10 = get_data.load_full_corpus_streaming(split, max_docs=10)

    assert out_5 == [{"doc_id": "5", "text": "five", "metadata": {}}]
    assert out_10 == [{"doc_id": "10", "text": "ten", "metadata": {}}]


def test_partial_cache_file_is_returned_as_is(monkeypatch, patch_cache_dir):
    """
    Current caching behavior: if the cache file exists, it is read fully and returned
    without validating "expected size" vs requested max_docs.

    This test documents that behavior so changes are intentional.
    """
    split = "code_retrieval"
    max_docs = 100
    cache_file = get_data.get_cache_dir(split, max_docs) / "documents.jsonl"

    # only 1 cached doc even though max_docs=100
    cached = [{"doc_id": "1", "text": "only one", "metadata": {}}]
    _write_jsonl(cache_file, cached)

    monkeypatch.setattr(get_data, "load_dataset", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no net")))

    out = get_data.load_full_corpus_streaming(split, max_docs=max_docs)
    assert out == cached