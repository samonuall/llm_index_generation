"""
Tests for BM25 indexing, retrieval, aggregation, and evaluation metrics.

Owner: [Person 2]

Key things to test:
- BM25Index construction and chunk mapping
- search() returns relevant chunks ranked correctly
- search_documents() aggregation modes: max, sum, avg
- Deterministic tie-breaking (same query twice = same ranking)
- Edge cases: single chunk, query with no matches
- Metric math: manually compute Recall@k and nDCG@10, verify evaluate() matches
- BM25 server endpoints (POST /index, POST /search) if applicable
"""

import pytest
from build_index import BM25Index
from schema import Chunk


# -- Example tests to get started (fill in the rest) --


class TestBM25IndexConstruction:
    def test_chunk_count_matches(self, sample_chunks):
        index = BM25Index(sample_chunks)
        assert len(index._chunks) == len(sample_chunks)

    def test_idx_to_chunk_mapping(self, sample_chunks):
        index = BM25Index(sample_chunks)
        for i, chunk in enumerate(sample_chunks):
            assert index._idx_to_chunk[i].chunk_id == chunk.chunk_id


class TestChunkSearch:
    def test_cat_query_returns_cat_doc(self, built_index):
        results = built_index.search("furry cat mammal retractable claws", top_k=3)
        top_doc_ids = [chunk.doc_id for chunk, _ in results]
        assert "doc_001" in top_doc_ids, f"Expected doc_001 in results, got {top_doc_ids}"

    def test_search_returns_scores_in_descending_order(self, built_index):
        results = built_index.search("planets solar system", top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


class TestDocumentSearch:
    def test_basic_retrieval(self, built_index):
        results = built_index.search_documents("programming language whitespace", top_k=3)
        top_doc_ids = [doc_id for doc_id, _ in results]
        assert "doc_003" in top_doc_ids

    def test_deterministic_ordering(self, built_index):
        """Same query twice must produce identical rankings."""
        q = "mountain highest Nepal Himalayas"
        r1 = built_index.search_documents(q, top_k=5)
        r2 = built_index.search_documents(q, top_k=5)
        assert r1 == r2


class TestAggregationModes:
    def test_max_aggregation(self, multi_chunks):
        """TODO: Build index with agg='max', verify doc scores = max of chunk scores."""
        pass

    def test_sum_aggregation(self, multi_chunks):
        """TODO: Build index with agg='sum', verify doc scores = sum of chunk scores."""
        pass

    def test_avg_aggregation(self, multi_chunks):
        """TODO: Build index with agg='avg', verify doc scores = mean of chunk scores."""
        pass


class TestMetricComputation:
    def test_recall_at_k_perfect(self, built_index, sample_queries):
        """TODO: For queries where BM25 retrieves the right doc in top-10,
        verify recall computation matches manual count."""
        pass

    def test_ndcg_at_10_manual(self):
        """TODO: Construct a known ranking, compute nDCG@10 by hand,
        verify the evaluate() formula matches."""
        pass
