"""
Test data loading functionality for CRUMB dataset.

Tests the src.evaluation.scripts.get_data module including:
- Query loading and caching
- Corpus streaming and caching
- Error handling and edge cases
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List, Dict

from src.evaluation.scripts.get_data import (
    load_queries,
    load_full_corpus_streaming,
    SPLIT_MAP,
)


class TestQueryLoading:
    """Test query loading and caching."""
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_load_queries_from_dataset(self, mock_load_dataset, tmp_path):
        """Test loading queries from HuggingFace dataset."""
        # Mock dataset response
        mock_dataset = [
            {
                "query_id": "q1",
                "query_content": "What is machine learning?",
                "full_document_qrels": [{"id": "doc1", "label": 1}],
            },
            {
                "query_id": "q2",
                "query_content": "Python tutorial",
                "full_document_qrels": [{"id": "doc2", "label": 1}],
            }
        ]
        mock_load_dataset.return_value = mock_dataset
        
        # Mock cache directory
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            queries = load_queries("paper_retrieval")
            
            assert len(queries) == 2
            assert queries[0]["query_id"] == "q1"
            assert queries[0]["query_content"] == "What is machine learning?"
            assert "doc1" in queries[0]["relevant_doc_ids"]
    
    def test_load_queries_from_cache(self, tmp_path):
        """Test loading queries from cached file."""
        # Create mock cache file
        cache_dir = tmp_path / "test_split"
        cache_dir.mkdir()
        cache_file = cache_dir / "queries.jsonl"
        
        test_queries = [
            {"query_id": "q1", "query_content": "test query", "relevant_doc_ids": ["doc1"]},
            {"query_id": "q2", "query_content": "another query", "relevant_doc_ids": ["doc2"]},
        ]
        
        with cache_file.open("w") as f:
            for q in test_queries:
                f.write(json.dumps(q) + "\n")
        
        # Mock get_cache_dir to return our tmp_path
        with patch('src.evaluation.scripts.get_data.get_cache_dir', return_value=cache_dir):
            queries = load_queries("paper_retrieval")
            
            assert len(queries) == 2
            assert queries[0]["query_id"] == "q1"
            assert queries[1]["query_id"] == "q2"
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_load_queries_with_limit(self, mock_load_dataset, tmp_path):
        """Test loading limited number of queries."""
        # Create 10 mock queries
        mock_dataset = [
            {
                "query_id": f"q{i}",
                "query_content": f"query {i}",
                "full_document_qrels": [{"id": f"doc{i}", "label": 1}],
            }
            for i in range(10)
        ]
        mock_load_dataset.return_value = mock_dataset
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            queries = load_queries("paper_retrieval", n_queries=3)
            
            assert len(queries) == 3
            assert queries[0]["query_id"] == "q0"
            assert queries[2]["query_id"] == "q2"
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_load_queries_filters_empty_qrels(self, mock_load_dataset, tmp_path):
        """Test that queries without relevant documents are filtered out."""
        mock_dataset = [
            {
                "query_id": "q1",
                "query_content": "valid query",
                "full_document_qrels": [{"id": "doc1", "label": 1}],
            },
            {
                "query_id": "q2",
                "query_content": "no relevant docs",
                "full_document_qrels": [{"id": "doc2", "label": 0}],  # label=0, should be filtered
            },
            {
                "query_id": "q3",
                "query_content": "empty qrels",
                "full_document_qrels": [],  # empty, should be filtered
            },
        ]
        mock_load_dataset.return_value = mock_dataset
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            queries = load_queries("paper_retrieval")
            
            # Only q1 should be included
            assert len(queries) == 1
            assert queries[0]["query_id"] == "q1"


class TestCorpusLoading:
    """Test corpus loading with streaming."""
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_load_corpus_streaming(self, mock_load_dataset, tmp_path):
        """Test loading corpus using streaming mode."""
        # Mock streaming dataset
        mock_docs = [
            {"document_id": "doc1", "document_content": "Content 1"},
            {"document_id": "doc2", "document_content": "Content 2"},
            {"document_id": "doc3", "document_content": "Content 3"},
        ]
        mock_load_dataset.return_value = iter(mock_docs)
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            docs = load_full_corpus_streaming("paper_retrieval")
            
            assert len(docs) == 3
            assert docs[0]["doc_id"] == "doc1"
            assert docs[0]["text"] == "Content 1"
            assert "metadata" in docs[0]
    
    def test_load_corpus_from_cache(self, tmp_path):
        """Test loading corpus from cached file."""
        cache_dir = tmp_path / "test_split"
        cache_dir.mkdir()
        cache_file = cache_dir / "documents.jsonl"
        
        test_docs = [
            {"doc_id": "doc1", "text": "Test document 1", "metadata": {}},
            {"doc_id": "doc2", "text": "Test document 2", "metadata": {}},
        ]
        
        with cache_file.open("w") as f:
            for doc in test_docs:
                f.write(json.dumps(doc) + "\n")
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir', return_value=cache_dir):
            docs = load_full_corpus_streaming("paper_retrieval")
            
            assert len(docs) == 2
            assert docs[0]["doc_id"] == "doc1"
            assert docs[1]["doc_id"] == "doc2"
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_load_corpus_with_limit(self, mock_load_dataset, tmp_path):
        """Test loading corpus with document limit."""
        # Mock 100 documents but we'll limit to 10
        mock_docs = [
            {"document_id": f"doc{i}", "document_content": f"Content {i}"}
            for i in range(100)
        ]
        mock_load_dataset.return_value = iter(mock_docs)
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            docs = load_full_corpus_streaming("paper_retrieval", max_docs=10)
            
            assert len(docs) == 10
            assert docs[0]["doc_id"] == "doc0"
            assert docs[9]["doc_id"] == "doc9"
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_load_corpus_handles_large_datasets(self, mock_load_dataset, tmp_path):
        """Test that large datasets are handled efficiently."""
        # Mock 100k documents
        def doc_generator():
            for i in range(100000):
                yield {"document_id": f"doc{i}", "document_content": f"Content {i}"}
        
        mock_load_dataset.return_value = doc_generator()
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            # Load with limit to keep test fast
            docs = load_full_corpus_streaming("paper_retrieval", max_docs=1000)
            
            assert len(docs) == 1000


class TestSplitConfiguration:
    """Test split configuration and validation."""
    
    def test_split_map_contains_all_splits(self):
        """Test that SPLIT_MAP contains all expected splits."""
        expected_splits = [
            "tip_of_the_tongue",
            "paper_retrieval",
            "stack_exchange",
            "clinical_trial",
            "legal_qa",
            "theorem_retrieval",
            "code_retrieval",
            "set_operation_entity_retrieval",
        ]
        
        for split in expected_splits:
            assert split in SPLIT_MAP
            assert SPLIT_MAP[split] == split  # They map to themselves
    


class TestDataIntegrity:
    """Test data integrity and validation."""
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_query_structure_validation(self, mock_load_dataset, tmp_path):
        """Test that loaded queries have correct structure."""
        mock_dataset = [
            {
                "query_id": "q1",
                "query_content": "test query",
                "full_document_qrels": [{"id": "doc1", "label": 1}],
            }
        ]
        mock_load_dataset.return_value = mock_dataset
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            queries = load_queries("paper_retrieval")
            
            # Validate structure
            assert all("query_id" in q for q in queries)
            assert all("query_content" in q for q in queries)
            assert all("relevant_doc_ids" in q for q in queries)
            assert all(isinstance(q["relevant_doc_ids"], list) for q in queries)
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_document_structure_validation(self, mock_load_dataset, tmp_path):
        """Test that loaded documents have correct structure."""
        mock_docs = [
            {"document_id": "doc1", "document_content": "Content 1"},
        ]
        mock_load_dataset.return_value = iter(mock_docs)
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            docs = load_full_corpus_streaming("paper_retrieval")
            
            # Validate structure
            assert all("doc_id" in doc for doc in docs)
            assert all("text" in doc for doc in docs)
            assert all("metadata" in doc for doc in docs)
            assert all(isinstance(doc["metadata"], dict) for doc in docs)
    
    def test_cache_file_format_jsonl(self, tmp_path):
        """Test that cache files are valid JSONL format."""
        cache_dir = tmp_path / "test_split"
        cache_dir.mkdir()
        cache_file = cache_dir / "queries.jsonl"
        
        # Write test data
        test_data = [
            {"query_id": "q1", "query_content": "test", "relevant_doc_ids": []},
            {"query_id": "q2", "query_content": "test2", "relevant_doc_ids": []},
        ]
        
        with cache_file.open("w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        
        # Read and validate
        with cache_file.open("r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            
            # Each line should be valid JSON
            for line in lines:
                data = json.loads(line)
                assert isinstance(data, dict)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_empty_dataset_handling(self, mock_load_dataset, tmp_path):
        """Test handling of empty dataset."""
        mock_load_dataset.return_value = []
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            queries = load_queries("paper_retrieval")
            assert len(queries) == 0
    
    @patch('src.evaluation.scripts.get_data.load_dataset')
    def test_malformed_query_handling(self, mock_load_dataset, tmp_path):
        """Test handling of malformed queries."""
        mock_dataset = [
            {
                "query_id": "q1",
                "query_content": "valid",
                "full_document_qrels": [{"id": "doc1", "label": 1}],
            },
            {
                "query_id": "q2",
                # Missing query_content
                "full_document_qrels": [{"id": "doc2", "label": 1}],
            },
        ]
        mock_load_dataset.return_value = mock_dataset
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir') as mock_cache:
            cache_dir = tmp_path / "test_split"
            cache_dir.mkdir()
            mock_cache.return_value = cache_dir
            
            # Should handle gracefully (might raise or filter)
            try:
                queries = load_queries("paper_retrieval")
                # If it doesn't raise, check that it filtered appropriately
                assert all("query_content" in q for q in queries)
            except (KeyError, ValueError):
                # Expected if validation is strict
                pass
    
    def test_corrupted_cache_file(self, tmp_path):
        """Test handling of corrupted cache file."""
        cache_dir = tmp_path / "test_split"
        cache_dir.mkdir()
        cache_file = cache_dir / "queries.jsonl"
        
        # Write invalid JSON
        with cache_file.open("w") as f:
            f.write("not valid json\n")
            f.write('{"valid": "json"}\n')
        
        with patch('src.evaluation.scripts.get_data.get_cache_dir', return_value=cache_dir):
            with pytest.raises(json.JSONDecodeError):
                load_queries("paper_retrieval")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])