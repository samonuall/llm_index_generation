"""
test_data_storage.py — Tests for saving/loading JSONL data files.

Tests JSONL write/read operations for documents, queries, and chunks.
"""

import json
import pytest
from pathlib import Path
from typing import List

from src.evaluation.schema import Document, Chunk, EvalQuery


class TestDocumentStorage:
    """Test saving and loading Document objects to/from JSONL."""

    def test_write_documents_to_jsonl(self, tmp_path, sample_documents):
        """Write documents to JSONL and verify file format."""
        output_file = tmp_path / "documents.jsonl"
        
        # Write documents
        with output_file.open("w") as f:
            for doc in sample_documents:
                f.write(json.dumps({
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }) + "\n")
        
        # Verify file exists
        assert output_file.exists()
        
        # Verify line count
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == len(sample_documents)
        
        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "doc_id" in data
            assert "text" in data
            assert "metadata" in data

    def test_read_documents_from_jsonl(self, tmp_path, sample_documents):
        """Read documents from JSONL and reconstruct Document objects."""
        output_file = tmp_path / "documents.jsonl"
        
        # Write sample data
        with output_file.open("w") as f:
            for doc in sample_documents:
                f.write(json.dumps({
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }) + "\n")
        
        # Read back
        loaded_docs = []
        with output_file.open("r") as f:
            for line in f:
                data = json.loads(line)
                loaded_docs.append(Document(**data))
        
        # Verify count and content
        assert len(loaded_docs) == len(sample_documents)
        assert loaded_docs[0].doc_id == sample_documents[0].doc_id
        assert loaded_docs[0].text == sample_documents[0].text

    def test_write_empty_documents(self, tmp_path):
        """Handle writing empty document list."""
        output_file = tmp_path / "empty_documents.jsonl"
        
        with output_file.open("w") as f:
            for doc in []:
                f.write(json.dumps({
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }) + "\n")
        
        assert output_file.exists()
        assert output_file.read_text() == ""

    def test_document_with_special_characters(self, tmp_path):
        """Handle documents with special characters and unicode."""
        output_file = tmp_path / "special_docs.jsonl"
        
        special_doc = Document(
            doc_id="doc_special",
            text='Text with "quotes", newlines\n, and emoji 🚀',
            metadata={"key": "value with 中文"}
        )
        
        # Write
        with output_file.open("w") as f:
            f.write(json.dumps({
                "doc_id": special_doc.doc_id,
                "text": special_doc.text,
                "metadata": special_doc.metadata,
            }) + "\n")
        
        # Read back
        with output_file.open("r") as f:
            loaded = json.loads(f.readline())
        
        assert loaded["text"] == special_doc.text
        assert loaded["metadata"]["key"] == special_doc.metadata["key"]

    def test_overwrite_existing_file(self, tmp_path, sample_documents):
        """Overwriting a file replaces old content."""
        output_file = tmp_path / "documents.jsonl"
        
        # Write first batch
        with output_file.open("w") as f:
            f.write(json.dumps({"doc_id": "old", "text": "old", "metadata": {}}) + "\n")
        
        # Overwrite with sample documents
        with output_file.open("w") as f:
            for doc in sample_documents:
                f.write(json.dumps({
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }) + "\n")
        
        # Verify only new content exists
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == len(sample_documents)
        assert "old" not in lines[0]


class TestQueryStorage:
    """Test saving and loading EvalQuery objects to/from JSONL."""

    def test_write_queries_to_jsonl(self, tmp_path, sample_queries):
        """Write queries to JSONL and verify format."""
        output_file = tmp_path / "queries.jsonl"
        
        with output_file.open("w") as f:
            for q in sample_queries:
                f.write(json.dumps({
                    "query_id": q.query_id,
                    "query_content": q.query_text,
                    "relevant_doc_ids": q.relevant_doc_ids,
                }) + "\n")
        
        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == len(sample_queries)
        
        # Verify structure
        first_query = json.loads(lines[0])
        assert "query_id" in first_query
        assert "query_content" in first_query
        assert "relevant_doc_ids" in first_query
        assert isinstance(first_query["relevant_doc_ids"], list)

    def test_read_queries_from_jsonl(self, tmp_path, sample_queries):
        """Read queries from JSONL and reconstruct EvalQuery objects."""
        output_file = tmp_path / "queries.jsonl"
        
        # Write
        with output_file.open("w") as f:
            for q in sample_queries:
                f.write(json.dumps({
                    "query_id": q.query_id,
                    "query_content": q.query_text,
                    "relevant_doc_ids": q.relevant_doc_ids,
                }) + "\n")
        
        # Read back (matching conftest.py pattern)
        loaded_queries = []
        with output_file.open("r") as f:
            for line in f:
                data = json.loads(line)
                loaded_queries.append(EvalQuery(
                    query_id=data["query_id"],
                    query_text=data["query_content"],
                    relevant_doc_ids=data["relevant_doc_ids"],
                ))
        
        assert len(loaded_queries) == len(sample_queries)
        assert loaded_queries[0].query_id == sample_queries[0].query_id
        assert loaded_queries[0].relevant_doc_ids == sample_queries[0].relevant_doc_ids

    def test_query_with_multiple_relevant_docs(self, tmp_path):
        """Handle queries with multiple relevant document IDs."""
        output_file = tmp_path / "multi_queries.jsonl"
        
        query = EvalQuery(
            query_id="q_multi",
            query_text="Test query",
            relevant_doc_ids=["doc1", "doc2", "doc3", "doc4"]
        )
        
        # Write
        with output_file.open("w") as f:
            f.write(json.dumps({
                "query_id": query.query_id,
                "query_content": query.query_text,
                "relevant_doc_ids": query.relevant_doc_ids,
            }) + "\n")
        
        # Read
        with output_file.open("r") as f:
            loaded = json.loads(f.readline())
        
        assert len(loaded["relevant_doc_ids"]) == 4
        assert loaded["relevant_doc_ids"] == query.relevant_doc_ids

    def test_query_with_empty_relevant_docs(self, tmp_path):
        """Handle queries with no relevant documents."""
        output_file = tmp_path / "empty_rel_queries.jsonl"
        
        query = EvalQuery(
            query_id="q_empty",
            query_text="Query with no relevant docs",
            relevant_doc_ids=[]
        )
        
        # Write
        with output_file.open("w") as f:
            f.write(json.dumps({
                "query_id": query.query_id,
                "query_content": query.query_text,
                "relevant_doc_ids": query.relevant_doc_ids,
            }) + "\n")
        
        # Read
        with output_file.open("r") as f:
            loaded = json.loads(f.readline())
        
        assert loaded["relevant_doc_ids"] == []


class TestChunkStorage:
    """Test saving and loading Chunk objects to/from JSONL."""

    def test_write_chunks_to_jsonl(self, tmp_path, sample_chunks):
        """Write chunks to JSONL and verify format."""
        output_file = tmp_path / "chunks.jsonl"
        
        with output_file.open("w") as f:
            for chunk in sample_chunks:
                f.write(json.dumps({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }) + "\n")
        
        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == len(sample_chunks)

    def test_read_chunks_from_jsonl(self, tmp_path, sample_chunks):
        """Read chunks from JSONL and reconstruct Chunk objects."""
        output_file = tmp_path / "chunks.jsonl"
        
        # Write
        with output_file.open("w") as f:
            for chunk in sample_chunks:
                f.write(json.dumps({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }) + "\n")
        
        # Read
        loaded_chunks = []
        with output_file.open("r") as f:
            for line in f:
                loaded_chunks.append(Chunk(**json.loads(line)))
        
        assert len(loaded_chunks) == len(sample_chunks)
        assert loaded_chunks[0].chunk_id == sample_chunks[0].chunk_id
        assert loaded_chunks[0].doc_id == sample_chunks[0].doc_id


class TestDirectoryManagement:
    """Test directory creation and path handling."""

    def test_create_nested_directories(self, tmp_path):
        """Automatically create nested directories for storage."""
        nested_dir = tmp_path / "data" / "test_split" / "subdirectory"
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = nested_dir / "documents.jsonl"
        output_file.write_text('{"doc_id": "test", "text": "test", "metadata": {}}\n')
        
        assert output_file.exists()
        assert output_file.parent.exists()

    def test_mkdir_idempotent(self, tmp_path):
        """mkdir with exist_ok=True doesn't fail if directory exists."""
        target_dir = tmp_path / "my_dir"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)  # Should not raise
        
        assert target_dir.exists()
        assert target_dir.is_dir()

    def test_storage_in_split_subdirectories(self, tmp_path):
        """Store data in split-specific subdirectories."""
        splits = ["paper_retrieval", "tip_of_the_tongue", "code_retrieval"]
        
        for split in splits:
            split_dir = tmp_path / "data" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            doc_file = split_dir / "documents.jsonl"
            doc_file.write_text(f'{{"split": "{split}"}}\n')
            
            assert doc_file.exists()


class TestErrorHandling:
    """Test error conditions and edge cases."""

    def test_malformed_json_line(self, tmp_path):
        """Reading malformed JSON raises JSONDecodeError."""
        output_file = tmp_path / "malformed.jsonl"
        output_file.write_text("not valid json\n")
        
        with pytest.raises(json.JSONDecodeError):
            with output_file.open("r") as f:
                json.loads(f.readline())

    def test_missing_required_field(self, tmp_path):
        """Reading data with missing required field raises error."""
        output_file = tmp_path / "incomplete.jsonl"
        output_file.write_text('{"doc_id": "test"}\n')  # Missing 'text'
        
        with pytest.raises((KeyError, TypeError)):
            with output_file.open("r") as f:
                Document(**json.loads(f.readline()))

    def test_write_to_readonly_directory(self, tmp_path):
        """Writing to read-only directory raises PermissionError."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        try:
            output_file = readonly_dir / "test.jsonl"
            with pytest.raises(PermissionError):
                with output_file.open("w") as f:
                    f.write("{}\n")
        finally:
            readonly_dir.chmod(0o755)  # Restore permissions for cleanup

    def test_read_nonexistent_file(self, tmp_path):
        """Reading non-existent file raises FileNotFoundError."""
        missing_file = tmp_path / "does_not_exist.jsonl"
        
        with pytest.raises(FileNotFoundError):
            with missing_file.open("r") as f:
                f.read()


class TestRoundTripConsistency:
    """Test that data survives write → read cycles without corruption."""

    def test_document_roundtrip(self, tmp_path, sample_documents):
        """Documents survive write → read cycle unchanged."""
        output_file = tmp_path / "roundtrip_docs.jsonl"
        
        # Write
        with output_file.open("w") as f:
            for doc in sample_documents:
                f.write(json.dumps({
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }) + "\n")
        
        # Read
        loaded_docs = []
        with output_file.open("r") as f:
            for line in f:
                loaded_docs.append(Document(**json.loads(line)))
        
        # Compare
        for original, loaded in zip(sample_documents, loaded_docs):
            assert original.doc_id == loaded.doc_id
            assert original.text == loaded.text
            assert original.metadata == loaded.metadata

    def test_query_roundtrip(self, tmp_path, sample_queries):
        """Queries survive write → read cycle unchanged."""
        output_file = tmp_path / "roundtrip_queries.jsonl"
        
        # Write
        with output_file.open("w") as f:
            for q in sample_queries:
                f.write(json.dumps({
                    "query_id": q.query_id,
                    "query_content": q.query_text,
                    "relevant_doc_ids": q.relevant_doc_ids,
                }) + "\n")
        
        # Read
        loaded_queries = []
        with output_file.open("r") as f:
            for line in f:
                data = json.loads(line)
                loaded_queries.append(EvalQuery(
                    query_id=data["query_id"],
                    query_text=data["query_content"],
                    relevant_doc_ids=data["relevant_doc_ids"],
                ))
        
        # Compare
        for original, loaded in zip(sample_queries, loaded_queries):
            assert original.query_id == loaded.query_id
            assert original.query_text == loaded.query_text
            assert original.relevant_doc_ids == loaded.relevant_doc_ids