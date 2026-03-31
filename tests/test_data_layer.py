"""
Tests for the data layer: schema classes, JSONL I/O, chunk contract validation.

Owner: [Person 1]

Key things to test:
- Document / Chunk / EvalQuery construction and field access
- JSONL round-trip (write -> read -> compare)
- _load_documents() and _load_queries() with missing/empty/malformed files
- Chunk contracts: every doc_id exists in source docs, all chunk_ids unique,
  no empty text, at least 1 chunk per document
- Type coercion: doc_ids should always be strings
"""

from schema import Document, Chunk, EvalQuery


# -- Example tests to get started (fill in the rest) --


class TestDocumentConstruction:
    def test_basic_fields(self):
        doc = Document(doc_id="123", text="hello world", metadata={"key": "val"})
        assert doc.doc_id == "123"
        assert doc.text == "hello world"
        assert doc.metadata == {"key": "val"}

    def test_metadata_defaults_to_empty_dict(self):
        doc = Document(doc_id="1", text="t")
        assert doc.metadata == {}


class TestChunkContracts:
    """Validate the invariants that preprocess() output must satisfy."""

    def test_all_docs_have_at_least_one_chunk(self, sample_documents, sample_chunks):
        doc_ids_in = {d.doc_id for d in sample_documents}
        doc_ids_out = {c.doc_id for c in sample_chunks}
        assert doc_ids_in == doc_ids_out

    def test_chunk_ids_are_unique(self, sample_chunks):
        ids = [c.chunk_id for c in sample_chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_no_empty_chunk_text(self, sample_chunks):
        for c in sample_chunks:
            assert c.text.strip(), f"Chunk {c.chunk_id} has empty text"

    def test_doc_ids_are_strings(self, sample_chunks):
        for c in sample_chunks:
            assert isinstance(c.doc_id, str), f"Chunk {c.chunk_id} doc_id is not a string"


class TestJsonlRoundTrip:
    """Test that writing and reading JSONL preserves data."""

    def test_documents_round_trip(self, tmp_data_dir):
        """TODO: Read documents from tmp_data_dir/test_split/documents.jsonl
        and verify they match the originals."""
        pass

    def test_queries_round_trip(self, tmp_data_dir):
        """TODO: Read queries from tmp_data_dir/test_split/queries.jsonl
        and verify they match the originals."""
        pass


class TestLoadEdgeCases:
    """Test _load_documents / _load_queries with bad input."""

    def test_missing_file_raises(self, tmp_path):
        """TODO: Call _load_documents with a nonexistent split, expect FileNotFoundError."""
        pass

    def test_empty_file(self, tmp_path):
        """TODO: Create an empty documents.jsonl, verify behavior."""
        pass
