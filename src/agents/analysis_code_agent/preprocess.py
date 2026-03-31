import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List

from schema import Document, Chunk
from base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    name = "passthrough"
    description = "Simple passthrough preprocessor that converts documents to chunks without modification"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """Convert each document to a single chunk with no preprocessing."""
        chunks = []
        for doc in docs:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_0",
                    doc_id=doc.doc_id,
                    text=doc.text,
                    metadata=doc.metadata,
                )
            )
        return chunks
