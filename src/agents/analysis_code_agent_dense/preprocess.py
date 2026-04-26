import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    name = "analysis_code_agent_dense"
    description = "Passthrough preprocessor: emits a single chunk per document containing the full original text."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in docs:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_0",
                doc_id=doc.doc_id,
                text=doc.text,
                metadata=doc.metadata,
            ))
        return chunks
