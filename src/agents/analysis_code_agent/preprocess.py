import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "analysis_code_agent"  # MUST BE EXACTLY THIS - DO NOT MODIFY
    description = "starting preprocessor that does a simple passthrough."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks = []
        for doc in docs:
            chunk_id = f"{doc.doc_id}_0"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=augmented_text,
                metadata=doc.metadata
            ))
        return chunks
