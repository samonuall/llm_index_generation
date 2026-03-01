import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "lite_llm_agent"
    description = "LiteLLM agent-generated preprocessor"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Simple baseline: one chunk per document with raw text.
        The agent will iteratively improve this.
        """
        chunks = []
        for doc in docs:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_0",
                doc_id=doc.doc_id,
                text=doc.text
            ))
        return chunks
