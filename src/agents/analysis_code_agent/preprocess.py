"""
Baseline preprocessor: one chunk per document, raw text, no modification.

Starting point for the analysis_code_agent — will be iteratively improved by the agent.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List

from schema import Document, Chunk
from base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    name = "analysis_code_agent"
    description = "Passthrough – one chunk per document, raw text, no modification."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}_0",
                doc_id=doc.doc_id,
                text=doc.text,
            )
            for doc in docs
        ]
