import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    name = "analysis_code_agent"  # MUST BE EXACTLY THIS - DO NOT MODIFY
    description = "Repeat the first 200 words of each document at the end to boost TF for key intro terms."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks = []
        for doc in docs:
            text = doc.text
            words = text.split()
            first_200 = " ".join(words[:200])
            # Append the first 200 words at the end to boost TF for intro terms
            augmented_text = text + "\n\n" + first_200

            chunk_id = f"{doc.doc_id}_0"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=augmented_text,
                metadata=doc.metadata
            ))
        return chunks
