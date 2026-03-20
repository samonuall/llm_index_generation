import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    name = "combined_title_prepend_and_emphasized"
    description = "Combines title prepending, overlapping sliding windows, and title/entity emphasis."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks = []
        grouped_docs = {}

        # Group documents by their article ID (everything before `:` in doc_id)
        for doc in docs:
            article_id = doc.doc_id.split(":")[0]
            grouped_docs.setdefault(article_id, []).append(doc)

        for group in grouped_docs.values():
            intro_text = None

            # Extract introduction text (if exists)
            for doc in group:
                if doc.doc_id.endswith(":0"):
                    intro_text = doc.text
                    break

            for doc in group:
                base_text = f"{intro_text}\n\n{doc.text}" if intro_text else doc.text

                # Create original chunk
                original_chunk_id = f"{doc.doc_id}_0"
                chunks.append(Chunk(chunk_id=original_chunk_id, doc_id=doc.doc_id, text=doc.text))

                # Create title/emphasized chunk if intro text exists
                if intro_text:
                    emphasized_chunk_id = f"{doc.doc_id}_emphasized"
                    chunks.append(Chunk(chunk_id=emphasized_chunk_id, doc_id=doc.doc_id, text=base_text))

                # Apply overlapping sliding windows to the text
                words = base_text.split()
                window_size = 250
                overlap = 100
                start = 0

                while start < len(words):
                    text_chunk = " ".join(words[start:start + window_size])
                    sliding_chunk_id = f"{doc.doc_id}_window_{start//100}"
                    chunks.append(Chunk(chunk_id=sliding_chunk_id, doc_id=doc.doc_id, text=text_chunk))
                    start += window_size - overlap

        return chunks
