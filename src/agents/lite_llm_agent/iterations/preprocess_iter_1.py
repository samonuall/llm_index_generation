"""
Preprocessing template for lite_llm_agent.
Split into semantically coherent chunks for BM25 retrieval.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    name = "lite_llm_agent"
    description = "Document segmentation into smaller, cohesive chunks to improve retrieval granularity"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Split documents into semantically coherent chunks.
        
        Args:
            docs: List of Document objects with .doc_id and .text fields
            
        Returns:
            List of Chunk objects with unique chunk_id and matching doc_id
        """
        chunks = []

        for doc in docs:
            # Split the document into sentences
            sentences = doc.text.split(". ")  # Basic sentence splitting using periods
            
            # Group sentences into cohesive chunks with a target limit on text length
            current_chunk = []
            current_length = 0
            chunk_index = 0

            for sentence in sentences:
                sentence = sentence.strip()  # Clean up whitespace
                sentence_length = len(sentence)

                # If the chunk exceeds the target size, finalize it and start a new chunk
                if current_length + sentence_length > 300:  # Adjust length threshold
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}_{chunk_index}",
                            doc_id=doc.doc_id,
                            text=chunk_text
                        )
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0

                # Add the sentence to the current chunk
                current_chunk.append(sentence)
                current_length += sentence_length

            # Create a chunk for any remaining sentences
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_{chunk_index}",
                        doc_id=doc.doc_id,
                        text=chunk_text
                    )
                )

        return chunks
