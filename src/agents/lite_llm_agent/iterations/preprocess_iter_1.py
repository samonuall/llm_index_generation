import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor
import re

class Preprocessor(BasePreprocessor):
    name = "lite_llm_agent"  # MUST BE EXACTLY THIS - DO NOT MODIFY
    description = (
        "Preprocessor that splits documents into overlapping, token-based chunks "
        "to improve BM25 retrieval effectiveness."
    )

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Transform documents into chunks for BM25 indexing.
        
        Args:
            docs: List of Document objects with .doc_id and .text fields
            
        Returns:
            List of Chunk objects with unique chunk_id and matching doc_id
        """
        chunks = []
        chunk_size = 256  # Target number of tokens per chunk
        overlap_size = 50  # Overlap size between chunks (number of tokens)
        
        for doc in docs:
            doc_id = doc.doc_id
            text = doc.text

            if not text:
                continue  # Skip documents with empty text
            
            # Tokenize text (simple whitespace tokenization)
            tokens = re.findall(r'\S+', text)
            total_tokens = len(tokens)
            
            # Generate overlapping chunks
            chunk_start = 0
            while chunk_start < total_tokens:
                chunk_end = min(chunk_start + chunk_size, total_tokens)
                chunk_tokens = tokens[chunk_start:chunk_end]
                chunk_text = " ".join(chunk_tokens)
                
                # Generate unique chunk_id
                chunk_id = f"{doc_id}_{chunk_start}"
                
                # Append chunk
                chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text))
                
                # Move sliding window forward
                chunk_start += chunk_size - overlap_size  # Adjust by overlap

        return chunks
