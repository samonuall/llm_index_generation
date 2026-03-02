import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor
import re

class Preprocessor(BasePreprocessor):
    name = "lite_llm_agent"  # MUST BE EXACTLY THIS - DO NOT MODIFY
    description = (
        "Splits documents into smaller chunks by sentences or fixed sizes, "
        "removes non-alphanumeric characters, and lowercases text for BM25 indexing."
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
        chunk_size = 200  # Fixed maximum size for each chunk
        
        for doc in docs:
            # Step 1: Normalize the text (lowercase and remove non-alphanumeric characters)
            normalized_text = re.sub(r'[^a-zA-Z0-9\s]', '', doc.text).lower()
            
            # Step 2: Split text into chunks of fixed size
            token_list = normalized_text.split()  # Tokenize by whitespace
            for i in range(0, len(token_list), chunk_size):
                chunk_tokens = token_list[i:i+chunk_size]
                chunk_text = " ".join(chunk_tokens)
                
                # Create Chunk object
                chunk_id = f"{doc.doc_id}_chunk_{i // chunk_size}"
                chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=chunk_text))
        
        return chunks
