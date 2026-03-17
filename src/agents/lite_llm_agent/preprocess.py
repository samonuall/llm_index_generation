import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "lite_llm_agent"  # MUST BE EXACTLY THIS - DO NOT MODIFY
    description = "Splits documents into chunks for indexing and retrieval. Removes stopwords, converts text to lowercase, and performs basic tokenization."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Transform documents into chunks for BM25 indexing.
        
        Args:
            docs: List of Document objects with .doc_id and .text fields
            
        Returns:
            List of Chunk objects with unique chunk_id and matching doc_id
        """
        def basic_tokenizer(text: str) -> List[str]:
            """Tokenizes text by splitting on whitespace and basic punctuation."""
            return text.replace("\n", " ").lower().split()

        def remove_stopwords(words: List[str]) -> List[str]:
            """Removes common English stopwords."""
            # Minimal stopword list to avoid external library dependency
            stopwords = set([
                'a', 'an', 'the', 'and', 'or', 'but', 'so', 'with', 'is', 
                'are', 'was', 'were', 'be', 'to', 'of', 'in', 'on', 'for', 
                'by', 'at', 'it', 'this', 'that', 'these', 'those', 'there'
            ])
            return [word for word in words if word not in stopwords]

        chunks = []
        for doc in docs:
            words = basic_tokenizer(doc.text)
            filtered_words = remove_stopwords(words)
            processed_text = " ".join(filtered_words)

            # Split text into chunks for better retrieval performance
            chunk_size = 200  # Approximate number of words per chunk
            for i in range(0, len(filtered_words), chunk_size):
                chunk_text = " ".join(filtered_words[i:i+chunk_size])
                chunk_id = f"{doc.doc_id}_chunk_{i // chunk_size}"
                chunks.append(Chunk(doc_id=doc.doc_id, chunk_id=chunk_id, text=chunk_text))

        return chunks
