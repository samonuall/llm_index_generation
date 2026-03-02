import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor
import re
from nltk.tokenize import sent_tokenize

class Preprocessor(BasePreprocessor):
    name = "lite_llm_agent"  # MUST BE EXACTLY THIS - DO NOT MODIFY
    description = (
        "Splits document text into semantically coherent chunks based on sentence boundaries, "
        "uses normalization (lowercasing and removing non-alphanumeric characters), "
        "and combines sentences ensuring chunks are within a token limit."
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
        max_tokens_per_chunk = 150  # Limit the number of tokens in each chunk for efficiency
        
        for doc in docs:
            # Step 1: Normalize the text (lowercase and remove non-alphanumeric characters while preserving spaces)
            normalized_text = re.sub(r'[^a-zA-Z0-9\s]', '', doc.text).lower()
            
            # Step 2: Split text into sentences
            sentences = sent_tokenize(normalized_text)

            # Step 3: Group sentences to form chunks while adhering to token limit
            current_chunk_tokens = []
            chunk_counter = 0
            
            for sentence in sentences:
                sentence_tokens = sentence.split()  # Tokenize the sentence
                
                # Check if adding the current sentence exceeds the token limit
                if len(current_chunk_tokens) + len(sentence_tokens) > max_tokens_per_chunk:
                    # Create a chunk
                    chunk_text = " ".join(current_chunk_tokens)
                    chunk_id = f"{doc.doc_id}_chunk_{chunk_counter}"
                    chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=chunk_text))
                    
                    # Reset for the next chunk
                    current_chunk_tokens = []
                    chunk_counter += 1
                
                # Add sentence tokens to the current chunk
                current_chunk_tokens.extend(sentence_tokens)
            
            # Add the last chunk if there's remaining content
            if current_chunk_tokens:
                chunk_text = " ".join(current_chunk_tokens)
                chunk_id = f"{doc.doc_id}_chunk_{chunk_counter}"
                chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=chunk_text))
        
        return chunks
