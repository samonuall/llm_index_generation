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
        "Improved preprocessor utilizing sentence-based chunking with better overlap handling "
        "and normalization to improve Recall@100 and nDCG@10 in BM25 retrieval."
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
        chunk_size_target = 200  # Target number of tokens per chunk
        overlap_percentage = 0.2  # Target overlap between chunks (percentage of chunk_size_target)

        for doc in docs:
            doc_id = doc.doc_id
            text = doc.text
            
            if not text:
                continue  # Skip documents with empty text
            
            # Clean and normalize text
            text = self.normalize_text(text)
            
            # Tokenize text into sentences
            sentences = sent_tokenize(text)
            total_tokens = sum(len(re.findall(r'\S+', sent)) for sent in sentences)
            
            current_chunk_tokens = []
            chunk_start_token_idx = 0
            sentence_idx = 0

            while sentence_idx < len(sentences):
                # Add sentences to the chunk until we hit the target chunk size
                sentence_tokens = re.findall(r'\S+', sentences[sentence_idx])
                
                if len(current_chunk_tokens) + len(sentence_tokens) <= chunk_size_target or not current_chunk_tokens:
                    current_chunk_tokens.extend(sentence_tokens)
                else:                
                    # Create the chunk text
                    chunk_text = " ".join(current_chunk_tokens)
                    
                    # Generate unique chunk_id
                    chunk_id = f"{doc_id}_{chunk_start_token_idx}_{sentence_idx}"
                    
                    # Append chunk
                    chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text))
                    
                    # Add neighboring overlap percentage carefully.>"
