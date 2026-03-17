# Preprocessing Code Requirements

You are writing Python code to preprocess documents for BM25 retrieval. Your goal is to maximize Recall@k and nDCG@k metrics.

## CRITICAL: Required Code Structure

You MUST generate Python code with this EXACT structure:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "lite_llm_agent"  # MUST BE EXACTLY THIS - DO NOT MODIFY
    description = "Brief description of your preprocessing approach"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Transform documents into chunks for BM25 indexing.
        
        Args:
            docs: List of Document objects with .doc_id and .text fields
            
        Returns:
            List of Chunk objects with unique chunk_id and matching doc_id
        """
        # Your implementation here
        pass