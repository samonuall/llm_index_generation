import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "gemini_sdk"
    description = "Sliding window (200/60) with title-header injection. Removed lead-text from header to improve discriminative ranking and reduce clumping."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        all_chunks = []
        # Window size 200 provides a broader semantic context than smaller windows.
        # Reducing overlap from 50% to 30% reduces the redundancy of results in the top-k list.
        chunk_size = 200
        overlap = 60

        for doc in docs:
            # Extract the title from metadata to use as a contextual anchor.
            title = ""
            if doc.metadata and "title" in doc.metadata:
                title = str(doc.metadata["title"])
            
            content = doc.text or ""
            words = content.split()
            
            # Repetitive title injection ensures the entity name is weighted in BM25.
            # We omit the first sentence/lead from the header to prevent similar documents 
            # from sharing too many identical terms across all their chunks.
            header = f"{title} {title}"
            
            if not words:
                # Requirement: Return at least one Chunk per Document.
                all_chunks.append(Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}_0",
                    text=title
                ))
                continue

            start = 0
            chunk_idx = 0
            while True:
                end = start + chunk_size
                chunk_slice = words[start:end]
                
                # Prepend the title-based header to provide global document context to the local slice.
                chunk_text = f"{header} {' '.join(chunk_slice)}".strip()
                
                all_chunks.append(Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}_{chunk_idx}",
                    text=chunk_text
                ))
                
                chunk_idx += 1
                
                # Break condition ensures we cover the entire document.
                if end >= len(words):
                    break
                
                # Advance the window by (size - overlap).
                start += (chunk_size - overlap)
                    
        return all_chunks
