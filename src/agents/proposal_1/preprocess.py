import sys, pathlib
sys.path.insert(0, str(path_to_this_file.parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "proposal_1"
    description = "Preprocesses full Wikipedia articles into smaller, overlapping text chunks for BM25 indexing to improve retrieval ranking."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Preprocesses a list of Documents for BM25 indexing by splitting them
        into smaller, overlapping text chunks.

        This strategy aims to improve BM25 retrieval ranking (MRR) by:
        1.  Reducing chunk length: For very long Wikipedia articles, query terms
            can be diluted, reducing the score of specific relevant passages.
            Smaller chunks allow these focused passages to score higher.
        2.  Overlapping chunks: Mitigates the risk of splitting relevant information
            across chunk boundaries, ensuring context is maintained across splits.

        The BM25 retriever's specified Snowball English stemmer will then
        process the `text` field of each Chunk.
        """
        chunks: List[Chunk] = []
        # These values are chosen based on common heuristics for document chunking
        # in retrieval systems, aiming for a balance between context and specificity.
        # Approximate character counts:
        # 3000 chars roughly equals 500 words (assuming avg 6 chars/word).
        # 600 chars roughly equals 100 words for overlap.
        chunk_size_chars = 3000 
        overlap_chars = 600   

        for doc in docs:
            doc_text = doc.text
            
            # Skip empty documents as they cannot yield meaningful chunks.
            if not doc_text:
                continue

            start_idx = 0
            chunk_idx = 0

            # If the document is short (less than or equal to the desired chunk size),
            # treat the entire document as a single chunk.
            if len(doc_text) <= chunk_size_chars:
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_{chunk_idx}",
                    doc_id=doc.doc_id,
                    text=doc_text,
                    metadata=doc.metadata
                )
                chunks.append(chunk)
                continue # Move to the next document

            # Process longer documents by splitting them into overlapping chunks.
            while start_idx < len(doc_text):
                # Determine the raw end index for the current chunk, up to `chunk_size_chars`.
                raw_end_idx = min(start_idx + chunk_size_chars, len(doc_text))
                
                # Adjust `effective_end_idx` to end on a word boundary if possible.
                # This improves readability and avoids cutting words in half, which can be
                # slightly beneficial for downstream processing though BM25 tokenizers
                # often handle this.
                effective_end_idx = raw_end_idx
                if raw_end_idx < len(doc_text): # Only try to adjust if not at the very end of the document
                    # Find the last space within the potential chunk text segment.
                    last_space_pos = doc_text.rfind(' ', start_idx, raw_end_idx)
                    # If a space is found and it's not at the very beginning of the segment,
                    # use that space as the boundary. Otherwise, use the raw_end_idx.
                    if last_space_pos != -1 and last_space_pos > start_idx:
                        effective_end_idx = last_space_pos
                
                chunk_text = doc_text[start_idx:effective_end_idx].strip()

                # Only create a chunk if it contains content after stripping whitespace.
                if chunk_text:
                    chunk = Chunk(
                        chunk_id=f"{doc.doc_id}_{chunk_idx}",
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        metadata=doc.metadata
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

                # Calculate the start index for the next chunk, incorporating overlap.
                # Ensure `chunk_size_chars` is greater than `overlap_chars` to make progress.
                start_idx += (chunk_size_chars - overlap_chars)
                
                # The loop condition `start_idx < len(doc_text)` combined with the
                # `min` operation for `raw_end_idx` correctly ensures that all parts
                # of the document, including the tail, are eventually captured as chunks.

        return chunks
