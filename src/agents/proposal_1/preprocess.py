import re
import sys, pathlib
from typing import List

# Ensure the evaluation modules are accessible
eval_path = str(pathlib.Path(__file__).parents[2] / "evaluation")
if eval_path not in sys.path:
    sys.path.insert(0, eval_path)

from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "proposal_1"
    description = "Section-aware Wikipedia chunking with triple title propagation and full plot coverage for BM25."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Preprocesses Wikipedia documents by partitioning them into sections and applying 
        title propagation. This strategy prioritizes plot details and specific scenes, 
        which are critical for 'tip of the tongue' queries.
        """
        all_chunks = []
        
        # Wikipedia sections that are typically noise for plot-based descriptive queries
        excluded_sections = {
            "references", "external links", "see also", "further reading", 
            "notes", "bibliography", "sources", "navigation", "gallery",
            "citations", "layout", "external media", "links", "works cited"
        }

        for doc in docs:
            doc_id = doc.doc_id
            
            # Extract a meaningful title from metadata, or fallback to doc_id
            title = doc.metadata.get("title", "").strip()
            if not title:
                # If metadata title is missing, try to infer from the first line of text
                text_lines = doc.text.split('\n') if doc.text else []
                title = text_lines[0].strip() if text_lines else doc_id
            
            text = doc.text.strip() if doc.text else ""
            if not text:
                all_chunks.append(Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_0",
                    text=f"{title}. {title}. {title}."
                ))
                continue
            
            # Split the document into sections based on Wikipedia header markers (== Section ==)
            # A lookahead is used to split *before* each header, keeping the header text with its content.
            # We prefix with a newline to catch the first header if the text starts with one.
            segments = re.split(r'\n+(?=={2,}.+?={2,})', "\n" + text)
            
            chunk_count = 0
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue
                
                # Detect the section header (e.g., == Plot == or === Cast ===)
                header_match = re.match(r'^(={2,})\s*(.*?)\s*\1', segment)
                if header_match:
                    header = header_match.group(2).strip()
                    # The content follows the header line
                    content = segment[header_match.end():].strip()
                else:
                    # Content before the first formal header (usually the article lead)
                    header = "Summary"
                    content = segment
                
                # Filter out sections that contain repetitive metadata or navigation noise
                if header.lower() in excluded_sections:
                    continue
                
                if not content:
                    continue

                # Tokenize by whitespace for sliding window chunking
                words = content.split()
                if not words:
                    continue
                
                # BM25 performance is improved by smaller, keyword-dense chunks.
                # 300 words provides enough context while 100-word overlap preserves narrative flow.
                chunk_size = 300
                stride = 200 # results in 100 words of overlap
                
                for i in range(0, len(words), stride):
                    chunk_words = words[i : i + chunk_size]
                    if not chunk_words:
                        break
                    
                    body = " ".join(chunk_words)
                    
                    # Title Propagation:
                    # 1. Triple Title Boost: Increases the BM25 term frequency for the entity name.
                    # 2. Section Context: Adds weight to 'Plot' or 'Synopsis' headers, which are 
                    #    highly relevant for scene descriptions.
                    # 3. Body: The descriptive text from the article.
                    chunk_text = f"{title}. {title}. {title}. {header}. {body}"
                    
                    all_chunks.append(Chunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_{chunk_count}",
                        text=chunk_text
                    ))
                    chunk_count += 1
                    
                    # Stop if we have covered the remaining content in the current window
                    if i + chunk_size >= len(words):
                        break

            # Fallback to ensure every document is represented in the index
            if chunk_count == 0:
                all_chunks.append(Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_0",
                    text=f"{title}. {title}. {title}."
                ))

        return all_chunks
