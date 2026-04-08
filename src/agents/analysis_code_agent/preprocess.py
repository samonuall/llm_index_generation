import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List, Dict
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "triple_granularity"
    description = "Full article + section pairs + individual sections for maximum coverage"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        # Group documents by article ID
        articles: Dict[str, List[Document]] = {}
        for doc in docs:
            article_id = doc.doc_id.split(":")[0]
            if article_id not in articles:
                articles[article_id] = []
            articles[article_id].append(doc)
        
        chunks = []
        
        for article_id, article_docs in articles.items():
            # Sort by section index
            article_docs.sort(key=lambda d: int(d.doc_id.split(":")[1]))
            
            # Find title section (section 0)
            title_text = ""
            title_doc = None
            for doc in article_docs:
                if doc.doc_id.endswith(":0"):
                    title_text = doc.text.strip()
                    title_doc = doc
                    break
            
            # Use section 0 if available, otherwise first section
            anchor_doc = title_doc if title_doc else article_docs[0]
            
            # Strategy 1: Full article chunk
            combined_text = "\n\n".join(doc.text for doc in article_docs)
            chunks.append(Chunk(
                chunk_id=f"{anchor_doc.doc_id}_full",
                doc_id=anchor_doc.doc_id,
                text=combined_text,
                metadata=anchor_doc.metadata
            ))
            
            # Strategy 2: Section pairs
            content_sections = [d for d in article_docs if not d.doc_id.endswith(":0")]
            for i in range(len(content_sections) - 1):
                doc1, doc2 = content_sections[i], content_sections[i + 1]
                
                if title_text:
                    pair_text = f"{title_text}\n\n{doc1.text}\n\n{doc2.text}"
                else:
                    pair_text = f"{doc1.text}\n\n{doc2.text}"
                
                chunks.append(Chunk(
                    chunk_id=f"{doc1.doc_id}_pair",
                    doc_id=doc1.doc_id,
                    text=pair_text,
                    metadata=doc1.metadata
                ))
            
            # Strategy 3: Individual sections
            for doc in article_docs:
                section_idx = int(doc.doc_id.split(":")[1])
                
                if section_idx > 0 and title_text:
                    enriched_text = f"{title_text}\n\n{doc.text}"
                else:
                    enriched_text = doc.text
                
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_single",
                    doc_id=doc.doc_id,
                    text=enriched_text,
                    metadata=doc.metadata
                ))
        
        return chunks
