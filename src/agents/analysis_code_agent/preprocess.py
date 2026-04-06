import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "title_prepend_with_overlap"
    description = "Prepend title text to sections and apply overlapping windows on content for better term alignment."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks = []
        articles = {}

        # Group documents by article ID
        for doc in docs:
            article_id = doc.doc_id.split(":")[0]
            if article_id not in articles:
                articles[article_id] = {}
            articles[article_id][doc.doc_id] = doc
        
        def split_into_chunks(text: str, chunk_size=300, overlap=150):
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk = " ".join(words[start:end])
                yield chunk
                if end == len(words):
                    break
                start += chunk_size - overlap

        for article_id, sections in articles.items():
            title_text = sections.get(f"{article_id}:0", Document("", "")).text

            for doc_id, doc in sections.items():
                if doc_id.endswith(":0"):
                    chunks.append(Chunk(chunk_id=f"{doc_id}_0", doc_id=doc_id, text=doc.text, metadata=doc.metadata))
                else:
                    enriched_text = f"{title_text}\n\n{doc.text}" if title_text else doc.text
                    idx = 0
                    for chunk in split_into_chunks(enriched_text):
                        chunks.append(Chunk(chunk_id=f"{doc_id}_{idx}", doc_id=doc_id, text=chunk, metadata=doc.metadata))
                        idx += 1

        return chunks
