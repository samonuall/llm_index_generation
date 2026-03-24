import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "overlap_with_article_context"
    description = "Sliding overlapping windows enriched with the full article for context."

    def __init__(self):
        self.window_words = 250
        self.stride_words = 125

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks = []
        articles = {}
        current_id = 0

        for doc in docs:
            page_id, section_idx = doc.doc_id.split(":")
            if page_id not in articles:
                articles[page_id] = []
            articles[page_id].append((section_idx, doc))

        for page_id, sections in articles.items():
            # Sort sections by their order
            sections.sort(key=lambda x: int(x[0]))
            full_article = " ".join(section[1].text for section in sections)
            words = full_article.split()

            # Sliding overlapping windows
            for start in range(0, len(words), self.stride_words):
                window = words[start:start + self.window_words]
                if not window:
                    break

                chunk_text = " ".join(window)
                first_doc = sections[0][1]
                chunk = Chunk(
                    chunk_id=f"{first_doc.doc_id}_{current_id}",
                    doc_id=first_doc.doc_id,
                    text=chunk_text
                )
                chunks.append(chunk)
                current_id += 1

        return chunks
