import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor
import re

class Preprocessor(BasePreprocessor):
    name = "overlap_window_lead_boost"
    description = (
        "Create overlapping fixed-size word windows (300w, stride 150) and repeat the lead "
        "(first paragraph head) at the start of each chunk to boost TF for lead terms. Also "
        "emit a coarse truncated full-document chunk of 1200 words with heavy lead repeat."
    )

    def __init__(self):
        self.window_words = 300
        self.stride_words = 150
        self.lead_words = 40
        self.lead_repeat_per_chunk = 4
        self.lead_repeat_coarse = 8
        self.coarse_truncate = 1200

    def _words(self, text: str) -> List[str]:
        if not text:
            return []
        # simple whitespace split preserves tokens for BM25 stemmer
        return text.split()

    def _first_paragraph(self, text: str) -> str:
        if not text:
            return ""
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if parts:
            return parts[0]
        # fallback to initial slice
        return text.strip().splitlines()[0] if text.strip().splitlines() else text.strip()

    def _repeat_block(self, text: str, repeats: int) -> str:
        if not text or repeats <= 0:
            return ""
        # prefix repetition to boost term frequency
        return (" " + text) * repeats + " "

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        global_counter = 0

        for doc in docs:
            doc_id = doc.doc_id
            raw = (doc.text or "").strip()

            # extract lead (first paragraph head)
            lead_para = self._first_paragraph(raw)
            lead_head = " ".join(self._words(lead_para)[: self.lead_words]).strip()
            lead_block_chunk = self._repeat_block(lead_head, self.lead_repeat_per_chunk) if lead_head else ""
            lead_block_coarse = self._repeat_block(lead_head, self.lead_repeat_coarse) if lead_head else ""

            all_words = self._words(raw)
            n = len(all_words)

            # Sliding overlapping windows
            if n == 0:
                # Ensure at least one chunk per doc (empty placeholder)
                chunks.append(Chunk(chunk_id=f"{doc_id}_{global_counter}", doc_id=doc_id, text=""))
                global_counter += 1
                continue

            i = 0
            created = 0
            while i < n:
                window = all_words[i : i + self.window_words]
                if not window:
                    break
                window_text = " ".join(window)
                chunk_text = f"{lead_block_chunk}{window_text}".strip()
                chunks.append(Chunk(chunk_id=f"{doc_id}_{global_counter}", doc_id=doc_id, text=chunk_text))
                global_counter += 1
                created += 1
                # advance by stride (overlap)
                i += self.stride_words

            # Add a coarse truncated whole-document chunk (preserve document-level context)
            coarse_words = all_words[: self.coarse_truncate] if n > 0 else []
            if coarse_words:
                coarse_text = " ".join(coarse_words)
                coarse_chunk = f"{lead_block_coarse}{coarse_text}".strip()
                chunks.append(Chunk(chunk_id=f"{doc_id}_{global_counter}", doc_id=doc_id, text=coarse_chunk))
                global_counter += 1
                created += 1

            # Guard: ensure at least one chunk was created
            if created == 0:
                chunks.append(Chunk(chunk_id=f"{doc_id}_{global_counter}", doc_id=doc_id, text=lead_head or ""))
                global_counter += 1

        return chunks
