# """
# build_index.py – Static BM25 index builder using bm25s + PyStemmer.

# Exposes BM25Index: tokenises chunk text with an English stemmer,
# builds a bm25s index in memory, and provides search() -> ranked Chunks.

# Not called directly; imported by test_preprocessing.py.
# """

# from __future__ import annotations

# import sys
# import pathlib

# # Make src/evaluation/ importable (for schema)
# sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# from typing import List, Tuple

# import bm25s
# import Stemmer

# from schema import Chunk


# class BM25Index:
#     def __init__(self, chunks: List[Chunk]) -> None:
#         self._chunks = chunks
#         self._stemmer = Stemmer.Stemmer("english")

#         corpus_texts = [c.text for c in chunks]
#         corpus_tokens = bm25s.tokenize(
#             corpus_texts, stopwords="en", stemmer=self._stemmer
#         )

#         self._retriever = bm25s.BM25()
#         self._retriever.index(corpus_tokens)

#     def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
#         """Return up to top_k (chunk, score) pairs ranked by BM25 score."""
#         k = min(top_k, len(self._chunks))
#         query_tokens = bm25s.tokenize(
#             [query], stopwords="en", stemmer=self._stemmer
#         )
#         # results/scores are shape (n_queries, k); we have 1 query
#         results, scores = self._retriever.retrieve(query_tokens, k=k)

#         return [
#             (self._chunks[int(idx)], float(score))
#             for idx, score in zip(results[0], scores[0])
#         ]

"""
build_index.py – Static BM25 index builder using bm25s.

Exposes BM25Index: tokenises chunk text, builds a bm25s index in memory,
and provides search() -> ranked Chunks.

Not called directly; imported by test_preprocessing.py.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from typing import List, Tuple

import bm25s

from schema import Chunk


class BM25Index:
    def __init__(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks

        corpus_texts = [c.text for c in chunks]
        corpus_tokens = bm25s.tokenize(corpus_texts)

        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Return up to top_k (chunk, score) pairs ranked by BM25 score."""
        k = min(top_k, len(self._chunks))
        query_tokens = bm25s.tokenize([query])
        results, scores = self._retriever.retrieve(query_tokens, k=k)

        return [
            (self._chunks[int(idx)], float(score))
            for idx, score in zip(results[0], scores[0])
        ]