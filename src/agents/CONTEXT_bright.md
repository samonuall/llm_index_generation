# Agent Context – BRIGHT Benchmark

## Dataset

- **Source**: [BRIGHT](https://huggingface.co/datasets/xlangai/bright) — reasoning-intensive retrieval benchmark (ICLR 2025)
- **Task used**: `sustainable_living` (StackExchange posts about sustainable living practices)
- **Corpus**: Articles, blog posts, news, and reports about sustainable living topics.
  Each `Document` has:
  - `doc_id` (str): unique identifier
  - `text` (str): full document text (articles/reports, typically 300–2000 words)
  - `metadata` (dict): currently empty for BRIGHT documents

- **Queries**: Real StackExchange questions about sustainable living — long, reasoning-heavy,
  with complex information needs (e.g. "What is the most cost-effective way to reduce home
  energy consumption while also minimising environmental impact?").
  Each query has one or more `relevant_doc_ids` (ground truth) and `excluded_ids`
  (documents cited in the question itself — excluded from evaluation to prevent data leakage).

## Why BRIGHT is hard for BM25

Queries and documents rarely share vocabulary directly. A query might describe a *problem*
using high-level reasoning, while the relevant document explains a *solution* using technical
vocabulary. BM25 needs the chunk text to surface terms that actually appear in (or closely
match) the query.

## Preprocessor Interface

Create `src/agents/<name>/preprocess.py` with a class named exactly `Preprocessor`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "my_agent"
    description = "One-line summary"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        ...
```

**Constraints:**
- Return at least one `Chunk` per `Document`
- Each `Chunk.doc_id` must match its source `Document.doc_id`
- `chunk_id` must be globally unique (e.g. `f"{doc_id}_{i}"`)
- `preprocess()` runs offline (index-build time) — it can do anything to the text

## Retriever (static – do not modify)

- **Algorithm**: BM25 via `bm25s`
- **Tokeniser**: English Snowball stemmer + stopword removal

## Evaluation Metric

Primary: **nDCG@10** — matches the BRIGHT paper's reported numbers.
Also reported: Recall@10, MRR.

**Paper BM25 baseline (no preprocessing): nDCG@10 ≈ 14.8** (average across all tasks).
BM25 with GPT-4 query-side reasoning augmentation achieves ≈ 26.5.
Your goal: improve nDCG@10 on the document side alone through better text transformations.
