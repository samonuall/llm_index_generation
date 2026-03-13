# LLM Index Generation — Architecture

## Overview

This project investigates whether an LLM-driven agent can iteratively write document-preprocessing code to improve BM25 retrieval quality. The agent gets numeric feedback from a static evaluation harness each loop and refines its preprocessing strategy.

The retrieval task is the CRUMB "tip-of-the-tongue" benchmark: given a vague natural-language memory of a movie, book, or other entity, find the relevant Wikipedia article out of ~10,000 candidates.

---

## Repository Layout

```
llm_index_generation/
├── main.py                          # CLI entry point — registers and runs agent loops
├── pyproject.toml                   # Python deps (managed via uv)
├── .env                             # API keys (not committed)
├── ARCHITECTURE.md                  # This file
│
├── data/
│   └── tip_of_the_tongue/
│       ├── documents.jsonl          # ~10,000 Wikipedia docs (931 relevant + 9,069 distractors)
│       └── queries.jsonl            # 135 EvalQuery objects with ground-truth doc IDs
│
└── src/
    ├── evaluation/                  # Static harness — DO NOT MODIFY
    │   ├── schema.py                # Shared dataclasses: Document, Chunk, EvalQuery
    │   ├── base.py                  # BasePreprocessor ABC
    │   └── scripts/
    │       ├── get_data.py          # One-time data download from HuggingFace
    │       ├── build_index.py       # Builds a bm25s index from a list of Chunks
    │       └── test_preprocessing.py  # Runs eval queries; returns Recall@k, nDCG, MRR
    │
    └── agents/
        ├── CONTEXT.md               # Dataset and interface reference for LLM prompts
        ├── agent_runner.py          # Abstract AgentRunner base class
        ├── baseline_results.json    # Pre-computed baseline metrics for the 10k subset
        ├── baseline/
        │   └── preprocess.py        # Passthrough (one chunk = one document)
        └── analysis_code_agent/
            └── ...                  # See src/agents/analysis_code_agent/README.md
```

---

## Core Data Classes (`src/evaluation/schema.py`)

```python
@dataclass
class Document:
    doc_id: str       # Unique ID matching CRUMB corpus
    text: str         # Raw Wikipedia article text
    metadata: dict    # Extra fields — currently empty in this dataset

@dataclass
class Chunk:
    chunk_id: str     # Globally unique ID, e.g. "doc_123_0"
    doc_id: str       # Parent document ID — used for retrieval scoring
    text: str         # Text that will be indexed by BM25
    metadata: dict    # Optional extra fields (unused by harness)

@dataclass
class EvalQuery:
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]   # Ground-truth document IDs
```

---

## Evaluation Harness

The harness is static — agents cannot modify it.

### Pipeline

```
documents.jsonl  →  Preprocessor.preprocess()  →  List[Chunk]
                                                        ↓
                                              bm25s index (English Snowball stemmer)
                                                        ↓
                                        queries.jsonl → ranked results
                                                        ↓
                                    Recall@k, nDCG@10, MRR reported
```

**Retrieval model:** `bm25s` with MaxP aggregation — the highest-scoring chunk from each document is used for document-level ranking. This means multiple chunks per document are fine; only the best one competes.

**Tokenization:** English Snowball stemmer applied at index-build time. Stopword removal is handled by BM25, not the preprocessor.

### Running Evals

```bash
# Evaluate a specific agent's preprocess.py
uv run python src/evaluation/scripts/test_preprocessing.py --agent analysis_code_agent

# Change top-k
uv run python src/evaluation/scripts/test_preprocessing.py --agent analysis_code_agent --top-k 20

# Programmatic use inside an agent
from test_preprocessing import evaluate
results = evaluate(Preprocessor(), top_k=100)
# returns: { agent, recall_at_k, ndcg, top_k, n_queries, n_chunks, query_results }
```

### Baseline

The `baseline` agent emits one chunk per document (passthrough). Pre-computed results for the 10k subset are stored in `src/agents/baseline_results.json`:

```json
{
  "agent": "baseline",
  "recall_at_k": 0.5778,
  "ndcg": 0.1160,
  "n_queries": 135,
  "n_chunks": 10000
}
```

---

## Agent Interface

Each agent lives in `src/agents/<agent_name>/` and must provide a `preprocess.py` that defines a `Preprocessor` class inheriting from `BasePreprocessor`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "my_agent"
    description = "One-line summary of the strategy"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        # - Must return at least one Chunk per Document
        # - chunk.doc_id must match the source document's doc_id
        # - chunk_id must be globally unique across all returned chunks
        ...
```

### AgentRunner Base Class

`AgentRunner` (`src/agents/agent_runner.py`) is an abstract base class for iterative LLM-driven agents. It handles loading `preprocess.py`, running the harness each iteration, and storing results.

Subclass it by implementing `build_prompt()` and `call_llm()`, then call `runner.run(n_loops)`:

```python
from agent_runner import AgentRunner

class MyAgent(AgentRunner):
    agent_name = "my_agent"

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        ...

    def call_llm(self, prompt: str, iteration: int) -> None:
        ...
```

`run_eval()` dynamically reloads `preprocess.py` each iteration using `importlib.util`, so code changes take effect immediately without restarting the process.

---

## CLI Entry Point

Agents are registered in `main.py` and run from the command line:

```bash
uv run python main.py --agent analysis_code_agent --loops 3

# Omit raw query text from prompts (useful if safety filters block dataset content)
uv run python main.py --agent analysis_code_agent --loops 3 --no-query-text
```

---

## Data Preparation

Run once to build the `data/` directory:

```bash
# Default: 50 queries
uv run python src/evaluation/scripts/get_data.py

# Larger subset (~10k docs using reservoir sampling)
uv run python src/evaluation/scripts/get_data.py --n-queries 200
```

The script streams from HuggingFace (`jfkback/crumb`, `tip_of_the_tongue` split), guarantees all relevant documents are included, and fills the remainder with random distractors via reservoir sampling.

---

## Adding a New Agent

1. Create `src/agents/<agent_name>/preprocess.py` with a `Preprocessor(BasePreprocessor)` class.
2. Add a new `elif` branch in `main.py` that instantiates the agent.
3. Run `uv run python main.py --agent <agent_name> --loops N`.

The harness validates that `Preprocessor` is named exactly `Preprocessor` and inherits from `BasePreprocessor`. Agents that fail this check will not be evaluated.

---

## Dependencies

Managed via `uv`. Key packages:

| Package | Role |
|---------|------|
| `bm25s` | Fast BM25 retrieval with stemmer support |
| `litellm` | Unified LLM API (supports OpenAI, Gemini, etc.) |
| `fastapi` / `uvicorn` | In-process BM25 HTTP server for hypothesis testing |
| `python-dotenv` | Load `.env` API keys |
| `pyyaml` | Agent config files |

Add new dependencies with `uv add <package>`. Never use `pip install` directly.
