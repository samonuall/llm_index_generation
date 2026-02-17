# LLM Index Generation

Research project testing LLM-driven approaches for iteratively writing preprocessing code to improve BM25 retrieval quality. An agent receives feedback from a static evaluation harness and iterates on its own preprocessing implementation.

## Overview

The core idea: an LLM agent only controls a single `preprocess()` function. Everything else — the retriever, tokenizer, evaluation metrics, and dataset — is fixed. The agent's goal is to transform raw Wikipedia documents into chunks that maximize Recall@k and MRR against a set of "tip of the tongue" queries.

```
Raw Documents → preprocess() → Chunks → BM25 Index → Eval Queries → Recall@k + MRR
                    ↑ (agent controls only this)
```

## Repository Structure

```
llm_index_generation/
├── main.py                            # CLI entry point for running agents
├── proposals.md                       # Design proposals and tradeoffs
├── pyproject.toml
├── data/                              # Pre-fetched corpus subset (generated, not committed)
│   ├── documents.jsonl                # Wikipedia documents
│   └── queries.jsonl                  # Eval queries (EvalQuery objects)
└── src/
    ├── evaluation/                    # Static harness – DO NOT MODIFY
    │   ├── schema.py                  # Data classes: Document, Chunk, EvalQuery
    │   ├── base.py                    # BasePreprocessor ABC
    │   └── scripts/
    │       ├── get_data.py            # One-time data download from HuggingFace
    │       ├── build_index.py         # BM25 index builder (bm25s + English stemmer)
    │       └── test_preprocessing.py  # Eval harness: Recall@k + MRR
    └── agents/
        ├── CONTEXT.md                 # Dataset and interface docs for agent prompts
        ├── agent_runner.py            # Abstract base class for iterative LLM agents
        ├── baseline/
        │   └── preprocess.py          # Passthrough reference implementation
        └── gemini_sdk/
            ├── preprocess.py          # Wikipedia section-aware chunker (agent-generated)
            ├── agent.py               # Gemini-backed iterative agent
            └── context/
                └── SYSTEM_INSTRUCTION.md  # System prompt template
```

## Setup

```bash
# Install dependencies
uv sync

# Download the dataset (run once)
uv run python src/evaluation/scripts/get_data.py            # 50 queries (default)
uv run python src/evaluation/scripts/get_data.py --n-queries 100
```

Requires a `.env` file with `GOOGLE_API_KEY` for the Gemini-backed agents.

## Running Evaluation

Evaluate any agent preprocessor against the static BM25 harness:

```bash
uv run python src/evaluation/scripts/test_preprocessing.py --agent baseline
uv run python src/evaluation/scripts/test_preprocessing.py --agent gemini_sdk
uv run python src/evaluation/scripts/test_preprocessing.py --agent <agent_name> --top-k 20
```

## Running an Agent

Run an iterative LLM agent (eval → improve loop):

```bash
uv run python main.py --agent gemini_sdk --loops 5

# If the LLM provider's safety filters block dataset content, omit raw query text from prompts:
uv run python main.py --agent gemini_sdk --loops 5 --no-query-text
```

The agent evaluates the current `preprocess.py`, builds a prompt with per-query feedback (misses, ranks, metrics), calls the LLM, extracts the updated code, and repeats. `--no-query-text` strips the raw query strings from that feedback while preserving query IDs, doc IDs, and rank signals.

## Dataset

- **Source**: [CRUMB](https://huggingface.co/datasets/jfkback/crumb) — `tip_of_the_tongue` split
- **Corpus**: Full Wikipedia articles (streamed from HuggingFace, not stored)
- **Queries**: "Tip of the tongue" natural language descriptions of Wikipedia entities
- **Scope**: 50 eval queries, each with one or more ground-truth `relevant_doc_ids`

## Agent Interface

Each agent lives in `src/agents/<name>/` and must define a `Preprocessor` class in `preprocess.py`:

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
        # Return at least one Chunk per Document.
        # chunk.doc_id must match doc.doc_id.
        # chunk_id must be globally unique.
        ...
```

Agents can create any additional files within their own folder.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Recall@k | Fraction of queries where a relevant doc appears in the top-k results |
| MRR | Mean Reciprocal Rank — average of 1/rank for the first relevant result |

The retriever is BM25 (`bm25s`) with an English Snowball stemmer. Agents cannot modify the retriever.

## Agents

| Agent | Strategy | Notes |
|-------|----------|-------|
| `baseline` | One chunk per document, raw text | Performance floor |
| `gemini_sdk` | Section-aware Wikipedia chunking with title propagation | Gemini-generated, iteratively improved |

## Adding a New Agent

1. Create `src/agents/<name>/preprocess.py` with a `Preprocessor` class (see interface above)
2. Optionally add an `agent.py` subclassing `AgentRunner` for the iterative loop
3. Evaluate: `uv run python src/evaluation/scripts/test_preprocessing.py --agent <name>`

## Conventions

- Use `uv add <package>` to add dependencies, never `pip install`
- Never modify anything in `src/evaluation/` — it is the static ground truth
- Agent code stays entirely within `src/agents/<name>/`
- `data/` is generated; run `get_data.py` to populate it
