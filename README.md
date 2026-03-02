# LLM Index Generation

Research project testing LLM-driven approaches for iteratively writing preprocessing code to improve BM25 retrieval quality. An agent receives feedback from a static evaluation harness and iterates on its own preprocessing implementation.

## Overview

The core idea: an LLM agent only controls a single `preprocess()` function. Everything else — the retriever, tokenizer, evaluation metrics, and dataset — is fixed. The agent's goal is to transform raw Wikipedia documents into chunks that maximize Recall@k and MRR against a set of "tip of the tongue" queries.

## Big Picture

```
Raw Documents → preprocess() → Chunks → BM25 Index → Eval Queries → Recall@k + MRR
                    ↑ (agent controls only this)
```

## Dataset

- **Source**: [CRUMB](https://huggingface.co/datasets/jfkback/crumb) — `tip_of_the_tongue` split
- **Corpus**: Full Wikipedia articles (streamed from HuggingFace, not stored)
- **Queries**: "Tip of the tongue" natural language descriptions of Wikipedia entities
- **Scope**: 135 eval queries (default), each with one or more ground-truth `relevant_doc_ids`


## Agents

| Agent | Strategy | Notes |
|-------|----------|-------|
| `baseline` | One chunk per document, raw text | Performance floor |
| `gemini_sdk` | Section-aware Wikipedia chunking with title propagation | Gemini-generated, iteratively improved |
| `lite_llm_agent` | Configurable via `config.yaml`; uses LiteLLM for provider-agnostic LLM calls | Supports any model accessible via LiteLLM |
| `test_agent` | Same as `lite_llm_agent` with `test_mode=True` | Returns a mock LLM response; no API calls made |

## Algos

Download full corpora for CRUMB benchmark splits:

| Split Name | Flag | Documents | Queries |
|------------|------|-----------|---------|
| Paper Retrieval | `paper_retrieval` | 363,133 | 72 |
| Tip of the Tongue | `tip_of_the_tongue` | 1,083,337 | 135 |
| Clinical Trial | `clinical_trial` | 914,628 | 113 |
| Code Retrieval | `code_retrieval` | 232,444 | 3,665 |
| Legal QA | `legal_qa` | 1,182,626 | 6,753 |
| Set Operations | `set_operation_entity_retrieval` | 651,704 | 423 |
| Theorem Retrieval | `theorem_retrieval` | 23,839 | 69 |
| Stack Exchange | `stack_exchange` | 40,956 | 107 |

  

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
    |       └── aggregate_results.py             # Compare results across splits/agents
    │       ├── get_data.py            # One-time data download from HuggingFace
    │       ├── build_index.py         # BM25 index builder (bm25s + English stemmer)
    │       └── test_preprocessing_split.py  # Eval harness: Recall@k + MRR
    └── agents/
        ├── CONTEXT.md                 # Dataset and interface docs for agent prompts
        ├── agent_runner.py            # Abstract base class for iterative LLM agents
        ├── baseline_results.json      # Baseline eval numbers loaded by AgentRunner
        ├── baseline/
        │   └── preprocess.py          # Passthrough reference implementation
        ├── gemini_sdk/
        │   ├── preprocess.py          # Wikipedia section-aware chunker (agent-generated)
        │   ├── agent.py               # Gemini-backed iterative agent
        │   └── context/
        │       └── SYSTEM_INSTRUCTION.md  # System prompt template
        └── lite_llm_agent/
            ├── preprocess.py          # Agent preprocessor
            ├── agent.py               # LiteLLM-backed iterative agent (supports test mode)
            ├── config.yaml            # Model, temperature, and API settings
            └── context/
                └── SYSTEM_INSTRUCTION.md  # System prompt template
```

## Background

Split	Documents	Queries
paper_retrieval	363,133	72
tip_of_the_tongue	1,083,337	135
clinical_trial	914,628	113
code_retrieval	232,444	3,665
legal_qa	1,182,626	6,753
set_operation_entity_retrieval	651,704	423
theorem_retrieval	23,839	69
stack_exchange	40,956	107

## Setup

```bash
# Install dependencies
uv sync

# Set up API keys in .env
GOOGLE_API_KEY=...        # For Gemini agents
LITELLM_API_KEY=...       # For LiteLLM agents

## Downloading Data

```bash
# Install dependencies
uv sync
```

## Download Data

Download individual splits (full corpus + queries) to /data/**
```
uv run python -m src.evaluation.scripts.get_data_extended --split paper_retrieval
uv run python -m src.evaluation.scripts.get_data_extended --split tip_of_the_tongue
uv run python -m src.evaluation.scripts.get_data_extended --split clinical_trial
uv run python -m src.evaluation.scripts.get_data_extended --split code_retrieval
uv run python -m src.evaluation.scripts.get_data_extended --split theorem_retrieval
uv run python -m src.evaluation.scripts.get_data_extended --split stack_exchange
uv run python -m src.evaluation.scripts.get_data_extended --split legal_qa
uv run python -m src.evaluation.scripts.get_data_extended --split set_operation_entity_retrieval
```

## Running Evaluations

```
# Basic evaluation
uv run python -m src.evaluation.scripts.test_preprocessing_split --agent baseline --split paper_retrieval

# With comparison to previous runs
uv run python -m src.evaluation.scripts.test_preprocessing_split --agent baseline --split paper_retrieval --compare

# Custom top-k
uv run python -m src.evaluation.scripts.test_preprocessing_split --agent lite_llm_agent --split code_retrieval --top-k 20

# List available splits
uv run python -m src.evaluation.scripts.test_preprocessing_split --agent baseline
```

## Comparing Aggregate Results
Metrics are based on recall!

| Metric | Description |
|--------|-------------|
| Recall@k | Fraction of queries where a relevant doc appears in the top-k results |
| MRR | Mean Reciprocal Rank — average of 1/rank for the first relevant result |

The retriever is BM25 (`bm25s`) with an English Snowball stemmer. Agents cannot modify the retriever.


```
# View all results
uv run python -m src.evaluation.scripts.aggregate_results

# Filter by split
uv run python -m src.evaluation.scripts.aggregate_results --split paper_retrieval

# Filter by agent
uv run python -m src.evaluation.scripts.aggregate_results --agent baseline

# Group by split
uv run python -m src.evaluation.scripts.aggregate_results --by-split

# Export to CSV
uv run python -m src.evaluation.scripts.aggregate_results --export all_results.csv
```


## Running an Agent

Requires a `.env` file with API keys for the relevant agent:
- `GOOGLE_API_KEY` — for `gemini_sdk`
- `LITELLM_API_KEY` — for `lite_llm_agent` / `test_agent` (or configure via `lite_llm_agent/config.yaml`)

Run an iterative LLM agent (eval → improve loop):

```bash
uv run python -m main --agent gemini_sdk --loops 5
uv run python -m main --agent lite_llm_agent --loops 5

# If the LLM provider's safety filters block dataset content, omit raw query text from prompts:
uv run python -m main --agent gemini_sdk --loops 5 --no-query-text

# Enable MLflow tracing for LiteLLM calls (lite_llm_agent only):
uv run python -m main --agent lite_llm_agent --loops 5 --enable_tracing

# Dry-run with a mock LLM response (no API calls, useful for testing the pipeline):
uv run python -m main --agent test_agent --loops 1
```

The agent evaluates the current `preprocess.py`, builds a prompt with per-query feedback (misses, ranks, metrics), calls the LLM, extracts the updated code, and repeats. `--no-query-text` strips the raw query strings from that feedback while preserving query IDs, doc IDs, and rank signals. `--enable_tracing` enables MLflow auto-logging for LiteLLM calls via `mlflow.litellm.autolog()` — only supported for `lite_llm_agent`. `test_agent` uses `LiteLLMAgent` with `test_mode=True`, which injects a mock response instead of making a real API call — useful for validating the pipeline without incurring API costs.

## Seeing Traces
Run the following to see traces recorded when `--enable_tracing` is used with `lite_llm_agent`:

```bash
uv run mlflow ui
```

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

## Adding a New Agent

1. Create `src/agents/<name>/preprocess.py` with a `Preprocessor` class (see interface above)
2. Optionally add an `agent.py` subclassing `AgentRunner` for the iterative loop
3. Evaluate: `uv run python -m src.evaluation.scripts.test_preprocessing --agent <name>`

## Conventions

- Use `uv add <package>` to add dependencies, never `pip install`
- Never modify anything in `src/evaluation/` — it is the static ground truth
- Agent code stays entirely within `src/agents/<name>/`
- `data/` is generated; run `get_data.py` to populate it
- Update `src/agents/baseline_results.json` whenever baseline numbers change (e.g. after re-running with a different dataset size); `AgentRunner` loads this file at runtime

## Quick Run

Download datasets
```
for split in paper_retrieval tip_of_the_tongue clinical_trial code_retrieval legal_qa set_operation_entity_retrieval theorem_retrieval stack_exchange; do
  echo "Downloading $split..."
  uv run python -m src.evaluation.scripts.get_data_extended --split $split
done
```

Baselines
```
for split in paper_retrieval tip_of_the_tongue clinical_trial code_retrieval legal_qa set_operation_entity_retrieval theorem_retrieval stack_exchange; do
  echo "Evaluating baseline on $split..."
  uv run python -m src.evaluation.scripts.test_preprocessing_split --agent baseline --split $split
done
```
