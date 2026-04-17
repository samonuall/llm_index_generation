# LLM Index Generation

Research project testing LLM-driven approaches for iteratively writing preprocessing code to improve BM25 retrieval quality. An agent receives feedback from a static evaluation harness and iterates on its own preprocessing implementation.

## Overview

The core idea: an LLM agent only controls a single `preprocess()` function. Everything else — the retriever, tokenizer, evaluation metrics, and dataset — is fixed. The agent's goal is to transform raw Wikipedia documents into chunks that maximize Recall@k and nDCG against a set of "tip of the tongue" queries.

```
Raw Documents → preprocess() → Chunks → BM25 Index → Eval Queries → Recall@k + nDCG
                    ↑ (agent controls only this)
```

## Dataset

- **Source**: [CRUMB](https://huggingface.co/datasets/jfkback/crumb)
- **Corpus**: Full Wikipedia articles (streamed from HuggingFace, cached locally)
- **Queries**: Natural language descriptions of Wikipedia entities

| Split | Documents | Queries |
|-------|-----------|---------|
| `tip_of_the_tongue` | 1,083,337 | 135 |
| `paper_retrieval` | 363,133 | 72 |
| `clinical_trial` | 914,628 | 113 |
| `code_retrieval` | 232,444 | 3,665 |
| `legal_qa` | 1,182,626 | 6,753 |
| `set_operation_entity_retrieval` | 651,704 | 423 |
| `theorem_retrieval` | 23,839 | 69 |
| `stack_exchange` | 40,956 | 107 |

## Agents

| Agent | Strategy |
|-------|----------|
| `baseline` | One chunk per document, raw text — performance floor |
| `analysis_code_agent` | Iterative LLM agent with hypothesis-driven analysis loop |
| `one_shot` | Single LLM call, no iteration |
| `gemini_sdk` | Section-aware Wikipedia chunking with title propagation |
| `lite_llm_agent` | Configurable via `config.yaml`; uses LiteLLM for provider-agnostic calls |

## Repository Structure

```
llm_index_generation/
├── main.py                            # CLI entry point for running agents
├── run_experiments.sh                 # Reproduces all ablation conditions
├── pyproject.toml
├── data/                              # Downloaded corpus data (not committed)
│   └── {split}/
│       ├── documents.jsonl            # Full corpus documents
│       ├── queries.jsonl              # Eval queries
│       ├── gold_docs_cache.json       # Cached gold docs (shared across runs)
│       └── distractors_seed{s}_size{n}.json  # Cached distractor sample
└── src/
    ├── evaluation/                    # Static harness – DO NOT MODIFY
    │   ├── schema.py                  # Data classes: Document, Chunk, EvalQuery
    │   ├── base.py                    # BasePreprocessor ABC
    │   └── scripts/
    │       ├── get_data.py            # One-time data download from HuggingFace
    │       ├── build_index.py         # BM25 index builder (bm25s + English stemmer)
    │       ├── test_preprocessing_split.py  # Eval harness: Recall@k + nDCG
    │       ├── aggregate_results.py   # Compare results across splits/agents
    │       ├── plot_iterations.py     # Visualize metric improvement over iterations
    │       ├── archive_agent_iterations.py  # Archive old iterations
    │       └── cleanup_iterations.py  # Clean up old artifacts
    └── agents/
        ├── CONTEXT.md                 # Dataset and interface docs for agent prompts
        ├── agent_runner.py            # Abstract base class for iterative agents
        ├── baseline/
        │   └── preprocess.py          # Passthrough reference implementation
        ├── analysis_code_agent/
        │   ├── preprocess.py          # Agent preprocessor (reset to baseline before each run)
        │   ├── agent.py               # Main iterative agent loop
        │   ├── analysis_agent.py      # Hypothesis generation + corpus analysis
        │   ├── code_agent.py          # Code rewriting from hypotheses
        │   ├── one_shot_agent.py      # Single-call baseline variant
        │   ├── config.yaml            # All tunable settings (see Config section)
        │   └── context/               # System prompts and corpus descriptions
        └── gemini_sdk/
            ├── preprocess.py
            ├── agent.py
            └── context/
                └── SYSTEM_INSTRUCTION.md
```

## Setup

```bash
uv sync
```

Create a `.env` file at the project root:

```
GOOGLE_API_KEY=...        # for gemini_sdk
LITE_LLM_KEY=...          # for analysis_code_agent / one_shot via UMass proxy
```

## Downloading Data

Data is stored per-split under `data/{split}/`. Download before running:

```bash
# Single split
uv run python -m src.evaluation.scripts.get_data --split tip_of_the_tongue

# All splits
for split in tip_of_the_tongue paper_retrieval clinical_trial code_retrieval legal_qa set_operation_entity_retrieval theorem_retrieval stack_exchange; do
  uv run python -m src.evaluation.scripts.get_data --split $split
done
```

On first run the agent streams and caches the corpus. Subsequent runs load from `gold_docs_cache.json` and `distractors_seed{s}_size{n}.json` — no re-download needed.

## Running an Agent

```bash
# analysis_code_agent (main agent) — ablation conditions:
uv run python main.py --agent analysis_code_agent --loops 5 --condition agent_history
uv run python main.py --agent analysis_code_agent --loops 3 --condition agent
uv run python main.py --agent analysis_code_agent --loops 3 --condition agent_contrastive

# One-shot baseline (single LLM call, no loop):
uv run python main.py --agent one_shot --split tip_of_the_tongue

# Override model and/or API base:
uv run python main.py --agent analysis_code_agent --loops 3 --model gemini/gemini-2.5-pro
uv run python main.py --agent one_shot --model openai/gpt4o --api-base https://thekeymaker.umass.edu/

# Limit distractor corpus size:
uv run python main.py --agent one_shot --max-distractors 5000

# Other agents:
uv run python main.py --agent gemini_sdk --loops 5
uv run python main.py --agent lite_llm_agent --loops 5 --split paper_retrieval
```

## Config (`analysis_code_agent/config.yaml`)

All tunable settings for `analysis_code_agent` and `one_shot`:

```yaml
analysis_model: "openai/claude-haiku-4-5"   # model for hypothesis/analysis calls
code_model: "openai/claude-haiku-4-5"        # model for code generation
api_base: "https://thekeymaker.umass.edu/"   # UMass LiteLLM proxy; null for native API

server_port: 8765
server_startup_timeout: 30        # seconds to wait for BM25 server; increase for large indexes
preprocess_timeout_seconds: 120   # seconds per preprocess() call; increase for large corpora
bm25_batch_size: 100000           # chunks per HTTP request to BM25 server

corpus_size: 5000                 # null = full corpus; integer = reservoir-sample this many docs
                                  # gold docs are always included; remainder sampled randomly
max_distractors: 9000             # max non-relevant docs in subset (overridable via --max-distractors)
max_hypotheses: 4
analysis_max_turns: 4
use_tools: true                   # set false for models without tool-calling support
```

## Running Ablation Experiments

```bash
# Default: runs agent_history on tip_of_the_tongue
bash run_experiments.sh

# Different split or distractor count:
bash run_experiments.sh --split paper_retrieval
bash run_experiments.sh --max-distractors 5000

# Override model:
bash run_experiments.sh --model gemini/gemini-2.5-pro
bash run_experiments.sh --model openai/gpt4o --api-base https://thekeymaker.umass.edu/
```

Resets `preprocess.py` to baseline before each condition. Results saved to `results/` and `ablation_experiments/`.

## Running Evaluations

```bash
# Evaluate a preprocessor directly
uv run python -m src.evaluation.scripts.test_preprocessing_split --agent baseline --split tip_of_the_tongue
uv run python -m src.evaluation.scripts.test_preprocessing_split --agent analysis_code_agent --split paper_retrieval --top-k 20

# Aggregate results across all runs
uv run python -m src.evaluation.scripts.aggregate_results
uv run python -m src.evaluation.scripts.aggregate_results --split paper_retrieval
uv run python -m src.evaluation.scripts.aggregate_results --export all_results.csv

# Plot iteration progress for an experiment
uv run python -m src.evaluation.scripts.plot_iterations --agent analysis_code_agent --split tip_of_the_tongue
```

## Metrics

| Metric | Description |
|--------|-------------|
| Recall@k | Fraction of queries where a relevant doc appears in the top-k results |
| nDCG@k | Normalized Discounted Cumulative Gain at rank k |
| MRR | Mean Reciprocal Rank |

The retriever is BM25 (`bm25s`) with an English Snowball stemmer. Agents cannot modify the retriever.

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

## Conventions

- Use `uv add <package>` to add dependencies, never `pip install`
- Never modify anything in `src/evaluation/` — it is the static ground truth
- Agent code stays entirely within `src/agents/<name>/`
- `data/` is generated; run `get_data.py` to populate it

