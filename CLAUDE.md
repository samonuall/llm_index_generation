# LLM Index Generation – CLAUDE.md

## Project Goal

Research project testing approaches for an agent that iteratively writes preprocessing code to
improve retrieval quality. The agent gets feedback from a static evaluation harness and iterates.

## Repository Structure

```
llm_index_generation/
├── main.py                            # CLI entry point: runs agent loops
├── proposals.md                       # Design proposals and tradeoffs
├── pyproject.toml
├── data/                              # Pre-fetched corpus subset (NEVER modified by agents)
│   ├── documents.jsonl                # Documents from CRUMB tip-of-the-tongue
│   └── queries.jsonl                  # Matching EvalQuery objects
└── src/
    ├── evaluation/                    # Static harness – DO NOT MODIFY
    │   ├── schema.py                  # Shared data classes (Document, Chunk, EvalQuery)
    │   ├── base.py                    # BasePreprocessor ABC – agents subclass this
    │   └── scripts/
    │       ├── get_data.py            # One-time data prep: downloads & saves to data/
    │       ├── build_index.py         # Builds a BM25 index (bm25s) from Chunks
    │       └── test_preprocessing.py  # Runs eval queries, returns Recall@k + MRR
    └── agents/                        # One sub-folder per agent / experiment
        ├── CONTEXT.md                 # Dataset and interface docs for agent prompts
        ├── agent_runner.py            # Abstract base class for iterative LLM agents
        ├── baseline/
        │   └── preprocess.py          # Reference passthrough agent
        ├── gemini_sdk/
        │   ├── preprocess.py          # Must define Preprocessor(BasePreprocessor)
        │   ├── agent.py               # Gemini-backed iterative agent
        │   └── context/
        │       └── SYSTEM_INSTRUCTION.md  # System prompt template
        └── <agent_name>/
            ├── preprocess.py          # Must define Preprocessor(BasePreprocessor)
            └── ...                    # Agent can create any other files here freely
```

## Standard Data Classes (`src/evaluation/schema.py`)

```python
@dataclass
class Document:
    doc_id: str       # Unique identifier matching CRUMB corpus
    text: str         # Raw document text
    metadata: dict    # Extra fields from source dataset

@dataclass
class Chunk:
    chunk_id: str     # Unique id (e.g. f"{doc_id}_0")
    doc_id: str       # Parent document id – used to score retrieval
    text: str         # Text that will be indexed via BM25
    metadata: dict    # Optional extra fields

@dataclass
class EvalQuery:
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]   # Ground-truth doc ids
```

Agents import these by adding `src/evaluation/` to `sys.path`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from schema import Document, Chunk
from base import BasePreprocessor
```

## Agent Interface

Each agent must provide `src/agents/<agent_name>/preprocess.py` with a class named `Preprocessor`
that inherits from `BasePreprocessor`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "my_agent"                  # short identifier, used in eval reports
    description = "One-line summary"   # what strategy this agent uses

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Transform raw documents into chunks ready for BM25 indexing.
        - Must return at least one Chunk per Document.
        - chunk.doc_id must match the source document's doc_id.
        - chunk_id must be globally unique across all returned chunks.
        - Multiple chunks per document are allowed and encouraged.
        """
```

## AgentRunner Base Class (`src/agents/agent_runner.py`)

`AgentRunner` is an abstract base class for iterative LLM-driven agents. Subclass it and implement
`build_prompt()` and `call_llm()`, then call `run(n_loops)`:

```python
from agent_runner import AgentRunner

class MyAgent(AgentRunner):
    agent_name = "my_agent"   # must match the folder under src/agents/

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        ...  # construct a prompt from current code + eval feedback

    def call_llm(self, prompt: str, iteration: int) -> None:
        ...  # call the LLM, extract code, write it to preprocess.py
```

`AgentRunner.run_eval()` dynamically reloads `preprocess.py` and runs the static harness each
iteration, so code changes take effect without restarting.

## CLI Entry Point (`main.py`)

```bash
uv run python main.py --agent gemini_sdk --loops 5
```

Agents are registered in `main.py`. Add a new `elif` branch to support a new agent.

## Evaluation Pipeline

The retriever (BM25 via `bm25s`) and eval scripts are **static and cannot be changed by agents**.
Agents only control `preprocess()`.

Pipeline:
1. Load `data/documents.jsonl` → `List[Document]`
2. Call `Preprocessor().preprocess(docs)` → `List[Chunk]`
3. `build_index.py` tokenises chunks with an English stemmer and builds a `bm25s` index
4. `test_preprocessing.py` runs all eval queries, computes **Recall@k** and **MRR**, prints report

Run an agent's eval:
```bash
uv run python src/evaluation/scripts/test_preprocessing.py --agent <agent_name>
uv run python src/evaluation/scripts/test_preprocessing.py --agent <agent_name> --top-k 20
```

Programmatic use (e.g. from inside an agent folder):
```python
from test_preprocessing import evaluate
results = evaluate(Preprocessor(), top_k=10)
# returns dict: { agent, recall_at_k, mrr, top_k, n_queries, n_chunks, query_results }
# query_results: list of per-query dicts with hit, rank, reciprocal_rank, retrieved_doc_ids
```

## Data Preparation (`get_data.py`)

Run manually to refresh `data/`:

```bash
uv run python src/evaluation/scripts/get_data.py            # default: 50 queries
uv run python src/evaluation/scripts/get_data.py --n-queries 100
```

Streams from HuggingFace (`jfkback/crumb`, `tip_of_the_tongue` split) — no large files stored.
Agents never call or import from `get_data.py`.

## Corpus

- Dataset: [CRUMB](https://huggingface.co/datasets/jfkback/crumb) – `tip_of_the_tongue` task
- Scope: 50 eval queries + their referenced documents (each query guaranteed to have a match)
- Wikipedia full-document corpus; queries are "tip of the tongue" descriptions of entities

## Conventions

- Use `uv add <package>` to add dependencies (never `pip install` directly).
- Harness code (`src/evaluation/`) is Python 3.11+, stdlib + `pyproject.toml` packages only.
- Agent code lives entirely inside its own `agents/<name>/` folder.
- `data/` is generated; add it to `.gitignore` or commit explicitly after running `get_data.py`.
- The harness validates that `Preprocessor` is named exactly `Preprocessor` and inherits from
  `BasePreprocessor`. Agents that fail this check will not be evaluated.
- Requires a `.env` file at the project root with provider API keys (e.g. `GOOGLE_API_KEY`).
