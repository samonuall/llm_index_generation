# LLM Index Generation – CLAUDE.md

## Project Goal

Research project testing approaches for an agent that iteratively writes preprocessing code to
improve retrieval quality. The agent gets feedback from a static evaluation harness and iterates.

## Repository Structure

```
llm_index_generation/
├── data/                      # Pre-fetched corpus subset (NEVER modified by agents)
│   ├── documents.jsonl        # 50 Documents from CRUMB tip-of-the-tongue
│   └── queries.jsonl          # Matching EvalQuery objects
├── src/
│   └── scripts/               # Static harness – DO NOT MODIFY
│       ├── schema.py          # Shared data classes (Document, Chunk, EvalQuery)
│       ├── get_data.py        # One-time data prep: downloads & saves to data/
│       ├── build_index.py     # Builds a BM25/embedding index from Chunks
│       └── test_preprocessing.py  # Runs eval queries, returns retrieval metrics
├── agents/                    # One sub-folder per agent / experiment
│   └── <agent_name>/
│       ├── preprocess.py      # Must expose preprocess(docs) -> List[Chunk]
│       └── ...                # Agent can create any other files here freely
├── pyproject.toml
└── README.md
```

## Standard Data Classes  (`src/scripts/schema.py`)

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
    text: str         # Text that will be indexed / embedded
    metadata: dict    # Optional extra fields

@dataclass
class EvalQuery:
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]   # Ground-truth doc ids
```

Harness scripts import these with `from schema import ...` (same directory).
Agents should copy this import pattern, adding `src/scripts/` to `sys.path` if needed:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "src" / "scripts"))
from schema import Document, Chunk
```

## Agent Interface

Each agent must provide `agents/<agent_name>/preprocess.py` exposing:

```python
def preprocess(docs: List[Document]) -> List[Chunk]:
    """
    Transform raw documents into chunks ready for indexing.
    - Must return at least one Chunk per Document.
    - chunk.doc_id must match the source document's doc_id.
    """
```

The harness loads Documents from `data/documents.jsonl`, calls `preprocess(docs)`,
indexes the returned Chunks, runs the queries from `data/queries.jsonl`, and reports metrics.

## Evaluation Pipeline (`src/scripts/`)

1. Load data from `data/documents.jsonl` and `data/queries.jsonl`.
2. Call agent's `preprocess(docs)` → `List[Chunk]`.
3. `build_index.py` indexes chunks (BM25 by default).
4. `test_preprocessing.py` runs eval queries, computes **Recall@k** and **MRR**, prints report.
   Entry point: `evaluate(preprocess_fn, top_k=10) -> dict`

## Data Preparation (`get_data.py`)

Run manually to refresh `data/`:

```bash
uv run python src/scripts/get_data.py            # default: 50 queries
uv run python src/scripts/get_data.py --n-queries 100
```

Streams from HuggingFace (`jfkback/crumb`, `tip_of_the_tongue` split) — no large files stored.
Agents never call or import from `get_data.py`.

## Corpus

- Dataset: [CRUMB](https://huggingface.co/datasets/jfkback/crumb) – `tip_of_the_tongue` task
- Scope: 50 eval queries + their referenced documents (each query guaranteed to have a match)
- Wikipedia full-document corpus; queries are "tip of the tongue" descriptions of entities

## Conventions

- Use `uv add <package>` to add dependencies (never `pip install` directly).
- Harness scripts (`src/scripts/`) are Python 3.11+, stdlib + `pyproject.toml` packages only.
- Agent code lives entirely inside its own `agents/<name>/` folder.
- `data/` is generated; add it to `.gitignore` or commit explicitly after running `get_data.py`.
