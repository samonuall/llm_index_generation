"""
test_preprocessing.py – Static evaluation harness. DO NOT MODIFY.

CLI usage:
    uv run python src/evaluation/scripts/test_preprocessing.py --agent <name>
    uv run python src/evaluation/scripts/test_preprocessing.py --agent <name> --top-k 20

Programmatic usage (from an agent's own test runner):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "src" / "evaluation" / "scripts"))
    from test_preprocessing import evaluate
    from my_preprocessor import Preprocessor
    results = evaluate(Preprocessor(), top_k=10)
"""

from __future__ import annotations

import sys
import pathlib
import json
import argparse
import importlib
from typing import List

# Make src/evaluation/ importable (for schema, base, build_index)
_EVAL_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_EVAL_DIR / "scripts"))

# Make src/ importable so `agents.<name>.preprocess` resolves (agents live in src/agents/)
_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from schema import Document, EvalQuery
from base import BasePreprocessor
from build_index import BM25Index

DATA_DIR = _PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_documents() -> List[Document]:
    docs = []
    with (DATA_DIR / "documents.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(Document(**d))
    return docs


def _load_queries() -> List[EvalQuery]:
    queries = []
    with (DATA_DIR / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries.append(EvalQuery(**q))
    return queries


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(preprocessor: BasePreprocessor, top_k: int = 10) -> dict:
    """
    Run the full eval pipeline for a given preprocessor.

    Returns a dict with keys: recall_at_k, mrr, top_k, n_queries, n_chunks.
    """
    docs = _load_documents()
    queries = _load_queries()

    label = preprocessor.name or type(preprocessor).__name__
    print(f"\n{'='*60}")
    print(f"Agent       : {label}")
    print(f"Description : {preprocessor.description or '(none)'}")
    print(f"{'='*60}")
    print(f"Preprocessing {len(docs)} documents ...")

    chunks = preprocessor.preprocess(docs)
    print(f"  -> {len(chunks)} chunks  ({len(chunks) / len(docs):.2f} avg per doc)")

    print("Building BM25 index ...")
    index = BM25Index(chunks)

    recall_hits = 0
    mrr_total = 0.0

    for query in queries:
        results = index.search(query.query_text, top_k=top_k)
        retrieved_doc_ids = [chunk.doc_id for chunk, _ in results]
        relevant = set(query.relevant_doc_ids)

        # Recall@k
        if any(doc_id in relevant for doc_id in retrieved_doc_ids):
            recall_hits += 1

        # MRR: reciprocal rank of first relevant doc
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant:
                mrr_total += 1.0 / rank
                break

    n = len(queries)
    recall_at_k = recall_hits / n
    mrr = mrr_total / n

    print(f"\nResults  ({n} queries, top-{top_k}):")
    print(f"  Recall@{top_k:<3}: {recall_at_k:.4f}")
    print(f"  MRR       : {mrr:.4f}")
    print(f"{'='*60}\n")

    return {
        "agent": label,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "top_k": top_k,
        "n_queries": n,
        "n_chunks": len(chunks),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent preprocessor against the static BM25 harness."
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent folder name under agents/  (e.g. 'baseline')",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    module_path = f"agents.{args.agent}.preprocess"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        print(f"Error: could not import '{module_path}': {e}")
        sys.exit(1)

    if not hasattr(module, "Preprocessor"):
        print(f"Error: {module_path} must define a class named 'Preprocessor'.")
        sys.exit(1)

    preprocessor = module.Preprocessor()

    if not isinstance(preprocessor, BasePreprocessor):
        print(
            f"Error: {module_path}.Preprocessor must inherit from BasePreprocessor."
        )
        sys.exit(1)

    evaluate(preprocessor, top_k=args.top_k)


if __name__ == "__main__":
    main()
