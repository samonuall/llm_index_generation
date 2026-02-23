"""
test_preprocessing.py – Static evaluation harness. DO NOT MODIFY.

CLI usage:
    uv run python src/evaluation/scripts/test_preprocessing.py --agent <name>
    uv run python src/evaluation/scripts/test_preprocessing.py --agent <name> --top-k 20
"""

from __future__ import annotations

import sys
import math
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

def evaluate(preprocessor: BasePreprocessor, top_k: int = 100, ndcg_k: int = 10) -> dict:
    """
    Run the full eval pipeline for a given preprocessor.

    Returns a dict with keys: recall_at_k, ndcg, ndcg_k, top_k, n_queries, n_chunks.
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
    ndcg_total = 0.0
    query_results = []

    for query in queries:
        results = index.search(query.query_text, top_k=top_k)
        retrieved_doc_ids = [chunk.doc_id for chunk, _ in results]
        relevant = set(query.relevant_doc_ids)

        # Recall@top_k
        hit = any(doc_id in relevant for doc_id in retrieved_doc_ids)
        if hit:
            recall_hits += 1

        # Rank of first relevant doc (for per-query breakdown)
        rank_of_first_hit = None
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant:
                rank_of_first_hit = rank
                break

        # nDCG@ndcg_k
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids[:ndcg_k], start=1):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_count = min(len(relevant), ndcg_k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
        ndcg_score = dcg / idcg if idcg > 0 else 0.0
        ndcg_total += ndcg_score

        query_results.append({
            "query_id": query.query_id,
            "query_text": query.query_text,
            "relevant_doc_ids": list(query.relevant_doc_ids),
            "retrieved_doc_ids": retrieved_doc_ids,
            "hit": hit,
            "rank": rank_of_first_hit,
            "ndcg": ndcg_score,
        })

    n = len(queries)
    recall_at_k = recall_hits / n
    ndcg = ndcg_total / n

    print(f"\nResults  ({n} queries, top-{top_k}):")
    print(f"  Recall@{top_k:<3} : {recall_at_k:.4f}")
    print(f"  nDCG@{ndcg_k:<5} : {ndcg:.4f}")
    print(f"{'='*60}\n")

    return {
        "agent": label,
        "recall_at_k": recall_at_k,
        "ndcg": ndcg,
        "ndcg_k": ndcg_k,
        "top_k": top_k,
        "n_queries": n,
        "n_chunks": len(chunks),
        "query_results": query_results,
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
    parser.add_argument("--top-k", type=int, default=100)
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
