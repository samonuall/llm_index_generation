"""
test_preprocessing_bright.py – BRIGHT evaluation harness. DO NOT MODIFY.

Mirrors test_preprocessing.py but targets the BRIGHT benchmark:
  - Loads from data/bright/{task}/
  - Primary metric: nDCG@10 (matches the BRIGHT paper's reported numbers)
  - Also reports Recall@k and MRR for reference
  - Handles excluded_ids: per-query docs that must be filtered from results
    before scoring (they appeared in the question itself)

CLI usage:
    uv run python src/evaluation/scripts/test_preprocessing_bright.py --agent <name>
    uv run python src/evaluation/scripts/test_preprocessing_bright.py --agent <name> --task sustainable_living --top-k 10

Programmatic usage (e.g. from an agent):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
    from test_preprocessing_bright import evaluate_bright
    results = evaluate_bright(Preprocessor(), task="sustainable_living", top_k=10)

Paper BM25 baseline (no preprocessing): nDCG@10 ≈ 14.8 (average across all tasks).
"""

from __future__ import annotations

import sys
import pathlib
import json
import argparse
import importlib
import math
from typing import List

_EVAL_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_EVAL_DIR / "scripts"))

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from schema import Document
from base import BasePreprocessor
from build_index import BM25Index

DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_TASK = "sustainable_living"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_documents(task: str) -> List[Document]:
    path = DATA_DIR / "bright" / task / "documents.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"BRIGHT documents not found at {path}. "
            f"Run: uv run python src/evaluation/scripts/get_data_bright.py --task {task}"
        )
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(Document(**d))
    return docs


def _load_queries(task: str) -> List[dict]:
    """Load query records as plain dicts (includes excluded_ids field)."""
    path = DATA_DIR / "bright" / task / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"BRIGHT queries not found at {path}. "
            f"Run: uv run python src/evaluation/scripts/get_data_bright.py --task {task}"
        )
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ndcg_at_k(retrieved_doc_ids: List[str], relevant: set, k: int) -> float:
    """Compute nDCG@k with binary relevance (0/1 gains)."""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bright(
    preprocessor: BasePreprocessor,
    task: str = DEFAULT_TASK,
    top_k: int = 10,
) -> dict:
    """
    Run the full BRIGHT eval pipeline for a given preprocessor.

    Returns a dict with keys:
      agent, task, ndcg_at_k, recall_at_k, mrr, top_k, n_queries, n_chunks,
      query_results (list of per-query dicts)
    """
    docs = _load_documents(task)
    query_records = _load_queries(task)

    label = preprocessor.name or type(preprocessor).__name__
    print(f"\n{'='*60}")
    print(f"Agent       : {label}")
    print(f"Task        : {task}")
    print(f"Description : {preprocessor.description or '(none)'}")
    print(f"{'='*60}")
    print(f"Preprocessing {len(docs)} documents ...")

    chunks = preprocessor.preprocess(docs)
    print(f"  -> {len(chunks)} chunks  ({len(chunks) / len(docs):.2f} avg per doc)")

    print("Building BM25 index ...")
    index = BM25Index(chunks)

    ndcg_total = 0.0
    recall_hits = 0
    mrr_total = 0.0
    query_results = []

    for q in query_records:
        excluded = set(q.get("excluded_ids", []))
        relevant = set(q["relevant_doc_ids"])

        # Retrieve more than top_k to allow for excluded_ids filtering
        raw_results = index.search(q["query_text"], top_k=top_k + len(excluded))
        # Filter excluded docs, then truncate to top_k
        filtered = [
            (chunk, score)
            for chunk, score in raw_results
            if chunk.doc_id not in excluded
        ][:top_k]

        retrieved_doc_ids = [chunk.doc_id for chunk, _ in filtered]

        # nDCG@k
        ndcg = _ndcg_at_k(retrieved_doc_ids, relevant, top_k)
        ndcg_total += ndcg

        # Recall@k
        hit = any(doc_id in relevant for doc_id in retrieved_doc_ids)
        if hit:
            recall_hits += 1

        # MRR
        reciprocal_rank = 0.0
        rank_of_first_hit = None
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                rank_of_first_hit = rank
                mrr_total += reciprocal_rank
                break

        query_results.append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "relevant_doc_ids": list(relevant),
            "excluded_ids": list(excluded),
            "retrieved_doc_ids": retrieved_doc_ids,
            "ndcg": ndcg,
            "hit": hit,
            "rank": rank_of_first_hit,
            "reciprocal_rank": reciprocal_rank,
        })

    n = len(query_records)
    ndcg_at_k = ndcg_total / n
    recall_at_k = recall_hits / n
    mrr = mrr_total / n

    print(f"\nResults  ({n} queries, top-{top_k}, task={task}):")
    print(f"  nDCG@{top_k:<3}  : {ndcg_at_k:.4f}   <- compare against paper BM25 baseline ~14.8")
    print(f"  Recall@{top_k:<3}: {recall_at_k:.4f}")
    print(f"  MRR       : {mrr:.4f}")
    print(f"{'='*60}\n")

    return {
        "agent": label,
        "task": task,
        "ndcg_at_k": ndcg_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
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
        description="Evaluate an agent preprocessor against the BRIGHT BM25 harness."
    )
    parser.add_argument("--agent", required=True, help="Agent folder name under agents/")
    parser.add_argument("--task", default=DEFAULT_TASK, help="BRIGHT task (default: sustainable_living)")
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
        print(f"Error: {module_path}.Preprocessor must inherit from BasePreprocessor.")
        sys.exit(1)

    evaluate_bright(preprocessor, task=args.task, top_k=args.top_k)


if __name__ == "__main__":
    main()
