"""
get_data.py — Downloads corpus + queries for CRUMB splits using STREAMING.

Usage:
    python -m src.evaluation.scripts.get_data --split paper_retrieval
    python -m src.evaluation.scripts.get_data --split tip_of_the_tongue
    python -m src.evaluation.scripts.get_data --split code_retrieval --limit 5000

Caches to: data/<split>/documents.jsonl and data/<split>/queries.jsonl
Re-running with --limit overwrites existing cached data for that split.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm

SPLIT_MAP = {
    "tip_of_the_tongue":              "tip_of_the_tongue",
    "paper_retrieval":                "paper_retrieval",
    "stack_exchange":                 "stack_exchange",
    "clinical_trial":                 "clinical_trial",
    "legal_qa":                       "legal_qa",
    "theorem_retrieval":              "theorem_retrieval",
    "code_retrieval":                 "code_retrieval",
    "set_operation_entity_retrieval": "set_operation_entity_retrieval",
}


def get_cache_dir(split: str) -> Path:
    """Get cache directory for a split: data/<split>/"""
    base_dir = Path(__file__).parents[3] / "data"
    cache_dir = base_dir / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_queries(split: str, n_queries: int = None) -> List[Dict]:
    crumb_name = SPLIT_MAP[split]
    cache_dir = get_cache_dir(split)
    cache_file = cache_dir / "queries.jsonl"

    if cache_file.exists():
        print(f"✓ Loading cached queries from {cache_file}")
        with cache_file.open("r") as f:
            queries = [json.loads(line) for line in f]
        print(f"  Loaded {len(queries)} queries")
        return queries

    print(f"Downloading queries for {split}...")
    dataset = load_dataset("jfkback/crumb", "evaluation_queries", split=crumb_name)

    queries = []
    for item in dataset:
        qrels = item.get("full_document_qrels") or item.get("passage_qrels") or []
        relevant_ids = [q["id"] for q in qrels if q.get("label", 0) > 0]
        if relevant_ids:
            queries.append({
                "query_id": item["query_id"],
                "query_content": item["query_content"],
                "relevant_doc_ids": relevant_ids,
            })

    if n_queries and n_queries < len(queries):
        queries = queries[:n_queries]

    print(f"Caching {len(queries)} queries to {cache_file}")
    with cache_file.open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    return queries


def load_full_corpus_streaming(split: str, max_docs: Optional[int] = None) -> List[Dict]:
    """
    Download corpus using STREAMING to avoid downloading all splits.

    Args:
        split: CRUMB split name
        max_docs: Maximum number of documents to download (None = all)
    """
    crumb_name = SPLIT_MAP[split]
    cache_dir = get_cache_dir(split)
    cache_file = cache_dir / "documents.jsonl"

    if cache_file.exists():
        print(f"✓ Loading cached corpus from {cache_file}")
        with cache_file.open("r") as f:
            docs = [json.loads(line) for line in f]
        print(f"  Loaded {len(docs)} documents")
        return docs

    if max_docs:
        print(f"Downloading corpus for {split} (streaming mode, limit={max_docs:,})...")
    else:
        print(f"Downloading corpus for {split} (streaming mode)...")

    # Use streaming=True to only fetch this split
    corpus_dataset = load_dataset(
        "jfkback/crumb",
        "full_document_corpus",
        split=crumb_name,
        streaming=True,
    )

    docs = []
    print("  Streaming documents...")

    # Use tqdm for progress if available
    try:
        pbar = tqdm(total=max_docs, unit=" docs")
        for item in corpus_dataset:
            docs.append({
                "doc_id": str(item["document_id"]),
                "text": item["document_content"],
                "metadata": {},
            })
            pbar.update(1)

            if max_docs and len(docs) >= max_docs:
                break
        pbar.close()
    except ImportError:
        for item in corpus_dataset:
            docs.append({
                "doc_id": str(item["document_id"]),
                "text": item["document_content"],
                "metadata": {},
            })
            if len(docs) % 10000 == 0:
                print(f"    Downloaded {len(docs):,} documents...")

            if max_docs and len(docs) >= max_docs:
                break

    print(f"\n  Total downloaded: {len(docs):,} documents")

    print(f"Caching to {cache_file}")
    with cache_file.open("w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    return docs


def main():
    parser = argparse.ArgumentParser(description="Download CRUMB corpus + queries (STREAMING)")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=list(SPLIT_MAP.keys()),
        help="CRUMB split name"
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=None,
        help="Limit number of queries (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to download (default: all)"
    )

    args = parser.parse_args()

    cache_dir = get_cache_dir(args.split)
    print(f"\n{'='*70}")
    print(f"CRUMB Data Download (Streaming) — Split: {args.split}")
    if args.limit:
        print(f"Document limit: {args.limit:,}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*70}\n")

    queries = load_queries(args.split, args.n_queries)
    docs = load_full_corpus_streaming(args.split, args.limit)

    # Report stats
    relevant_ids = {str(rid) for q in queries for rid in q["relevant_doc_ids"]}
    print(f"\n{'='*70}")
    print(f"✓ Download complete")
    print(f"  Split       : {args.split}")
    print(f"  Queries     : {len(queries)}")
    print(f"  Total docs  : {len(docs):,}")
    print(f"  Relevant    : {len(relevant_ids)}")
    print(f"  Distractors : {len(docs) - len(relevant_ids):,}")
    print(f"  Location    : {cache_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
