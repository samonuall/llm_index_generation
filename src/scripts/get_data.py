"""
get_data.py – One-time data preparation script (run manually, never by agents).

Downloads documents and eval queries from the CRUMB tip-of-the-tongue dataset
(jfkback/crumb on HuggingFace) using streaming so the full ~1.7 GB corpus is
never stored on disk, then saves the selected subset to data/ as JSONL files.

Output files (overwritten on each run):
  data/documents.jsonl  –  one Document per line
  data/queries.jsonl    –  one EvalQuery per line

Usage:
  uv run python src/scripts/get_data.py
  uv run python src/scripts/get_data.py --n-queries 100
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
from pathlib import Path
from typing import Any, List, Set, Tuple

from datasets import load_dataset  # type: ignore

from schema import Document, EvalQuery

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO   = "jfkback/crumb"
HF_SPLIT  = "tip_of_the_tongue"
DATA_DIR  = Path(__file__).parents[2] / "data"
N_QUERIES = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_qrels(raw: Any) -> List[str]:
    """Return doc IDs with positive relevance from a full_document_qrels value."""
    if isinstance(raw, str):
        qrels = ast.literal_eval(raw)
    elif raw is None:
        return []
    else:
        qrels = raw
    return [str(q["id"]) for q in qrels if q.get("label", 0) > 0]


def _fetch(n_queries: int) -> Tuple[List[Document], List["EvalQuery"]]:
    # --- Step 1: stream eval queries, collect target doc IDs ----------------
    print(f"Streaming first {n_queries} evaluation queries from {HF_REPO} ...")
    q_stream = load_dataset(
        HF_REPO, "evaluation_queries", split=HF_SPLIT, streaming=True
    )

    queries: List[EvalQuery] = []
    target_doc_ids: Set[str] = set()

    for row in q_stream:
        rel_ids = _parse_qrels(row.get("full_document_qrels"))
        if not rel_ids:
            continue
        queries.append(
            EvalQuery(
                query_id=str(row["query_id"]),
                query_text=row["query_content"],
                relevant_doc_ids=rel_ids,
            )
        )
        target_doc_ids.update(rel_ids)
        if len(queries) >= n_queries:
            break

    print(f"  {len(queries)} queries referencing {len(target_doc_ids)} unique docs.")

    # --- Step 2: stream full-doc corpus, fetch only the target docs ----------
    print("Streaming full-document corpus to retrieve referenced docs ...")
    doc_stream = load_dataset(
        HF_REPO, "full_document_corpus", split=HF_SPLIT, streaming=True
    )

    docs: List[Document] = []
    remaining = set(target_doc_ids)

    for row in doc_stream:
        doc_id = str(row["document_id"])
        if doc_id in remaining:
            docs.append(
                Document(
                    doc_id=doc_id,
                    text=row["document_content"],
                    metadata={"parent_id": row.get("parent_id")},
                )
            )
            remaining.discard(doc_id)
            if not remaining:
                break

    if remaining:
        print(f"  Warning: {len(remaining)} doc IDs not found in corpus.")

    print(f"  {len(docs)} documents retrieved.")
    return docs, queries


def _save(docs: List[Document], queries: List[EvalQuery]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    doc_path = DATA_DIR / "documents.jsonl"
    with doc_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(dataclasses.asdict(doc)) + "\n")
    print(f"Saved {len(docs)} documents -> {doc_path}")

    q_path = DATA_DIR / "queries.jsonl"
    with q_path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(dataclasses.asdict(q)) + "\n")
    print(f"Saved {len(queries)} queries  -> {q_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download CRUMB data to data/")
    parser.add_argument("--n-queries", type=int, default=N_QUERIES)
    args = parser.parse_args()

    docs, queries = _fetch(args.n_queries)
    _save(docs, queries)
    print("Done.")


if __name__ == "__main__":
    main()
