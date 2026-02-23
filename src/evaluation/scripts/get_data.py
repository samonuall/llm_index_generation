"""
get_data.py – One-time data preparation script (run manually, never by agents).

Downloads documents and eval queries from the CRUMB tip-of-the-tongue dataset
(jfkback/crumb on HuggingFace) using streaming so the full ~1.7 GB corpus is
never stored on disk, then saves the selected subset to data/ as JSONL files.

Output files (overwritten on each run):
  data/documents.jsonl  –  one Document per line
  data/queries.jsonl    –  one EvalQuery per line

Usage:
  uv run python -m src.evaluation.scripts.get_data
  uv run python -m src.evaluation.scripts.get_data --n-queries 100
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

# Make src/evaluation/ importable (for schema)
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset  # type: ignore
from tqdm import tqdm

from schema import Document, EvalQuery

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO   = "jfkback/crumb"
HF_SPLIT  = "tip_of_the_tongue"
DATA_DIR  = Path(__file__).parents[3] / "data"
N_QUERIES = 135


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_qrels(raw: Any) -> list[str]:
    """Return doc IDs with positive relevance from a full_document_qrels value."""
    if isinstance(raw, str):
        qrels = ast.literal_eval(raw)
    elif raw is None:
        return []
    else:
        qrels = raw
    return [str(q["id"]) for q in qrels if q.get("label", 0) > 0]


def _stream_queries(n_queries: int) -> None:
    """Stream evaluation queries directly to disk."""
    print(f"Streaming evaluation queries from {HF_REPO} (want {n_queries}) ...")
    q_stream = load_dataset(
        HF_REPO, "evaluation_queries", split=HF_SPLIT, streaming=True
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    q_path = DATA_DIR / "queries.jsonl"
    n_written = 0
    q_scanned = 0

    with q_path.open("w", encoding="utf-8") as f, \
         tqdm(total=n_queries, desc="Queries", unit="query") as pbar:
        for row in q_stream:
            q_scanned += 1
            rel_ids = _parse_qrels(row.get("full_document_qrels"))
            if not rel_ids:
                pbar.set_postfix(scanned=q_scanned, no_qrels=q_scanned - n_written)
                continue
            query = EvalQuery(
                query_id=str(row["query_id"]),
                query_text=row["query_content"],
                relevant_doc_ids=rel_ids,
            )
            f.write(json.dumps(dataclasses.asdict(query)) + "\n")
            n_written += 1
            pbar.update(1)
            pbar.set_postfix(scanned=q_scanned)
            if n_written >= n_queries:
                break

    print(f"  {n_written} queries saved -> {q_path} (scanned {q_scanned} rows).")


def _stream_docs(max_docs: int | None = None) -> None:
    """Stream the full document corpus directly to disk."""
    cap_msg = f" (capped at {max_docs})" if max_docs else ""
    print(f"Streaming full-document corpus{cap_msg} ...")
    doc_stream = load_dataset(
        HF_REPO, "full_document_corpus", split=HF_SPLIT, streaming=True
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = DATA_DIR / "documents.jsonl"
    n_written = 0

    with doc_path.open("w", encoding="utf-8") as f, \
         tqdm(desc="Documents", unit="doc") as pbar:
        for row in doc_stream:
            doc = Document(
                doc_id=str(row["document_id"]),
                text=row["document_content"],
                metadata={"parent_id": row.get("parent_id")},
            )
            f.write(json.dumps(dataclasses.asdict(doc)) + "\n")
            n_written += 1
            pbar.update(1)
            if max_docs and n_written >= max_docs:
                break

    print(f"  {n_written} documents saved -> {doc_path}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download CRUMB data to data/")
    parser.add_argument("--n-queries", type=int, default=N_QUERIES)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap the number of corpus documents saved (default: all)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--queries-only",
        action="store_true",
        help="Download and save only queries (skip documents)",
    )
    mode.add_argument(
        "--docs-only",
        action="store_true",
        help="Download and save only documents (skip queries)",
    )
    args = parser.parse_args()

    if args.queries_only:
        _stream_queries(args.n_queries)
    elif args.docs_only:
        _stream_docs(max_docs=args.max_docs)
    else:
        _stream_queries(args.n_queries)
        _stream_docs(max_docs=args.max_docs)
    print("Done.")


if __name__ == "__main__":
    main()
