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
import random
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


def _stream_queries(n_queries: int) -> set[str]:
    """Stream evaluation queries directly to disk. Returns the union of all relevant doc IDs."""
    print(f"Streaming evaluation queries from {HF_REPO} (want {n_queries}) ...")
    q_stream = load_dataset(
        HF_REPO, "evaluation_queries", split=HF_SPLIT, streaming=True
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    q_path = DATA_DIR / "queries.jsonl"
    n_written = 0
    q_scanned = 0
    relevant_doc_ids: set[str] = set()

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
            relevant_doc_ids.update(rel_ids)
            n_written += 1
            pbar.update(1)
            pbar.set_postfix(scanned=q_scanned)
            if n_written >= n_queries:
                break

    print(f"  {n_written} queries saved -> {q_path} (scanned {q_scanned} rows).")
    return relevant_doc_ids


def _load_relevant_doc_ids_from_disk() -> set[str]:
    """Read the already-saved queries.jsonl and return the union of all relevant doc IDs."""
    q_path = DATA_DIR / "queries.jsonl"
    if not q_path.exists():
        return set()
    relevant: set[str] = set()
    with q_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                relevant.update(obj.get("relevant_doc_ids", []))
    return relevant


def _stream_docs(
    relevant_doc_ids: set[str],
    max_docs: int | None = None,
) -> None:
    """Stream documents, guaranteeing all relevant docs are saved.

    If max_docs > len(relevant_doc_ids), the remainder is filled with a
    uniformly random sample of non-relevant corpus documents (reservoir
    sampling, single pass). If max_docs is None, all corpus documents are
    saved (relevant ones first, then the rest).
    """
    n_relevant = len(relevant_doc_ids)
    if max_docs is not None and max_docs <= n_relevant:
        print(
            f"  Note: max_docs={max_docs} <= {n_relevant} relevant docs; "
            "saving relevant docs only."
        )
        max_docs = None  # collect all relevant, skip reservoir logic

    n_extra_slots = None if max_docs is None else max_docs - n_relevant
    cap_msg = (
        f" ({n_relevant} relevant + up to {n_extra_slots} random extras)"
        if n_extra_slots is not None
        else ""
    )
    print(f"Streaming full-document corpus{cap_msg} ...")

    doc_stream = load_dataset(
        HF_REPO, "full_document_corpus", split=HF_SPLIT, streaming=True
    )

    relevant_docs: dict[str, Document] = {}
    reservoir: list[Document] = []   # random sample of non-relevant docs
    n_seen_extra = 0

    with tqdm(desc="Documents", unit="doc") as pbar:
        for row in doc_stream:
            doc = Document(
                doc_id=str(row["document_id"]),
                text=row["document_content"],
                metadata={"parent_id": row.get("parent_id")},
            )
            pbar.update(1)
            if doc.doc_id in relevant_doc_ids:
                relevant_docs[doc.doc_id] = doc
            else:
                if n_extra_slots is None:
                    # no cap – keep everything
                    reservoir.append(doc)
                elif n_extra_slots > 0:
                    # reservoir sampling (Knuth / Algorithm R)
                    n_seen_extra += 1
                    if len(reservoir) < n_extra_slots:
                        reservoir.append(doc)
                    else:
                        j = random.randint(0, n_seen_extra - 1)
                        if j < n_extra_slots:
                            reservoir[j] = doc

            # Early exit: all relevant docs found AND reservoir is full
            if (
                n_extra_slots is not None
                and len(relevant_docs) == n_relevant
                and len(reservoir) == n_extra_slots
            ):
                # Can't stop early – we need to scan the whole corpus so that
                # the reservoir sample is unbiased. Keep going.
                pass

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = DATA_DIR / "documents.jsonl"
    all_docs = list(relevant_docs.values()) + reservoir

    with doc_path.open("w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(dataclasses.asdict(doc)) + "\n")

    missing = relevant_doc_ids - relevant_docs.keys()
    if missing:
        print(f"  WARNING: {len(missing)} relevant doc(s) not found in corpus: {missing}")
    print(
        f"  {len(relevant_docs)} relevant + {len(reservoir)} random docs "
        f"= {len(all_docs)} total saved -> {doc_path}."
    )


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
        relevant_doc_ids = _load_relevant_doc_ids_from_disk()
        if not relevant_doc_ids:
            print("  Warning: no saved queries found; relevant docs cannot be guaranteed.")
        _stream_docs(relevant_doc_ids=relevant_doc_ids, max_docs=args.max_docs)
    else:
        relevant_doc_ids = _stream_queries(args.n_queries)
        _stream_docs(relevant_doc_ids=relevant_doc_ids, max_docs=args.max_docs)
    print("Done.")


if __name__ == "__main__":
    main()
