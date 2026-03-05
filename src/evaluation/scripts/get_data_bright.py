"""
get_data_bright.py – One-time data preparation script for the BRIGHT benchmark.

Downloads documents and eval queries from xlangai/BRIGHT on HuggingFace and
saves them to data/bright/{task}/ as JSONL files.

Output files (overwritten on each run):
  data/bright/{task}/documents.jsonl  –  one Document per line (doc_id, text, metadata)
  data/bright/{task}/queries.jsonl    –  one record per line with fields:
                                         query_id, query_text, relevant_doc_ids,
                                         excluded_ids (BRIGHT-specific: docs seen
                                         in the question, excluded from eval scoring)

Usage:
  uv run python src/evaluation/scripts/get_data_bright.py
  uv run python src/evaluation/scripts/get_data_bright.py --task sustainable_living
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import pathlib
from typing import List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from datasets import load_dataset  # type: ignore
from schema import Document

HF_REPO = "xlangai/bright"
DEFAULT_TASK = "sustainable_living"
DATA_DIR = pathlib.Path(__file__).parents[3] / "data"

ALL_TASKS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa",
]


def _fetch(task: str) -> Tuple[List[Document], List[dict]]:
    """Return (documents, query_records).

    Each query_record is a plain dict with:
      query_id, query_text, relevant_doc_ids, excluded_ids
    We don't use EvalQuery here because EvalQuery has no excluded_ids field
    and we must not modify schema.py.
    """
    print(f"Loading BRIGHT documents for task='{task}' from {HF_REPO} ...")
    doc_dataset = load_dataset(HF_REPO, "documents")[task]
    docs: List[Document] = []
    for row in doc_dataset:
        docs.append(
            Document(
                doc_id=str(row["id"]),
                text=row["content"],
                metadata={},
            )
        )
    print(f"  {len(docs)} documents loaded.")

    print(f"Loading BRIGHT examples (queries) for task='{task}' ...")
    ex_dataset = load_dataset(HF_REPO, "examples")[task]
    query_records: List[dict] = []
    for row in ex_dataset:
        gold_ids = [str(g) for g in (row.get("gold_ids") or [])]
        excluded_ids = [str(e) for e in (row.get("excluded_ids") or [])]
        if not gold_ids:
            continue
        query_records.append({
            "query_id": str(row["id"]),
            "query_text": row["query"],
            "relevant_doc_ids": gold_ids,
            "excluded_ids": excluded_ids,
        })
    print(f"  {len(query_records)} queries loaded.")
    return docs, query_records


def _save(docs: List[Document], query_records: List[dict], task: str) -> None:
    out_dir = DATA_DIR / "bright" / task
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_path = out_dir / "documents.jsonl"
    with doc_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(dataclasses.asdict(doc)) + "\n")
    print(f"Saved {len(docs)} documents -> {doc_path}")

    q_path = out_dir / "queries.jsonl"
    with q_path.open("w", encoding="utf-8") as f:
        for record in query_records:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(query_records)} queries  -> {q_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BRIGHT data to data/bright/{task}/")
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        choices=ALL_TASKS,
        help=f"BRIGHT task/subset to download (default: {DEFAULT_TASK})",
    )
    args = parser.parse_args()

    docs, query_records = _fetch(args.task)
    _save(docs, query_records, args.task)
    print("Done.")


if __name__ == "__main__":
    main()
