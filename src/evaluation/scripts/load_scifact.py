"""
load_scifact.py – Download SciFact (BEIR) and convert to the llm_index_generation
data format, writing data/documents.jsonl and data/queries.jsonl.

Drop this file at the repo root and run:
    uv run python load_scifact.py
    uv run python load_scifact.py --split test      # default
    uv run python load_scifact.py --split train
    uv run python load_scifact.py --max-queries 50  # limit queries
    uv run python load_scifact.py --max-docs 2000   # limit corpus size

No HuggingFace streaming — SciFact downloads as a ~3 MB zip, so it's near-instant.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import csv
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
# Make src/evaluation importable so we can reuse the dataclasses
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from schema import Document, EvalQuery  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
DATA_DIR = ROOT / "data"
CACHE_ZIP = DATA_DIR / "scifact.zip"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_zip() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_ZIP.exists():
        print(f"  Found cached zip at {CACHE_ZIP}, skipping download.")
        return
    print(f"Downloading SciFact (~3 MB) from BEIR ...")
    urlretrieve(SCIFACT_URL, CACHE_ZIP)
    print(f"  Saved to {CACHE_ZIP}")


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def _read_jsonl_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(name) as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_tsv_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        reader = csv.DictReader(text, delimiter="\t")
        return list(reader)


def _load_scifact(split: str) -> tuple[list[Document], list[EvalQuery]]:
    """Read SciFact from the cached zip and return Documents + EvalQueries."""
    with zipfile.ZipFile(CACHE_ZIP) as zf:
        # --- Corpus (same for all splits) ---
        corpus_rows = _read_jsonl_from_zip(zf, "scifact/corpus.jsonl")

        # --- Queries ---
        queries_rows = _read_jsonl_from_zip(zf, f"scifact/queries.jsonl")

        # --- Qrels (relevance judgements) ---
        qrels_rows = _read_tsv_from_zip(zf, f"scifact/qrels/{split}.tsv")

    # Build qrels map: query_id -> set of relevant doc_ids (score > 0)
    qrels: dict[str, list[str]] = {}
    for row in qrels_rows:
        qid = row["query-id"]
        did = row["corpus-id"]
        score = int(row.get("score", 1))
        if score > 0:
            qrels.setdefault(qid, []).append(did)

    # --- Convert queries ---
    # Only keep queries that have at least one relevant doc in this split
    eval_queries: list[EvalQuery] = []
    for q in queries_rows:
        qid = str(q["_id"])
        rel_ids = qrels.get(qid)
        if not rel_ids:
            continue
        eval_queries.append(
            EvalQuery(
                query_id=qid,
                query_text=q["text"],
                relevant_doc_ids=rel_ids,
            )
        )

    # --- Determine which doc IDs are needed ---
    needed_doc_ids: set[str] = set()
    for eq in eval_queries:
        needed_doc_ids.update(eq.relevant_doc_ids)

    # --- Convert corpus ---
    # Always include relevant docs; pad with others up to max_docs (handled outside)
    relevant_docs: list[Document] = []
    extra_docs: list[Document] = []
    for row in corpus_rows:
        did = str(row["_id"])
        text = (row.get("title", "") + "\n\n" + row.get("text", "")).strip()
        doc = Document(
            doc_id=did,
            text=text,
            metadata={"title": row.get("title", "")},
        )
        if did in needed_doc_ids:
            relevant_docs.append(doc)
        else:
            extra_docs.append(doc)

    return relevant_docs, extra_docs, eval_queries


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(dataclasses.asdict(item)) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load SciFact into data/")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Which SciFact split to use for queries/qrels (default: test)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries saved (default: all)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap total corpus size; relevant docs always included (default: all)",
    )
    args = parser.parse_args()

    _download_zip()

    print(f"Parsing SciFact ({args.split} split) ...")
    relevant_docs, extra_docs, eval_queries = _load_scifact(args.split)

    # Optionally limit queries
    if args.max_queries is not None:
        eval_queries = eval_queries[: args.max_queries]
        # Recompute needed docs after query trim
        needed = {did for eq in eval_queries for did in eq.relevant_doc_ids}
        relevant_docs = [d for d in relevant_docs if d.doc_id in needed]

    # Optionally limit total corpus
    n_rel = len(relevant_docs)
    if args.max_docs is not None and args.max_docs > n_rel:
        extra_docs = extra_docs[: args.max_docs - n_rel]
    elif args.max_docs is None:
        pass  # keep all extras
    else:
        extra_docs = []  # max_docs <= n_rel, skip extras

    all_docs = relevant_docs + extra_docs

    # Write
    doc_path = DATA_DIR / "documents.jsonl"
    q_path = DATA_DIR / "queries.jsonl"
    _write_jsonl(doc_path, all_docs)
    _write_jsonl(q_path, eval_queries)

    print(
        f"\n  ✓ {len(relevant_docs)} relevant + {len(extra_docs)} extra docs "
        f"= {len(all_docs)} total  →  {doc_path}"
    )
    print(f"  ✓ {len(eval_queries)} queries  →  {q_path}")
    print("\nDone! You can now run:")
    print("  uv run python -m src.evaluation.scripts.test_preprocessing --agent baseline")


if __name__ == "__main__":
    main()