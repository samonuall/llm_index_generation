"""
download_val_queries.py — migrate / re-split existing query caches.

For each split that has cached data, pools all existing query records (from
validation_queries.jsonl, evaluation_queries.jsonl, or the old queries.jsonl)
and rewrites both files according to query_splits.json (1:5 val:eval).

Run once after pulling the change_queries branch:
    uv run python download_val_queries.py
"""
import json
from pathlib import Path

SPLITS = [
    "tip_of_the_tongue",
    "paper_retrieval",
    "stack_exchange",
    "clinical_trial",
    "legal_qa",
    "theorem_retrieval",
    "code_retrieval",
    "set_operation_entity_retrieval",
]

BASE = Path("data")
SPLITS_JSON = Path("query_splits.json")


def _apply_split(all_queries, split):
    if SPLITS_JSON.exists():
        config = json.loads(SPLITS_JSON.read_text())
        if split in config:
            by_id = {q["query_id"]: q for q in all_queries}
            val = [by_id[qid] for qid in config[split]["validation"] if qid in by_id]
            evl = [by_id[qid] for qid in config[split]["evaluation"] if qid in by_id]
            return val, evl
    n_val = max(1, len(all_queries) // 6)
    return all_queries[:n_val], all_queries[n_val:]


def _read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l]


def main():
    if not SPLITS_JSON.exists():
        print("ERROR: query_splits.json not found. Run generate_query_splits.py first.")
        return

    for split in SPLITS:
        cache_dir = BASE / split
        if not cache_dir.exists():
            continue

        print(f"\nProcessing {split}...")

        # Pool from all possible source files
        seen = {}
        for fname in ("queries.jsonl", "validation_queries.jsonl", "evaluation_queries.jsonl"):
            f = cache_dir / fname
            if f.exists():
                for q in _read_jsonl(f):
                    seen[q["query_id"]] = q

        if not seen:
            print("  No query files found, skipping")
            continue

        all_queries = list(seen.values())
        print(f"  Pooled {len(all_queries)} total queries")

        val_queries, eval_queries = _apply_split(all_queries, split)
        print(f"  -> {len(val_queries)} val, {len(eval_queries)} eval")

        val_file = cache_dir / "validation_queries.jsonl"
        eval_file = cache_dir / "evaluation_queries.jsonl"

        with val_file.open("w") as f:
            for q in val_queries:
                f.write(json.dumps(q) + "\n")

        with eval_file.open("w") as f:
            for q in eval_queries:
                f.write(json.dumps(q) + "\n")

        # Remove legacy file now that both split files are written
        old = cache_dir / "queries.jsonl"
        if old.exists():
            old.unlink()
            print(f"  Removed legacy {old.name}")

        print(f"  Wrote {val_file.name} and {eval_file.name}")


if __name__ == "__main__":
    main()
