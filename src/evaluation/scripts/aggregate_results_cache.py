"""
aggregate_results.py

Scan results/ directory and produce a summary table:
Agent | Split | Top-K | nDCG@10

Usage:
    python -m src.evaluation.scripts.aggregate_results
    python -m src.evaluation.scripts.aggregate_results --split paper_retrieval
    python -m src.evaluation.scripts.aggregate_results --csv summary.csv
"""

from __future__ import annotations
import pathlib
import json
import argparse
from collections import defaultdict

PROJECT_ROOT = pathlib.Path(__file__).parents[3]
RESULTS_DIR = PROJECT_ROOT / "results"


def load_all_results(split_filter: str | None = None):
    rows = []

    if not RESULTS_DIR.exists():
        print("No results/ directory found.")
        return rows

    for split_dir in RESULTS_DIR.iterdir():
        if not split_dir.is_dir():
            continue

        split = split_dir.name

        if split_filter and split != split_filter:
            continue

        for file in split_dir.glob("*.json"):
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            rows.append({
                "agent": data["agent"],
                "split": split,
                "top_k": data["config"]["top_k"],
                "recall": data["metrics"]["recall_at_k"],
                "ndcg": data["metrics"]["ndcg"],
            })

    return rows


def print_table(rows):
    if not rows:
        print("No results found.")
        return

    # Group by agent
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["agent"]].append(r)

    for agent in sorted(grouped.keys()):
        print("\n" + "=" * 80)
        print(f"{agent}")
        print("=" * 80)
        print(f"{'Split':<25} {'Top-K':<8} {'Recall@K':<12} {'nDCG@10':<10}")
        print("-" * 80)

        agent_rows = sorted(
            grouped[agent],
            key=lambda r: (r["split"], r["top_k"])
        )

        for r in agent_rows:
            print(
                f"{r['split']:<25} "
                f"{r['top_k']:<8} "
                f"{r['recall']:<12.4f} "
                f"{r['ndcg']:<10.4f}"
            )

    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="Filter by split name")
    parser.add_argument("--csv", type=str, help="Export to CSV file")
    args = parser.parse_args()

    rows = load_all_results(split_filter=args.split)

    print_table(rows)

    if args.csv:
        export_csv(rows, pathlib.Path(args.csv))


if __name__ == "__main__":
    main()