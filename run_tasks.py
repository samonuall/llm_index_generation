"""
run_tasks.py – Run the gemini_sdk_bright agent on multiple BRIGHT tasks sequentially.

For each task:
  1. Download data if not already present
  2. Run baseline eval (passthrough, no preprocessing) to get the internal floor
  3. Run the agent loop with best-of-N tracking
  4. Append results to results/bright_results.jsonl

Usage:
    uv run python run_tasks.py
    uv run python run_tasks.py --loops 5 --no-query-text
"""

from __future__ import annotations

import sys
import json
import pathlib
import argparse
import importlib.util
import datetime
import subprocess

_PROJECT_ROOT = pathlib.Path(__file__).parent

# Paper BM25 nDCG@10 baselines per task (Table 2, BRIGHT ICLR 2025).
# Values are decimals (paper reports as percentages, e.g. 15.0 → 0.150).
PAPER_BM25 = {
    "biology": 0.189,
    "earth_science": 0.272,
    "economics": 0.149,
    "psychology": 0.125,
    "robotics": 0.136,
    "stackoverflow": 0.184,
    "sustainable_living": 0.150,
    "leetcode": 0.244,
    "pony": 0.079,
    "aops": 0.062,
    "theoremqa_questions": 0.104,
    "theoremqa_theorems": 0.049,
}

RESULTS_FILE = _PROJECT_ROOT / "results" / "bright_results.jsonl"


def _add_paths() -> None:
    for p in [
        str(_PROJECT_ROOT / "src" / "evaluation" / "scripts"),
        str(_PROJECT_ROOT / "src" / "evaluation"),
        str(_PROJECT_ROOT / "src" / "agents"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_preprocessor(path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.Preprocessor()


def run_baseline(task: str) -> float:
    """Evaluate the passthrough baseline on a task; returns nDCG@10."""
    _add_paths()
    from test_preprocessing_bright import evaluate_bright  # type: ignore

    baseline_path = _PROJECT_ROOT / "src" / "agents" / "baseline" / "preprocess.py"
    preprocessor = _load_preprocessor(baseline_path, "_baseline_preprocess")
    results = evaluate_bright(preprocessor, task=task, top_k=10)
    return results["ndcg_at_k"]


def ensure_data(task: str) -> None:
    """Download BRIGHT data for a task if not already on disk."""
    data_dir = _PROJECT_ROOT / "data" / "bright" / task
    if data_dir.exists() and (data_dir / "documents.jsonl").exists():
        return
    print(f"\n[run_tasks] Downloading data for task: {task} ...")
    subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "src" / "evaluation" / "scripts" / "get_data_bright.py"),
            "--task", task,
        ],
        check=True,
    )


def append_result(record: dict) -> None:
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def print_summary(summary: list[dict]) -> None:
    print(f"\n\n{'=' * 72}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 72}")
    header = f"{'Task':<22} {'Baseline':>10} {'Best':>10} {'Paper BM25':>12} {'Δ vs Paper':>12}"
    print(header)
    print("-" * 72)
    for r in summary:
        paper = r.get("paper_bm25", 0.0) or 0.0
        delta = r["best_ndcg"] - paper
        beat = "✓" if delta > 0 else " "
        print(
            f"{r['task']:<22} {r['baseline_ndcg']:>10.4f} {r['best_ndcg']:>10.4f}"
            f" {paper:>12.4f} {delta:>+11.4f} {beat}"
        )
    print(f"{'=' * 72}")
    print(f"Full results saved to: {RESULTS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gemini_sdk_bright on multiple BRIGHT tasks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["biology", "earth_science", "economics", "psychology", "robotics"],
        help="Tasks to run (default: 5 StackExchange tasks)",
    )
    parser.add_argument("--loops", type=int, default=5, help="Agent loops per task (default: 5)")
    parser.add_argument(
        "--no-query-text",
        action="store_true",
        default=False,
        help="Omit raw query text from prompts (use when safety filters block content)",
    )
    args = parser.parse_args()

    _add_paths()
    from gemini_sdk_bright.agent import GeminiSdkBrightAgent  # type: ignore

    summary: list[dict] = []

    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"  TASK: {task}")
        print(f"{'=' * 60}")

        ensure_data(task)

        print(f"\n[run_tasks] Running baseline eval for {task} ...")
        baseline_ndcg = run_baseline(task)
        print(f"[run_tasks] Baseline nDCG@10: {baseline_ndcg:.4f}")

        agent = GeminiSdkBrightAgent(task=task, include_query_text=not args.no_query_text)
        agent.run(n_loops=args.loops)

        best_ndcg = getattr(agent, "_best_ndcg", baseline_ndcg)

        record = {
            "task": task,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_loops": args.loops,
            "baseline_ndcg": round(baseline_ndcg, 4),
            "best_ndcg": round(best_ndcg, 4),
            "paper_bm25": PAPER_BM25.get(task),
            "delta_vs_baseline": round(best_ndcg - baseline_ndcg, 4),
            "delta_vs_paper": round(best_ndcg - (PAPER_BM25.get(task) or 0.0), 4),
        }
        append_result(record)
        summary.append(record)

        print(
            f"\n[run_tasks] {task} done:"
            f" baseline={baseline_ndcg:.4f}"
            f" best={best_ndcg:.4f}"
            f" paper_bm25={PAPER_BM25.get(task)}"
        )

    print_summary(summary)


if __name__ == "__main__":
    main()
