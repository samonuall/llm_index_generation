"""
Aggregate and compare all saved results across algorithms, agents, and splits.

Usage:
    python -m src.evaluation.scripts.aggregate_results
    python -m src.evaluation.scripts.aggregate_results --split paper_retrieval
    python -m src.evaluation.scripts.aggregate_results --algorithm bm25
    python -m src.evaluation.scripts.aggregate_results --export results_summary.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd

_PROJECT_ROOT = Path(__file__).parents[3]
RESULTS_DIR = _PROJECT_ROOT / "results"


def load_all_results(
    algorithm: str = None,
    agent: str = None,
    split: str = None
) -> List[Dict]:
    """Load all result JSON files, optionally filtered."""
    if not RESULTS_DIR.exists():
        return []
    
    results = []
    
    # Traverse: results/<algorithm>/<agent>/*.json
    for algo_dir in RESULTS_DIR.iterdir():
        if not algo_dir.is_dir():
            continue
        if algorithm and algo_dir.name != algorithm:
            continue
        
        for agent_dir in algo_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            if agent and agent_dir.name != agent:
                continue
            
            for result_file in agent_dir.glob("*.json"):
                try:
                    with result_file.open('r') as f:
                        data = json.load(f)
                    
                    # Filter by split if specified
                    if split and data.get("split") != split:
                        continue
                    
                    # Flatten for easy tabular display
                    flat = {
                        "algorithm": data.get("metadata", {}).get("algorithm", algo_dir.name),
                        "agent": data.get("metadata", {}).get("agent", agent_dir.name),
                        "split": data.get("split", "unknown"),
                        "n_docs": data.get("config", {}).get("n_docs", 0),
                        "top_k": data.get("config", {}).get("top_k", 0),
                        "timestamp": data.get("metadata", {}).get("timestamp", "unknown"),
                    }
                    
                    # Add CRUMB metrics
                    crumb = data.get("crumb_metrics", {}) or {}
                    for k, v in crumb.items():
                        flat[k] = v
                    
                    results.append(flat)
                    
                except Exception as e:
                    print(f"⚠️  Error loading {result_file}: {e}")
    
    return results


def print_results_table(results: List[Dict], sort_by: str = "nDCG@10"):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return
    
    df = pd.DataFrame(results)
    
    # Sort by metric (descending, with NaN last)
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False, na_position='last')
    
    # Select columns to display
    display_cols = ["algorithm", "agent", "split", "n_docs", "nDCG@10", "nDCG@5", "R@100", "P@10", "RR@10"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    print("\n" + "="*100)
    print("AGGREGATED RESULTS")
    print("="*100)
    print(df[display_cols].to_string(index=False))
    print("="*100)
    print(f"\nTotal runs: {len(results)}")


def export_to_csv(results: List[Dict], output_path: str):
    """Export results to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Exported {len(results)} results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--algorithm", type=str, help="Filter by algorithm (e.g., 'bm25')")
    parser.add_argument("--agent", type=str, help="Filter by agent")
    parser.add_argument("--split", type=str, help="Filter by split")
    parser.add_argument("--sort-by", type=str, default="nDCG@10", help="Metric to sort by")
    parser.add_argument("--export", type=str, help="Export to CSV file")
    
    args = parser.parse_args()
    
    print(f"\nLoading results from: {RESULTS_DIR}")
    if args.algorithm:
        print(f"  Filter: algorithm = {args.algorithm}")
    if args.agent:
        print(f"  Filter: agent = {args.agent}")
    if args.split:
        print(f"  Filter: split = {args.split}")
    
    results = load_all_results(
        algorithm=args.algorithm,
        agent=args.agent,
        split=args.split
    )
    
    if not results:
        print("\n⚠️  No results found matching filters.")
        print("\nAvailable results structure:")
        if RESULTS_DIR.exists():
            for algo in RESULTS_DIR.iterdir():
                if algo.is_dir():
                    agents = [a.name for a in algo.iterdir() if a.is_dir()]
                    print(f"  {algo.name}/ → agents: {', '.join(agents)}")
        return
    
    print_results_table(results, sort_by=args.sort_by)
    
    if args.export:
        export_to_csv(results, args.export)


if __name__ == "__main__":
    main()