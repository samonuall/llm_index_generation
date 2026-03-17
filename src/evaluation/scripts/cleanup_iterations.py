"""
cleanup_iterations.py – Clean up iteration artifacts while keeping summaries

Usage:
    # Clean up all but keep summaries and final iteration
    python -m src.evaluation.scripts.cleanup_iterations --split code_retrieval --agent lite_llm_agent
    
    # Keep last N iterations
    python -m src.evaluation.scripts.cleanup_iterations --split code_retrieval --agent lite_llm_agent --keep-last 2
    
    # Dry run (see what would be deleted)
    python -m src.evaluation.scripts.cleanup_iterations --split code_retrieval --agent lite_llm_agent --dry-run
    
    # Clean ALL agents on a split
    python -m src.evaluation.scripts.cleanup_iterations --split code_retrieval --all-agents
"""

import argparse
import pathlib
import sys
from typing import List

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
RESULTS_DIR = _PROJECT_ROOT / "results"


def find_iteration_files(split: str, agent: str) -> dict:
    """Find all iteration-related files for an agent."""
    results_dir = RESULTS_DIR / split
    if not results_dir.exists():
        return {}
    
    files = {
        'iterations': [],
        'crumb': [],
        'results': [],
        'summary': None,
        'plots': []
    }
    
    # Find all iteration files
    for f in results_dir.glob(f"{agent}_*_iter*.json"):
        files['iterations'].append(f)
    
    for f in results_dir.glob(f"{agent}_*_iter*_crumb.jsonl"):
        files['crumb'].append(f)
    
    for f in results_dir.glob(f"{agent}_*_iter*_results.json"):
        files['results'].append(f)
    
    # Find summary
    summary_file = results_dir / f"{agent}_iterations.json"
    if summary_file.exists():
        files['summary'] = summary_file
    
    # Find plots
    plots_dir = results_dir / "plots"
    if plots_dir.exists():
        for f in plots_dir.glob(f"{agent}_*.png"):
            files['plots'].append(f)
    
    return files


def cleanup_iterations(
    split: str,
    agent: str,
    keep_last: int = 1,
    dry_run: bool = False
) -> dict:
    """
    Clean up iteration files, keeping summary and optionally last N iterations.
    
    Returns dict with counts of files deleted/kept.
    """
    files = find_iteration_files(split, agent)
    
    stats = {
        'deleted': 0,
        'kept': 0,
        'size_freed': 0
    }
    
    if not files['summary']:
        print(f"⚠️  No summary file found for {agent} - skipping cleanup")
        return stats
    
    print(f"\n{'='*60}")
    print(f"Cleanup: {agent} on {split}")
    print(f"{'='*60}")
    print(f"Summary file: {files['summary'].name} (will keep)")
    print(f"Plots: {len(files['plots'])} files (will keep)")
    
    # Sort iteration files by iteration number
    def get_iter_num(path):
        import re
        match = re.search(r'_iter(\d+)', path.name)
        return int(match.group(1)) if match else 0
    
    files['iterations'].sort(key=get_iter_num)
    files['crumb'].sort(key=get_iter_num)
    files['results'].sort(key=get_iter_num)
    
    # Determine which to keep
    to_delete = []
    to_keep = []
    
    # Keep last N iterations
    if keep_last > 0 and len(files['iterations']) > keep_last:
        to_keep.extend(files['iterations'][-keep_last:])
        to_delete.extend(files['iterations'][:-keep_last])
    else:
        to_keep.extend(files['iterations'])
    
    # Delete ALL crumb and results files (they're redundant)
    to_delete.extend(files['crumb'])
    to_delete.extend(files['results'])
    
    print(f"\nIteration files: {len(files['iterations'])} total")
    print(f"  Keeping: {len([f for f in files['iterations'] if f in to_keep])}")
    print(f"  Deleting: {len([f for f in files['iterations'] if f in to_delete])}")
    
    print(f"\nCRUMB files: {len(files['crumb'])} (deleting all)")
    print(f"Query results files: {len(files['results'])} (deleting all)")
    
    # Calculate space
    total_size = sum(f.stat().st_size for f in to_delete)
    
    print(f"\n💾 Space to free: {total_size / 1024 / 1024:.2f} MB")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files deleted")
        print("\nWould delete:")
        for f in to_delete[:5]:
            print(f"  - {f.name}")
        if len(to_delete) > 5:
            print(f"  ... and {len(to_delete) - 5} more")
    else:
        print("\n🗑️  Deleting files...")
        for f in to_delete:
            try:
                f.unlink()
                stats['deleted'] += 1
                stats['size_freed'] += f.stat().st_size if f.exists() else 0
            except Exception as e:
                print(f"  Error deleting {f.name}: {e}")
        
        stats['kept'] = len(to_keep) + 1 + len(files['plots'])  # +1 for summary
        
        print(f"✓ Deleted {stats['deleted']} files")
        print(f"✓ Freed {stats['size_freed'] / 1024 / 1024:.2f} MB")
        print(f"✓ Kept {stats['kept']} important files")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean up iteration artifacts")
    parser.add_argument("--split", required=True, help="Split name")
    parser.add_argument("--agent", help="Agent name")
    parser.add_argument("--all-agents", action="store_true", 
                       help="Clean all agents on this split")
    parser.add_argument("--keep-last", type=int, default=1,
                       help="Keep last N iterations (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without deleting")
    
    args = parser.parse_args()
    
    if not args.agent and not args.all_agents:
        print("Error: Must specify --agent or --all-agents")
        sys.exit(1)
    
    agents = []
    if args.all_agents:
        # Find all agents with iteration files
        results_dir = RESULTS_DIR / args.split
        if not results_dir.exists():
            print(f"No results found for split: {args.split}")
            sys.exit(1)
        
        for f in results_dir.glob("*_iterations.json"):
            agent_name = f.name.replace("_iterations.json", "")
            agents.append(agent_name)
        
        if not agents:
            print(f"No agents with iterations found for split: {args.split}")
            sys.exit(1)
    else:
        agents = [args.agent]
    
    total_stats = {'deleted': 0, 'kept': 0, 'size_freed': 0}
    
    for agent in agents:
        stats = cleanup_iterations(args.split, agent, args.keep_last, args.dry_run)
        total_stats['deleted'] += stats['deleted']
        total_stats['kept'] += stats['kept']
        total_stats['size_freed'] += stats['size_freed']
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Agents cleaned: {len(agents)}")
    print(f"Files deleted: {total_stats['deleted']}")
    print(f"Files kept: {total_stats['kept']}")
    print(f"Space freed: {total_stats['size_freed'] / 1024 / 1024:.2f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()