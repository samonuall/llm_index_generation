"""
archive_agent_iterations.py – Archive and clean up agent iteration code files

Archives iteration code files (preprocess_iter_*.py) to timestamped zips
and optionally clears them from the agent's iterations/ folder.

Usage:
    # Archive iterations for specific agent
    python -m src.evaluation.scripts.archive_agent_iterations --agent lite_llm_agent
    
    # Archive and clear
    python -m src.evaluation.scripts.archive_agent_iterations --agent lite_llm_agent --clear
    
    # Archive all agents
    python -m src.evaluation.scripts.archive_agent_iterations --all-agents
    
    # Archive all and clear
    python -m src.evaluation.scripts.archive_agent_iterations --all-agents --clear
    
    # Keep only best iteration (based on nDCG)
    python -m src.evaluation.scripts.archive_agent_iterations --agent lite_llm_agent --keep-best

Archives saved to: results/archives/<agent>_iterations_<timestamp>.zip
"""

import argparse
import shutil
import pathlib
import json
import sys
from datetime import datetime
from typing import List, Optional

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
AGENTS_DIR = _PROJECT_ROOT / "src" / "agents"
RESULTS_DIR = _PROJECT_ROOT / "results"
ARCHIVE_DIR = RESULTS_DIR / "archives"


def find_all_agents() -> List[str]:
    """Find all agent directories."""
    if not AGENTS_DIR.exists():
        return []
    
    agents = []
    for d in AGENTS_DIR.iterdir():
        if d.is_dir() and (d / "preprocess.py").exists():
            agents.append(d.name)
    
    return sorted(agents)


def get_best_iteration(agent: str) -> Optional[int]:
    """
    Find the best iteration based on results files.
    Looks for highest nDCG@10 across all splits.
    """
    best_iter = None
    best_ndcg = -1
    
    # Search all splits for this agent's iteration results
    for split_dir in RESULTS_DIR.iterdir():
        if not split_dir.is_dir():
            continue
        
        summary_file = split_dir / f"{agent}_iterations.json"
        if not summary_file.exists():
            continue
        
        try:
            with summary_file.open('r') as f:
                data = json.load(f)
                for it in data.get('iterations', []):
                    ndcg = it.get('metrics', {}).get('ndcg_at_10', 0)
                    iter_num = it.get('iteration', 0)
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_iter = iter_num
        except Exception:
            continue
    
    return best_iter


def archive_agent_iterations(
    agent: str,
    clear: bool = False,
    keep_best: bool = False
) -> dict:
    """
    Archive iteration code files for an agent.
    
    Returns dict with stats.
    """
    agent_dir = AGENTS_DIR / agent
    iterations_dir = agent_dir / "iterations"
    
    stats = {
        'agent': agent,
        'files_found': 0,
        'files_archived': 0,
        'files_deleted': 0,
        'archive_path': None
    }
    
    if not iterations_dir.exists():
        print(f"⚠️  No iterations folder for {agent}")
        return stats
    
    files = sorted(iterations_dir.glob("preprocess_iter_*.py"))
    if not files:
        print(f"⚠️  No iteration files found for {agent}")
        return stats
    
    stats['files_found'] = len(files)
    
    print(f"\n{'='*60}")
    print(f"Agent: {agent}")
    print(f"{'='*60}")
    print(f"Found {len(files)} iteration files")
    
    # Create archive
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{agent}_iterations_{timestamp}"
    archive_path = ARCHIVE_DIR / archive_name
    
    print(f"\n📦 Creating archive...")
    shutil.make_archive(str(archive_path), 'zip', iterations_dir)
    stats['archive_path'] = f"{archive_path}.zip"
    stats['files_archived'] = len(files)
    
    archive_size = pathlib.Path(stats['archive_path']).stat().st_size / 1024
    print(f"✓ Archived to: {archive_path}.zip ({archive_size:.1f} KB)")
    
    # Handle clearing/keeping best
    if keep_best:
        best_iter = get_best_iteration(agent)
        if best_iter:
            print(f"\n🏆 Keeping best iteration: iter_{best_iter}")
            for f in files:
                if f.name != f"preprocess_iter_{best_iter}.py":
                    f.unlink()
                    stats['files_deleted'] += 1
            print(f"✓ Deleted {stats['files_deleted']} other iterations")
        else:
            print("⚠️  Could not determine best iteration (no results found)")
    
    elif clear:
        print(f"\n🗑️  Clearing iteration files...")
        for f in files:
            f.unlink()
            stats['files_deleted'] += 1
        print(f"✓ Deleted {stats['files_deleted']} files")
    
    else:
        print(f"\n💡 Iteration files preserved (use --clear to delete)")
    
    return stats


def list_archives():
    """List all existing archives."""
    if not ARCHIVE_DIR.exists():
        print("No archives found")
        return
    
    archives = sorted(ARCHIVE_DIR.glob("*.zip"))
    if not archives:
        print("No archives found")
        return
    
    print(f"\n{'='*60}")
    print(f"EXISTING ARCHIVES ({len(archives)} total)")
    print(f"{'='*60}")
    
    # Group by agent
    by_agent = {}
    for arch in archives:
        # Extract agent name from filename (agent_iterations_timestamp.zip)
        parts = arch.stem.split('_iterations_')
        if len(parts) == 2:
            agent = parts[0]
            by_agent.setdefault(agent, []).append(arch)
    
    for agent, agent_archives in sorted(by_agent.items()):
        print(f"\n{agent}:")
        for arch in sorted(agent_archives)[-5:]:  # Show last 5
            size_kb = arch.stat().st_size / 1024
            date_str = arch.stem.split('_')[-2:]  # date and time
            print(f"  - {arch.name} ({size_kb:.1f} KB)")
    
    total_size = sum(a.stat().st_size for a in archives) / 1024 / 1024
    print(f"\nTotal archives: {len(archives)} ({total_size:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Archive and clean up agent iteration code files"
    )
    parser.add_argument("--agent", type=str, help="Agent name")
    parser.add_argument("--all-agents", action="store_true",
                       help="Process all agents")
    parser.add_argument("--clear", action="store_true",
                       help="Delete iteration files after archiving")
    parser.add_argument("--keep-best", action="store_true",
                       help="Keep only the best iteration (by nDCG)")
    parser.add_argument("--list", action="store_true",
                       help="List existing archives and exit")
    
    args = parser.parse_args()
    
    if args.list:
        list_archives()
        sys.exit(0)
    
    if not args.agent and not args.all_agents:
        print("Error: Must specify --agent or --all-agents")
        print("\nAvailable agents:")
        for agent in find_all_agents():
            print(f"  - {agent}")
        sys.exit(1)
    
    agents = []
    if args.all_agents:
        agents = find_all_agents()
        if not agents:
            print("No agents found")
            sys.exit(1)
    else:
        agents = [args.agent]
    
    all_stats = []
    for agent in agents:
        stats = archive_agent_iterations(agent, args.clear, args.keep_best)
        if stats['files_found'] > 0:
            all_stats.append(stats)
    
    # Summary
    if all_stats:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Agents processed: {len(all_stats)}")
        print(f"Total files archived: {sum(s['files_archived'] for s in all_stats)}")
        print(f"Total files deleted: {sum(s['files_deleted'] for s in all_stats)}")
        print(f"\nArchives saved to: {ARCHIVE_DIR}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()