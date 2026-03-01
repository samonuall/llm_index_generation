"""
Helper to load cached data for a specific split.

Usage in your evaluation scripts:

    from src.evaluation.scripts.load_split_data import load_split_data
    
    queries, docs = load_split_data("paper_retrieval")
    queries, docs = load_split_data("tip_of_the_tongue")
    queries, docs = load_split_data("legal_qa")
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

def load_split_data(split: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load cached queries and documents for a specific split.
    
    Args:
        split: Split name (e.g., "paper_retrieval", "tip_of_the_tongue")
    
    Returns:
        (queries, documents) tuple
    
    Raises:
        FileNotFoundError: If data hasn't been downloaded for this split yet
    """
    base_dir = Path(__file__).parents[2] / "data"
    split_dir = base_dir / split
    
    queries_file = split_dir / "queries.jsonl"
    docs_file = split_dir / "documents.jsonl"
    
    if not queries_file.exists():
        raise FileNotFoundError(
            f"Queries not found for split '{split}'. "
            f"Run: python -m src.evaluation.scripts.get_data_extended --split {split}"
        )
    
    if not docs_file.exists():
        raise FileNotFoundError(
            f"Documents not found for split '{split}'. "
            f"Run: python -m src.evaluation.scripts.get_data_extended --split {split}"
        )
    
    # Load queries
    queries = []
    with open(queries_file, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    
    # Load documents
    docs = []
    with open(docs_file, 'r') as f:
        for line in f:
            docs.append(json.loads(line))
    
    print(f"✓ Loaded {split}: {len(queries)} queries, {len(docs)} documents")
    
    return queries, docs

def list_available_splits() -> List[str]:
    """List all splits that have been cached."""
    base_dir = Path(__file__).parents[2] / "data"
    
    if not base_dir.exists():
        return []
    
    splits = []
    for split_dir in base_dir.iterdir():
        if split_dir.is_dir() and (split_dir / "queries.jsonl").exists():
            splits.append(split_dir.name)
    
    return sorted(splits)

if __name__ == "__main__":
    # Demo: list available splits
    splits = list_available_splits()
    
    if not splits:
        print("No cached splits found.")
        print("\nDownload data with:")
        print("  python -m src.evaluation.scripts.get_data_extended --split paper_retrieval")
    else:
        print(f"Available cached splits: {', '.join(splits)}\n")
        
        for split in splits:
            queries, docs = load_split_data(split)
            print(f"  {split}: {len(queries)} queries, {len(docs)} docs")