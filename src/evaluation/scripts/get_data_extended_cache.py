"""
Modified get_data_extended.py that caches data by split name.

Usage:
    python -m src.evaluation.scripts.get_data_extended --split paper_retrieval
    python -m src.evaluation.scripts.get_data_extended --split tip_of_the_tongue
    python -m src.evaluation.scripts.get_data_extended --split legal_qa --max-docs 2000

Data will be cached in:
    data/paper_retrieval/documents.jsonl
    data/paper_retrieval/queries.jsonl
    data/tip_of_the_tongue/documents.jsonl
    data/tip_of_the_tongue/queries.jsonl
    etc.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set
from datasets import load_dataset

# Map split names to CRUMB dataset names
SPLIT_MAP = {
    "tip_of_the_tongue": "tip_of_the_tongue",
    "paper_retrieval": "paper_retrieval",
    "stack_exchange": "stack_exchange",
    "clinical_trial": "clinical_trial",
    "legal_qa": "legal_qa",
    "theorem_retrieval": "theorem_retrieval",
    "code_retrieval": "code_retrieval",
    "set_operation_entity_retrieval": "set_operation_entity_retrieval",
}

def get_cache_dir(split: str) -> Path:
    """Get cache directory for this split."""
    base_dir = Path(__file__).parents[3] / "data"
    cache_dir = base_dir / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def load_queries(split: str, n_queries: int = None):
    """Load queries for a split."""
    crumb_name = SPLIT_MAP[split]
    cache_dir = get_cache_dir(split)
    cache_file = cache_dir / "queries.jsonl"
    
    # Check cache first
    if cache_file.exists():
        print(f"✓ Loading cached queries from {cache_file}")
        queries = []
        with open(cache_file, 'r') as f:
            for line in f:
                queries.append(json.loads(line))
        print(f"  Loaded {len(queries)} cached queries")
        return queries
    
    # Download from HuggingFace
    print(f"Downloading queries for {split} from CRUMB...")
    dataset = load_dataset("jfkback/crumb", "evaluation_queries", split=crumb_name)
    
    queries = []
    for item in dataset:
        # Get relevant doc IDs
        qrels = item.get('passage_qrels') or item.get('full_document_qrels') or []
        relevant_ids = [qrel['id'] for qrel in qrels if qrel.get('label', 0) > 0]
        
        if relevant_ids:
            queries.append({
                'query_id': item['query_id'],
                'query_content': item['query_content'],
                'relevant_doc_ids': relevant_ids
            })
    
    # Limit if requested
    if n_queries and n_queries < len(queries):
        queries = queries[:n_queries]
    
    # Cache queries
    print(f"Caching {len(queries)} queries to {cache_file}")
    with open(cache_file, 'w') as f:
        for q in queries:
            f.write(json.dumps(q) + '\n')
    
    return queries

def load_documents(split: str, queries: List[Dict], max_docs: int = None):
    """Load documents for a split."""
    crumb_name = SPLIT_MAP[split]
    cache_dir = get_cache_dir(split)
    cache_file = cache_dir / "documents.jsonl"
    
    # Check cache first
    if cache_file.exists() and not max_docs:
        print(f"✓ Loading cached documents from {cache_file}")
        docs = []
        with open(cache_file, 'r') as f:
            for line in f:
                docs.append(json.loads(line))
        print(f"  Loaded {len(docs)} cached documents")
        return docs
    
    # Get relevant doc IDs from queries
    relevant_ids = set()
    for q in queries:
        relevant_ids.update(q['relevant_doc_ids'])
    
    print(f"Need {len(relevant_ids)} relevant documents")
    
    # Download corpus
    print(f"Downloading corpus for {split} from CRUMB...")
    corpus_dataset = load_dataset("jfkback/crumb", "passage_corpus", split=crumb_name, streaming=True)
    
    docs = []
    doc_ids_seen = set()
    
    # First pass: collect all relevant docs
    print("Collecting relevant documents...")
    for item in corpus_dataset:
        doc_id = item['document_id']
        if doc_id in relevant_ids:
            docs.append({
                'doc_id': doc_id,
                'text': item['document_content'],
                'metadata': {}
            })
            doc_ids_seen.add(doc_id)
            
            if len(doc_ids_seen) >= len(relevant_ids):
                break
    
    print(f"Collected {len(docs)} relevant documents")
    
    # If max_docs specified and we need more, add random extras
    if max_docs and len(docs) < max_docs:
        print(f"Adding random documents to reach {max_docs}...")
        extra_needed = max_docs - len(docs)
        
        # Reservoir sampling for extras
        corpus_dataset = load_dataset("jfkback/crumb", "passage_corpus", split=crumb_name, streaming=True)
        reservoir = []
        count = 0
        
        for item in corpus_dataset:
            doc_id = item['document_id']
            if doc_id not in doc_ids_seen:
                count += 1
                if len(reservoir) < extra_needed:
                    reservoir.append({
                        'doc_id': doc_id,
                        'text': item['document_content'],
                        'metadata': {}
                    })
                else:
                    # Reservoir sampling: replace with decreasing probability
                    j = random.randint(0, count - 1)
                    if j < extra_needed:
                        reservoir[j] = {
                            'doc_id': doc_id,
                            'text': item['document_content'],
                            'metadata': {}
                        }
        
        docs.extend(reservoir)
        print(f"Added {len(reservoir)} random documents")
    
    # Cache documents
    print(f"Caching {len(docs)} documents to {cache_file}")
    with open(cache_file, 'w') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')
    
    return docs

def main():
    parser = argparse.ArgumentParser(description="Download CRUMB data with split-specific caching")
    parser.add_argument(
        "--split",
        type=str,
        default="tip_of_the_tongue",
        choices=list(SPLIT_MAP.keys()),
        help="Which CRUMB split to download"
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=None,
        help="Limit number of queries (default: all)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to include (adds random extras if needed)"
    )
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="Only re-download documents (use existing queries)"
    )
    
    args = parser.parse_args()
    
    cache_dir = get_cache_dir(args.split)
    print(f"\n{'='*60}")
    print(f"CRUMB Data Download - Split: {args.split}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*60}\n")
    
    # Load queries
    if not args.docs_only:
        queries = load_queries(args.split, args.n_queries)
    else:
        # Load from cache
        cache_file = cache_dir / "queries.jsonl"
        if not cache_file.exists():
            print(f"ERROR: No cached queries found at {cache_file}")
            print("Run without --docs-only first to download queries.")
            return
        queries = []
        with open(cache_file, 'r') as f:
            for line in f:
                queries.append(json.loads(line))
        print(f"Using {len(queries)} cached queries")
    
    # Load documents
    docs = load_documents(args.split, queries, args.max_docs)
    
    print(f"\n{'='*60}")
    print(f"✓ Download complete for {args.split}")
    print(f"  Queries: {len(queries)}")
    print(f"  Documents: {len(docs)}")
    print(f"  Location: {cache_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()