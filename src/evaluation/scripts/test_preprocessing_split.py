# """
# test_preprocessing_split.py – Evaluation harness with auto-fallback, results caching, and document-level metrics.

# CLI usage:
#     uv run python src/evaluation/scripts/test_preprocessing_split.py --agent baseline --split paper_retrieval
#     uv run python src/evaluation/scripts/test_preprocessing_split.py --agent lite_llm_agent --split tip_of_the_tongue --top-k 20

# Results are automatically saved to: results/<split>/<agent>_k<top_k>.json
# """

# from __future__ import annotations

# import sys
# import math
# import pathlib
# import json
# import argparse
# import importlib
# from typing import List, Dict

# # Make src/evaluation/ importable
# _EVAL_DIR = pathlib.Path(__file__).parent.parent
# sys.path.insert(0, str(_EVAL_DIR))
# sys.path.insert(0, str(_EVAL_DIR / "scripts"))

# # Make src/ importable so `agents.<name>.preprocess` resolves
# _PROJECT_ROOT = pathlib.Path(__file__).parents[3]
# sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# from schema import Document, EvalQuery, Chunk
# from base import BasePreprocessor
# from build_index import BM25Index

# DATA_DIR = _PROJECT_ROOT / "data"
# RESULTS_DIR = _PROJECT_ROOT / "results"


# # ---------------------------------------------------------------------------
# # Data loading
# # ---------------------------------------------------------------------------

# def _load_documents(split: str = None) -> tuple[List[Document], str]:
#     """Load documents from split-specific cache, or fallback to default."""
#     if split:
#         docs_file = DATA_DIR / split / "documents.jsonl"
#         if docs_file.exists():
#             print(f"✓ Using cached data for split: {split}")
#             docs = [Document(**json.loads(line)) for line in docs_file.open(encoding="utf-8")]
#             return docs, split

#     default_docs = DATA_DIR / "documents.jsonl"
#     if default_docs.exists():
#         print(f"⚠️  Split '{split}' not cached, using default data/")
#         docs = [Document(**json.loads(line)) for line in default_docs.open(encoding="utf-8")]
#         return docs, "default"

#     raise FileNotFoundError(
#         f"No data found! Run get_data_extended_cache or get_data for default data."
#     )


# def _load_queries(split: str = None) -> tuple[List[EvalQuery], str]:
#     """Load queries from split-specific cache, or fallback to default."""
#     if split:
#         queries_file = DATA_DIR / split / "queries.jsonl"
#         if queries_file.exists():
#             queries = []
#             for line in queries_file.open(encoding="utf-8"):
#                 q = json.loads(line)
#                 queries.append(EvalQuery(
#                     query_id=q['query_id'],
#                     query_text=q['query_content'],
#                     relevant_doc_ids=q['relevant_doc_ids']
#                 ))
#             return queries, split

#     default_queries = DATA_DIR / "queries.jsonl"
#     if default_queries.exists():
#         queries = [EvalQuery(**json.loads(line)) for line in default_queries.open(encoding="utf-8")]
#         return queries, "default"

#     raise FileNotFoundError(
#         f"No queries found! Run get_data_extended_cache for split {split}."
#     )


# def _list_available_splits() -> List[str]:
#     """List all splits that have been cached."""
#     if not DATA_DIR.exists():
#         return []
#     return sorted(
#         d.name for d in DATA_DIR.iterdir()
#         if d.is_dir() and (d / "queries.jsonl").exists()
#     )


# # ---------------------------------------------------------------------------
# # Results saving
# # ---------------------------------------------------------------------------

# def _results_path(split: str, agent: str, top_k: int) -> pathlib.Path:
#     return RESULTS_DIR / split / f"{agent}_k{top_k}.json"


# def _save_results(results: dict, split: str, agent: str, top_k: int) -> pathlib.Path:
#     path = _results_path(split, agent, top_k)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open('w', encoding='utf-8') as f:
#         json.dump(results, f, indent=2)
#     print(f"\n✓ Results saved to: {path}")
#     return path


# def _load_all_results(split: str) -> List[dict]:
#     split_results_dir = RESULTS_DIR / split
#     if not split_results_dir.exists():
#         return []
#     return [json.load(f.open('r', encoding='utf-8')) for f in split_results_dir.glob("*.json")]


# # ---------------------------------------------------------------------------
# # Evaluation
# # ---------------------------------------------------------------------------

# def evaluate(
#     preprocessor: BasePreprocessor,
#     split: str,
#     top_k: int = 100,
#     ndcg_k: int = 10,
#     save_results: bool = True,
# ) -> dict:
#     """Run the full eval pipeline with realistic document-level metrics."""
#     docs, actual_split = _load_documents(split)
#     queries, _ = _load_queries(split)

#     label = preprocessor.name or type(preprocessor).__name__
#     print(f"\n{'='*60}")
#     print(f"Agent       : {label}")
#     print(f"Split       : {actual_split}")
#     print(f"Description : {preprocessor.description or '(none)'}")
#     print(f"{'='*60}")
#     print(f"Preprocessing {len(docs)} documents ...")

#     # Run preprocessor -> get chunks
#     chunks = preprocessor.preprocess(docs)
#     print(f"  -> {len(chunks)} chunks  ({len(chunks)/len(docs):.2f} avg per doc)")

#     # Only augment if a document is really short
#     MIN_CHUNK_SIZE = 10  # threshold for splitting
#     chunk_counter = 0
#     augmented_chunks: List[Chunk] = []

#     for doc in docs:
#         text = doc.text
#         if len(text) <= MIN_CHUNK_SIZE:
#             # Short docs → single chunk
#             augmented_chunks.append(Chunk(doc_id=doc.doc_id, chunk_id=f"{doc.doc_id}_0", text=text))
#         else:
#             # Long docs → split into chunks of ~MIN_CHUNK_SIZE
#             for i in range(0, len(text), MIN_CHUNK_SIZE):
#                 augmented_chunks.append(
#                     Chunk(doc_id=doc.doc_id, chunk_id=f"{doc.doc_id}_{chunk_counter}", text=text[i:i+MIN_CHUNK_SIZE])
#                 )
#                 chunk_counter += 1

#     chunks = augmented_chunks
#     print(f"  -> Processed into {len(chunks)} chunks (~{len(chunks)/len(docs):.2f} avg per doc)")

#     print("Building BM25 index ...")
#     index = BM25Index(chunks)

#     # Compute metrics at document level
#     recall_hits = 0
#     ndcg_total = 0.0

#     for query in queries:
#         chunk_results = index.search(query.query_text, top_k=top_k)
#         retrieved_docs: Dict[str, float] = {}
#         for chunk, score in chunk_results:
#             if chunk.doc_id not in retrieved_docs or score > retrieved_docs[chunk.doc_id]:
#                 retrieved_docs[chunk.doc_id] = score

#         ranked_doc_ids = [doc_id for doc_id, _ in sorted(retrieved_docs.items(), key=lambda x: x[1], reverse=True)][:top_k]

#         relevant = set(query.relevant_doc_ids)

#         # Recall@k
#         if any(doc_id in relevant for doc_id in ranked_doc_ids):
#             recall_hits += 1

#         # nDCG@k
#         dcg = 0.0
#         for rank, doc_id in enumerate(ranked_doc_ids[:ndcg_k], start=1):
#             if doc_id in relevant:
#                 dcg += 1.0 / math.log2(rank + 1)
#         idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), ndcg_k) + 1))
#         ndcg_total += (dcg / idcg) if idcg > 0 else 0.0

#     n = len(queries)
#     recall_at_k = recall_hits / n
#     ndcg = ndcg_total / n

#     print(f"\nResults  ({n} queries, top-{top_k}):")
#     print(f"  Recall@{top_k:<3} : {recall_at_k:.4f}")
#     print(f"  nDCG@{ndcg_k:<5} : {ndcg:.4f}")
#     print(f"{'='*60}\n")

#     results = {
#         "agent": label,
#         "split": actual_split,
#         "config": {
#             "top_k": top_k,
#             "ndcg_k": ndcg_k,
#             "n_docs": len(docs),
#             "n_queries": n,
#             "n_chunks": len(chunks),
#             "chunks_per_doc": len(chunks)/len(docs),
#         },
#         "metrics": {
#             "recall_at_k": recall_at_k,
#             "ndcg": ndcg,
#         },
#         "summary": {
#             "recall_at_k": recall_at_k,
#             "ndcg": ndcg,
#         },
#     }

#     if save_results:
#         _save_results(results, actual_split, label, top_k)

#     return results


# # ---------------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------------

# def main() -> None:
#     parser = argparse.ArgumentParser(description="Evaluate an agent preprocessor on a CRUMB split.")
#     parser.add_argument("--agent", required=True, help="Agent folder name under agents/")
#     parser.add_argument("--split", type=str, help="CRUMB split name (e.g., 'paper_retrieval').")
#     parser.add_argument("--top-k", type=int, default=100)
#     parser.add_argument("--no-save", action="store_true")
#     parser.add_argument("--compare", action="store_true")

#     args = parser.parse_args()

#     if not args.split:
#         available = _list_available_splits()
#         if not available:
#             print("No cached splits found.\nDownload data with: python -m src.evaluation.scripts.get_data_extended_cache --split paper_retrieval")
#         else:
#             print("Available cached splits:")
#             for split in available:
#                 split_dir = DATA_DIR / split
#                 n_docs = sum(1 for _ in (split_dir / "documents.jsonl").open())
#                 n_queries = sum(1 for _ in (split_dir / "queries.jsonl").open())
#                 print(f"  - {split}: {n_queries} queries, {n_docs} docs")
#         sys.exit(0)

#     module_path = f"agents.{args.agent}.preprocess"
#     try:
#         module = importlib.import_module(module_path)
#     except ModuleNotFoundError as e:
#         print(f"Error: could not import '{module_path}': {e}")
#         sys.exit(1)

#     if not hasattr(module, "Preprocessor"):
#         print(f"Error: {module_path} must define a class named 'Preprocessor'.")
#         sys.exit(1)

#     preprocessor = module.Preprocessor()
#     if not isinstance(preprocessor, BasePreprocessor):
#         print(f"Error: {module_path}.Preprocessor must inherit from BasePreprocessor.")
#         sys.exit(1)

#     current_results = evaluate(
#         preprocessor,
#         split=args.split,
#         top_k=args.top_k,
#         save_results=not args.no_save,
#     )

#     if args.compare:
#         print("\n" + "="*60)
#         print("COMPARISON WITH ALL SAVED RUNS")
#         print("="*60)
#         all_results = _load_all_results(args.split)
#         if len(all_results) <= 1:
#             print("No previous results to compare with.")
#         else:
#             all_results.sort(key=lambda x: (x['agent'], x['config']['top_k']))
#             print(f"\n{'Agent':<20} {'Top-K':<8} {'Docs':<8} {'Recall':<10} {'nDCG@10':<10} {'Chunks/Doc'}")
#             print("-"*70)
#             for r in all_results:
#                 cfg = r['config']
#                 m = r['metrics']
#                 print(
#                     f"{r['agent']:<20} {cfg['top_k']:<8} {cfg['n_docs']:<8} "
#                     f"{m['recall_at_k']:<10.4f} {m['ndcg']:<10.4f} {cfg['chunks_per_doc']:.2f}"
#                 )
#             print("="*60)


# if __name__ == "__main__":
#     main()

"""
test_preprocessing_split.py – Evaluation harness using official CRUMB eval library.

CLI usage:
    uv run python -m src.evaluation.scripts.test_preprocessing_split --agent baseline --split paper_retrieval
    uv run python -m src.evaluation.scripts.test_preprocessing_split --agent lite_llm_agent --split tip_of_the_tongue --compare
    
Results saved to:
- results/<split>/<agent>_<n_docs>docs_k<top_k>.json (config + metrics)
- results/<split>/<agent>_<n_docs>docs_k<top_k>_crumb.jsonl (CRUMB format for their eval)
"""

from __future__ import annotations

import sys
import pathlib
import json
import argparse
import importlib
from typing import List, Dict
from datetime import datetime

# Make src/evaluation/ importable
_EVAL_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_EVAL_DIR / "scripts"))

# Make src/ importable
_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from schema import Document, EvalQuery
from base import BasePreprocessor
from build_index import BM25Index

DATA_DIR = _PROJECT_ROOT / "data"
RESULTS_DIR = _PROJECT_ROOT / "results"

# Try to import CRUMB eval (optional but recommended)
try:
    from crumb_eval import evaluate as crumb_evaluate
    CRUMB_EVAL_AVAILABLE = True
except ImportError:
    CRUMB_EVAL_AVAILABLE = False
    print("\n⚠️  CRUMB eval not installed. Install with:")
    print("    pip install git+https://github.com/jfkback/crumb#subdirectory=crumb_eval\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_documents(split: str = None) -> tuple[List[Document], str]:
    """Load documents from split-specific cache, or fallback to default."""
    if split:
        docs_file = DATA_DIR / split / "documents.jsonl"
        if docs_file.exists():
            print(f"✓ Using cached data for split: {split}")
            docs = []
            with docs_file.open(encoding="utf-8") as f:
                for line in f:
                    docs.append(Document(**json.loads(line)))
            return docs, split

    default_docs = DATA_DIR / "documents.jsonl"
    if default_docs.exists():
        print(f"⚠️  Split '{split}' not cached, using default data/")
        docs = []
        with default_docs.open(encoding="utf-8") as f:
            for line in f:
                docs.append(Document(**json.loads(line)))
        return docs, "default"

    raise FileNotFoundError(
        f"No data found! Run: python -m src.evaluation.scripts.get_data_extended_cache --split {split}"
    )


def _load_queries(split: str = None) -> tuple[List[EvalQuery], str]:
    """Load queries from split-specific cache, or fallback to default."""
    if split:
        queries_file = DATA_DIR / split / "queries.jsonl"
        if queries_file.exists():
            queries = []
            with queries_file.open(encoding="utf-8") as f:
                for line in f:
                    q = json.loads(line)
                    queries.append(EvalQuery(
                        query_id=q['query_id'],
                        query_text=q['query_content'],
                        relevant_doc_ids=q['relevant_doc_ids']
                    ))
            return queries, split

    default_queries = DATA_DIR / "queries.jsonl"
    if default_queries.exists():
        queries = []
        with default_queries.open(encoding="utf-8") as f:
            for line in f:
                queries.append(EvalQuery(**json.loads(line)))
        return queries, "default"

    raise FileNotFoundError(f"No queries found!")


def _list_available_splits() -> List[str]:
    """List all cached splits."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and (d / "queries.jsonl").exists()
    )


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def _save_results(results: dict, split: str, agent: str, n_docs: int, top_k: int) -> pathlib.Path:
    """Save results with full config in filename."""
    results_dir = RESULTS_DIR / split
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{agent}_{n_docs}docs_k{top_k}.json"
    path = results_dir / filename
    
    with path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {path}")
    return path


def _save_crumb_format(
    query_results: List[Dict], 
    split: str, 
    agent: str, 
    n_docs: int, 
    top_k: int
) -> pathlib.Path:
    """
    Save in CRUMB eval format (JSONL).
    
    Format per line:
    {
        "query": {"id": "query_id"},
        "items": [{"id": "doc_id", "score": float}, ...]
    }
    """
    results_dir = RESULTS_DIR / split
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{agent}_{n_docs}docs_k{top_k}_crumb.jsonl"
    path = results_dir / filename
    
    with path.open('w', encoding='utf-8') as f:
        for qr in query_results:
            entry = {
                "query": {"id": qr["query_id"]},
                "items": [
                    {"id": doc_id, "score": score}
                    for doc_id, score in qr["ranked_docs"]
                ]
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"✓ CRUMB format saved to: {path}")
    return path


def _load_all_results(split: str) -> List[dict]:
    """Load all saved results for comparison."""
    results_dir = RESULTS_DIR / split
    if not results_dir.exists():
        return []
    
    results = []
    for f in results_dir.glob("*.json"):
        # Skip crumb format files
        if "_crumb" not in f.name:
            with f.open('r', encoding='utf-8') as fh:
                results.append(json.load(fh))
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    preprocessor: BasePreprocessor,
    split: str,
    top_k: int = 100,
    save_results: bool = True,
) -> dict:
    """
    Run evaluation and use CRUMB eval library for official metrics.
    
    This function:
    1. Runs preprocessing
    2. Builds BM25 index
    3. Retrieves results for each query
    4. Saves in CRUMB format
    5. Calls official CRUMB eval library
    """
    docs, actual_split = _load_documents(split)
    queries, _ = _load_queries(split)

    agent_name = preprocessor.name or type(preprocessor).__name__
    
    print(f"\n{'='*60}")
    print(f"Agent       : {agent_name}")
    print(f"Split       : {actual_split}")
    print(f"Description : {preprocessor.description or '(none)'}")
    print(f"{'='*60}")
    print(f"Preprocessing {len(docs)} documents ...")

    # Preprocess documents into chunks
    chunks = preprocessor.preprocess(docs)
    print(f"  -> {len(chunks)} chunks  ({len(chunks)/len(docs):.2f} avg per doc)")

    print("Building BM25 index ...")
    index = BM25Index(chunks)

    # Retrieve for each query
    query_results = []
    
    for query in queries:
        # Get chunk-level results
        chunk_results = index.search(query.query_text, top_k=top_k * 10)
        
        # Aggregate to document level using MaxP (max score per doc)
        doc_scores: Dict[str, float] = {}
        for chunk, score in chunk_results:
            if chunk.doc_id not in doc_scores or score > doc_scores[chunk.doc_id]:
                doc_scores[chunk.doc_id] = score
        
        # Sort and take top-k
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        query_results.append({
            "query_id": query.query_id,
            "ranked_docs": ranked_docs,  # List of (doc_id, score) tuples
        })

    # Build results structure
    results = {
        "agent": agent_name,
        "split": actual_split,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": top_k,
            "n_docs": len(docs),
            "n_queries": len(queries),
            "n_chunks": len(chunks),
            "chunks_per_doc": len(chunks) / len(docs),
        },
        "crumb_metrics": None,  # Will be populated if CRUMB eval runs
    }

    if save_results:
        # Save our format
        _save_results(results, actual_split, agent_name, len(docs), top_k)
        
        # Save CRUMB format
        crumb_path = _save_crumb_format(query_results, actual_split, agent_name, len(docs), top_k)
        
        # Run CRUMB eval if available
        if CRUMB_EVAL_AVAILABLE:
            print("\n" + "="*60)
            print("RUNNING CRUMB EVAL (Official Metrics)")
            print("="*60)
            try:
                # Call official CRUMB eval
                # It prints metrics directly and may return them
                crumb_result = crumb_evaluate(
                    run_path=str(crumb_path),
                    task_name=actual_split,
                    max_p="auto",  # Handles MaxP automatically per task
                )
                
                # Try to capture metrics if returned
                if crumb_result and isinstance(crumb_result, dict):
                    results["crumb_metrics"] = crumb_result
                    # Re-save with CRUMB metrics
                    _save_results(results, actual_split, agent_name, len(docs), top_k)
                else:
                    print("\n✓ CRUMB eval completed (metrics printed above)")
                
            except Exception as e:
                print(f"\n⚠️  CRUMB eval error: {e}")
                import traceback
                traceback.print_exc()
            
            print("="*60)
        else:
            print("\n⚠️  CRUMB eval not available. Skipping official metrics.")
            print("    Install: pip install git+https://github.com/jfkback/crumb#subdirectory=crumb_eval")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate agent with official CRUMB metrics."
    )
    parser.add_argument(
        "--agent", 
        required=True, 
        help="Agent folder name (e.g., 'baseline', 'lite_llm_agent')"
    )
    parser.add_argument(
        "--split",
        type=str,
        help="CRUMB split name (e.g., 'paper_retrieval', 'tip_of_the_tongue')"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=100,
        help="Number of documents to retrieve (default: 100)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison with all previous runs"
    )

    args = parser.parse_args()

    # If no split, show available
    if not args.split:
        available = _list_available_splits()
        if not available:
            print("No cached splits found.")
            print("\nDownload with:")
            print("  python -m src.evaluation.scripts.get_data_extended_cache --split paper_retrieval")
        else:
            print("Available splits:")
            for split in available:
                split_dir = DATA_DIR / split
                n_docs = sum(1 for _ in (split_dir / "documents.jsonl").open())
                n_queries = sum(1 for _ in (split_dir / "queries.jsonl").open())
                print(f"  - {split}: {n_queries} queries, {n_docs} docs")
            print(f"\nUsage:")
            print(f"  python -m src.evaluation.scripts.test_preprocessing_split --agent baseline --split <split>")
        sys.exit(0)

    # Import agent
    module_path = f"agents.{args.agent}.preprocess"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        print(f"Error: Could not import '{module_path}': {e}")
        sys.exit(1)

    if not hasattr(module, "Preprocessor"):
        print(f"Error: {module_path} must define 'Preprocessor' class")
        sys.exit(1)

    preprocessor = module.Preprocessor()
    if not isinstance(preprocessor, BasePreprocessor):
        print("Error: Preprocessor must inherit from BasePreprocessor")
        sys.exit(1)

    # Run evaluation
    evaluate(
        preprocessor,
        split=args.split,
        top_k=args.top_k,
        save_results=not args.no_save,
    )

    # Show comparison if requested
    if args.compare:
        print("\n" + "="*70)
        print("COMPARISON WITH ALL SAVED RUNS")
        print("="*70)
        
        all_results = _load_all_results(args.split)
        if len(all_results) == 0:
            print("No saved results found.")
        else:
            # Sort by agent name, then doc count, then top-k
            all_results.sort(key=lambda x: (
                x['agent'], 
                x['config']['n_docs'], 
                x['config']['top_k']
            ))
            
            print(f"\n{'Agent':<20} {'Docs':<8} {'Top-K':<8} {'Chunks/Doc':<12} {'Timestamp'}")
            print("-"*70)
            
            for r in all_results:
                cfg = r['config']
                timestamp = r.get('timestamp', 'unknown')[:19]
                
                print(
                    f"{r['agent']:<20} {cfg['n_docs']:<8} {cfg['top_k']:<8} "
                    f"{cfg['chunks_per_doc']:<12.2f} {timestamp}"
                )
            
            print("\n💡 Tip: CRUMB metrics are printed during evaluation")
            print("   Check the _crumb.jsonl files for official eval format")
            print("="*70)


if __name__ == "__main__":
    main()