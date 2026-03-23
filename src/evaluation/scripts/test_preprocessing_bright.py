"""
test_preprocessing_bright.py – BRIGHT evaluation harness.

Mirrors test_preprocessing.py but targets the BRIGHT benchmark:
  - Loads from data/bright/{task}/
  - Primary metric: nDCG@10 (matches the BRIGHT paper's reported numbers)
  - Also reports Recall@k and MRR for reference
  - Handles excluded_ids: per-query docs that must be filtered from results
    before scoring (they appeared in the question itself)
  - Diagnostic fields returned per-query (for agent feedback):
      rank_of_relevant  – rank of highest relevant doc in extended top-200 search
      score_of_relevant – BM25 score of that doc
      score_of_rank1    – BM25 score of rank-1 retrieved doc
      terms_present     – query content-words found in relevant doc chunks
      terms_missing     – query content-words NOT found in relevant doc chunks
      rank1_snippet     – first 200 chars of rank-1 doc when it is wrong

CLI usage:
    uv run python src/evaluation/scripts/test_preprocessing_bright.py --agent <name>
    uv run python src/evaluation/scripts/test_preprocessing_bright.py --agent <name> --task sustainable_living --top-k 10

Programmatic usage (e.g. from an agent):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
    from test_preprocessing_bright import evaluate_bright
    results = evaluate_bright(Preprocessor(), task="sustainable_living", top_k=10)

Paper BM25 baseline (no preprocessing): nDCG@10 ≈ 14.8 (average across all tasks).
"""

from __future__ import annotations

import sys
import re as _re
import pathlib
import json
import argparse
import importlib
import math
from typing import List

_EVAL_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_EVAL_DIR / "scripts"))

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from schema import Document
from base import BasePreprocessor
from build_index import BM25Index

import Stemmer as _StemmerLib

DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_TASK = "sustainable_living"

# ---------------------------------------------------------------------------
# Term-overlap helpers (uses same Snowball stemmer as BM25Index)
# ---------------------------------------------------------------------------

_STEMMER_OVERLAP = _StemmerLib.Stemmer("english")

_STOPWORDS_OVERLAP = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "not", "no", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
    "our", "their", "so", "if", "then", "than", "when", "where", "which", "who",
    "what", "how", "all", "also", "just", "more", "about", "up", "out", "into",
    "there", "s", "t", "re", "m", "d", "ve", "ll", "use", "used", "using",
    "one", "two", "get", "got", "can", "use",
}


def _tokenize_overlap(text: str) -> dict[str, str]:
    """Return {stem: representative_word} for content words. Word length > 2."""
    words = _re.findall(r"\b[a-z]+\b", text.lower())
    result: dict[str, str] = {}
    for w in words:
        if w not in _STOPWORDS_OVERLAP and len(w) > 2:
            stem = _STEMMER_OVERLAP.stemWord(w)
            if stem not in result:
                result[stem] = w
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_documents(task: str) -> List[Document]:
    path = DATA_DIR / "bright" / task / "documents.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"BRIGHT documents not found at {path}. "
            f"Run: uv run python src/evaluation/scripts/get_data_bright.py --task {task}"
        )
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(Document(**d))
    return docs


def _load_queries(task: str) -> List[dict]:
    """Load query records as plain dicts (includes excluded_ids field)."""
    path = DATA_DIR / "bright" / task / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"BRIGHT queries not found at {path}. "
            f"Run: uv run python src/evaluation/scripts/get_data_bright.py --task {task}"
        )
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ndcg_at_k(retrieved_doc_ids: List[str], relevant: set, k: int) -> float:
    """Compute nDCG@k with binary relevance (0/1 gains)."""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bright(
    preprocessor: BasePreprocessor,
    task: str = DEFAULT_TASK,
    top_k: int = 10,
) -> dict:
    """
    Run the full BRIGHT eval pipeline for a given preprocessor.

    Returns a dict with keys:
      agent, task, ndcg_at_k, recall_at_k, mrr, top_k, n_queries, n_chunks,
      query_results (list of per-query dicts)

    Each per-query dict includes diagnostic fields for agent feedback:
      rank_of_relevant, score_of_relevant, score_of_rank1,
      terms_present, terms_missing, rank1_snippet
    """
    docs = _load_documents(task)
    query_records = _load_queries(task)

    label = preprocessor.name or type(preprocessor).__name__
    print(f"\n{'='*60}")
    print(f"Agent       : {label}")
    print(f"Task        : {task}")
    print(f"Description : {preprocessor.description or '(none)'}")
    print(f"{'='*60}")
    print(f"Preprocessing {len(docs)} documents ...")

    chunks = preprocessor.preprocess(docs)
    print(f"  -> {len(chunks)} chunks  ({len(chunks) / len(docs):.2f} avg per doc)")

    print("Building BM25 index ...")
    index = BM25Index(chunks)

    # Build doc_id -> chunks map for term overlap computation
    doc_to_chunks: dict[str, list] = {}
    for c in chunks:
        doc_to_chunks.setdefault(c.doc_id, []).append(c)

    ndcg_total = 0.0
    recall_hits = 0
    mrr_total = 0.0
    query_results = []

    # Extended search budget for rank diagnostics (200 extra beyond scoring window)
    diag_extra = 200

    for q in query_records:
        excluded = set(q.get("excluded_ids", []))
        relevant = set(q["relevant_doc_ids"])

        # Single extended search covering both scoring and rank diagnostics
        search_k = min(top_k + len(excluded) + diag_extra, len(chunks))
        raw_results = index.search(q["query_text"], top_k=search_k)

        # Filter excluded docs
        filtered_all = [
            (chunk, score)
            for chunk, score in raw_results
            if chunk.doc_id not in excluded
        ]

        # Top-k for actual scoring
        filtered = filtered_all[:top_k]
        retrieved_doc_ids = [chunk.doc_id for chunk, _ in filtered]

        # nDCG@k
        ndcg = _ndcg_at_k(retrieved_doc_ids, relevant, top_k)
        ndcg_total += ndcg

        # Recall@k
        hit = any(doc_id in relevant for doc_id in retrieved_doc_ids)
        if hit:
            recall_hits += 1

        # MRR
        reciprocal_rank = 0.0
        rank_of_first_hit = None
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                rank_of_first_hit = rank
                mrr_total += reciprocal_rank
                break

        # --- Diagnostics ---

        # Rank of highest relevant doc in the extended filtered list
        rank_of_relevant = None
        score_of_relevant = 0.0
        for rank, (chunk, score) in enumerate(filtered_all, start=1):
            if chunk.doc_id in relevant:
                rank_of_relevant = rank
                score_of_relevant = score
                break

        # Rank-1 doc info
        score_of_rank1 = filtered_all[0][1] if filtered_all else 0.0
        rank1_doc_id = filtered_all[0][0].doc_id if filtered_all else None
        rank1_is_wrong = (rank1_doc_id not in relevant) if rank1_doc_id else True
        rank1_snippet = ""
        if rank1_is_wrong and filtered_all:
            rank1_snippet = filtered_all[0][0].text[:200].replace("\n", " ").strip()

        # Term overlap: which stemmed query words appear/don't appear in relevant chunks
        query_stems = _tokenize_overlap(q["query_text"])  # {stem: word}
        rel_texts = " ".join(
            c.text for rel_id in relevant for c in doc_to_chunks.get(rel_id, [])
        )
        doc_stems = set(_tokenize_overlap(rel_texts).keys())
        terms_present = sorted(query_stems[s] for s in query_stems if s in doc_stems)
        terms_missing = sorted(query_stems[s] for s in query_stems if s not in doc_stems)

        query_results.append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "relevant_doc_ids": list(relevant),
            "excluded_ids": list(excluded),
            "retrieved_doc_ids": retrieved_doc_ids,
            "ndcg": ndcg,
            "hit": hit,
            "rank": rank_of_first_hit,
            "reciprocal_rank": reciprocal_rank,
            # Diagnostic fields
            "rank_of_relevant": rank_of_relevant,
            "score_of_relevant": round(score_of_relevant, 4),
            "score_of_rank1": round(score_of_rank1, 4),
            "terms_present": terms_present,
            "terms_missing": terms_missing,
            "rank1_snippet": rank1_snippet,
        })

    n = len(query_records)
    ndcg_at_k = ndcg_total / n
    recall_at_k = recall_hits / n
    mrr = mrr_total / n

    print(f"\nResults  ({n} queries, top-{top_k}, task={task}):")
    print(f"  nDCG@{top_k:<3}  : {ndcg_at_k:.4f}   <- compare against paper BM25 baseline ~14.8")
    print(f"  Recall@{top_k:<3}: {recall_at_k:.4f}")
    print(f"  MRR       : {mrr:.4f}")
    print(f"{'='*60}\n")

    return {
        "agent": label,
        "task": task,
        "ndcg_at_k": ndcg_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "top_k": top_k,
        "n_queries": n,
        "n_chunks": len(chunks),
        "query_results": query_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent preprocessor against the BRIGHT BM25 harness."
    )
    parser.add_argument("--agent", required=True, help="Agent folder name under agents/")
    parser.add_argument("--task", default=DEFAULT_TASK, help="BRIGHT task (default: sustainable_living)")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    module_path = f"agents.{args.agent}.preprocess"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        print(f"Error: could not import '{module_path}': {e}")
        sys.exit(1)

    if not hasattr(module, "Preprocessor"):
        print(f"Error: {module_path} must define a class named 'Preprocessor'.")
        sys.exit(1)

    preprocessor = module.Preprocessor()

    if not isinstance(preprocessor, BasePreprocessor):
        print(f"Error: {module_path}.Preprocessor must inherit from BasePreprocessor.")
        sys.exit(1)

    evaluate_bright(preprocessor, task=args.task, top_k=args.top_k)


if __name__ == "__main__":
    main()
