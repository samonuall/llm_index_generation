"""
eval_utils.py - Partial evaluation utilities using the BM25 HTTP client.

Provides subset evaluation (run_subset_eval) and dynamic preprocessor loading
(load_preprocessor_from_code) for hypothesis testing without filesystem writes.
"""

from __future__ import annotations

import math
import sys
import types
import pathlib
from dataclasses import dataclass, field


@dataclass
class SubsetEvalResult:
    query_id: str
    hit_at_10: bool
    hit_at_100: bool
    rank: int | None  # 1-indexed, None if not in top_k
    ndcg_at_10: float
    retrieved_doc_ids: list[str]


@dataclass
class SubsetEvalSummary:
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    n_queries: int
    per_query: list[SubsetEvalResult] = field(default_factory=list)


def run_subset_eval(
    index_name: str,
    queries: list,
    client,
    top_k: int = 100,
) -> SubsetEvalSummary:
    """
    Query the BM25 server for the given queries against a named index.
    Computes recall@10, recall@100, nDCG@10 from the ranked results.

    Args:
        index_name: Name of the BM25 index on the server.
        queries: List of EvalQuery objects (query_id, query_text, relevant_doc_ids).
        client: BM25Client instance.
        top_k: Number of results to retrieve per query.
    """
    batch_results = client.batch_retrieve(index_name, queries, top_k=top_k)
    results_by_qid = {r["query_id"]: r["ranked_docs"] for r in batch_results}

    per_query = []
    hits_at_10 = 0
    hits_at_100 = 0
    ndcg_sum = 0.0

    for q in queries:
        ranked_docs = results_by_qid.get(q.query_id, [])
        retrieved_doc_ids = [d["doc_id"] for d in ranked_docs]

        # Find best rank of any relevant doc
        rank = None
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in q.relevant_doc_ids:
                rank = i + 1  # 1-indexed
                break

        hit_10 = rank is not None and rank <= 10
        hit_100 = rank is not None and rank <= 100

        # nDCG@10: for single relevant doc, nDCG = 1/log2(rank+1) if rank<=10
        ndcg = (1.0 / math.log2(rank + 1)) if (rank is not None and rank <= 10) else 0.0

        per_query.append(
            SubsetEvalResult(
                query_id=q.query_id,
                hit_at_10=hit_10,
                hit_at_100=hit_100,
                rank=rank,
                ndcg_at_10=ndcg,
                retrieved_doc_ids=retrieved_doc_ids[:10],
            )
        )

        hits_at_10 += int(hit_10)
        hits_at_100 += int(hit_100)
        ndcg_sum += ndcg

    n = len(queries) or 1
    return SubsetEvalSummary(
        recall_at_10=hits_at_10 / n,
        recall_at_100=hits_at_100 / n,
        ndcg_at_10=ndcg_sum / n,
        n_queries=len(queries),
        per_query=per_query,
    )


def load_preprocessor_from_code(code: str):
    """
    exec() a code string into a fresh module namespace and return Preprocessor().
    Used for hypothesis testing without filesystem writes.
    """
    eval_dir = str(pathlib.Path(__file__).parents[2] / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    module = types.ModuleType("_hypothesis_preprocess")
    # Set __file__ to a real path so pathlib.Path(__file__).parents[2] resolves
    # to the agents dir (matching what preprocess.py expects)
    fake_path = str(pathlib.Path(__file__).parent / "_hypothesis_preprocess.py")
    module.__file__ = fake_path
    exec(code, module.__dict__)
    return module.Preprocessor()
