"""
eval_utils.py — Subset evaluation utilities for the dense agent.

Operates against any client that exposes ``batch_retrieve(name, queries, top_k)``
returning ``[{"query_id": ..., "ranked_docs": [{"doc_id": ..., "rank": ...}]}, ...]``
— the dense_client.DenseClient and bm25_client.BM25Client both satisfy this.
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
    recall_at_10: float
    recall_at_100: float
    ranks: list[int]
    ndcg_at_10: float
    retrieved_doc_ids: list[str]

    @property
    def hit_at_10(self) -> bool:
        return self.recall_at_10 > 0

    @property
    def hit_at_100(self) -> bool:
        return self.recall_at_100 > 0

    @property
    def rank(self) -> int | None:
        return min(self.ranks) if self.ranks else None


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
    parent_map: dict[str, str] | None = None,
) -> SubsetEvalSummary:
    """Query the dense server for the given queries and compute metrics.

    Args:
        parent_map: Optional mapping from passage_id → parent_doc_id.
                    When provided, retrieved passage scores are MaxP-aggregated
                    to parent-document level, and relevant_doc_ids are also
                    mapped to parent level before computing metrics.
                    This is used for CRUMB passage corpus on splits with
                    document-level labels (clinical_trial, tip_of_the_tongue,
                    set_operation_entity_retrieval).
    """
    batch_results = client.batch_retrieve(index_name, queries, top_k=top_k)
    results_by_qid = {r["query_id"]: r["ranked_docs"] for r in batch_results}

    per_query = []
    recall_10_sum = 0.0
    recall_100_sum = 0.0
    ndcg_sum = 0.0

    for q in queries:
        ranked_docs = results_by_qid.get(q.query_id, [])
        retrieved_doc_ids = [d["doc_id"] for d in ranked_docs]

        # MaxP parent-document aggregation for passage corpus.
        if parent_map:
            parent_scores: dict[str, float] = {}
            for d in ranked_docs:
                pid = parent_map.get(d["doc_id"], d["doc_id"])
                score = d.get("score", 0.0)
                if pid not in parent_scores or score > parent_scores[pid]:
                    parent_scores[pid] = score
            sorted_parents = sorted(parent_scores.items(), key=lambda x: (-x[1], x[0]))
            retrieved_doc_ids = [pid for pid, _ in sorted_parents]
            relevant_set = set(
                parent_map.get(str(rid), str(rid)) for rid in q.relevant_doc_ids
            )
        else:
            relevant_set = set(q.relevant_doc_ids)

        n_relevant = len(relevant_set) or 1

        ranks = [
            i + 1 for i, doc_id in enumerate(retrieved_doc_ids)
            if doc_id in relevant_set
        ]

        recall_10 = len([r for r in ranks if r <= 10]) / n_relevant
        recall_100 = len([r for r in ranks if r <= 100]) / n_relevant

        dcg = sum(
            1.0 / math.log2(i + 1)
            for i, doc_id in enumerate(retrieved_doc_ids[:10], start=1)
            if doc_id in relevant_set
        )
        idcg = sum(
            1.0 / math.log2(i + 1)
            for i in range(1, min(len(relevant_set), 10) + 1)
        )
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        per_query.append(
            SubsetEvalResult(
                query_id=q.query_id,
                recall_at_10=recall_10,
                recall_at_100=recall_100,
                ranks=ranks,
                ndcg_at_10=ndcg,
                retrieved_doc_ids=retrieved_doc_ids[:10],
            )
        )

        recall_10_sum += recall_10
        recall_100_sum += recall_100
        ndcg_sum += ndcg

    n = len(queries) or 1
    return SubsetEvalSummary(
        recall_at_10=recall_10_sum / n,
        recall_at_100=recall_100_sum / n,
        ndcg_at_10=ndcg_sum / n,
        n_queries=len(queries),
        per_query=per_query,
    )


def load_preprocessor_from_code(code: str):
    """exec() a code string into a fresh module namespace and return Preprocessor()."""
    eval_dir = str(pathlib.Path(__file__).parents[2] / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    module = types.ModuleType("_hypothesis_preprocess")
    fake_path = str(pathlib.Path(__file__).parent / "_hypothesis_preprocess.py")
    module.__file__ = fake_path
    exec(code, module.__dict__)
    return module.Preprocessor()
