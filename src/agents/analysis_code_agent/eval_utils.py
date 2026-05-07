"""
eval_utils.py - Partial evaluation utilities using the BM25 HTTP client.

Provides subset evaluation (run_subset_eval) and dynamic preprocessor loading
(load_preprocessor_from_code) for hypothesis testing without filesystem writes.
"""

from __future__ import annotations

import hashlib
import math
import sys
import types
import pathlib
from dataclasses import dataclass, field


@dataclass
class SubsetEvalResult:
    query_id: str
    recall_at_10: float  # fractional: |relevant ∩ retrieved[:10]| / |relevant|
    recall_at_100: float  # fractional: |relevant ∩ retrieved[:100]| / |relevant|
    ranks: list[int]  # 1-indexed ranks of all relevant docs found (empty if none)
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
        """Best (lowest) rank among relevant docs, for backward compat."""
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
    recall_10_sum = 0.0
    recall_100_sum = 0.0
    ndcg_sum = 0.0

    for q in queries:
        ranked_docs = results_by_qid.get(q.query_id, [])
        retrieved_doc_ids = [d["doc_id"] for d in ranked_docs]
        relevant_set = set(q.relevant_doc_ids)
        n_relevant = len(relevant_set) or 1

        # Collect ranks of all relevant docs
        ranks = [
            i + 1 for i, doc_id in enumerate(retrieved_doc_ids)
            if doc_id in relevant_set
        ]

        # Fractional recall
        recall_10 = len([r for r in ranks if r <= 10]) / n_relevant
        recall_100 = len([r for r in ranks if r <= 100]) / n_relevant

        # nDCG@10: DCG over all relevant docs in top-10, normalized by IDCG
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


def sanitize_docs_for_preprocessing(docs: list) -> tuple[list, dict]:
    """Replace doc_ids with opaque sha256 hashes before passing to agent code.

    Returns (sanitized_docs, reverse_map) where reverse_map maps hash→original_id.
    Apply remap_chunk_doc_ids() to chunks after preprocessing to restore originals.
    """
    eval_dir = str(pathlib.Path(__file__).parents[2] / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from schema import Document

    reverse_map: dict[str, str] = {}
    sanitized = []
    for d in docs:
        hashed = "doc_" + hashlib.sha256(d.doc_id.encode()).hexdigest()[:16]
        reverse_map[hashed] = d.doc_id
        sanitized.append(Document(doc_id=hashed, text=d.text, metadata=d.metadata))
    return sanitized, reverse_map


def remap_chunk_doc_ids(chunks: list, reverse_map: dict) -> list:
    """Restore original doc_ids in chunks produced by sanitized preprocessing."""
    for chunk in chunks:
        if chunk.doc_id in reverse_map:
            chunk.doc_id = reverse_map[chunk.doc_id]
    return chunks


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


def _preprocess_worker(code, docs, output_path):
    import pickle, traceback, os
    try:
        preprocessor = load_preprocessor_from_code(code)
        chunks = preprocessor.preprocess(docs)
        tmp = output_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(("ok", chunks), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(tmp, output_path)
    except BaseException as e:
        try:
            with open(output_path, "wb") as f:
                pickle.dump(("error", repr(e), traceback.format_exc()), f)
        except Exception:
            pass


def run_preprocess_with_timeout(code: str, docs: list, timeout: float) -> list:
    """Run an untrusted preprocess() in a child process with a hard wall-clock timeout.

    Raises TimeoutError("preprocess() timed out after Ns") on timeout — the worker
    is killed via SIGTERM/SIGKILL so it cannot leak GIL-holding work into the parent.
    Linux only: uses fork() for cheap copy-on-write of docs / spaCy models.
    """
    import multiprocessing as mp
    import pickle, tempfile, os, signal
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory(prefix="preprocess_") as tdir:
        out_path = os.path.join(tdir, "result.pkl")
        proc = ctx.Process(target=_preprocess_worker, args=(code, docs, out_path))
        proc.start()
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            raise TimeoutError(f"preprocess() timed out after {int(timeout)}s")
        if not os.path.exists(out_path):
            sig_name = ""
            if proc.exitcode is not None and proc.exitcode < 0:
                try:
                    sig_name = f" (killed by {signal.Signals(-proc.exitcode).name})"
                except Exception:
                    pass
            raise RuntimeError(
                f"preprocess() worker exited with code {proc.exitcode}{sig_name}"
            )
        try:
            with open(out_path, "rb") as f:
                result = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"preprocess() worker output truncated: {e}")
        if result[0] == "error":
            raise RuntimeError(f"preprocess() raised: {result[1]}")
        return result[1]
