"""
prompt_utils.py – Shared prompt-building utilities for preprocessing agents.
"""

from __future__ import annotations


def build_eval_prompt(
    system_instruction: str,
    current_code: str,
    eval_results: dict | None,
    include_query_text: bool = True,
) -> str:
    """
    Assemble a full LLM prompt from the system instruction, current preprocess.py
    code, and the latest eval results.

    Returns the system instruction alone when there is no code or eval results yet.
    """
    if not current_code or eval_results is None:
        return system_instruction

    k = eval_results["top_k"]

    query_results = eval_results.get("query_results", [])
    misses = [r for r in query_results if not r["hit"]]
    hits = [r for r in query_results if r["hit"]]

    # Format missed queries
    missed_lines = []
    for r in misses:
        if include_query_text:
            missed_lines.append(
                f"  - [{r['query_id']}] \"{r['query_text']}\"\n"
                f"    Expected doc(s): {r['relevant_doc_ids']}\n"
                f"    Retrieved docs : {r['retrieved_doc_ids']}"
            )
        else:
            missed_lines.append(
                f"  - [{r['query_id']}]"
                f"    Expected doc(s): {r['relevant_doc_ids']}\n"
                f"    Retrieved docs : {r['retrieved_doc_ids']}"
            )

    # Format hits, sorted by rank (worst first)
    hit_lines = []
    for r in sorted(hits, key=lambda x: x["rank"] or 0, reverse=True):
        if include_query_text:
            hit_lines.append(
                f"  - [{r['query_id']}] rank={r['rank']} rr={r['reciprocal_rank']:.3f}  \"{r['query_text']}\""
            )
        else:
            hit_lines.append(
                f"  - [{r['query_id']}] rank={r['rank']} rr={r['reciprocal_rank']:.3f}"
            )

    missed_section = (
        f"### Missed queries ({len(misses)} / {len(query_results)}):\n"
        + ("\n".join(missed_lines) if missed_lines else "  (none)")
    )
    hit_section = (
        f"### Retrieved queries ({len(hits)} / {len(query_results)}) — sorted worst rank first:\n"
        + ("\n".join(hit_lines) if hit_lines else "  (none)")
    )

    return (
        f"{system_instruction}\n\n"
        f"## Current implementation\n```python\n{current_code}\n```\n\n"
        f"## Last eval results (top-{k})\n"
        f"- Recall@{k}: {eval_results['recall_at_k']:.4f}\n"
        f"- MRR: {eval_results['mrr']:.4f}\n"
        f"- Chunks indexed: {eval_results['n_chunks']}\n\n"
        f"## Per-query breakdown\n"
        f"{missed_section}\n\n"
        f"{hit_section}\n\n"
        "Improve the implementation to increase Recall and MRR."
    )
