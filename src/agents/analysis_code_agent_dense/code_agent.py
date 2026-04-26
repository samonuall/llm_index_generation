"""
code_agent.py — Hypothesis generation, testing, and final code synthesis (dense variant).

Differences from analysis_code_agent.code_agent:
  - Hypothesis index names are loop-scoped: ``hyp_loop{i}_{H.id}``.
  - The "current" baseline index name is configurable per-call so the orchestrator
    can use ``current_loop{i}`` and drop it at end of loop.
  - Prompts are tuned for dense retrieval (no BM25 length-norm cautions, with a
    note about embedding token caps).
"""
from __future__ import annotations

import logging
import os
import re
import json
import pathlib
import time
import concurrent.futures
from dataclasses import dataclass, field

from dotenv import load_dotenv
from .llm_call import completion

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent

logger = logging.getLogger("analysis_code_agent_dense")


@dataclass
class Hypothesis:
    id: str
    description: str
    rationale: str
    code: str
    query_ids_to_test: list[str] = field(default_factory=list)
    falsifying_condition: str = ""


@dataclass
class HypothesisResult:
    hypothesis: Hypothesis
    hypothesis_recall_100: float = 0.0
    baseline_recall_100: float = 0.0
    delta_recall_100: float = 0.0
    hypothesis_recall_10: float = 0.0
    baseline_recall_10: float = 0.0
    delta_recall_10: float = 0.0
    delta_ndcg_10: float = 0.0
    proven: bool = False
    error: str | None = None
    notes: str = ""
    improved_query_ids: list[str] = field(default_factory=list)
    regressed_query_ids: list[str] = field(default_factory=list)


class CodeAgent:

    def __init__(self, config: dict, tracker=None, split: str = "tip_of_the_tongue", log_dir: pathlib.Path | None = None) -> None:
        self._config = config
        self._tracker = tracker
        self._model = config.get("code_model", "openai/gpt-4o")
        self._temperature = config.get("code_temperature", 0.7)
        self._api_base = config.get("api_base")
        _proxy_key = os.environ.get("LITE_LLM_KEY", os.environ.get("LITELLM_API_KEY", ""))
        self._api_key = _proxy_key if self._api_base else None
        self._llm_timeout = config.get("code_llm_timeout", None)
        self._recall_threshold = config.get("recall_improvement_threshold", 0.05)
        self._max_hypotheses = config.get("max_hypotheses", 4)
        self._split = split
        self._log_dir = log_dir
        self._max_input_tokens = int(config.get("embedding_max_input_tokens", 8192))

        # Load system prompt
        system_path = _AGENT_DIR / "context" / "CODE_SYSTEM.md"
        self._system_prompt = system_path.read_text(encoding="utf-8")

    def _log_call(self, label: str, messages: list[dict], response_text: str) -> None:
        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        log_path = self._log_dir / f"code_{label}_{timestamp}.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"=== Code Agent | {label} | {timestamp} ===\n\n")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").upper()
                f.write(f"--- MESSAGE {i} [{role}] ---\n")
                f.write(f"{msg.get('content', '')}\n\n")
            f.write("--- RESPONSE ---\n")
            f.write(response_text)
        print(f"[code_agent] Log: {log_path}")

    # ------------------------------------------------------------------
    # Hypothesis idea generation (Phase 1)
    # ------------------------------------------------------------------

    def _build_past_section(
        self,
        past_hypotheses: list[dict] | None,
        query_lookup: dict[str, str] | None,
    ) -> str:
        if not past_hypotheses:
            return ""
        all_failed = all(not ph["proven"] for ph in past_hypotheses)
        chunking_variations = sum(
            1 for ph in past_hypotheses
            if any(w in ph["description"].lower() for w in ["chunk", "window", "overlap", "paragraph", "sentence"])
        )
        lines = []
        for ph in past_hypotheses:
            lines.append(
                f"- **{ph['id']}: {ph['description']}** → "
                f"delta_recall@100={ph['delta_recall_100']:+.4f}, "
                f"delta_ndcg@10={ph['delta_ndcg_10']:+.4f}, "
                f"proven={ph['proven']}. {ph.get('notes', '')}"
            )
            if query_lookup:
                improved = ph.get("improved_query_ids", [])[:5]
                regressed = ph.get("regressed_query_ids", [])[:5]
                if improved or regressed:
                    lines.append("  **What changed (contrastive):**")
                for qid in improved:
                    qt = query_lookup.get(qid, "")[:120]
                    lines.append(f"  fixed   [{qid}] \"{qt}\"")
                for qid in regressed:
                    qt = query_lookup.get(qid, "")[:120]
                    lines.append(f"  broke   [{qid}] \"{qt}\"")

        diagnosis = ""
        if all_failed and chunking_variations >= 3:
            diagnosis = (
                "\nPATTERN DETECTED: Multiple chunking/window variations have all failed. "
                "The retrieval problem is NOT about chunk boundaries -- it is about semantics. "
                "Do NOT generate any more chunking or window strategies.\n"
            )
        elif all_failed:
            diagnosis = (
                "\nAll previous hypotheses failed. Every new hypothesis must be "
                "mechanically different -- not a renaming or minor tweak of what was tried.\n"
            )

        diversity_instruction = (
            "\n## Diversity Requirement\n"
            "The approaches already tried are listed above. "
            "Each new hypothesis must be different from all of them.\n"
        )
        return (
            "\n## Previously Tested Hypotheses (DO NOT repeat these)\n"
            + "\n".join(lines)
            + diagnosis
            + diversity_instruction
        )

    def _build_persistent_section(self, persistent_failure_ids: list[str] | None) -> str:
        if not persistent_failure_ids:
            return ""
        pf_ids_str = ", ".join(persistent_failure_ids[:30]) + ("..." if len(persistent_failure_ids) > 30 else "")
        return (
            f"## Persistent Failures (failing EVERY iteration so far -- highest priority)\n"
            f"These {len(persistent_failure_ids)} query IDs have never been retrieved correctly: {pf_ids_str}\n"
            f"At least one hypothesis MUST specifically target these queries.\n\n"
        )

    def _dense_retrieval_notes(self) -> str:
        return (
            "IMPORTANT NOTES (dense retrieval):\n"
            "- The retriever uses cosine similarity over L2-normalised embeddings from a Qwen embedding model.\n"
            f"- Each chunk is embedded after being truncated to {self._max_input_tokens} tokens. Anything beyond that limit is silently dropped during embedding -- prefer multiple shorter chunks over a single oversize one when document text exceeds the budget.\n"
            "- Synonym injection / lexical expansion is much less helpful than for BM25; semantic clarity wins. Useful operations include: shorter focused chunks for long documents, query-style summary chunks, title prepending, removing pure boilerplate.\n"
            "- The documents in this dataset have EMPTY metadata dicts (no title, no aliases). Do NOT rely on doc.metadata for anything.\n"
            "- Documents are full-length documents (potentially thousands of words), not pre-chunked passages.\n"
        )

    def _regression_prevention(self) -> str:
        return (
            "## CRITICAL: Regression Prevention\n"
            "- ALWAYS keep the original full-document chunk as chunk_0 alongside any new chunks. The eval uses max-score aggregation per doc_id, so additional chunks can only help -- removing the original full-doc chunk risks losing queries that currently succeed.\n"
            "- Keep additional chunks to 1-3 per document max. Over-chunking gives the index many short snippets that compete with the full-document chunk during top-k cutoffs.\n"
            "- Do NOT aggressively filter or remove text. Only ADD targeted chunks alongside the original.\n"
        )

    def generate_hypotheses(
        self,
        analysis_summary: str,
        current_code: str,
        n: int = 4,
        past_hypotheses: list[dict] | None = None,
        persistent_failure_ids: list[str] | None = None,
        query_lookup: dict[str, str] | None = None,
    ) -> list[Hypothesis]:
        """Single LLM call to generate N hypotheses with code inline."""

        past_section = self._build_past_section(past_hypotheses, query_lookup)
        persistent_section = self._build_persistent_section(persistent_failure_ids)

        prompt = f"""## Analysis Summary
{analysis_summary}

## Current preprocess.py
```python
{current_code}
```

{past_section}
{persistent_section}Generate exactly {n} hypotheses to improve the preprocessing code.
Each hypothesis must be a complete, working preprocess.py implementation.

{self._dense_retrieval_notes()}

{self._regression_prevention()}

Output each hypothesis as a SEPARATE block using this format (do NOT use JSON):

### H1: <description>
Rationale: <rationale>
Query IDs: <comma-separated query_ids>
Falsifying: <condition>
```python
<complete preprocess.py code>
```

Repeat for H2, H3, H4.

The code MUST start with the standard imports:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor
```

IMPORTANT: Each hypothesis code must be complete and self-contained. It should define `class Preprocessor(BasePreprocessor)` with a `preprocess(self, docs: List[Document]) -> List[Chunk]` method.
"""

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            _t0 = time.time()
            response = completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
                timeout=self._llm_timeout,
            )
            if self._tracker:
                self._tracker.record_llm_call(response, time.time() - _t0, agent="code")
            text = response.choices[0].message.content or ""
            self._log_call("generate_hypotheses", messages, text)
        except Exception as e:
            logger.exception("Hypothesis generation LLM call failed (model=%s)", self._model)
            print(f"[code_agent] Hypothesis generation failed: {e}")
            return []

        raw = self._parse_hypotheses_blocks(text)
        if raw is None:
            raw = self._parse_hypotheses_json(text)

        if not raw:
            logger.warning("Failed to parse hypotheses. Raw model output:\n%s", text)
            print("[code_agent] No hypotheses parsed. Returning empty.")
            return []

        hypotheses = []
        for h in raw[:n]:
            hypotheses.append(
                Hypothesis(
                    id=h.get("id", f"H{len(hypotheses) + 1}"),
                    description=h.get("description", ""),
                    rationale=h.get("rationale", ""),
                    code=h.get("code", ""),
                    query_ids_to_test=h.get("query_ids_to_test", []),
                    falsifying_condition=h.get("falsifying_condition", ""),
                )
            )

        return hypotheses

    # ------------------------------------------------------------------
    # Two-phase hypothesis generation (Phase 1 ideas, Phase 2 code in parallel)
    # ------------------------------------------------------------------

    async def generate_hypotheses_async(
        self,
        analysis_summary: str,
        current_code: str,
        n: int = 4,
        past_hypotheses: list[dict] | None = None,
        persistent_failure_ids: list[str] | None = None,
        query_lookup: dict[str, str] | None = None,
    ) -> list[Hypothesis]:
        ideas = self._generate_hypothesis_ideas(
            analysis_summary, current_code, n,
            past_hypotheses=past_hypotheses,
            persistent_failure_ids=persistent_failure_ids,
            query_lookup=query_lookup,
        )
        if not ideas:
            return []

        print(f"[code_agent] Phase 1: generated {len(ideas)} hypothesis ideas, generating code in parallel ...")

        import asyncio
        tasks = [
            self._generate_code_for_idea(idea, current_code, analysis_summary)
            for idea in ideas
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        hypotheses = []
        for idea, result in zip(ideas, results):
            if isinstance(result, Exception):
                print(f"[code_agent] Code generation failed for {idea['id']}: {result}")
                continue
            if result is not None:
                hypotheses.append(result)

        print(f"[code_agent] Phase 2: {len(hypotheses)}/{len(ideas)} hypotheses generated successfully.")
        return hypotheses

    def _generate_hypothesis_ideas(
        self,
        analysis_summary: str,
        current_code: str,
        n: int = 4,
        past_hypotheses: list[dict] | None = None,
        persistent_failure_ids: list[str] | None = None,
        query_lookup: dict[str, str] | None = None,
    ) -> list[dict]:
        past_section = self._build_past_section(past_hypotheses, query_lookup)
        persistent_section = self._build_persistent_section(persistent_failure_ids)

        prompt = f"""## Analysis Summary
{analysis_summary}

## Current preprocess.py
```python
{current_code}
```

{past_section}
{persistent_section}Generate exactly {n} hypothesis IDEAS to improve the preprocessing code.
Do NOT include code -- just describe each idea clearly.

{self._dense_retrieval_notes()}

{self._regression_prevention()}

Output each hypothesis idea using this format (NO code):

### H1: <description>
Rationale: <detailed rationale explaining why this approach should improve retrieval>
Query IDs: <comma-separated query_ids this targets>
Falsifying: <condition that would prove this wrong>

Repeat for H2, H3, H4.
"""

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            _t0 = time.time()
            response = completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
                timeout=self._llm_timeout,
            )
            if self._tracker:
                self._tracker.record_llm_call(response, time.time() - _t0, agent="code")
            text = response.choices[0].message.content or ""
            self._log_call("generate_ideas", messages, text)
        except Exception as e:
            logger.exception("Idea generation LLM call failed (model=%s)", self._model)
            print(f"[code_agent] Idea generation failed: {e}")
            return []

        raw = self._parse_hypotheses_blocks(text, require_code=False)
        if raw is None:
            raw = self._parse_hypotheses_json(text)
        if not raw:
            print("[code_agent] No ideas parsed from Phase 1 response.")
            return []

        for i, idea in enumerate(raw):
            if "id" not in idea:
                idea["id"] = f"H{i + 1}"

        return raw[:n]

    async def _generate_code_for_idea(
        self,
        idea: dict,
        current_code: str,
        analysis_summary: str,
    ) -> Hypothesis | None:
        from .llm_call import async_completion

        prompt = f"""## Task
Write a complete, working `preprocess.py` implementation for this hypothesis:

### {idea['id']}: {idea['description']}
Rationale: {idea.get('rationale', '')}

## Current preprocess.py (base to build on)
```python
{current_code}
```

## Brief Analysis Context
{analysis_summary[:2000]}

{self._dense_retrieval_notes()}

## Requirements
- Output ONLY a single ```python``` code block with the complete preprocess.py
- The code MUST start with the standard imports:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor
```
- Define `class Preprocessor(BasePreprocessor)` with a `preprocess(self, docs: List[Document]) -> List[Chunk]` method
- chunk.doc_id must exactly match the source Document.doc_id
- The documents have EMPTY metadata dicts -- do NOT rely on doc.metadata
- Build on top of the current code, don't throw it away
- CRITICAL: Always emit the original full-document text as chunk_0 (the passthrough chunk). Any additional chunks are extras alongside it, not replacements. This prevents regressions on currently-working queries.
- Keep additional chunks to 1-3 per document max. Do NOT create many small chunks per doc.
"""

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        _t0 = time.time()
        response = await async_completion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            api_key=self._api_key,
            api_base=self._api_base,
            timeout=self._llm_timeout,
        )
        if self._tracker:
            self._tracker.record_llm_call(response, time.time() - _t0, agent="code")
        text = response.choices[0].message.content or ""
        self._log_call(f"generate_code_{idea['id']}", messages, text)

        code_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if not code_match:
            logger.warning("No code block found for %s. Raw response:\n%s", idea.get("id"), text)
            print(f"[code_agent] No code block found for {idea['id']}")
            return None

        code = code_match.group(1).strip()
        return Hypothesis(
            id=idea.get("id", "H?"),
            description=idea.get("description", ""),
            rationale=idea.get("rationale", ""),
            code=code,
            query_ids_to_test=idea.get("query_ids_to_test", []),
            falsifying_condition=idea.get("falsifying_condition", ""),
        )

    def _parse_hypotheses_json(self, text: str) -> list[dict] | None:
        match = re.search(r"<hypotheses>(.*?)</hypotheses>", text, re.DOTALL)
        if not match:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return None
        try:
            raw_text = match.group(1) if "<hypotheses>" in match.group(0) else match.group(0)
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning("Hypothesis JSON parse failed: %s", e)
            return None

    def _parse_hypotheses_blocks(self, text: str, require_code: bool = True) -> list[dict] | None:
        header_pattern = r"###\s+(H\d+)\s*:\s*(.+?)(?:\n|$)"
        code_pattern = r"```python\s*\n(.*?)```"

        headers = list(re.finditer(header_pattern, text))
        if not headers:
            return None

        codes = list(re.finditer(code_pattern, text, re.DOTALL))
        if require_code and not codes:
            return None

        results = []
        for i, header in enumerate(headers):
            h_id = header.group(1)
            desc = header.group(2).strip()
            header_end = header.end()
            next_header_start = headers[i + 1].start() if i + 1 < len(headers) else len(text)

            code = None
            for c in codes:
                if header_end <= c.start() < next_header_start:
                    code = c.group(1).strip()
                    break

            if require_code and not code:
                continue

            between = text[header_end:next_header_start]
            rationale_match = re.search(r"Rationale:\s*(.+?)(?:\n|$)", between)
            qids_match = re.search(r"Query IDs?:\s*(.+?)(?:\n|$)", between)
            falsify_match = re.search(r"Falsif(?:ying|ication):\s*(.+?)(?:\n|$)", between)

            query_ids = []
            if qids_match:
                query_ids = [q.strip().strip("[]\"'") for q in qids_match.group(1).split(",")]

            entry = {
                "id": h_id,
                "description": desc,
                "rationale": rationale_match.group(1).strip() if rationale_match else "",
                "query_ids_to_test": query_ids,
                "falsifying_condition": falsify_match.group(1).strip() if falsify_match else "",
            }
            if code is not None:
                entry["code"] = code
            results.append(entry)

        return results if results else None

    def _validate_code(self, code: str, documents: list) -> str | None:
        from .eval_utils import load_preprocessor_from_code
        try:
            sample = documents[:20]
            valid_doc_ids = {d.doc_id for d in sample}
            preprocessor = load_preprocessor_from_code(code)
            chunks = preprocessor.preprocess(sample)
            if not chunks:
                return "preprocess() returned empty list on sample docs"
            for c in chunks:
                if not hasattr(c, "doc_id") or not hasattr(c, "text"):
                    return f"Chunk missing doc_id or text: {c}"
                if c.doc_id not in valid_doc_ids:
                    return (
                        f"Chunk has invalid doc_id '{c.doc_id}' -- "
                        f"chunk.doc_id must exactly match one of the input document doc_ids. "
                        f"Valid example: '{next(iter(valid_doc_ids))}'"
                    )
            return None
        except Exception as e:
            logger.exception("Validation of generated code raised")
            return str(e)

    # ------------------------------------------------------------------
    # Hypothesis testing — per-loop named index, dropped after eval
    # ------------------------------------------------------------------

    def test_hypothesis(
        self,
        hypothesis: Hypothesis,
        documents: list,
        queries: list,
        current_code: str,
        client,
        *,
        iteration: int | None = None,
        current_index_name: str = "current",
    ) -> HypothesisResult:
        """Test a single hypothesis by building a temp index and running subset eval.

        The hypothesis index name follows the per-loop / per-hypothesis lifecycle:
            ``hyp_loop{iteration}_{H.id}`` if ``iteration`` is given, else ``hyp_{H.id}``.
        The index is always dropped in a ``finally`` block, even on failure.
        """
        from .eval_utils import load_preprocessor_from_code, run_subset_eval

        result = HypothesisResult(hypothesis=hypothesis)
        index_name = (
            f"hyp_loop{iteration}_{hypothesis.id}"
            if iteration is not None
            else f"hyp_{hypothesis.id}"
        )

        validation_error = self._validate_code(hypothesis.code, documents)
        if validation_error:
            result.error = f"Validation failed: {validation_error}"
            result.notes = f"Code rejected at validation: {validation_error[:120]}"
            logger.error("Hypothesis %s validation failed: %s", hypothesis.id, validation_error)
            print(f"[code_agent] {hypothesis.id} validation error: {validation_error[:120]}")
            return result

        preprocess_timeout = self._config.get("preprocess_timeout_seconds", 120)

        try:
            test_queries = queries

            preprocessor = load_preprocessor_from_code(hypothesis.code)
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = ex.submit(preprocessor.preprocess, documents)
            try:
                chunks = future.result(timeout=preprocess_timeout)
            except concurrent.futures.TimeoutError:
                ex.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"preprocess() timed out after {preprocess_timeout}s")
            else:
                ex.shutdown(wait=True)

            client.build_index(index_name, chunks, persist=False)

            hyp_eval = run_subset_eval(index_name, test_queries, client)
            current_eval = run_subset_eval(current_index_name, test_queries, client)

            result.hypothesis_recall_100 = hyp_eval.recall_at_100
            result.baseline_recall_100 = current_eval.recall_at_100
            result.delta_recall_100 = hyp_eval.recall_at_100 - current_eval.recall_at_100

            result.hypothesis_recall_10 = hyp_eval.recall_at_10
            result.baseline_recall_10 = current_eval.recall_at_10
            result.delta_recall_10 = hyp_eval.recall_at_10 - current_eval.recall_at_10
            result.delta_ndcg_10 = hyp_eval.ndcg_at_10 - current_eval.ndcg_at_10

            result.proven = result.delta_recall_100 >= self._recall_threshold

            result.improved_query_ids = [
                h_q.query_id
                for h_q, c_q in zip(hyp_eval.per_query, current_eval.per_query)
                if h_q.hit_at_100 and not c_q.hit_at_100
            ]
            result.regressed_query_ids = [
                c_q.query_id
                for h_q, c_q in zip(hyp_eval.per_query, current_eval.per_query)
                if c_q.hit_at_100 and not h_q.hit_at_100
            ]
            result.notes = (
                f"@100: improved {len(result.improved_query_ids)}, "
                f"regressed {len(result.regressed_query_ids)} of {len(test_queries)} queries"
            )

            print(
                f"[code_agent] {hypothesis.id}: "
                f"delta_recall@100={result.delta_recall_100:+.4f} "
                f"delta_recall@10={result.delta_recall_10:+.4f} "
                f"delta_ndcg@10={result.delta_ndcg_10:+.4f} "
                f"proven={result.proven}"
            )

        except Exception as e:
            logger.exception(
                "test_hypothesis failed for %s (desc=%r, targeted_queries=%s)",
                hypothesis.id, hypothesis.description, hypothesis.query_ids_to_test,
            )
            result.error = str(e)
            result.notes = f"Error: {e}"
            print(f"[code_agent] {hypothesis.id} error: {e}")

        finally:
            try:
                client.delete_index(index_name)
            except Exception:
                logger.debug("delete_index(%s) failed during cleanup", index_name, exc_info=True)

        return result

    def generate_final_code(
        self,
        analysis_summary: str,
        proven_results: list[HypothesisResult],
        current_code: str,
    ) -> str | None:
        """Generate final preprocess.py from analysis + proven hypotheses."""
        proven_text = ""
        for r in proven_results:
            h = r.hypothesis
            proven_text += f"""### {h.id}: {h.description}
- Rationale: {h.rationale}
- Delta recall@100: {r.delta_recall_100:+.4f}
- Delta recall@10: {r.delta_recall_10:+.4f}
- Delta nDCG@10: {r.delta_ndcg_10:+.4f}
- Notes: {r.notes}
- Code:
```python
{h.code}
```

"""

        prompt = f"""## Analysis Summary
{analysis_summary}

## Proven Hypotheses
{proven_text}

## Current preprocess.py
```python
{current_code}
```

{self._dense_retrieval_notes()}

Synthesize ALL proven hypotheses into a single, final preprocess.py implementation.
Combine the best ideas from each proven hypothesis.

Output ONLY the complete Python code for the final preprocess.py inside a ```python ... ``` block.

The code MUST:
1. Start with the standard imports (sys.path, schema, base)
2. Define `class Preprocessor(BasePreprocessor)` with name and description attributes
3. Implement `def preprocess(self, docs: List[Document]) -> List[Chunk]`
4. Return at least one Chunk per Document
5. Ensure chunk_id is globally unique and chunk.doc_id matches source doc_id
"""

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            _t0 = time.time()
            response = completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
                timeout=self._llm_timeout,
            )
            if self._tracker:
                self._tracker.record_llm_call(response, time.time() - _t0, agent="code")
            text = response.choices[0].message.content or ""
            self._log_call("generate_final_code", messages, text)
        except Exception as e:
            print(f"[code_agent] Final code generation failed: {e}")
            return None

        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if not match:
            print("[code_agent] No python block in final code response")
            return None

        code = match.group(1).strip()

        if "class Preprocessor" not in code:
            print("[code_agent] Final code missing 'class Preprocessor' - rejected")
            return None

        return code
