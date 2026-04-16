"""
code_agent.py - Hypothesis generation, testing, and final code synthesis.
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

logger = logging.getLogger("analysis_code_agent")


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
        self._tracker = tracker
        self._model = config.get("code_model", "openai/gpt-4o")
        self._temperature = config.get("code_temperature", 0.7)
        self._api_base = config.get("api_base")  # None = use provider's native endpoint
        # Only pass api_key explicitly for proxy; native providers read key from env.
        _proxy_key = os.environ.get("LITE_LLM_KEY", os.environ.get("LITELLM_API_KEY", ""))
        self._api_key = _proxy_key if self._api_base else None
        self._llm_timeout = config.get("code_llm_timeout", None)
        self._recall_threshold = config.get("recall_improvement_threshold", 0.05)
        self._max_hypotheses = config.get("max_hypotheses", 4)
        self._split = split
        self._log_dir = log_dir

        # Load system prompt
        system_path = _AGENT_DIR / "context" / "CODE_SYSTEM.md"
        self._system_prompt = system_path.read_text(encoding="utf-8")

    def _log_call(self, label: str, messages: list[dict], response_text: str) -> None:
        """Write a verbose log of a code agent LLM call, matching analysis agent style."""
        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        # Use a timestamp suffix to avoid collisions between calls
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

    def generate_hypotheses(
        self,
        analysis_summary: str,
        current_code: str,
        n: int = 4,
        past_hypotheses: list[dict] | None = None,
        persistent_failure_ids: list[str] | None = None,
        query_lookup: dict[str, str] | None = None,
    ) -> list[Hypothesis]:
        """Single LLM call to generate N hypotheses. Output JSON inside <hypotheses>...</hypotheses> tags."""

        # Build past attempts section with pattern diagnosis
        past_section = ""
        if past_hypotheses:
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
                # Contrastive table: show query text for improved and regressed queries
                if query_lookup:
                    improved = ph.get("improved_query_ids", [])[:5]
                    regressed = ph.get("regressed_query_ids", [])[:5]
                    if improved or regressed:
                        lines.append("  **What changed (contrastive):**")
                    for qid in improved:
                        qt = query_lookup.get(qid, "")[:120]
                        lines.append(f"  ✓ fixed   [{qid}] \"{qt}\"")
                    for qid in regressed:
                        qt = query_lookup.get(qid, "")[:120]
                        lines.append(f"  ✗ broke   [{qid}] \"{qt}\"")

            diagnosis = ""
            if all_failed and chunking_variations >= 3:
                diagnosis = (
                    "\n⚠ PATTERN DETECTED: Multiple chunking/window variations have all failed. "
                    "The retrieval problem is NOT about chunk boundaries — it is about vocabulary. "
                    "Do NOT generate any more chunking or window strategies.\n"
                )
            elif all_failed:
                diagnosis = (
                    "\n⚠ All previous hypotheses failed. Every new hypothesis must be "
                    "mechanically different — not a renaming or minor tweak of what was tried.\n"
                )

            past_descriptions = [ph["description"] for ph in past_hypotheses]
            diversity_instruction = (
                "\n## Diversity Requirement\n"
                "The approaches already tried are listed above. Each new hypothesis MUST be "
                "mechanically different from all of them — different *operation* on the text, "
                "not just a different parameter or a renaming.\n"
                f"Already tried: {'; '.join(past_descriptions)}\n"
            )

            past_section = (
                "\n## Previously Tested Hypotheses (DO NOT repeat these)\n"
                + "\n".join(lines)
                + diagnosis
                + diversity_instruction
            )

        if persistent_failure_ids:
            pf_ids_str = ", ".join(persistent_failure_ids[:30]) + ("..." if len(persistent_failure_ids) > 30 else "")
            persistent_section = (
                f"## Persistent Failures (failing EVERY iteration so far — highest priority)\n"
                f"These {len(persistent_failure_ids)} query IDs have never been retrieved correctly: {pf_ids_str}\n"
                f"At least one hypothesis MUST specifically target these queries.\n\n"
            )
        else:
            persistent_section = ""

        prompt = f"""## Analysis Summary
{analysis_summary}

## Current preprocess.py
```python
{current_code}
```

{past_section}
{persistent_section}Generate exactly {n} hypotheses to improve the preprocessing code.
Each hypothesis must be a complete, working preprocess.py implementation.

IMPORTANT NOTES:
- The documents in this dataset have EMPTY metadata dicts (no title, no aliases). Do NOT rely on doc.metadata for anything.
- The BM25 tokenizer lowercases and stems text. Stopword removal is NOT done by the preprocessor — it's handled by BM25.
- Documents are full-length documents (potentially thousands of words), not pre-chunked passages.

## CRITICAL: Regression Prevention
- **ALWAYS keep the original full-document chunk** alongside any new chunks. The eval uses max-score aggregation per doc_id, so extra chunks can only help — but removing the original full-doc chunk risks losing queries that currently succeed.
- Do NOT split documents into sections/paragraphs WITHOUT also keeping the full original text as chunk 0. The safest pattern: chunk_0 = full original text, chunk_1..N = additional targeted chunks.
- Do NOT aggressively filter or remove text from the original document. Only add to it.
- The corpus has 200K+ docs. Splitting each into 10-20 small chunks creates millions of index entries, inflating the index with short metadata-heavy chunks that score high due to BM25 length normalization. Keep additional chunks to 1-3 per document maximum.

Output each hypothesis as a SEPARATE block using this format (do NOT use JSON):

### H1: <description>
Mechanism: <your own short label for the core operation, e.g. TITLE-INJECTION>
Rationale: <rationale>
Query IDs: <comma-separated query_ids>
Falsifying: <condition>
```python
<complete preprocess.py code>
```

Repeat for H2, H3, H4. Each must have a DIFFERENT Mechanism label.

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

        # Parse hypotheses — try markdown blocks first (our default format), then JSON
        raw = self._parse_hypotheses_blocks(text)
        if raw is None:
            raw = self._parse_hypotheses_json(text)

        if raw is None:
            # Retry with explicit instructions
            print("[code_agent] Parse failed. Retrying with structured format...")
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Could not parse hypotheses. Please output each hypothesis "
                        "as a SEPARATE block using this exact format:\n\n"
                        "### H1: <description>\n"
                        "Rationale: <rationale>\n"
                        "Query IDs: <comma-separated query_ids>\n"
                        "Falsifying: <condition>\n"
                        "```python\n<complete preprocess.py code>\n```\n\n"
                        "Repeat for H2, H3, H4."
                    ),
                }
            )
            try:
                response = completion(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    api_key=self._api_key,
                    api_base=self._api_base,
                )
                text2 = response.choices[0].message.content or ""
                self._log_call("generate_hypotheses_retry", messages, text2)
                raw = self._parse_hypotheses_blocks(text2)
                if raw is None:
                    raw = self._parse_hypotheses_json(text2)
            except Exception as e:
                logger.exception("Hypothesis generation retry failed")
                print(f"[code_agent] Retry failed: {e}")

        if not raw:
            logger.warning(
                "Failed to parse hypotheses. Raw model output:\n%s",
                locals().get("text2", text),
            )
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
    # Two-phase hypothesis generation (reduces per-call token load)
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
        """Two-phase hypothesis generation: ideas first, then code in parallel.

        Phase 1: Single LLM call to generate N hypothesis ideas (no code).
        Phase 2: N parallel async LLM calls to generate code for each idea.
        """
        # --- Phase 1: Generate ideas (no code) ---
        ideas = self._generate_hypothesis_ideas(
            analysis_summary, current_code, n,
            past_hypotheses=past_hypotheses,
            persistent_failure_ids=persistent_failure_ids,
            query_lookup=query_lookup,
        )
        if not ideas:
            return []

        print(f"[code_agent] Phase 1: generated {len(ideas)} hypothesis ideas, generating code in parallel ...")

        # --- Phase 2: Generate code for each idea in parallel ---
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
        """Phase 1: Generate hypothesis ideas without code (single LLM call)."""

        # Reuse the same past_section / persistent_section logic from generate_hypotheses
        past_section = ""
        if past_hypotheses:
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
                        lines.append(f"  ✓ fixed   [{qid}] \"{qt}\"")
                    for qid in regressed:
                        qt = query_lookup.get(qid, "")[:120]
                        lines.append(f"  ✗ broke   [{qid}] \"{qt}\"")

            diagnosis = ""
            if all_failed and chunking_variations >= 3:
                diagnosis = (
                    "\n⚠ PATTERN DETECTED: Multiple chunking/window variations have all failed. "
                    "The retrieval problem is NOT about chunk boundaries — it is about vocabulary. "
                    "Do NOT generate any more chunking or window strategies.\n"
                )
            elif all_failed:
                diagnosis = (
                    "\n⚠ All previous hypotheses failed. Every new hypothesis must be "
                    "mechanically different — not a renaming or minor tweak of what was tried.\n"
                )

            past_descriptions = [ph["description"] for ph in past_hypotheses]
            diversity_instruction = (
                "\n## Diversity Requirement\n"
                "The approaches already tried are listed above. Each new hypothesis MUST be "
                "mechanically different from all of them — different *operation* on the text, "
                "not just a different parameter or a renaming.\n"
                f"Already tried: {'; '.join(past_descriptions)}\n"
            )

            past_section = (
                "\n## Previously Tested Hypotheses (DO NOT repeat these)\n"
                + "\n".join(lines)
                + diagnosis
                + diversity_instruction
            )

        persistent_section = ""
        if persistent_failure_ids:
            pf_ids_str = ", ".join(persistent_failure_ids[:30]) + ("..." if len(persistent_failure_ids) > 30 else "")
            persistent_section = (
                f"## Persistent Failures (failing EVERY iteration so far — highest priority)\n"
                f"These {len(persistent_failure_ids)} query IDs have never been retrieved correctly: {pf_ids_str}\n"
                f"At least one hypothesis MUST specifically target these queries.\n\n"
            )

        prompt = f"""## Analysis Summary
{analysis_summary}

## Current preprocess.py
```python
{current_code}
```

{past_section}
{persistent_section}Generate exactly {n} hypothesis IDEAS to improve the preprocessing code.
Do NOT include code — just describe each idea clearly.

IMPORTANT NOTES:
- The documents in this dataset have EMPTY metadata dicts (no title, no aliases). Do NOT rely on doc.metadata for anything.
- The BM25 tokenizer lowercases and stems text. Stopword removal is NOT done by the preprocessor — it's handled by BM25.
- Documents are full-length documents (potentially thousands of words), not pre-chunked passages.

## CRITICAL: Regression Prevention
- **ALWAYS keep the original full-document chunk** alongside any new chunks. The eval uses max-score aggregation per doc_id, so extra chunks can only help — but removing the original full-doc chunk risks losing queries that currently succeed.
- Do NOT propose splitting documents into sections/paragraphs WITHOUT also keeping the full original text as chunk 0.
- Do NOT propose aggressively filtering or removing text from the original document. Only add to it.
- The corpus has 200K+ docs. Avoid creating many small chunks per doc (keep to 1-3 additional chunks max) — over-chunking inflates the index with short metadata-heavy chunks that score high due to BM25 length normalization.

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

        return raw[:n]

    async def _generate_code_for_idea(
        self,
        idea: dict,
        current_code: str,
        analysis_summary: str,
    ) -> Hypothesis | None:
        """Phase 2: Generate code for a single hypothesis idea (async LLM call)."""
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
- The documents have EMPTY metadata dicts — do NOT rely on doc.metadata
- Build on top of the current code, don't throw it away
- **CRITICAL**: Always emit the original full-document text as chunk_0 (the passthrough chunk). Any additional chunks are extras alongside it, not replacements. This prevents regressions on currently-working queries.
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

        # Extract code block
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
        """Try to parse hypotheses from <hypotheses>JSON</hypotheses> tags."""
        match = re.search(r"<hypotheses>(.*?)</hypotheses>", text, re.DOTALL)
        if not match:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return None
        try:
            raw_text = match.group(1) if "<hypotheses>" in match.group(0) else match.group(0)
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning("Hypothesis JSON parse failed: %s. Raw:\n%s", e, raw_text if 'raw_text' in locals() else match.group(0))
            print(f"[code_agent] JSON parse error: {e}")
            return None

    def _parse_hypotheses_blocks(self, text: str, require_code: bool = True) -> list[dict] | None:
        """Parse hypotheses from markdown blocks: ### H1: desc + ```python code```.

        If require_code is False, hypotheses without code blocks are still returned
        (used for Phase 1 idea-only parsing).
        """
        # Find all hypothesis headers
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
            # Find the code block that follows this header
            header_end = header.end()
            next_header_start = headers[i + 1].start() if i + 1 < len(headers) else len(text)

            code = None
            for c in codes:
                if header_end <= c.start() < next_header_start:
                    code = c.group(1).strip()
                    break

            if require_code and not code:
                continue

            # Extract fields from text between header and code
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
        """Quick exec + preprocess on a tiny sample. Returns error string or None if OK."""
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
                        f"Chunk has invalid doc_id '{c.doc_id}' — "
                        f"chunk.doc_id must exactly match one of the input document doc_ids. "
                        f"Valid example: '{next(iter(valid_doc_ids))}'"
                    )
            return None
        except Exception as e:
            logger.exception("Validation of generated code raised")
            logger.debug("Offending code:\n%s", code)
            return str(e)

    def test_hypothesis(
        self,
        hypothesis: Hypothesis,
        documents: list,
        queries: list,
        current_code: str,
        client,
    ) -> HypothesisResult:
        """Test a single hypothesis by building a temp index and running subset eval."""
        from .eval_utils import load_preprocessor_from_code, run_subset_eval

        result = HypothesisResult(hypothesis=hypothesis)
        index_name = f"hyp_{hypothesis.id}"

        # Validate code on tiny sample before full preprocessing
        validation_error = self._validate_code(hypothesis.code, documents)
        if validation_error:
            result.error = f"Validation failed: {validation_error}"
            result.notes = f"Code rejected at validation: {validation_error[:120]}"
            logger.error(
                "Hypothesis %s validation failed: %s",
                hypothesis.id, validation_error,
            )
            logger.debug("Hypothesis %s code:\n%s", hypothesis.id, hypothesis.code)
            print(f"[code_agent] {hypothesis.id} validation error: {validation_error[:120]}")
            return result

        preprocess_timeout = 1200  # seconds — full corpus can be 200K+ docs

        try:
            # Always test on all queries for reliable delta measurement.
            test_queries = queries

            # Load hypothesis preprocessor and run with timeout
            preprocessor = load_preprocessor_from_code(hypothesis.code)
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = ex.submit(preprocessor.preprocess, documents)
            try:
                chunks = future.result(timeout=preprocess_timeout)
            except concurrent.futures.TimeoutError:
                # Avoid blocking indefinitely on shutdown if preprocess() is still running.
                ex.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"preprocess() timed out after {preprocess_timeout}s")
            else:
                # Normal completion: wait for worker thread to finish cleanly.
                ex.shutdown(wait=True)

            # Build hypothesis index on server
            client.build_index(index_name, chunks, persist=False)

            # Run subset eval on hypothesis index
            hyp_eval = run_subset_eval(index_name, test_queries, client)

            # Run subset eval on current index for comparison
            current_eval = run_subset_eval("current", test_queries, client)

            # recall@100 for proven decision (more granular)
            result.hypothesis_recall_100 = hyp_eval.recall_at_100
            result.baseline_recall_100 = current_eval.recall_at_100
            result.delta_recall_100 = hyp_eval.recall_at_100 - current_eval.recall_at_100

            # recall@10 and nDCG@10 for info
            result.hypothesis_recall_10 = hyp_eval.recall_at_10
            result.baseline_recall_10 = current_eval.recall_at_10
            result.delta_recall_10 = hyp_eval.recall_at_10 - current_eval.recall_at_10
            result.delta_ndcg_10 = hyp_eval.ndcg_at_10 - current_eval.ndcg_at_10

            # Proven if recall@100 improves (more granular than @10)
            result.proven = result.delta_recall_100 >= self._recall_threshold

            # Build per-query improvement/regression lists
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
            # Clean up hypothesis index
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

        # Build proven hypotheses section
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

        # Extract python code block
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if not match:
            print("[code_agent] No python block in final code response")
            return None

        code = match.group(1).strip()

        # Validate it has class Preprocessor
        if "class Preprocessor" not in code:
            print("[code_agent] Final code missing 'class Preprocessor' - rejected")
            return None

        return code
