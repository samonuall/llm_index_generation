"""
code_agent.py - Hypothesis generation, testing, and final code synthesis.
"""
from __future__ import annotations

import os
import re
import json
import pathlib
from dataclasses import dataclass, field

from dotenv import load_dotenv
from litellm import completion

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent


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


class CodeAgent:
    def __init__(self, config: dict) -> None:
        self._model = config.get("code_model", "openai/gpt-4o")
        self._temperature = config.get("code_temperature", 0.7)
        self._api_key = os.environ.get("LITELLM_API_KEY", "")
        self._api_base = config.get("api_base", "https://thekeymaker.umass.edu/")
        self._recall_threshold = config.get("recall_improvement_threshold", 0.05)
        self._max_hypotheses = config.get("max_hypotheses", 4)

        # Load system prompt
        system_path = _AGENT_DIR / "context" / "CODE_SYSTEM.md"
        self._system_prompt = system_path.read_text(encoding="utf-8")

    def generate_hypotheses(
        self,
        analysis_summary: str,
        current_code: str,
        n: int = 4,
        past_hypotheses: list[dict] | None = None,
    ) -> list[Hypothesis]:
        """Single LLM call to generate N hypotheses. Output JSON inside <hypotheses>...</hypotheses> tags."""

        dataset_info = (_AGENT_DIR.parent / "CONTEXT.md").read_text(encoding="utf-8")

        # Build past attempts section
        past_section = ""
        if past_hypotheses:
            lines = []
            for ph in past_hypotheses:
                lines.append(
                    f"- **{ph['id']}: {ph['description']}** → "
                    f"delta_recall@100={ph['delta_recall_100']:+.4f}, "
                    f"delta_ndcg@10={ph['delta_ndcg_10']:+.4f}, "
                    f"proven={ph['proven']}. {ph.get('notes', '')}"
                )
            past_section = (
                "\n## Previously Tested Hypotheses (DO NOT repeat these)\n"
                + "\n".join(lines)
                + "\n\nLearn from what worked and what didn't. "
                "Do NOT regenerate hypotheses that are similar to failed ones above.\n"
            )

        prompt = f"""## Analysis Summary
{analysis_summary}

## Current preprocess.py
```python
{current_code}
```

## Dataset Info
{dataset_info}
{past_section}
Generate exactly {n} hypotheses to improve the preprocessing code.
Each hypothesis must be a complete, working preprocess.py implementation.

IMPORTANT NOTES:
- The documents in this dataset have EMPTY metadata dicts (no title, no aliases). Do NOT rely on doc.metadata for anything.
- The BM25 tokenizer lowercases and stems text. Stopword removal is NOT done by the preprocessor — it's handled by BM25.
- Naive paragraph splitting hurts BM25 because short chunks lose term co-occurrence. If chunking, use overlapping windows or keep chunks substantial (200+ words).

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
            response = completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
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
                raw = self._parse_hypotheses_blocks(text2)
                if raw is None:
                    raw = self._parse_hypotheses_json(text2)
            except Exception as e:
                print(f"[code_agent] Retry failed: {e}")

        if not raw:
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
            print(f"[code_agent] JSON parse error: {e}")
            return None

    def _parse_hypotheses_blocks(self, text: str) -> list[dict] | None:
        """Parse hypotheses from markdown blocks: ### H1: desc + ```python code```."""
        # Find all hypothesis headers
        header_pattern = r"###\s+(H\d+)\s*:\s*(.+?)(?:\n|$)"
        code_pattern = r"```python\s*\n(.*?)```"

        headers = list(re.finditer(header_pattern, text))
        codes = list(re.finditer(code_pattern, text, re.DOTALL))

        if not headers or not codes:
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

            if not code:
                continue

            # Extract rationale and query_ids from text between header and code
            between = text[header_end:next_header_start]
            rationale_match = re.search(r"Rationale:\s*(.+?)(?:\n|$)", between)
            qids_match = re.search(r"Query IDs?:\s*(.+?)(?:\n|$)", between)
            falsify_match = re.search(r"Falsif(?:ying|ication):\s*(.+?)(?:\n|$)", between)

            query_ids = []
            if qids_match:
                # Parse comma-separated query IDs
                query_ids = [q.strip().strip("[]\"'") for q in qids_match.group(1).split(",")]

            results.append({
                "id": h_id,
                "description": desc,
                "rationale": rationale_match.group(1).strip() if rationale_match else "",
                "code": code,
                "query_ids_to_test": query_ids,
                "falsifying_condition": falsify_match.group(1).strip() if falsify_match else "",
            })

        return results if results else None

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

        try:
            # Always test on all queries for reliable delta measurement.
            # The query_ids_to_test are logged but not used for filtering,
            # since testing only on failure cases yields 0/0 comparisons.
            test_queries = queries

            # Load hypothesis preprocessor from code string
            preprocessor = load_preprocessor_from_code(hypothesis.code)
            chunks = preprocessor.preprocess(documents)

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

            # Build notes
            improved_100 = sum(
                1
                for h_q, c_q in zip(hyp_eval.per_query, current_eval.per_query)
                if h_q.hit_at_100 and not c_q.hit_at_100
            )
            regressed_100 = sum(
                1
                for h_q, c_q in zip(hyp_eval.per_query, current_eval.per_query)
                if c_q.hit_at_100 and not h_q.hit_at_100
            )
            result.notes = (
                f"@100: improved {improved_100}, regressed {regressed_100} of "
                f"{len(test_queries)} queries"
            )

            print(
                f"[code_agent] {hypothesis.id}: "
                f"delta_recall@100={result.delta_recall_100:+.4f} "
                f"delta_recall@10={result.delta_recall_10:+.4f} "
                f"delta_ndcg@10={result.delta_ndcg_10:+.4f} "
                f"proven={result.proven}"
            )

        except Exception as e:
            result.error = str(e)
            result.notes = f"Error: {e}"
            print(f"[code_agent] {hypothesis.id} error: {e}")

        finally:
            # Clean up hypothesis index
            try:
                client.delete_index(index_name)
            except Exception:
                pass

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
            response = completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
            )
            text = response.choices[0].message.content or ""
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
