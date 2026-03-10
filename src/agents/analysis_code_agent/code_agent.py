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
        self, analysis_summary: str, current_code: str, n: int = 4
    ) -> list[Hypothesis]:
        """Single LLM call to generate N hypotheses. Output JSON inside <hypotheses>...</hypotheses> tags."""

        dataset_info = (_AGENT_DIR.parent / "CONTEXT.md").read_text(encoding="utf-8")

        prompt = f"""## Analysis Summary
{analysis_summary}

## Current preprocess.py
```python
{current_code}
```

## Dataset Info
{dataset_info}

Generate exactly {n} hypotheses to improve the preprocessing code.
Each hypothesis must be a complete, working preprocess.py implementation.

Output your hypotheses as a JSON array inside <hypotheses>...</hypotheses> tags.
Each hypothesis object must have these fields:
- "id": "H1", "H2", etc.
- "description": one-line summary of the change
- "rationale": why this should help based on the analysis
- "code": complete Python code for preprocess.py (must define class Preprocessor(BasePreprocessor) with preprocess method)
- "query_ids_to_test": list of query_ids most likely affected (from the analysis), at least 3
- "falsifying_condition": what would disprove this hypothesis

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

        # Parse hypotheses from <hypotheses>...</hypotheses> tags
        match = re.search(r"<hypotheses>(.*?)</hypotheses>", text, re.DOTALL)
        if not match:
            # Try parsing the whole thing as JSON
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                print("[code_agent] No hypotheses found in response")
                return []

        try:
            raw_text = match.group(1) if "<hypotheses>" in (match.group(0) or "") else match.group(0)
            raw = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print(f"[code_agent] JSON parse error: {e}. Retrying...")
            # Retry once with error correction
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"JSON parse error: {e}. Please output the hypotheses again "
                        "as valid JSON inside <hypotheses>...</hypotheses> tags."
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
                match2 = re.search(r"<hypotheses>(.*?)</hypotheses>", text2, re.DOTALL)
                if match2:
                    raw = json.loads(match2.group(1))
                else:
                    print("[code_agent] Retry also failed. Returning empty.")
                    return []
            except Exception:
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
            # Filter queries to test
            test_queries = queries
            if hypothesis.query_ids_to_test:
                test_qids = set(hypothesis.query_ids_to_test)
                filtered = [q for q in queries if q.query_id in test_qids]
                if len(filtered) >= 3:
                    test_queries = filtered
                # If fewer than 3 matching queries, use all

            # Load hypothesis preprocessor from code string
            preprocessor = load_preprocessor_from_code(hypothesis.code)
            chunks = preprocessor.preprocess(documents)

            # Build hypothesis index on server
            client.build_index(index_name, chunks, persist=False)

            # Run subset eval on hypothesis index
            hyp_eval = run_subset_eval(index_name, test_queries, client)

            # Run subset eval on current index for comparison
            current_eval = run_subset_eval("current", test_queries, client)

            result.hypothesis_recall_10 = hyp_eval.recall_at_10
            result.baseline_recall_10 = current_eval.recall_at_10
            result.delta_recall_10 = hyp_eval.recall_at_10 - current_eval.recall_at_10
            result.delta_ndcg_10 = hyp_eval.ndcg_at_10 - current_eval.ndcg_at_10
            result.proven = result.delta_recall_10 >= self._recall_threshold

            # Build notes
            improved = sum(
                1
                for h_q, c_q in zip(hyp_eval.per_query, current_eval.per_query)
                if h_q.hit_at_10 and not c_q.hit_at_10
            )
            regressed = sum(
                1
                for h_q, c_q in zip(hyp_eval.per_query, current_eval.per_query)
                if c_q.hit_at_10 and not h_q.hit_at_10
            )
            result.notes = (
                f"Improved {improved}, regressed {regressed} of {len(test_queries)} test queries"
            )

            print(
                f"[code_agent] {hypothesis.id}: delta_recall@10={result.delta_recall_10:+.4f} "
                f"delta_ndcg@10={result.delta_ndcg_10:+.4f} proven={result.proven}"
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
