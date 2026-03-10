"""
analysis_agent.py - Multi-turn bash-loop analysis agent.

Uses LiteLLM to call a model that can run bash commands to investigate
BM25 retrieval failures and produce a structured analysis summary.
"""
from __future__ import annotations

import os
import re
import subprocess
import pathlib
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from litellm import completion

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent


@dataclass
class AnalysisResult:
    summary: str
    turns: int
    conversation: list[dict]  # full message history for logging


class AnalysisAgent:
    def __init__(self, config: dict) -> None:
        self._model = config.get("analysis_model", "openai/gpt-4o-mini")
        self._temperature = config.get("analysis_temperature", 0.3)
        self._max_turns = config.get("analysis_max_turns", 8)
        self._bash_timeout = config.get("bash_timeout_seconds", 30)
        self._api_key = os.environ.get("LITELLM_API_KEY", "")
        self._api_base = config.get("api_base", "https://thekeymaker.umass.edu/")
        self._server_url = f"http://localhost:{config.get('server_port', 8765)}"

        # Load system prompt
        system_path = _AGENT_DIR / "context" / "ANALYSIS_SYSTEM.md"
        self._system_prompt = system_path.read_text(encoding="utf-8")

    def analyze(
        self,
        eval_results: dict,
        baseline_results: dict,
        current_code: str,
        documents: list,
        queries: list,
        client,
    ) -> AnalysisResult:
        """Run multi-turn analysis loop. Returns AnalysisResult with summary."""

        # Build candidate analysis targets
        candidates = self._build_candidates(eval_results, baseline_results)

        # Build initial user message
        initial_msg = self._build_initial_context(
            eval_results=eval_results,
            baseline_results=baseline_results,
            current_code=current_code,
            candidates=candidates,
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": initial_msg},
        ]

        for turn in range(self._max_turns):
            # Call LLM
            text = self._call_llm(messages, turn)
            if text is None:
                break

            messages.append({"role": "assistant", "content": text})

            # Check for bash blocks
            bash_match = re.search(r"<bash>(.*?)</bash>", text, re.DOTALL)
            if not bash_match:
                # No bash block - this is the final summary
                break

            # Execute bash command
            cmd = bash_match.group(1).strip()
            bash_output = self._run_bash(cmd)
            messages.append({"role": "user", "content": bash_output})

        # Extract summary from last assistant message
        summary = self._extract_summary(messages)

        # If no clean summary found, ask for one
        if not summary:
            summary = self._request_summary(messages)

        return AnalysisResult(
            summary=summary,
            turns=len([m for m in messages if m["role"] == "assistant"]),
            conversation=messages,
        )

    def _call_llm(self, messages: list[dict], turn: int) -> str | None:
        """Call LLM with retry logic. Returns response text or None on failure."""
        try:
            response = completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[analysis_agent] LLM call failed (turn {turn}): {e}")
            time.sleep(5)
            try:
                response = completion(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    api_key=self._api_key,
                    api_base=self._api_base,
                )
                return response.choices[0].message.content or ""
            except Exception as e2:
                print(f"[analysis_agent] LLM retry failed: {e2}")
                return None

    def _run_bash(self, cmd: str) -> str:
        """Execute a bash command and return formatted output."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._bash_timeout,
                cwd=str(_PROJECT_ROOT),
            )
            output = f"[BASH EXIT CODE: {result.returncode}]\n"
            combined = (result.stdout or "") + (result.stderr or "")
            if len(combined) > 4000:
                combined = combined[:4000] + "\n... [truncated]"
            output += f"[BASH OUTPUT]:\n{combined}"
        except subprocess.TimeoutExpired:
            output = f"[bash: timeout after {self._bash_timeout}s]"
        except Exception as e:
            output = f"[bash: error: {e}]"
        return output

    def _extract_summary(self, messages: list[dict]) -> str:
        """Extract the final summary from the conversation (last assistant msg without bash)."""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                if not re.search(r"<bash>", msg["content"]):
                    return msg["content"]
        return ""

    def _request_summary(self, messages: list[dict]) -> str:
        """Ask the LLM for a final summary when none was produced naturally."""
        messages.append({
            "role": "user",
            "content": (
                "Please summarize your findings now. No more bash commands. "
                "Provide a structured summary of failure patterns and recommendations."
            ),
        })
        try:
            response = completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
            )
            summary = response.choices[0].message.content or "No summary generated."
            messages.append({"role": "assistant", "content": summary})
            return summary
        except Exception:
            return "Analysis failed to produce summary."

    def _build_candidates(self, eval_results: dict, baseline_results: dict) -> dict:
        """Build candidate analysis targets from eval results."""
        query_results = eval_results.get("query_results", [])
        baseline_qr = {
            r["query_id"]: r for r in baseline_results.get("query_results", [])
        }

        # Failures (regressions): baseline had hit but current doesn't
        failures = []
        for r in query_results:
            baseline_r = baseline_qr.get(r["query_id"])
            if baseline_r and baseline_r.get("hit") and not r.get("hit"):
                failures.append(r)

        # Hard negatives: missed queries, top-10 retrieved that aren't gold
        misses = [r for r in query_results if not r.get("hit")]
        hard_negatives = []
        for r in misses[:5]:
            wrong_docs = [
                doc_id
                for doc_id in r.get("retrieved_doc_ids", [])[:10]
                if doc_id not in r.get("relevant_doc_ids", [])
            ][:3]
            if wrong_docs:
                hard_negatives.append({
                    "query_id": r["query_id"],
                    "query_text": r.get("query_text", ""),
                    "relevant_doc_ids": r.get("relevant_doc_ids", []),
                    "wrong_docs": wrong_docs,
                })

        # Successes: queries with hits, sorted by worst rank first
        hits = [r for r in query_results if r.get("hit")]
        successes = sorted(
            hits, key=lambda x: x.get("rank") or 0, reverse=True
        )[:8]

        return {
            "failures": failures,
            "hard_negatives": hard_negatives,
            "successes": successes,
        }

    def _build_initial_context(
        self,
        eval_results: dict,
        baseline_results: dict,
        current_code: str,
        candidates: dict,
    ) -> str:
        """Build the initial user message with all context."""

        failures = candidates["failures"]
        hard_negatives = candidates["hard_negatives"]
        successes = candidates["successes"]

        # Format failures section
        if failures:
            lines = []
            for r in failures:
                lines.append(
                    f"  - [{r['query_id']}] \"{r.get('query_text', 'N/A')}\""
                )
                lines.append(f"    Expected: {r.get('relevant_doc_ids', [])}")
                lines.append(
                    f"    Retrieved: {r.get('retrieved_doc_ids', [])[:5]}"
                )
            failures_text = (
                f"### Regressions ({len(failures)} queries "
                f"-- baseline hit, current missed):\n" + "\n".join(lines)
            )
        else:
            failures_text = "### Regressions: none"

        # Format hard negatives
        if hard_negatives:
            lines = []
            for hn in hard_negatives:
                lines.append(
                    f"  - [{hn['query_id']}] \"{hn['query_text']}\""
                )
                lines.append(
                    f"    Gold: {hn['relevant_doc_ids']}, "
                    f"Wrong top docs: {hn['wrong_docs']}"
                )
            hn_text = (
                f"### Hard negatives ({len(hard_negatives)} queries):\n"
                + "\n".join(lines)
            )
        else:
            hn_text = "### Hard negatives: none"

        # Format successes
        if successes:
            lines = [
                f"  - [{r['query_id']}] rank={r.get('rank', 'N/A')} "
                f"\"{r.get('query_text', 'N/A')}\""
                for r in successes
            ]
            succ_text = (
                f"### Successes (hit but poor rank, worst first, "
                f"{len(successes)} shown):\n" + "\n".join(lines)
            )
        else:
            succ_text = "### Successes: none"

        # Get metrics
        metrics = eval_results.get("metrics", {})
        recall_100 = metrics.get(
            "recall_at_100", eval_results.get("recall_at_k", 0)
        )
        ndcg_10 = metrics.get("ndcg_at_10", eval_results.get("ndcg", 0))
        baseline_recall = baseline_results.get("recall_at_k", 0)
        baseline_ndcg = baseline_results.get("ndcg", 0)

        data_dir = _PROJECT_ROOT / "data"
        server = self._server_url

        return (
            f"## Current Evaluation\n"
            f"- Recall@100: {recall_100:.4f} (baseline: {baseline_recall:.4f})\n"
            f"- nDCG@10: {ndcg_10:.4f} (baseline: {baseline_ndcg:.4f})\n"
            f"\n"
            f"## Current preprocess.py\n"
            f"```python\n"
            f"{current_code}\n"
            f"```\n"
            f"\n"
            f"## Analysis Targets\n"
            f"{failures_text}\n"
            f"\n"
            f"{hn_text}\n"
            f"\n"
            f"{succ_text}\n"
            f"\n"
            f"## Available Resources\n"
            f"- BM25 Server: {server}\n"
            f"  - Query: `curl -X POST {server}/index/current/retrieve "
            f"-H 'Content-Type: application/json' "
            f"""-d '{{"query": "...", "top_k": 10}}'`\n"""
            f"  - Or via Python: `import requests; "
            f"r = requests.post('{server}/index/current/retrieve', "
            f"json={{'query': '...', 'top_k': 10}}); print(r.json())`\n"
            f"- Data files: `{data_dir}/documents.jsonl`, "
            f"`{data_dir}/queries.jsonl`\n"
            f"\n"
            f"Investigate the failures and patterns above. "
            f"Use <bash>...</bash> blocks to run commands.\n"
            f"When done investigating, provide your final analysis summary "
            f"(no bash block).\n"
        )
