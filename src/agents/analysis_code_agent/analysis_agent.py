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
    def __init__(self, config: dict, tracker=None) -> None:
        self._tracker = tracker
        self._model = config.get("analysis_model", "openai/gpt-4o-mini")
        self._temperature = config.get("analysis_temperature", 0.3)
        self._max_turns = config.get("analysis_max_turns", 8)
        self._bash_timeout = config.get("bash_timeout_seconds", 30)
        self._api_key = os.environ.get("LITE_LLM_KEY", os.environ.get("LITELLM_API_KEY", ""))
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
        split: str = "tip_of_the_tongue",
        journal_summary: str | None = None,
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
            split=split,
            journal_summary=journal_summary,
        )

        min_bash_turns = 3  # require at least this many bash turns before accepting a summary

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": initial_msg},
        ]

        bash_turns_completed = 0

        for turn in range(self._max_turns):
            # Call LLM
            text = self._call_llm(messages, turn)
            if text is None:
                break

            messages.append({"role": "assistant", "content": text})

            # Check for bash blocks
            bash_match = re.search(r"<bash>(.*?)</bash>", text, re.DOTALL)
            if not bash_match:
                if bash_turns_completed < min_bash_turns:
                    # Haven't done enough bash yet — nudge the agent to investigate
                    nudge = (
                        f"You haven't run enough bash commands yet ({bash_turns_completed}/{min_bash_turns} done). "
                        f"You MUST investigate the actual data before summarizing. "
                        f"Pick a specific failing query from the list and run the curl command to see what BM25 retrieved for it. "
                        f"Then look at the gold document. Do NOT summarize until you have done this {min_bash_turns} times."
                    )
                    messages.append({"role": "user", "content": nudge})
                    continue
                else:
                    # Enough bash done — this is the final summary
                    break

            # Execute bash command
            cmd = bash_match.group(1).strip()
            bash_output = self._run_bash(cmd)
            bash_turns_completed += 1
            messages.append({"role": "user", "content": bash_output})

        # Always request a dedicated final summary call.
        # This avoids accidentally treating a planning/status assistant message
        # as the final analysis output.
        summary = self._request_summary(messages)

        return AnalysisResult(
            summary=summary,
            turns=len([m for m in messages if m["role"] == "assistant"]),
            conversation=messages,
        )

    def _call_llm(self, messages: list[dict], turn: int) -> str | None:
        """Call LLM with retry logic. Returns response text or None on failure."""
        def _do_call(msgs):
            t0 = time.time()
            response = completion(
                model=self._model,
                messages=msgs,
                temperature=self._temperature,
                api_key=self._api_key,
                api_base=self._api_base,
            )
            if self._tracker:
                self._tracker.record_llm_call(response, time.time() - t0, agent="analysis")
            return response.choices[0].message.content or ""

        try:
            return _do_call(messages)
        except Exception as e:
            print(f"[analysis_agent] LLM call failed (turn {turn}): {e}")
            # If content policy violation, strip bash outputs from history and retry
            if "ContentPolicyViolation" in type(e).__name__ or "content_policy" in str(e).lower() or "content management policy" in str(e).lower():
                sanitized = self._sanitize_messages(messages)
                # Update the original messages history in place so future turns use sanitized content
                messages[:] = sanitized
                try:
                    return _do_call(sanitized)
                except Exception as e2:
                    print(f"[analysis_agent] LLM retry (sanitized) failed: {e2}")
                    return None
            time.sleep(5)
            try:
                return _do_call(messages)
            except Exception as e2:
                print(f"[analysis_agent] LLM retry failed: {e2}")
                return None

    def _sanitize_messages(self, messages: list[dict]) -> list[dict]:
        """Return messages with bash outputs truncated to 300 chars to avoid content policy violations."""
        sanitized = []
        for m in messages:
            if m["role"] == "user" and "[BASH OUTPUT]" in m.get("content", ""):
                content = m["content"]
                if len(content) > 400:
                    content = content[:400] + "\n... [truncated to avoid content policy filter]"
                sanitized.append({**m, "content": content})
            else:
                sanitized.append(m)
        return sanitized

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
            if len(combined) > 2000:
                combined = combined[:2000] + "\n... [truncated]"
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
                "Now provide the FINAL analysis summary. This is a dedicated summary step. "
                "No more bash commands. Provide a structured summary of failure patterns and "
                "recommendations with concrete evidence (query IDs/doc IDs)."
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
            # Guardrail: if model still emits bash, retry once with stricter instruction.
            if re.search(r"<bash>.*?</bash>", summary, re.DOTALL):
                messages.append({"role": "assistant", "content": summary})
                messages.append({
                    "role": "user",
                    "content": (
                        "Do not include any <bash> blocks. Output only the final written summary."
                    ),
                })
                response2 = completion(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    api_key=self._api_key,
                    api_base=self._api_base,
                )
                summary = response2.choices[0].message.content or "No summary generated."

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
        split: str = "tip_of_the_tongue",
        journal_summary: str | None = None,
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

        data_dir = _PROJECT_ROOT / "data" / split
        server = self._server_url

        journal_section = f"\n{journal_summary}\n" if journal_summary else ""

        return (
            f"{journal_section}"
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
