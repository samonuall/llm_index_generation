"""
agent.py – LiteLLM Agent: iterative preprocessing agent backed by any litellm-compatible model.

Reads model name and hyperparameters from config.yaml in the same directory.
The API key is read from LITE_LLM_KEY in the project-root .env file.
"""

from __future__ import annotations

import os
import re
import pathlib
import datetime

import yaml
from dotenv import load_dotenv
from litellm import completion

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent

from ..agent_runner import AgentRunner


def _load_config() -> dict:
    config_path = _AGENT_DIR / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class LiteLLMAgent(AgentRunner):
    agent_name = "lite_llm_agent"

    def __init__(self, include_query_text: bool = True) -> None:
        config = _load_config()
        self._model: str = config["model"]
        self._temperature: float = float(config.get("temperature", 0.7))
        self._api_key: str = os.environ.get("LITE_LLM_KEY", "")

        context_dir = _AGENT_DIR / "context"
        dataset_info = (_AGENT_DIR.parent / "CONTEXT.md").read_text(encoding="utf-8")
        template = (context_dir / "SYSTEM_INSTRUCTION.md").read_text(encoding="utf-8")
        self._system_instruction = template.replace("{dataset_info}", dataset_info)
        self._include_query_text = include_query_text

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        preprocess_path = _AGENT_DIR / "preprocess.py"
        current_code = preprocess_path.read_text(encoding="utf-8").strip()

        if not current_code or eval_results is None:
            return self._system_instruction

        k = eval_results["top_k"]

        query_results = eval_results.get("query_results", [])
        misses = [r for r in query_results if not r["hit"]]
        hits = [r for r in query_results if r["hit"]]

        # Format missed queries
        missed_lines = []
        for r in misses:
            if self._include_query_text:
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
            if self._include_query_text:
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
            f"{self._system_instruction}\n\n"
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

    def call_llm(self, prompt: str, iteration: int) -> None:
        logs_dir = _AGENT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / f"iteration_{iteration + 1}.log"

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        header = f"=== Iteration {iteration + 1} | {timestamp} | model={self._model} ===\n\n"

        # Log the prompt
        prompt_log_path = logs_dir / f"iteration_{iteration + 1}_prompt.log"
        prompt_log_path.write_text(header + prompt, encoding="utf-8")

        print(f"[lite_llm_agent] Calling {self._model} (iteration {iteration + 1}) ...")

        response = completion(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            api_key=self._api_key,
            api_base="https://thekeymaker.umass.edu/"
        )

        try:
            text: str = response.choices[0].message.content or ""
        except Exception as exc:
            text = ""
            print(f"[lite_llm_agent] Failed to extract response text: {exc}")

        finish_reason = None
        try:
            finish_reason = response.choices[0].finish_reason
        except Exception:
            pass

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(header)
            log_file.write(f"--- finish_reason: {finish_reason}\n\n")
            log_file.write(text)

        if not text:
            print(f"[lite_llm_agent] Empty response from API. finish_reason={finish_reason}")
        else:
            print(text)

        # Extract the first ```python ... ``` block and write it to preprocess.py
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1).rstrip()
            preprocess_path = _AGENT_DIR / "preprocess.py"
            preprocess_path.write_text(code + "\n", encoding="utf-8")
            print(f"[lite_llm_agent] preprocess.py updated ({len(code.splitlines())} lines).")
        else:
            print("[lite_llm_agent] Warning: no ```python``` block found in response — preprocess.py unchanged.")

        print(f"[lite_llm_agent] Iteration {iteration + 1} complete. Log: {log_path}")
