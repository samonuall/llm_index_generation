"""
agent.py – Gemini SDK: Gemini-backed iterative preprocessing agent.

Uses the google-genai Python SDK to iteratively improve preprocess.py based
on eval feedback from the static BM25 harness.
"""

from __future__ import annotations

import re
import pathlib
import datetime

from dotenv import load_dotenv
from google import genai

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent

from ..agent_runner import AgentRunner
from ..utils.prompt_utils import build_eval_prompt

_MODEL = "gemini-2.0-flash"


class GeminiSdkAgent(AgentRunner):
    agent_name = "gemini_sdk"

    def __init__(self, include_query_text: bool = True) -> None:
        context_dir = _AGENT_DIR / "context"
        dataset_info = (_AGENT_DIR.parent / "CONTEXT.md").read_text(encoding="utf-8")
        template = (context_dir / "SYSTEM_INSTRUCTION.md").read_text(encoding="utf-8")
        self._system_instruction_template = template.replace("{dataset_info}", dataset_info)
        self._system_instruction = self._system_instruction_template.replace("{baseline_info}", "(not yet computed)")
        self._client = genai.Client()
        self._include_query_text = include_query_text

    def on_baseline_complete(self, baseline_results: dict) -> None:
        k = baseline_results["top_k"]
        ndcg_k = baseline_results.get("ndcg_k", 10)
        info = (
            f"- Recall@{k}:    {baseline_results['recall_at_k']:.4f}\n"
            f"- nDCG@{ndcg_k}:    {baseline_results['ndcg']:.4f}\n"
            f"- Chunks:       {baseline_results['n_chunks']}"
        )
        self._system_instruction = self._system_instruction_template.replace("{baseline_info}", info)

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        current_code = (_AGENT_DIR / "preprocess.py").read_text(encoding="utf-8").strip()
        return build_eval_prompt(
            current_code=current_code,
            eval_results=eval_results,
            include_query_text=self._include_query_text,
        )

    def call_llm(self, prompt: str, iteration: int) -> None:
        logs_dir = _AGENT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / f"iteration_{iteration + 1}.log"

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        header = f"=== Iteration {iteration + 1} | {timestamp} ===\n\n"

        # Log the prompt
        prompt_log_path = logs_dir / f"iteration_{iteration + 1}_prompt.log"
        prompt_log_path.write_text(header + prompt, encoding="utf-8")

        print(f"[gemini_sdk] Calling Gemini (iteration {iteration + 1}) ...")

        response = self._client.models.generate_content(
            model=_MODEL,
            contents=self._system_instruction + "\n\n" + prompt,
        )

        # Collect response metadata for logging
        finish_reason = None
        safety_ratings = []
        try:
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                safety_ratings = response.candidates[0].safety_ratings or []
        except Exception:
            pass

        prompt_feedback = getattr(response, "prompt_feedback", None)

        try:
            text = response.text or ""
        except Exception as exc:
            text = ""
            print(f"[gemini_sdk] response.text raised: {exc}")

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(header)
            log_file.write(f"--- finish_reason: {finish_reason}\n")
            if safety_ratings:
                log_file.write(f"--- safety_ratings: {safety_ratings}\n")
            if prompt_feedback:
                log_file.write(f"--- prompt_feedback: {prompt_feedback}\n")
            log_file.write("\n")
            log_file.write(text)

        if not text:
            print(f"[gemini_sdk] Empty response from API. finish_reason={finish_reason}")
            if prompt_feedback:
                print(f"[gemini_sdk] prompt_feedback: {prompt_feedback}")
        else:
            print(text)

        # Extract the first ```python ... ``` block and write it to preprocess.py
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1).rstrip()
            preprocess_path = _AGENT_DIR / "preprocess.py"
            preprocess_path.write_text(code + "\n", encoding="utf-8")
            print(f"[gemini_sdk] preprocess.py updated ({len(code.splitlines())} lines).")
        else:
            print("[gemini_sdk] Warning: no ```python``` block found in response — preprocess.py unchanged.")

        print(f"[gemini_sdk] Iteration {iteration + 1} complete. Log: {log_path}")
