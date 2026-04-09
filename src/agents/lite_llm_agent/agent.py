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
from ..utils.prompt_utils import build_eval_prompt


def _load_config() -> dict:
    config_path = _AGENT_DIR / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class LiteLLMAgent(AgentRunner):
    agent_name = "lite_llm_agent"

    def __init__(self, include_query_text: bool = True, test_mode: bool = False) -> None:
        config = _load_config()
        self._model: str = config["model"]
        self._temperature: float = float(config.get("temperature", 1.0))
        self._api_key: str = os.environ.get("LITELLM_API_KEY", "")

        context_dir = _AGENT_DIR / "context"
        dataset_info = (_AGENT_DIR.parent / "CONTEXT.md").read_text(encoding="utf-8")
        template = (context_dir / "SYSTEM_INSTRUCTION.md").read_text(encoding="utf-8")
        self._system_instruction_template = template.replace("{dataset_info}", dataset_info)
        self._system_instruction = self._system_instruction_template.replace("{baseline_info}", "(not yet computed)")
        self._include_query_text = include_query_text
        self._test_mode = test_mode

    def on_baseline_complete(self, baseline_results: dict) -> None:
        """Format baseline results for display - supports both recall@100 and recall@1000."""
        # Get recall@100 and recall@1000 if available
        recall_100 = baseline_results.get('recall_at_100', baseline_results.get('recall_at_k', 0))
        recall_1000 = baseline_results.get('recall_at_1000', 0)
        ndcg_k = baseline_results.get("ndcg_k", 10)
        ndcg = baseline_results.get('ndcg', 0)
        n_chunks = baseline_results.get('n_chunks', 0)
        
        info = (
            f"- Recall@100:  {recall_100:.4f}\n"
            f"- Recall@1000: {recall_1000:.4f}\n"
            f"- nDCG@{ndcg_k}:     {ndcg:.4f}\n"
            f"- Chunks:      {n_chunks}"
        )
        self._system_instruction = self._system_instruction_template.replace("{baseline_info}", info)

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        current_code = (_AGENT_DIR / "preprocess.py").read_text(encoding="utf-8").strip()
        return build_eval_prompt(
            current_code=current_code,
            eval_results=eval_results,
            include_query_text=self._include_query_text,
        )

    def _completion(self, prompt: str):
        if self._test_mode:
            return completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temperature,
                api_key=self._api_key,
                api_base="https://thekeymaker.umass.edu/",
                mock_response="---This is a test response.---"
            )
        else:
            return completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temperature,
                api_key=self._api_key,
                api_base="https://thekeymaker.umass.edu/"
            )
    
    def call_llm(self, prompt: str, iteration: int) -> None:
        logs_dir = _AGENT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create iterations directory
        iterations_dir = _AGENT_DIR / "iterations"
        iterations_dir.mkdir(exist_ok=True)
        
        log_path = logs_dir / f"iteration_{iteration + 1}.log"
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        header = f"=== Iteration {iteration + 1} | {timestamp} | model={self._model} ===\n\n"

        # Log the prompt
        prompt_log_path = logs_dir / f"iteration_{iteration + 1}_prompt.log"
        prompt_log_path.write_text(header + self._system_instruction + "\n\n" + prompt, encoding="utf-8")

        print(f"[lite_llm_agent] Calling {self._model} (iteration {iteration + 1}) ...")

        response = self._completion(prompt)

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

        # Extract the first ```python ... ``` block
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1).rstrip()
            
            # Fix name field to ensure consistency
            code = re.sub(
                r'name\s*=\s*"[^"]*"',
                'name = "lite_llm_agent"',
                code
            )
            
            # Save to iteration-specific file
            iteration_file = iterations_dir / f"preprocess_iter_{iteration + 1}.py"
            iteration_file.write_text(code + "\n", encoding="utf-8")
            print(f"[lite_llm_agent] Saved iteration {iteration + 1} to: {iteration_file}")
            
            # Also update the main preprocess.py (for compatibility)
            preprocess_path = _AGENT_DIR / "preprocess.py"
            preprocess_path.write_text(code + "\n", encoding="utf-8")
            print(f"[lite_llm_agent] Updated preprocess.py ({len(code.splitlines())} lines).")
        else:
            print("[lite_llm_agent] Warning: no ```python``` block found in response — preprocess.py unchanged.")

        print(f"[lite_llm_agent] Iteration {iteration + 1} complete. Log: {log_path}")