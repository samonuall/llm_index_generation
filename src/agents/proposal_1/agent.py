"""
agent.py – Proposal 1: Gemini-backed iterative preprocessing agent.

Uses the `gemini` CLI to iteratively improve preprocess.py based on eval
feedback from the static BM25 harness.
"""

from __future__ import annotations

import subprocess
import sys
import pathlib
import datetime

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
_AGENT_DIR = pathlib.Path(__file__).parent

# Make src/agents importable so AgentRunner resolves
_SRC_AGENTS_DIR = _PROJECT_ROOT / "src" / "agents"
if str(_SRC_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_AGENTS_DIR))

from agent_runner import AgentRunner  # type: ignore


class Proposal1Agent(AgentRunner):
    agent_name = "proposal_1"

    def __init__(self) -> None:
        context_dir = _AGENT_DIR / "context"
        dataset_info = (context_dir / "DATASET_INFO.md").read_text(encoding="utf-8")
        template = (context_dir / "SYSTEM_INSTRUCTION.md").read_text(encoding="utf-8")
        self._system_instruction = template.replace("{dataset_info}", dataset_info)

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        preprocess_path = _AGENT_DIR / "preprocess.py"
        current_code = preprocess_path.read_text(encoding="utf-8").strip()

        if not current_code or eval_results is None:
            return self._system_instruction

        k = eval_results["top_k"]
        return (
            f"{self._system_instruction}\n\n"
            f"## Current implementation\n```python\n{current_code}\n```\n\n"
            f"## Last eval results (top-{k})\n"
            f"- Recall@{k}: {eval_results['recall_at_k']:.4f}\n"
            f"- MRR: {eval_results['mrr']:.4f}\n"
            f"- Chunks indexed: {eval_results['n_chunks']}\n\n"
            "Improve the implementation to increase Recall and MRR."
        )

    def call_llm(self, prompt: str, iteration: int) -> None:
        logs_dir = _AGENT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / f"iteration_{iteration + 1}.log"

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        header = f"=== Iteration {iteration + 1} | {timestamp} ===\n\n"

        print(f"[proposal_1] Calling Gemini (iteration {iteration + 1}) ...")

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(header)
            log_file.flush()

            process = subprocess.Popen(
                [
                    "gemini",
                    "-y",
                    "-m", "gemini-3-flash-preview",
                    "-p", prompt,
                ],
                cwd=str(_PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )

            for line in process.stdout:  # type: ignore[union-attr]
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()

            process.wait()

        if process.returncode != 0:
            print(
                f"[proposal_1] Warning: gemini exited with code {process.returncode}. "
                f"See {log_path}"
            )
        else:
            print(f"[proposal_1] Iteration {iteration + 1} complete. Log: {log_path}")
