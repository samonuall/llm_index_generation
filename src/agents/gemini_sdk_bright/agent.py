"""
agent.py – BRIGHT-specific Gemini-backed iterative preprocessing agent.

Extends the CRUMB GeminiSdkAgent with:
  - BRIGHT dataset context and system instruction
  - nDCG@10 as the primary feedback metric (not Recall@k + MRR)
  - Per-query excluded_ids awareness in feedback formatting
  - Overrides run_eval() to call test_preprocessing_bright.evaluate_bright()
"""

from __future__ import annotations

import re
import sys
import pathlib
import importlib.util
import datetime

from dotenv import load_dotenv
from google import genai

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent
_SRC_AGENTS_DIR = _PROJECT_ROOT / "src" / "agents"

if str(_SRC_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_AGENTS_DIR))

from agent_runner import AgentRunner  # type: ignore

_MODEL = "gemini-2.5-pro"


class GeminiSdkBrightAgent(AgentRunner):
    agent_name = "gemini_sdk_bright"

    def __init__(self, task: str = "sustainable_living", include_query_text: bool = True) -> None:
        self._task = task
        self._include_query_text = include_query_text

        context_dir = _AGENT_DIR / "context"
        dataset_info = (_SRC_AGENTS_DIR / "CONTEXT_bright.md").read_text(encoding="utf-8")
        template = (context_dir / "SYSTEM_INSTRUCTION_BRIGHT.md").read_text(encoding="utf-8")
        self._system_instruction = template.replace("{dataset_info}", dataset_info)
        self._client = genai.Client()

    # ------------------------------------------------------------------
    # Override run_eval() to use the BRIGHT harness instead of CRUMB
    # ------------------------------------------------------------------

    def run_eval(self) -> dict:
        eval_scripts_dir = _PROJECT_ROOT / "src" / "evaluation" / "scripts"
        eval_dir = _PROJECT_ROOT / "src" / "evaluation"
        src_dir = _PROJECT_ROOT / "src"

        for p in [str(eval_scripts_dir), str(eval_dir), str(src_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from test_preprocessing_bright import evaluate_bright  # type: ignore

        preprocess_path = _PROJECT_ROOT / "src" / "agents" / self.agent_name / "preprocess.py"
        spec = importlib.util.spec_from_file_location(
            f"_agent_{self.agent_name}_preprocess", preprocess_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        preprocessor = module.Preprocessor()
        return evaluate_bright(preprocessor, task=self._task, top_k=10)

    # ------------------------------------------------------------------
    # build_prompt: uses nDCG@10 as the primary metric signal
    # ------------------------------------------------------------------

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        preprocess_path = _AGENT_DIR / "preprocess.py"
        current_code = preprocess_path.read_text(encoding="utf-8").strip()

        if not current_code or eval_results is None:
            return self._system_instruction

        k = eval_results["top_k"]
        query_results = eval_results.get("query_results", [])

        # Split into misses and hits
        misses = [r for r in query_results if not r["hit"]]
        hits = [r for r in query_results if r["hit"]]

        # Format missed queries
        missed_lines = []
        for r in misses:
            if self._include_query_text:
                missed_lines.append(
                    f"  - [{r['query_id']}] \"{r['query_text'][:200]}{'...' if len(r['query_text']) > 200 else ''}\"\n"
                    f"    Expected doc(s): {r['relevant_doc_ids']}\n"
                    f"    Retrieved docs : {r['retrieved_doc_ids']}"
                )
            else:
                missed_lines.append(
                    f"  - [{r['query_id']}]\n"
                    f"    Expected doc(s): {r['relevant_doc_ids']}\n"
                    f"    Retrieved docs : {r['retrieved_doc_ids']}"
                )

        # Format hits, sorted by nDCG (worst first to focus attention)
        hit_lines = []
        for r in sorted(hits, key=lambda x: x.get("ndcg", 0.0)):
            if self._include_query_text:
                hit_lines.append(
                    f"  - [{r['query_id']}] rank={r['rank']} ndcg={r.get('ndcg', 0):.3f}  "
                    f"\"{r['query_text'][:120]}{'...' if len(r['query_text']) > 120 else ''}\""
                )
            else:
                hit_lines.append(
                    f"  - [{r['query_id']}] rank={r['rank']} ndcg={r.get('ndcg', 0):.3f}"
                )

        missed_section = (
            f"### Missed queries ({len(misses)} / {len(query_results)}):\n"
            + ("\n".join(missed_lines) if missed_lines else "  (none)")
        )
        hit_section = (
            f"### Retrieved queries ({len(hits)} / {len(query_results)}) — sorted by nDCG (worst first):\n"
            + ("\n".join(hit_lines) if hit_lines else "  (none)")
        )

        return (
            f"{self._system_instruction}\n\n"
            f"## Current implementation\n```python\n{current_code}\n```\n\n"
            f"## Last eval results (top-{k}, task={self._task})\n"
            f"- **nDCG@{k}**: {eval_results['ndcg_at_k']:.4f}   (paper BM25 baseline ≈ 0.148)\n"
            f"- Recall@{k}: {eval_results['recall_at_k']:.4f}\n"
            f"- MRR: {eval_results['mrr']:.4f}\n"
            f"- Chunks indexed: {eval_results['n_chunks']}\n\n"
            f"## Per-query breakdown\n"
            f"{missed_section}\n\n"
            f"{hit_section}\n\n"
            "Improve the implementation to increase nDCG@10."
        )

    # ------------------------------------------------------------------
    # run: override AgentRunner.run() to add best-of-N tracking
    # ------------------------------------------------------------------

    def run(self, n_loops: int) -> None:
        """Eval-improve loop with best-of-N tracking.

        After every eval, if nDCG@10 improved we save a copy as
        preprocess_best_{task}.py and restore preprocess.py to that
        version at the end of the run.
        """
        preprocess_path = _AGENT_DIR / "preprocess.py"
        self._best_ndcg: float = -1.0
        best_code: str | None = None

        for i in range(n_loops):
            print(f"\n{'#' * 60}")
            print(f"# Iteration {i + 1} / {n_loops}")
            print(f"{'#' * 60}")

            if not preprocess_path.read_text(encoding="utf-8").strip():
                print("[gemini_sdk_bright] preprocess.py is empty, skipping eval.")
                eval_results = None
            else:
                try:
                    eval_results = self.run_eval()
                except Exception as e:
                    print(f"[gemini_sdk_bright] Eval failed (iteration {i + 1}): {e}")
                    eval_results = None

            # Best-of-N: save whenever we see a new high score
            if eval_results is not None:
                current_ndcg = eval_results.get("ndcg_at_k", 0.0)
                if current_ndcg > self._best_ndcg:
                    self._best_ndcg = current_ndcg
                    best_code = preprocess_path.read_text(encoding="utf-8")
                    best_path = _AGENT_DIR / f"preprocess_best_{self._task}.py"
                    best_path.write_text(best_code, encoding="utf-8")
                    print(
                        f"[gemini_sdk_bright] New best nDCG@10: {self._best_ndcg:.4f}"
                        f" -> saved to {best_path.name}"
                    )

            prompt = self.build_prompt(iteration=i, eval_results=eval_results)
            self.call_llm(prompt=prompt, iteration=i)

        # Restore preprocess.py to the best version seen this run
        if best_code is not None:
            preprocess_path.write_text(best_code, encoding="utf-8")
            print(
                f"\n[gemini_sdk_bright] Run complete."
                f" Restored preprocess.py to best version (nDCG={self._best_ndcg:.4f})"
            )

    # ------------------------------------------------------------------
    # call_llm: writes logs to per-task subdirectory
    # ------------------------------------------------------------------

    def call_llm(self, prompt: str, iteration: int) -> None:
        logs_dir = _AGENT_DIR / "logs" / self._task
        logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        header = f"=== Iteration {iteration + 1} | {timestamp} ===\n\n"

        prompt_log_path = logs_dir / f"iteration_{iteration + 1}_prompt.log"
        prompt_log_path.write_text(header + prompt, encoding="utf-8")

        print(f"[gemini_sdk_bright:{self._task}] Calling Gemini (iteration {iteration + 1}) ...")

        response = self._client.models.generate_content(
            model=_MODEL,
            contents=prompt,
        )

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
            print(f"[gemini_sdk_bright] response.text raised: {exc}")

        log_path = logs_dir / f"iteration_{iteration + 1}.log"
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
            print(f"[gemini_sdk_bright] Empty response. finish_reason={finish_reason}")
            if prompt_feedback:
                print(f"[gemini_sdk_bright] prompt_feedback: {prompt_feedback}")
        else:
            print(text)

        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1).rstrip()
            preprocess_path = _AGENT_DIR / "preprocess.py"
            preprocess_path.write_text(code + "\n", encoding="utf-8")
            print(f"[gemini_sdk_bright] preprocess.py updated ({len(code.splitlines())} lines).")
        else:
            print("[gemini_sdk_bright] Warning: no ```python``` block found — preprocess.py unchanged.")

        print(f"[gemini_sdk_bright] Iteration {iteration + 1} complete. Log: {log_path}")
