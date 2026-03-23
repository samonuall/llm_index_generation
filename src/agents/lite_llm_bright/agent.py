"""
agent.py – LiteLLM-backed iterative preprocessing agent for BRIGHT.

Key improvements over gemini_sdk_bright:
  - Model-agnostic via LiteLLM (config.yaml controls model + temperature)
  - Hypothesis-driven prompt: shows score delta, code diff, error traceback,
    term overlap (which query words are absent from relevant chunks), and
    rank of relevant doc — making the feedback loop legible and causal
  - MLflow tracking: logs nDCG@10, recall, MRR, n_chunks per iteration
  - Best-of-N: restores preprocess.py to the highest-scoring version at end
  - Error feedback: exceptions from preprocess code are shown to the LLM
"""

from __future__ import annotations

import difflib
import importlib.util
import os
import pathlib
import re
import sys
import traceback as _traceback
import datetime

import yaml
from dotenv import load_dotenv
from litellm import completion
import mlflow

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent
_SRC_AGENTS_DIR = _PROJECT_ROOT / "src" / "agents"

if str(_SRC_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_AGENTS_DIR))

from agent_runner import AgentRunner  # type: ignore


def _load_config() -> dict:
    with (_AGENT_DIR / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class LiteLLMBrightAgent(AgentRunner):
    agent_name = "lite_llm_bright"

    def __init__(self, task: str = "sustainable_living", include_query_text: bool = True) -> None:
        self._task = task
        self._include_query_text = include_query_text

        config = _load_config()
        self._model: str = config["model"]
        self._temperature: float = float(config.get("temperature", 1.0))
        self._api_key: str = os.environ.get("LITE_LLM_KEY", "")

        dataset_info = (_SRC_AGENTS_DIR / "CONTEXT_bright.md").read_text(encoding="utf-8")
        template = (_AGENT_DIR / "context" / "SYSTEM_INSTRUCTION_BRIGHT.md").read_text(encoding="utf-8")
        self._system_instruction = template.replace("{dataset_info}", dataset_info)

    # ------------------------------------------------------------------
    # Eval: BRIGHT harness
    # ------------------------------------------------------------------

    def run_eval(self) -> dict:
        eval_scripts_dir = _PROJECT_ROOT / "src" / "evaluation" / "scripts"
        eval_dir = _PROJECT_ROOT / "src" / "evaluation"
        for p in [str(eval_scripts_dir), str(eval_dir)]:
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
    # Prompt: hypothesis-driven with full diagnostic context
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        iteration: int,
        eval_results: dict | None,
        eval_error: str | None = None,
        code_diff: str | None = None,
        prev_ndcg: float | None = None,
        run_history: list[dict] | None = None,
    ) -> str:
        preprocess_path = _AGENT_DIR / "preprocess.py"
        current_code = preprocess_path.read_text(encoding="utf-8").strip()

        # First iteration with no results yet: rely solely on the system instruction
        # passed via the dedicated system role; no additional user-facing context.
        if iteration == 0 and eval_results is None and eval_error is None:
            return ""

        sections: list[str] = [""]

        # --- Run history (episodic memory across iterations) ---
        if run_history:
            rows = []
            for h in run_history:
                ndcg_str = f"{h['ndcg']:.4f}" if h["ndcg"] is not None else "ERROR"
                delta_str = f"({h['delta']:+.4f})" if h["delta"] is not None else "(first)"
                icon = "✓" if (h["delta"] and h["delta"] > 0) else ("✗" if (h["delta"] and h["delta"] < 0) else " ")
                hyp = h["hypothesis"].replace("\n", " ")[:120]
                rows.append(f"  iter {h['iteration']}: nDCG={ndcg_str} {delta_str} {icon}  \"{hyp}\"")
            sections.append("## Run history — what you've tried so far\n" + "\n".join(rows))

        # --- Score delta ---
        if eval_results is not None:
            curr_ndcg = eval_results["ndcg_at_k"]
            if prev_ndcg is not None:
                delta = curr_ndcg - prev_ndcg
                arrow = "✓ IMPROVED" if delta > 0 else ("✗ WORSE" if delta < 0 else "= NO CHANGE")
                sections.append(
                    f"## Score (iteration {iteration + 1})\n"
                    f"Previous: {prev_ndcg:.4f} → Current: **{curr_ndcg:.4f}** "
                    f"({delta:+.4f}) {arrow}"
                )
            else:
                sections.append(
                    f"## Score (iteration {iteration + 1})\n"
                    f"Current: **{curr_ndcg:.4f}**  "
                    f"(Recall@10: {eval_results['recall_at_k']:.4f}, MRR: {eval_results['mrr']:.4f}, "
                    f"chunks: {eval_results['n_chunks']})"
                )
        elif eval_error:
            sections.append(
                f"## Error — preprocessing code crashed (iteration {iteration + 1})\n"
                f"```\n{eval_error.strip()}\n```\n"
                f"Fix this error first. The score cannot improve while the code crashes."
            )

        # --- Code diff ---
        if code_diff and code_diff.strip():
            sections.append(f"## What changed from last iteration\n```diff\n{code_diff}\n```")

        # --- Per-query breakdown ---
        if eval_results is not None:
            k = eval_results["top_k"]
            query_results = eval_results.get("query_results", [])
            misses = [r for r in query_results if not r["hit"]]
            hits = [r for r in query_results if r["hit"]]

            # Missed queries — rich diagnostic output
            miss_lines: list[str] = []
            for r in misses:
                rank_info = (
                    f"rank_of_relevant={r['rank_of_relevant']}"
                    if r["rank_of_relevant"] is not None
                    else f"rank_of_relevant=>200"
                )
                score_info = (
                    f"score={r['score_of_relevant']:.3f} vs rank1={r['score_of_rank1']:.3f}"
                )
                present_str = ", ".join(r["terms_present"][:15]) or "(none)"
                missing_str = ", ".join(r["terms_missing"][:20]) or "(none)"

                line = (
                    f"  - [{r['query_id']}] {rank_info}  {score_info}\n"
                    f"    terms_present : {present_str}\n"
                    f"    terms_missing : **{missing_str}**"
                )
                if self._include_query_text:
                    q_snippet = r["query_text"][:180]
                    if len(r["query_text"]) > 180:
                        q_snippet += "..."
                    line = (
                        f"  - [{r['query_id']}] \"{q_snippet}\"\n"
                        f"    {rank_info}  {score_info}\n"
                        f"    terms_present : {present_str}\n"
                        f"    terms_missing : **{missing_str}**"
                    )

                if r.get("rank1_snippet"):
                    line += f"\n    rank1_competitor: \"{r['rank1_snippet'][:150]}\""

                miss_lines.append(line)

            # Hit queries sorted worst-first
            hit_lines: list[str] = []
            for r in sorted(hits, key=lambda x: x.get("ndcg", 0.0)):
                missing_str = ", ".join(r["terms_missing"][:10])
                entry = f"  - [{r['query_id']}] rank={r['rank']} ndcg={r.get('ndcg', 0):.3f}"
                if missing_str:
                    entry += f"  still missing: {missing_str}"
                if self._include_query_text:
                    entry += f"  \"{r['query_text'][:100]}\""
                hit_lines.append(entry)

            sections.append(
                f"## Per-query breakdown (task={self._task}, top-{k})\n\n"
                f"### Missed ({len(misses)}/{len(query_results)}) — terms_missing is the vocabulary gap to close:\n"
                + ("\n".join(miss_lines) if miss_lines else "  (none — all retrieved!)")
                + f"\n\n### Retrieved ({len(hits)}/{len(query_results)}) — sorted by nDCG worst-first:\n"
                + ("\n".join(hit_lines) if hit_lines else "  (none)")
            )

        # --- Best exemplar (FunSearch-style: show best seen so far) ---
        best_path = _AGENT_DIR / f"preprocess_best_{self._task}.py"
        if best_path.exists() and self._best_ndcg > 0:
            best_code = best_path.read_text(encoding="utf-8").strip()
            if best_code != current_code:
                sections.append(
                    f"## Best solution seen this run (nDCG={self._best_ndcg:.4f})\n"
                    f"```python\n{best_code}\n```\n"
                    f"If the current implementation is weaker, consider building on this instead."
                )

        # --- Current code ---
        sections.append(f"## Current implementation\n```python\n{current_code}\n```")

        sections.append(
            "Analyze the term overlap evidence and run history above. "
            "State your hypothesis (2-3 lines): what pattern do you see in terms_missing, "
            "and what specific change will you make? "
            "Then output the complete updated preprocess.py."
        )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Run: best-of-N + MLflow + code diff tracking
    # ------------------------------------------------------------------

    def run(self, n_loops: int) -> None:
        preprocess_path = _AGENT_DIR / "preprocess.py"
        self._best_ndcg: float = -1.0
        best_code: str | None = None

        mlflow.set_experiment(f"bright_{self._task}")
        run_name = f"{self._task}_{self._model}_{datetime.datetime.now():%Y%m%d_%H%M%S}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "task": self._task,
                "model": self._model,
                "temperature": self._temperature,
                "n_loops": n_loops,
                "include_query_text": self._include_query_text,
            })

            prev_code = preprocess_path.read_text(encoding="utf-8")
            prev_ndcg: float | None = None
            run_history: list[dict] = []  # episodic memory across iterations

            for i in range(n_loops):
                print(f"\n{'#' * 60}")
                print(f"# Iteration {i + 1} / {n_loops}  (task={self._task})")
                print(f"{'#' * 60}")

                current_code = preprocess_path.read_text(encoding="utf-8")

                # Compute diff between what was eval'd last time and current code
                code_diff: str | None = None
                if i > 0:
                    diff_lines = list(difflib.unified_diff(
                        prev_code.splitlines(),
                        current_code.splitlines(),
                        fromfile="previous",
                        tofile="current",
                        lineterm="",
                    ))
                    if diff_lines:
                        code_diff = "\n".join(diff_lines)

                # Eval
                eval_results: dict | None = None
                eval_error: str | None = None
                if not current_code.strip():
                    print(f"[lite_llm_bright] preprocess.py is empty, skipping eval.")
                else:
                    try:
                        eval_results = self.run_eval()
                    except Exception:
                        eval_error = _traceback.format_exc()
                        print(f"[lite_llm_bright] Eval crashed (iteration {i + 1}):\n{eval_error}")

                # Best-of-N
                if eval_results is not None:
                    curr_ndcg = eval_results["ndcg_at_k"]
                    mlflow.log_metric("ndcg_at_10", curr_ndcg, step=i)
                    mlflow.log_metric("recall_at_10", eval_results["recall_at_k"], step=i)
                    mlflow.log_metric("mrr", eval_results["mrr"], step=i)
                    mlflow.log_metric("n_chunks", eval_results["n_chunks"], step=i)

                    if curr_ndcg > self._best_ndcg:
                        self._best_ndcg = curr_ndcg
                        best_code = current_code
                        best_path = _AGENT_DIR / f"preprocess_best_{self._task}.py"
                        best_path.write_text(best_code, encoding="utf-8")
                        print(
                            f"[lite_llm_bright] New best nDCG@10: {self._best_ndcg:.4f}"
                            f" -> saved to {best_path.name}"
                        )

                prompt = self.build_prompt(
                    iteration=i,
                    eval_results=eval_results,
                    eval_error=eval_error,
                    code_diff=code_diff,
                    prev_ndcg=prev_ndcg,
                    run_history=run_history,
                )
                hypothesis = self.call_llm(prompt=prompt, iteration=i)

                # Record this iteration in episodic memory
                curr_ndcg_for_history = eval_results["ndcg_at_k"] if eval_results else None
                run_history.append({
                    "iteration": i + 1,
                    "ndcg": curr_ndcg_for_history,
                    "delta": (curr_ndcg_for_history - prev_ndcg)
                             if (curr_ndcg_for_history is not None and prev_ndcg is not None)
                             else None,
                    "hypothesis": hypothesis,
                })

                # Update state for next iteration
                prev_code = current_code
                if eval_results is not None:
                    prev_ndcg = eval_results["ndcg_at_k"]

            mlflow.log_metric("best_ndcg", self._best_ndcg)

        # Restore best version
        if best_code is not None:
            preprocess_path.write_text(best_code, encoding="utf-8")
            print(
                f"\n[lite_llm_bright] Run complete."
                f" Restored preprocess.py to best version (nDCG={self._best_ndcg:.4f})"
            )

    # ------------------------------------------------------------------
    # LLM call: LiteLLM with per-task logs
    # ------------------------------------------------------------------

    def call_llm(self, prompt: str, iteration: int) -> str:
        """Call LLM, write logs, update preprocess.py. Returns the hypothesis text."""
        logs_dir = _AGENT_DIR / "logs" / self._task
        logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        header = f"=== Iteration {iteration + 1} | {timestamp} | model={self._model} ===\n\n"

        (logs_dir / f"iteration_{iteration + 1}_prompt.log").write_text(
            header + prompt, encoding="utf-8"
        )

        print(f"[lite_llm_bright:{self._task}] Calling {self._model} (iteration {iteration + 1}) ...")

        try:
            response = completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temperature,
                api_key=self._api_key,
                api_base="https://thekeymaker.umass.edu/",
            )
            text: str = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
        except Exception as exc:
            text = ""
            finish_reason = f"exception: {exc}"
            print(f"[lite_llm_bright] LiteLLM call failed: {exc}")

        log_path = logs_dir / f"iteration_{iteration + 1}.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"--- finish_reason: {finish_reason}\n\n")
            f.write(text)

        if not text:
            print(f"[lite_llm_bright] Empty response. finish_reason={finish_reason}")
        else:
            print(text)

        # Extract hypothesis: text before the ```python block
        hypothesis = ""
        if text and "```python" in text:
            pre_code = text[:text.find("```python")].strip()
            hypothesis = " ".join(pre_code.split())[:400]  # single-line, max 400 chars

        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1).rstrip()
            (_AGENT_DIR / "preprocess.py").write_text(code + "\n", encoding="utf-8")
            print(f"[lite_llm_bright] preprocess.py updated ({len(code.splitlines())} lines).")
        else:
            print("[lite_llm_bright] Warning: no ```python``` block found — preprocess.py unchanged.")

        print(f"[lite_llm_bright] Iteration {iteration + 1} complete. Log: {log_path}")
        return hypothesis
