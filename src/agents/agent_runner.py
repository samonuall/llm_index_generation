"""
agent_runner.py – Abstract base class for LLM-driven preprocessing agents.

Each agent subclass implements build_prompt() and call_llm(), then calls
run(n_loops) to start the iterative eval-improve loop.
"""

from __future__ import annotations

import json
import sys
import importlib.util
import pathlib
from abc import ABC, abstractmethod

_BASELINE_RESULTS_PATH = pathlib.Path(__file__).parent / "baseline_results.json"

_PROJECT_ROOT = pathlib.Path(__file__).parents[2]


class AgentRunner(ABC):
    agent_name: str  # set by subclass
    baseline_results: dict | None = None  # set before first agent loop
    _system_instruction: str = ""  # set by subclass, can be updated after baseline eval

    def run(self, n_loops: int) -> None:
        """Main eval-improve loop."""
        preprocess_path = (
            _PROJECT_ROOT / "src" / "agents" / self.agent_name / "preprocess.py"
        )

        # Baseline results loaded from src/agents/baseline_results.json
        print(f"\n{'#'*60}")
        print(f"# Baseline (raw documents, no preprocessing) — from baseline_results.json")
        print(f"{'#'*60}")
        self.baseline_results = json.loads(_BASELINE_RESULTS_PATH.read_text(encoding="utf-8"))
        print(f"  Recall@100 : {self.baseline_results['recall_at_k']:.4f}")
        print(f"  nDCG@10    : {self.baseline_results['ndcg']:.4f}")
        self.on_baseline_complete(self.baseline_results)

        for i in range(n_loops):
            print(f"\n{'#'*60}")
            print(f"# Iteration {i + 1} / {n_loops}")
            print(f"{'#'*60}")

            prompt = None
            if not preprocess_path.read_text(encoding="utf-8").strip():
                print("[agent_runner] preprocess.py is empty, skipping eval.")
                eval_results = None
                prompt = "[agent_runner] No eval results available, using empty prompt."
            else:
                try:
                    eval_results = self.run_eval()
                except Exception as e:
                    print(f"[agent_runner] Eval failed (iteration {i + 1}): {e}")
                    eval_results = None
                    prompt = f"[agent_runner] Eval failed with error: {e}, current implementation:\n{preprocess_path.read_text(encoding='utf-8')}"

            if eval_results:
                prompt = self.build_prompt(iteration=i, eval_results=eval_results)
            
            self.call_llm(prompt=prompt, iteration=i)

        # Run final eval to show results of the last generated preprocess.py
        print(f"\n{'#'*60}")
        print(f"# Final eval (after {n_loops} loop{'s' if n_loops != 1 else ''})")
        print(f"{'#'*60}")
        if not preprocess_path.read_text(encoding="utf-8").strip():
            print("[agent_runner] preprocess.py is empty, skipping final eval.")
        else:
            try:
                self.run_eval()
            except Exception as e:
                print(f"[agent_runner] Final eval failed: {e}")

    def run_eval(self) -> dict:
        """
        Dynamically load Preprocessor from the agent's preprocess.py and run
        the static evaluate() harness. Returns the results dict.
        """
        eval_scripts_dir = _PROJECT_ROOT / "src" / "evaluation" / "scripts"
        eval_dir = _PROJECT_ROOT / "src" / "evaluation"
        src_dir = _PROJECT_ROOT / "src"

        for p in [str(eval_scripts_dir), str(eval_dir), str(src_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Import evaluate() from the static harness
        from test_preprocessing import evaluate  # type: ignore

        # Reload preprocess.py fresh each iteration so code changes take effect
        preprocess_path = (
            _PROJECT_ROOT / "src" / "agents" / self.agent_name / "preprocess.py"
        )
        spec = importlib.util.spec_from_file_location(
            f"_agent_{self.agent_name}_preprocess", preprocess_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        preprocessor = module.Preprocessor()
        return evaluate(preprocessor, top_k=100)

    def on_baseline_complete(self, baseline_results: dict) -> None:
        """Called after baseline eval; override to inject baseline numbers into system instruction."""

    @abstractmethod
    def build_prompt(self, iteration: int, eval_results: dict | None) -> str: ...

    @abstractmethod
    def call_llm(self, prompt: str, iteration: int) -> None: ...
