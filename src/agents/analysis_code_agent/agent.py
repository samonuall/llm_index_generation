"""
agent.py - AnalysisCodeAgent: two-stage analysis + hypothesis-testing orchestrator.

Overrides AgentRunner.run() to implement:
1. Analysis agent investigates failures via multi-turn bash loop
2. Code agent generates hypotheses, tests each on BM25 server, synthesizes final code
"""

from __future__ import annotations

import atexit
import json
import subprocess
import sys
import time
import pathlib
import datetime

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_AGENT_DIR = pathlib.Path(__file__).parent

# Add evaluation to path
_EVAL_DIR = _PROJECT_ROOT / "src" / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from ..agent_runner import AgentRunner
from .analysis_agent import AnalysisAgent
from .code_agent import CodeAgent
from .bm25_client import BM25Client


def _load_config() -> dict:
    config_path = _AGENT_DIR / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_data(split: str = "tip_of_the_tongue"):
    """Load documents and queries from data/ directory."""
    from schema import Document, EvalQuery

    data_dir = _PROJECT_ROOT / "data" / split
    if not data_dir.exists():
        data_dir = _PROJECT_ROOT / "data"

    docs = []
    docs_path = data_dir / "documents.jsonl"
    with docs_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(Document(
                doc_id=d["doc_id"],
                text=d["text"],
                metadata=d.get("metadata", {}),
            ))

    queries = []
    queries_path = data_dir / "queries.jsonl"
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries.append(EvalQuery(
                query_id=q["query_id"],
                query_text=q["query_text"],
                relevant_doc_ids=q["relevant_doc_ids"],
            ))

    return docs, queries


class AnalysisCodeAgent(AgentRunner):
    agent_name = "analysis_code_agent"

    def __init__(self) -> None:
        self._config = _load_config()
        self._server_process = None
        self._client = BM25Client(
            base_url=f"http://localhost:{self._config.get('server_port', 8765)}",
        )

    # --- AgentRunner ABC stubs (we override run() instead) ---

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        return ""  # not used

    def call_llm(self, prompt: str, iteration: int) -> None:
        pass  # not used

    # --- Server management ---

    def _ensure_server_running(self) -> None:
        """Start the BM25 FastAPI server if not already up."""
        if self._client.health():
            print("[agent] BM25 server already running.")
            return

        port = self._config.get("server_port", 8765)
        persist_dir = self._config.get("server_persist_dir", ".bm25_cache")
        server_path = _AGENT_DIR / "bm25_server.py"

        print(f"[agent] Starting BM25 server on port {port} ...")
        self._server_process = subprocess.Popen(
            ["uv", "run", "python", str(server_path), "--port", str(port), "--persist-dir", persist_dir],
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self._kill_server)

        # Wait for server to be ready (up to 15s)
        for i in range(30):
            time.sleep(0.5)
            if self._client.health():
                print(f"[agent] BM25 server ready (took {(i+1)*0.5:.1f}s).")
                return

        raise RuntimeError(
            f"BM25 server failed to start after 15s. "
            f"Check: uv run python {server_path} --port {port}"
        )

    def _kill_server(self) -> None:
        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            print("[agent] BM25 server stopped.")

    # --- Preprocessing helper ---

    def _preprocess_with_current_code(self, documents: list, current_code: str) -> list:
        """Load preprocessor from current code and run it on documents."""
        from .eval_utils import load_preprocessor_from_code
        preprocessor = load_preprocessor_from_code(current_code)
        return preprocessor.preprocess(documents)

    # --- Logging ---

    def _log_analysis(self, iteration: int, analysis_result) -> None:
        logs_dir = _AGENT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        # Full conversation log
        log_path = logs_dir / f"iteration_{iteration}_analysis.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"=== Analysis Agent | Iteration {iteration} | {timestamp} ===\n\n")

            # Context summary
            n_turns = 0
            for msg in analysis_result.conversation:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user" and n_turns == 0:
                    f.write("--- CONTEXT SUMMARY ---\n")
                    f.write(msg["content"][:500] + "\n...\n\n")
                    n_turns += 1
                elif msg["role"] == "assistant":
                    f.write(f"--- TURN {n_turns} ---\n")
                    f.write(f"[ASSISTANT]: {msg['content']}\n\n")
                    n_turns += 1
                elif msg["role"] == "user":
                    f.write(f"{msg['content']}\n\n")

            f.write("--- FINAL SUMMARY ---\n")
            f.write(analysis_result.summary)

        # Summary-only file
        summary_path = logs_dir / f"iteration_{iteration}_analysis_summary.txt"
        summary_path.write_text(analysis_result.summary, encoding="utf-8")

        print(f"[agent] Analysis log: {log_path}")

    def _log_hypotheses(self, iteration: int, hypothesis_results: list) -> None:
        logs_dir = _AGENT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)

        data = []
        for r in hypothesis_results:
            h = r.hypothesis
            data.append({
                "id": h.id,
                "description": h.description,
                "rationale": h.rationale,
                "code": h.code,
                "query_ids_to_test": h.query_ids_to_test,
                "falsifying_condition": h.falsifying_condition,
                "test_results": {
                    "hypothesis_recall_10": r.hypothesis_recall_10,
                    "baseline_recall_10": r.baseline_recall_10,
                    "delta_recall_10": r.delta_recall_10,
                    "delta_ndcg_10": r.delta_ndcg_10,
                    "proven": r.proven,
                    "notes": r.notes,
                    "error": r.error,
                },
            })

        log_path = logs_dir / f"iteration_{iteration}_hypotheses.json"
        log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[agent] Hypotheses log: {log_path}")

    def _log_final_code(self, iteration: int, code: str) -> None:
        logs_dir = _AGENT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / f"iteration_{iteration}_final_code.py"
        log_path.write_text(code, encoding="utf-8")
        print(f"[agent] Final code log: {log_path}")

    def _write_preprocess(self, code: str) -> None:
        preprocess_path = _AGENT_DIR / "preprocess.py"
        preprocess_path.write_text(code + "\n", encoding="utf-8")
        print(f"[agent] preprocess.py updated ({len(code.splitlines())} lines).")

    # --- Main loop ---

    def run(self, n_loops: int) -> None:
        """Override AgentRunner.run() with analysis+hypothesis loop."""

        # Load baseline results
        baseline_path = pathlib.Path(__file__).parent.parent / "baseline_results.json"
        print(f"\n{'#'*60}")
        print(f"# Baseline (raw documents, no preprocessing) — from baseline_results.json")
        print(f"{'#'*60}")
        baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))
        print(f"  Recall@100 : {baseline_results['recall_at_k']:.4f}")
        print(f"  nDCG@10    : {baseline_results['ndcg']:.4f}")

        # Load data
        documents, queries = _load_data(self.split)
        print(f"[agent] Loaded {len(documents)} documents, {len(queries)} queries.")

        # Start BM25 server
        self._ensure_server_running()

        # Create sub-agents
        analysis_agent = AnalysisAgent(self._config)
        code_agent = CodeAgent(self._config)
        max_hypotheses = self._config.get("max_hypotheses", 4)

        for i in range(n_loops):
            print(f"\n{'#'*60}")
            print(f"# Loop {i + 1} / {n_loops}")
            print(f"{'#'*60}")

            # Full harness eval
            try:
                raw_results = self.run_eval(iteration=i * 2)
                # Save eval results
                eval_log = _AGENT_DIR / "logs" / f"iteration_{i}_eval.json"
                eval_log.parent.mkdir(exist_ok=True)
                eval_log.write_text(json.dumps(raw_results, indent=2, default=str), encoding="utf-8")
            except Exception as e:
                print(f"[agent] Eval failed (loop {i + 1}): {e}")
                import traceback
                traceback.print_exc()
                continue

            current_code = (_AGENT_DIR / "preprocess.py").read_text(encoding="utf-8")

            # Rebuild "current" BM25 index on server
            print("[agent] Building 'current' index on BM25 server ...")
            try:
                chunks = self._preprocess_with_current_code(documents, current_code)
                self._client.build_index("current", chunks, persist=True)
                print(f"[agent] 'current' index built with {len(chunks)} chunks.")
            except Exception as e:
                print(f"[agent] Index build failed: {e}")
                continue

            # Analysis agent
            print("[agent] Running analysis agent ...")
            try:
                analysis_result = analysis_agent.analyze(
                    eval_results=raw_results,
                    baseline_results=baseline_results,
                    current_code=current_code,
                    documents=documents,
                    queries=queries,
                    client=self._client,
                )
                self._log_analysis(i, analysis_result)
            except Exception as e:
                print(f"[agent] Analysis failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Hypothesis generation
            print(f"[agent] Generating {max_hypotheses} hypotheses ...")
            hypotheses = code_agent.generate_hypotheses(
                analysis_result.summary, current_code, n=max_hypotheses
            )
            print(f"[agent] Generated {len(hypotheses)} hypotheses.")

            if not hypotheses:
                print("[agent] No hypotheses generated — skipping.")
                continue

            # Hypothesis testing
            print("[agent] Testing hypotheses ...")
            hypothesis_results = []
            for h in hypotheses:
                print(f"[agent] Testing {h.id}: {h.description}")
                result = code_agent.test_hypothesis(
                    h, documents, queries, current_code, self._client
                )
                hypothesis_results.append(result)
            self._log_hypotheses(i, hypothesis_results)

            # Final code generation
            proven = [r for r in hypothesis_results if r.proven]
            print(f"[agent] {len(proven)} / {len(hypothesis_results)} hypotheses proven.")

            if proven:
                print("[agent] Generating final code from proven hypotheses ...")
                final_code = code_agent.generate_final_code(
                    analysis_result.summary, proven, current_code
                )
                if final_code:
                    self._log_final_code(i, final_code)
                    self._write_preprocess(final_code)
                else:
                    print("[agent] Final code generation failed — preprocess.py unchanged.")
            else:
                print("[agent] No proven hypotheses — preprocess.py unchanged.")

        # Final eval
        print(f"\n{'#'*60}")
        print(f"# Final eval (after {n_loops} loop{'s' if n_loops != 1 else ''})")
        print(f"{'#'*60}")
        try:
            self.run_eval()
        except Exception as e:
            print(f"[agent] Final eval failed: {e}")

        # Clean up server
        self._kill_server()
