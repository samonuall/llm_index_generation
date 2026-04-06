"""
agent.py - AnalysisCodeAgent: two-stage analysis + hypothesis-testing orchestrator.

Overrides AgentRunner.run() to implement:
1. Analysis agent investigates failures via multi-turn bash loop
2. Code agent generates hypotheses, tests each on BM25 server, synthesizes final code
"""

from __future__ import annotations

import atexit
import json
import random
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
from .run_journal import RunJournal
from .run_tracker import RunTracker


def _load_config() -> dict:
    config_path = _AGENT_DIR / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_data(split: str = "tip_of_the_tongue", max_distractors: int = 9000, seed: int = 42):
    """Load queries and a manageable corpus: all relevant docs + sampled distractors.

    With 1M+ doc corpora, loading everything is impractical for fast iteration.
    We keep ALL relevant documents (gold answers) and sample max_distractors from
    the rest, giving a realistic but fast eval corpus (~10K docs total by default).
    """
    from schema import Document, EvalQuery

    data_dir = _PROJECT_ROOT / "data" / split
    if not data_dir.exists():
        data_dir = _PROJECT_ROOT / "data"

    # Load queries first to know which doc_ids are relevant
    queries = []
    queries_path = data_dir / "queries.jsonl"
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries.append(EvalQuery(
                query_id=q["query_id"],
                query_text=q.get("query_text") or q.get("query_content", ""),
                relevant_doc_ids=q["relevant_doc_ids"],
            ))

    relevant_ids: set[str] = set()
    for q in queries:
        relevant_ids.update(q.relevant_doc_ids)

    # Stream documents: always keep relevant, reservoir-sample distractors
    rng = random.Random(seed)
    relevant_docs: list[Document] = []
    distractor_reservoir: list[Document] = []

    docs_path = data_dir / "documents.jsonl"
    distractor_count = 0
    with docs_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            doc = Document(doc_id=d["doc_id"], text=d["text"], metadata=d.get("metadata", {}))
            if doc.doc_id in relevant_ids:
                relevant_docs.append(doc)
            else:
                distractor_count += 1
                if len(distractor_reservoir) < max_distractors:
                    distractor_reservoir.append(doc)
                else:
                    # Reservoir sampling: replace with decreasing probability
                    j = rng.randint(0, distractor_count - 1)
                    if j < max_distractors:
                        distractor_reservoir[j] = doc

    docs = relevant_docs + distractor_reservoir
    rng.shuffle(docs)
    print(
        f"[data] Corpus: {len(relevant_docs)} relevant + {len(distractor_reservoir)} distractors "
        f"= {len(docs)} docs total (from {distractor_count + len(relevant_docs)} available)"
    )
    return docs, queries


class AnalysisCodeAgent(AgentRunner):
    agent_name = "analysis_code_agent"

    def __init__(self, use_history: bool = True, use_contrastive: bool = True) -> None:
        self._config = _load_config()
        self._use_history = use_history
        self._use_contrastive = use_contrastive
        self._server_process = None
        self._client = BM25Client(
            base_url=f"http://localhost:{self._config.get('server_port', 8765)}",
        )
        # Set by run() after data loading; used by run_eval()
        self._documents: list | None = None
        self._queries: list | None = None

    # --- AgentRunner ABC stubs (we override run() instead) ---

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        return ""  # not used

    def call_llm(self, prompt: str, iteration: int) -> None:
        pass  # not used

    # --- Eval: use sampled corpus via BM25 server (not the full 1M-doc harness) ---

    def run_eval(self, iteration: int | None = None) -> dict:
        """Evaluate current preprocess.py against the sampled corpus via BM25 server.

        Overrides AgentRunner.run_eval() to avoid reloading 1M docs from disk.
        Builds a 'harness_eval' index on the server and runs all queries.
        """
        from .eval_utils import load_preprocessor_from_code, run_subset_eval

        if self._documents is None or self._queries is None:
            raise RuntimeError("run_eval() called before documents/queries were loaded.")

        preprocess_path = _AGENT_DIR / "preprocess.py"
        code = preprocess_path.read_text(encoding="utf-8")
        preprocessor = load_preprocessor_from_code(code)

        print(f"[agent] Preprocessing {len(self._documents)} documents ...")
        chunks = preprocessor.preprocess(self._documents)
        print(f"[agent] Built {len(chunks)} chunks. Pushing to BM25 server ...")
        self._client.build_index("harness_eval", chunks, persist=False)

        summary = run_subset_eval("harness_eval", self._queries, self._client, top_k=100)

        agent_name = getattr(preprocessor, "name", type(preprocessor).__name__)
        iter_str = f" (Iteration {iteration})" if iteration is not None else ""
        print(
            f"\n{'='*60}\n"
            f"Agent       : {agent_name}{iter_str}\n"
            f"{'='*60}\n"
            f"  Recall@10  : {summary.recall_at_10:.4f}\n"
            f"  Recall@100 : {summary.recall_at_100:.4f}\n"
            f"  nDCG@10    : {summary.ndcg_at_10:.4f}\n"
        )

        return {
            "agent": agent_name,
            "config": {
                "top_k": 100,
                "n_docs": len(self._documents),
                "n_queries": len(self._queries),
                "n_chunks": len(chunks),
                "chunks_per_doc": len(chunks) / max(len(self._documents), 1),
            },
            "metrics": {
                "recall_at_10": summary.recall_at_10,
                "recall_at_100": summary.recall_at_100,
                "ndcg_at_10": summary.ndcg_at_10,
            },
            "query_results": [],  # enriched separately by _enrich_eval_results
        }

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

    # --- Per-query enrichment ---

    def _enrich_eval_results(
        self, raw_results: dict, queries: list, client
    ) -> dict:
        """
        The eval harness returns metrics but no per-query hit/rank data.
        Enrich raw_results with query_results using the BM25 server's 'current' index.
        """
        from .eval_utils import run_subset_eval

        eval_summary = run_subset_eval("current", queries, client, top_k=100)

        query_results = []
        for pq, q in zip(eval_summary.per_query, queries):
            query_results.append({
                "query_id": pq.query_id,
                "query_text": q.query_text,
                "hit": pq.hit_at_100,
                "rank": pq.rank,
                "relevant_doc_ids": q.relevant_doc_ids,
                "retrieved_doc_ids": pq.retrieved_doc_ids,
            })

        raw_results["query_results"] = query_results
        return raw_results

    # --- Preprocessing helper ---

    def _compute_baseline(self) -> dict:
        """Run baseline preprocessor on the current sampled corpus via BM25 server.

        Overrides AgentRunner._compute_baseline() so that baseline results are
        computed on the exact same document set as the current eval (R1.B).
        """
        from .eval_utils import run_subset_eval

        _EVAL_DIR = _PROJECT_ROOT / "src" / "evaluation"
        _AGENTS_DIR = _PROJECT_ROOT / "src" / "agents"
        for p in [str(_EVAL_DIR), str(_AGENTS_DIR)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from baseline.preprocess import Preprocessor as BaselinePreprocessor

        if self._documents is None or self._queries is None:
            raise RuntimeError("_compute_baseline() called before documents/queries were loaded.")

        baseline_preprocessor = BaselinePreprocessor()
        baseline_chunks = baseline_preprocessor.preprocess(self._documents)
        print(f"[agent] Baseline: {len(baseline_chunks)} chunks from {len(self._documents)} docs")
        self._client.build_index("baseline", baseline_chunks, persist=False)

        baseline_summary = run_subset_eval("baseline", self._queries, self._client, top_k=100)
        return {
            "recall_at_k": baseline_summary.recall_at_100,
            "ndcg": baseline_summary.ndcg_at_10,
            "query_results": [
                {
                    "query_id": pq.query_id,
                    "hit": pq.hit_at_100,
                    "rank": pq.rank,
                    "retrieved_doc_ids": pq.retrieved_doc_ids,
                }
                for pq in baseline_summary.per_query
            ],
        }

    def _preprocess_with_current_code(self, documents: list, current_code: str) -> list:
        """Load preprocessor from current code and run it on documents."""
        from .eval_utils import load_preprocessor_from_code
        preprocessor = load_preprocessor_from_code(current_code)
        return preprocessor.preprocess(documents)

    # --- Logging ---

    def _log_analysis(self, iteration: int, analysis_result) -> None:
        logs_dir = self._experiment_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        # Full conversation log
        log_path = logs_dir / f"iteration_{iteration}_analysis.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"=== Analysis Agent | Iteration {iteration} | {timestamp} ===\n\n")

            for i, msg in enumerate(analysis_result.conversation):
                role = msg.get("role", "unknown").upper()
                f.write(f"--- MESSAGE {i} [{role}] ---\n")

                if msg.get("content"):
                    f.write(f"{msg['content']}\n")

                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        f.write(f"[TOOL CALL] id={tc.get('id')} name={fn.get('name')} args={fn.get('arguments')}\n")

                if role == "TOOL":
                    f.write(f"[TOOL RESULT] tool_call_id={msg.get('tool_call_id')}\n")

                f.write("\n")

            f.write("--- FINAL SUMMARY ---\n")
            f.write(analysis_result.summary)

        # Summary-only file
        summary_path = logs_dir / f"iteration_{iteration}_analysis_summary.txt"
        summary_path.write_text(analysis_result.summary, encoding="utf-8")

        print(f"[agent] Analysis log: {log_path}")

    def _log_hypotheses(self, iteration: int, hypothesis_results: list) -> None:
        logs_dir = self._experiment_dir
        logs_dir.mkdir(parents=True, exist_ok=True)

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
                    "hypothesis_recall_100": r.hypothesis_recall_100,
                    "baseline_recall_100": r.baseline_recall_100,
                    "delta_recall_100": r.delta_recall_100,
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
        logs_dir = self._experiment_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"iteration_{iteration}_final_code.py"
        log_path.write_text(code, encoding="utf-8")
        print(f"[agent] Final code log: {log_path}")

    def _write_preprocess(self, code: str) -> None:
        preprocess_path = _AGENT_DIR / "preprocess.py"
        preprocess_path.write_text(code + "\n", encoding="utf-8")
        print(f"[agent] preprocess.py updated ({len(code.splitlines())} lines).")

    @property
    def condition(self) -> str:
        if self._use_history and self._use_contrastive:
            return "agent_contrastive"
        if self._use_history:
            return "agent_history"
        if self._use_contrastive:
            return "agent_contrastive_no_history"
        return "agent"

    def _write_results(
        self,
        tracker: RunTracker,
        n_loops: int,
        n_docs: int,
        n_queries: int,
        baseline_results: dict,
        final_results: dict | None,
    ) -> None:
        results_dir = _PROJECT_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"{self.condition}_{timestamp}.json"

        metrics = final_results.get("metrics", {}) if final_results else {}
        payload = {
            "condition": self.condition,
            "loops": n_loops,
            "split": getattr(self, "split", "tip_of_the_tongue"),
            "seed": 42,
            "n_docs": n_docs,
            "n_queries": n_queries,
            "baseline_recall_100": baseline_results.get("recall_at_k"),
            "baseline_ndcg_10": baseline_results.get("ndcg"),
            "final_recall_100": metrics.get("recall_at_100"),
            "final_ndcg_10": metrics.get("ndcg_at_10"),
            "improvement_recall_100": (
                round(metrics.get("recall_at_100", 0) - baseline_results.get("recall_at_k", 0), 4)
                if final_results else None
            ),
            "latency": tracker.to_dict(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[agent] Results saved → {out_path}")

        # Also save a copy in the experiment directory
        if hasattr(self, '_experiment_dir') and self._experiment_dir:
            exp_results = self._experiment_dir / "results.json"
            exp_results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[agent] Results copy → {exp_results}")

    # --- Main loop ---

    def run(self, n_loops: int) -> None:
        """Override AgentRunner.run() with analysis+hypothesis loop."""

        # Load data
        max_distractors = self._config.get("max_distractors", 9000)
        seed = self._config.get("seed", 42)
        documents, queries = _load_data(self.split, max_distractors=max_distractors, seed=seed)
        self._documents = documents
        self._queries = queries
        print(f"[agent] Loaded {len(documents)} documents, {len(queries)} queries.")

        # Start BM25 server (must be up before baseline eval)
        self._ensure_server_running()

        # Compute baseline by running baseline preprocessor on the current corpus
        print(f"\n{'#'*60}")
        print(f"# Baseline (raw documents, no preprocessing) — computed on current corpus")
        print(f"{'#'*60}")
        baseline_results = self._compute_baseline()
        print(f"  Recall@100 : {baseline_results['recall_at_k']:.4f}")
        print(f"  nDCG@10    : {baseline_results['ndcg']:.4f}")

        # Create per-experiment log directory
        model_name = self._config.get("code_model", "unknown_model").replace("/", "_")
        exp_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._experiment_dir = _PROJECT_ROOT / "ablation_experiments" / f"{model_name}_{self.condition}_{exp_timestamp}"
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"[agent] Experiment logs → {self._experiment_dir}")

        # Create tracker + sub-agents + journal
        tracker = RunTracker()
        analysis_agent = AnalysisAgent(self._config, tracker=tracker, split=self.split)
        code_agent = CodeAgent(self._config, tracker=tracker, split=self.split, log_dir=self._experiment_dir)
        max_hypotheses = self._config.get("max_hypotheses", 4)
        all_past_hypotheses: list[dict] = []  # track across loops
        journal = RunJournal(self._experiment_dir)

        # Track globally best code + recall@100 (from harness eval) across all loops
        best_recall_100: float = baseline_results.get("recall_at_k", 0.0)
        best_code: str = (_AGENT_DIR / "preprocess.py").read_text(encoding="utf-8")

        for i in range(n_loops):
            print(f"\n{'#'*60}")
            print(f"# Loop {i + 1} / {n_loops}")
            print(f"{'#'*60}")

            # Full harness eval — authoritative recall@100 for this loop's starting point
            try:
                raw_results = self.run_eval(iteration=i * 2)
                eval_log = self._experiment_dir / f"iteration_{i}_eval.json"
                eval_log.write_text(json.dumps(raw_results, indent=2, default=str), encoding="utf-8")
            except Exception as e:
                print(f"[agent] Eval failed (loop {i + 1}): {e}")
                import traceback
                traceback.print_exc()
                continue

            # Anchor: harness recall@100 at the start of this loop
            loop_start_recall_100 = raw_results["metrics"]["recall_at_100"]
            print(f"[agent] Loop {i+1} starting recall@100: {loop_start_recall_100:.4f} "
                  f"(global best: {best_recall_100:.4f})")

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

            # Enrich eval results with per-query data from BM25 server
            print("[agent] Enriching eval results with per-query data ...")
            try:
                raw_results = self._enrich_eval_results(raw_results, queries, self._client)
                print(f"[agent] Enriched with {len(raw_results.get('query_results', []))} query results.")
            except Exception as e:
                print(f"[agent] Enrichment failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Record iteration in journal
            journal.record_iteration(iteration=i, eval_results=raw_results)

            # Analysis agent
            print("[agent] Running analysis agent ...")
            try:
                analysis_result = analysis_agent.analyze(
                    eval_results=raw_results,
                    baseline_results=baseline_results,
                    current_code=current_code,
                    client=self._client,
                    split=self.split,
                    journal_summary=journal.summary_for_prompt() if self._use_history else None,
                )
                self._log_analysis(i, analysis_result)
            except Exception as e:
                print(f"[agent] Analysis failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Hypothesis generation
            print(f"[agent] Generating {max_hypotheses} hypotheses ...")
            persistent_fails = journal.persistent_failure_ids(min_iters=len(journal.iterations))
            query_lookup = {q.query_id: q.query_text for q in queries} if self._use_contrastive else None
            hypotheses = code_agent.generate_hypotheses(
                analysis_result.summary,
                current_code,
                n=max_hypotheses,
                past_hypotheses=all_past_hypotheses if (all_past_hypotheses and self._use_history) else None,
                persistent_failure_ids=persistent_fails if (persistent_fails and self._use_history) else None,
                query_lookup=query_lookup,
            )
            print(f"[agent] Generated {len(hypotheses)} hypotheses.")

            if not hypotheses:
                print("[agent] No hypotheses generated — skipping.")
                continue

            # Hypothesis testing — compare each against the BM25 server "current" index
            print("[agent] Testing hypotheses ...")
            hypothesis_results = []
            for h in hypotheses:
                print(f"[agent] Testing {h.id}: {h.description}")
                result = code_agent.test_hypothesis(
                    h, documents, queries, current_code, self._client
                )
                hypothesis_results.append(result)
            self._log_hypotheses(i, hypothesis_results)

            # Track all tested hypotheses for future loops + journal
            for r in hypothesis_results:
                all_past_hypotheses.append({
                    "id": r.hypothesis.id,
                    "description": r.hypothesis.description,
                    "delta_recall_100": r.delta_recall_100,
                    "delta_recall_10": r.delta_recall_10,
                    "delta_ndcg_10": r.delta_ndcg_10,
                    "proven": r.proven,
                    "notes": r.notes,
                    "improved_query_ids": r.improved_query_ids,
                    "regressed_query_ids": r.regressed_query_ids,
                })

            # Pick the single best hypothesis by recall@100 on the BM25 server.
            # Always adopt the best if it improves over the current loop's starting point.
            valid = [r for r in hypothesis_results if not r.error]
            if not valid:
                print("[agent] All hypotheses errored — preprocess.py unchanged.")
                continue

            best_hyp = max(valid, key=lambda r: r.hypothesis_recall_100)
            print(f"[agent] Best hypothesis: {best_hyp.hypothesis.id} "
                  f"recall@100={best_hyp.hypothesis_recall_100:.4f} "
                  f"(Δ{best_hyp.delta_recall_100:+.4f} vs server current, "
                  f"server baseline={best_hyp.baseline_recall_100:.4f})")

            # Record all hypothesis results in journal (mark adopted=False first)
            for r in hypothesis_results:
                adopted = (r is best_hyp and best_hyp.delta_recall_100 > 0)
                journal.record_hypothesis(
                    iteration=i,
                    h_id=r.hypothesis.id,
                    description=r.hypothesis.description,
                    rationale=r.hypothesis.rationale,
                    targeted_query_ids=r.hypothesis.query_ids_to_test,
                    delta_recall_100=r.delta_recall_100,
                    delta_recall_10=r.delta_recall_10,
                    delta_ndcg_10=r.delta_ndcg_10,
                    proven=r.proven,
                    adopted=adopted,
                    improved_query_ids=r.improved_query_ids,
                    regressed_query_ids=r.regressed_query_ids,
                    error=r.error,
                )

            proven_results = [r for r in valid if r.proven]

            if best_hyp.delta_recall_100 > 0:
                journal.set_iteration_adoption(i, best_hyp.hypothesis.id)

                if best_hyp.regressed_query_ids:
                    print(
                        f"[agent] ⚠ Overfitting: regresses {len(best_hyp.regressed_query_ids)} queries "
                        f"({', '.join(best_hyp.regressed_query_ids[:5])}{'...' if len(best_hyp.regressed_query_ids) > 5 else ''})"
                    )

                # If multiple hypotheses proved, try synthesis instead of just picking best
                was_synthesized = False
                synthesized_from_ids: list[str] = []
                if len(proven_results) > 1:
                    print(f"[agent] {len(proven_results)} hypotheses proved — attempting synthesis ...")
                    synthesized = code_agent.generate_final_code(
                        analysis_result.summary, proven_results, current_code
                    )
                    if synthesized:
                        # Validate and quick-test the synthesized code
                        val_err = code_agent._validate_code(synthesized, documents)
                        if val_err:
                            print(f"[agent] Synthesis validation failed: {val_err[:80]} — falling back to best hypothesis.")
                            final_code = best_hyp.hypothesis.code
                        else:
                            print(f"[agent] Adopting synthesized code.")
                            final_code = synthesized
                            was_synthesized = True
                            synthesized_from_ids = [r.hypothesis.id for r in proven_results]
                    else:
                        print(f"[agent] Synthesis failed — falling back to best hypothesis.")
                        final_code = best_hyp.hypothesis.code
                else:
                    print(f"[agent] Adopting {best_hyp.hypothesis.id} directly.")
                    final_code = best_hyp.hypothesis.code

                # Write candidate code and run authoritative harness eval
                self._log_final_code(i, final_code)
                self._write_preprocess(final_code)
                print("[agent] Running authoritative harness eval on candidate code ...")
                try:
                    candidate_results = self.run_eval(iteration=i * 2 + 1)
                    candidate_recall_100 = candidate_results["metrics"]["recall_at_100"]
                    print(f"[agent] Candidate recall@100={candidate_recall_100:.4f} "
                          f"(global best so far: {best_recall_100:.4f})")
                except Exception as e:
                    print(f"[agent] Harness eval of candidate failed: {e} — reverting to pre-loop code.")
                    self._write_preprocess(current_code)
                    continue

                # Write accepted hypothesis JSON with authoritative recall
                accepted_data = {
                    "iteration": i,
                    "adopted_hypothesis": {
                        "id": best_hyp.hypothesis.id,
                        "description": best_hyp.hypothesis.description,
                        "rationale": best_hyp.hypothesis.rationale,
                        "delta_recall_100": best_hyp.delta_recall_100,
                        "delta_recall_10": best_hyp.delta_recall_10,
                        "delta_ndcg_10": best_hyp.delta_ndcg_10,
                    },
                    "synthesized": was_synthesized,
                    "synthesized_from": synthesized_from_ids,
                    "proven_hypotheses": [
                        {"id": r.hypothesis.id, "description": r.hypothesis.description}
                        for r in proven_results
                    ],
                    "candidate_recall_100": candidate_recall_100,
                    "global_best_recall_100_before": best_recall_100,
                }
                accepted_path = self._experiment_dir / f"iteration_{i}_accepted.json"
                accepted_path.write_text(json.dumps(accepted_data, indent=2), encoding="utf-8")

                if candidate_recall_100 > best_recall_100:
                    best_recall_100 = candidate_recall_100
                    best_code = final_code
                    print(f"[agent] Global best updated → recall@100={best_recall_100:.4f}")
                else:
                    print(f"[agent] Candidate did not beat global best — reverting preprocess.py.")
                    self._write_preprocess(current_code)
            else:
                # No hypothesis improved — write accepted JSON indicating no adoption
                accepted_data = {
                    "iteration": i,
                    "adopted_hypothesis": None,
                    "reason": "no improvement",
                }
                accepted_path = self._experiment_dir / f"iteration_{i}_accepted.json"
                accepted_path.write_text(json.dumps(accepted_data, indent=2), encoding="utf-8")
                print(f"[agent] No hypothesis improved over current — preprocess.py unchanged.")

        # Final eval
        print(f"\n{'#'*60}")
        print(f"# Final eval (after {n_loops} loop{'s' if n_loops != 1 else ''})")
        print(f"{'#'*60}")
        final_results = None
        try:
            final_results = self.run_eval()
            final_recall = final_results["metrics"]["recall_at_100"]
            baseline_recall = baseline_results.get("recall_at_k", 0.0)
            print(f"\n[agent] Improvement: recall@100 {baseline_recall:.4f} → {final_recall:.4f} "
                  f"({final_recall - baseline_recall:+.4f})")
        except Exception as e:
            print(f"[agent] Final eval failed: {e}")

        # Write results JSON
        self._write_results(
            tracker=tracker,
            n_loops=n_loops,
            n_docs=len(documents),
            n_queries=len(queries),
            baseline_results=baseline_results,
            final_results=final_results,
        )

        # Clean up server
        self._kill_server()
