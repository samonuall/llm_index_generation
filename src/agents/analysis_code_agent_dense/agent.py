"""
agent.py - AnalysisCodeAgentDense: two-stage analysis + hypothesis-testing
orchestrator backed by a dense LanceDB vector retriever.

Mirrors ``analysis_code_agent.agent.AnalysisCodeAgent`` with two key changes:

1. The retrieval substrate is LanceDB (HNSW + cosine) via ``DenseClient`` /
   ``dense_server.py``. Per-loop and per-hypothesis indexes are created with
   loop-scoped names and dropped at end of loop / end of test.
2. Embedding is done by ``embedder.Embedder``, which truncates each input to
   the embedding model's max-token budget before sending it to the endpoint.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
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

_EVAL_DIR = _PROJECT_ROOT / "src" / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from ..agent_runner import AgentRunner
from .analysis_agent import AnalysisAgent
from .code_agent import CodeAgent
from .dense_client import DenseClient
from .run_journal import RunJournal
from .run_tracker import RunTracker


_DEBUG_LOGGER_NAME = "analysis_code_agent_dense"
logger = logging.getLogger(_DEBUG_LOGGER_NAME)


def _setup_debug_logger(experiment_dir: pathlib.Path) -> logging.Logger:
    log = logging.getLogger(_DEBUG_LOGGER_NAME)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    fh = logging.FileHandler(experiment_dir / "debug.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(fh)
    log.info("Debug log initialized at %s", experiment_dir / "debug.log")
    return log


def _load_config(overrides: dict | None = None) -> dict:
    config_path = _AGENT_DIR / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def _load_data(split: str = "tip_of_the_tongue", corpus_size: int | None = None, seed: int = 42):
    """Load queries and corpus from data/.

    If corpus_size is None, loads every document. Otherwise uses reservoir
    sampling to select corpus_size documents, always retaining every gold doc.
    """
    import random
    from schema import Document, EvalQuery

    data_dir = _PROJECT_ROOT / "data" / split
    if not data_dir.exists():
        data_dir = _PROJECT_ROOT / "data"

    def _load_queries_file(queries_path) -> list[EvalQuery]:
        queries = []
        if queries_path.exists():
            with queries_path.open(encoding="utf-8") as f:
                for line in f:
                    q = json.loads(line)
                    queries.append(EvalQuery(
                        query_id=q["query_id"],
                        query_text=q.get("query_text") or q.get("query_content", ""),
                        relevant_doc_ids=q["relevant_doc_ids"],
                    ))
        return queries

    val_queries = _load_queries_file(data_dir / "validation_queries.jsonl")
    eval_queries = _load_queries_file(data_dir / "evaluation_queries.jsonl")
    if not val_queries and not eval_queries:
        val_queries = _load_queries_file(data_dir / "queries.jsonl")
        eval_queries = val_queries

    docs_path = data_dir / "documents.jsonl"
    gold_doc_ids = {doc_id for q in val_queries + eval_queries for doc_id in q.relevant_doc_ids}
    target_non_gold = max(0, corpus_size - len(gold_doc_ids)) if corpus_size is not None else None

    gold_docs: list[Document] = []
    reservoir: list[Document] = []
    rng = random.Random(seed)
    n_non_gold_seen = 0

    with docs_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            doc = Document(doc_id=d["doc_id"], text=d["text"], metadata=d.get("metadata", {}))
            if doc.doc_id in gold_doc_ids:
                gold_docs.append(doc)
            elif target_non_gold is None:
                reservoir.append(doc)
            else:
                if n_non_gold_seen < target_non_gold:
                    reservoir.append(doc)
                else:
                    j = rng.randint(0, n_non_gold_seen)
                    if j < target_non_gold:
                        reservoir[j] = doc
                n_non_gold_seen += 1

    docs = gold_docs + reservoir
    print(f"[data] Corpus: {len(docs)} docs ({len(gold_docs)} gold + {len(reservoir)} non-gold), "
          f"{len(val_queries)} val queries, {len(eval_queries)} eval queries")
    return docs, val_queries, eval_queries


class AnalysisCodeAgentDense(AgentRunner):
    agent_name = "analysis_code_agent_dense"

    def __init__(
        self,
        use_history: bool = True,
        use_contrastive: bool = True,
        model: str | None = None,
        api_base: str | None = None,
    ) -> None:
        overrides: dict = {}
        if model:
            overrides["analysis_model"] = model
            overrides["code_model"] = model
            overrides["api_base"] = api_base
        elif api_base is not None:
            overrides["api_base"] = api_base
        self._config = _load_config(overrides or None)
        self._use_history = use_history
        self._use_contrastive = use_contrastive
        self._server_process = None
        self._server_log_path: pathlib.Path | None = None
        port = self._config.get("dense_server_port", self._config.get("server_port", 8766))
        self._client = DenseClient(
            base_url=f"http://localhost:{port}",
            batch_size=self._config.get("dense_batch_size", 5_000),
        )
        self._documents: list | None = None
        self._queries: list | None = None
        self._val_queries: list | None = None
        self._eval_queries: list | None = None

    # --- AgentRunner ABC stubs (we override run() instead) ---

    def build_prompt(self, iteration: int, eval_results: dict | None) -> str:
        return ""

    def call_llm(self, prompt: str, iteration: int) -> None:
        pass

    # --- Eval: build a fresh dense index on the server, run, drop ---

    def _eval_index_name(self, kind: str, iteration: int | None = None) -> str:
        ts = datetime.datetime.now().strftime("%H%M%S%f")
        suffix = f"_loop{iteration}" if iteration is not None else ""
        return f"{kind}{suffix}_{ts}"

    def run_eval(self, iteration: int | None = None, queries: list | None = None) -> dict:
        """Evaluate current preprocess.py against the corpus via the dense server.

        Each call creates a fresh, ephemeral index, runs the eval, and drops it.
        """
        from .eval_utils import load_preprocessor_from_code, run_subset_eval

        eval_queries = queries if queries is not None else self._queries
        if self._documents is None or eval_queries is None:
            raise RuntimeError("run_eval() called before documents/queries were loaded.")

        preprocess_path = _AGENT_DIR / "preprocess.py"
        code = preprocess_path.read_text(encoding="utf-8")
        preprocessor = load_preprocessor_from_code(code)

        print(f"[agent] Preprocessing {len(self._documents)} documents ...")
        chunks = preprocessor.preprocess(self._documents)
        print(f"[agent] Built {len(chunks)} chunks. Building dense index ...")

        index_name = self._eval_index_name("harness_eval", iteration)
        try:
            self._client.build_index(index_name, chunks, persist=False)
            summary = run_subset_eval(index_name, eval_queries, self._client, top_k=100)
        finally:
            self._client.delete_index_safe(index_name)

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
                "n_queries": len(eval_queries),
                "n_chunks": len(chunks),
                "chunks_per_doc": len(chunks) / max(len(self._documents), 1),
            },
            "metrics": {
                "recall_at_10": summary.recall_at_10,
                "recall_at_100": summary.recall_at_100,
                "ndcg_at_10": summary.ndcg_at_10,
            },
            "query_results": [],
        }

    # --- Server management ---

    def _ensure_server_running(self) -> None:
        if self._client.health():
            print("[agent] Dense server already running.")
            return

        port = self._config.get("dense_server_port", self._config.get("server_port", 8766))
        lance_uri = self._config.get("lance_uri", ".lance_cache")
        server_path = _AGENT_DIR / "dense_server.py"
        config_path = _AGENT_DIR / "config.yaml"
        self._server_log_path = _PROJECT_ROOT / "dense_server.log"

        print(f"[agent] Starting Dense server on port {port} ...")
        server_log = open(self._server_log_path, "w", encoding="utf-8")
        self._server_process = subprocess.Popen(
            [
                "uv", "run", "python", str(server_path),
                "--port", str(port),
                "--lance-uri", str(lance_uri),
                "--config-path", str(config_path),
            ],
            cwd=str(_PROJECT_ROOT),
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self._kill_server)

        max_wait = self._config.get("server_startup_timeout", 300)
        for i in range(max_wait * 2):
            if self._server_process.poll() is not None:
                tail = ""
                if self._server_log_path and self._server_log_path.exists():
                    try:
                        lines = self._server_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        tail = "\n".join(lines[-40:])
                    except Exception:
                        tail = "(failed reading dense_server.log)"
                raise RuntimeError(
                    "Dense server exited before becoming healthy "
                    f"(exit code {self._server_process.returncode}). "
                    f"See {self._server_log_path}.\n"
                    f"{tail}"
                )
            time.sleep(0.5)
            if self._client.health():
                print(f"[agent] Dense server ready (took {(i+1)*0.5:.1f}s).")
                return

        tail = ""
        if self._server_log_path and self._server_log_path.exists():
            try:
                lines = self._server_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-40:])
            except Exception:
                tail = "(failed reading dense_server.log)"
        raise RuntimeError(
            f"Dense server failed to start after {max_wait}s. "
            f"Check: uv run python {server_path} --port {port} "
            f"--lance-uri {lance_uri} --config-path {config_path}\n"
            f"Last dense_server.log lines:\n{tail}"
        )

    def _kill_server(self) -> None:
        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            print("[agent] Dense server stopped.")

    # --- Per-query enrichment ---

    def _enrich_eval_results(
        self, raw_results: dict, queries: list, client, current_index_name: str = "current"
    ) -> dict:
        """Enrich raw_results with per-query data using the named "current" index."""
        from .eval_utils import run_subset_eval

        eval_summary = run_subset_eval(current_index_name, queries, client, top_k=100)

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

    # --- Baseline ---

    def _compute_baseline(self, queries: list | None = None) -> dict:
        """Run baseline preprocessor on current corpus via dense server, then drop the index."""
        from .eval_utils import run_subset_eval

        _AGENTS_DIR = _PROJECT_ROOT / "src" / "agents"
        for p in [str(_EVAL_DIR), str(_AGENTS_DIR)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from baseline.preprocess import Preprocessor as BaselinePreprocessor

        eval_queries = queries if queries is not None else self._queries
        if self._documents is None or eval_queries is None:
            raise RuntimeError("_compute_baseline() called before documents/queries were loaded.")

        baseline_preprocessor = BaselinePreprocessor()
        baseline_chunks = baseline_preprocessor.preprocess(self._documents)
        print(f"[agent] Baseline: {len(baseline_chunks)} chunks from {len(self._documents)} docs")

        index_name = self._eval_index_name("baseline")
        try:
            self._client.build_index(index_name, baseline_chunks, persist=False)
            baseline_summary = run_subset_eval(index_name, eval_queries, self._client, top_k=100)
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
        finally:
            self._client.delete_index_safe(index_name)

    def _preprocess_with_current_code(self, documents: list, current_code: str) -> list:
        from .eval_utils import load_preprocessor_from_code
        preprocessor = load_preprocessor_from_code(current_code)
        return preprocessor.preprocess(documents)

    # --- Logging ---

    def _log_analysis(self, iteration: int, analysis_result) -> None:
        logs_dir = self._experiment_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

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

    def _model_folder(self) -> str:
        model = self._config.get("code_model", "unknown")
        return model.split("/")[-1].replace(".", "-")

    def _write_results(
        self,
        tracker: RunTracker,
        n_loops: int,
        n_docs: int,
        n_queries: int,
        baseline_results: dict,
        final_results: dict | None,
        baseline_val_results: dict | None = None,
        final_val_results: dict | None = None,
    ) -> None:
        results_dir = _PROJECT_ROOT / "results" / "dense" / self._model_folder()
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"{self.condition}_{timestamp}.json"

        eval_m = final_results.get("metrics", {}) if final_results else {}
        val_m = final_val_results.get("metrics", {}) if final_val_results else {}
        payload = {
            "condition": self.condition,
            "agent": self.agent_name,
            "model": self._config.get("code_model"),
            "embedding_model": self._config.get("embedding_model"),
            "loops": n_loops,
            "split": getattr(self, "split", "tip_of_the_tongue"),
            "seed": 42,
            "n_docs": n_docs,
            "n_queries": n_queries,
            "baseline_recall_100": baseline_results.get("recall_at_k"),
            "baseline_ndcg_10": baseline_results.get("ndcg"),
            "final_recall_100": eval_m.get("recall_at_100"),
            "final_ndcg_10": eval_m.get("ndcg_at_10"),
            "improvement_recall_100": (
                round(eval_m.get("recall_at_100", 0) - baseline_results.get("recall_at_k", 0), 4)
                if final_results else None
            ),
            "baseline_val_recall_100": baseline_val_results.get("recall_at_k") if baseline_val_results else None,
            "baseline_val_ndcg_10": baseline_val_results.get("ndcg") if baseline_val_results else None,
            "final_val_recall_100": val_m.get("recall_at_100"),
            "final_val_ndcg_10": val_m.get("ndcg_at_10"),
            "latency": tracker.to_dict(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[agent] Results saved -> {out_path}")

        if hasattr(self, '_experiment_dir') and self._experiment_dir:
            exp_results = self._experiment_dir / "results.json"
            exp_results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[agent] Results copy -> {exp_results}")

    # --- Main loop ---

    def run(self, n_loops: int) -> None:
        """Override AgentRunner.run() with analysis+hypothesis loop, dense backend."""

        corpus_size = self._config.get("corpus_size", None)
        documents, val_queries, eval_queries = _load_data(self.split, corpus_size=corpus_size)
        self._documents = documents
        self._val_queries = val_queries
        self._eval_queries = eval_queries
        print(f"\n{'='*60}")
        print(f"  Agent       : {self.agent_name}")
        print(f"  Split       : {self.split}")
        print(f"  Documents   : {len(documents)}")
        print(f"  Val queries : {len(val_queries)}")
        print(f"  Eval queries: {len(eval_queries)}")
        print(f"  Embed model : {self._config.get('embedding_model')}")
        print(f"  Embed URL   : {self._config.get('embedding_endpoint_url')}")
        print(f"{'='*60}\n")

        self._ensure_server_running()

        print(f"\n{'#'*60}")
        print(f"# Baseline (raw documents, no preprocessing) -- computed on current corpus")
        print(f"{'#'*60}")
        val_baseline_results = self._compute_baseline(queries=val_queries)
        eval_baseline_results = self._compute_baseline(queries=eval_queries)
        print(f"  Eval Recall@100 : {eval_baseline_results['recall_at_k']:.4f}")
        print(f"  Eval nDCG@10    : {eval_baseline_results['ndcg']:.4f}")

        model_name = self._config.get("code_model", "unknown_model").replace("/", "_")
        exp_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._experiment_dir = _PROJECT_ROOT / "ablation_experiments" / f"dense_{model_name}_{self.condition}_{exp_timestamp}"
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        _setup_debug_logger(self._experiment_dir)
        print(f"[agent] Experiment logs -> {self._experiment_dir}")
        logger.info(
            "Run start: model=%s condition=%s split=%s loops=%d baseline_recall@100=%.4f baseline_ndcg@10=%.4f",
            model_name, self.condition, self.split, n_loops,
            eval_baseline_results.get("recall_at_k", 0.0),
            eval_baseline_results.get("ndcg", 0.0),
        )

        tracker = RunTracker()
        analysis_agent = AnalysisAgent(self._config, tracker=tracker, split=self.split)
        code_agent = CodeAgent(self._config, tracker=tracker, split=self.split, log_dir=self._experiment_dir)
        max_hypotheses = self._config.get("max_hypotheses", 4)
        all_past_hypotheses: list[dict] = []
        journal = RunJournal(self._experiment_dir)

        best_recall_100: float = eval_baseline_results.get("recall_at_k", 0.0)
        best_code: str = (_AGENT_DIR / "preprocess.py").read_text(encoding="utf-8")  # noqa: F841

        for i in range(n_loops):
            print(f"\n{'#'*60}")
            print(f"# Loop {i + 1} / {n_loops}")
            print(f"{'#'*60}")
            logger.info("=== Loop %d/%d start ===", i + 1, n_loops)

            current_index_name = f"current_loop{i}"

            try:
                raw_results = self.run_eval(iteration=i * 2, queries=eval_queries)
                eval_log = self._experiment_dir / f"iteration_{i}_eval.json"
                eval_log.write_text(json.dumps(raw_results, indent=2, default=str), encoding="utf-8")
            except Exception as e:
                logger.exception("Eval failed on loop %d", i + 1)
                print(f"[agent] Eval failed (loop {i + 1}): {e}")
                continue

            loop_start_recall_100 = raw_results["metrics"]["recall_at_100"]
            print(f"[agent] Loop {i+1} starting Eval recall@100: {loop_start_recall_100:.4f} "
                  f"(global best: {best_recall_100:.4f})")

            current_code = (_AGENT_DIR / "preprocess.py").read_text(encoding="utf-8")

            # Build per-loop "current" dense index that hypotheses & analysis will compare against.
            print(f"[agent] Building '{current_index_name}' dense index ...")
            try:
                chunks = self._preprocess_with_current_code(documents, current_code)
                self._client.build_index(current_index_name, chunks, persist=False)
                print(f"[agent] '{current_index_name}' built with {len(chunks)} chunks.")
            except Exception as e:
                logger.exception("Index build failed on loop %d", i + 1)
                print(f"[agent] Index build failed: {e}")
                continue

            try:
                # Validation enrichment using the loop-scoped current index
                print("[agent] Enriching eval results with validation per-query data ...")
                try:
                    from .eval_utils import run_subset_eval
                    val_summary = run_subset_eval(current_index_name, val_queries, self._client, top_k=100)
                    val_raw_results = {
                        "metrics": {
                            "recall_at_10": val_summary.recall_at_10,
                            "recall_at_100": val_summary.recall_at_100,
                            "ndcg_at_10": val_summary.ndcg_at_10,
                        }
                    }
                    val_raw_results = self._enrich_eval_results(
                        val_raw_results, val_queries, self._client,
                        current_index_name=current_index_name,
                    )
                    print(f"[agent] Enriched with {len(val_raw_results.get('query_results', []))} val query results.")
                    val_log = self._experiment_dir / f"iteration_{i}_val.json"
                    val_log.write_text(json.dumps(val_raw_results, indent=2, default=str), encoding="utf-8")
                except Exception as e:
                    logger.exception("Val Enrichment failed on loop %d", i + 1)
                    print(f"[agent] Val Enrichment failed: {e}")
                    continue

                journal.record_iteration(
                    iteration=i,
                    eval_results=val_raw_results,
                    eval_results_harness=raw_results,
                )

                print("[agent] Running analysis agent on validation data ...")
                try:
                    # The analysis agent uses the dense client directly via vector_retrieve;
                    # it expects the index named "current". Build a temporary alias.
                    self._client.build_index("current", chunks, persist=False)
                    try:
                        analysis_result = analysis_agent.analyze(
                            eval_results=val_raw_results,
                            baseline_results=val_baseline_results,
                            current_code=current_code,
                            client=self._client,
                            split=self.split,
                            journal_summary=journal.summary_for_prompt() if self._use_history else None,
                        )
                    finally:
                        self._client.delete_index_safe("current")
                    self._log_analysis(i, analysis_result)
                except Exception as e:
                    logger.exception("Analysis agent failed on loop %d", i + 1)
                    print(f"[agent] Analysis failed: {e}")
                    continue

                print(f"[agent] Generating {max_hypotheses} hypotheses ...")
                persistent_fails = journal.persistent_failure_ids(min_iters=len(journal.iterations))
                query_lookup = {q.query_id: q.query_text for q in val_queries} if self._use_contrastive else None
                hypotheses = asyncio.run(code_agent.generate_hypotheses_async(
                    analysis_result.summary,
                    current_code,
                    n=max_hypotheses,
                    past_hypotheses=all_past_hypotheses if (all_past_hypotheses and self._use_history) else None,
                    persistent_failure_ids=persistent_fails if (persistent_fails and self._use_history) else None,
                    query_lookup=query_lookup,
                ))
                print(f"[agent] Generated {len(hypotheses)} hypotheses.")

                if not hypotheses:
                    print("[agent] No hypotheses generated -- skipping.")
                    continue

                print("[agent] Testing hypotheses on validation queries ...")
                hypothesis_results = []
                for h in hypotheses:
                    print(f"[agent] Testing {h.id}: {h.description}")
                    result = code_agent.test_hypothesis(
                        h, documents, val_queries, current_code, self._client,
                        iteration=i, current_index_name=current_index_name,
                    )
                    hypothesis_results.append(result)
                self._log_hypotheses(i, hypothesis_results)

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

                valid = [r for r in hypothesis_results if not r.error]
                if not valid:
                    print("[agent] All hypotheses errored -- preprocess.py unchanged.")
                    continue

                best_hyp = max(valid, key=lambda r: r.hypothesis_recall_100)
                print(f"[agent] Best hypothesis: {best_hyp.hypothesis.id} "
                      f"val_recall@100={best_hyp.hypothesis_recall_100:.4f} "
                      f"(delta {best_hyp.delta_recall_100:+.4f} vs val current, "
                      f"val baseline={best_hyp.baseline_recall_100:.4f})")

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
                candidate_eval_recall_100 = best_recall_100

                if best_hyp.delta_recall_100 > 0:
                    journal.set_iteration_adoption(i, best_hyp.hypothesis.id)

                    if best_hyp.regressed_query_ids:
                        print(
                            f"[agent] Overfitting risk: regresses {len(best_hyp.regressed_query_ids)} val queries "
                            f"({', '.join(best_hyp.regressed_query_ids[:5])}{'...' if len(best_hyp.regressed_query_ids) > 5 else ''})"
                        )

                    was_synthesized = False
                    synthesized_from_ids: list[str] = []
                    if len(proven_results) > 1:
                        print(f"[agent] {len(proven_results)} hypotheses proved -- attempting synthesis ...")
                        synthesized = code_agent.generate_final_code(
                            analysis_result.summary, proven_results, current_code
                        )
                        if synthesized:
                            val_err = code_agent._validate_code(synthesized, documents)
                            if val_err:
                                logger.error("Synthesis validation failed on loop %d:\n%s", i, val_err)
                                print(f"[agent] Synthesis validation failed: {val_err[:80]} -- falling back to best hypothesis.")
                                final_code = best_hyp.hypothesis.code
                            else:
                                synth_index = f"synthesized_loop{i}"
                                try:
                                    from .eval_utils import run_subset_eval as _rse
                                    synth_chunks = self._preprocess_with_current_code(documents, synthesized)
                                    self._client.build_index(synth_index, synth_chunks, persist=False)
                                    synth_summary = _rse(synth_index, val_queries, self._client, top_k=100)
                                    synth_recall = synth_summary.recall_at_100
                                    print(f"[agent] Synthesized val recall@100={synth_recall:.4f} vs best hypothesis {best_hyp.hypothesis_recall_100:.4f}")
                                    if synth_recall > best_hyp.hypothesis_recall_100:
                                        print(f"[agent] Synthesis beats best hypothesis -- adopting.")
                                        final_code = synthesized
                                        was_synthesized = True
                                        synthesized_from_ids = [r.hypothesis.id for r in proven_results]
                                    else:
                                        print(f"[agent] Synthesis did not beat best hypothesis -- falling back to {best_hyp.hypothesis.id}.")
                                        final_code = best_hyp.hypothesis.code
                                except Exception as e:
                                    logger.exception("Synthesis val eval failed on loop %d", i)
                                    print(f"[agent] Synthesis val eval failed: {e} -- falling back to best hypothesis.")
                                    final_code = best_hyp.hypothesis.code
                                finally:
                                    self._client.delete_index_safe(synth_index)
                        else:
                            print(f"[agent] Synthesis failed -- falling back to best hypothesis.")
                            final_code = best_hyp.hypothesis.code
                    else:
                        print(f"[agent] Adopting {best_hyp.hypothesis.id} directly.")
                        final_code = best_hyp.hypothesis.code

                    self._log_final_code(i, final_code)

                    self._write_preprocess(final_code)
                    print("[agent] Running authoritative harness eval on adopted code ...")
                    try:
                        candidate_results = self.run_eval(iteration=i * 2 + 1, queries=eval_queries)
                        candidate_eval_recall_100 = candidate_results["metrics"]["recall_at_100"]
                        print(f"[agent] Adopted Eval recall@100={candidate_eval_recall_100:.4f} "
                              f"(global best so far: {best_recall_100:.4f})")
                    except Exception as e:
                        logger.exception("Harness eval of adopted code failed on loop %d", i)
                        print(f"[agent] Harness eval of adopted code failed: {e} -- reverting to pre-loop code.")
                        self._write_preprocess(current_code)
                        continue

                    accepted_data = {
                        "iteration": i,
                        "adopted_hypothesis": {
                            "id": best_hyp.hypothesis.id,
                            "description": best_hyp.hypothesis.description,
                            "rationale": best_hyp.hypothesis.rationale,
                            "delta_val_recall_100": best_hyp.delta_recall_100,
                            "delta_val_recall_10": best_hyp.delta_recall_10,
                            "delta_val_ndcg_10": best_hyp.delta_ndcg_10,
                        },
                        "synthesized": was_synthesized,
                        "synthesized_from": synthesized_from_ids,
                        "proven_hypotheses": [
                            {"id": r.hypothesis.id, "description": r.hypothesis.description}
                            for r in proven_results
                        ],
                        "candidate_eval_recall_100": candidate_eval_recall_100,
                        "global_best_recall_100_before": best_recall_100,
                    }
                    accepted_path = self._experiment_dir / f"iteration_{i}_accepted.json"
                    accepted_path.write_text(json.dumps(accepted_data, indent=2), encoding="utf-8")

                    if candidate_eval_recall_100 > best_recall_100:
                        best_recall_100 = candidate_eval_recall_100
                        best_code = final_code  # noqa: F841
                        print(f"[agent] Global best updated -> eval recall@100={best_recall_100:.4f}")
                    else:
                        print(f"[agent] Candidate adopted, but eval recall did not beat global best.")

                    logger.info(
                        "Loop %d end: adopted=%s synthesized=%s candidate_eval_recall@100=%.4f best=%.4f",
                        i + 1, best_hyp.hypothesis.id, was_synthesized,
                        candidate_eval_recall_100, best_recall_100,
                    )
                else:
                    accepted_data = {
                        "iteration": i,
                        "adopted_hypothesis": None,
                        "reason": "no improvement",
                    }
                    accepted_path = self._experiment_dir / f"iteration_{i}_accepted.json"
                    accepted_path.write_text(json.dumps(accepted_data, indent=2), encoding="utf-8")
                    print(f"[agent] No hypothesis improved val recall over current -- preprocess.py unchanged.")
                    logger.info("Loop %d end: no adoption (best=%.4f)", i + 1, best_recall_100)
            finally:
                # Drop the per-loop "current" index regardless of what happened above.
                self._client.delete_index_safe(current_index_name)

        # Final eval
        print(f"\n{'#'*60}")
        print(f"# Final eval (after {n_loops} loop{'s' if n_loops != 1 else ''})")
        print(f"{'#'*60}")
        final_results = None
        final_val_results = None
        try:
            final_results = self.run_eval(queries=eval_queries)
            final_recall = final_results["metrics"]["recall_at_100"]
            baseline_recall = eval_baseline_results.get("recall_at_k", 0.0)
            print(f"\n[agent] Improvement: Eval recall@100 {baseline_recall:.4f} -> {final_recall:.4f} "
                  f"({final_recall - baseline_recall:+.4f})")
        except Exception as e:
            logger.exception("Final eval failed")
            print(f"[agent] Final eval failed: {e}")

        try:
            final_val_results = self.run_eval(queries=val_queries)
            final_val_recall = final_val_results["metrics"]["recall_at_100"]
            baseline_val_recall = val_baseline_results.get("recall_at_k", 0.0)
            print(f"[agent] Improvement: Val  recall@100 {baseline_val_recall:.4f} -> {final_val_recall:.4f} "
                  f"({final_val_recall - baseline_val_recall:+.4f})")
        except Exception as e:
            logger.exception("Final val eval failed")
            print(f"[agent] Final val eval failed: {e}")

        self._write_results(
            tracker=tracker,
            n_loops=n_loops,
            n_docs=len(documents),
            n_queries=len(eval_queries),
            baseline_results=eval_baseline_results,
            final_results=final_results,
            baseline_val_results=val_baseline_results,
            final_val_results=final_val_results,
        )

        self._kill_server()
