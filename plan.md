Plan: Analysis + Code Agent (analysis_code_agent)
Context
The project currently has iterative LLM agents that generate preprocessing code for BM25 retrieval, receive eval feedback, and improve. This builds a smarter two-stage system:

Analysis Agent: A mini SWE-style bash-loop agent (gpt-4o-mini) that queries a live BM25 server, inspects failures/hard negatives/successes, and produces a structured summary.
Code Agent: A stronger model (gpt-4o) that generates N hypotheses from the analysis, tests each on a query subset via the BM25 server, discards disproven ones, and synthesizes surviving ideas into the final preprocess.py.
BM25 FastAPI Server: Hosts named BM25 indexes in memory, persisting between runs. Supports async batch retrieval.
Architecture

[FastAPI BM25 Server] ← always running, multiple named indexes in memory
         ↑  ↑  ↑
run_eval() → AnalysisAgent.analyze() [queries "current" index via HTTP]
          → CodeAgent.generate_hypotheses()
          → CodeAgent.test_hypothesis() [builds "hyp_H1" index, queries, deletes]
          → filter proven → CodeAgent.generate_final_code()
          → rebuild "current" index → run_eval()
Files to Create

src/agents/analysis_code_agent/
├── __init__.py                    # exports AnalysisCodeAgent
├── agent.py                       # AnalysisCodeAgent(AgentRunner) - orchestrator
├── preprocess.py                  # Baseline passthrough to start
├── analysis_agent.py              # AnalysisAgent - multi-turn bash loop
├── code_agent.py                  # CodeAgent - hypothesis gen, test, synthesis
├── bm25_server.py                 # FastAPI BM25 index server
├── bm25_client.py                 # HTTP client for BM25 server
├── eval_utils.py                  # Subset eval using bm25_client
├── config.yaml                    # Model configs + server settings
└── context/
    ├── ANALYSIS_SYSTEM.md         # Analysis agent system prompt + taxonomy
    └── CODE_SYSTEM.md             # Code agent system prompt
Existing files to modify:

main.py — add "analysis_code_agent" to choices + elif registration
src/agents/__init__.py — add from .analysis_code_agent import *
New dependencies to add:


uv add fastapi uvicorn httpx
BM25 FastAPI Server (bm25_server.py)
Runs on a configurable port (default 8765). Maintains a dict of named BM25Index objects in memory. Can persist the "current" index to disk (bm25s native save format) for cross-run reuse.

Endpoints

GET  /health
     → {"status": "ok", "indexes": ["current", "hyp_H1"]}

GET  /indexes
     → {"indexes": ["current"], "n_chunks": {"current": 2847}}

POST /index/{name}/build
     body: {"chunks": [{"chunk_id": str, "doc_id": str, "text": str, "metadata": dict}],
             "persist": bool}   # persist=True saves to disk for cross-run reuse
     → {"status": "built", "n_chunks": 2847}

POST /index/{name}/retrieve
     body: {"query": str, "top_k": int}
     → {"results": [{"doc_id": str, "score": float, "rank": int}]}

POST /index/{name}/batch_retrieve
     body: {"queries": [{"query_id": str, "query_text": str}], "top_k": int}
     → {"results": [{"query_id": str,
                      "ranked_docs": [{"doc_id": str, "score": float, "rank": int}]}]}

DELETE /index/{name}
     → {"status": "deleted"}
Server startup / management
AnalysisCodeAgent checks GET /health on init. If no response (connection refused), it starts the server as a subprocess:


subprocess.Popen(
    ["uv", "run", "python", str(SERVER_PATH), "--port", str(port)],
    cwd=PROJECT_ROOT,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
Registers atexit handler to kill the process on clean exit. Re-checks health with retries (up to 10s).

The server CLI: python bm25_server.py --port 8765 --persist-dir .bm25_cache/

BM25 Client (bm25_client.py)
Thin httpx-based client with retry wrapper on all methods. Because agent loops can run for extended periods (multi-turn LLM calls, hypothesis testing), httpx sessions may go stale or timeout. All HTTP methods use a retry decorator:


def _with_retry(max_attempts: int = 3, backoff: float = 2.0):
    """Decorator: on httpx.RequestError or httpx.TimeoutException, wait and retry.
    Creates a fresh httpx.Client per attempt to avoid stale connections."""

class BM25Client:
    def __init__(self, base_url: str = "http://localhost:8765",
                 timeout: float = 120.0,   # long timeout for large index builds
                 max_retries: int = 3) -> None: ...

    @_with_retry()
    def build_index(self, name: str, chunks: list[Chunk], persist: bool = False) -> None:
        """POST /index/{name}/build — uses fresh httpx.Client per call"""

    @_with_retry()
    def retrieve(self, name: str, query: str, top_k: int = 100) -> list[dict]:
        """POST /index/{name}/retrieve → list of {doc_id, score, rank}"""

    @_with_retry()
    def batch_retrieve(
        self,
        name: str,
        queries: list[EvalQuery],
        top_k: int = 100,
    ) -> list[dict]:
        """POST /index/{name}/batch_retrieve → list of {query_id, ranked_docs}"""

    @_with_retry()
    def delete_index(self, name: str) -> None:
        """DELETE /index/{name}"""

    def health(self) -> bool:
        """GET /health → True if server is up (no retry; used for polling)"""
Each method creates a new httpx.Client instance per call (rather than reusing a long-lived session) to prevent stale connection issues. The retry wrapper catches httpx.RequestError and httpx.TimeoutException, sleeps with exponential backoff, and re-raises after max_attempts failures.

eval_utils.py — Partial Evaluation via Server

@dataclass
class SubsetEvalResult:
    query_id: str
    hit_at_10: bool
    hit_at_100: bool
    rank: int | None          # 1-indexed, None if not in top_k
    ndcg_at_10: float
    retrieved_doc_ids: list[str]

@dataclass
class SubsetEvalSummary:
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    n_queries: int
    per_query: list[SubsetEvalResult]

def run_subset_eval(
    index_name: str,
    queries: list[EvalQuery],
    client: BM25Client,
    top_k: int = 100,
) -> SubsetEvalSummary:
    """
    Queries the BM25 server for the given queries against a named index.
    Computes recall@10, recall@100, nDCG@10 from the ranked results.
    """

def load_preprocessor_from_code(code: str) -> BasePreprocessor:
    """
    exec() code string into a fresh module namespace and return Preprocessor().
    Used for hypothesis testing without filesystem writes.
    """
analysis_agent.py — Multi-Turn Bash Loop
Loop mechanics (max 8 turns):

Build initial context with: domain info, current preprocess.py, eval summary, candidate tuples, server URL, and data file paths
Model responds with <bash>...</bash> blocks (or plain text to finish)
Execute bash via subprocess.run(shell=True, capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT)
Truncate combined stdout+stderr to 4000 chars
Append to messages, loop
Stop when: no <bash> block OR max_turns reached
Final non-bash response = analysis summary; fallback: one more "summarize your findings" call
Candidate analysis targets built from eval_results:

Failures (regressions): query_ids where baseline had hit_at_100=True but current doesn't
Hard negatives: top-10 retrieved doc_ids that aren't gold, for missed queries (limit: 5 queries × 3 wrong docs)
Successes: queries with hits, sorted worst rank first
Analysis agent can query the BM25 server via bash, e.g.:


import requests
r = requests.post('http://localhost:8765/index/current/retrieve',
    json={'query': 'silent film actress 1920s', 'top_k': 10})
print(r.json())
ANALYSIS_SYSTEM.md failure taxonomy:

CHUNKING TOO AGGRESSIVE — splits destroy term co-occurrence
CHUNKING TOO COARSE — entity name buried, IDF dampened
STOPWORD REMOVAL HURTS — proper nouns / titles stripped (e.g. "The Who")
METADATA NOT INDEXED — title/aliases only in metadata, not chunk text
TERM FREQUENCY DILUTION — long docs lower TF of rare terms
NO FIELD BOOSTING — title should outweigh body
STEMMING MISMATCH — query vs entity name stem differently
code_agent.py — Hypothesis Testing
Hypothesis generation: Single LLM call. Output JSON inside <hypotheses>...</hypotheses> tags.


[{
  "id": "H1",
  "description": "Add title repetition at start of each chunk",
  "rationale": "Analysis shows entity name often only in article title...",
  "code": "import sys...\nclass Preprocessor(BasePreprocessor):\n    ...",
  "query_ids_to_test": ["q_023", "q_041", "q_087"],
  "falsifying_condition": "recall@10 on test queries does not improve by 0.05"
}]
Parse with json.loads; retry once on parse failure.

Hypothesis testing:

Filter queries to query_ids_to_test (use all if empty; require ≥3)
load_preprocessor_from_code(hypothesis.code) → get chunks
Build "hyp_{H.id}" index on server via client.build_index(...)
run_subset_eval("hyp_{H.id}", queries, client)
Also run_subset_eval("current", queries, client) for baseline comparison
proven = delta_recall_10 >= recall_threshold (default 0.05)
client.delete_index("hyp_{H.id}")
Final code generation: Single LLM call with analysis summary + proven hypothesis results. Validates class Preprocessor in response before writing to disk.

agent.py — Orchestrator
AnalysisCodeAgent(AgentRunner) overrides run(). Implements build_prompt and call_llm as no-op stubs to satisfy ABC.

run(n_loops) flow:


baseline_results = load from baseline_results.json
on_baseline_complete(baseline_results)
documents, queries = _load_data()   # cached from data/<split>/
_ensure_server_running()            # start FastAPI server if not up

for i in range(n_loops):
    header(f"Loop {i+1}/{n_loops}")

    # Full harness eval (writes results file)
    raw_results = run_eval(iteration=i*2)
    current_code = (AGENT_DIR / "preprocess.py").read_text()

    # Rebuild "current" BM25 index on server from current preprocess.py
    chunks = _preprocess_with_current_code(documents, current_code)
    client.build_index("current", chunks, persist=True)

    # Analysis agent
    analysis_result = analysis_agent.analyze(
        eval_results=raw_results,
        baseline_results=baseline_results,
        current_code=current_code,
        documents=documents,
        queries=queries,
        client=client,
    )
    _log_analysis(i, analysis_result)           # logs/iteration_{i}_analysis.log

    # Hypothesis generation
    hypotheses = code_agent.generate_hypotheses(
        analysis_result.summary, current_code, n=max_hypotheses
    )

    # Hypothesis testing
    hypothesis_results = []
    for h in hypotheses:
        result = code_agent.test_hypothesis(h, documents, queries, current_code, client)
        hypothesis_results.append(result)
    _log_hypotheses(i, hypothesis_results)      # logs/iteration_{i}_hypotheses.json

    # Final code generation
    proven = [r for r in hypothesis_results if r.proven]
    if proven:
        final_code = code_agent.generate_final_code(
            analysis_result.summary, proven, current_code
        )
        _log_final_code(i, final_code)          # logs/iteration_{i}_final_code.py
        _write_preprocess(final_code)
    else:
        print("[agent] No proven hypotheses — preprocess.py unchanged")

run_eval()  # final eval
Logging Structure
All logs in src/agents/analysis_code_agent/logs/:


logs/
├── iteration_0_analysis.log           # Full turn-by-turn conversation (bash + output)
├── iteration_0_analysis_summary.txt   # Just the final summary text
├── iteration_0_hypotheses.json        # All hypotheses with full details:
│                                      #   {id, description, rationale, code,
│                                      #    query_ids_to_test, falsifying_condition,
│                                      #    delta_recall_10, delta_ndcg_10, proven, notes}
├── iteration_0_final_code.py          # Full preprocess.py generated
├── iteration_0_eval.json              # run_eval() raw results (also saved by harness)
└── ...                                # (repeated for each loop iteration)
iteration_N_analysis.log format:


=== Analysis Agent | Iteration 0 | 2026-03-10T14:23:01 ===

--- CONTEXT SUMMARY ---
Failures (regressions): 3 queries
Hard negatives: 5 queries × 3 docs
Successes (hit but poor rank): 8 queries

--- TURN 0 ---
[ASSISTANT]: I'll start by examining the documents for failed query q_023...
<bash>python3 -c "import json; ..."</bash>

[BASH EXIT CODE: 0]
[BASH STDOUT]:
{"doc_id": "doc_123", "text": "Silent film actress..."}
...

--- TURN 1 ---
...

--- FINAL SUMMARY ---
## ANALYSIS SUMMARY
1. ...
2. ...
iteration_N_hypotheses.json format:


[{
  "id": "H1",
  "description": "...",
  "rationale": "...",
  "code": "...",
  "query_ids_to_test": ["q_023", "q_041"],
  "falsifying_condition": "...",
  "test_results": {
    "hypothesis_recall_10": 0.67,
    "baseline_recall_10": 0.50,
    "delta_recall_10": 0.17,
    "delta_ndcg_10": 0.08,
    "proven": true,
    "notes": "Improved 2/2 test queries"
  }
}]
config.yaml

analysis_model: "openai/gpt-4o-mini"
analysis_temperature: 0.3
code_model: "openai/gpt4o"
code_temperature: 0.7
api_base: "https://thekeymaker.umass.edu/"

server_port: 8765
server_persist_dir: ".bm25_cache"

max_hypotheses: 4
recall_improvement_threshold: 0.05
analysis_max_turns: 8
bash_timeout_seconds: 30
Error Handling
Error	Handling
Server not starting after 10s	Raise RuntimeError with helpful message
Bash timeout	Send [bash: timeout after Ns] to model, continue loop
Max analysis turns hit	Use last assistant text as summary; fallback summarization call
JSON parse failure (hypotheses)	Retry once with error correction; return empty list on 2nd failure
exec() raises on hypothesis code	Mark as errored (not proven), log exception
Hypothesis server index build fails	Mark as errored, log
Final code has no class Preprocessor	Keep current preprocess.py, log warning, skip _write_preprocess
LiteLLM API error	One retry after 5s; skip iteration on 2nd failure
run_eval() failure	Wrap in try/except, log traceback, continue loop
main.py Changes
Add "analysis_code_agent" to choices list
Add elif branch:

elif args.agent == "analysis_code_agent":
    from src.agents.analysis_code_agent import AnalysisCodeAgent
    agent = AnalysisCodeAgent()
Verification

# Add dependencies
uv add fastapi uvicorn httpx

# Run for 1 loop
uv run python main.py --agent analysis_code_agent --loops 1

# Check server is running separately if desired
uv run python src/agents/analysis_code_agent/bm25_server.py --port 8765

# Inspect logs
cat src/agents/analysis_code_agent/logs/iteration_0_analysis.log
cat src/agents/analysis_code_agent/logs/iteration_0_hypotheses.json
cat src/agents/analysis_code_agent/logs/iteration_0_final_code.py

# Verify preprocess.py is valid
uv run python src/evaluation/scripts/test_preprocessing.py --agent analysis_code_agent
Implementation Order
bm25_server.py — FastAPI server (can test standalone)
bm25_client.py — HTTP client
eval_utils.py — subset eval using client
preprocess.py — baseline passthrough (name = "analysis_code_agent")
config.yaml
context/ANALYSIS_SYSTEM.md and context/CODE_SYSTEM.md
analysis_agent.py
code_agent.py
agent.py
__init__.py
main.py + src/agents/__init__.py