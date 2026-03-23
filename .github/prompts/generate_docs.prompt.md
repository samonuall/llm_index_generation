---
mode: agent
description: Generate source-of-truth documentation for analysis_code_agent
---

Go through each major component of the agent setup in `src/agents/analysis_code_agent/`. Ask yourself questions and answer them by reading the code (for example: how does the run journal insert its data into prompts? how is the run journal stored? in what order does analysis and code happen? how are the best hypotheses chosen to implement with the coding agent?).

Write a source-of-truth file at `docs/analysis_code_agent.md` containing the most up-to-date description of the agentic system and its eval loop. Break down the system into components with:
- Justification for each component
- A tag indicating where to find the implementation (file path, and line range if useful)

At the top of the document include:
1. An ASCII diagram of the full system flow
2. An executive summary explaining the whole system and why it works

Cover these components:

**Analysis Agent** (`src/agents/analysis_code_agent/analysis_agent.py`)
- Multi-turn bash loop: how turns work, when it stops, how bash output feeds back in
- How the initial context is built (eval results, failures, hard negatives, successes)
- How the final summary is extracted or requested

**Coding Agent** (`src/agents/analysis_code_agent/code_agent.py`)
- Hypothesis generation: how many, what format (markdown blocks), retry logic
- How past hypotheses are passed in to avoid repetition
- Hypothesis testing: how each hypothesis is evaluated, what queries are used
- Selection rule: exactly how the best hypothesis is chosen and adopted

**Run Journal / Logging** (`src/agents/analysis_code_agent/agent.py` `_log_*` methods, `logs/` directory)
- What is logged per iteration (eval JSON, analysis log, hypotheses JSON, final code)
- How logs feed back into subsequent iterations (past_hypotheses list)

**Feedback mechanism**
- How harness eval results (recall@100, nDCG@10) are passed to the analysis agent
- How per-query hit/rank data is enriched via the BM25 server
- How analysis summary flows into hypothesis generation prompt

**Evaluation pipeline** (`src/agents/analysis_code_agent/eval_utils.py`, `src/evaluation/scripts/test_preprocessing_split.py`)
- Exact metrics: Recall@100, Recall@10, nDCG@10 — how each is computed
- Difference between harness eval (authoritative, runs full pipeline) and subset eval (fast, via BM25 server)
- How nDCG@10 is computed for single-relevant-doc queries

**BM25 retrieval server** (`src/agents/analysis_code_agent/bm25_server.py`, `bm25_client.py`)
- FastAPI server with named in-memory indexes
- Tokenization: lowercase only (`t.lower()`), then `bm25s.tokenize`
- MaxP aggregation: best chunk score per doc across all chunks
- Endpoints used: build, batch_retrieve, delete
- How the "current" index and per-hypothesis indexes are managed

**Hypothesis testing and selection** (`src/agents/analysis_code_agent/agent.py`, `code_agent.py`)
- Each hypothesis gets its own temp index on the server, compared against "current"
- Decision rule: adopt if `delta_recall_100 > 0` (best hypothesis by `hypothesis_recall_100`)
- `proven` flag: `delta_recall_100 >= recall_improvement_threshold` (from config)
- Global best tracking: how `best_recall_100` and `best_code` are updated across loops
- Final restore: if later loops regress, best code is restored before final eval

**Global best tracking** (`src/agents/analysis_code_agent/agent.py` `run()`)
- How `best_recall_100` is initialized (from `baseline_results.json`)
- How it is updated optimistically using `loop_start_recall_100 + delta_recall_100`
- The final restore logic that guards against regression

For each component make explicit:
- The exact metrics used for feedback and decisions at each step
- Any validation or self-reflection steps (e.g. parse retry in code agent, summary request in analysis agent)

Focus only on `analysis_code_agent`. Do not describe other agents.
Create the `docs/` directory if it does not exist.
