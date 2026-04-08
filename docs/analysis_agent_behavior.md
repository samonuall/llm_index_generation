# AnalysisAgent — High-Level Behavior Document

## Overview

The `AnalysisAgent` is the diagnostic half of a two-stage agent loop. It receives evaluation results from a BM25 retrieval system, investigates why specific queries fail via a multi-turn LLM conversation with structured tool calls, and produces a structured summary of failure patterns. This summary is then consumed by the downstream `CodeAgent`, which generates preprocessing hypotheses to fix the identified problems.

The analysis agent does **not** write or modify code. Its sole output is a natural-language analysis summary with concrete evidence (query IDs, doc IDs, failure patterns).

---

## Current Data Flow: How Inputs Are Determined

### 1. Corpus Construction (happens in `agent.py:_load_data`)

The corpus is **not** the full 1M+ document Wikipedia dataset. Before any agent runs, the orchestrator (`AnalysisCodeAgent.run()`) constructs a **sampled corpus**:

- **Split**: Determined by config (e.g. `"tip_of_the_tongue"`). NOT hardcoded — the same `split` value flows through the orchestrator to data loading, evaluation, and the analysis agent.
- **Queries**: All queries from `data/<split>/queries.jsonl` are loaded (currently ~50 EvalQuery objects).
- **Relevant documents**: Every document whose `doc_id` appears in any query's `relevant_doc_ids` is **always kept** (guarantees gold answers exist in the corpus).
- **Distractor documents**: From the remaining documents in `documents.jsonl`, up to `max_distractors` (configurable, default 9000) are selected via **reservoir sampling** with a configurable seed (default 42). This gives a realistic but fast eval corpus (~10K docs total).
- The final corpus is shuffled and stored as `self._documents`.

**Key invariant**: The analysis agent and the coding agent MUST operate on the exact same document corpus and query set. The `split`, `seed`, and `max_distractors` config values that determine corpus construction flow from a single config source to both agents.

### 2. Evaluation (happens before analysis agent is called)

Before the analysis agent runs, the orchestrator:

1. **Runs the eval harness** (`run_eval`) — preprocesses all documents with the current `preprocess.py`, builds a BM25 index on the server, and computes recall@10, recall@100, nDCG@10.
2. **Enriches results** (`_enrich_eval_results`) — queries the BM25 server's "current" index for every query, producing per-query data: `hit` (bool), `rank` (int or None), `retrieved_doc_ids` (top results), `relevant_doc_ids` (gold answers), `query_text`.

### 3. Baseline Results (current implementation — flagged for change)

**Currently**: Baseline results are loaded from a static `src/agents/baseline_results.json` file at the start of `run()`. This file contains metrics from a previous run of the baseline preprocessor against a potentially different corpus.

**Problem**: The baseline was computed on a different document set than the current loop's sampled corpus, making metric deltas (current vs baseline) not directly comparable.

**Planned change (R1.B)**: Baseline should be computed on the same sampled corpus as the current eval, not loaded from a static file.

---

## Inputs to `AnalysisAgent.analyze()`

The `analyze()` method receives structured data as arguments. It does NOT receive the full document or query lists — instead, it investigates data selectively through its tools.

| Parameter | Type | Source | Description |
|---|---|---|---|
| `eval_results` | `dict` | `run_eval()` + `_enrich_eval_results()` | Current iteration's metrics and per-query results. Contains `metrics` (recall@100, nDCG@10, recall@10) and `query_results` (list of per-query dicts with `hit`, `rank`, `query_text`, `relevant_doc_ids`, `retrieved_doc_ids`). |
| `baseline_results` | `dict` | Computed on same corpus (planned) | Baseline metrics for comparison: `recall_at_k`, `ndcg`. Used to identify regressions and compute deltas. Must be computed on the same sampled corpus as `eval_results`. |
| `current_code` | `str` | Read from `preprocess.py` | The current preprocessing code as a string. Included in the LLM context so the analysis agent understands what transformations are being applied. |
| `client` | `BM25Client` | Orchestrator | BM25 server client. Used internally by the `bm25_retrieve` tool to query the index. |
| `split` | `str` | Config | Dataset split name. Determines which `data/<split>/` directory the file reader and grep tools operate on. Must come from config, not hardcoded. |
| `journal_summary` | `str \| None` | `RunJournal.summary_for_prompt()` | If `use_history=True`, a structured summary of past iterations: score history, persistent failures, overfitting cases, hypothesis outcomes. `None` if history mode is disabled or first iteration. |

**Removed parameters** (planned): `documents` and `queries` lists should NOT be passed to `analyze()`. The analysis agent reads documents and queries selectively using its tools against the data files on disk. This avoids passing large lists through the LLM context and ensures the agent works with the same on-disk data as the rest of the pipeline.

### What the analysis agent does with these inputs

The analysis agent's `analyze()` method processes inputs into three intermediate structures before the LLM conversation begins:

1. **Candidates** (via `_build_candidates`): Categorizes `query_results` into:
   - **Failures/regressions**: Up to 5 queries where baseline had `hit=True` but current has `hit=False`
   - **Hard negatives**: Up to 5 missed queries with up to 3 wrong doc IDs each
   - **Successes**: Up to 8 hit queries sorted by worst rank first

2. **Initial context message** (via `_build_initial_context`): A formatted string containing:
   - Journal summary (if provided)
   - Current metrics vs baseline metrics (4 decimal places)
   - Current `preprocess.py` code in a fenced Python block
   - Formatted candidate sections (regressions, hard negatives, successes)
   - Tool descriptions and usage instructions
   - Data file paths for the split

3. **System prompt**: Loaded from `context/ANALYSIS_SYSTEM.md` during `__init__`, not per-call.

---

## Outputs from `AnalysisAgent.analyze()`

Returns an `AnalysisResult` dataclass:

| Field | Type | Description |
|---|---|---|
| `summary` | `str` | The final structured analysis summary. Contains identified failure patterns (using the taxonomy from ANALYSIS_SYSTEM.md), specific query/doc IDs as evidence, and concrete preprocessing recommendations. This is the primary output consumed by the CodeAgent. |
| `turns` | `int` | Number of assistant messages in the conversation (not loop iterations). Useful for tracking how much investigation the agent performed. |
| `conversation` | `list[dict]` | The full message history: system prompt, initial user context, all assistant responses, tool call results, nudge messages, and the final summary request. Used for logging and debugging. |

### How the summary is consumed

The orchestrator passes `analysis_result.summary` directly to:
- `CodeAgent.generate_hypotheses(analysis_summary, current_code, ...)` — the code agent uses this summary to generate preprocessing hypotheses
- Log files: `iteration_N_analysis_summary.txt` (summary only) and `iteration_N_analysis.log` (full conversation)

---

## Tools Available to the Analysis Agent

The analysis agent uses LiteLLM's OpenAI-compatible tool-calling API. The LLM receives tool schemas via the `tools` parameter and invokes them via structured `tool_calls` in its responses. The agent infrastructure dispatches each call to the corresponding Python function and returns the result as a tool-result message.

**No bash access.** The agent cannot execute arbitrary shell commands. All data access is through the three structured tools below, each sandboxed to `data/<split>/`.

### Tool 1: `bm25_retrieve`

Query the current BM25 index to see what documents are retrieved for a given query.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | str | yes | — | The query text to search |
| `top_k` | int | no | 10 | Number of results to return |
| `index_name` | str | no | `"current"` | Which BM25 index to query |

**Returns**: JSON list of results, each with `doc_id`, `score`, `rank`.

**Implementation**: Wraps `BM25Client.retrieve()`. When the BM25Client changes, this tool's behavior changes accordingly — no separate curl logic to maintain.

### Tool 2: `read_file`

Read file contents from the data directory. Equivalent to `cat` with a character limit.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file_path` | str | yes | — | Path relative to `data/<split>/` |
| `max_chars` | int | no | 800 | Maximum characters to return |
| `filter_id` | str | no | — | For JSONL files: return only lines where `doc_id` or `query_id` matches |

**Returns**: File contents as string, truncated with `... [truncated]` if exceeding `max_chars`.

**Path restriction**: Only paths under `data/<split>/` are allowed. Paths containing `..`, absolute paths, or symlinks resolving outside the data directory are rejected with an error.

### Tool 3: `grep_search`

Search file contents using regex patterns.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `pattern` | str | yes | — | Regex pattern to search for |
| `file_path` | str | yes | — | Path relative to `data/<split>/` |
| `max_results` | int | no | 10 | Maximum matching lines to return |

**Returns**: Matching lines with line numbers, up to `max_results`.

**Path restriction**: Same as `read_file` — only `data/<split>/` paths allowed.

### What the agent does NOT have access to

- **No bash/shell execution**: Cannot run arbitrary commands. All operations go through the three tools above.
- **No file access outside `data/<split>/`**: Cannot read `preprocess.py`, `src/`, config files, or any other project files at runtime. The current preprocessing code is provided in the initial context message instead.
- **No code modification**: Cannot write or modify any file. Its role is purely diagnostic.
- **No index building**: Cannot create or rebuild BM25 indexes. It queries the existing "current" index via the `bm25_retrieve` tool.

---

## Multi-Turn Conversation Flow

```
Orchestrator                    AnalysisAgent                         LLM
    |                                |                                 |
    |-- analyze(eval, baseline, ...) |                                 |
    |                                |-- _build_candidates()           |
    |                                |-- _build_initial_context()      |
    |                                |                                 |
    |                                |-- [system prompt + ctx + tools] |
    |                                |-------------------------------->|
    |                                |                                 |
    |                                |    tool_calls: [bm25_retrieve]  |
    |                                |<--------------------------------|
    |                                |                                 |
    |                                |-- dispatch bm25_retrieve(...)   |
    |                                |-- [tool result message]         |
    |                                |-------------------------------->|
    |                                |                                 |
    |                                |    tool_calls: [read_file, ...] |
    |                                |<--------------------------------|
    |                                |                                 |
    |                                |-- dispatch read_file(...)       |
    |                                |-- [tool result messages]        |
    |                                |-------------------------------->|
    |                                |                                 |
    |                                |    (repeats for up to max_turns)|
    |                                |    (nudges if < min_tool_turns) |
    |                                |                                 |
    |                                |    <summary>...</summary>       |
    |                                |<--------------------------------|
    |                                |                                 |
    |                                |-- _request_summary() if needed  |
    |                                |                                 |
    |<-- AnalysisResult              |                                 |
```

### Loop behavior summary

1. LLM receives system prompt + initial context (candidates, metrics, code) + tool schemas
2. LLM responds with tool calls and/or text content
3. If tool calls present: dispatch each tool, append results as tool-result messages, increment tool turn counter
4. If no tool calls and `tool_turns_completed < min_tool_turns` (configurable, default 3): append nudge message asking LLM to investigate more
5. If no tool calls and text contains `<summary>` tag: extract summary, exit loop
6. If no tool calls, no `<summary>`, and enough tool turns done: exit loop
7. After loop: call `_request_summary()` only if no `<summary>` tag was found in any assistant message
8. Return `AnalysisResult(summary, turns, conversation)`

---

## Configuration

All configuration comes from `config.yaml` and can be overridden in the config dict passed to `__init__`:

| Key | Default | Description |
|---|---|---|
| `analysis_model` | `"openai/gpt-4o-mini"` | LiteLLM model identifier for analysis LLM calls |
| `analysis_temperature` | `0.3` | Temperature for analysis LLM calls |
| `analysis_max_turns` | `8` | Maximum loop iterations |
| `server_port` | `8765` | BM25 server port (used by BM25Client) |
| `api_base` | `"https://thekeymaker.umass.edu/"` | LiteLLM API base URL |
| `min_tool_turns` | 3 (currently hardcoded) | Minimum tool-using turns before accepting summary (should be configurable) |
| `split` | `"tip_of_the_tongue"` (from AgentRunner) | Dataset split — determines data directory and query/document files |
| `max_distractors` | `9000` | Number of non-relevant documents in sampled corpus |
| `seed` | `42` (currently hardcoded) | Random seed for corpus sampling (should be configurable) |

---

## What Needs to Change (from Requirements Plan)

| Area | Current Behavior | Planned Change |
|---|---|---|
| Baseline source | Static `baseline_results.json` | Compute baseline on same sampled corpus |
| Tool protocol | XML `<bash>` tags parsed from text | LiteLLM tool-calling API with structured tool schemas |
| Available tools | Arbitrary bash execution | Three sandboxed tools: `bm25_retrieve`, `read_file`, `grep_search` |
| BM25 access | curl commands to HTTP server | `bm25_retrieve` tool wrapping `BM25Client.retrieve()` |
| Data access | Bash commands reading any file | `read_file` and `grep_search` restricted to `data/<split>/` |
| Response parsing | Infers summary from absence of `<bash>` | Require explicit `<summary>` XML tags in text content |
| Summary request | Always called after loop | Only called if no `<summary>` tag found in conversation |
| `analyze()` params | Receives `documents`, `queries`, `client` | Remove `documents`/`queries` params; keep `client` for tool use |
| Split | Comes from `AgentRunner.split` class attr | Must flow from config to both agents |
| Seed | Hardcoded to 42 | Configurable via config |
| Failures cap | No limit on regressions | Cap at 5 |
| `min_tool_turns` | Hardcoded to 3 | Configurable via config |
| `_extract_summary()` | Dead code (never called) | Remove |
| `_run_bash()` | Executes arbitrary shell commands | Remove — replaced by structured tools |
