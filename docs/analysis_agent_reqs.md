# AnalysisAgent Requirements — Behavioral Contracts for Testing

## Context

The `AnalysisAgent` (in `src/agents/analysis_code_agent/analysis_agent.py`) is the diagnostic component of the two-stage analysis→code agent loop. It runs a multi-turn LLM conversation with structured tool calls to investigate BM25 retrieval failures, then produces a structured summary for the downstream CodeAgent. We need a clear requirements doc to (a) understand the component at a high level and (b) design tests around its fundamental capabilities and contracts.

The output of this plan is a requirements document (not code). The test stubs in `tests/test_analysis_agent.py` already exist and will be filled in based on these requirements.

---

## R1: Initialization & Configuration

**File:** `analysis_agent.py:32-44`

| Requirement | Contract |
|---|---|
| R1.1 Config defaults | If a config key is missing, these defaults apply: `analysis_model="openai/gpt-4o-mini"`, `analysis_temperature=0.3`, `analysis_max_turns=8`, `server_port=8765`, `api_base="https://thekeymaker.umass.edu/"`, `max_distractors=9000`, `seed=42` |
| R1.2 Config override | Every default can be overridden by passing the key in `config` dict. This includes `split`, `max_distractors`, and `seed`. |
| R1.3 System prompt loading | Constructor reads `context/ANALYSIS_SYSTEM.md` from the agent directory; failure to find this file is a hard error |
| R1.4 Context files validated | All context files fed into the analysis agent must be verified as loaded and injected correctly: (a) `ANALYSIS_SYSTEM.md` as the system prompt, (b) the initial user message built by `_build_initial_context` containing eval results, current code, candidate targets, and tool descriptions. Each context source should be testable independently. |
| R1.5 Tracker optional | `tracker=None` is valid; LLM calls succeed without a tracker |
| R1.6 Split from config | The data split (e.g. `"tip_of_the_tongue"`) must come from the orchestrator's config, NOT be hardcoded. The analysis agent receives the `split` parameter and uses it to construct data paths. The same split must be used by the orchestrator for data loading, eval, and analysis. |
| R1.7 Corpus size configurable | `max_distractors` (total non-relevant docs in the sampled corpus) must be configurable via config. Default 9000. |
| R1.8 Seed configurable | The random seed used for corpus sampling must be configurable via config. Default 42. |

---

## R1.B: Baseline Results Source

**Design change:** The analysis agent must NOT depend on a static `baseline_results.json` file. Instead, the `baseline_results` passed to `analyze()` must come from evaluating the baseline preprocessor against the **current loop's corpus** (the same sampled document set used for the current iteration). This ensures the baseline comparison is consistent with the corpus the agent is actually analyzing.

| Requirement | Contract |
|---|---|
| R1.B.1 Baseline from current corpus | The `baseline_results` dict passed to `analyze()` must be computed on the same document set as `eval_results`. The analysis agent should not read or depend on `src/agents/baseline_results.json`. |
| R1.B.2 Baseline metrics are comparable | Because both baseline and current eval use the same corpus sample, metric deltas (recall, nDCG) are directly comparable without corpus-size bias. |

---

## R1.C: Corpus Consistency

**Critical invariant:** The analysis agent must operate on the **exact same** document corpus and query set as the coding agent. There must be no scenario where the analysis agent sees different documents or queries than what the coding agent's hypotheses are evaluated against.

| Requirement | Contract |
|---|---|
| R1.C.1 Same document corpus | The documents available to the analysis agent (via its tools) must be identical to the documents used by the coding agent for hypothesis testing. Both agents operate on the same sampled corpus (all relevant docs + sampled distractors). |
| R1.C.2 Same query set | The queries the analysis agent investigates must be the same queries used for evaluation. No separate query loading or filtering. |
| R1.C.3 No direct document/query injection | The analysis agent does NOT receive the full `documents` or `queries` lists as parameters. Instead, it reads documents and queries selectively using its tools (file reader, grep) against the data files on disk. This avoids passing large lists through the LLM context. |
| R1.C.4 Single source of truth | The `split`, `seed`, and `max_distractors` config values that determine corpus construction must flow from a single config source to both agents. |

---

## R2: Candidate Building (`_build_candidates`)

**File:** `analysis_agent.py:244-285`

**Inputs:** `eval_results` dict (with `query_results` list), `baseline_results` dict (with `query_results` list)

| Requirement | Contract |
|---|---|
| R2.1 Failures = regressions only, capped at 5 | A query is a "failure" iff baseline had `hit=True` AND current has `hit=False`. Queries that both miss are NOT failures. Returns at most 5 failures. **Design change: current code has no cap on failures; needs to limit to 5.** |
| R2.2 Hard negatives from misses | Takes up to 5 missed queries (current `hit=False`). For each, collects up to 3 retrieved doc_ids that are NOT in `relevant_doc_ids`. Only includes queries that have at least one wrong doc. |
| R2.3 Successes = hits by worst rank | Takes queries with `hit=True`, sorts by rank descending (worst first), returns up to 8. |
| R2.4 Empty inputs | If `query_results` is empty or missing, all three categories return empty lists. |
| R2.5 No baseline match | If a query_id exists in current results but not baseline, it is never classified as a failure. |
| R2.6 Return structure | Returns `{"failures": [...], "hard_negatives": [...], "successes": [...]}` — always all three keys. |

---

## R3: Initial Context Building (`_build_initial_context`)

**File:** `analysis_agent.py:287-399`

| Requirement | Contract |
|---|---|
| R3.1 Journal inclusion | If `journal_summary` is not None, it appears in the output. If None, it's omitted (no empty section). |
| R3.2 Metrics display | Output contains current recall@100 and nDCG@10 alongside baseline values, formatted to 4 decimal places. |
| R3.3 Current code included | The `current_code` string appears inside a ```python fenced block. |
| R3.4 All candidate sections present | Output contains sections for regressions, hard negatives, and successes — even if empty (shows "none"). |
| R3.5 Tool descriptions | Output describes the available tools (BM25 retrieval, file reader, grep) and their parameters, so the LLM knows how to invoke them. No raw curl commands or server URLs — the tools abstract these away. |
| R3.6 Data path uses split | Data file paths referenced in context include the `split` parameter (e.g., `data/tip_of_the_tongue/`). The split value comes from config, not hardcoded. |

---

## R4: Multi-Turn Tool-Calling Loop (`analyze`)

**File:** `analysis_agent.py:46-121`

**Design change:** The analysis agent uses LiteLLM's OpenAI-compatible tool-calling API (`tools` parameter) instead of parsing XML `<bash>` tags from text responses. The LLM receives tool schemas and invokes them via structured function calls. This is safer (no arbitrary bash), easier to test (mock individual tools), and follows standard LLM tool-use patterns.

This is the core loop. Key behavioral contracts:

| Requirement | Contract |
|---|---|
| R4.1 Minimum tool turns enforced | If the LLM produces a response with no tool calls before completing `min_tool_turns` tool-using turns, the agent appends a nudge message and continues the loop (does NOT accept it as summary). `min_tool_turns` should be configurable via config (default 3). |
| R4.2 Nudge message content | The nudge tells the LLM how many tool-using turns have been done vs. required, and instructs it to investigate specific failing queries using the available tools. |
| R4.3 Multiple tool calls per turn | The LLM may invoke multiple tools in a single turn (LiteLLM/OpenAI tool-calling supports parallel tool calls). All tool calls in a single response are executed and their results returned as tool-result messages before the next LLM turn. |
| R4.4 Max turns cap | The loop runs at most `analysis_max_turns` iterations (default 8), regardless of tool turn count. |
| R4.5 Summary detection via `<summary>` tag | The LLM's text content (non-tool-call portion of the response) is checked for `<summary>...</summary>` XML tags. A summary is only accepted when `<summary>` tags are present — a non-tool-call response without `<summary>` is NOT treated as a summary. **Design change from current code which infers summary from absence of bash.** |
| R4.6 Conditional summary request | After the loop exits, `_request_summary()` is called ONLY if no `<summary>` tag was found in any assistant message during the run. If the LLM already produced a `<summary>`, that content is used directly. |
| R4.7 LLM failure terminates loop | If `_call_llm` returns `None`, the loop breaks immediately. |
| R4.8 Tool results appended correctly | Tool call results are appended as `{"role": "tool", "tool_call_id": "...", "content": "..."}` messages per the OpenAI tool-calling protocol. |
| R4.9 Conversation history | `AnalysisResult.conversation` contains the full message list including system, user (initial + nudges), assistant (with tool_calls), and tool result messages. |
| R4.10 Turn count | `AnalysisResult.turns` equals the number of assistant-role messages in the conversation (not loop iterations). |

---

## R5: Analysis Tools

**Design change:** Replaces `_run_bash()` with three structured tools exposed via LiteLLM's tool-calling API. Each tool is a Python function with a defined schema. This eliminates arbitrary bash execution entirely — the analysis agent can only perform the specific operations these tools allow.

All tools are sandboxed to `data/<split>/` — they cannot access files outside the data directory.

### R5.A: BM25 Retrieval Tool (`bm25_retrieve`)

Wraps `BM25Client.retrieve()` so the analysis agent can query the current BM25 index without raw curl commands.

| Requirement | Contract |
|---|---|
| R5.A.1 Parameters | `query` (str, required), `top_k` (int, optional, default 10), `index_name` (str, optional, default `"current"`) |
| R5.A.2 Returns | JSON-serialized list of results, each with `doc_id`, `score`, `rank`. Same data as the BM25 server's `/retrieve` endpoint. |
| R5.A.3 Uses BM25Client | Must call `BM25Client.retrieve()` internally — not subprocess/curl. When `BM25Client` changes, this tool's behavior changes accordingly. |
| R5.A.4 Error handling | If the client call fails, returns a structured error message (not an exception). |

### R5.B: File Reader Tool (`read_file`)

Reads file contents from the data directory. Equivalent to `cat` with an optional character limit, restricted to `data/<split>/`.

| Requirement | Contract |
|---|---|
| R5.B.1 Parameters | `file_path` (str, required — relative to `data/<split>/`), `max_chars` (int, optional, default 800) |
| R5.B.2 Path restriction | Only paths under `data/<split>/` are allowed. Any path that resolves outside this directory (via `..`, symlinks, or absolute paths) must be rejected with an error message. |
| R5.B.3 Returns | File contents as a string, truncated to `max_chars` with a `... [truncated]` notice if exceeded. |
| R5.B.4 JSONL-aware | When reading `.jsonl` files, supports an optional `filter_id` parameter to return only lines where `doc_id` or `query_id` matches the given value, avoiding full file reads. |
| R5.B.5 Error handling | Returns a structured error for missing files or access violations. |

### R5.C: Grep/Search Tool (`grep_search`)

Searches file contents using regex patterns, restricted to `data/<split>/`.

| Requirement | Contract |
|---|---|
| R5.C.1 Parameters | `pattern` (str, required — regex), `file_path` (str, required — relative to `data/<split>/`), `max_results` (int, optional, default 10) |
| R5.C.2 Path restriction | Same restriction as R5.B.2 — only paths under `data/<split>/`. |
| R5.C.3 Returns | Matching lines (up to `max_results`), each with line number and content. |
| R5.C.4 Error handling | Returns a structured error for invalid regex, missing files, or access violations. |

### R5.D: Tool Schema Registration

| Requirement | Contract |
|---|---|
| R5.D.1 OpenAI tool format | Tools are defined as OpenAI-compatible function schemas and passed via the `tools` parameter to `litellm.completion()`. |
| R5.D.2 Tool dispatch | When the LLM response contains `tool_calls`, the agent dispatches each call to the corresponding Python function, passing the parsed arguments. |
| R5.D.3 Unknown tool rejection | If the LLM calls a tool name not in the registered set, the agent returns an error message as the tool result (does not crash). |
| R5.D.4 Testability | Each tool function is independently callable and testable without LLM involvement. Tool functions are pure (no global state mutation) and can be tested with synthetic inputs. |

---

## R6: LLM Call & Error Recovery (`_call_llm`)

**File:** `analysis_agent.py:123-157`

| Requirement | Contract |
|---|---|
| R6.1 Tracker recording | If tracker is provided, every successful LLM call records wall_time and response via `tracker.record_llm_call()`. |
| R6.2 Tools passed to LLM | Every `completion()` call includes the `tools` parameter with the registered tool schemas (R5.D.1). |
| R6.3 Content policy retry | On ContentPolicyViolation (detected by exception class name or "content_policy"/"content management policy" in message), messages are sanitized and the call is retried once. |
| R6.4 In-place sanitization | On content policy error, the original `messages` list is mutated in-place (`messages[:] = sanitized`) so future turns use sanitized history. |
| R6.5 Generic retry | On non-policy errors, waits 5 seconds then retries once with original messages. |
| R6.6 Returns None on exhausted retries | If all retry attempts fail, returns `None` (does not raise). |
| R6.7 Empty response handling | If LLM returns empty content, returns `""` (not None). |

---

## R7: Message Sanitization (`_sanitize_messages`)

**File:** `analysis_agent.py:159-170`

| Requirement | Contract |
|---|---|
| R7.1 Only tool results affected | Only tool-result messages (`role="tool"`) are candidates for truncation. |
| R7.2 Length threshold | Tool result messages over 400 chars are truncated to 400 chars + truncation notice. Messages ≤400 chars are unchanged. |
| R7.3 Non-tool messages preserved | System, user, and assistant messages pass through unchanged regardless of content. |
| R7.4 New list returned | Returns a new list; does not mutate the input list's message dicts (creates new dicts via `{**m, "content": ...}`). |

---

## R8: Summary Request (`_request_summary`)

**File:** `analysis_agent.py:202-242`

| Requirement | Contract |
|---|---|
| R8.1 Summary prompt appended | Appends a user message requesting the final summary to the messages list. |
| R8.2 Tool-call guardrail | If the LLM's summary response contains tool calls, the agent retries once with a stricter "no tool calls, text summary only" instruction. |
| R8.3 Fallback on exception | If the LLM call raises any exception, returns `"Analysis failed to produce summary."`. |
| R8.4 Fallback on empty | If LLM returns empty/None content, returns `"No summary generated."`. |
| R8.5 Messages mutated | The summary prompt and response are appended to the messages list (side effect — this is intentional so the conversation log is complete). |

---

## Gaps & Implementation Notes

| Item | Status | Note |
|---|---|---|
| `min_tool_turns` hardcoded to 3 | **Gap** | Should be added to config.yaml and read in `__init__`. Tests should use the configurable value. |
| Split hardcoded to `tip_of_the_tongue` | **Gap** | Split comes from `AgentRunner.split` class attribute. Should flow from config to both orchestrator and analysis agent. Not hardcoded anywhere. |
| `seed` not configurable | **Gap** | `_load_data` uses hardcoded `seed=42`. Should come from config. |
| `max_distractors` partially configurable | **Gap** | Already in config.yaml but should be documented as part of the analysis agent's contract (R1.7). |
| Failures uncapped (R2.1) | **Design change** | Current code returns all regressions. Needs cap at 5 to match hard negatives. |
| Bash → structured tools (R5) | **Design change** | Current code uses `_run_bash()` with `subprocess.run(shell=True)`. Replace entirely with three structured tools (`bm25_retrieve`, `read_file`, `grep_search`) exposed via LiteLLM tool-calling API. Delete `_run_bash()`. |
| XML bash protocol → LiteLLM tool-calling (R4) | **Design change** | Current code parses `<bash>` XML tags from text responses. Replace with LiteLLM's `tools` parameter and `tool_calls` response handling. System prompt updated to describe tools instead of bash syntax. |
| Summary detection via `<summary>` tag (R4.5) | **Design change** | Current code infers summary from absence of bash. Needs update to require explicit `<summary>` tags in text content. |
| Conditional summary request (R4.6) | **Design change** | Current code always calls `_request_summary()`. Needs conditional check: only call if no `<summary>` tag found in conversation. |
| Baseline from current corpus (R1.B) | **Design change** | Current code accepts `baseline_results` from static JSON. Caller must pass baseline computed on same corpus as current eval. Analysis agent should not depend on `baseline_results.json`. |
| Corpus consistency (R1.C) | **Design change** | Need to verify analysis agent uses same docs/queries as coding agent. Remove `documents` and `queries` parameters from `analyze()` — agent reads selectively via tools. |
| `_extract_summary()` | **Remove** | Dead code — not called in `analyze()`. Should be deleted from implementation. |
| `_run_bash()` | **Remove** | Replaced by structured tools (R5.A, R5.B, R5.C). |
| `read_documents.py` analysis tool | **Remove/Replace** | Functionality absorbed by the `read_file` tool (R5.B). The standalone script may be kept for manual debugging but is no longer invoked by the agent. |

---

## Test Design Mapping

| Requirement Group | Test Strategy |
|---|---|
| R1 (Config) | Unit: construct with partial/empty config, verify defaults (including split, seed, max_distractors). Verify override. Verify all context files loaded. |
| R1.B (Baseline) | Unit: verify baseline_results passed to analyze() are from same corpus, not static JSON. |
| R1.C (Corpus consistency) | Integration: verify analysis agent's tools read from the same `data/<split>/` directory used by eval. Verify `documents`/`queries` are NOT passed as parameters to `analyze()`. |
| R2 (Candidates) | Unit: synthetic eval_results/baseline_results dicts → verify categorization. Verify failures capped at 5. Edge cases: empty results, no baseline match, all hits, all misses, >5 regressions. |
| R3 (Context) | Unit: call with known inputs → assert substrings in output. Test journal present/absent. Verify tool descriptions present instead of curl commands. |
| R4 (Loop) | Integration w/ mocked LLM: script responses with tool_calls and `<summary>` tags. Verify: parallel tool calls dispatched correctly, nudge behavior, summary detection via `<summary>` tag, conditional `_request_summary()` call. |
| R5.A (BM25 tool) | Unit: mock BM25Client → verify tool calls `client.retrieve()`, returns correct format, handles errors. |
| R5.B (File reader) | Unit: create temp data dir → verify reads, truncation, JSONL filtering, path restriction (reject `..`, absolute paths, paths outside `data/<split>/`). |
| R5.C (Grep tool) | Unit: create temp data files → verify regex matching, max_results cap, path restriction. |
| R5.D (Tool registration) | Unit: verify tool schemas are valid OpenAI format, dispatch routes correctly, unknown tools return error. |
| R6 (LLM calls) | Unit w/ mocked litellm: simulate ContentPolicyViolation, generic errors, success → verify retry logic, tracker calls, tools parameter present, None return. |
| R7 (Sanitize) | Unit: craft message lists with tool results of varying lengths → verify truncation logic. |
| R8 (Summary) | Unit w/ mocked LLM: simulate tool-calls-in-summary, empty response, exception → verify guardrail and fallbacks. Only called when no `<summary>` tag found. |

---

## Critical Files

- **Implementation:** `src/agents/analysis_code_agent/analysis_agent.py`
- **Tools (new):** `src/agents/analysis_code_agent/analysis_tools/` — tool function implementations for `bm25_retrieve`, `read_file`, `grep_search`
- **BM25 client (wrapped by tool):** `src/agents/analysis_code_agent/bm25_client.py`
- **System prompt:** `src/agents/analysis_code_agent/context/ANALYSIS_SYSTEM.md`
- **Config:** `src/agents/analysis_code_agent/config.yaml`
- **Test file:** `tests/test_analysis_agent.py`
- **Fixtures:** `tests/conftest.py`

## Verification

- Run `uv run pytest tests/test_analysis_agent.py -v` after implementing tests
- Each requirement should map to at least one test case
- Tests should use mocked LLM (no real API calls) and mocked BM25Client (no real server)
- Tool functions should be tested independently with synthetic data directories
