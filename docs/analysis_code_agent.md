# analysis_code_agent Source of Truth

## Full System Flow (ASCII)

```text
baseline_results.json
        |
        v
+---------------------------+
| AnalysisCodeAgent.run()   |
| (n loops)                 |
+---------------------------+
        |
        | load sampled corpus (all relevant docs + sampled distractors)
        | start BM25 FastAPI server
        v
+---------------------------+
| Loop i starts             |
+---------------------------+
        |
        | run_eval(): preprocess.py -> chunks
        | build server index "harness_eval"
        | run_subset_eval(all queries)
        v
+---------------------------+
| loop_start metrics        |
| recall@10, recall@100,    |
| nDCG@10                   |
+---------------------------+
        |
        | rebuild server index "current"
        | enrich query-level hit/rank via server
        | record iteration in RunJournal
        v
+---------------------------+
| AnalysisAgent.analyze()   |
| multi-turn LLM + <bash>   |
| -> analysis summary       |
+---------------------------+
        |
        | summary + current code + run journal + past hypotheses
        v
+---------------------------+
| CodeAgent.generate_...    |
| -> H1..Hk hypotheses      |
+---------------------------+
        |
        | for each H:
        |   validate code
        |   build temp index "hyp_H*"
        |   subset eval vs "current"
        |   compute deltas
        |   delete temp index
        v
+---------------------------+
| Select best hypothesis    |
| by hypothesis_recall_100  |
+---------------------------+
        |
        | if delta_recall_100 > 0:
        |   adopt best directly, or
        |   synthesize if multiple proven
        |   write preprocess.py
        |   update global best estimate
        | else keep current code
        v
+---------------------------+
| end loops                 |
+---------------------------+
        |
        | restore globally best code if needed
        | final eval
        v
final reported improvement
```

## Executive Summary

`analysis_code_agent` is a two-stage iterative optimizer for retrieval preprocessing:

1. It runs an eval loop to measure current preprocessing quality (`Recall@100`, `Recall@10`, `nDCG@10`).
2. It asks an analysis model to inspect concrete failures using a constrained multi-turn bash workflow.
3. It asks a coding model to generate multiple full-code hypotheses, tests each hypothesis on the same query set, and keeps only improvements.
4. It tracks a global best code snapshot across loops and restores it before final eval to avoid ending in a regression.

Why this works:
- Analysis is grounded in observed failures (query-level misses/ranks), not just aggregate metrics.
- Hypotheses are experimentally verified against a shared baseline index (`current`) before adoption.
- The decision rule is strict and metric-based (`delta_recall_100 > 0` for adoption).
- Long-horizon memory (run journal + past hypotheses + persistent failures) reduces repeated bad ideas.
- Final global-best restore gives monotonic safety across loops.

---

## Analysis Agent

Implementation tag: `src/agents/analysis_code_agent/analysis_agent.py:45-120,223-377`

### What it does
- Runs a multi-turn conversation where assistant outputs can include `<bash>...</bash>`.
- If no bash block appears before minimum investigation depth, it is explicitly nudged to keep investigating.
- Produces a final summary used as direct input to hypothesis generation.

### Multi-turn bash loop mechanics
- `analyze()` builds initial messages then iterates for `analysis_max_turns` turns.
- `min_bash_turns = 3` is enforced before allowing summary-only completion.
- Each assistant bash block is executed with timeout (`bash_timeout_seconds`) and output is injected back as a user message.
- Loop stops when assistant returns non-bash content after minimum bash turns, or when max turns / call failures stop progress.

Implementation tag: `src/agents/analysis_code_agent/analysis_agent.py:71-109,170-190`

### Initial context construction
- Candidate failures are built from enriched query results:
  - Regressions: baseline hit but current miss.
  - Hard negatives: misses with wrong top docs.
  - Successes: hits sorted by worst rank first.
- Initial prompt includes:
  - Current metrics vs baseline (`Recall@100`, `nDCG@10`).
  - Current `preprocess.py`.
  - Candidate sections above.
  - BM25 server commands and data file pointers.
  - Optional run-journal summary prepended when available.

Implementation tag: `src/agents/analysis_code_agent/analysis_agent.py:223-264,266-377`

### Final summary extraction / fallback
- First tries to extract last assistant message that does not contain `<bash>`.
- If none is found, explicitly asks for a no-bash structured summary.
- Also includes a sanitization retry path to truncate bash outputs on content-policy failures.

Implementation tag: `src/agents/analysis_code_agent/analysis_agent.py:112-116,142-168,192-221`

### Justification
- The mandatory bash investigation creates evidence-based analysis.
- Structured candidate construction focuses the model on actionable retrieval errors.
- Summary fallback and sanitization increase robustness in long conversations.

---

## Coding Agent

Implementation tag: `src/agents/analysis_code_agent/code_agent.py:61-233,255-307,329-430,432-518`

### Hypothesis generation
- Single LLM call asks for exactly `n` full `preprocess.py` hypotheses (default from config: 4).
- Required format is markdown blocks:
  - `### Hk: ...`
  - `Rationale:`
  - `Query IDs:`
  - `Falsifying:`
  - Python code block with complete implementation.
- Parser first tries markdown-block parser, then JSON parser.
- If parse fails, it retries with stricter formatting instructions.

Implementation tag: `src/agents/analysis_code_agent/code_agent.py:61-154,184-223,236-304`

### Passing past hypotheses to avoid repetition
- Prompt includes a "Previously Tested Hypotheses" section when history exists.
- It includes each prior hypothesis + deltas + proven flag.
- Adds pattern diagnosis:
  - If all failed and many are chunking variants, forces fundamentally different strategies.
- Also injects persistent failure IDs when available and requires at least one targeted hypothesis.

Implementation tag: `src/agents/analysis_code_agent/code_agent.py:75-123`

### Hypothesis testing
- Every hypothesis is validated first on a small sample (syntax/runtime + chunk integrity checks).
- For each valid hypothesis:
  - Build temp index name `hyp_{hypothesis.id}` on BM25 server.
  - Evaluate all queries with `run_subset_eval(index_name, queries, client)`.
  - Evaluate baseline comparator on `run_subset_eval("current", queries, client)`.
  - Compute:
    - `delta_recall_100`
    - `delta_recall_10`
    - `delta_ndcg_10`
  - Record improved/regressed query IDs.
  - Always delete temp hypothesis index in `finally`.

Implementation tag: `src/agents/analysis_code_agent/code_agent.py:306-430`

### Selection rule used by orchestrator
- The orchestrator (`agent.py`) chooses best hypothesis by highest absolute `hypothesis_recall_100` among non-error results.
- Adoption criterion is strict improvement over current: `best_hyp.delta_recall_100 > 0`.
- If multiple hypotheses are `proven`, optional synthesis combines them into one final code candidate.
- Synthesis output is validated; if invalid, fallback to the best single hypothesis.

Implementation tag: `src/agents/analysis_code_agent/agent.py:507-543` and `src/agents/analysis_code_agent/code_agent.py:432-518`

### Justification
- Multi-hypothesis generation broadens search over code strategies.
- Parse retry + validation prevents malformed/unsafe code from entering eval.
- Head-to-head testing against current index converts ideas into measurable evidence.

---

## Run Journal / Logging

Implementation tag: `src/agents/analysis_code_agent/run_journal.py:48-256` and `src/agents/analysis_code_agent/agent.py:258-337,404-506`

### What is logged per iteration
- `logs/iteration_{i}_eval.json` from loop-start eval.
- `logs/iteration_{i}_analysis.log` full analysis conversation.
- `logs/iteration_{i}_analysis_summary.txt` analysis summary only.
- `logs/iteration_{i}_hypotheses.json` all hypotheses + metrics/deltas/proven/error.
- `logs/iteration_{i}_final_code.py` adopted/synthesized code written that loop.
- `logs/run_journal.json` structured history:
  - Iteration records (metrics + hit/miss query IDs).
  - Hypothesis records (deltas, targeted queries, improved/regressed IDs, adoption).

### How logs feed back into prompts
- `RunJournal.summary_for_prompt()` is passed into analysis context (`journal_summary`).
- `persistent_failure_ids()` are passed into `CodeAgent.generate_hypotheses(...)`.
- `all_past_hypotheses` list (in-memory, built from test results each loop) is passed to code generation prompt to discourage repeats.

Implementation tag: `src/agents/analysis_code_agent/agent.py:404-449`, `src/agents/analysis_code_agent/run_journal.py:129-242`

### Justification
- Keeps both human-auditable artifacts and machine-usable structured memory.
- Enables longitudinal adaptation (persistent failures, overfitting awareness, win-rate trends).

---

## Feedback Mechanism

Implementation tag: `src/agents/analysis_code_agent/agent.py:129-179,223-256,404-449` and `src/agents/analysis_code_agent/analysis_agent.py:266-377`

### How harness-style metrics are passed to analysis
- Loop computes metrics (`recall_at_10`, `recall_at_100`, `ndcg_at_10`) in `run_eval()`.
- Results are enriched with per-query data, then passed to `analysis_agent.analyze(...)`.
- Analysis initial context explicitly shows current-vs-baseline summary metrics.

### How per-query hit/rank data is enriched
- `_enrich_eval_results()` runs `run_subset_eval("current", queries, client, top_k=100)`.
- Converts per-query results into fields consumed downstream:
  - `query_id`, `query_text`, `hit`, `rank`, `relevant_doc_ids`, `retrieved_doc_ids`.

### How analysis summary flows into hypothesis generation
- `analysis_result.summary` is passed directly into `code_agent.generate_hypotheses(...)`.
- That summary becomes top section of coding prompt before current code and history sections.

### Justification
- Aggregate metrics guide optimization target.
- Per-query data localizes concrete retrieval failure modes.
- Analysis summary creates a compressed bridge from diagnosis to code generation.

---

## Evaluation Pipeline

Implementation tag: `src/agents/analysis_code_agent/eval_utils.py:36-100` and `src/evaluation/scripts/test_preprocessing_split.py:308-413`

### Exact metrics and formulas
- `Recall@10`: fraction of queries with at least one relevant doc in top 10.
- `Recall@100`: fraction of queries with at least one relevant doc in top 100.
- `nDCG@10` (subset eval path):
  - If relevant doc appears at rank `r <= 10`, score is `1 / log2(r + 1)`.
  - Else `0`.
  - Average over queries.
- `nDCG@10` (harness script path):
  - Computes DCG from ranked top-10 relevant hits.
  - Computes IDCG as ideal ranking up to `min(#relevant, 10)`.
  - Uses `DCG/IDCG`.
  - For single-relevant-doc queries, this reduces to the same single-hit expression above.

Implementation tag: `src/agents/analysis_code_agent/eval_utils.py:63-96` and `src/evaluation/scripts/test_preprocessing_split.py:375-390`

### Harness eval vs subset eval in this system
- Canonical static harness (`test_preprocessing_split.evaluate`) runs full preprocessing + local BM25Index construction + metric computation.
- Fast subset eval (`run_subset_eval`) assumes index already exists on BM25 server and computes metrics from server retrieval.
- `analysis_code_agent` intentionally overrides `run_eval()` to use BM25-server subset eval over sampled corpus for speed, even though comments label it as harness-like loop eval.

Implementation tag: `src/agents/agent_runner.py:91-129` vs `src/agents/analysis_code_agent/agent.py:129-179`

### Justification
- Static harness is canonical and reproducible for general agents.
- Server subset eval enables much faster inner-loop hypothesis testing.

---

## BM25 Retrieval Server

Implementation tag: `src/agents/analysis_code_agent/bm25_server.py:27-214` and `src/agents/analysis_code_agent/bm25_client.py:43-99`

### Architecture
- FastAPI service stores multiple named indexes in memory (`_indexes`).
- Optional persistence to disk under `.bm25_cache` (or configured directory).
- Client (`BM25Client`) provides typed wrappers with retry logic.

### Tokenization behavior
- Corpus tokenization: lowercase each chunk text (`t.lower()`), then `bm25s.tokenize(corpus)`.
- Query tokenization: lowercase query, then `bm25s.tokenize([query_text])`.

Implementation tag: `src/agents/analysis_code_agent/bm25_server.py:64-68,80-82`

### MaxP aggregation
- Retrieval runs at chunk level then aggregates to document scores by taking max chunk score per doc.
- Final ranking is doc-level sorted by score descending and `doc_id` tie-break.

Implementation tag: `src/agents/analysis_code_agent/bm25_server.py:83-109`

### Endpoints used by the agent
- Build: `POST /index/{name}/build`
- Batch retrieve: `POST /index/{name}/batch_retrieve`
- Delete: `DELETE /index/{name}`
- Health: `GET /health` for startup polling

Implementation tag: `src/agents/analysis_code_agent/bm25_server.py:112-183` and `src/agents/analysis_code_agent/bm25_client.py:43-99`

### Index lifecycle (`current` and per-hypothesis)
- Loop rebuilds persistent `current` index from current preprocess code.
- Each hypothesis gets temp index `hyp_{id}` for isolated A/B comparison.
- Temp index is deleted after test (success or failure).
- Loop eval uses index name `harness_eval`.

Implementation tag: `src/agents/analysis_code_agent/agent.py:398-402`, `src/agents/analysis_code_agent/code_agent.py:339-430`

### Justification
- Named indexes support side-by-side testing without rewriting files.
- MaxP at doc level matches retrieval objective and harness behavior.

---

## Hypothesis Testing and Selection

Implementation tag: `src/agents/analysis_code_agent/agent.py:468-553` and `src/agents/analysis_code_agent/code_agent.py:329-418`

### Testing protocol
- Evaluate each hypothesis against identical query set and identical sampled corpus.
- Comparator is always server index `current`.
- Metric deltas are computed per hypothesis result.

### Decision and adoption rules
- Candidate pool excludes errored hypotheses.
- Best hypothesis is argmax of `hypothesis_recall_100`.
- Adopt only if `delta_recall_100 > 0`.
- `proven` is separate from adoption and defined as:
  - `delta_recall_100 >= recall_improvement_threshold`.
- Threshold is from config (`recall_improvement_threshold`, default `0.00001`).

Implementation tag: `src/agents/analysis_code_agent/config.yaml:10` and `src/agents/analysis_code_agent/code_agent.py:392`

### Global best tracking interaction
- After adoption, optimistic estimate is:
  - `new_recall_estimate = loop_start_recall_100 + best_hyp.delta_recall_100`.
- If estimate beats `best_recall_100`, update `best_recall_100` and `best_code`.
- If later loops regress, global best restore at end prevents final regression.

Implementation tag: `src/agents/analysis_code_agent/agent.py:545-558`

### Justification
- Adoption rule prevents drift from non-improving changes.
- Proven threshold enables stronger filtering and synthesis gating.
- Global-best restore gives end-of-run safety.

---

## Global Best Tracking

Implementation tag: `src/agents/analysis_code_agent/agent.py:343-370,545-564`

### Initialization
- Baseline loaded from `src/agents/baseline_results.json`.
- `best_recall_100` initialized from `baseline_results["recall_at_k"]`.
- `best_code` initialized from current `preprocess.py`.

### Update rule
- Uses loop anchor `loop_start_recall_100` from start-of-loop eval.
- On adopted improvement, computes optimistic estimate using loop delta.
- Updates global best if estimate is larger.

### Final restore guard
- Before final eval, if current `preprocess.py` differs from `best_code`, it rewrites `preprocess.py` with global best snapshot.
- Final eval then runs from restored best state.

### Justification
- Ensures best-known solution survives later exploratory regressions.

---

## Validation and Self-Reflection Mechanisms (cross-cutting)

Implementation tag: `src/agents/analysis_code_agent/analysis_agent.py:91-103,112-120,200-221` and `src/agents/analysis_code_agent/code_agent.py:184-223,306-327`

- Analysis self-reflection:
  - Explicit nudge when insufficient bash investigation is performed.
  - Explicit summary request when model does not naturally conclude.
- Code self-reflection:
  - Parse retry when hypothesis format is not parseable.
  - Pre-execution validation on sample documents before expensive eval.
  - Timeout guard around preprocessing call.

These mechanisms reduce silent failure modes and keep the loop progressing with machine-checkable outputs.
