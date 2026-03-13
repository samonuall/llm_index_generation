# analysis_code_agent

A two-stage LLM agent that iteratively improves BM25 document preprocessing by:

1. **Analysis stage** — an LLM investigates retrieval failures using bash commands (looking at documents, querying the BM25 server, reading eval results).
2. **Hypothesis stage** — a second LLM generates N candidate `preprocess.py` implementations, each is tested empirically against the live BM25 server, and the best-performing one is adopted.

---

## Files

```
analysis_code_agent/
├── agent.py            # Main orchestrator — overrides AgentRunner.run()
├── analysis_agent.py   # Multi-turn bash-loop analysis agent
├── code_agent.py       # Hypothesis generation, testing, and selection
├── eval_utils.py       # Subset eval utilities (BM25 server-based)
├── bm25_server.py      # FastAPI server for in-memory BM25 index management
├── bm25_client.py      # HTTP client for the BM25 server
├── preprocess.py       # Current best preprocessing code (written by the agent)
├── config.yaml         # All tunable settings
├── context/
│   ├── ANALYSIS_SYSTEM.md  # System prompt for the analysis agent
│   └── CODE_SYSTEM.md      # System prompt for the code agent
└── logs/
    ├── iteration_N_analysis.log        # Full analysis conversation
    ├── iteration_N_analysis_summary.txt # Final summary from analysis agent
    ├── iteration_N_hypotheses.json     # Hypothesis test results
    ├── iteration_N_eval.json           # Harness eval results at loop start
    └── iteration_N_final_code.py       # Adopted code for this iteration
```

---

## Architecture

### Loop Overview

```
for each loop:
    1. run_eval()           — authoritative harness recall@100 at loop start
    2. build "current" index on BM25 server (from current preprocess.py)
    3. enrich eval results with per-query hit/rank data (BM25 server)
    4. AnalysisAgent.analyze()   — multi-turn bash investigation → summary
    5. CodeAgent.generate_hypotheses()  — N candidate preprocess.py implementations
    6. CodeAgent.test_hypothesis()  × N  — empirical recall@100 on BM25 server
    7. adopt best hypothesis if delta_recall_100 > 0

after all loops:
    restore globally best code if later loops degraded things
    run final harness eval and print improvement summary
```

### Analysis Agent (`analysis_agent.py`)

A multi-turn agentic loop where the LLM can issue `<bash>...</bash>` commands at each turn to investigate failures. The loop runs until the LLM stops producing bash blocks (its final message is treated as the summary) or until `analysis_max_turns` is reached.

**What it receives as context:**
- Current `preprocess.py` code
- Current recall@100 and nDCG@10 vs. baseline
- Up to 5 hard negatives (queries that miss entirely, with wrong top docs shown)
- Regressions vs. baseline (queries that were previously hits but are now misses)
- Worst-ranked hits (queries found but with poor rank)
- Paths to `documents.jsonl` and `queries.jsonl`
- BM25 server URL and curl/Python snippets for live retrieval

**Bash loop behavior:**
- The LLM runs arbitrary shell commands (capped at `bash_timeout_seconds`)
- stdout+stderr are returned, truncated to 4000 chars
- The agent can inspect documents, diff chunks, run BM25 queries, read code, etc.

**Output:** A structured text summary of failure taxonomy, concrete examples, and preprocessing recommendations — passed directly to the code agent.

### Code Agent (`code_agent.py`)

Generates N hypothesis `preprocess.py` implementations in a single LLM call, then tests each one empirically.

**Prompt includes:**
- Analysis agent's summary
- Current `preprocess.py`
- Dataset info (`CONTEXT.md`)
- Past tested hypotheses with their delta scores (so the agent doesn't repeat failed strategies)
- Hard notes: metadata is empty, naive paragraph splitting hurts BM25, chunks must be ≥200 words

**Output format (markdown blocks, not JSON):**
```
### H1: <description>
Rationale: <rationale>
Query IDs: <comma-separated>
Falsifying: <condition>
```python
<complete preprocess.py code>
```
```

**Testing:** Each hypothesis is tested by:
1. Loading the code via `exec()` into a fresh module (no disk write)
2. Running `preprocessor.preprocess(all_documents)` → chunks
3. Building a temporary BM25 index on the server (`hyp_H1`, etc.)
4. Running subset eval against all 135 queries
5. Comparing recall@100 and recall@10 to the "current" baseline index
6. Deleting the temp index

**Selection:** The hypothesis with the highest absolute `hypothesis_recall_100` is adopted if `delta_recall_100 > 0`. No synthesis step — the hypothesis code is used directly.

### BM25 Server (`bm25_server.py` / `bm25_client.py`)

A local FastAPI server that manages named in-memory BM25 indexes. The agent starts it as a subprocess and communicates via HTTP.

**Key endpoints:**
- `POST /index/{name}/build` — build an index from a list of chunks
- `POST /index/{name}/retrieve` — query an index
- `DELETE /index/{name}` — delete a temporary hypothesis index
- `GET /health` — liveness check

This allows hypothesis testing without touching disk-based indexes or restarting processes.

### Global Best Tracking

`best_recall_100` and `best_code` are tracked across all loops. If a later loop's adopted code degrades results, the globally best code is restored before the final harness eval.

---

## Configuration (`config.yaml`)

```yaml
# LLM settings
analysis_model: "openai/gpt-5-mini"   # Model for analysis agent
analysis_temperature: 1.0
code_model: "openai/gpt-5-mini"       # Model for code agent
code_temperature: 1.0
api_base: "https://thekeymaker.umass.edu/"  # LiteLLM proxy base URL

# BM25 server
server_port: 8765
server_persist_dir: ".bm25_cache"

# Agent behavior
max_hypotheses: 4                     # N hypotheses per loop
recall_improvement_threshold: 0.00001 # Minimum delta_recall_100 for "proven" flag
analysis_max_turns: 8                 # Max bash turns for analysis agent
bash_timeout_seconds: 30              # Per-command timeout
```

**`recall_improvement_threshold`**: Used to set the `proven` flag on `HypothesisResult`. The adoption logic is independent — the best hypothesis is always adopted if `delta_recall_100 > 0` regardless of this threshold. Setting it near zero (0.00001) makes `proven=true` whenever there's any positive improvement at all.

**`api_base`**: The LiteLLM proxy at thekeymaker.umass.edu. Model IDs available: `gpt4o`, `gpt-5-mini`, etc. (these differ from OpenAI's public naming).

---

## Running

```bash
# Single loop, default split
uv run python main.py --agent analysis_code_agent --loops 1

# Multiple loops
uv run python main.py --agent analysis_code_agent --loops 3

# Omit raw query text from prompts (if safety filters block dataset content)
uv run python main.py --agent analysis_code_agent --loops 3 --no-query-text
```

Logs are written to `src/agents/analysis_code_agent/logs/` each loop.

---

## Example Output

### Loop 1 — Analysis Summary (from `logs/iteration_0_analysis_summary.txt`)

The analysis agent investigated why BM25 was missing gold documents for several "tip-of-the-tongue" queries. Key findings (condensed):

> **Most failures are not due to BM25 hyperparameters; rather they stem from document preprocessing decisions that reduce useful term co-occurrence.**
>
> **1. CHUNKING TOO AGGRESSIVE (high priority)**
> Long texts are split into sliding windows of 300 words with a 150-word stride. For narrative, identificatory terms (character descriptions, key scenes) can be distributed across adjacent chunks such that no single chunk has enough co-occurring terms to match the query well.
>
> **2. TERM FREQUENCY DILUTION (medium-high priority)**
> The preprocessor repeats the lead in each chunk to boost lead TF, but creates many chunks. For long documents, rare but identifying terms scattered through the document are diluted across many chunks.
>
> **4. NO FIELD BOOSTING / METADATA NOT INDEXED (medium priority)**
> The preprocessor operates only on `doc.text`. Titles/aliases that might identify a document are not present in metadata (metadata is empty in this dataset).
>
> **Concrete recommendations:**
> - Increase `window_words` from 300 → 600–800
> - Reduce `lead_repeat_per_chunk` to 0 or 1
> - Add multiple coarse chunks (head/mid/tail) rather than a single 1200-word truncation

### Loop 1 — Hypothesis Results (from `logs/iteration_0_hypotheses.json`)

Four hypotheses were generated and tested. Baseline (passthrough): recall@100 = 0.5926.

| Hypothesis | Description | Δ recall@100 | Δ recall@10 | Δ nDCG@10 | Proven |
|-----------|-------------|-------------|------------|----------|--------|
| **H1** | Overlapping sliding windows (300w, stride 150) + lead repeat × 4 + coarse 1200w chunk | **+0.0222** | +0.0444 | +0.0252 | ✓ |
| **H2** | Paragraph-merge with sentence-overlap (~200–500w chunks, 2-sentence overlap) | **+0.0222** | +0.0370 | +0.0295 | ✓ |
| H3 | Sentence-window grouping (250–350w chunks, 1-sentence overlap) | −0.0148 | +0.0222 | +0.0208 | ✗ |
| H4 | Larger sentence windows (350–500w) + coarse 1500w chunk | +0.0074 | +0.0370 | +0.0260 | ✓ |

**Adopted:** H1 (tied with H2 on recall@100; H1 selected as it had the highest absolute score). H1 improved 3 queries without regressing any.

**Note on recall@100 vs @10:** H1 improved recall@100 by +2.2% and recall@10 by +4.4%. H3, despite improving @10 by +2.2%, hurt @100 by −1.5% — the relevant docs were being found in the top 100 but knocked out entirely by H3's chunking. This is why recall@100 is used as the primary adoption signal.

### Loop 2 — Hypothesis Results (from `logs/iteration_1_hypotheses.json`)

Baseline for loop 2: recall@100 = 0.6148 (the H1 code from loop 1). The analysis agent noted that the current `overlap_window_lead_boost` strategy still has heavy lead repetition and a single coarse chunk that may miss later-document content.

| Hypothesis | Description | Δ recall@100 | Δ recall@10 | Proven |
|-----------|-------------|-------------|------------|--------|
| H1 | Larger windows (800w, stride 400) + single lead prefix + head/mid/tail coarse chunks | −0.0148 | −0.0074 | ✗ |
| H2 | Paragraph grouping (~700w) + 1-paragraph overlap + single lead prefix | −0.0222 | +0.0148 | ✗ |
| H3 | (see logs) | ... | ... | ✗ |
| H4 | (see logs) | ... | ... | ✗ |

No hypothesis improved over the loop 2 baseline — `preprocess.py` was left unchanged.

### Final Result

```
Improvement: recall@100  0.5778 → 0.6148  (+0.0370)
```

The agent improved recall@100 by **+3.7%** (from 57.8% to 61.5%) in one productive loop, recovering 3 additional relevant documents across 135 queries.

---

## Key Design Decisions

**Why recall@100 (not @10)?**
BM25 with many chunks per document can push relevant docs to ranks 11–50 even when they are found. A hypothesis that improves coarse coverage (+recall@100) but scatters chunks (+rank noise) would show zero delta@10 despite a real improvement. Recall@100 is more sensitive to whether the relevant document is retrieved at all.

**Why best-of-N adoption (not synthesis)?**
An LLM synthesis step adds another LLM call, introduces hallucination risk, and is hard to evaluate. Direct adoption of the empirically best hypothesis is simpler and fully grounded — we know exactly what recall@100 the adopted code will achieve.

**Why a separate BM25 server?**
The harness (`test_preprocessing.py`) writes indexes to disk and reloads from disk each run — too slow for testing 4 hypotheses × 10k docs per loop. The FastAPI server keeps all indexes in memory and allows multiple named indexes to coexist simultaneously, making hypothesis testing fast (~5–15s per hypothesis).

**Why `exec()` for hypothesis loading?**
Hypotheses are generated as code strings. Writing them to disk for each test would require either overwriting `preprocess.py` (losing the current code) or managing many temp files. `exec()` into a fresh module namespace avoids both problems and keeps hypothesis testing self-contained.
