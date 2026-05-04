# Cross-split analysis: why stack_exchange worked and the others didn't

**Run**: `run_splits_throwaway.py`, condition `agent_history`, model `openai/claude-sonnet-4-6`, 5 loops
**Splits compared**: stack_exchange (great), code_retrieval (plateau), paper_retrieval (no improvement), clinical_trial (one-shot then stuck)

---

## 1. The headline numbers

| Split | Baseline R@100 | Final R@100 | Δ R@100 | Baseline nDCG@10 | Final nDCG@10 | Δ nDCG | Loops adopted |
|---|---|---|---|---|---|---|---|
| **stack_exchange** | 0.6224 | **0.7602** | **+0.138** | 0.1772 | 0.2251 | +0.048 | iter 0,1,2,3 (4/5) |
| code_retrieval | 0.0467 | 0.0547 | +0.008 | 0.0434 | 0.0585 | +0.015 | iter 0 only (1/5) |
| paper_retrieval | 0.3234 | 0.3208 | **−0.001** | 0.6607 | 0.6553 | −0.005 | iter 2,3,4 (3/5) but net negative |
| clinical_trial | 0.1949 | 0.1974 | +0.003 | 0.4721 | 0.4707 | −0.001 | iter 0 only (1/3 completed) |

Stack_exchange is the only split where every accepted iteration *also* moved the held-out eval up. Everywhere else either nothing was adopted, or what was adopted didn't generalize.

Stack_exchange validation trajectory (from the iter-3 run journal): 0.5250 → 0.6750 → 0.7250 → 0.8250 — every iteration adds val-recall, four hypotheses adopted (H3, H3, H3 — all three of stack_exchange's "H3" choices were combination/structural-summary chunks).

---

## 2. What went well in stack_exchange

### 2.1 Hypotheses build on each other compositionally
Every adopted hypothesis is a *new chunk type*, not a tweak to an existing chunk:
- iter 0 → "topic + first paragraph" augmented chunk (chunks/doc 1 → 2)
- iter 1 → "folder siblings context" expansion chunk (chunks/doc → 3)
- iter 2 → "structural summary" chunk (folder topic + filename + first sentence per paragraph) (chunks/doc → ~5)
- iter 3 → no improvement, but the cumulative state is at recall 0.76

Each chunk targets a *different* failure mode (topic missing → topic chunk, intra-folder competition → siblings expansion, terms buried in long doc → structural summary). The agent did not redo the same idea — it kept adding orthogonal signals.

### 2.2 The corpus matched the prompt's strategy hints
`stack_exchange.md` explicitly recommends "weight the question title", "extract code identifiers", "separate question from answer text". The CRUMB stack_exchange corpus does have folder-name disambiguation (e.g. `sex_duration/Handicap_principle.txt`) and short titles, so these strategies actually worked. The prompt was right.

### 2.3 Small corpus, headroom to improve
5,035 documents, baseline recall 0.62 → there's room for a 200-line preprocessing change to materially shift the index. With BM25 already finding the right *folder* most of the time, surfacing the right *file* in that folder is a tractable task.

### 2.4 The validation set was representative of failures
Stack_exchange validation has 20 queries and the analysis sees 3 hard negatives + 8 poor-rank successes per iter. The persistent failures (`economics:80`, `robotics:31`, `sustainable_living:18`) genuinely span different failure types (sparse doc, vocab mismatch, semantic gap) so patterns abstracted from them generalized.

---

## 3. What went wrong in the other splits

### 3.1 code_retrieval — the agent fixated on `ALGORITHM_PATTERNS`

After iter 0 adopted "function-name token injection" (H4), every subsequent analysis converged on the same recommendation:

| iter | top-priority recommendation |
|---|---|
| 1 | "Expand ALGORITHM_PATTERNS for greedy/lexicographic" |
| 2 | "Extend ALGORITHM_PATTERNS to cover problem-story vocabulary" |
| 3 | "Extend ALGORITHM_PATTERNS to cover problem-story vocabulary" (again) |
| 4 | "Expand ALGORITHM_PATTERNS for string manipulation, greedy, character-level" |

Every iter asks for the same fix, just with different example regexes. None were adopted. This is **prompt fixation** — once the codebase had `ALGORITHM_PATTERNS`, the analysis agent kept looking at it and proposing extensions to it, instead of stepping back and asking whether the whole regex-pattern approach is the wrong frame for this corpus.

The hypotheses across iter 1-4 are also less diverse than stack_exchange's — most are variants of "detect pattern X in code, emit keyword Y":
- iter 1: H1=identifier splitting (already done), H2=string literals, H3=more patterns, H4=function names (already done)
- iter 2: H1=palindrome regex, H2=2-char abbrev expansion, H3=input-pattern domain inference
- iter 3: H1=string-literal extraction, H2=structural NL synthesis, H3=more algorithm vocab, H4=output-format tokens
- iter 4: H1=character-to-word mapping, H2=stoplist reduction, H3=more patterns

The hypotheses **converge** rather than diverge — each iter is increasingly similar to the last. Compare with stack_exchange iter 0 → iter 2 where the four hypotheses span topic, paragraph, augmentation, term-cloud → strip-boilerplate, distinctive-words, sibling-context, last-sentences.

The deeper issue: **competitive programming code with no NL is fundamentally outside what BM25 + preprocessing can solve.** Baseline 4.7% recall says BM25 has almost no signal to amplify. The agent correctly diagnoses this in every iter ("the queries describe problems in NL, the docs are pure code") but cannot escape the prompt's framing of "preprocessing fixes the problem". A more honest output would be "this corpus needs a different retriever; recommend abandoning further iteration."

### 3.2 paper_retrieval — the agent chases noise

Paper_retrieval has the inverse problem: the validation queries are mostly *successes* at rank 1-3. The iter 0 analysis says it explicitly:

> "ALL shown queries are already successes (just at ranks 1-3, not rank 0). The system is NOT failing catastrophically — the queries listed are the worst-performing successes."
> "The validation queries shown are cherry-picked successes."
> "Medium confidence — the validation set only shows successful queries (rank 1-3), making it hard to diagnose the 64% of queries where the gold document isn't retrieved in top 100."

Despite recognizing this, iter 0-1 adopted nothing (no hypothesis beat baseline) and iter 2-4 each adopted small tweaks (CamelCase splitting, hyphenated-term expansion, section-header chunks) that **didn't move the needle on val** (R@100 stayed at 0.32) and **regressed on eval** (0.3234 → 0.3208, −0.001).

What's wrong:
- The validation set is showing the agent the *wrong distribution*: rank-1-to-3 successes when the real problem is the 64% that aren't in top-100. The agent is optimising for a tail that already works.
- The hypotheses iter-over-iter become increasingly nitpicky: suffix stripping, parenthetical acronym expansion, mathematical-formula token normalization. These are micro-optimizations that can't generalize to the unseen 200K docs in any reliable way.
- The corpus_description for paper_retrieval (12 lines) gives the same three suggestions as stack_exchange ("title prepending, expand acronyms, repeat technical terms"). After iter 0 already used title prepending, the agent has nothing fresh to propose.

### 3.3 clinical_trial — adopted once then plateau

Iter 0 adopted a "Title + Conditions + Interventions" mini-chunk. It bumped val recall +0.0025 but **dropped val nDCG by −0.0519**. Iter 1's analysis caught this:

> "Adding mini-chunks creates duplicate representations… the mini-chunk ranking is taking up slots that could go to other relevant docs."
> "The most impactful fix would be: Remove the Mini-Chunk Strategy."

But the run_journal already showed H1 was adopted, so the next iters proposed alternatives — none of which were adopted (synthesis even gets rejected: "Synthesis did not beat best hypothesis — falling back to H1"). The split was killed at loop 4 of 5 by the wall time, but it had already plateaued at 0.1974 by iter 1.

Hypotheses are very narrow: every single one is "extract section X (title / conditions / interventions / eligibility / brief summary / outcomes) and put it in a chunk." The agent does not propose anything structurally different (e.g. abbreviation/expansion mappings, sentence-level dedup of the heavy boilerplate, condition-name normalization across the corpus).

### 3.4 The common failure shape

All three failing splits show the same pattern:

1. **One genuinely useful idea early** (iter 0 or 1) that gets adopted.
2. **Iterations 2-4 keep proposing variations on that same idea**, occasionally with a new angle that doesn't beat baseline.
3. **The analysis agent's per-iter input doesn't change much** — the same persistent failures, the same poor-rank successes — so it produces the same recommendations.

Stack_exchange escaped this because (a) the early ideas actually moved the val score by 10-15 points each, which kicked the analysis input into a new state with new failures to investigate, and (b) the corpus structure (folders, short docs, NL) had multiple orthogonal signals to extract.

---

## 4. Specific prompting and bug observations

### 4.1 The system prompt constrains the search space

`ANALYSIS_SYSTEM.md` and the per-split corpus_descriptions push hard toward "vocabulary mismatch" and "additive chunking". Every analysis report frames the failure as "vocabulary mismatch" — even paper_retrieval which is mostly a ranking problem. This is the prompt talking, not the data.

The corpus_descriptions are **3-line strategy hints disguised as objective descriptions**:
- stack_exchange.md ends with "weight the question title… extract code identifiers… separate question from answer text"
- code_retrieval.md ends with "split camelCase… include the function signature… strip boilerplate"

These get baked into hypotheses immediately. Stack_exchange iter 0 H1 = folder name; iter 2 H1 = "Add a Chunk Consisting of the Stack Exchange Question Title Repeated Multiple Times" — the Question Title strategy was *literally in the prompt* (line 2 of stack_exchange.md). Code_retrieval iter 0 H1 = "Identifier Splitting via camelCase and snake_case" — also literally in the prompt.

This isn't necessarily bad — the prompt suggestions are genuinely good — but it means the agent is **only really exploring the prompt's strategies**. When those don't work (code_retrieval, paper_retrieval), there's no second-best alternative for it to fall back on.

### 4.2 The "persistent failures" header encourages overfitting

The run journal sent into iter 2+ has a section like:

```
### Persistent failures (3 queries failing every iteration)
  economics:80, robotics:31, sustainable_living:18
  → These queries are the hardest to fix. Prioritise them.
```

This explicit "prioritise them" instruction makes the agent investigate the same 3 queries every iteration. In stack_exchange iter 3, all 8 queries the agent inspected were the same persistent failures from iter 1 + the same poor-rank successes. With only 3-5 queries driving every recommendation, hypotheses become narrative-driven around those specific queries:

- "robotics:31 has a sparse README" → recommend "boost short docs"
- "economics:80 has navigation boilerplate" → recommend "strip boilerplate"
- "sustainable_living:18 is a true semantic gap" → "no fix, recommend abandoning"

All three are corpus-specific stories, not generalizable patterns — and the analysis agent itself flags this in iter 4 ("These persistent failures are fundamentally difficult for BM25 without semantic understanding").

### 4.3 Successes are surfaced but not actually used

The "## Successes (hit but poor rank)" section gives 8 queries, but the agent's tool calls almost exclusively investigate the *failures* (hard negatives + persistent failures). In stack_exchange iter 3 the agent made 4 `bm25_retrieve` calls and 3 `read_file` calls — all on hard-negative gold docs, none on the rank-19/22/25 successes. The analysis treats successes as "things that already work, we should just push them up" rather than "what is making these successful, and is the new chunk going to break them?"

This contributes to nDCG regressions: the new chunks help some recall@100 cases but inadvertently push existing rank-1 hits down to rank-3.

### 4.4 The agent never investigates *why successes succeed*

The prompt only asks "what's missing in the gold doc that's in the wrong doc?" — there's no instruction to also examine "what's working in the docs that *are* succeeding?" If the agent compared the 3 persistent failures against the 8 poor-rank successes and the 7 unseen full-rank-1 successes, it would more clearly see which chunk types *currently get rewarded* and avoid breaking them.

### 4.5 Hypotheses within an iteration are not diverse enough

Looking at each iter's 4 hypotheses, in many cases 3 of the 4 are variants of the same idea:

- stack_exchange iter 0: H1=folder name, H2=first paragraph, H3=combined H1+H2, H4=noun phrases. H3 is just "do H1 AND H2 together" — not really a third hypothesis. (The agent picked H3, which works, but only because H1 and H2 are individually fine.)
- clinical_trial iter 0: H1=title+conditions+interventions, H2=conditions repetition, H3=eligibility key terms. All "extract a section, repeat it." H4 came in iter 1 with "Title Prepended to Full Document" which is yet another variant.
- code_retrieval iter 4: H1=character-to-word mapping (e.g. `'['` → "bracket"), but the synthesis iter 4 also tried it. There's no fundamentally different approach proposed (e.g. "skip BM25 entirely for the empty-doc cluster, fall back to filename matching").

The code_agent generates 4 hypotheses in parallel, but they're seeded by a single ideas-generation call. So if the ideas generator is in a narrow mode, all 4 hypotheses share that bias.

### 4.6 No "give up gracefully" mechanism

When the analysis agent realises BM25-on-this-corpus is fundamentally limited (code_retrieval iter 4: "this requires semantic/embedding-based retrieval which is outside scope"), it's still required to produce 4 hypotheses. So it produces 4 hypotheses it knows won't work, the code_agent generates 4 implementations, none beat baseline, and we burn ~25 min of wall time per iteration. There's no termination signal.

### 4.7 The analysis agent does not look at enough queries

In every iteration, the agent makes 9-16 tool calls. Of those, ~5-8 are `bm25_retrieve` and ~3-5 are `read_file`. It investigates 4-5 queries per iteration. With a validation set of 11-21 queries per split, a third of the population is examined per iter, and **the same third every iter** (since persistent failures are pinned).

The system prompt says "pick 5-8 diverse failure cases" but in practice the agent picks ~4-5, often skewed toward the persistent failures because the journal banner tells it to.

### 4.8 Mini-chunk dilution is a real bug, not a prompt issue

Clinical_trial iter 0 → iter 1 documented this: adding short mini-chunks alongside the full document inflates BM25's score for those chunks (length normalization), pushing their parent doc up in `recall@100` but pushing other relevant docs *down* in `nDCG@10`. The current evaluation shows R@100 +0.0025 / nDCG@10 −0.0519 — a clear net loss that the system *adopted anyway* because the proven criterion is "any of recall_at_100 / recall_at_10 / ndcg_at_10 improved by ≥ threshold." When the threshold is symmetric across metrics but the metrics move in opposite directions, the wrong hypothesis gets adopted.

### 4.9 No "you've tried X already" signal

The hypothesis history in the run journal shows old hypotheses by name+description, but the agent doesn't seem to check whether its proposed new hypothesis is similar to a failed old one. Code_retrieval iter 1 proposes "Extract and Repeat the Document's Most Distinctive Long Words" → fails (regressed). Iter 2 proposes "Cross-Reference Chunk That Combines Folder Topic + Filename + First Sentence" → similar shape, succeeds. There's no explicit "this proposal overlaps with a failed proposal" check.

---

## 5. Why stack_exchange specifically worked, in one paragraph

The CRUMB stack_exchange corpus is **structured the way the prompt assumes the world is**: short docs (~1 page each), folder-name disambiguation built into the doc_id, an explicit question-title-then-body-then-answer layout, NL throughout. The baseline at 62% recall@100 means BM25 already finds the right neighborhood; the agent just needs to reranker-style boost the right doc within it. Each of the four chunk types adopted (folder topic, sibling context, structural summary, title repetition) attacks an *orthogonal* signal, and because the val→eval transfer is good (the validation queries genuinely sample the full distribution), each adopted hypothesis carries over to held-out gains. None of those preconditions hold for the other three splits.

---

## 6. Possible improvements (no code changes yet — this section is for discussion)

### 6.1 Prompt-level changes

**A. Stop priming with strategy hints in the corpus descriptions.** The 3-line "Key preprocessing strategies to consider" section in each corpus description is doing two harmful things:
- Limiting the search space to those 3 strategies
- Causing the analysis agent to recommend the strategy *back* to itself in its first hypothesis ("title repetition" is in the prompt; the agent then writes a hypothesis called "title repetition")

Replace the strategy hints with a *neutral structural description*: "documents are typically 200-500 words, contain a leading title followed by free text, may include code blocks." Let the agent derive strategies from the data.

**B. Investigate successes as well as failures.** Add an explicit step: "for at least 2 of the queries currently succeeding at rank 1, examine what about the gold doc makes it succeed. Does your proposed change preserve that signal?" This would address both the diversity problem and the nDCG regression problem.

**C. Force genuinely diverse hypotheses.** The ideas-generation prompt should explicitly require: "the 4 hypotheses must use 4 different mechanisms — e.g. one chunking change, one filtering change, one expansion change, one reordering change. Do not propose 4 variants of the same idea." Currently 3 of 4 hypotheses tend to be variants.

**D. Allow the agent to recommend "no further iteration."** If the analysis agent thinks the corpus is fundamentally outside what BM25-preprocessing can fix, it should output a `<give-up>` signal and the loop should stop. Right now it has to produce 4 hypotheses no matter what — wasted budget.

**E. Drop the "prioritise persistent failures" line from the journal.** Replace with "investigate at least 2 *new* queries you haven't seen in prior iterations." This breaks the same-3-queries fixation.

### 6.2 Process changes

**F. Promotion criterion should require *both* recall and nDCG to not regress.** Currently a hypothesis can be adopted if any one of {recall_at_100, recall_at_10, ndcg_at_10} improves by the threshold. Clinical_trial iter 0 adopted a hypothesis that gained 0.0025 R@100 at the cost of 0.0519 nDCG@10. A simple guard `delta_ndcg_at_10 >= -0.005` would have rejected it.

**G. Fold-out validation.** Right now the validation set is fixed at 15-21 queries. For paper_retrieval, those 15 are mostly easy queries where the gold doc is already at rank 1-3. The held-out eval set has the *real* failure distribution. Either resample validation each iter, or use a stratified split that preserves the eval failure distribution in val.

**H. Increase the number of failures the analysis agent inspects.** Currently 4-5 per iter. If the budget allowed for 8-12, the patterns would be less narrative-driven. Cost: ~50% more tool-call latency per iter (~5 min added per iter, ~25 min total over 5 iters).

**I. Detect "same idea proposed before" and reject.** Before generating code for a hypothesis, embed-similarity-check it against the last 3 iterations' hypotheses. If cosine > 0.85, reject and ask for a different hypothesis. This breaks the code_retrieval ALGORITHM_PATTERNS fixation.

### 6.3 Corpus-aware changes

**J. Pre-flight check: "is this corpus solvable by BM25-preprocessing?"** A trivial heuristic: if baseline R@100 < 0.10, recommend skipping the agent entirely or running with fewer loops. Code_retrieval at 4.7% baseline was never going to work — running 5 loops cost 84 min for +0.008 recall.

**K. Per-split loop budgets.** Stack_exchange got plenty of mileage from 5 loops; clinical_trial plateaued at iter 1. A simple early-stop ("if 2 consecutive iters adopt nothing, stop") would save wall time and let other splits fit in the 6-hour budget.

### 6.4 Validation/diagnosis questions to answer first

Before making prompt changes, answer these with targeted experiments:

- **Q1.** If we re-run code_retrieval *without* the corpus_description hints, do we get more diverse hypotheses?
- **Q2.** If we add the nDCG-no-regress guard, does clinical_trial iter 1 reject H1 and adopt something else? Does that something-else generalize?
- **Q3.** If we double the analysis agent's tool-call budget, do paper_retrieval's hypotheses get less nitpicky?
- **Q4.** Is the validation split actually representative of the eval split for paper_retrieval and code_retrieval? If we look at the rank distribution of val gold docs vs eval gold docs, are they similar?

These four would tell us which lever (prompt content, promotion criteria, exploration budget, validation choice) is doing the most damage.

---

## 7. Single biggest takeaway

Stack_exchange worked because the baseline left enough headroom and the corpus structure matched the prompt's hints. The other three split failures share **one root cause**: the agent generates each iter's hypotheses from a near-identical input (same persistent failures, same prompt suggestions, same successes-list it doesn't actually read), so the hypotheses converge to variants of one idea. Whatever fixes the multi-split outcome needs to inject **new information** into the analysis agent's context each iteration — whether that's a different sample of failures, an explicit "you've already tried this" signal, or a different decomposition of the failure space.
