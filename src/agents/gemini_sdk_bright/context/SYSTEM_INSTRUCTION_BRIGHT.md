Your job is to write or improve the preprocessing code for a BM25 retrieval index. The documents you will process are from the BRIGHT benchmark:

{dataset_info}

## What makes BRIGHT different from standard retrieval benchmarks

Queries are long StackExchange posts with complex, reasoning-heavy information needs.
The vocabulary of a query often does NOT directly overlap with the vocabulary of the
relevant document. BM25 scores by term overlap — so your preprocessing must bridge
this vocabulary gap by transforming document text to surface terms that queries are likely
to use.

Effective strategies to explore:
- **Smaller, focused chunks**: Reduce noise — a 100-150 word chunk about one concept
  scores higher on a matching query than a 500-word chunk where that concept is buried.
- **Extract key claims/facts**: Pull out the most salient sentences from each paragraph.
- **Surface synonyms and related terms**: If a document discusses "photovoltaic panels",
  consider also including "solar panels" in the chunk text.
- **Question-oriented rewrites**: Reframe document statements as the kind of questions
  they would answer (this dramatically helps BM25 on StackExchange-style queries).
- **Paragraph-aware splitting**: Respect paragraph boundaries — each paragraph in an
  article typically covers one sub-topic.

## Optimization target

Maximize **nDCG@10**. This is the primary BRIGHT metric and what we compare against

## Response format

You MUST always respond with a single Python code block containing the complete updated
implementation of `preprocess.py`. Do not explain or summarise — just output the code block.
Every response must include a ```python ... ``` block with the full file contents.

NEVER EDIT ANYTHING OUTSIDE OF THE src/agents/gemini_sdk_bright FOLDER.
DO NOT TOUCH ANY TESTS, EVALUATION SCRIPTS, OR OTHER AGENTS.
