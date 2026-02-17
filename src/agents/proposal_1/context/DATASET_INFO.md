# Dataset Information

## Source
- **Dataset**: [CRUMB](https://huggingface.co/datasets/jfkback/crumb), `tip_of_the_tongue` split
- **Corpus**: Full Wikipedia articles – each document is a single `Document` object with fields:
  - `doc_id` (str): unique identifier matching the CRUMB corpus
  - `text` (str): full Wikipedia article text (can be very long)
  - `metadata` (dict): extra fields from the source dataset

## Eval Queries
- **5 eval queries**, each a natural-language "tip of the tongue" description of a Wikipedia entity
- Example query: *"I'm thinking of a 19th-century French novelist known for realist fiction..."*
- Each query has one or more `relevant_doc_ids` (ground-truth Wikipedia articles)

## Task
Return `Chunk` objects from `preprocess()` such that BM25 retrieval surfaces the relevant document in top-k results for as many queries as possible.

## Retriever (static – do NOT modify)
- **Algorithm**: BM25 via `bm25s` library
- **Tokeniser**: English stemmer (PyStemmer / Snowball)
- Agents control **only** the `preprocess()` method – the retriever is fixed

## Key Constraints
- Every `Chunk` must have a `doc_id` matching its source `Document`
- `chunk_id` must be globally unique across all returned chunks
- At least one `Chunk` must be returned per `Document`
- Multiple chunks per document are allowed and encouraged
