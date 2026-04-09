## CRITICAL: Corpus Structure

The corpus contains **full academic paper abstracts and metadata**. Each entry in `documents.jsonl` is a complete paper record, including the title, authors, abstract, and potentially venue or year information.

The `doc_id` is a unique identifier for each paper.

**Query characteristics**: Queries describe a specific paper using natural-language paraphrases of the abstract or title — often without using the exact words from the paper. The primary retrieval challenge is **vocabulary mismatch**: queries use informal or general language while papers use domain-specific terminology, and vice versa.

**Key preprocessing strategies to consider**:
- Prepend the paper title prominently to increase title term weight
- Expand acronyms or abbreviations if inferable from the abstract
- Extract and repeat key noun phrases or technical terms that characterize the paper's contribution
