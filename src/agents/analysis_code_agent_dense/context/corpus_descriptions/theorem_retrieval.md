## CRITICAL: Corpus Structure

The corpus contains **mathematical theorems, lemmas, and proofs** from formal mathematics. Each entry in `documents.jsonl` is a complete theorem record, including the theorem statement, proof, and potentially surrounding context (e.g., definitions, corollaries).

The `doc_id` is a unique identifier for each theorem.

**Query characteristics**: Queries describe a mathematical result in natural language or informal notation, seeking the formal theorem that matches. The primary retrieval challenge is **notation and terminology mismatch**: queries may use informal descriptions while documents use formal symbolic notation, LaTeX, or discipline-specific terms.

**Key preprocessing strategies to consider**:
- Strip or normalize LaTeX/symbolic notation to make terms BM25-searchable (e.g., convert `\mathbb{R}` to "real numbers")
- Extract the theorem name or label if present and repeat it prominently
- Include surrounding definitional context that explains the notation used in the theorem
