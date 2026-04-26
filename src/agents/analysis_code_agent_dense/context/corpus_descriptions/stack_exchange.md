## CRITICAL: Corpus Structure

The corpus contains **Stack Exchange question-and-answer posts**. Each entry in `documents.jsonl` is a complete Q&A thread, including the question title, question body, accepted answer, and potentially top-voted answers.

The `doc_id` is a unique identifier for each Q&A thread.

**Query characteristics**: Queries describe a programming problem or technical question in natural language. The goal is to find a Stack Exchange thread that answers the same question, even if phrased differently. The primary retrieval challenge is **vocabulary mismatch**: queries may describe the problem abstractly while the thread uses specific API names, error messages, or code snippets.

**Key preprocessing strategies to consider**:
- Weight the question title more heavily (repeat it or place it first)
- Extract code identifiers, function names, and error strings as high-signal terms
- Separate the question from the answer text — the question text is often more query-aligned
