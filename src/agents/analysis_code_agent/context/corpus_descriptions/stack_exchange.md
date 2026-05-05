## CRITICAL: Corpus Structure

The corpus contains **Stack Exchange question-and-answer posts**. Each entry in `documents.jsonl` is a complete Q&A thread, including the question title, question body, accepted answer, and potentially top-voted answers.

The `doc_id` is a unique identifier for each Q&A thread.

**Goal**: Given a natural-language query describing a programming or technical question, find the Stack Exchange thread(s) that answer the same question.

Key preprocessing strategies to consider:

Weight the question title more heavily (repeat it or place it first)
Extract code identifiers, function names, and error strings as high-signal terms
Separate the question from the answer text — the question text is often more query-aligned
