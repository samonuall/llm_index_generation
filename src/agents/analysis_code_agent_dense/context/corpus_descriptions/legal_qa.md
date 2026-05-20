## CRITICAL: Corpus Structure

The corpus contains **legal documents**, such as case law opinions, statutes, or legal Q&A passages. Each entry in `documents.jsonl` is a full legal document or passage, potentially spanning several paragraphs with formal legal language.

The `doc_id` is a unique identifier for each document.

**Query characteristics**: Queries ask legal questions in plain language. The goal is to find the document that answers or most directly addresses the question. The primary retrieval challenge is the **gap between plain-language queries and formal legal terminology** — queries use everyday words while documents use statutory terms, citations, and legal jargon.

**Key preprocessing strategies to consider**:
- Extract and surface key legal concepts, statutes referenced, or jurisdiction markers
- Split long documents into meaningful sub-sections (e.g., by paragraph or legal clause)
- Normalize legal abbreviations (e.g., expand "§" to "section", "v." to "versus")
