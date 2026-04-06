## CRITICAL: Corpus Structure

The corpus is **pre-chunked Wikipedia articles about movies and TV shows**. Each entry in `documents.jsonl` is a single section (not a full article). The `doc_id` format is `"{wikipedia_page_id}:{section_index}"`:

- `"24073089:0"` → title/intro section of Wikipedia article 24073089
- `"24073089:1"` → second section 
- `"24073089:2"` → third section, etc.

All sections sharing the same prefix (e.g. `"24073089"`) belong to the **same Wikipedia article**.
