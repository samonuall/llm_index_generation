## CRITICAL: Corpus Structure

The corpus contains **full Wikipedia articles about named entities** (people, places, organizations, events, etc.). Each entry in `documents.jsonl` is a complete Wikipedia article, potentially several thousand words long with multiple sections.

The `doc_id` is a unique identifier for each article.

**Query characteristics**: Queries specify multiple attributes that the target entity must satisfy simultaneously — effectively a set intersection (e.g., "a musician born in France who won a Grammy before 1990"). The goal is to find the entity whose article satisfies all stated conditions. The primary retrieval challenge is that **query attributes are spread across different sections** of a long article, and BM25 may not surface documents where all terms co-occur.

**Key preprocessing strategies to consider**:
- Split articles into sections and index each section separately — this avoids term dilution across a long document
- Prepend the entity name (article title) to every section chunk so each chunk is identifiable
- Extract structured facts (birth dates, nationalities, awards) from infobox-style text if present, and include them as explicit terms in chunks
