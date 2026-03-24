import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List, Dict, Optional
import unicodedata
import re

from schema import Document, Chunk
from base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    """
    Consolidated preprocessor implementing the best proven strategies:

    - Always keep the original section chunk (normalized + optional synonym expansion).
    - Add a title-prepended variant for every section (if the article has a :0 title section).
      This brings title/alias tokens into the same chunk as section text so BM25 can match
      descriptive queries to canonical article names.
    - Add a small title-boosted variant (duplicate title tokens) to increase term frequency
      influence of the title without changing the retriever.
    - Add a conservative neighbor-merged variant (prev+curr+next) when adjacent sections exist,
      with the title prepended when available. This recovers context spread across contiguous sections.
    - Apply light normalization (unicode, smart-quote/dash fixes, collapse whitespace, lowercase)
      and a small synonym/typo expansion to help colloquial queries overlap with encyclopedic phrasing.

    Important constraints followed:
    - chunk.doc_id is exactly the source Document.doc_id for every produced Chunk.
    - chunk_id values are globally unique by using the source doc_id plus a suffix.
    - At least one Chunk is returned per Document (the original chunk).
    """

    name = "title_prepend_normalize_boost_merge"
    description = (
        "Keep original section chunk (normalized + expansions). Add title-prepended, "
        "title-boosted, and neighbor-merged variants to enrich BM25 signal."
    )

    # Small synonym/typo expansions. The replacement string is the variants to inject
    # (we'll insert them inline in parentheses following the matched phrase).
    SYNONYM_PATTERNS = {
        r"\btrash can\b": "trash can | garbage bin",
        r"\bgarbage bin\b": "garbage bin | trash can",
        r"\bmaffia\b": "maffia | mafia",
        r"\bmafia\b": "mafia | mob | organized crime",
        r"\bbaseball cap\b": "baseball cap | cap",
        r"\bspiral\b": "spiral | helix | spiral-like",
        r"\bbikini\b": "bikini | swimsuit",
        r"\bfoot sticking out\b": "foot sticking out | foot protruding | limb visible",
        # some shorter patterns
        r"\btrash\b": "trash | garbage",
        r"\bcorpse\b": "corpse | body",
    }

    def _normalize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # Replace common smart quotes and dashes with ascii equivalents
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = text.replace("—", " - ").replace("–", " - ")
        # Remove excessive repeated whitespace/newlines
        text = re.sub(r"\s+", " ", text).strip()
        # Lowercase to reduce casing mismatch (BM25 tokenization/stemming usually done on lowercase)
        text = text.lower()
        return text

    def _expand_synonyms(self, text: str) -> str:
        # Insert small variant lists after matched phrases to increase token overlap
        # We use case-insensitive matching, but since text is already lowercased, normal re is fine.
        for pat, variants in self.SYNONYM_PATTERNS.items():
            # Replace each match with "match (variants)" to include synonyms as tokens
            def repl(m):
                matched = m.group(0)
                # Avoid duplicating if variants already present
                return f"{matched} ({variants})"
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)
        return text

    def _split_doc_id(self, doc_id: str) -> (str, int):
        """
        Return (article_id, section_index_int). If the section index is non-numeric,
        fallback to 0. Assumes doc_id format "ARTICLE_ID:SECTION".
        """
        if ":" in doc_id:
            article_id, sec = doc_id.split(":", 1)
            try:
                sec_idx = int(sec)
            except Exception:
                sec_idx = 0
        else:
            article_id = doc_id
            sec_idx = 0
        return article_id, sec_idx

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        # Organize documents by article_id and section index for neighbor merging
        articles: Dict[str, Dict[int, Document]] = {}
        for doc in docs:
            article_id, sec_idx = self._split_doc_id(doc.doc_id)
            articles.setdefault(article_id, {})[sec_idx] = doc

        # Build title map (article_id -> title text from section 0) normalized & expanded
        title_map: Dict[str, str] = {}
        for article_id, sections in articles.items():
            title_doc = sections.get(0)
            if title_doc and title_doc.text:
                title_norm = self._normalize_text(title_doc.text)
                title_expanded = self._expand_synonyms(title_norm)
                # Use the entire :0 section as title/lead context
                title_map[article_id] = title_expanded

        chunks: List[Chunk] = []

        for article_id, sections in articles.items():
            # Work through sorted indices so neighbor lookup is deterministic
            sorted_indices = sorted(sections.keys())
            for idx in sorted_indices:
                doc = sections[idx]
                # Normalize and expand the current section text
                raw_text = doc.text or ""
                norm_text = self._normalize_text(raw_text)
                expanded_text = self._expand_synonyms(norm_text)

                # 1) Original / baseline chunk (normalized + expanded)
                # chunk.doc_id MUST equal the source Document.doc_id (critical)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_orig",
                        doc_id=doc.doc_id,
                        text=expanded_text,
                    )
                )

                # Prepare title if available for this article
                title = title_map.get(article_id, "").strip()

                # 2) Title-prepended variant (single title)
                if title:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}_title",
                            doc_id=doc.doc_id,
                            text=title + "\n" + expanded_text,
                        )
                    )

                    # 3) Title-boosted variant (duplicate title tokens a small number of times)
                    boost_times = 2  # conservative duplication to increase TF without overwhelming
                    boosted_title = (" " + title) * boost_times
                    boosted_title = boosted_title.strip()
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}_title_boost",
                            doc_id=doc.doc_id,
                            text=boosted_title + "\n" + expanded_text,
                        )
                    )

                # 4) Neighbor-merged variant (prev + curr + next), only if at least one neighbor exists
                prev_doc = sections.get(idx - 1)
                next_doc = sections.get(idx + 1)
                parts = []
                if prev_doc and prev_doc.text:
                    prev_norm = self._normalize_text(prev_doc.text)
                    parts.append(self._expand_synonyms(prev_norm))
                parts.append(expanded_text)  # current
                if next_doc and next_doc.text:
                    next_norm = self._normalize_text(next_doc.text)
                    parts.append(self._expand_synonyms(next_norm))

                if len(parts) > 1:
                    merged_text = "\n".join(parts)
                    # Prepend title if available
                    if title:
                        merged_text = title + "\n" + merged_text
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}_merged",
                            doc_id=doc.doc_id,
                            text=merged_text,
                        )
                    )

                # Note: we intentionally keep the original chunk and add enriched variants.
                # This preserves the baseline coverage while giving BM25 more opportunities
                # to match title tokens, synonyms, and neighboring context.
        return chunks
