import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

import re
from typing import List
from schema import Document, Chunk
from base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    name = "analysis_code_agent"  # MUST BE EXACTLY THIS - DO NOT MODIFY
    description = "Full-doc passthrough + title/conditions/interventions mini-chunk for BM25 length normalization boost."

    # Patterns to extract key sections from clinical trial text
    SECTION_PATTERNS = {
        "title": [
            r"(?:Brief Title|Official Title|Study Title)[:\s]+(.+?)(?=\n[A-Z]|\n\n|$)",
            r"Title[:\s]+(.+?)(?=\n[A-Z]|\n\n|$)",
        ],
        "conditions": [
            r"Conditions?[:\s]+(.+?)(?=\n[A-Z]|\n\n|Interventions?|Eligibility|$)",
            r"Disease(?:s)?[:\s]+(.+?)(?=\n[A-Z]|\n\n|$)",
            r"Condition(?:s)? Studied[:\s]+(.+?)(?=\n[A-Z]|\n\n|$)",
        ],
        "interventions": [
            r"Interventions?[:\s]+(.+?)(?=\n[A-Z]|\n\n|Eligibility|Outcomes?|$)",
            r"Drug(?:s)?[:\s]+(.+?)(?=\n[A-Z]|\n\n|$)",
            r"Treatment(?:s)?[:\s]+(.+?)(?=\n[A-Z]|\n\n|$)",
        ],
        "keywords": [
            r"Keywords?[:\s]+(.+?)(?=\n[A-Z]|\n\n|$)",
            r"MeSH Terms?[:\s]+(.+?)(?=\n[A-Z]|\n\n|$)",
        ],
        "summary": [
            r"Brief Summary[:\s]+(.+?)(?=\n[A-Z]|\n\n|Detailed Description|$)",
            r"Summary[:\s]+(.+?)(?=\n[A-Z]|\n\n|$)",
        ],
    }

    def _extract_section(self, text: str, patterns: list) -> str:
        """Try each pattern and return the first match found."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Limit to reasonable length (first 300 chars)
                if len(content) > 300:
                    content = content[:300]
                return content
        return ""

    def _build_mini_chunk(self, text: str) -> str:
        """
        Extract title, conditions, interventions, keywords, and brief summary
        into a compact mini-chunk to boost BM25 scoring for key terms.
        """
        parts = []

        title = self._extract_section(text, self.SECTION_PATTERNS["title"])
        if title:
            parts.append(f"Title: {title}")

        conditions = self._extract_section(text, self.SECTION_PATTERNS["conditions"])
        if conditions:
            parts.append(f"Conditions: {conditions}")

        interventions = self._extract_section(text, self.SECTION_PATTERNS["interventions"])
        if interventions:
            parts.append(f"Interventions: {interventions}")

        keywords = self._extract_section(text, self.SECTION_PATTERNS["keywords"])
        if keywords:
            parts.append(f"Keywords: {keywords}")

        summary = self._extract_section(text, self.SECTION_PATTERNS["summary"])
        if summary:
            parts.append(f"Summary: {summary}")

        return "\n".join(parts)

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks = []
        for doc in docs:
            # Chunk 0: Always keep the original full-document text (passthrough)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_0",
                doc_id=doc.doc_id,
                text=doc.text,
                metadata=doc.metadata
            ))

            # Chunk 1: Mini-chunk with title + conditions + interventions + keywords + summary
            mini_text = self._build_mini_chunk(doc.text)
            if mini_text and len(mini_text) > 20:
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_1",
                    doc_id=doc.doc_id,
                    text=mini_text,
                    metadata=doc.metadata
                ))

        return chunks
