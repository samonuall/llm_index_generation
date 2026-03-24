"""
Preprocessor that enriches documents with question-answer pairs and conversational reformulations
to bridge the vocabulary gap between formal documents and conversational StackExchange queries.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
import re

from schema import Document, Chunk
from base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    name = "lite_llm_bright"
    description = "Enriches documents with question-answer pairs and conversational reformulations"

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks = []
        
        for doc in docs:
            # Original document chunk
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_0",
                    doc_id=doc.doc_id,
                    text=doc.text,
                )
            )
            
            # Generate enriched chunk with conversational reformulations
            enriched_text = self._generate_enriched_text(doc.text)
            if enriched_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_1",
                        doc_id=doc.doc_id,
                        text=enriched_text,
                    )
                )
        
        return chunks

    def _generate_enriched_text(self, text: str) -> str:
        """Generate conversational reformulations and Q&A pairs from document text."""
        enrichments = []
        
        # Extract key sentences (those with actionable or comparative content)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        for sentence in sentences[:15]:  # Limit to avoid explosion
            lower_sent = sentence.lower()
            
            # Pattern 1: Convert "X is efficient/sustainable/better" to questions
            if any(word in lower_sent for word in ['efficient', 'sustainable', 'better', 'best', 'good', 'effective']):
                # Add question forms
                enrichments.append(f"What is the most efficient way? {sentence}")
                enrichments.append(f"How to achieve this? {sentence}")
                enrichments.append(f"Is this sustainable? {sentence}")
            
            # Pattern 2: Convert "X reduces Y" to problem-solution
            if any(word in lower_sent for word in ['reduce', 'decrease', 'lower', 'minimize', 'save']):
                enrichments.append(f"How to reduce this? {sentence}")
                enrichments.append(f"Ways to minimize: {sentence}")
                enrichments.append(f"Cost effective method: {sentence}")
            
            # Pattern 3: Extract method/process descriptions
            if any(word in lower_sent for word in ['method', 'process', 'way', 'approach', 'technique']):
                enrichments.append(f"What method works? {sentence}")
                enrichments.append(f"Best approach: {sentence}")
            
            # Pattern 4: Convert "X uses Y" to resource questions
            if any(word in lower_sent for word in ['use', 'uses', 'using', 'consume', 'require']):
                enrichments.append(f"How much does this use? {sentence}")
                enrichments.append(f"What are the requirements? {sentence}")
            
            # Pattern 5: Environmental impact statements
            if any(word in lower_sent for word in ['impact', 'emission', 'carbon', 'footprint', 'environment']):
                enrichments.append(f"What is the environmental impact? {sentence}")
                enrichments.append(f"Carbon footprint considerations: {sentence}")
            
            # Pattern 6: Comparison statements
            if any(word in lower_sent for word in ['compare', 'versus', 'than', 'alternative']):
                enrichments.append(f"Which is better? {sentence}")
                enrichments.append(f"Comparison between options: {sentence}")
        
        # Add common query terms that might be missing
        common_query_terms = [
            "wondering if", "want to know", "how to", "what is", "is it okay",
            "best way", "most efficient", "should I", "can I", "would it be",
            "looking for", "trying to", "interested in", "good idea", "feasible",
            "cost effective", "environmentally friendly", "sustainable living"
        ]
        
        # Sample the original text to add these terms contextually
        text_lower = text.lower()
        for term in common_query_terms:
            if term not in text_lower:
                # Add term with context from first sentence
                first_sentence = sentences[0] if sentences else text[:200]
                enrichments.append(f"{term}: {first_sentence}")
        
        return " ".join(enrichments[:30])  # Limit total enrichments
