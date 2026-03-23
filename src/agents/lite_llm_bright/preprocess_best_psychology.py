"""
Preprocessor that enriches documents with conversational reformulations
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
    description = "Enriches documents with conversational reformulations and natural question phrasings"

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
        """Generate conversational reformulations from document text."""
        enrichments = []
        
        # Extract sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        for sentence in sentences[:15]:
            lower_sent = sentence.lower()
            
            # Add modal verbs and hedging language
            modal_variants = [
                f"Can {sentence}",
                f"Could {sentence}",
                f"May {sentence}",
                f"Might {sentence}",
                f"Should {sentence}",
                f"Would {sentence}",
                f"Is it possible that {sentence}",
                f"Does this mean {sentence}",
                f"Maybe {sentence}",
                f"Perhaps {sentence}",
                f"Probably {sentence}",
                f"It seems {sentence}",
                f"It appears {sentence}"
            ]
            enrichments.extend(modal_variants[:3])
            
            # Add conversational markers
            conversational = [
                f"Something like {sentence}",
                f"Anything about {sentence}",
                f"Even {sentence}",
                f"Just {sentence}",
                f"Really {sentence}",
                f"Actually {sentence}",
                f"Someone said {sentence}",
                f"People say {sentence}",
                f"I heard {sentence}",
                f"I read {sentence}",
                f"I think {sentence}",
                f"I know {sentence}",
                f"I believe {sentence}"
            ]
            enrichments.extend(conversational[:3])
            
            # Question patterns with wh-words
            if any(word in lower_sent for word in ['cause', 'reason', 'because', 'due to']):
                enrichments.extend([
                    f"Why {sentence}",
                    f"What causes {sentence}",
                    f"How come {sentence}",
                    f"Why does {sentence}",
                    f"Why is {sentence}",
                    f"What is the reason {sentence}"
                ])
            
            if any(word in lower_sent for word in ['process', 'mechanism', 'works', 'functions']):
                enrichments.extend([
                    f"How {sentence}",
                    f"How does {sentence}",
                    f"How can {sentence}",
                    f"What is the mechanism {sentence}",
                    f"How do you {sentence}"
                ])
            
            if any(word in lower_sent for word in ['different', 'similar', 'compare', 'versus']):
                enrichments.extend([
                    f"Which {sentence}",
                    f"What is the difference {sentence}",
                    f"Which is better {sentence}",
                    f"Are they different {sentence}"
                ])
            
            # Uncertainty patterns
            if any(word in lower_sent for word in ['may', 'might', 'could', 'possible']):
                enrichments.extend([
                    f"Is it possible {sentence}",
                    f"Can this {sentence}",
                    f"Could this {sentence}",
                    f"Might this {sentence}"
                ])
            
            # Subjective perspective
            personal_forms = [
                f"Someone who {sentence}",
                f"A person {sentence}",
                f"People {sentence}",
                f"I wonder if {sentence}",
                f"I want to know {sentence}",
                f"I was wondering {sentence}"
            ]
            enrichments.extend(personal_forms[:2])
        
        # Add query-style reformulations of key sentences
        if len(sentences) >= 3:
            for i in [0, len(sentences)//2, -1]:
                sent = sentences[i]
                enrichments.extend([
                    f"wondering if {sent}",
                    f"want to know {sent}",
                    f"can someone explain {sent}",
                    f"does anyone know {sent}",
                    f"looking for {sent}",
                    f"trying to understand {sent}",
                    f"curious about {sent}",
                    f"help with {sent}",
                    f"question about {sent}",
                    f"confused about {sent}"
                ])
        
        return " ".join(enrichments[:50])
