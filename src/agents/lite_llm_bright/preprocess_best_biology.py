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
        
        for sentence in sentences[:20]:
            lower_sent = sentence.lower()
            
            # Pattern 1: Causality - "X causes Y" → "Why does Y happen?"
            if any(word in lower_sent for word in ['cause', 'causes', 'due to', 'because', 'result in']):
                enrichments.append(f"Why does this happen? {sentence}")
                enrichments.append(f"What causes this? {sentence}")
                enrichments.append(f"I was wondering why {sentence}")
            
            # Pattern 2: Process/mechanism - "X occurs through Y" → "How does X work?"
            if any(word in lower_sent for word in ['process', 'mechanism', 'occurs', 'happens', 'works']):
                enrichments.append(f"How does this work? {sentence}")
                enrichments.append(f"Can someone explain how {sentence}")
                enrichments.append(f"What's the mechanism behind {sentence}")
            
            # Pattern 3: Properties - "X is Y" → "Why is X like this?"
            if any(word in lower_sent for word in ['is a', 'are ', 'has ', 'have ', 'contains']):
                enrichments.append(f"Why is it like this? {sentence}")
                enrichments.append(f"I noticed that {sentence}")
                enrichments.append(f"Claim about {sentence}")
            
            # Pattern 4: Effects - "X affects Y" → "Does X really affect Y?"
            if any(word in lower_sent for word in ['affect', 'influence', 'impact', 'change']):
                enrichments.append(f"Does this really work? {sentence}")
                enrichments.append(f"I read somewhere that {sentence}")
                enrichments.append(f"Is it true that {sentence}")
            
            # Pattern 5: Comparisons → "Which is better?"
            if any(word in lower_sent for word in ['more', 'less', 'better', 'worse', 'compared', 'than']):
                enrichments.append(f"Which is better? {sentence}")
                enrichments.append(f"I'm wondering if {sentence}")
                enrichments.append(f"Is there a difference between {sentence}")
            
            # Pattern 6: Methods/techniques → "How do I do this?"
            if any(word in lower_sent for word in ['method', 'way', 'technique', 'approach', 'can be']):
                enrichments.append(f"How do I do this? {sentence}")
                enrichments.append(f"What's the best way to {sentence}")
                enrichments.append(f"Looking for ways to {sentence}")
        
        # Add conversational framing with document content
        if sentences:
            first_sent = sentences[0]
            enrichments.extend([
                f"I was wondering about {first_sent}",
                f"Can anyone explain {first_sent}",
                f"Article about {first_sent}",
                f"Why does {first_sent}",
                f"How come {first_sent}",
                f"I noticed that {first_sent}",
                f"Someone told me that {first_sent}",
                f"I read that {first_sent}",
                f"Is it true that {first_sent}",
                f"Claim in article about {first_sent}"
            ])
        
        # Add common query patterns
        query_starters = [
            "wondering if", "want to know", "can someone", "I was wondering", 
            "why does", "how come", "is it possible", "can you explain",
            "I noticed", "I read", "claim about", "article says",
            "does this mean", "is this true", "someone told me",
            "I'm curious about", "what causes", "looking for", "trying to understand"
        ]
        
        # Add query starters with context from middle of document
        if len(sentences) > 5:
            mid_sent = sentences[len(sentences)//2]
            for starter in query_starters[:10]:
                enrichments.append(f"{starter} {mid_sent}")
        
        return " ".join(enrichments[:40])
