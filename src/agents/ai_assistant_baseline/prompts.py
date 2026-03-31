"""
Prompts for AI coding assistants.
"""

from typing import List
from schema import Document


def create_chunking_prompt(documents: List[Document], task_description: str) -> str:
    """Create prompt for AI assistant to generate chunking code."""
    
    # Sample documents for context
    sample_docs = documents[:3]
    doc_samples = "\n\n".join([
        f"Document {i+1}:\nDoc ID: {doc.doc_id}\nContent: {doc.text[:500]}..."
        for i, doc in enumerate(sample_docs)
    ])
    
    prompt = f"""You are an expert at document preprocessing for information retrieval.

TASK: {task_description}

DATASET INFO:
- Total documents: {len(documents)}
- Average doc length: {sum(len(d.text) for d in documents) / len(documents):.0f} chars
- Domain: Academic papers (titles, abstracts, technical content)

CURRENT BASELINE PERFORMANCE:
- Strategy: 1 chunk per document (no splitting)
- Recall@10: 0.6944
- Recall@100: 0.7222
- nDCG@10: 0.1492

YOUR GOAL: BEAT THE BASELINE by creating BETTER chunks!

CONSTRAINTS:
- Must create searchable text chunks for BM25 retrieval
- Balance between chunk size (specificity) and chunk count (coverage)
- Goal: Maximize recall@10, recall@100, and nDCG@10

WHY CHUNKING MATTERS:
- Long documents dilute keyword signals → worse BM25 scores
- Smaller, focused chunks = more precise matches
- BUT: Too small = lose context, too many = noise
- Optimal: 3-10 chunks per document with semantic coherence

SAMPLE DOCUMENTS:
{doc_samples}

YOUR TASK:
Write a Python function that chunks documents BETTER than the baseline (1 chunk per doc).

REQUIREMENTS:
1. Function signature: `def chunk_documents(documents: List[Document]) -> List[Chunk]`
2. Input: List of Document objects with attributes: doc_id, text
3. Output: List of Chunk objects with attributes: chunk_id, doc_id, text
4. MUST create multiple chunks per document (aim for 3-10 chunks per doc)
5. Each chunk should be 200-800 characters (not too small, not too large)
6. Use smart splitting: by paragraphs, sentences, or semantic boundaries
7. Consider adding slight overlap (50-100 chars) between chunks
8. Each chunk must have a unique chunk_id and reference its parent doc_id

STRATEGIES TO CONSIDER:
- Split by double newlines (paragraphs)
- Split by sentences if paragraphs too long
- Add chunk overlap for context
- Keep chunks between 200-800 chars
- Preserve complete sentences

Example structure:
```python
def chunk_documents(documents: List[Document]) -> List[Chunk]:
    chunks = []
    for doc in documents:
        # Split text into paragraphs/sentences
        # Create multiple chunks with overlap
        # Aim for 3-10 chunks per document
        text_parts = ...  # your smart splitting logic
        
        for i, part in enumerate(text_parts):
            chunk = Chunk(
                chunk_id=f"{{doc.doc_id}}_{{i}}",
                doc_id=doc.doc_id,
                text=part
            )
            chunks.append(chunk)
    return chunks

Return ONLY the Python function code, no explanations."""

    return prompt