"""Execute LLM-generated chunking code safely."""

from typing import List
from schema import Document, Chunk

def execute_chunking_code(code: str, documents: List[Document]) -> List[Chunk]:
    """
    Execute generated chunking code.
    
    Args:
        code: Python code that defines a chunk_documents function
        documents: Documents to chunk
        
    Returns:
        List of chunks
    """
    namespace = {
        'Document': Document,
        'Chunk': Chunk,
        'documents': documents,
        'List': List,
    }
    
    exec(code, namespace)
    
    if 'chunk_documents' in namespace:
        chunks = namespace['chunk_documents'](documents)
    else:
        raise ValueError("Generated code must define 'chunk_documents' function")
    
    return chunks
