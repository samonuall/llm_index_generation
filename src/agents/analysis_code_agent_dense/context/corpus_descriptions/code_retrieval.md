## CRITICAL: Corpus Structure

The corpus contains **code files or functions**, potentially from open-source repositories. Each entry in `documents.jsonl` is a complete code unit — a function, class, or file — along with any associated docstring or comments.

The `doc_id` is a unique identifier for each code unit.

**Query characteristics**: Queries describe a programming task or functionality in natural language (e.g., "function that sorts a list of dictionaries by a key"). The goal is to find the code that implements the described functionality. The primary retrieval challenge is the **semantic gap between natural-language descriptions and code tokens**: code uses identifiers, operators, and syntax that may not match query words.

**Key preprocessing strategies to consider**:
- Prioritize docstrings and comments — they bridge natural language and code
- Split camelCase and snake_case identifiers into component words (e.g., `getBestScore` → "get best score")
- Include the function signature prominently as it often captures intent concisely
- Strip boilerplate (imports, decorators) that adds noise without retrieval signal
