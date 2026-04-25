import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List
import re
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "analysis_code_agent"  # MUST BE EXACTLY THIS - DO NOT MODIFY
    description = "Domain-specific terminology bridging via expansion dictionary and synonym injection."

    def __init__(self):
        super().__init__()
        
        # Mapping of LaTeX patterns to English expansions
        self.latex_expansions = [
            # Binomial and combinatorial notation
            (r'\$?\\binom\{([^}]+)\}\{([^}]+)\}\$?', r'binomial coefficient \1 choose \2'),
            (r'\$?\\comb\{([^}]+)\}\{([^}]+)\}\$?', r'combination \1 choose \2'),
            (r'\$?P\(([^)]+)\)\$?', r'probability of \1'),
            (r'\$?E\[([^\]]+)\]\$?', r'expected value of \1'),
            
            # Common mathematical functions
            (r'\$?\\log\(([^)]+)\)\$?', r'logarithm of \1'),
            (r'\$?\bln\(([^)]+)\)\$?', r'natural logarithm of \1'),
            (r'\$?\bexp\(([^)]+)\)\$?', r'exponential of \1'),
            (r'\$?\\sqrt\{([^}]+)\}\$?', r'square root of \1'),
            
            # Utility and function notation
            (r'\$?U\(([^)]+)\)\$?', r'utility function of \1'),
            (r'\$?u\(([^)]+)\)\$?', r'utility function of \1'),
            (r'\$?V\(([^)]+)\)\$?', r'value function of \1'),
            (r'\$?v\(([^)]+)\)\$?', r'value function of \1'),
            (r'\$?f\(([^)]+)\)\$?', r'function of \1'),
            
            # Summation and products
            (r'\$?\\sum\$?', r'sum'),
            (r'\$?\\prod\$?', r'product'),
            (r'\$?\\int\$?', r'integral'),
            
            # Set notation
            (r'\$?\\in\$?', r'in'),
            (r'\$?\\notin\$?', r'not in'),
            (r'\$?\\subset\$?', r'subset'),
            (r'\$?\\cup\$?', r'union'),
            (r'\$?\\cap\$?', r'intersection'),
            
            # Inequality and equals
            (r'\$?\\geq\$?', r'greater than or equal to'),
            (r'\$?\\leq\$?', r'less than or equal to'),
            (r'\$?\\neq\$?', r'not equal to'),
            
            # Greek letters (common in mathematics)
            (r'\$?\\alpha\$?', r'alpha'),
            (r'\$?\\beta\$?', r'beta'),
            (r'\$?\\gamma\$?', r'gamma'),
            (r'\$?\\delta\$?', r'delta'),
            (r'\$?\\theta\$?', r'theta'),
            (r'\$?\\lambda\$?', r'lambda'),
            (r'\$?\\mu\$?', r'mu'),
            (r'\$?\\sigma\$?', r'sigma'),
            
            # Subscripts and superscripts (basic handling)
            (r'\$?([a-zA-Z])_\{([^}]+)\}\$?', r'\1 subscript \2'),
            (r'\$?([a-zA-Z])\^\{([^}]+)\}\$?', r'\1 to the power of \2'),
        ]
        
        # Domain-specific terminology expansion dictionary
        # Maps code patterns/variable names to problem domain terminology
        self.domain_expansions = {
            # Data structure patterns
            'ddict': 'dictionary trie autocomplete prefix matching',
            'trie': 'trie prefix tree dictionary structure',
            'heap': 'heap priority queue',
            'queue': 'queue fifo first in first out',
            'stack': 'stack lifo last in first out',
            'deque': 'double ended queue deque',
            'set': 'set collection unique elements',
            'dict': 'dictionary map hash table',
            'list': 'list array sequence',
            'tree': 'tree graph structure nodes edges',
            'graph': 'graph nodes edges vertices',
            'node': 'node vertex point',
            'edge': 'edge connection link',
            
            # Algorithmic patterns
            'dfs': 'depth first search dfs graph traversal',
            'bfs': 'breadth first search bfs graph traversal',
            'dijkstra': 'dijkstra shortest path algorithm',
            'floyd': 'floyd warshall all pairs shortest path',
            'kruskal': 'kruskal minimum spanning tree mst',
            'prim': 'prim minimum spanning tree mst',
            'union': 'union find disjoint set data structure',
            'dp': 'dynamic programming optimization memoization',
            'greedy': 'greedy algorithm optimization',
            'binary_search': 'binary search sorted array logarithmic',
            'merge_sort': 'merge sort sorting algorithm divide conquer',
            'quick_sort': 'quick sort sorting algorithm divide conquer',
            'segment_tree': 'segment tree range query range update',
            'fenwick': 'fenwick tree binary indexed tree range sum',
            'bit': 'binary indexed tree range sum fenwick',
            
            # Common operation patterns
            'add': 'insert add element operation',
            'remove': 'delete remove element operation',
            'find': 'search find lookup query element',
            'update': 'update modify change element',
            'query': 'query retrieve fetch data',
            'sort': 'sort ordering arrangement',
            'search': 'search find lookup element',
            'insert': 'insert add element operation',
            'delete': 'delete remove element operation',
            'build': 'construct build create initialize',
            
            # Mathematical/logical patterns
            'min': 'minimum smallest value',
            'max': 'maximum largest value',
            'sum': 'sum total aggregate',
            'count': 'count enumerate frequency',
            'xor': 'xor exclusive or bitwise operation',
            'and': 'and bitwise and operation',
            'or': 'or bitwise or operation',
            'gcd': 'greatest common divisor gcd',
            'lcm': 'least common multiple lcm',
            'prime': 'prime number primality',
            'factorial': 'factorial combinatorial',
            'modulo': 'modulo remainder modular arithmetic',
            
            # Problem domain patterns
            'autocompletion': 'autocompletion autocomplete text editor prefix',
            'editor': 'editor text editing',
            'undo': 'undo revert operation history',
            'redo': 'redo restore operation history',
            'prefix': 'prefix matching prefix search',
            'suffix': 'suffix matching suffix search',
            'substring': 'substring subsequence pattern matching',
            'subsequence': 'subsequence substring pattern',
            'pattern': 'pattern matching search string',
            'palindrome': 'palindrome symmetric string',
            'anagram': 'anagram permutation rearrangement',
            'permutation': 'permutation arrangement combination',
            'combination': 'combination selection arrangement',
            'subset': 'subset power set elements',
            'path': 'path route navigation',
            'cycle': 'cycle circular loop',
            'component': 'connected component graph connectivity',
            'clique': 'clique fully connected subgraph',
            'matching': 'matching bipartite graph',
            'flow': 'maximum flow network flow',
            'spanning_tree': 'spanning tree minimum spanning tree mst',
        }

    def expand_latex(self, text: str) -> str:
        """
        Expand LaTeX notation to English descriptions.
        Applies regex substitutions for common mathematical notation.
        """
        expanded = text
        
        for pattern, replacement in self.latex_expansions:
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        
        # Remove remaining LaTeX dollar signs and braces that weren't matched
        expanded = re.sub(r'\$+', ' ', expanded)
        expanded = re.sub(r'\\', '', expanded)  # Remove backslashes
        
        # Clean up multiple spaces
        expanded = re.sub(r'\s+', ' ', expanded)
        
        return expanded.strip()

    def extract_title_from_doc_id(self, doc_id: str) -> str:
        """
        Extract a likely title from the doc_id by parsing filename conventions.
        Examples: "24073089:1" -> "24073089", "Handicap_principle.txt" -> "Handicap principle"
        """
        # Remove the chunk suffix if present (e.g., ":1" from "24073089:1")
        base_id = doc_id.split(':')[0]
        
        # If it looks like a filename with underscores/hyphens, convert to title case
        if '_' in base_id or '-' in base_id:
            # Replace underscores/hyphens with spaces and title case
            title = base_id.replace('_', ' ').replace('-', ' ')
            # Remove common file extensions
            title = re.sub(r'\.(txt|md|pdf)$', '', title, flags=re.IGNORECASE)
            return title
        
        return base_id

    def extract_key_sections(self, text: str, limit_chars: int = 2000) -> str:
        """
        Extract the beginning/most important sections of text.
        Focus on opening paragraphs which often contain key definitions and context.
        """
        # Split into paragraphs (multiple newlines or double spaces indicate breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        selected = []
        char_count = 0
        
        # Include first 3-4 paragraphs or until we reach limit
        for i, para in enumerate(paragraphs[:4]):
            para = para.strip()
            if not para:
                continue
            
            selected.append(para)
            char_count += len(para)
            
            if char_count >= limit_chars:
                break
        
        return '\n\n'.join(selected)

    def extract_and_rephrase_identifiers(self, text: str) -> str:
        """
        Extract class names, function names, and variable names from code.
        Rephrase them as natural language query-like terms.
        
        For example:
        - class Ddict -> "ddict data structure dictionary"
        - def prune_tree() -> "prune pruning tree operation function"
        - variable longest_subsegment -> "longest subsegment length"
        
        This bridges the semantic gap between code implementation and problem description.
        """
        identifiers = []
        
        # Extract class definitions: class ClassName
        class_pattern = r'\bclass\s+([A-Za-z_][A-Za-z0-9_]*)'
        for match in re.finditer(class_pattern, text):
            class_name = match.group(1)
            # Rephrase: "class ClassName" -> "ClassName data structure class"
            rephrased = self._rephrase_identifier(class_name)
            identifiers.append(f"{rephrased} data structure class")
        
        # Extract function/method definitions: def function_name
        func_pattern = r'\bdef\s+([A-Za-z_][A-Za-z0-9_]*)'
        for match in re.finditer(func_pattern, text):
            func_name = match.group(1)
            # Rephrase: "def function_name" -> "function_name operation function"
            rephrased = self._rephrase_identifier(func_name)
            identifiers.append(f"{rephrased} operation function")
        
        # Extract variable assignments (common patterns)
        # Looking for patterns like: variable_name = ... (first 30 unique assignments)
        var_pattern = r'\b([a-z_][a-z0-9_]*)\s*='
        var_names = set()
        for match in re.finditer(var_pattern, text):
            var_name = match.group(1)
            # Skip common single-letter vars and very common names
            if len(var_name) > 1 and var_name not in ('self', 'cls', 'result', 'data', 'value'):
                var_names.add(var_name)
                if len(var_names) >= 15:  # Limit to avoid noise
                    break
        
        for var_name in list(var_names)[:15]:
            rephrased = self._rephrase_identifier(var_name)
            identifiers.append(f"{rephrased} variable")
        
        # Combine all rephrased identifiers
        identifier_text = ' '.join(identifiers)
        
        # Clean up and remove duplicates while preserving order
        seen = set()
        tokens = identifier_text.split()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)
        
        return ' '.join(unique_tokens)

    def _rephrase_identifier(self, identifier: str) -> str:
        """
        Convert camelCase or snake_case identifier to natural language.
        Examples:
        - "autoComplete" -> "auto complete"
        - "longest_subsegment" -> "longest subsegment"
        - "Ddict" -> "ddict"
        """
        # Convert camelCase to snake_case first
        # Insert underscore before uppercase letters (but not at the start)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', identifier)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        
        # Convert to lowercase and replace underscores with spaces
        result = s2.lower().replace('_', ' ')
        
        return result

    def inject_domain_synonyms(self, text: str) -> str:
        """
        Inject domain-specific terminology based on detected code patterns.
        Scans text for identifiers matching the domain expansion dictionary
        and adds enriched terminology to bridge the semantic gap.
        
        H4: Domain-Specific Terminology Bridging
        """
        injected_terms = []
        
        # Look for each code pattern in the text
        for pattern, expansion in self.domain_expansions.items():
            # Create case-insensitive pattern with word boundaries
            # e.g., "ddict" matches "ddict", "Ddict", "DDICT", etc.
            regex_pattern = r'\b' + re.escape(pattern) + r'\b'
            
            if re.search(regex_pattern, text, re.IGNORECASE):
                # Add the expansion once per pattern found
                injected_terms.append(expansion)
        
        # Also check for underscored versions
        for identifier_match in re.finditer(r'\b([a-z_][a-z0-9_]*)\b', text, re.IGNORECASE):
            identifier = identifier_match.group(1).lower()
            # Replace underscores with nothing for lookup (e.g., binary_search -> binarysearch)
            # but also try with underscores (binary_search)
            for check_id in [identifier, identifier.replace('_', '')]:
                if check_id in self.domain_expansions:
                    injected_terms.append(self.domain_expansions[check_id])
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in injected_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        # Combine injected terms
        injected_text = ' '.join(unique_terms)
        
        return injected_text

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        chunks = []
        
        for doc in docs:
            # ===== CHUNK 0: Original full-document (ALWAYS KEEP) =====
            chunk_id_0 = f"{doc.doc_id}_0"
            chunks.append(Chunk(
                chunk_id=chunk_id_0,
                doc_id=doc.doc_id,
                text=doc.text,
                metadata=doc.metadata
            ))
            
            # ===== CHUNK 1: LaTeX-to-English expansion =====
            expanded_text = self.expand_latex(doc.text)
            
            # Only add expansion chunk if it differs meaningfully from original
            if expanded_text != doc.text and len(expanded_text) > 0:
                chunk_id_1 = f"{doc.doc_id}_1"
                chunks.append(Chunk(
                    chunk_id=chunk_id_1,
                    doc_id=doc.doc_id,
                    text=expanded_text,
                    metadata=doc.metadata
                ))
            
            # ===== CHUNK 2: Domain-Specific Terminology Bridge (H4) =====
            # H4: Inject domain-specific synonyms and algorithmic terminology
            domain_synonyms = self.inject_domain_synonyms(doc.text)
            
            # Extract and rephrase code identifiers (from H5)
            extracted_identifiers = self.extract_and_rephrase_identifiers(doc.text)
            
            # Extract key sections for context
            key_sections = self.extract_key_sections(doc.text, limit_chars=1500)
            
            # Create a comprehensive terminology bridge chunk:
            # domain synonyms + extracted identifiers + key problem context
            bridge_parts = []
            if domain_synonyms:
                bridge_parts.append(domain_synonyms)
            if extracted_identifiers:
                bridge_parts.append(extracted_identifiers)
            if key_sections:
                bridge_parts.append(key_sections)
            
            bridge_text = ' '.join(bridge_parts)
            
            if len(bridge_text) > 200:  # Only add if substantial
                chunk_id_2 = f"{doc.doc_id}_2"
                chunks.append(Chunk(
                    chunk_id=chunk_id_2,
                    doc_id=doc.doc_id,
                    text=bridge_text,
                    metadata=doc.metadata
                ))
        
        return chunks
