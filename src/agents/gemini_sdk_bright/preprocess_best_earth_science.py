import sys
import re
import pathlib
from typing import List, Tuple
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from schema import Document, Chunk
from base import BasePreprocessor

# Base list of English stopwords
BASE_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot',
    'com', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', 'don', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he',
    "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i',
    "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 'just', 'k',
    'let', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'r', 'same', 'shan', "shan't",
    'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll",
    "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'us', 'very', 'was',
    "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's",
    'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would',
    "wouldn't", 'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
}

class Preprocessor(BasePreprocessor):
    name = "gemini_sdk_bright"
    description = (
        "Paragraph chunking with a dense, label-free header of concepts and diverse synthetic questions to improve BM25 relevance."
    )

    # Chunking parameters
    PARAGRAPH_WORD_LIMIT = 150
    CHUNK_SIZE_SENTENCES = 4
    CHUNK_OVERLAP_SENTENCES = 2

    # Augmentation parameters
    DOC_LEVEL_CONCEPTS = 10
    CHUNK_LEVEL_CONCEPTS = 8
    MAX_TOTAL_CONCEPTS = 15
    MAX_QUESTIONS_PER_CHUNK = 10

    def __init__(self):
        super().__init__()
        self.nltk = None
        self.stopwords = BASE_STOPWORDS
        try:
            import nltk
            self.nltk = nltk
            for pkg, path in [
                ('punkt', 'tokenizers/punkt'),
                ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
                ('stopwords', 'corpora/stopwords'),
            ]:
                try:
                    self.nltk.data.find(path)
                except LookupError:
                    self.nltk.download(pkg, quiet=True)

            from nltk.corpus import stopwords as nltk_stopwords
            self.stopwords = BASE_STOPWORDS.union(set(nltk_stopwords.words('english')))
        except ImportError:
            print("Warning: NLTK not found. Augmentation features will be disabled.", file=sys.stderr)

    def _extract_title_and_content(self, text: str) -> Tuple[str, str]:
        """Extracts the first non-empty line as a title if it's short and not a full sentence."""
        lines = text.strip().split('\n')
        title = ""
        content_start_index = 0
        if lines:
            first_line = lines[0].strip()
            # Heuristic: A short line that doesn't end with sentence punctuation is likely a title.
            if 0 < len(first_line.split()) < 20 and not first_line.endswith(('.', '?', '!')):
                title = first_line
                content_start_index = 1
        content = '\n'.join(lines[content_start_index:])
        return title, content

    def _generate_diverse_questions(self, concepts: List[str], title: str) -> List[str]:
        """Generates a diverse set of hypothetical questions and statements from key concepts."""
        if not concepts:
            return []

        questions = set()
        primary_concept = concepts[0]

        # --- Questions about the primary concept ---
        questions.add(f"What is {primary_concept}?")
        questions.add(f"How does {primary_concept} work?")
        questions.add(f"Why is {primary_concept} important?")
        questions.add(f"What are the benefits of {primary_concept}?")
        questions.add(f"What are the pros and cons of {primary_concept}?")
        questions.add(f"What are the effects of {primary_concept}?")
        questions.add(f"What causes {primary_concept}?")
        questions.add(f"How long does {primary_concept} take?")
        questions.add(f"Example of {primary_concept}.")
        questions.add(f"Applications of {primary_concept}.")
        questions.add(f"Define {primary_concept}.")
        questions.add(f"Explanation of {primary_concept}.")
        
        # --- Relational questions ---
        if len(concepts) > 1:
            for i in range(1, min(5, len(concepts))):
                secondary_concept = concepts[i]
                questions.add(f"What is the relationship between {primary_concept} and {secondary_concept}?")
                questions.add(f"Compare {primary_concept} and {secondary_concept}.")
                questions.add(f"How does {primary_concept} affect {secondary_concept}?")
                questions.add(f"How does {secondary_concept} influence {primary_concept}?")

        # --- Title-related questions ---
        if title:
            title_lower = title.lower()
            questions.add(f"What is {title_lower}?")
            questions.add(f"Summary of {title_lower}.")
            if primary_concept.lower() not in title_lower:
                questions.add(f"How does {primary_concept} relate to {title_lower}?")

        # Simple transformation query for BM25 to act as a related terms list
        questions.add(f"Information about {', '.join(concepts[:4])}.")

        return sorted(list(questions))[:self.MAX_QUESTIONS_PER_CHUNK]

    def _extract_key_concepts(self, text: str, title: str, num_concepts: int) -> List[str]:
        """Extracts and scores salient noun phrases, boosting concepts from definitions and topic sentences."""
        if not self.nltk or num_concepts == 0: return []
        try:
            words = self.nltk.word_tokenize(text)
            if not words: return []

            tagged = self.nltk.pos_tag(words)
            grammar = r"NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}"
            parser = self.nltk.RegexpParser(grammar)
            tree = parser.parse(tagged)

            raw_phrases = [" ".join(word for word, _ in subtree.leaves()) for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP')]
            if not raw_phrases: return []

            phrase_map, cleaned_phrases_list = {}, []
            for phrase in raw_phrases:
                words_lower = phrase.lower().split()
                while words_lower and words_lower[0] in self.stopwords:
                    words_lower.pop(0)
                if not words_lower: continue

                cleaned_phrase = " ".join(words_lower)
                if len(cleaned_phrase) < 4 or cleaned_phrase in self.stopwords: continue

                cleaned_phrases_list.append(cleaned_phrase)
                if cleaned_phrase not in phrase_map:
                    phrase_map[cleaned_phrase] = phrase

            if not cleaned_phrases_list: return []

            freq_counts = Counter(cleaned_phrases_list)
            phrase_scores = Counter()
            title_lower = title.lower()
            
            sents = []
            try:
                sents = self.nltk.sent_tokenize(text)
            except Exception: pass
            
            first_sentence = sents[0].lower() if sents else ""
            
            definition_sents = set()
            definition_pattern = re.compile(r'\b(is|are|was|were|refers to|is defined as)\s+(a|an|the)\b', re.IGNORECASE)
            if sents:
                for sent in sents:
                    if definition_pattern.search(sent):
                        definition_sents.add(sent.lower())

            for phrase, count in freq_counts.items():
                score = float(count)
                original_phrase = phrase_map[phrase]
                
                if len(phrase.split()) > 1: score *= 1.2
                if any(w.isupper() for w in original_phrase.split() if len(w) > 1): score *= 1.5
                if title_lower and phrase in title_lower: score *= 2.0
                if first_sentence and phrase in first_sentence: score *= 1.8
                
                if definition_sents:
                    for def_sent in definition_sents:
                        if phrase in def_sent:
                            match = definition_pattern.search(def_sent)
                            if match and def_sent.find(phrase) < match.start():
                                score *= 2.5
                                break
                
                phrase_scores[phrase] = score
            
            top_phrases_cleaned = [p for p, _ in phrase_scores.most_common(num_concepts)]
            return [phrase_map.get(p, p) for p in top_phrases_cleaned]

        except Exception:
            return []

    def _create_chunk(self, doc_id: str, chunk_id_counter: int, chunk_content: str, title: str, doc_level_concepts: List[str]) -> Chunk:
        """Creates an augmented chunk with a dense header for better term matching."""
        summary = ""
        if self.nltk:
            try:
                sentences = self.nltk.sent_tokenize(chunk_content)
                # Use first sentence as a summary if it's a reasonable length
                if sentences and len(sentences[0].split()) > 4:
                    summary = sentences[0]
            except Exception:
                pass

        chunk_level_concepts = self._extract_key_concepts(chunk_content, title, num_concepts=self.CHUNK_LEVEL_CONCEPTS)

        # Merge and deduplicate concepts, prioritizing chunk-level (more specific) ones
        combined_concepts = []
        seen_concepts_lower = set()
        for concept in chunk_level_concepts + doc_level_concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen_concepts_lower:
                combined_concepts.append(concept)
                seen_concepts_lower.add(concept_lower)
        
        final_concepts = combined_concepts[:self.MAX_TOTAL_CONCEPTS]
        questions = self._generate_diverse_questions(final_concepts, title)

        # Create a clean, label-free augmentation header
        header_parts = []
        if title:
            header_parts.append(title)
        if summary:
            header_parts.append(summary)
        if final_concepts:
            header_parts.append(", ".join(final_concepts))
        if questions:
            # Clean up questions to be more like a text block
            cleaned_queries = [q.lower().replace('?', '').replace('.', '') for q in questions]
            header_parts.append('. '.join(cleaned_queries))
        
        augmentation_header = ". ".join(filter(None, header_parts))
        if augmentation_header and not augmentation_header.endswith('.'):
            augmentation_header += "."
        
        chunk_text = f"{augmentation_header}\n\n{chunk_content}".strip()
        
        return Chunk(
            chunk_id=f"{doc_id}_{chunk_id_counter}",
            doc_id=doc_id,
            text=chunk_text,
            metadata={},
        )

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """Splits documents into smaller, paragraph-based chunks and augments them."""
        all_chunks = []
        for doc in docs:
            doc_text = doc.text or ""
            if not doc_text.strip():
                continue
            
            title, content = self._extract_title_and_content(doc_text)
            
            # Extract document-level concepts once to provide global context to all chunks
            doc_level_text = f"{title}. {content}" if title else content
            doc_level_concepts = self._extract_key_concepts(doc_level_text, title, num_concepts=self.DOC_LEVEL_CONCEPTS)

            doc_chunks = []
            chunk_idx_counter = 0

            paragraphs = re.split(r'\n\s*\n', content)

            for para in paragraphs:
                para_text = para.strip()
                if not para_text:
                    continue
                
                if len(para_text.split()) > self.PARAGRAPH_WORD_LIMIT and self.nltk:
                    try:
                        sentences = self.nltk.sent_tokenize(para_text)
                    except Exception:
                        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', para_text) if s.strip()]

                    if not sentences: continue
                    
                    stride = self.CHUNK_SIZE_SENTENCES - self.CHUNK_OVERLAP_SENTENCES
                    stride = max(1, stride)
                    
                    for i in range(0, len(sentences), stride):
                        chunk_sentences = sentences[i : i + self.CHUNK_SIZE_SENTENCES]
                        if not chunk_sentences: continue
                        chunk_content = " ".join(chunk_sentences)
                        chunk = self._create_chunk(doc.doc_id, chunk_idx_counter, chunk_content, title, doc_level_concepts)
                        doc_chunks.append(chunk)
                        chunk_idx_counter += 1
                else:
                    chunk = self._create_chunk(doc.doc_id, chunk_idx_counter, para_text, title, doc_level_concepts)
                    doc_chunks.append(chunk)
                    chunk_idx_counter += 1

            if not doc_chunks and doc_text.strip():
                full_text_for_chunking = f"{title}\n{content}".strip()
                if full_text_for_chunking:
                     chunk = self._create_chunk(doc.doc_id, 0, full_text_for_chunking, title, doc_level_concepts)
                     doc_chunks.append(chunk)

            all_chunks.extend(doc_chunks)

        return all_chunks
