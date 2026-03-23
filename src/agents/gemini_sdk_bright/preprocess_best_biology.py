import sys
import re
import pathlib
from typing import List, Tuple
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from schema import Document, Chunk
from base import BasePreprocessor

# Base list of English stopwords from the original implementation
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
        "Creates a dense 'meta' chunk per doc plus paragraph chunks, all augmented with concepts, synonyms, and questions."
    )

    # Chunking parameters
    MIN_PARAGRAPH_WORDS = 20

    # Augmentation parameters
    DOC_LEVEL_CONCEPTS = 15
    CHUNK_LEVEL_CONCEPTS = 8
    MAX_TOTAL_CONCEPTS = 25
    MAX_QUESTIONS_PER_CHUNK = 10
    MAX_DEFINITIONAL_QUESTIONS = 7
    MAX_SYNONYMS = 5
    
    # Stopwords for filtering out low-value extracted concepts
    WEAK_CONCEPT_STOPWORDS = {
        'term', 'name', 'phenomenon', 'process', 'effect', 'study', 'result', 'example',
        'information', 'summary', 'concept', 'idea', 'way', 'part', 'list', 'type', 'form',
        'question', 'answer', 'paper', 'article', 'author', 'research', 'analysis', 'role',
        'function', 'method', 'system', 'approach', 'number', 'level', 'quality', 'source',
        'kind', 'factor', 'aspect', 'field', 'area', 'topic', 'model', 'group', 'case',
        'data', 'evidence', 'impact', 'feature', 'property', 'characteristic', 'component',
        'element', 'structure', 'condition', 'context', 'problem', 'solution', 'change',
        'increase', 'decrease', 'use', 'value', 'rate', 'theory', 'hypothesis', 'principle', 
        'mechanism', 'application', 'benefit', 'disadvantage', 'limitation', 'implication', 
        'importance', 'advantage', 'relationship', 'difference', 'similarity', 'variety',
        'range', 'collection', 'set', 'work', 'studies', 'study', 'results', 'examples',
        'others', 'time', 'people', 'many'
    }

    def __init__(self):
        super().__init__()
        self.nltk = None
        self.wordnet = None
        self.stopwords = BASE_STOPWORDS
        try:
            import nltk
            self.nltk = nltk
            for pkg, path in [
                ('punkt', 'tokenizers/punkt'),
                ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
                ('stopwords', 'corpora/stopwords'),
                ('wordnet', 'corpora/wordnet'),
                ('omw-1.4', 'corpora/omw-1.4'),
            ]:
                try:
                    self.nltk.data.find(path)
                except LookupError:
                    self.nltk.download(pkg, quiet=True)

            from nltk.corpus import stopwords as nltk_stopwords
            from nltk.corpus import wordnet
            self.stopwords = BASE_STOPWORDS.union(set(nltk_stopwords.words('english')))
            self.wordnet = wordnet

        except ImportError:
            print("Warning: NLTK not found. Augmentation features will be disabled.", file=sys.stderr)

    def _get_concepts_from_doc_id(self, doc_id: str) -> List[str]:
        """Extracts key concepts from the structured doc_id (filepath)."""
        parts = doc_id.split('/')
        if len(parts) < 2:
            return []
        
        filename = parts[-1]
        file_concept = re.sub(r'(_\d+)?\.txt$', '', filename).replace('_', ' ').strip()
        topic_concept = parts[-2].replace('_', ' ').strip()

        concepts = []
        if file_concept and not file_concept.isdigit():
            concepts.append(file_concept)
        if topic_concept and topic_concept.lower() != file_concept.lower():
            concepts.append(topic_concept)
        return concepts

    def _extract_title_and_content(self, text: str) -> Tuple[str, str]:
        """Extracts the first non-empty line as a title if it's short and not a full sentence."""
        lines = text.strip().split('\n')
        title = ""
        content_start_index = 0
        if lines:
            first_line = lines[0].strip()
            # A good title is relatively short and doesn't look like a full sentence.
            if 0 < len(first_line.split()) < 20 and not first_line.endswith(('.', '?', '!')):
                title = first_line
                content_start_index = 1
        content = '\n'.join(lines[content_start_index:])
        return title, content

    def _generate_diverse_questions(self, concepts: List[str]) -> List[str]:
        """Generates targeted questions from a prioritized list of concepts."""
        if not concepts:
            return []

        questions = set()
        primary_concept = concepts[0]
        context_concept = concepts[1] if len(concepts) > 1 else None
        
        # Definitional
        questions.add(f"What is {primary_concept}?")
        # Functional / Causal
        questions.add(f"How does {primary_concept} work?")
        questions.add(f"What are the effects of {primary_concept}?")
        # Contextual (link specific term to broader topic)
        if context_concept and context_concept.lower() not in primary_concept.lower():
            questions.add(f"How does {primary_concept} relate to {context_concept}?")
            questions.add(f"What is the role of {primary_concept} in {context_concept}?")
            questions.add(f"How is {primary_concept} an explanation for {context_concept}?")
        # Relational
        if len(concepts) > 2:
            questions.add(f"What is the difference between {primary_concept} and {concepts[2]}?")
        
        return sorted(list(questions))[:self.MAX_QUESTIONS_PER_CHUNK]

    def _generate_definition_questions(self, text: str) -> List[str]:
        """Generates 'what is' and 'what is the term for' questions from definition sentences."""
        if not self.nltk: return []
        questions = set()
        try:
            sents = self.nltk.sent_tokenize(text)
        except Exception:
            sents = text.split('.') # Fallback
        
        patterns = [
            # "Term is a/an/the definition." (e.g., "A phosphene is the phenomenon of...")
            re.compile(r'^(?P<term>[\w\s-]{3,30})\s+(?:is|are)\s+(?:a|an|the)\s+(?P<definition>.+)\.?$', re.IGNORECASE),
            # "This phenomenon is called Term."
            re.compile(r'^(?:this|the)\s+(?:phenomenon|process|effect|term)\s+(?:is|are)\s+(?:called|known as)\s+(?P<term>.+)\.?$', re.IGNORECASE),
            # "Term (definition)."
            re.compile(r'^(?P<term>[\w\s-]{3,30})\s+\((?P<definition>[^)]{10,})\)\s*.*$', re.IGNORECASE),
        ]

        for sent in sents:
            sent = sent.strip()
            if not sent: continue

            for pattern in patterns:
                match = pattern.match(sent)
                if match:
                    d = match.groupdict()
                    term = re.sub(r'^(a|an|the)\s+', '', d.get('term', '').strip().rstrip('.'), flags=re.IGNORECASE).strip()
                    definition = d.get('definition', '').strip()
                    
                    if not term or not definition or len(term.split()) > 7 or term.lower() in self.stopwords or len(definition.split()) < 4:
                        continue
                    
                    questions.add(f"What is {term}?")
                    questions.add(f"What is the term for {definition}?")
                    break 

        glossary_pattern = re.compile(r'^(?P<term>[A-Z][\w\s-]{3,30}):\s+(?P<definition>.{10,})$', re.MULTILINE)
        for match in glossary_pattern.finditer(text):
            term, definition = match.group('term').strip(), match.group('definition').strip()
            questions.add(f"What is {term}?")
            questions.add(f"What is the term for {definition}?")

        return sorted(list(questions))[:self.MAX_DEFINITIONAL_QUESTIONS]

    def _extract_key_concepts(self, text: str, title: str, num_concepts: int) -> List[str]:
        """Extracts, scores, and filters salient noun phrases."""
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
                words_lower = [w for w in phrase.lower().split() if w not in self.stopwords]
                if not words_lower: continue
                cleaned_phrase = " ".join(words_lower)
                if len(cleaned_phrase) < 4: continue
                cleaned_phrases_list.append(cleaned_phrase)
                if cleaned_phrase not in phrase_map:
                    phrase_map[cleaned_phrase] = phrase

            if not cleaned_phrases_list: return []

            freq_counts = Counter(cleaned_phrases_list)
            phrase_scores = Counter()
            title_lower = title.lower()
            
            sents = self.nltk.sent_tokenize(text)
            first_sentence = sents[0].lower() if sents else ""
            
            for phrase, count in freq_counts.items():
                score = float(count)
                num_words = len(phrase.split())
                if num_words > 1: score *= 1.5 # Prioritize multi-word concepts
                
                if any(w.isupper() for w in phrase_map[phrase].split() if len(w) > 1): score *= 1.5
                if title_lower and phrase in title_lower: score *= 2.0
                if first_sentence and phrase in first_sentence: score *= 1.8
                phrase_scores[phrase] = score
            
            top_phrases_candidates = [p for p, _ in phrase_scores.most_common(num_concepts * 3)]
            top_phrases_candidates.sort(key=len, reverse=True)

            final_phrases, seen_phrases_lower = [], set()
            for p in top_phrases_candidates:
                if len(final_phrases) >= num_concepts: break
                p_lower = p.lower()
                p_words = p_lower.split()
                if p_lower in self.WEAK_CONCEPT_STOPWORDS or p_words[-1] in self.WEAK_CONCEPT_STOPWORDS or len(p_words) > 6:
                    continue
                if any(p_lower in seen_p for seen_p in seen_phrases_lower):
                    continue
                final_phrases.append(phrase_map.get(p, p))
                seen_phrases_lower.add(p_lower)

            final_phrases.sort(key=lambda x: phrase_scores.get(x.lower(), 0), reverse=True)
            return final_phrases
        except Exception:
            return []
    
    def _get_synonyms(self, phrase: str, num_synonyms: int) -> List[str]:
        if not self.wordnet or not phrase: return []
        synonyms = set()
        lemmas_to_try = [phrase.replace(' ', '_')] + [w for w in phrase.split() if w not in self.stopwords]
        for lemma in lemmas_to_try:
            if len(synonyms) >= num_synonyms: break
            for syn in self.wordnet.synsets(lemma.lower()):
                if len(synonyms) >= num_synonyms: break
                if syn.pos() not in ('n', 'v'): continue
                for lem in syn.lemmas():
                    syn_word = lem.name().replace('_', ' ')
                    if syn_word.lower() != phrase.lower() and syn_word.lower() != lemma.lower():
                        synonyms.add(syn_word)
                        if len(synonyms) >= num_synonyms: break
        return sorted(list(synonyms))[:num_synonyms]

    def _get_topic_sentence(self, text: str, title: str, concepts: List[str]) -> str:
        if not self.nltk or not text: return ""
        try:
            sents = self.nltk.sent_tokenize(text)
            if not sents: return ""
            if len(sents) <= 2: return sents[0]

            important_words = set(self.nltk.word_tokenize(title.lower()))
            important_words.update(word for c in concepts for word in self.nltk.word_tokenize(c.lower()))
            important_words -= self.stopwords

            if not important_words: return sents[0]
            best_sentence, max_score = sents[0], -1
            for i, sent in enumerate(sents):
                sent_words = set(w for w in self.nltk.word_tokenize(sent.lower()) if w not in self.stopwords)
                score = len(sent_words.intersection(important_words)) / (len(sent_words) + 1)
                score *= (1.0 - (0.5 * i / len(sents))) # Position bonus, less aggressive
                if score > max_score:
                    max_score, best_sentence = score, sent
            return best_sentence
        except Exception:
            return text.split('.')[0] if '.' in text else ""

    def _create_meta_chunk(self, doc: Document, title: str, doc_level_concepts: List[str]) -> Chunk:
        """Creates a dense, header-only chunk representing the entire document's topics."""
        synonyms = set()
        for concept in doc_level_concepts[:7]:
            for syn in self._get_synonyms(concept, 2):
                synonyms.add(syn)
        
        template_questions = self._generate_diverse_questions(doc_level_concepts)
        definition_questions = self._generate_definition_questions(doc.text)
        all_questions = sorted(list(set(definition_questions + template_questions)))
        
        summary_sentence = self._get_topic_sentence(doc.text, title, doc_level_concepts)

        header_parts = []
        if title: header_parts.append(f"Title: {title}")
        if summary_sentence: header_parts.append(f"Summary: {summary_sentence.strip()}")
        if doc_level_concepts: header_parts.append(f"Key Topics: {', '.join(doc_level_concepts)}")
        if synonyms: header_parts.append(f"Related Terms: {', '.join(sorted(list(synonyms))[:self.MAX_SYNONYMS+2])}")
        
        if all_questions:
            queries = [q.replace('?', '.').strip() for q in all_questions]
            queries = [q if q.endswith('.') else q + '.' for q in queries]
            header_parts.append(f"Potential Questions Answered: {' '.join(queries)}")

        meta_text = "\n".join(header_parts)
        
        return Chunk(
            chunk_id=f"{doc.doc_id}_meta",
            doc_id=doc.doc_id,
            text=meta_text.strip(),
            metadata={"chunk_type": "meta"},
        )

    def _create_chunk(self, doc_id: str, chunk_id_counter: int, chunk_content: str, title: str, doc_level_concepts: List[str]) -> Chunk:
        """Creates an augmented chunk from a text block (e.g., a paragraph)."""
        chunk_level_concepts = self._extract_key_concepts(chunk_content, title, num_concepts=self.CHUNK_LEVEL_CONCEPTS)
        
        combined_concepts = list(doc_level_concepts)
        seen_concepts = {c.lower() for c in doc_level_concepts}
        for concept in chunk_level_concepts:
            if concept.lower() not in seen_concepts:
                combined_concepts.append(concept)
                seen_concepts.add(concept.lower())
        final_concepts = combined_concepts[:self.MAX_TOTAL_CONCEPTS]
        
        synonyms = set()
        for concept in final_concepts[:5]:
            for syn in self._get_synonyms(concept, 1):
                synonyms.add(syn)
        
        template_questions = self._generate_diverse_questions(final_concepts)
        definition_questions = self._generate_definition_questions(chunk_content)
        all_questions = sorted(list(set(definition_questions + template_questions)))

        header_parts = []
        if title: header_parts.append(f"Title: {title}")
        summary_sentence = self._get_topic_sentence(chunk_content, title, final_concepts)
        if summary_sentence: header_parts.append(f"Summary: {summary_sentence.strip()}")
        if final_concepts: header_parts.append(f"Key Topics: {', '.join(final_concepts)}")
        if synonyms: header_parts.append(f"Related Terms: {', '.join(sorted(list(synonyms)))}")
        if all_questions:
            queries = [q.replace('?', '.').strip() for q in all_questions]
            queries = [q if q.endswith('.') else q + '.' for q in queries]
            header_parts.append(f"Potential Questions Answered: {' '.join(queries)}")

        meta_header = "\n".join(header_parts)
        chunk_text = f"{meta_header}\n\n---\n\n{chunk_content}".strip()
        
        return Chunk(
            chunk_id=f"{doc_id}_{chunk_id_counter}",
            doc_id=doc_id, text=chunk_text, metadata={},
        )

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """
        Processes documents into two types of chunks:
        1. A single, dense 'meta' chunk per document with aggregated concepts and questions.
        2. Multiple paragraph-based chunks, each augmented with its own header.
        """
        all_chunks = []
        for doc in docs:
            doc_text = doc.text or ""
            if not doc_text.strip(): continue
            
            title, content = self._extract_title_and_content(doc_text)
            
            # 1. Perform holistic document analysis for concepts
            doc_id_concepts = self._get_concepts_from_doc_id(doc.doc_id)
            text_based_concepts = self._extract_key_concepts(f"{title}. {content}", title, num_concepts=self.DOC_LEVEL_CONCEPTS)
            
            doc_level_concepts = list(doc_id_concepts)
            seen_concepts = {c.lower() for c in doc_id_concepts}
            for concept in text_based_concepts:
                if concept.lower() not in seen_concepts:
                    doc_level_concepts.append(concept)
                    seen_concepts.add(concept.lower())
            
            # 2. Create and add the dense 'meta' chunk for the whole document
            meta_chunk = self._create_meta_chunk(doc, title, doc_level_concepts)
            all_chunks.append(meta_chunk)

            # 3. Create paragraph-based chunks
            doc_para_chunks = []
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
            paragraphs = [p for p in paragraphs if len(p.split()) >= self.MIN_PARAGRAPH_WORDS]

            for i, para in enumerate(paragraphs):
                chunk = self._create_chunk(doc.doc_id, i, para, title, doc_level_concepts)
                doc_para_chunks.append(chunk)
            
            if not doc_para_chunks and content.strip():
                # Fallback: if no paragraphs, create one chunk for the whole content
                chunk = self._create_chunk(doc.doc_id, 0, content.strip(), title, doc_level_concepts)
                doc_para_chunks.append(chunk)

            all_chunks.extend(doc_para_chunks)

        return all_chunks
