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
        "Paragraph-based chunking with aggressive augmentation (Questions, Keywords, Summary) to bridge vocabulary gaps."
    )

    # Chunking parameters adjusted for more context and focus
    PARAGRAPH_WORD_LIMIT = 180
    CHUNK_SIZE_SENTENCES = 5
    CHUNK_OVERLAP_SENTENCES = 2
    
    # Augmentation parameters
    NUM_KEYPHRASES = 5
    MAX_QUESTIONS_PER_CHUNK = 3

    def __init__(self):
        super().__init__()
        self.nltk = None
        self.stopwords = BASE_STOPWORDS
        try:
            import nltk
            self.nltk = nltk
            # Ensure required NLTK packages are downloaded
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
            # A potential title is non-empty, short, and doesn't look like a paragraph.
            if 0 < len(first_line.split()) < 20 and not first_line.endswith(('.', '?', '!')):
                title = first_line
                content_start_index = 1
        content = '\n'.join(lines[content_start_index:])
        return title, content

    def _generate_diverse_questions(self, concepts: List[str], title: str) -> List[str]:
        """Generates a diverse set of hypothetical questions from key concepts and title."""
        if not concepts:
            return []

        questions = set()
        primary_concept = concepts[0]

        # Basic templates for the primary concept
        questions.add(f"What is {primary_concept}?")
        if len(primary_concept.split()) > 1:
            questions.add(f"How does {primary_concept} work?")
        if title:
            questions.add(f"What is the role of {primary_concept} in {title.lower()}?")

        # Templates for relationships between concepts
        if len(concepts) > 1:
            secondary_concept = concepts[1]
            questions.add(f"What is the relationship between {primary_concept} and {secondary_concept}?")
            questions.add(f"How does {primary_concept} affect {secondary_concept}?")
        
        return sorted(list(questions))[:self.MAX_QUESTIONS_PER_CHUNK]

    def _extract_key_concepts(self, text: str, title: str) -> List[str]:
        """Extracts and scores salient noun phrases using frequency and heuristics."""
        if not self.nltk: return []
        try:
            words = self.nltk.word_tokenize(text)
            if not words: return []

            tagged = self.nltk.pos_tag(words)
            # Grammar for noun phrases: (optional det/possessive), (0+ adj), (1+ noun)
            grammar = r"NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}"
            parser = self.nltk.RegexpParser(grammar)
            tree = parser.parse(tagged)

            # Extract raw phrases from the text
            raw_phrases = []
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                raw_phrases.append(" ".join(word for word, tag in subtree.leaves()))
            
            if not raw_phrases: return []

            # Clean and map phrases to their original casing
            phrase_map = {} # Maps cleaned phrase to original cased phrase
            cleaned_phrases_list = []
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

            # Score phrases based on frequency and heuristics
            freq_counts = Counter(cleaned_phrases_list)
            phrase_scores = Counter()
            title_lower = title.lower()

            for phrase, count in freq_counts.items():
                score = float(count)
                original_phrase = phrase_map[phrase]
                
                if any(w.isupper() for w in original_phrase.split() if len(w) > 1):
                    score *= 1.5  # Boost for capitalization
                if phrase in title_lower:
                    score *= 2.0  # Boost for being in the title
                if len(phrase.split()) > 1:
                    score *= 1.2  # Boost for being multi-word
                
                phrase_scores[phrase] = score
            
            top_phrases_cleaned = [p for p, _ in phrase_scores.most_common(self.NUM_KEYPHRASES)]
            return [phrase_map[p] for p in top_phrases_cleaned]

        except Exception:
            return []

    def _create_chunk(self, doc_id: str, chunk_id_counter: int, chunk_content: str, title: str) -> Chunk:
        """Creates an augmented chunk with a summary, keywords, and hypothetical questions."""
        summary = ""
        if self.nltk:
            try:
                sentences = self.nltk.sent_tokenize(chunk_content)
                if sentences:
                    summary = sentences[0]
            except Exception:
                pass

        concepts = self._extract_key_concepts(chunk_content, title)
        questions = self._generate_diverse_questions(concepts, title)

        augmented_parts = []
        if title:
            augmented_parts.append(f"Title: {title}.")
        if questions:
            augmented_parts.append(f"Potential Questions: {' '.join(questions)}")
        if concepts:
            augmented_parts.append(f"Keywords: {', '.join(concepts)}.")
        if summary and summary.strip() != chunk_content.strip():
             augmented_parts.append(f"Summary: {summary}")
        
        augmentation_header = "\n".join(filter(None, augmented_parts))
        chunk_text = f"{augmentation_header}\n\n{chunk_content}".strip()
        
        return Chunk(
            chunk_id=f"{doc_id}_{chunk_id_counter}",
            doc_id=doc_id,
            text=chunk_text,
            metadata={},
        )

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        """Splits documents into paragraph-based chunks and augments them."""
        all_chunks = []
        for doc in docs:
            doc_text = doc.text or ""
            if not doc_text.strip():
                continue
            
            title, content = self._extract_title_and_content(doc_text)
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
                        doc_chunks.append(self._create_chunk(doc.doc_id, chunk_idx_counter, chunk_content, title))
                        chunk_idx_counter += 1
                else:
                    doc_chunks.append(self._create_chunk(doc.doc_id, chunk_idx_counter, para_text, title))
                    chunk_idx_counter += 1

            if not doc_chunks and doc_text.strip():
                full_text_for_chunking = f"{title}\n{content}".strip()
                if full_text_for_chunking:
                     doc_chunks.append(self._create_chunk(doc.doc_id, 0, full_text_for_chunking, title))

            all_chunks.extend(doc_chunks)

        return all_chunks
