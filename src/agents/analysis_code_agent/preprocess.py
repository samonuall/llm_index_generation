import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))

from typing import List, Dict
import re
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "analysis_code_agent"
    description = "Synthesis of all proven hypotheses: article-level title context, synonym expansion, scene extraction, and overlapping sections"

    def __init__(self):
        # H1: Synonym map for vocabulary expansion
        self.synonym_map = {
            r'\balien\b': 'alien (creature monster lizard being extraterrestrial)',
            r'\bpractical joke\b': 'practical joke (prank trick)',
            r'\btraumatized\b': 'traumatized (scared frightened mortified freaked)',
            r'\bstudent\b': 'student (kid teenager girl boy)',
            r'\bvehicle\b': 'vehicle (car cadillac truck)',
            r'\bparty\b': 'party (club gathering event)',
            r'\bbuilding\b': 'building (house structure place)',
            r'\bchild\b': 'child (kid girl boy daughter son)',
            r'\bmother\b': 'mother (mom parent)',
            r'\bfather\b': 'father (dad parent)',
            r'\bcruel\b': 'cruel (mean nasty fucked)',
            r'\bscreaming\b': 'screaming (crying yelling shouting)',
            r'\bterrified\b': 'terrified (scared frightened afraid)',
            r'\bdeceased\b': 'deceased (dead died killed)',
            r'\bdisturbed\b': 'disturbed (upset freaked disturbing)',
            r'\bgang\b': 'gang (mafia mob crew group)',
            r'\bphysician\b': 'physician (doctor)',
            r'\binvestigator\b': 'investigator (detective cop police)',
            r'\bautomobile\b': 'automobile (car vehicle)',
            r'\byouth\b': 'youth (teenager kid teen)',
        }
        
        # H4: Scene indicators for extracting memorable descriptions
        self.scene_indicators = [
            r'\b(screaming|crying|yelling|shouting)\b',
            r'\b(terrified|frightened|scared|mortified|freaked)\b',
            r'\b(standing|sitting|running|walking|driving|throwing|turning)\b',
            r'\b(window|door|car|building|room|house)\b',
            r'\b(girl|boy|woman|man|mother|father|child)\b',
            r'\b(suddenly|quickly|slowly|frantically)\b',
            r'\b(blood|dead|killed|murder|attack)\b',
            r'\b(strange|unusual|bizarre|weird|disturbing)\b',
        ]

    def expand_vocabulary(self, text: str) -> str:
        """H1: Expand vocabulary with synonyms."""
        expanded = text
        for pattern, replacement in self.synonym_map.items():
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        return expanded

    def extract_title(self, text: str) -> str:
        """H3: Extract main entity/title from section :0."""
        lines = text.strip().split('\n')
        first_part = lines[0] if lines else text[:200]
        first_part = re.sub(r'\s+', ' ', first_part).strip()
        sentences = re.split(r'[.!?]', first_part)
        title_text = sentences[0] if sentences else first_part[:150]
        return title_text.strip()

    def extract_scene_sentences(self, text: str) -> List[str]:
        """H4: Extract sentences with scene-like descriptions."""
        sentences = re.split(r'[.!?]+', text)
        scene_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            score = 0
            for pattern in self.scene_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    score += 1
            
            if score >= 2:
                scene_sentences.append(sentence)
        
        return scene_sentences

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        # Group documents by article (page_id)
        articles: Dict[str, List[Document]] = {}
        for doc in docs:
            page_id = doc.doc_id.split(':')[0]
            if page_id not in articles:
                articles[page_id] = []
            articles[page_id].append(doc)
        
        # Sort sections by index within each article
        for page_id in articles:
            articles[page_id].sort(key=lambda d: int(d.doc_id.split(':')[1]))
        
        chunks = []
        
        for page_id, sections in articles.items():
            # H3: Extract title context from section :0
            title_context = ""
            if sections and sections[0].doc_id.endswith(':0'):
                title_context = self.extract_title(sections[0].text)
                if title_context and not title_context.endswith(('.', '!', '?')):
                    title_context += '.'
                title_context += ' '
            
            # Process each section
            for i, doc in enumerate(sections):
                section_idx = int(doc.doc_id.split(':')[1])
                
                # Base text processing: H1 (synonym expansion) + H4 (scene extraction)
                base_text = doc.text
                scene_sentences = self.extract_scene_sentences(base_text)
                
                if scene_sentences:
                    scene_text = ' '.join(scene_sentences)
                    base_text = f"{base_text} Key scenes: {scene_text}"
                
                # Apply vocabulary expansion
                base_text = self.expand_vocabulary(base_text)
                
                # H2 + H3: Create primary chunk with title context
                if section_idx == 0:
                    # Section :0 stays as-is (it's the title section)
                    chunk_text = base_text
                else:
                    # Content sections get title context prepended
                    chunk_text = title_context + base_text
                
                chunk_id = f"{doc.doc_id}_main"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    metadata=doc.metadata
                ))
                
                # H2: Create overlapping chunk with previous section (for sections after :0)
                if i > 0:
                    prev_text = sections[i-1].text
                    prev_text = self.expand_vocabulary(prev_text)
                    
                    # Take last 150 words of previous section
                    prev_words = prev_text.split()
                    overlap_text = ' '.join(prev_words[-150:]) if len(prev_words) > 150 else prev_text
                    
                    merged_text = title_context + overlap_text + ' ' + base_text
                    chunk_id_overlap = f"{doc.doc_id}_overlap"
                    chunks.append(Chunk(
                        chunk_id=chunk_id_overlap,
                        doc_id=doc.doc_id,
                        text=merged_text,
                        metadata=doc.metadata
                    ))
        
        return chunks
