import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "evaluation"))
from typing import List
import re
from schema import Document, Chunk
from base import BasePreprocessor

class Preprocessor(BasePreprocessor):
    name = "scene_based_extraction"
    description = "Extracts scenes dynamically based on transition keywords for granular chunking."

    def preprocess(self, docs: List[Document]) -> List[Chunk]:
        def split_into_scenes(text: str) -> List[str]:
            scenes = re.split(r'\b(later|in one scene|another scene|eventually|finally)\b', text, flags=re.IGNORECASE)
            return [scene.strip() for scene in scenes if len(scene.strip()) > 50]

        chunks = []
        for doc in docs:
            scene_texts = split_into_scenes(doc.text)
            for i, scene_text in enumerate(scene_texts):
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_{i}",
                    doc_id=doc.doc_id,
                    text=scene_text,
                )
                chunks.append(chunk)
        return chunks
