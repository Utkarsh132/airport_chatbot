"""
Thin inference wrapper used by the Streamlit app.
Keeps app.py free of business logic -- imports from src/.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from kb_retriever import KnowledgeBaseRetriever  # noqa: E402
from fusion import MultimodalFusionEngine  # noqa: E402

_engine = None


def get_engine() -> MultimodalFusionEngine:
    global _engine
    if _engine is None:
        retriever = KnowledgeBaseRetriever()
        _engine = MultimodalFusionEngine(retriever)
    return _engine


def run_inference(text=None, image=None, audio_path=None) -> dict:
    engine = get_engine()
    return engine.respond(text=text, image=image, audio_path=audio_path)
