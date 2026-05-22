"""Singleton SentenceTransformer reused across the application."""
import logging
from typing import List
from config.settings import get_settings

logger = logging.getLogger(__name__)

_model = None

def get_embedding_model():
    """Load the singleton model on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_name = get_settings().embedding_model
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")
    return _model

def embed(text: str) -> List[float]:
    """Create an embedding for text."""
    return get_embedding_model().encode(text).tolist()
