"""Text embedding service using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

_model_cache: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Get cached embedding model (lazy loading)."""
    global _model_cache
    if _model_cache is None:
        logger.info(f"Loading embedding model: {model_name}")
        _model_cache = SentenceTransformer(model_name)
        logger.info(f"Model loaded with dimension: {_model_cache.get_sentence_embedding_dimension()}")
    return _model_cache


def encode_text(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Encode a single text to embedding vector."""
    model = get_embedding_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return normalize_vector(embedding)


def encode_batch(
    texts: List[str],
    batch_size: int = 32,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """Encode multiple texts efficiently with batching."""
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)

    # Normalize all vectors
    normalized = np.array([normalize_vector(emb) for emb in embeddings])
    return normalized


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2 normalize vector for cosine similarity."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
