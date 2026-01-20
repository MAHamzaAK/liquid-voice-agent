"""
Embeddings Module - BGE model for semantic matching

Phase II: Generates embeddings for transcripts, situations, and principles.
Uses BAAI/bge-small-en-v1.5 (384 dimensions) for good quality/speed balance.
"""

from typing import Union
import numpy as np

# Lazy loading to avoid import overhead when not needed
_model = None
_tokenizer = None

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


def _load_model():
    """Lazy load the embedding model."""
    global _model, _tokenizer
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: The text to embed

    Returns:
        List of floats (384 dimensions)
    """
    model = _load_model()
    # BGE models recommend prepending "query: " for queries
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts (batch processing).

    Args:
        texts: List of texts to embed

    Returns:
        List of embeddings (each 384 dimensions)
    """
    model = _load_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Generate embedding for a query (prepends "query: " for BGE models).

    Args:
        query: The query text (e.g., transcript)

    Returns:
        List of floats (384 dimensions)
    """
    model = _load_model()
    # BGE recommends prepending instruction for queries
    embedding = model.encode(f"query: {query}", normalize_embeddings=True)
    return embedding.tolist()


def embed_document(document: str) -> list[float]:
    """
    Generate embedding for a document (no prefix for BGE models).

    Args:
        document: The document text (e.g., situation signal, principle definition)

    Returns:
        List of floats (384 dimensions)
    """
    model = _load_model()
    embedding = model.encode(document, normalize_embeddings=True)
    return embedding.tolist()


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Similarity score between -1 and 1
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
