"""Embedding backends for semantic similarity.

Public API:
- ``SemanticBackendError`` — raised when semantic is requested but unavailable
- ``get_embedding_backend()`` — factory that returns the appropriate backend
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citeguard.similarity.models import SimilarityConfig

from .base import EmbeddingBackend


class SemanticBackendError(Exception):
    """Raised when semantic similarity is requested but the backend is unavailable.

    This can happen when:
    - ``sentence-transformers`` is not installed (``pip install citeguard[semantic]``)
    - The requested model is not in the local cache and ``--offline`` is active
    """

    pass


def get_embedding_backend(
    config: SimilarityConfig,
) -> EmbeddingBackend | None:
    """Return the embedding backend for the given configuration.

    Returns
    -------
    EmbeddingBackend or None
        - ``None`` when ``config.enable_semantic`` is ``False``
        - ``SentenceTransformerBackend`` when available
        - Raises ``SemanticBackendError`` when semantic is enabled but
          the backend cannot be loaded
    """
    if not config.enable_semantic:
        return None

    try:
        from .sentence_transformers import SentenceTransformerBackend
    except ImportError:
        raise SemanticBackendError(
            "Semantic similarity requires the sentence-transformers package. "
            "Install it with: pip install citeguard[semantic]"
        ) from None

    return SentenceTransformerBackend(
        model_name=config.embedding_model,
        allow_download=config.allow_model_download,
    )


__all__ = [
    "EmbeddingBackend",
    "SemanticBackendError",
    "get_embedding_backend",
]
