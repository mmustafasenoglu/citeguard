"""SentenceTransformer embedding backend.

Optional dependency — requires ``pip install citeguard[semantic]``
(or manually ``pip install sentence-transformers``).

The backend lazily imports ``sentence_transformers`` so that the core
package remains lightweight.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _is_model_cached(model_name: str) -> bool:
    """Return True if the model is already in the local HuggingFace cache."""
    try:
        from sentence_transformers import SentenceTransformer

        cache_dir = SentenceTransformer._model_card_variables.get(
            "cache_dir", None,
        )
        # Quick heuristic: try to load from cache without downloading
        SentenceTransformer(model_name, cache_folder=cache_dir)
        return True
    except Exception:
        return False


class SentenceTransformerBackend:
    """Local SentenceTransformer wrapper.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
        Default: ``paraphrase-multilingual-MiniLM-L12-v2``.
    allow_download:
        When *False*, raise an error if the model is not already cached.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        allow_download: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            from . import SemanticBackendError

            raise SemanticBackendError(
                "Semantic similarity requires the sentence-transformers package. "
                "Install it with: pip install citeguard[semantic]"
            ) from None

        if not allow_download and not _is_model_cached(model_name):
            from . import SemanticBackendError

            raise SemanticBackendError(
                f"Model '{model_name}' not in local cache. "
                f"Run once without --offline to download it, or install "
                f"the model manually into the HuggingFace cache."
            )

        self._model: SentenceTransformer = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Loaded SentenceTransformer model '%s' (dim=%d)",
            model_name,
            self._dimension,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into dense vectors.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), dimension)``.
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
