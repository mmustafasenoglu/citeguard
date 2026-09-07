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
    """Return True if the model is already in the local HuggingFace cache.

    This checks the filesystem only — no ``SentenceTransformer()``
    call, no HTTP, no download trigger.  The contract is:
    ``allow_model_download=False`` → zero network attempts.
    """
    from pathlib import Path

    try:
        # Default HF cache: ~/.cache/huggingface/hub
        cache_home = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache_home.exists():
            return False
        # HF cache structure: models--{org}--{model_name}/snapshots/
        sanitized = model_name.replace("/", "--")
        repo_path = cache_home / f"models--{sanitized}"
        if not repo_path.exists():
            return False
        snapshots = repo_path / "snapshots"
        if not snapshots.exists():
            return False
        # At least one snapshot dir with weight files
        for snap in snapshots.iterdir():
            if snap.is_dir():
                for f in snap.iterdir():
                    if f.suffix in (".bin", ".safetensors", ".pt"):
                        return True
        return False
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
