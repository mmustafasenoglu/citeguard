"""Embedding backend protocol for similarity retrieval.

Defines the abstract interface that all embedding backends must implement.
The backend is pluggable: the engine never imports a specific backend
library directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for embedding backends.

    Any backend must expose a fixed ``model_name``, ``dimension``, and
    batch ``encode`` method.  The engine uses only these attributes.
    """

    model_name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into dense vectors.

        Parameters
        ----------
        texts:
            List of strings to encode.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(len(texts), dimension)``.
        """
        ...
