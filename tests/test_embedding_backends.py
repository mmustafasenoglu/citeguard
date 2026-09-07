"""Tests for semantic embedding backends.

sentence-transformers is mocked so these tests stay offline.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from citeguard.similarity.embeddings import (
    SemanticBackendError,
    get_embedding_backend,
)
from citeguard.similarity.embeddings.base import EmbeddingBackend
from citeguard.similarity.models import SimilarityConfig

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_embedding_backend_protocol_structural() -> None:
    """EmbeddingBackend protocol has required attributes."""

    class DummyBackend:
        model_name = "test"
        dimension = 128

        def encode(self, texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), 128), dtype=np.float32)

    backend = DummyBackend()
    assert isinstance(backend, EmbeddingBackend)


# ---------------------------------------------------------------------------
# get_embedding_backend
# ---------------------------------------------------------------------------


def test_get_backend_returns_none_when_disabled() -> None:
    config = SimilarityConfig(enable_semantic=False)
    assert get_embedding_backend(config) is None


def test_get_backend_raises_when_dependency_missing() -> None:
    config = SimilarityConfig(enable_semantic=True)
    saved = sys.modules.pop(
        "citeguard.similarity.embeddings.sentence_transformers", None,
    )
    try:
        sys.modules["sentence_transformers"] = None
        with pytest.raises(SemanticBackendError, match="sentence-transformers"):
            get_embedding_backend(config)
    finally:
        if saved is not None:
            sys.modules["citeguard.similarity.embeddings.sentence_transformers"] = saved
        sys.modules.pop("sentence_transformers", None)


def test_get_backend_returns_st_backend_when_available() -> None:
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_st.SentenceTransformer.return_value = mock_model

    config = SimilarityConfig(enable_semantic=True, embedding_model="test-model")
    with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
        backend = get_embedding_backend(config)
        assert backend is not None
        assert backend.model_name == "test-model"
        assert backend.dimension == 384


# ---------------------------------------------------------------------------
# SentenceTransformerBackend (mocked)
# ---------------------------------------------------------------------------


def _make_mock_st_backend(dimension: int = 384):
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dimension
    mock_model.encode.return_value = np.random.rand(3, dimension).astype(np.float32)
    mock_st.SentenceTransformer.return_value = mock_model

    with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
        from citeguard.similarity.embeddings.sentence_transformers import (
            SentenceTransformerBackend,
        )

        return SentenceTransformerBackend(model_name="test-model"), mock_model


def test_st_backend_encode_returns_correct_shape() -> None:
    backend, _ = _make_mock_st_backend(dimension=128)
    result = backend.encode(["hello", "world", "test"])
    assert result.shape == (3, 128)
    assert result.dtype == np.float32


def test_st_backend_encode_empty_list() -> None:
    backend, _ = _make_mock_st_backend()
    result = backend.encode([])
    assert result.shape == (0, 384)


def test_st_backend_model_name() -> None:
    backend, _ = _make_mock_st_backend()
    assert backend.model_name == "test-model"


def test_st_backend_dimension() -> None:
    backend, _ = _make_mock_st_backend(dimension=256)
    assert backend.dimension == 256


# ---------------------------------------------------------------------------
# Offline mode (allow_download=False)
# ---------------------------------------------------------------------------


def test_st_backend_raises_when_download_blocked() -> None:
    mock_st = MagicMock()
    mock_st.SentenceTransformer.side_effect = OSError("no network")

    config = SimilarityConfig(
        enable_semantic=True, allow_model_download=False, embedding_model="test-model",
    )
    with patch.dict("sys.modules", {"sentence_transformers": mock_st}), pytest.raises(
        (SemanticBackendError, OSError),
    ):
        get_embedding_backend(config)


# ---------------------------------------------------------------------------
# SemanticBackendError
# ---------------------------------------------------------------------------


def test_semantic_backend_error_is_exception() -> None:
    with pytest.raises(SemanticBackendError, match="install"):
        raise SemanticBackendError("install citeguard[semantic]")
