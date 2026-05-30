"""Tests for speaker-embedding response parsing across backend shapes.

Three response shapes must be supported so the endpoint can be swapped between:
- Legacy GPUFARM omi-diarizer container (pyannote wespeaker, 512-d) — returns
  either a bare JSON array or `{"embedding": [...]}`.
- rtx6000 voice-extras (TitaNet-Large, 192-d) — returns OpenAI's embeddings
  shape: `{"data": [{"embedding": [...]}], "model": ..., "usage": ...}`.

The shared helper `_parse_embedding_response` normalises all three into a
numpy array of shape (1, D).
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock GCP client init at import time (same pattern as test_short_audio_embedding.py)
sys.modules.setdefault("database._client", MagicMock())

from utils.stt.speaker_embedding import _parse_embedding_response


class TestParseEmbeddingResponse:
    def test_bare_list_shape(self):
        """Legacy diarizer sometimes returns a bare JSON array."""
        payload = [0.1, 0.2, 0.3, 0.4]
        out = _parse_embedding_response(payload)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.shape == (1, 4)
        assert out.flatten().tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_legacy_embedding_key_shape(self):
        """Legacy diarizer's other shape: `{embedding: [...]}`."""
        payload = {"embedding": [0.5, -0.5, 0.0]}
        out = _parse_embedding_response(payload)
        assert out.shape == (1, 3)
        assert out.flatten().tolist() == pytest.approx([0.5, -0.5, 0.0])

    def test_openai_shape(self):
        """voice-extras returns OpenAI's embeddings response shape."""
        payload = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}],
            "model": "titanet-large",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
        out = _parse_embedding_response(payload)
        assert out.shape == (1, 6)
        assert out.dtype == np.float32
        assert out.flatten().tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_titanet_dimension(self):
        """TitaNet output is 192-d. The parser must not assume any specific dim."""
        payload = {"data": [{"embedding": [0.0] * 192}]}
        out = _parse_embedding_response(payload)
        assert out.shape == (1, 192)

    def test_wespeaker_dimension(self):
        """Wespeaker output is 512-d. Same parser path must handle it."""
        payload = {"embedding": [0.0] * 512}
        out = _parse_embedding_response(payload)
        assert out.shape == (1, 512)

    def test_empty_openai_data_raises(self):
        """OpenAI shape with empty data is a server error — surface it clearly."""
        payload = {"object": "list", "data": [], "model": "titanet-large"}
        with pytest.raises(ValueError, match="empty"):
            _parse_embedding_response(payload)

    def test_unknown_shape_raises(self):
        """Anything else is a contract violation."""
        with pytest.raises(ValueError, match="unrecognised"):
            _parse_embedding_response({"foo": "bar"})

    def test_none_payload_raises(self):
        with pytest.raises(ValueError):
            _parse_embedding_response(None)
