"""Unit tests for the local Whisper batch transcription path in pre_recorded.py.

Phase A of the Deepgram → local Whisper swap. The new helpers produce the
same word-dict shape Deepgram has always returned so postprocess_conversation
and friends keep working:

    [{'timestamp': [start, end], 'speaker': 'SPEAKER_XX', 'text': 'word'}, ...]

Coverage:
  - _merge_words_with_speakers: Whisper words + Sortformer segments -> labelled words
  - _wav_wrap_pcm: wraps raw PCM bytes in a WAV header so Whisper can decode them
  - local_whisper_prerecorded_from_bytes: HTTP-mocked end-to-end shape parity with
    deepgram_prerecorded_from_bytes
  - STT_BATCH_BACKEND routing: the public deepgram_prerecorded_from_bytes function
    delegates to the local Whisper path when the env selector is "whisper"
"""

import importlib
import importlib.util
import os
import sys
import types
import wave
from unittest.mock import MagicMock, patch

import httpx
import pytest

os.environ.setdefault(
    "ENCRYPTION_SECRET",
    "omi_ZwB2ZNqB2HHpMK6wStk7sTpavJiPTFg7gXUHnc4tFABPU6pZ2c2DKgehtfgi4RZv",
)

# Heavy deps that pre_recorded.py touches at import time but aren't used by the
# unit-tested code paths. fal_client and deepgram both initialise SDK clients at
# import time, which would otherwise require live credentials.
sys.modules.setdefault("fal_client", MagicMock())
sys.modules.setdefault("deepgram", MagicMock(DeepgramClient=MagicMock(), DeepgramClientOptions=MagicMock()))
# database/_client.py constructs a Firestore singleton at module load, which
# requires GCP credentials. Pre-stub it so transitive imports get the mock.
_db_client_stub = types.ModuleType("database._client")
_db_client_stub.db = MagicMock()
_db_client_stub.document_id_from_seed = lambda *a, **k: "fake-doc-id"
sys.modules["database._client"] = _db_client_stub


@pytest.fixture(scope="module")
def pre_recorded():
    from utils.stt import pre_recorded as mod

    importlib.reload(mod)  # pick up any env we set during the test session
    return mod


# ---------------------------------------------------------------------------
# _merge_words_with_speakers
# ---------------------------------------------------------------------------


class TestMergeWordsWithSpeakers:
    def _w(self, start, end, word, prob=0.9):
        return {"start": start, "end": end, "word": word, "probability": prob}

    def _seg(self, start, end, speaker):
        return {"start": start, "end": end, "duration": end - start, "speaker": speaker}

    def test_no_diarization_segments_all_speaker_00(self, pre_recorded):
        words = [self._w(0.0, 0.5, " hello"), self._w(0.5, 1.0, " world")]
        out = pre_recorded._merge_words_with_speakers(words, [])
        assert len(out) == 2
        assert all(w["speaker"] == "SPEAKER_00" for w in out)
        assert out[0]["text"] == "hello"
        assert out[0]["timestamp"] == [0.0, 0.5]

    def test_single_speaker_segments_all_speaker_00(self, pre_recorded):
        words = [self._w(0.5, 1.0, " hi"), self._w(1.2, 1.6, " there")]
        segs = [self._seg(0.0, 2.0, "speaker_0")]
        out = pre_recorded._merge_words_with_speakers(words, segs)
        assert [w["speaker"] for w in out] == ["SPEAKER_00", "SPEAKER_00"]

    def test_two_speakers_assigned_by_midpoint(self, pre_recorded):
        words = [
            self._w(0.0, 0.4, " a"),  # midpoint 0.2 → speaker_0
            self._w(0.5, 0.9, " b"),  # midpoint 0.7 → speaker_0
            self._w(1.5, 1.9, " c"),  # midpoint 1.7 → speaker_1
        ]
        segs = [
            self._seg(0.0, 1.0, "speaker_0"),
            self._seg(1.0, 2.0, "speaker_1"),
        ]
        out = pre_recorded._merge_words_with_speakers(words, segs)
        assert [w["speaker"] for w in out] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]

    def test_speaker_format_two_digits_for_index_10(self, pre_recorded):
        """Sortformer can emit "speaker_10" for densely-mic'd audio; format must keep two digits."""
        words = [self._w(0.0, 0.5, " x")]
        segs = [self._seg(0.0, 1.0, "speaker_10")]
        out = pre_recorded._merge_words_with_speakers(words, segs)
        assert out[0]["speaker"] == "SPEAKER_10"

    def test_word_outside_any_segment_defaults_to_speaker_00(self, pre_recorded):
        """If Sortformer's coverage misses a word (gap between speaker turns),
        fall back to SPEAKER_00 rather than crashing."""
        words = [self._w(5.0, 5.5, " orphan")]
        segs = [self._seg(0.0, 1.0, "speaker_0"), self._seg(2.0, 3.0, "speaker_1")]
        out = pre_recorded._merge_words_with_speakers(words, segs)
        assert out[0]["speaker"] == "SPEAKER_00"

    def test_text_field_strips_leading_space(self, pre_recorded):
        """Whisper word fields look like ' data' (leading space). Output text must be stripped."""
        words = [self._w(0.0, 0.5, "  trailing  "), self._w(0.5, 1.0, "")]
        out = pre_recorded._merge_words_with_speakers(words, [])
        assert out[0]["text"] == "trailing"
        # Empty words shouldn't crash the merge.
        assert out[1]["text"] == ""


# ---------------------------------------------------------------------------
# _wav_wrap_pcm
# ---------------------------------------------------------------------------


class TestWavWrapPcm:
    def test_wraps_raw_pcm_into_parseable_wav(self, pre_recorded):
        pcm = b"\x00\x00" * 16000  # 1 second of silence at 16 kHz 16-bit mono
        wav_bytes = pre_recorded._wav_wrap_pcm(pcm, sample_rate=16000, channels=1)
        # Has the RIFF header
        assert wav_bytes[:4] == b"RIFF"
        # Parseable by stdlib wave
        import io as _io

        with wave.open(_io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 16000


# ---------------------------------------------------------------------------
# local_whisper_prerecorded_from_bytes (HTTP-mocked)
# ---------------------------------------------------------------------------


class TestLocalWhisperPrerecordedFromBytes:
    @patch("utils.stt.pre_recorded.httpx.post")
    def test_returns_deepgram_shape_word_dicts(self, mock_post, pre_recorded):
        # Two mocked POSTs: first Whisper transcription, then Sortformer diarization.
        whisper_resp = MagicMock(status_code=200)
        whisper_resp.json.return_value = {
            "text": "hi there",
            "language": "en",
            "duration": 1.5,
            "words": [
                {"start": 0.0, "end": 0.4, "word": " hi"},
                {"start": 0.5, "end": 1.0, "word": " there"},
            ],
            "segments": [],
        }
        sortformer_resp = MagicMock(status_code=200)
        sortformer_resp.json.return_value = {
            "model": "sortformer-stream",
            "segments": [{"start": 0.0, "end": 1.5, "duration": 1.5, "speaker": "speaker_0"}],
            "rttm": [],
        }
        mock_post.side_effect = [whisper_resp, sortformer_resp]

        wav = pre_recorded._wav_wrap_pcm(b"\x00\x00" * 16000, sample_rate=16000, channels=1)
        words = pre_recorded.local_whisper_prerecorded_from_bytes(wav, sample_rate=16000, diarize=True)

        assert len(words) == 2
        assert words[0] == {"timestamp": [0.0, 0.4], "speaker": "SPEAKER_00", "text": "hi"}
        assert words[1] == {"timestamp": [0.5, 1.0], "speaker": "SPEAKER_00", "text": "there"}

    @patch("utils.stt.pre_recorded.httpx.post")
    def test_diarize_false_skips_sortformer_call(self, mock_post, pre_recorded):
        whisper_resp = MagicMock(status_code=200)
        whisper_resp.json.return_value = {
            "text": "hi",
            "language": "en",
            "duration": 0.5,
            "words": [{"start": 0.0, "end": 0.4, "word": " hi"}],
            "segments": [],
        }
        mock_post.side_effect = [whisper_resp]

        wav = pre_recorded._wav_wrap_pcm(b"\x00\x00" * 8000, sample_rate=16000, channels=1)
        words = pre_recorded.local_whisper_prerecorded_from_bytes(wav, diarize=False)

        # Single POST: Whisper only, no Sortformer.
        assert mock_post.call_count == 1
        assert len(words) == 1
        assert words[0]["speaker"] == "SPEAKER_00"

    @patch("utils.stt.pre_recorded.httpx.post")
    def test_return_language_true_returns_tuple(self, mock_post, pre_recorded):
        whisper_resp = MagicMock(status_code=200)
        whisper_resp.json.return_value = {
            "text": "hola",
            "language": "es",
            "duration": 0.5,
            "words": [{"start": 0.0, "end": 0.4, "word": " hola"}],
            "segments": [],
        }
        mock_post.side_effect = [whisper_resp]

        wav = pre_recorded._wav_wrap_pcm(b"\x00\x00" * 8000, sample_rate=16000, channels=1)
        result = pre_recorded.local_whisper_prerecorded_from_bytes(wav, diarize=False, return_language=True)
        assert isinstance(result, tuple)
        words, lang = result
        assert lang == "es"
        assert words[0]["text"] == "hola"

    @patch("utils.stt.pre_recorded.httpx.post")
    def test_empty_whisper_response_returns_empty_list(self, mock_post, pre_recorded):
        whisper_resp = MagicMock(status_code=200)
        whisper_resp.json.return_value = {"text": "", "language": "en", "duration": 1.0, "words": [], "segments": []}
        mock_post.side_effect = [whisper_resp]

        wav = pre_recorded._wav_wrap_pcm(b"\x00\x00" * 16000, sample_rate=16000, channels=1)
        words = pre_recorded.local_whisper_prerecorded_from_bytes(wav, diarize=False)
        assert words == []

    @patch("utils.stt.pre_recorded.httpx.post")
    def test_raw_pcm_input_gets_wav_wrapped_before_post(self, mock_post, pre_recorded):
        """Callers pass raw PCM with encoding='linear16'. The Whisper path must
        wrap that in a WAV header so the upstream can decode it."""
        whisper_resp = MagicMock(status_code=200)
        whisper_resp.json.return_value = {
            "text": "",
            "language": "en",
            "duration": 1.0,
            "words": [],
            "segments": [],
        }
        mock_post.side_effect = [whisper_resp]

        pcm = b"\x00\x00" * 16000
        pre_recorded.local_whisper_prerecorded_from_bytes(
            pcm, encoding="linear16", sample_rate=16000, channels=1, diarize=False
        )
        # The bytes uploaded must start with RIFF (proper WAV).
        sent_files = mock_post.call_args.kwargs.get("files") or mock_post.call_args[1].get("files")
        assert sent_files is not None
        # files is {'file': (filename, bytes, mimetype)}
        uploaded_bytes = sent_files["file"][1]
        assert uploaded_bytes[:4] == b"RIFF"


# ---------------------------------------------------------------------------
# Routing: STT_BATCH_BACKEND="whisper" makes the public function delegate
# ---------------------------------------------------------------------------


class TestLocalWhisperPrerecordedFromUrl:
    @patch("utils.stt.pre_recorded.httpx.get")
    @patch("utils.stt.pre_recorded.local_whisper_prerecorded_from_bytes")
    def test_fetches_url_and_delegates(self, mock_bytes, mock_get, pre_recorded):
        fetched = MagicMock(status_code=200, content=b"\x00" * 1000)
        fetched.raise_for_status = MagicMock()
        mock_get.return_value = fetched
        mock_bytes.return_value = [{"timestamp": [0, 1], "speaker": "SPEAKER_00", "text": "ok"}]

        out = pre_recorded.local_whisper_prerecorded("https://example.com/audio.wav", diarize=True, language="en")
        assert mock_get.call_count == 1
        # downloaded bytes must be forwarded to the bytes helper
        assert mock_bytes.call_count == 1
        forwarded = mock_bytes.call_args.args[0]
        assert forwarded == b"\x00" * 1000
        assert out == [{"timestamp": [0, 1], "speaker": "SPEAKER_00", "text": "ok"}]

    @patch("utils.stt.pre_recorded.httpx.get")
    @patch("utils.stt.pre_recorded.local_whisper_prerecorded_from_bytes")
    def test_return_language_propagates_tuple(self, mock_bytes, mock_get, pre_recorded):
        fetched = MagicMock(status_code=200, content=b"\x00" * 1000)
        fetched.raise_for_status = MagicMock()
        mock_get.return_value = fetched
        mock_bytes.return_value = ([], "es")

        result = pre_recorded.local_whisper_prerecorded("https://example.com/audio.wav", return_language=True)
        assert result == ([], "es")

    @patch("utils.stt.pre_recorded.httpx.get")
    def test_fetch_failure_raises_after_retry(self, mock_get, pre_recorded):
        mock_get.side_effect = httpx.RequestError("boom")
        # Function should retry once (attempts<1) then raise — patched httpx.get is hit twice.
        with pytest.raises(RuntimeError, match="Audio fetch failed"):
            pre_recorded.local_whisper_prerecorded("https://example.com/audio.wav")
        assert mock_get.call_count == 2


class TestPublicFunctionRouting:
    def test_whisper_backend_delegates_to_local(self, pre_recorded, monkeypatch):
        monkeypatch.setattr(pre_recorded, "STT_BATCH_BACKEND", "whisper")
        called = {}

        def _fake_local(audio_bytes, **kwargs):
            called["yes"] = True
            called["bytes_len"] = len(audio_bytes)
            called["kwargs"] = kwargs
            return [{"timestamp": [0, 1], "speaker": "SPEAKER_00", "text": "ok"}]

        monkeypatch.setattr(pre_recorded, "local_whisper_prerecorded_from_bytes", _fake_local)
        out = pre_recorded.deepgram_prerecorded_from_bytes(b"\x00" * 8, diarize=True, language="en")
        assert called.get("yes")
        assert out == [{"timestamp": [0, 1], "speaker": "SPEAKER_00", "text": "ok"}]

    def test_whisper_backend_routes_url_function(self, pre_recorded, monkeypatch):
        """Public deepgram_prerecorded (URL variant) must delegate to local_whisper_prerecorded."""
        monkeypatch.setattr(pre_recorded, "STT_BATCH_BACKEND", "whisper")
        called = {"yes": False}

        def _fake_url(url, **kwargs):
            called["yes"] = True
            called["url"] = url
            return [{"timestamp": [0, 1], "speaker": "SPEAKER_00", "text": "ok"}]

        monkeypatch.setattr(pre_recorded, "local_whisper_prerecorded", _fake_url)
        out = pre_recorded.deepgram_prerecorded("https://example.com/audio.wav", diarize=True)
        assert called["yes"]
        assert called["url"] == "https://example.com/audio.wav"
        assert out == [{"timestamp": [0, 1], "speaker": "SPEAKER_00", "text": "ok"}]

    def test_deepgram_backend_unchanged(self, pre_recorded, monkeypatch):
        """If STT_BATCH_BACKEND='deepgram', the local path is NOT called."""
        monkeypatch.setattr(pre_recorded, "STT_BATCH_BACKEND", "deepgram")
        called = {"local": False}

        def _fake_local(*a, **k):
            called["local"] = True
            return []

        monkeypatch.setattr(pre_recorded, "local_whisper_prerecorded_from_bytes", _fake_local)
        # Mock the Deepgram client so the real function returns cleanly without network.
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {"results": {"channels": [{"alternatives": [{"words": []}]}]}}
        mock_client.listen.rest.v.return_value.transcribe_file.return_value = mock_response
        monkeypatch.setattr(pre_recorded, "_deepgram_client", mock_client)

        pre_recorded.deepgram_prerecorded_from_bytes(b"\x00" * 8, encoding="linear16")
        assert called["local"] is False
