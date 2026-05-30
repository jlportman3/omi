"""Unit tests for the Kokoro TTS backend code paths in routers/tts.py.

The router now supports two TTS backends selected by the TTS_BACKEND env
variable: "elevenlabs" (legacy) and "kokoro" (local, via LiteLLM/speaches
on rtx6000). These tests cover the kokoro-specific helpers:

  - _is_valid_voice_id(voice_id, backend="kokoro") accepts the Kokoro
    voice-name convention (alphanumeric + underscore), but still rejects
    path traversal and oversized inputs.
  - _resolve_voice_for_kokoro(voice_id) maps known legacy ElevenLabs
    voice IDs (e.g. the default "Sloane") onto their closest Kokoro voice,
    and passes through Kokoro-shape IDs unchanged.

End-to-end synthesize wiring is covered by the existing test_tts.py and
by manual integration smoke against live voice-extras / LiteLLM.
"""

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

os.environ.setdefault(
    "ENCRYPTION_SECRET",
    "omi_ZwB2ZNqB2HHpMK6wStk7sTpavJiPTFg7gXUHnc4tFABPU6pZ2c2DKgehtfgi4RZv",
)


def _stub_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _stub_package(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


# Pre-stub heavy deps the router pulls in transitively.
for mod_name in [
    "firebase_admin",
    "firebase_admin.firestore",
    "firebase_admin.auth",
    "firebase_admin.credentials",
]:
    _stub_package(mod_name) if "." not in mod_name else _stub_module(mod_name)
redis_stub = _stub_module("redis")
redis_stub.Redis = MagicMock(return_value=MagicMock())


def _load_tts_router_module():
    endpoints_stub = types.ModuleType("utils.other.endpoints")

    def _fake_dep_factory():
        async def _dep():
            return "test-uid"

        return _dep

    endpoints_stub.get_current_user_uid = _fake_dep_factory()
    endpoints_stub.with_rate_limit = lambda _auth, _policy: _fake_dep_factory()
    sys.modules["utils.other.endpoints"] = endpoints_stub

    redis_db_stub = types.ModuleType("database.redis_db")
    redis_db_stub.check_tts_rate_limit = MagicMock(return_value=(0, 0))
    sys.modules["database.redis_db"] = redis_db_stub
    sys.modules.setdefault("database", _stub_package("database"))
    sys.modules["database"].redis_db = redis_db_stub

    http_client_stub = types.ModuleType("utils.http_client")
    http_client_stub.get_tts_client = MagicMock()
    http_client_stub.get_tts_semaphore = MagicMock()
    sys.modules["utils.http_client"] = http_client_stub

    log_sanitizer_stub = types.ModuleType("utils.log_sanitizer")
    log_sanitizer_stub.sanitize = lambda s: str(s)
    sys.modules["utils.log_sanitizer"] = log_sanitizer_stub

    sys.path.insert(0, str(BACKEND_DIR))
    spec = importlib.util.spec_from_file_location(
        "routers.tts",
        str(BACKEND_DIR / "routers" / "tts.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["routers.tts"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tts_router():
    return _load_tts_router_module()


# ---------------------------------------------------------------------------
# _is_valid_voice_id — kokoro backend
# ---------------------------------------------------------------------------


class TestValidVoiceIdKokoroBackend:
    def test_accepts_kokoro_voice_names(self, tts_router):
        assert tts_router._is_valid_voice_id("af_bella", backend="kokoro")
        assert tts_router._is_valid_voice_id("am_michael", backend="kokoro")
        assert tts_router._is_valid_voice_id("bf_isabella", backend="kokoro")
        assert tts_router._is_valid_voice_id("bm_george", backend="kokoro")

    def test_accepts_legacy_elevenlabs_id_for_translation(self, tts_router):
        """Old client builds may still send ElevenLabs IDs; validator must
        accept them so the translation layer downstream can map them."""
        assert tts_router._is_valid_voice_id("BAMYoBHLZM7lJgJAmFz0", backend="kokoro")

    def test_rejects_path_traversal(self, tts_router):
        """Underscore is allowed but slash, dot-dot, and other path separators
        must still be rejected — the URL template is still being formatted."""
        assert not tts_router._is_valid_voice_id("../../history", backend="kokoro")
        assert not tts_router._is_valid_voice_id("../v1/voices", backend="kokoro")
        assert not tts_router._is_valid_voice_id("foo/bar", backend="kokoro")
        assert not tts_router._is_valid_voice_id("af_bella/x", backend="kokoro")

    def test_rejects_other_special_chars(self, tts_router):
        assert not tts_router._is_valid_voice_id("id-with-dash", backend="kokoro")
        assert not tts_router._is_valid_voice_id("id with space", backend="kokoro")
        assert not tts_router._is_valid_voice_id("id?query=1", backend="kokoro")
        assert not tts_router._is_valid_voice_id("af.bella", backend="kokoro")

    def test_rejects_empty(self, tts_router):
        assert not tts_router._is_valid_voice_id("", backend="kokoro")

    def test_length_boundaries(self, tts_router):
        assert tts_router._is_valid_voice_id("a" * 128, backend="kokoro")
        assert not tts_router._is_valid_voice_id("a" * 129, backend="kokoro")


# ---------------------------------------------------------------------------
# _is_valid_voice_id — elevenlabs backend (unchanged behavior, regression)
# ---------------------------------------------------------------------------


class TestValidVoiceIdElevenLabsBackend:
    def test_underscore_still_rejected(self, tts_router):
        """Backend='elevenlabs' preserves the strict alnum-only rule so the
        legacy ElevenLabs URL template stays safe."""
        assert not tts_router._is_valid_voice_id("af_bella", backend="elevenlabs")
        assert not tts_router._is_valid_voice_id("id_with_underscore", backend="elevenlabs")

    def test_accepts_alphanumeric(self, tts_router):
        assert tts_router._is_valid_voice_id("BAMYoBHLZM7lJgJAmFz0", backend="elevenlabs")


# ---------------------------------------------------------------------------
# _resolve_voice_for_kokoro
# ---------------------------------------------------------------------------


class TestResolveVoiceForKokoro:
    def test_translates_default_sloane(self, tts_router):
        """The historical default ElevenLabs voice (Sloane) maps to a sensible
        warm-American-female Kokoro voice so existing client defaults still
        produce reasonable audio after the swap."""
        from models.tts import DEFAULT_VOICE_ID

        translated = tts_router._resolve_voice_for_kokoro(DEFAULT_VOICE_ID)
        assert translated.startswith("af_"), f"expected an American-female voice, got {translated!r}"
        assert translated != DEFAULT_VOICE_ID

    def test_passes_through_kokoro_voice_unchanged(self, tts_router):
        for voice in ("af_bella", "am_michael", "bf_emma", "bm_george"):
            assert tts_router._resolve_voice_for_kokoro(voice) == voice

    def test_passes_through_unknown_elevenlabs_id(self, tts_router):
        """Unknown legacy IDs pass through; voice-extras will respond with an
        error rather than silently substituting."""
        assert tts_router._resolve_voice_for_kokoro("abc123XYZ") == "abc123XYZ"
