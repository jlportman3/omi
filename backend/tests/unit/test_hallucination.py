"""Unit tests for the shared Whisper-hallucination detector.

Covers the three corpus-verified categories from
``utils/stt/hallucination.py`` — YouTube-residue phrases (A), repetition
loops (B), and the recurring phantom phrase (C) — and, critically, asserts
that REAL short backchannel ("Yeah.", "Okay.", "Good.", "I know.") is NEVER
flagged. Filtering real backchannel would silently delete genuine user
speech from the memory extractor, so those negative cases are the
load-bearing guardrail.

Pure unit tests: no network, no external services. The detector is stdlib-
only (os/re/string), so these run fast and hermetically.
"""

import pytest

from utils.stt.hallucination import (
    DEFAULT_HALLUCINATION_PHRASES,
    PHANTOM_PHRASES,
    _load_hallucination_phrases,
    _normalize_phrase,
    is_whisper_hallucination,
)


@pytest.fixture(autouse=True)
def _fresh_residue_set(monkeypatch):
    """Rebuild the residue set from defaults for every test.

    Ensures env-order independence: a stray ``REALTIME_HALLUCINATION_PHRASES``
    in the ambient environment must not weaken these assertions.
    """
    monkeypatch.delenv('REALTIME_HALLUCINATION_PHRASES', raising=False)
    import utils.stt.hallucination as hallucination_module

    monkeypatch.setattr(hallucination_module, '_HALLUCINATION_PHRASES', _load_hallucination_phrases())


# ---------------------------------------------------------------------------
# Category B — repetition loops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'text',
    [
        'Come here. Come here. Come here.',
        "Let's go. Let's go. Let's go. Let's go.",
        'come on, come on, come on',
    ],
)
def test_repetition_loops_flagged(text):
    assert is_whisper_hallucination(text) is True


def test_nine_times_loop_flagged():
    # "Let's go" repeated 9 times back-to-back — a classic Whisper loop.
    text = ' '.join(["Let's go."] * 9)
    assert is_whisper_hallucination(text) is True


# ---------------------------------------------------------------------------
# Category A — YouTube-residue phrases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'text',
    [
        'Thank you.',
        'you',
        'Bye.',
        'Thank you for watching.',
        'Subscribe.',
        '[Music]',
    ],
)
def test_residue_phrases_flagged(text):
    assert is_whisper_hallucination(text) is True


# ---------------------------------------------------------------------------
# Category C — recurring phantom phrase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'text',
    [
        'Yeah, I think we got all right.',
        'Yeah, I think we got allright.',
        # Whitespace / casing variants normalize identically.
        '  YEAH,  I THINK  WE GOT ALL RIGHT!  ',
    ],
)
def test_phantom_phrases_flagged(text):
    assert is_whisper_hallucination(text) is True


def test_phantom_set_is_normalized_form():
    # Sanity: the hard-coded phantom set is stored in normalized form so the
    # exact-match path can compare against incoming Whisper output directly.
    assert 'yeah i think we got all right' in PHANTOM_PHRASES
    assert 'yeah i think we got allright' in PHANTOM_PHRASES


# ---------------------------------------------------------------------------
# REAL backchannel — MUST NOT be flagged. This is the guardrail: filtering
# these would delete genuine user speech from the memory extractor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'text',
    [
        'Yeah.',
        'Okay.',
        'Good.',
        'No.',
        'I know.',
        "I don't know.",
        # Longer real speech that merely contains a residue word.
        'Thank you very much for that pitch.',
        'I want to subscribe to the streaming service.',
        # Real speech with a single incidental repeat is NOT a loop.
        "I think we should go. Let's go.",
        # Two consecutive "No" is below the >=3 loop threshold and the rest
        # of the sentence is unique real speech.
        'No, no, I already told you about the meeting yesterday.',
    ],
)
def test_real_backchannel_and_speech_not_flagged(text):
    assert is_whisper_hallucination(text) is False


def test_empty_and_whitespace_not_flagged():
    assert is_whisper_hallucination('') is False
    assert is_whisper_hallucination('   ') is False


# ---------------------------------------------------------------------------
# Env override + normalization sanity
# ---------------------------------------------------------------------------


def test_env_override_replaces_residue_set(monkeypatch):
    """``REALTIME_HALLUCINATION_PHRASES`` replaces the default residue set;
    loops + phantom still fire independently of the residue override."""
    import utils.stt.hallucination as hallucination_module

    monkeypatch.setenv('REALTIME_HALLUCINATION_PHRASES', 'Foo bar.,Baz!')
    monkeypatch.setattr(hallucination_module, '_HALLUCINATION_PHRASES', _load_hallucination_phrases())

    # Old default residue no longer flagged.
    assert is_whisper_hallucination('Thank you.') is False
    # New custom residue flagged.
    assert is_whisper_hallucination('Foo bar.') is True
    # Loops + phantom are independent of the residue override.
    assert is_whisper_hallucination('Come here. Come here. Come here.') is True
    assert is_whisper_hallucination('Yeah, I think we got all right.') is True


def test_empty_env_disables_residue_only(monkeypatch):
    """An explicit empty env value disables category A (residue) but NOT the
    loop / phantom categories."""
    import utils.stt.hallucination as hallucination_module

    monkeypatch.setenv('REALTIME_HALLUCINATION_PHRASES', '')
    monkeypatch.setattr(hallucination_module, '_HALLUCINATION_PHRASES', _load_hallucination_phrases())

    assert is_whisper_hallucination('Thank you.') is False
    assert is_whisper_hallucination('Come here. Come here. Come here.') is True
    assert is_whisper_hallucination('Yeah, I think we got all right.') is True


def test_default_residue_set_covers_expected():
    normalized = {_normalize_phrase(p) for p in DEFAULT_HALLUCINATION_PHRASES}
    assert 'thank you' in normalized
    assert 'thanks for watching' in normalized
    assert 'subscribe' in normalized
    assert 'music' in normalized
