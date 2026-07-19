"""Shared Whisper-hallucination detector.

Single source of truth for hallucination detection so the SAME logic serves
both the realtime STT provider (``utils/stt/realtime_provider.py``, filtering
NEW streaming transcripts) and the memory-extraction input filter
(``utils/conversations/process_conversation.py``, keeping hallucinated
segments out of the extractor even when ``is_user=True``).

A full-corpus scan of the Jarvis account (11,389 convs, 337,994 segments)
found three genuine hallucination categories — all distinct from REAL
backchannel like "Yeah." / "Okay." / "Good." / "I know." which must NEVER be
filtered:

  A. YouTube-residue phrases — Whisper's training corpus is saturated with
     YouTube-style sign-offs ("Thank you for watching.", "Don't forget to
     subscribe.", "[Music]") and short polite filler ("Thank you.", "Bye!",
     "you"). When fed silence / breathing / mic noise the model confabulates
     these. ~800 convs. Handled by exact normalized-phrase match against
     ``DEFAULT_HALLUCINATION_PHRASES`` (env-overridable via
     ``REALTIME_HALLUCINATION_PHRASES``). Exact-only, never substring — real
     speech that merely contains a residue phrase must survive.

  B. Repetition LOOPS — a phrase repeated 3+ times back-to-back inside one
     segment: "Come here. Come here. Come here." / "Let's go" x9 / comma-runs
     like "come on, come on, come on". 562 segments across 370 convs. NOT
     filtered before this module existed — the main gap.

  C. A specific recurring phantom — "Yeah, I think we got all right." and the
     variant "Yeah, I think we got allright." 703 occurrences, ~1.08s
     is_user=True segments, Whisper confabulating on a recurring ~1s audio
     signature.

``is_whisper_hallucination(text)`` returns True for any of the three
categories and False for everything else — including all normal short
backchannel.
"""

from __future__ import annotations

import os
import re
import string

# ---------------------------------------------------------------------------
# Category A — known YouTube-residue phrases (exact normalized match).
#
# Env override: REALTIME_HALLUCINATION_PHRASES, comma-separated. Empty string
# ("") explicitly DISABLES category A only (loops + phantom still fire).
# ---------------------------------------------------------------------------

DEFAULT_HALLUCINATION_PHRASES = [
    'Thank you.',
    'Thanks.',
    'Thank you for watching.',
    'Thanks for watching.',
    "Don't forget to subscribe.",
    'Subscribe.',
    'Bye.',
    'Bye!',
    'Bye bye.',
    'Bye bye!',
    'Goodbye.',
    '[Music]',
    'Music plays.',
    'Music.',
    '♪',
    '...',
    'You.',
]

# ---------------------------------------------------------------------------
# Category C — specific recurring phantom phrases (exact normalized match).
# Whisper confabulates these on a recurring ~1s audio signature. Kept
# separate from category A because they are NOT YouTube residue and are NOT
# env-overridable — they are a hard-coded corpus finding.
# ---------------------------------------------------------------------------

PHANTOM_PHRASES = {
    'yeah i think we got all right',
    'yeah i think we got allright',
}

# ---------------------------------------------------------------------------
# Category B — repetition-loop tuning constants.
# ---------------------------------------------------------------------------

# Minimum normalized length for a sentence-split phrase to count toward a
# loop. Below this we treat repeats as (possibly real) backchannel — e.g.
# "No. No. No." / "Yeah. Yeah." stays untouched. 8 chars is long enough that
# a genuine repeated word/short-phrase loop ("come here", "let's go") trips it
# while single-word backchannel does not.
_MIN_LOOP_PHRASE_LEN = 8

# Minimum consecutive repeats to call something a loop.
_MIN_LOOP_REPEATS = 3

# Punctuation set used by _normalize_phrase to strip leading/trailing residue.
# ``string.punctuation`` covers ASCII punctuation; we add em/en dash + ellipsis
# explicitly since the unicode variants are common in Whisper output.
_PHRASE_STRIP_CHARS = string.punctuation + '—–…' + ' \t\n\r'

# Sentence splitter for loop detection — split on . ! ? runs.
_SENTENCE_SPLIT_RE = re.compile(r'[.!?]+')


def _normalize_phrase(s: str) -> str:
    """Normalize a phrase for hallucination comparison.

    Applied identically to the configured phrase set (at load time), the
    phantom set, and to incoming transcript text (at filter time) so
    comparison is symmetric.

    Pipeline:
      1. ``.strip()`` outer whitespace.
      2. ``.lower()`` — case-insensitive match.
      3. Strip leading + trailing punctuation (ASCII punct + em/en dash +
         ellipsis + whitespace).
      4. Collapse internal whitespace runs to a single space.

    Examples:
      ``"Thank you."`` -> ``"thank you"``
      ``" THANK   YOU! "`` -> ``"thank you"``
      ``"[Music]"`` -> ``"music"`` (brackets are punctuation)
      ``"♪"`` -> ``"♪"`` (not stripped by string.punctuation)
    """
    s = s.strip().lower()
    s = s.strip(_PHRASE_STRIP_CHARS)
    return ' '.join(s.split())


# Internal-punctuation set for the exact-match normalizer. Whisper commonly
# inserts commas mid-phrase ("Yeah, I think we got all right."), so the
# exact-match categories (A residue + C phantom) strip ALL punctuation — not
# just leading/trailing — before comparison. Loop detection deliberately does
# NOT use this (it needs the raw text to sentence/comma-split).
_INTERNAL_PUNCT_RE = re.compile(r'[' + re.escape(string.punctuation + '—–…') + r']')


def _normalize_exact(s: str) -> str:
    """Normalize for the exact-match categories (residue + phantom).

    Like ``_normalize_phrase`` but strips ALL punctuation (internal too),
    replacing each punctuation run with a space and collapsing whitespace.
    So ``"Yeah, I think we got all right."`` -> ``"yeah i think we got all
    right"``, which matches the phantom set. ``"[Music]"`` -> ``"music"``.
    """
    s = s.lower()
    s = _INTERNAL_PUNCT_RE.sub(' ', s)
    return ' '.join(s.split())


def _load_hallucination_phrases() -> set:
    """Build the active category-A phrase set from env + defaults.

    ``REALTIME_HALLUCINATION_PHRASES`` unset/None -> defaults.
    ``REALTIME_HALLUCINATION_PHRASES=""`` (explicit empty) -> category A
        disabled (returns empty set; loops + phantom still fire).
    Otherwise comma-separated list replaces the defaults.
    """
    raw = os.getenv('REALTIME_HALLUCINATION_PHRASES')
    if raw is None:
        items = DEFAULT_HALLUCINATION_PHRASES
    elif raw.strip() == '':
        return set()
    else:
        items = raw.split(',')
    normalized = {_normalize_exact(p) for p in items if p.strip()}
    normalized.discard('')
    return normalized


# Module-level cache; tests can rebuild via _load_hallucination_phrases().
_HALLUCINATION_PHRASES = _load_hallucination_phrases()

# Phantom set normalized through the exact-match normalizer so it matches
# incoming text regardless of the comma Whisper inserts ("Yeah, I think ...").
_PHANTOM_PHRASES_NORM = {_normalize_exact(p) for p in PHANTOM_PHRASES}


def _is_residue_phrase(normalized: str) -> bool:
    """Category A — exact normalized match against the residue phrase set."""
    if not _HALLUCINATION_PHRASES:
        return False
    return normalized in _HALLUCINATION_PHRASES


def _is_phantom_phrase(normalized: str) -> bool:
    """Category C — exact normalized match against the phantom phrase set."""
    return normalized in _PHANTOM_PHRASES_NORM


def _max_consecutive_repeats(units) -> int:
    """Return the longest run of identical consecutive units in ``units``."""
    best = 0
    run = 0
    prev = None
    for u in units:
        if u and u == prev:
            run += 1
        else:
            run = 1
            prev = u
        if run > best:
            best = run
    return best


def _is_repetition_loop(text: str) -> bool:
    """Category B — a phrase repeated 3+ times back-to-back.

    Two independent triggers:

      1. Sentence-split loop: split on ``[.!?]``, normalize each fragment,
         and flag if any fragment >= ``_MIN_LOOP_PHRASE_LEN`` chars repeats
         ``_MIN_LOOP_REPEATS``+ times consecutively.
         ("Come here. Come here. Come here.")

      2. Comma-run loop: split on commas, normalize each fragment, and flag
         if the SAME short fragment repeats ``_MIN_LOOP_REPEATS``+ times
         consecutively. Comma-runs are usually short ("come on, come on,
         come on") so no min-length gate applies here — the >=3 consecutive
         repeat is itself the signal.

    A single incidental repeat ("I think we should go. Let's go.") never
    trips either trigger.
    """
    # Trigger 1 — sentence-split loop (gated on phrase length).
    sentences = [_normalize_phrase(s) for s in _SENTENCE_SPLIT_RE.split(text)]
    long_sentences = [s if len(s) >= _MIN_LOOP_PHRASE_LEN else '\x00' for s in sentences]
    if _max_consecutive_repeats(long_sentences) >= _MIN_LOOP_REPEATS:
        return True

    # Trigger 2 — comma-run loop (no length gate; the run itself is signal).
    if ',' in text:
        fragments = [_normalize_phrase(f) for f in text.split(',')]
        fragments = [f for f in fragments if f]
        if _max_consecutive_repeats(fragments) >= _MIN_LOOP_REPEATS:
            return True

    return False


def is_whisper_hallucination(text: str) -> bool:
    """True if ``text`` is a Whisper hallucination in any of the 3 categories.

      A. Exact YouTube-residue phrase (env-overridable set), OR
      B. Repetition loop (phrase >=8 chars repeated 3+ times back-to-back via
         sentence-split, OR a comma-run repeated >=3 times), OR
      C. A specific recurring phantom phrase.

    Returns False for everything else — crucially including normal short
    backchannel ("Yeah.", "Okay.", "Good.", "No.", "I know.", "I don't
    know.") which is REAL speech and must never be filtered.
    """
    normalized = _normalize_exact(text)
    if not normalized:
        return False
    if _is_residue_phrase(normalized):
        return True
    if _is_phantom_phrase(normalized):
        return True
    if _is_repetition_loop(text):
        return True
    return False


# Backward-compat alias — realtime_provider historically exported the
# private ``_is_whisper_hallucination`` name; keep it pointing at the shared
# public function so existing imports keep working.
_is_whisper_hallucination = is_whisper_hallucination


__all__ = [
    'DEFAULT_HALLUCINATION_PHRASES',
    'PHANTOM_PHRASES',
    'is_whisper_hallucination',
    '_is_whisper_hallucination',
    '_normalize_phrase',
    '_load_hallucination_phrases',
]
