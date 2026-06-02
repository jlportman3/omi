"""STT streaming provider abstraction (Phase A of the STT migration).

This module defines the **pure** interfaces — no I/O, no concrete provider
implementations, no httpx, no websockets. Concrete providers live in sibling
modules (``streaming.py`` for Deepgram, ``realtime_provider.py`` for the
speaches OpenAI-Realtime adapter targeting rtx6000).

Why a separate ABC module:
  - Lets unit tests import the ABC contract without dragging in DG SDK or
    websockets clients.
  - Documents the surface that ``routers/transcribe.py`` consumes today via
    ``process_audio_dg`` — every concrete provider must satisfy it so that
    ``transcribe.py`` does not change.

Spec: ``docs/superpowers/specs/2026-06-01-stt-migration-deepgram-to-local.md``
(Adapter design + Reprobe v1.2 sections).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, List, Optional


# ---------------------------------------------------------------------------
# Segment shape — matches what the existing pipeline emits today
# ---------------------------------------------------------------------------


@dataclass
class StreamingSegment:
    """A single transcript segment emitted by a streaming STT provider.

    Field shape mirrors the dict shape that ``stream_transcript`` is invoked
    with by today's Deepgram path in ``streaming.py``. Speaker IDs are
    ``SPEAKER_{n}`` strings (matching memory-attribution Layer 2 expectations
    in ``utils/stt/attribution.py``). For the Realtime API path the speaker
    label defaults to ``SPEAKER_0`` and is overwritten downstream by the
    Sortformer-driven ``speaker_identification`` pipeline — identical to how
    Deepgram's per-word speaker ints are processed today.

    ``start`` / ``end`` are seconds, wall-clock-relative to the session start
    (the VAD gate's ``dg_wall_mapper`` remaps Deepgram's audio-time to
    wall-clock; Realtime provider tracks wall-clock directly because the
    per-item ``audio_*_ms`` values reset on every utterance — see spec
    "Per-item audio_start_ms reset" mitigation).
    """

    speaker: str
    start: float
    end: float
    text: str
    is_user: bool = False
    person_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Return the legacy dict shape consumed by ``stream_transcript``."""
        return asdict(self)


# Callback signature accepted by ``open_session``. Providers MUST invoke this
# with a list of dicts (NOT ``StreamingSegment`` instances) to preserve
# backward compatibility with ``routers/transcribe.py`` which expects the
# same dict shape Deepgram emits today.
StreamTranscriptFn = Callable[[List[dict]], None]


# ---------------------------------------------------------------------------
# Streaming session ABC — analog of SafeDeepgramSocket's public surface
# ---------------------------------------------------------------------------


class StreamingTranscriptSession(ABC):
    """A long-lived push-bytes / emit-segments STT session.

    This is the contract ``routers/transcribe.py`` consumes today through
    ``SafeDeepgramSocket`` / ``GatedDeepgramSocket``. Methods are named —
    and intentionally **SYNC** — to match the existing duck-typed surface so
    transcribe.py can swap providers via the env var without source changes.

    Why sync, not async:
        transcribe.py runs an async WebSocket handler but treats the STT
        session as a sync object — it calls ``session.send(chunk)`` /
        ``session.finish()`` without ``await`` (see lines 2452, 2793, 2797
        in ``routers/transcribe.py``). ``SafeDeepgramSocket`` matches that
        contract because the Deepgram SDK's ``LiveConnection.send`` is sync.
        Any provider that needs async I/O internally MUST schedule it onto
        the caller's event loop (e.g. via ``asyncio.run_coroutine_threadsafe``
        or ``loop.create_task``) — do NOT change the public surface to
        ``async`` without also rewriting transcribe.py.

    Lifecycle:
        s = await provider.open_session(...)   # async — opens transport
        while audio:
            s.send(pcm_bytes)                  # sync — returns True/False
            # provider invokes stream_transcript(...) as segments arrive
        s.finalize()                           # sync; some providers no-op
        s.finish()                             # sync — closes transport
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier — e.g. ``'deepgram'`` or ``'realtime-local'``."""

    @property
    @abstractmethod
    def is_connection_dead(self) -> bool:
        """True once the underlying transport has been detected as dead.

        One-way latch — once True, MUST never flip back to False. The caller
        (transcribe.py:flush_stt_buffer) checks this before every ``send``
        and stops forwarding audio when it goes True (see #5870).

        Mirrors ``SafeDeepgramSocket.is_connection_dead``.
        """

    @abstractmethod
    def send(self, pcm_bytes: bytes) -> bool:
        """Forward a PCM16LE chunk to the underlying transport.

        SYNC by contract — see class docstring. Returns ``True`` if the bytes
        were scheduled / sent, ``False`` if the session is dead. Today's
        caller (transcribe.py) ignores the return value, but tests assert on
        it to catch dead-connection regressions early.

        Implementations SHOULD be silent on already-dead connections — never
        raise; just return ``False``.
        """

    @abstractmethod
    def finalize(self) -> None:
        """Flush any pending partial transcript.

        SYNC by contract. For server-VAD providers (Realtime API on speaches)
        this is a NO-OP because manual ``input_audio_buffer.commit`` triggers
        a server AssertionError (spec, T1 reprobe). For Deepgram this maps to
        the existing ``finalize()`` call on speech->silence transitions.
        """

    @abstractmethod
    def finish(self) -> None:
        """Close the underlying transport. SYNC by contract; MUST be idempotent.

        Implementations that need async cleanup MUST schedule it onto the
        caller's event loop and return — do NOT block waiting for completion
        (transcribe.py is mid-shutdown and cannot await).
        """


# ---------------------------------------------------------------------------
# Provider ABC — factory for sessions
# ---------------------------------------------------------------------------


@dataclass
class StreamingSessionConfig:
    """Optional per-session knobs passed through to providers.

    Kept as a separate dataclass (rather than expanding ``open_session``'s
    positional signature) so future providers can add knobs without churning
    every caller. Today only ``vad_gate`` / ``is_active`` / ``keywords`` /
    ``model`` / ``sample_width`` are consumed; the rest are reserved.

    ``sample_width`` is the *byte width* per audio sample as transmitted to
    ``send()`` — i.e. 2 for int16 LE (the canonical omi pipeline format) and
    1 for int8 (legacy / hypothetical 8-bit-source firmware). It is NOT the
    same as the omi WS ``codec=`` query param: the Friend pendant connects
    with ``?codec=pcm8&sample_rate=16000`` but the firmware actually emits
    int16 LE @ 16 kHz — the "8" in ``pcm8`` historically referred to the
    sample rate in kHz, not the bit depth (see codec mapping notes in
    ``app/lib/backend/schema/bt_device/bt_device.dart``). Callers therefore
    pass ``sample_width=2`` for both ``pcm8`` and ``pcm16``. Providers MAY
    use this field to validate / convert formats defensively — e.g. the
    realtime provider widens int8 → int16 LE before forwarding so a future
    truly-8-bit codec wouldn't silently feed Whisper garbage and trigger
    hallucinations on noise.
    """

    keywords: List[str] = field(default_factory=list)
    model: Optional[str] = None
    vad_gate: Optional[Any] = None  # utils.stt.vad_gate.VADStreamingGate
    is_active: Optional[Callable[[], bool]] = None
    sample_width: int = 2  # bytes per sample as transmitted to send(); 2 = int16 LE
    extra: dict = field(default_factory=dict)


class STTStreamingProvider(ABC):
    """Factory that opens a ``StreamingTranscriptSession``.

    Concrete providers:
      - ``DeepgramStreamingProvider`` (in ``utils/stt/streaming.py``)
      - ``RealtimeApiStreamingProvider`` (in ``utils/stt/realtime_provider.py``)
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @abstractmethod
    async def open_session(
        self,
        stream_transcript: StreamTranscriptFn,
        language: str,
        sample_rate: int,
        channels: int,
        config: Optional[StreamingSessionConfig] = None,
    ) -> Optional[StreamingTranscriptSession]:
        """Open a new streaming session.

        Returns ``None`` if the underlying transport could not be brought up
        (matches today's ``process_audio_dg`` contract — see retry loop in
        ``connect_to_deepgram_with_backoff``). Raises on unrecoverable errors.
        """


# Type alias for callers that want to pass an awaitable invoke explicitly.
SessionOpener = Callable[..., Awaitable[Optional[StreamingTranscriptSession]]]


__all__ = [
    'StreamingSegment',
    'StreamingTranscriptSession',
    'STTStreamingProvider',
    'StreamingSessionConfig',
    'StreamTranscriptFn',
    'SessionOpener',
]
