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
    ``SafeDeepgramSocket`` / ``GatedDeepgramSocket``. Methods are named to
    match the existing duck-typed surface so transcribe.py can swap providers
    via the env var without source changes.

    Lifecycle:
        s = await provider.open_session(...)
        while audio:
            await s.send_audio(pcm_bytes)
            # provider invokes stream_transcript(...) as segments arrive
        await s.finalize()  # idempotent; some providers no-op (server VAD)
        await s.finish()
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier — e.g. ``'deepgram'`` or ``'realtime-local'``."""

    @abstractmethod
    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Forward a PCM16LE chunk to the underlying transport.

        Implementations SHOULD be silent on already-dead connections — the
        caller does not check return values today (see
        ``routers/transcribe.py`` audio-write path).
        """

    @abstractmethod
    async def finalize(self) -> None:
        """Flush any pending partial transcript.

        For server-VAD providers (Realtime API on speaches) this is a NO-OP
        because manual ``input_audio_buffer.commit`` triggers a server
        AssertionError (spec, T1 reprobe). For Deepgram this maps to the
        existing ``finalize()`` call on speech->silence transitions.
        """

    @abstractmethod
    async def finish(self) -> None:
        """Close the underlying transport. MUST be idempotent."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Return ``False`` once the connection has been detected as dead.

        Implementations mirror today's ``SafeDeepgramSocket.is_connection_dead``
        semantics (one-way latch). The caller may stop sending audio once this
        returns ``False``.
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
    ``model`` are consumed; the rest are reserved.
    """

    keywords: List[str] = field(default_factory=list)
    model: Optional[str] = None
    vad_gate: Optional[Any] = None  # utils.stt.vad_gate.VADStreamingGate
    is_active: Optional[Callable[[], bool]] = None
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
