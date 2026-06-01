"""OpenAI Realtime API streaming provider (speaches on rtx6000).

Backs ``STT_STREAMING_PROVIDER=realtime-local``. Connects to the LiteLLM-fronted
speaches Realtime WS at ``ws://10.0.60.48:4000/v1/realtime`` and translates the
event stream into the segment shape that ``routers/transcribe.py`` consumes
today.

Probe-verified constraints encoded below (see spec
``docs/superpowers/specs/2026-06-01-stt-migration-deepgram-to-local.md``
Reprobe v1.2 section for the source-of-truth probe transcripts):

  * WS URL MUST include ``?model=whisper-large-v3`` — without it LiteLLM
    returns 403. ``?intent=transcription`` is accepted but is a no-op; we
    send it anyway for documentation purposes.
  * Auth: ``Authorization: Bearer ${OPENAI_API_KEY}`` (same key already wired
    in ``backend/.env`` for the LiteLLM proxy).
  * ``websockets==12.0`` — pass ``extra_headers=`` (NOT
    ``additional_headers``).
  * ``ping_interval=None`` — server does not respond to client pongs; a
    keepalive thread would tear the connection down at ~30s with 1011.
  * NEVER send ``input_audio_buffer.commit`` — speaches throws AssertionError
    on manual commits (T1 reprobe). We rely entirely on server VAD.
  * ``session.update`` IS sent right after ``session.created`` but most
    knobs are silently ignored — T2 reprobe confirmed only ``temperature``
    is genuinely applied on the current build. We send the update anyway
    so the wire log documents intent.
  * The ``.completed`` event's ``audio_start_ms`` / ``audio_end_ms`` RESET
    PER ITEM and cannot be trusted as absolute conversation time (T3
    reprobe). We anchor turn timestamps to our own wall-clock.
  * Each ``.completed`` event becomes its own ``StreamingSegment``. The
    server VAD chunks long utterances into several events; Sortformer +
    downstream consumers handle reassembly (T3 reprobe).
  * No ``.delta`` partial events are emitted; final-only stream. Matches
    omi's existing ``interim_results=False`` Deepgram config — zero UX
    regression.
  * No speaker labels in transcription events. Speaker IDs continue to come
    from the Sortformer service at ``utils/stt/voice-extras:8094`` (same
    parallel pipeline as for Deepgram today). We emit ``SPEAKER_0`` as the
    placeholder.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Optional

import websockets

from utils.stt.streaming_provider import (
    StreamTranscriptFn,
    StreamingSegment,
    StreamingSessionConfig,
    StreamingTranscriptSession,
    STTStreamingProvider,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — overridable via env vars per spec ``.env.template`` plan
# ---------------------------------------------------------------------------

DEFAULT_REALTIME_WS_URL = 'ws://10.0.60.48:4000/v1/realtime?model=whisper-large-v3&intent=transcription'
DEFAULT_REALTIME_TRANSCRIPTION_MODEL = 'Systran/faster-whisper-large-v3'

# Probe-derived: server VAD's ``silence_duration_ms=550`` plus distil-small
# transcription is ~10s end-to-end. We don't drive any client-side timer here;
# this is just the documented expected upper bound for tests.
EXPECTED_FINAL_LATENCY_SEC = 12.0


def _get_ws_url() -> str:
    return os.getenv('REALTIME_WS_URL', DEFAULT_REALTIME_WS_URL)


def _get_transcription_model() -> str:
    # NOTE: probe-verified silently ignored by speaches today (build-level
    # default applies). Kept for forward-compatibility once speaches honors
    # the override; see spec open question 9.
    return os.getenv('REALTIME_TRANSCRIPTION_MODEL', DEFAULT_REALTIME_TRANSCRIPTION_MODEL)


def _get_api_key() -> Optional[str]:
    return os.getenv('OPENAI_API_KEY')


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class RealtimeStreamingSession(StreamingTranscriptSession):
    """A single WebSocket-backed STT session against speaches.

    See module docstring for the probe-verified protocol invariants. The
    public method names match ``SafeDeepgramSocket`` so ``routers/transcribe.py``
    can swap providers via the factory without source changes.
    """

    provider_name_value = 'realtime-local'

    def __init__(
        self,
        stream_transcript: StreamTranscriptFn,
        language: str,
        sample_rate: int,
        channels: int,
        config: Optional[StreamingSessionConfig] = None,
    ):
        self._stream_transcript = stream_transcript
        self._language = language
        self._sample_rate = sample_rate
        self._channels = channels
        self._config = config or StreamingSessionConfig()
        self._vad_gate = self._config.vad_gate
        self._is_active = self._config.is_active

        self._ws: Optional[Any] = None  # websockets.WebSocketClientProtocol
        self._dead = False
        self._closed = False
        self._reader_task: Optional[asyncio.Task] = None

        # Wall-clock anchor — set when WS opens. ``audio_*_ms`` from the
        # server resets per item so we ignore them as absolute timestamps
        # and use this anchor + monotonic offsets instead.
        self._wall_clock_open: Optional[float] = None

        # Per-utterance wall-clock window: updated by speech_started /
        # speech_stopped and read by the .completed handler.
        self._current_utterance_start: Optional[float] = None
        self._current_utterance_end: Optional[float] = None

        # Counters for observability + tests.
        self._segments_emitted = 0
        self._errors_seen = 0

    # ----- public API --------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self.provider_name_value

    def is_alive(self) -> bool:
        return not self._dead and not self._closed

    async def connect(self) -> None:
        """Open the WS, send session.update, start the reader task.

        Raises on connection failure. Caller (``RealtimeApiStreamingProvider``)
        translates that into the ``None`` return contract preserved from
        ``connect_to_deepgram_with_backoff``.
        """
        url = _get_ws_url()
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY is not set; cannot connect to speaches Realtime endpoint')

        headers = {
            'Authorization': f'Bearer {api_key}',
            # Required per OpenAI Realtime docs even on third-party-fronted
            # endpoints. speaches ignores the value.
            'OpenAI-Beta': 'realtime=v1',
        }

        logger.info(
            'Connecting to Realtime API url=%s lang=%s sr=%s ch=%s',
            url,
            self._language,
            self._sample_rate,
            self._channels,
        )

        # websockets 12.0 ⇒ extra_headers; ping_interval=None per spec to
        # avoid the historical (and possibly still latent) 1011 keepalive
        # timeout. max_size=None — server may send unbounded events.
        self._ws = await websockets.connect(
            url,
            extra_headers=headers,
            ping_interval=None,
            max_size=None,
        )
        self._wall_clock_open = time.time()

        await self._send_session_update()

        # Spawn the background reader. We do NOT await it; ``send_audio`` /
        # ``finish`` interact with the same WS concurrently.
        self._reader_task = asyncio.create_task(self._event_loop(), name='realtime-stt-reader')

    async def _send_session_update(self) -> None:
        """Send the probe-verified working session.update payload.

        Per T2 reprobe, only ``temperature`` is actually applied; the model /
        modalities / language overrides are silently ignored on the current
        speaches build. We send them anyway so the wire log documents intent.
        """
        payload = {
            'type': 'session.update',
            'session': {
                'modalities': ['text'],
                'input_audio_transcription': {
                    'model': _get_transcription_model(),
                    'language': self._language if self._language and self._language != 'multi' else None,
                },
                'temperature': 0.0,
            },
        }
        await self._ws_send_json(payload)

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Forward a PCM16LE chunk as an ``input_audio_buffer.append`` event.

        If a VAD gate is attached, audio is routed through it first — same
        pattern Deepgram path uses (silence is gated out, finalize signals
        are absorbed because server VAD owns turn closing on the Realtime
        side).
        """
        if self._dead or self._closed or self._ws is None:
            return

        # Allow callers to register their own activity gate (same shape as
        # ``connect_to_deepgram_with_backoff``'s ``is_active`` parameter).
        if self._is_active is not None and not self._is_active():
            return

        audio_to_send = pcm_bytes
        if self._vad_gate is not None:
            try:
                gate_out = self._vad_gate.process_audio(pcm_bytes, time.time())
                audio_to_send = gate_out.audio_to_send
                # ``should_finalize`` is intentionally ignored — server VAD
                # owns turn boundaries; manual commit triggers AssertionError.
            except Exception:
                logger.exception('VAD gate process error in realtime provider; sending raw audio')
                audio_to_send = pcm_bytes

        if not audio_to_send:
            return

        encoded = base64.b64encode(audio_to_send).decode('ascii')
        event = {'type': 'input_audio_buffer.append', 'audio': encoded}
        await self._ws_send_json(event)

    async def finalize(self) -> None:
        """NO-OP on the Realtime path.

        Manual ``input_audio_buffer.commit`` triggers a server-side
        AssertionError (T1 reprobe). Server VAD handles turn closing
        based on ``silence_duration_ms=550`` (hardcoded build-level
        default — silently ignored override).
        """
        return None

    async def finish(self) -> None:
        """Close the WS and cancel the reader task. Idempotent."""
        if self._closed:
            return
        self._closed = True

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.debug('Error closing Realtime WS', exc_info=True)
            self._ws = None

    # ----- internal helpers --------------------------------------------

    async def _ws_send_json(self, payload: dict) -> None:
        """Send a JSON event over the WS, latching the connection on failure."""
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            logger.warning('Realtime WS send failed (%s: %s); marking dead', type(e).__name__, e)
            self._dead = True

    async def _event_loop(self) -> None:
        """Background reader: translate server events into segments."""
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                try:
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode('utf-8', errors='replace')
                    event = json.loads(raw)
                except Exception:
                    logger.warning('Realtime WS received non-JSON frame; skipping')
                    continue
                await self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning('Realtime WS reader terminated: %s: %s', type(e).__name__, e)
            self._dead = True

    async def _handle_event(self, event: dict) -> None:
        """Translate one Realtime event into the legacy segment dict shape.

        Only the events probe-verified to fire on the current speaches build
        are handled; ``.delta`` is intentionally absent because the server
        does not emit it.
        """
        etype = event.get('type', '')

        if etype == 'session.created' or etype == 'session.updated':
            logger.debug('Realtime session event: %s', etype)
            return

        if etype == 'input_audio_buffer.speech_started':
            # ``audio_start_ms`` resets per item — we anchor to wall-clock.
            now = time.time()
            anchor = self._wall_clock_open if self._wall_clock_open is not None else now
            self._current_utterance_start = max(0.0, now - anchor)
            return

        if etype == 'input_audio_buffer.speech_stopped':
            now = time.time()
            anchor = self._wall_clock_open if self._wall_clock_open is not None else now
            self._current_utterance_end = max(0.0, now - anchor)
            return

        if etype == 'input_audio_buffer.committed':
            # Server-side commit; await the matching .completed next.
            return

        if etype == 'conversation.item.created':
            # Placeholder; no segment yet.
            return

        if etype == 'conversation.item.input_audio_transcription.completed':
            await self._emit_segment_from_completed(event)
            return

        if etype == 'conversation.item.input_audio_transcription.delta':
            # Defensive: spec says server doesn't emit these on this build,
            # but if a future speaches version adds partials we ignore them
            # (omi runs final-only today).
            logger.debug('Ignoring unexpected .delta event from Realtime provider')
            return

        if etype == 'response.created':
            # In STT-only mode the assistant response side stays incomplete.
            # T1/T3 reprobe confirmed these do not fire on the patched build,
            # but we keep the branch as defensive doc.
            return

        if etype == 'error':
            self._errors_seen += 1
            err = event.get('error', {})
            msg = err.get('message', '')
            err_type = err.get('type', '')
            logger.warning('Realtime API error event type=%s message=%s', err_type, msg)
            # Only mark dead on connection-fatal errors. Benign
            # ``InternalServerError`` after a transcript turn does NOT close
            # the WS on the v1.2-patched build, so we leave the connection
            # alive and let the next .completed fire normally.
            if 'connection' in msg.lower() or 'closed' in msg.lower():
                self._dead = True
            return

        logger.debug('Unhandled Realtime event type=%s', etype)

    async def _emit_segment_from_completed(self, event: dict) -> None:
        """Build one ``StreamingSegment`` from a ``.completed`` event.

        Times use the wall-clock anchor + per-utterance start/stop captured
        from the matching ``speech_started`` / ``speech_stopped`` events.
        If the speech_started/stopped pair was missed (rare; first item on
        connect), fall back to a 0-duration stamp at the current wall offset.
        """
        text = (event.get('transcript') or '').strip()
        if not text:
            return

        now = time.time()
        anchor = self._wall_clock_open if self._wall_clock_open is not None else now

        start = self._current_utterance_start
        end = self._current_utterance_end
        if start is None:
            start = max(0.0, now - anchor)
        if end is None or end < start:
            end = start

        segment = StreamingSegment(
            speaker='SPEAKER_0',
            start=float(start),
            end=float(end),
            text=text,
            is_user=False,
            person_id=None,
        )

        # Reset per-utterance window so subsequent .completed events without
        # a fresh speech_started don't reuse the same stamps.
        self._current_utterance_start = None
        self._current_utterance_end = None

        self._segments_emitted += 1

        try:
            # transcribe.py's stream_transcript is a sync function. We call
            # it as-is to preserve the existing contract — matches the
            # Deepgram path which also calls a sync callback inside an async
            # context. If the callback raises we swallow + log so the reader
            # task survives.
            self._stream_transcript([segment.to_dict()])
        except Exception:
            logger.exception('stream_transcript callback raised in realtime provider')


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class RealtimeApiStreamingProvider(STTStreamingProvider):
    """Factory for ``RealtimeStreamingSession``.

    Behavior wraps a single attempt at WS connect. The retry/backoff policy
    that the Deepgram path uses today (``connect_to_deepgram_with_backoff``)
    is intentionally NOT mirrored here in Phase A — the speaches Realtime
    endpoint is wired up as opt-in only via env var, so a single failure
    leaves the operator to flip back to Deepgram. Phase B' adds the
    side-by-side soak that will inform whether we need exponential backoff
    here too.
    """

    provider_name_value = 'realtime-local'

    @property
    def provider_name(self) -> str:
        return self.provider_name_value

    async def open_session(
        self,
        stream_transcript: StreamTranscriptFn,
        language: str,
        sample_rate: int,
        channels: int,
        config: Optional[StreamingSessionConfig] = None,
        model: Optional[str] = None,
        keywords: Optional[list] = None,
        vad_gate: Optional[Any] = None,
        is_active: Optional[Any] = None,
        **_ignored,
    ) -> Optional[StreamingTranscriptSession]:
        # Backward-compat with the dispatcher signature in streaming.py:
        # process_audio_dg passes model/keywords/vad_gate/is_active as raw
        # kwargs (mirroring the Deepgram path). Pack them into a
        # StreamingSessionConfig if no explicit config object was supplied.
        if config is None:
            config = StreamingSessionConfig(
                model=model,
                keywords=list(keywords) if keywords else [],
                vad_gate=vad_gate,
                is_active=is_active,
            )
        session = RealtimeStreamingSession(
            stream_transcript=stream_transcript,
            language=language,
            sample_rate=sample_rate,
            channels=channels,
            config=config,
        )
        try:
            await session.connect()
        except Exception as e:
            logger.error('RealtimeApiStreamingProvider failed to open session: %s: %s', type(e).__name__, e)
            # Mirror connect_to_deepgram_with_backoff's None-on-failure path
            # so the caller in streaming.py can decide whether to fall back.
            try:
                await session.finish()
            except Exception:
                pass
            return None
        return session


__all__ = [
    'RealtimeApiStreamingProvider',
    'RealtimeStreamingSession',
]
