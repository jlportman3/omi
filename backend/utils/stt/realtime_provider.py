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
  * NEVER send ``conversation.item.create`` from the client. speaches owns
    the item lifecycle — it auto-creates items via VAD with server-assigned
    ``item_id`` values and pushes them through
    ``conversation.item.created`` events. If we (or any retry/echo loop)
    ever post a ``conversation.item.create`` with an existing id the
    server emits a benign ``Error adding item: ... already exists`` event
    that we recognize and demote in ``_handle_event``.
  * Per-sample byte width: speaches expects int16 LE PCM samples in
    ``input_audio_buffer.append``. The omi WS ``codec=pcm8`` Friend pendant
    transmits int16 LE @ 16 kHz despite the legacy name (firmware emits
    int16 from the PDM mic — verified empirically against the live audio
    in GCS). The provider therefore treats ``sample_width=2`` as the
    canonical input and only widens / narrows if a caller explicitly passes
    ``sample_width=1`` (defensive, in case a future device codec carries
    true int8 samples).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import string
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


# ---------------------------------------------------------------------------
# Whisper hallucination filter — known-residue phrases.
#
# Whisper's training corpus contains lots of YouTube-style sign-offs ("Thank
# you for watching.", "Don't forget to subscribe.", "[Music]") and short
# polite filler ("Thank you.", "Bye!"). When the model is fed silence,
# breathing, or mic noise it confabulates these. The DevKit recording for
# conv 3deec0e8-87cf-48de-8ece-e6ab0a85f112 had several confirmed cases
# (e.g. idx 3 'Thank you.', idx 16 'Thanks for watching!', idx 47
# 'Thank you for watching. Thank you for watching.') — Joe confirmed only
# the "Thank you." was a real hallucination; the other collapsed-timestamp
# segments were REAL speech. So the filter is exact-phrase only, never
# substring (avoids dropping real speech that happens to contain residue).
#
# Env override: REALTIME_HALLUCINATION_PHRASES, comma-separated. Empty
# string explicitly disables the filter.
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

# Punctuation set used by _normalize_phrase to strip leading/trailing
# residue. ``string.punctuation`` covers ASCII punctuation; we add em/en
# dash and ellipsis explicitly since the unicode variants are common in
# Whisper output.
_PHRASE_STRIP_CHARS = string.punctuation + '—–…' + ' \t\n\r'


def _normalize_phrase(s: str) -> str:
    """Normalize a phrase for hallucination comparison.

    Applied identically to both the configured phrase set (at load time)
    and to incoming transcript text (at filter time) so comparison is
    symmetric.

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


def _load_hallucination_phrases() -> set:
    """Build the active hallucination phrase set from env + defaults.

    ``REALTIME_HALLUCINATION_PHRASES`` unset/None -> defaults.
    ``REALTIME_HALLUCINATION_PHRASES=""`` (explicit empty) -> filter disabled.
    Otherwise comma-separated list replaces the defaults.
    """
    raw = os.getenv('REALTIME_HALLUCINATION_PHRASES')
    if raw is None:
        items = DEFAULT_HALLUCINATION_PHRASES
    elif raw.strip() == '':
        return set()
    else:
        items = raw.split(',')
    return {_normalize_phrase(p) for p in items if p.strip()}


# Module-level cache; tests can rebuild via _load_hallucination_phrases().
_HALLUCINATION_PHRASES = _load_hallucination_phrases()


def _is_whisper_hallucination(text: str) -> bool:
    """True if ``text`` (after normalization) matches a known residue phrase.

    Exact normalized-string match only — substring matching would drop
    real speech that contains a residue phrase as part of a longer
    utterance.
    """
    if not _HALLUCINATION_PHRASES:
        return False
    normalized = _normalize_phrase(text)
    if not normalized:
        return False
    return normalized in _HALLUCINATION_PHRASES


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
    public method names + sync/async signature match ``SafeDeepgramSocket``
    so ``routers/transcribe.py`` can swap providers via the factory without
    source changes.

    Sync/async boundary:
        The public surface (``send`` / ``finalize`` / ``finish`` /
        ``is_connection_dead``) is SYNC because transcribe.py calls them
        without ``await`` (see streaming_provider.py docstring). Internally
        the WebSocket I/O is async — we schedule it onto the caller's event
        loop (captured in ``connect()`` via ``asyncio.get_running_loop()``)
        using ``asyncio.run_coroutine_threadsafe`` / ``loop.create_task`` so
        the sync wrappers stay non-blocking.
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
        self._death_reason: Optional[str] = None  # Mirrors SafeDeepgramSocket.death_reason
        self._reader_task: Optional[asyncio.Task] = None

        # Event loop captured during connect() — used by the sync wrappers to
        # schedule async WS work without blocking. transcribe.py runs in this
        # same loop so loop.create_task is safe; we use
        # run_coroutine_threadsafe defensively in case a caller invokes
        # send/finish from a different loop or a worker thread.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Wall-clock anchor — set when WS opens. ``audio_*_ms`` from the
        # server resets per item so we ignore them as absolute timestamps
        # and use this anchor + monotonic offsets instead.
        self._wall_clock_open: Optional[float] = None

        # Per-utterance wall-clock window: updated by speech_started /
        # speech_stopped and read by the .completed handler.
        self._current_utterance_start: Optional[float] = None
        self._current_utterance_end: Optional[float] = None

        # Previous utterance window — retained after emit so duplicate
        # ``.completed`` events for the same item (a known speaches race
        # under load) can reuse the real wall-clock timing instead of
        # collapsing to start==end. ``_prev_item_id`` + ``_prev_text``
        # guard against reusing the window for genuinely-new items that
        # happen to also be missing VAD — only true duplicates (same
        # item_id OR same text) get the reuse path.
        self._prev_utterance_start: Optional[float] = None
        self._prev_utterance_end: Optional[float] = None
        self._prev_item_id: Optional[str] = None
        self._prev_text: Optional[str] = None

        # Monotonic floor for emitted segments. Used as a fallback when
        # both wall-clock VAD events were missed and the event payload
        # carries no usable duration — better than 0 for downstream
        # speaker attribution.
        self._last_emitted_end: Optional[float] = None

        # Counters for observability + tests.
        self._segments_emitted = 0
        self._errors_seen = 0
        self._hallucinations_filtered = 0

    # ----- public API --------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self.provider_name_value

    @property
    def is_connection_dead(self) -> bool:
        """True once the WS is closed or a fatal error has latched.

        One-way latch — mirrors ``SafeDeepgramSocket.is_connection_dead``.
        ``transcribe.py:flush_stt_buffer`` checks this before each ``send``
        and stops forwarding audio when it flips True.
        """
        return self._dead or self._closed

    @property
    def is_finished(self) -> bool:
        """True once ``finish()`` has been called. Idempotency guard for tests."""
        return self._closed

    @property
    def death_reason(self) -> Optional[str]:
        """Why the connection died, or None if still alive.

        Mirrors ``SafeDeepgramSocket.death_reason`` — transcribe.py reads this
        for the dead-connection log line at line 2438.
        """
        return self._death_reason

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

        # Capture the calling event loop so the sync public API
        # (send/finalize/finish) can schedule async WS work onto it without
        # blocking. transcribe.py's WS handler runs in this same loop.
        self._loop = asyncio.get_running_loop()

        await self._send_session_update()

        # Spawn the background reader. We do NOT await it; ``send`` /
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

    def send(self, pcm_bytes: bytes) -> bool:
        """Forward a PCM16LE chunk as an ``input_audio_buffer.append`` event.

        SYNC by contract (mirrors ``SafeDeepgramSocket.send``) — schedules the
        async WS send onto the captured event loop and returns immediately.

        Returns ``True`` if the send was scheduled, ``False`` if the session
        is dead/closed or has not been connected yet. transcribe.py ignores
        the return value, but unit tests assert on it.

        If a VAD gate is attached, audio is routed through it first — same
        pattern the Deepgram path uses (silence is gated out, finalize
        signals are absorbed because server VAD owns turn closing on the
        Realtime side).
        """
        if self._dead or self._closed or self._ws is None or self._loop is None:
            return False

        # Allow callers to register their own activity gate (same shape as
        # ``connect_to_deepgram_with_backoff``'s ``is_active`` parameter).
        if self._is_active is not None and not self._is_active():
            # Activity gate said skip — not a dead-connection condition; we
            # report success so the caller doesn't latch us as dead.
            return True

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
            return True

        # Schedule the async send fire-and-forget. We use
        # run_coroutine_threadsafe so the wrapper is safe whether the caller
        # is on self._loop or on a worker thread; either way we return
        # without blocking the caller.
        try:
            asyncio.run_coroutine_threadsafe(self._async_send(audio_to_send), self._loop)
        except RuntimeError as e:
            # Loop closed mid-flight — latch as dead so transcribe.py stops sending.
            if self._death_reason is None:
                self._death_reason = f'send schedule failed: {e}'
            self._dead = True
            return False
        return True

    def finalize(self) -> None:
        """NO-OP on the Realtime path.

        SYNC by contract (mirrors ``SafeDeepgramSocket.finalize``). Manual
        ``input_audio_buffer.commit`` triggers a server-side AssertionError
        (T1 reprobe). Server VAD handles turn closing based on
        ``silence_duration_ms=550`` (hardcoded build-level default — silently
        ignored override). We intentionally do NOT send a commit event here.
        """
        return None

    def finish(self) -> None:
        """Close the WS and cancel the reader task. SYNC; idempotent.

        Schedules the async close onto the captured event loop fire-and-forget
        (transcribe.py is shutting down and cannot await), and latches
        ``is_connection_dead`` so any racing ``send`` call returns False.
        """
        if self._closed:
            return
        self._closed = True
        if self._death_reason is None:
            self._death_reason = 'finish() called'

        if self._loop is None:
            # connect() never completed — nothing to schedule.
            self._reader_task = None
            self._ws = None
            return

        try:
            asyncio.run_coroutine_threadsafe(self._async_finish(), self._loop)
        except RuntimeError:
            # Loop already closed — best-effort; nothing else we can do
            # safely from a sync context.
            logger.debug('Realtime WS finish skipped: event loop already closed', exc_info=True)

    # ----- internal helpers --------------------------------------------

    async def _async_send(self, audio_bytes: bytes) -> None:
        """Async worker invoked by the sync ``send`` wrapper.

        Encodes the PCM chunk + dispatches the ``input_audio_buffer.append``
        event over the WS. Latches ``_dead`` + records ``_death_reason`` on
        any send failure so subsequent sync ``send`` calls short-circuit
        to ``False``.
        """
        if self._dead or self._closed or self._ws is None:
            return
        try:
            encoded = base64.b64encode(audio_bytes).decode('ascii')
            event = {'type': 'input_audio_buffer.append', 'audio': encoded}
            await self._ws.send(json.dumps(event))
        except Exception as e:
            if self._death_reason is None:
                self._death_reason = f'send {type(e).__name__}: {e}'
            logger.warning('Realtime WS send failed (%s: %s); marking dead', type(e).__name__, e)
            self._dead = True

    async def _async_finish(self) -> None:
        """Async worker invoked by the sync ``finish`` wrapper.

        Cancels the reader task and closes the WS. Best-effort: never raises
        because the sync caller (transcribe.py shutdown path) has nowhere to
        catch.
        """
        reader_task = self._reader_task
        self._reader_task = None
        if reader_task is not None:
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass

        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                logger.debug('Error closing Realtime WS', exc_info=True)

    async def _ws_send_json(self, payload: dict) -> None:
        """Send a JSON event over the WS, latching the connection on failure.

        Used for control-plane events (session.update) that still run from
        within ``connect()``'s async context.
        """
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            if self._death_reason is None:
                self._death_reason = f'control-plane send {type(e).__name__}: {e}'
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
            if self._death_reason is None:
                self._death_reason = f'reader terminated {type(e).__name__}: {e}'
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
            err = event.get('error', {})
            msg = err.get('message', '')
            err_type = err.get('type', '')
            # Benign known speaches races we explicitly tolerate:
            #   * "Error adding item: ... already exists" — speaches auto-creates
            #     items via VAD; under load it occasionally tries to add the
            #     same id twice on its own side. We never send
            #     conversation.item.create from the client (verified by grep),
            #     so this is purely server-side and harmless to our pipeline.
            #     Demote to debug so it stops alarming operators and don't
            #     increment _errors_seen (keeps the counter meaningful).
            if err_type == 'invalid_request_error' and 'already exists' in msg:
                logger.debug('Realtime API benign duplicate-item server race: %s', msg)
                return
            self._errors_seen += 1
            logger.warning('Realtime API error event type=%s message=%s', err_type, msg)
            # Only mark dead on connection-fatal errors. Benign
            # ``InternalServerError`` after a transcript turn does NOT close
            # the WS on the v1.2-patched build, so we leave the connection
            # alive and let the next .completed fire normally.
            if 'connection' in msg.lower() or 'closed' in msg.lower():
                if self._death_reason is None:
                    self._death_reason = f'server error: {err_type}: {msg}'
                self._dead = True
            return

        logger.debug('Unhandled Realtime event type=%s', etype)

    async def _emit_segment_from_completed(self, event: dict) -> None:
        """Build one ``StreamingSegment`` from a ``.completed`` event.

        Times use the wall-clock anchor + per-utterance start/stop captured
        from the matching ``speech_started`` / ``speech_stopped`` events.
        When those VAD events are missed (rare-but-real on speaches under
        load — observed in conv 3deec0e8-87cf-48de-8ece-e6ab0a85f112 for
        50% of segments) we apply a layered fallback so we never emit a
        collapsed ``start==end`` stamp.

        Also drops known-residue Whisper hallucinations BEFORE invoking
        ``stream_transcript`` — see ``_HALLUCINATION_PHRASES``.
        """
        text = (event.get('transcript') or '').strip()
        if not text:
            return

        # Drop known Whisper YouTube-residue hallucinations before any
        # downstream call. Still reset the per-utterance window so the
        # next real item starts clean.
        if _is_whisper_hallucination(text):
            self._hallucinations_filtered += 1
            logger.debug(
                'Realtime: dropping known-residue hallucination text=%r (normalized=%r)',
                text,
                _normalize_phrase(text),
            )
            # Cycle utterance windows so subsequent emits don't reuse
            # stamps that may have been intended for this dropped item.
            if self._current_utterance_start is not None or self._current_utterance_end is not None:
                self._prev_utterance_start = self._current_utterance_start
                self._prev_utterance_end = self._current_utterance_end
                self._current_utterance_start = None
                self._current_utterance_end = None
            return

        now = time.time()
        anchor = self._wall_clock_open if self._wall_clock_open is not None else now
        now_offset = max(0.0, now - anchor)

        # Per-item audio_*_ms reset per item (T3 reprobe — cannot be used
        # as absolute time), but they DO encode the duration of THIS turn.
        # Use that to back-derive missing endpoints.
        audio_start_ms = event.get('audio_start_ms')
        audio_end_ms = event.get('audio_end_ms')
        item_duration: Optional[float] = None
        if (
            isinstance(audio_start_ms, (int, float))
            and isinstance(audio_end_ms, (int, float))
            and audio_end_ms > audio_start_ms
        ):
            item_duration = (audio_end_ms - audio_start_ms) / 1000.0

        start = self._current_utterance_start
        end = self._current_utterance_end

        if start is None and end is None:
            # Both VAD events missed (or this is a duplicate ``.completed``
            # for an already-finalized item; the first emit reset the
            # window). Only reuse the previous window when this looks like
            # a true duplicate — same item_id OR identical text — so that
            # genuinely-new items missing VAD still get fresh wall-clock
            # anchoring instead of being clamped to the prior emit.
            event_item_id = event.get('item_id')
            is_duplicate = (
                self._prev_utterance_start is not None
                and self._prev_utterance_end is not None
                and (
                    (event_item_id is not None and event_item_id == self._prev_item_id)
                    or (self._prev_text is not None and text == self._prev_text)
                )
            )
            if is_duplicate:
                start = self._prev_utterance_start
                end = self._prev_utterance_end
            else:
                # True cold start (e.g. very first segment of session).
                # Anchor end at now, back-derive start.
                end = now_offset
                if item_duration is not None:
                    start = max(0.0, end - item_duration)
                elif self._last_emitted_end is not None:
                    # Use monotonic floor: previous emit's end becomes
                    # this segment's start, clamped to keep end > start.
                    start = min(self._last_emitted_end, end)
                    if start >= end:
                        start = max(0.0, end - 0.1)
                else:
                    start = max(0.0, end - 0.1)
        elif end is None:
            # Only ``speech_started`` fired.
            if item_duration is not None:
                end = start + item_duration
            else:
                end = max(start, now_offset)
        elif start is None:
            # Only ``speech_stopped`` fired.
            if item_duration is not None:
                start = max(0.0, end - item_duration)
            else:
                start = max(0.0, end - 0.1)

        # Guarantee non-collapsed monotonic segment. 100ms placeholder is
        # small enough to be ignored by Sortformer reassembly but big
        # enough that downstream attribution doesn't flag the segment as
        # anomalous.
        if end <= start:
            end = start + 0.1

        segment = StreamingSegment(
            speaker='SPEAKER_0',
            start=float(start),
            end=float(end),
            text=text,
            is_user=False,
            person_id=None,
        )

        # Cycle utterance windows: stash this item's window so a duplicate
        # ``.completed`` (server re-emit under load) can reuse it instead
        # of collapsing. Also remember item_id + text so the reuse path
        # only fires for actual duplicates.
        self._prev_utterance_start = start
        self._prev_utterance_end = end
        self._prev_item_id = event.get('item_id')
        self._prev_text = text
        self._current_utterance_start = None
        self._current_utterance_end = None

        self._segments_emitted += 1
        self._last_emitted_end = float(end)

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
            # finish() is now sync (fire-and-forget) — see SafeDeepgramSocket
            # contract in streaming_provider.py.
            try:
                session.finish()
            except Exception:
                pass
            return None
        return session


__all__ = [
    'RealtimeApiStreamingProvider',
    'RealtimeStreamingSession',
    'DEFAULT_HALLUCINATION_PHRASES',
    '_normalize_phrase',
    '_load_hallucination_phrases',
    '_is_whisper_hallucination',
]
