"""Unit tests for the STT streaming provider ABC + RealtimeApiStreamingProvider.

These are pure unit tests — no network, no rtx6000 dependency. The Realtime WS
is mocked at the ``websockets.connect`` boundary so we can drive arbitrary
event sequences through ``RealtimeStreamingSession._handle_event`` /
``_event_loop`` and assert on the segments emitted via the
``stream_transcript`` callback.

Integration tests against the live speaches endpoint live in
``tests/integration/test_realtime_provider_live.py``.
"""

import asyncio
import base64
import json
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.stt.streaming_provider import (
    StreamingSegment,
    StreamingSessionConfig,
    StreamingTranscriptSession,
    STTStreamingProvider,
)
from utils.stt.realtime_provider import (
    RealtimeApiStreamingProvider,
    RealtimeStreamingSession,
)


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------


def test_streaming_session_abc_cannot_instantiate():
    """The ABC must not be directly instantiable."""
    with pytest.raises(TypeError):
        StreamingTranscriptSession()  # type: ignore[abstract]


def test_streaming_provider_abc_cannot_instantiate():
    """The provider ABC must not be directly instantiable."""
    with pytest.raises(TypeError):
        STTStreamingProvider()  # type: ignore[abstract]


def test_streaming_segment_shape():
    """StreamingSegment dataclass round-trips through ``to_dict`` and matches
    the dict shape ``routers/transcribe.py`` consumes today."""
    seg = StreamingSegment(
        speaker='SPEAKER_0',
        start=1.25,
        end=2.5,
        text='hello world',
    )
    d = seg.to_dict()
    assert d == {
        'speaker': 'SPEAKER_0',
        'start': 1.25,
        'end': 2.5,
        'text': 'hello world',
        'is_user': False,
        'person_id': None,
    }


def test_streaming_segment_default_speaker_and_optional_person_id():
    """is_user defaults to False; person_id defaults to None."""
    seg = StreamingSegment(speaker='SPEAKER_2', start=0.0, end=1.0, text='hi')
    assert seg.is_user is False
    assert seg.person_id is None
    seg2 = StreamingSegment(
        speaker='SPEAKER_0',
        start=0.0,
        end=1.0,
        text='hi',
        is_user=True,
        person_id='person-abc',
    )
    assert seg2.is_user is True
    assert seg2.person_id == 'person-abc'


# ---------------------------------------------------------------------------
# Fixture helpers — fake WS + collected segments
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """Minimal stand-in for ``websockets.WebSocketClientProtocol``.

    Lets tests:
      - Drive events through the reader by pushing onto ``incoming``.
      - Capture all client-sent frames in ``sent`` for assertions.
      - Terminate the reader cleanly by setting ``stop``.
    """

    def __init__(self):
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: List[str] = []
        self.closed = False
        self.stop = False

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.closed = True
        self.stop = True
        # Unblock any pending reader.
        await self.incoming.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.stop:
            raise StopAsyncIteration
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item


def make_session(stream_transcript, **kwargs) -> RealtimeStreamingSession:
    """Construct a session without going through ``connect`` (no real WS)."""
    cfg = kwargs.pop('config', None) or StreamingSessionConfig()
    session = RealtimeStreamingSession(
        stream_transcript=stream_transcript,
        language=kwargs.pop('language', 'en'),
        sample_rate=kwargs.pop('sample_rate', 16000),
        channels=kwargs.pop('channels', 1),
        config=cfg,
    )
    return session


# ---------------------------------------------------------------------------
# Event translation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_realtime_provider_translates_completed_event_to_segment():
    """A ``.completed`` event becomes one segment, dispatched via stream_transcript."""
    captured: List[List[dict]] = []

    def stream_transcript(segments):
        captured.append(segments)

    session = make_session(stream_transcript)
    # Set the wall-clock anchor manually (normally set by ``connect``).
    session._wall_clock_open = 100.0

    # Simulate the natural event sequence: speech_started → speech_stopped → committed → .completed
    await session._handle_event({'type': 'input_audio_buffer.speech_started', 'audio_start_ms': 0, 'item_id': 'item_1'})
    await session._handle_event(
        {'type': 'input_audio_buffer.speech_stopped', 'audio_end_ms': 1500, 'item_id': 'item_1'}
    )
    await session._handle_event({'type': 'input_audio_buffer.committed', 'item_id': 'item_1'})
    await session._handle_event(
        {
            'type': 'conversation.item.input_audio_transcription.completed',
            'item_id': 'item_1',
            'transcript': 'So our LLM model.',
        }
    )

    assert len(captured) == 1
    segs = captured[0]
    assert len(segs) == 1
    seg = segs[0]
    assert seg['text'] == 'So our LLM model.'
    assert seg['speaker'] == 'SPEAKER_0'
    assert seg['is_user'] is False
    assert seg['person_id'] is None
    # Times are non-negative floats (wall-clock-anchored on our side, not from
    # audio_*_ms which resets per item).
    assert isinstance(seg['start'], float)
    assert isinstance(seg['end'], float)
    assert seg['start'] >= 0.0
    assert seg['end'] >= seg['start']


@pytest.mark.asyncio
async def test_realtime_provider_ignores_delta_events():
    """Defensive: server doesn't emit .delta on the current build, but if it
    ever does the adapter must not crash and must not emit a segment."""
    captured: List[List[dict]] = []

    def stream_transcript(segments):
        captured.append(segments)

    session = make_session(stream_transcript)
    session._wall_clock_open = 100.0

    await session._handle_event(
        {
            'type': 'conversation.item.input_audio_transcription.delta',
            'item_id': 'item_x',
            'delta': 'partial text',
        }
    )

    assert captured == []


@pytest.mark.asyncio
async def test_realtime_provider_handles_error_event_without_crash():
    """An ``error`` event is logged and counted but does NOT crash the loop
    and does NOT mark the connection dead for benign errors."""
    captured: List[List[dict]] = []

    def stream_transcript(segments):
        captured.append(segments)

    session = make_session(stream_transcript)
    session._wall_clock_open = 100.0

    await session._handle_event(
        {
            'type': 'error',
            'error': {
                'message': 'InternalServerError: Internal Server Error',
                'type': 'server_error',
            },
        }
    )

    assert session._errors_seen == 1
    # Benign InternalServerError must NOT mark the connection dead — v1.2
    # reprobe confirmed these no longer fire post-patch, but if they do
    # we leave the WS alive.
    assert session.is_alive() is True
    assert captured == []


@pytest.mark.asyncio
async def test_realtime_provider_handles_connection_error_marks_dead():
    """Connection-fatal error events must latch the session as dead."""

    def stream_transcript(segments):
        pass

    session = make_session(stream_transcript)
    session._wall_clock_open = 100.0
    assert session.is_alive() is True

    await session._handle_event(
        {
            'type': 'error',
            'error': {
                'message': 'WebSocket connection closed unexpectedly',
                'type': 'server_error',
            },
        }
    )

    assert session.is_alive() is False


@pytest.mark.asyncio
async def test_realtime_provider_send_audio_uses_base64():
    """send_audio emits ``input_audio_buffer.append`` events carrying
    base64-encoded PCM payloads."""

    def stream_transcript(segments):
        pass

    session = make_session(stream_transcript)
    fake_ws = FakeWebSocket()
    session._ws = fake_ws

    pcm = b'\x00\x01\x02\x03' * 100  # 400 bytes of fake PCM16
    await session.send_audio(pcm)

    assert len(fake_ws.sent) == 1
    event = json.loads(fake_ws.sent[0])
    assert event['type'] == 'input_audio_buffer.append'
    decoded = base64.b64decode(event['audio'])
    assert decoded == pcm


@pytest.mark.asyncio
async def test_realtime_provider_send_audio_noop_when_dead():
    """Once latched dead, send_audio is a no-op (matches SafeDeepgramSocket)."""

    def stream_transcript(segments):
        pass

    session = make_session(stream_transcript)
    fake_ws = FakeWebSocket()
    session._ws = fake_ws
    session._dead = True

    await session.send_audio(b'\x00' * 100)

    assert fake_ws.sent == []


@pytest.mark.asyncio
async def test_realtime_provider_finalize_is_noop():
    """finalize() must NOT send input_audio_buffer.commit — server
    AssertionErrors on manual commits (T1 reprobe)."""

    def stream_transcript(segments):
        pass

    session = make_session(stream_transcript)
    fake_ws = FakeWebSocket()
    session._ws = fake_ws

    await session.finalize()

    # No frames sent — finalize is intentionally a no-op on the Realtime
    # path because server VAD owns turn closing.
    assert fake_ws.sent == []


@pytest.mark.asyncio
async def test_realtime_provider_session_update_payload():
    """The session.update sent on connect carries the probe-verified working
    fields. Only ``temperature`` is actually applied by speaches today, but
    we send the others for documentation purposes."""

    def stream_transcript(segments):
        pass

    session = make_session(stream_transcript, language='en')
    fake_ws = FakeWebSocket()
    session._ws = fake_ws

    await session._send_session_update()

    assert len(fake_ws.sent) == 1
    event = json.loads(fake_ws.sent[0])
    assert event['type'] == 'session.update'
    s = event['session']
    assert s['modalities'] == ['text']
    assert s['temperature'] == 0.0
    iat = s['input_audio_transcription']
    # The transcription model is silently ignored by speaches today but we
    # still send the override so the wire log documents intent.
    assert iat['model'] == 'Systran/faster-whisper-large-v3'
    assert iat['language'] == 'en'


@pytest.mark.asyncio
async def test_realtime_provider_session_update_multi_language_pins_null():
    """``language='multi'`` should be sent as null so speaches' auto-detect
    runs (matches how Deepgram's ``multi`` is interpreted today)."""

    def stream_transcript(segments):
        pass

    session = make_session(stream_transcript, language='multi')
    fake_ws = FakeWebSocket()
    session._ws = fake_ws

    await session._send_session_update()

    event = json.loads(fake_ws.sent[0])
    assert event['session']['input_audio_transcription']['language'] is None


@pytest.mark.asyncio
async def test_realtime_provider_uses_vad_gate_when_provided():
    """When a VAD gate is attached, send_audio routes through it. Silence
    (empty ``audio_to_send``) is dropped; speech bytes go to the WS."""

    def stream_transcript(segments):
        pass

    gate = MagicMock()
    cfg = StreamingSessionConfig(vad_gate=gate)
    session = make_session(stream_transcript, config=cfg)
    fake_ws = FakeWebSocket()
    session._ws = fake_ws

    # First call: gate returns silence (no bytes to forward).
    gate.process_audio.return_value = MagicMock(audio_to_send=b'', should_finalize=False)
    await session.send_audio(b'\x00' * 200)
    assert fake_ws.sent == []

    # Second call: gate returns speech bytes — those should be forwarded.
    speech = b'\x10\x20' * 50
    gate.process_audio.return_value = MagicMock(audio_to_send=speech, should_finalize=True)
    await session.send_audio(speech)
    assert len(fake_ws.sent) == 1
    event = json.loads(fake_ws.sent[0])
    assert event['type'] == 'input_audio_buffer.append'
    assert base64.b64decode(event['audio']) == speech


@pytest.mark.asyncio
async def test_realtime_provider_send_audio_respects_is_active():
    """When ``is_active`` returns False, send_audio is a no-op."""

    def stream_transcript(segments):
        pass

    cfg = StreamingSessionConfig(is_active=lambda: False)
    session = make_session(stream_transcript, config=cfg)
    fake_ws = FakeWebSocket()
    session._ws = fake_ws

    await session.send_audio(b'\x00' * 64)
    assert fake_ws.sent == []


@pytest.mark.asyncio
async def test_realtime_provider_finish_is_idempotent():
    """finish() can be called multiple times safely."""

    def stream_transcript(segments):
        pass

    session = make_session(stream_transcript)
    fake_ws = FakeWebSocket()
    session._ws = fake_ws

    await session.finish()
    await session.finish()  # second call must not raise
    assert fake_ws.closed is True


@pytest.mark.asyncio
async def test_realtime_provider_open_session_returns_none_on_connect_failure():
    """The provider mirrors connect_to_deepgram_with_backoff's contract: a
    failed connect returns ``None`` rather than raising up to the caller."""

    def stream_transcript(segments):
        pass

    async def failing_connect(*args, **kwargs):
        raise ConnectionError('boom')

    with patch('utils.stt.realtime_provider.websockets.connect', new=failing_connect):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test'}, clear=False):
            provider = RealtimeApiStreamingProvider()
            result = await provider.open_session(stream_transcript, 'en', 16000, 1)

    assert result is None


@pytest.mark.asyncio
async def test_realtime_provider_open_session_raises_without_api_key(monkeypatch):
    """Without OPENAI_API_KEY the connect step raises (caught by
    open_session, surfaced as ``None``)."""

    def stream_transcript(segments):
        pass

    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    provider = RealtimeApiStreamingProvider()
    result = await provider.open_session(stream_transcript, 'en', 16000, 1)
    assert result is None


@pytest.mark.asyncio
async def test_realtime_provider_audio_start_end_anchored_to_wall_clock(monkeypatch):
    """The .completed event's stamps must be wall-clock-anchored on our side.
    audio_*_ms from the server resets per item and cannot be trusted (T3
    reprobe)."""

    captured: List[List[dict]] = []

    def stream_transcript(segments):
        captured.append(segments)

    session = make_session(stream_transcript)

    # Pin wall clock to control the math.
    wall_time = [100.0]

    def fake_time():
        return wall_time[0]

    monkeypatch.setattr('utils.stt.realtime_provider.time.time', fake_time)
    session._wall_clock_open = 100.0

    # Simulate a turn starting 5s after WS open and ending 7s after open.
    wall_time[0] = 105.0
    await session._handle_event({'type': 'input_audio_buffer.speech_started', 'audio_start_ms': 0})
    wall_time[0] = 107.0
    await session._handle_event({'type': 'input_audio_buffer.speech_stopped', 'audio_end_ms': 0})
    wall_time[0] = 107.4
    await session._handle_event(
        {
            'type': 'conversation.item.input_audio_transcription.completed',
            'transcript': 'wall clock test',
        }
    )

    seg = captured[0][0]
    assert seg['start'] == pytest.approx(5.0)
    assert seg['end'] == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_realtime_provider_each_completed_emits_own_segment():
    """Server VAD chunks long utterances into multiple .completed events.
    Each one becomes its own segment — no reassembly in the adapter (T3
    reprobe). The Sortformer + downstream merge handles reassembly."""

    captured: List[List[dict]] = []

    def stream_transcript(segments):
        captured.append(segments)

    session = make_session(stream_transcript)
    session._wall_clock_open = 0.0

    for i, text in enumerate(['chunk one', 'chunk two', 'chunk three']):
        await session._handle_event({'type': 'input_audio_buffer.speech_started', 'audio_start_ms': 0})
        await session._handle_event({'type': 'input_audio_buffer.speech_stopped', 'audio_end_ms': 0})
        await session._handle_event(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'transcript': text,
            }
        )

    assert len(captured) == 3
    assert [c[0]['text'] for c in captured] == ['chunk one', 'chunk two', 'chunk three']


@pytest.mark.asyncio
async def test_realtime_provider_event_loop_drives_segments_through_ws(monkeypatch):
    """End-to-end internal test: drive the WS reader with a real
    ``async for`` over the fake WS and verify segments come through."""

    captured: List[List[dict]] = []

    def stream_transcript(segments):
        captured.append(segments)

    session = make_session(stream_transcript)
    session._wall_clock_open = 0.0
    fake_ws = FakeWebSocket()
    session._ws = fake_ws

    # Push the canonical event sequence onto the fake WS.
    await fake_ws.incoming.put(json.dumps({'type': 'session.created'}))
    await fake_ws.incoming.put(json.dumps({'type': 'input_audio_buffer.speech_started', 'audio_start_ms': 0}))
    await fake_ws.incoming.put(json.dumps({'type': 'input_audio_buffer.speech_stopped', 'audio_end_ms': 0}))
    await fake_ws.incoming.put(
        json.dumps(
            {
                'type': 'conversation.item.input_audio_transcription.completed',
                'transcript': 'realtime smoke',
            }
        )
    )
    await fake_ws.incoming.put(None)  # terminator

    # Run the reader to completion.
    await session._event_loop()

    assert len(captured) == 1
    assert captured[0][0]['text'] == 'realtime smoke'


@pytest.mark.asyncio
async def test_realtime_provider_open_session_passes_extra_headers(monkeypatch):
    """Verify the WS connect call uses extra_headers (websockets 12.0 kwarg)
    with a Bearer auth header and ping_interval=None."""

    def stream_transcript(segments):
        pass

    monkeypatch.setenv('OPENAI_API_KEY', 'sk-unit-test')

    captured_kwargs = {}

    async def fake_connect(url, **kwargs):
        captured_kwargs['url'] = url
        captured_kwargs.update(kwargs)
        return FakeWebSocket()

    with patch('utils.stt.realtime_provider.websockets.connect', new=fake_connect):
        provider = RealtimeApiStreamingProvider()
        session = await provider.open_session(stream_transcript, 'en', 16000, 1)
        # Cleanup so the background reader task doesn't leak.
        if session is not None:
            await session.finish()

    assert session is not None
    assert 'extra_headers' in captured_kwargs
    headers = captured_kwargs['extra_headers']
    auth = headers['Authorization']
    assert auth.startswith('Bearer ')
    assert captured_kwargs.get('ping_interval') is None
    # WS URL must include ?model= so LiteLLM doesn't 403.
    assert 'model=' in captured_kwargs['url']


def test_realtime_provider_name():
    """Provider name is the stable env-var value operators flip to."""
    provider = RealtimeApiStreamingProvider()
    assert provider.provider_name == 'realtime-local'
    session = RealtimeStreamingSession(
        stream_transcript=lambda segs: None,
        language='en',
        sample_rate=16000,
        channels=1,
    )
    assert session.provider_name == 'realtime-local'
