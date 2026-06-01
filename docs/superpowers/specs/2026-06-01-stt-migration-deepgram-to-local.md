# STT Migration: Deepgram → rtx6000 local (Whisper / Voxtral)

**Date:** 2026-06-01
**Last modified:** 2026-06-01
**Status:** Design draft. Scout passes complete (rtx6000 HTTP probe + WS deep-probe + backend integration map). Awaiting Joe's answers to the open questions at the bottom before Phase A implementation kicks off.
**Updated 2026-06-01 (am):** rtx6000 exposed an OpenAI Realtime API WebSocket at `ws://10.0.60.48:4000/v1/realtime`; streaming migration plan revised.
**Updated 2026-06-01 (pm):** Deep-probe of the speaches Realtime endpoint surfaced hard blockers — server VAD fires exactly once per WS connection, `create_response=false` is silently ignored, no interim partials, no word-level timestamps. The Realtime endpoint is **not yet usable** as a Deepgram streaming replacement. Streaming cutover deferred until speaches is patched; in the interim the plan reverts to keep Deepgram for streaming and consider a chunked-batch shim if needed. See "Discovery v1.1" + "Realtime API deep-probe (2026-06-01 pm)".
**Owner:** Joe Portman (@jlportman3)
**Scope:** Backend STT layer only. Live streaming STT (BLE pendant + desktop sockets) and batch / pre-recorded STT (post-conversation re-transcribe, sync v2, voice messages, speech-profile sample verification, scripts). Touches `backend/utils/stt/*`, `backend/routers/{transcribe,sync,users,speech_profile}.py`, `backend/utils/{chat,speaker_sample,byok,subscription}.py`, `backend/utils/conversations/postprocess_conversation.py`, `backend/migrations/006_*`, `backend/pusher/requirements.txt`, `.env.template`. Out of scope: TTS (separate Kokoro/Orpheus spec), diarization (already local at `10.0.60.48:8094`), LLM (already on LiteLLM `10.0.60.48:4000`), language-pack expansion.

---

## Problem

Omi's STT layer is the single largest remaining third-party-cloud dependency in the listen pipeline. Joe's stated long-term goal is **fully local — zero third-party clouds** (see `~/.claude/projects/-mypool-home-baron-omi/memory/project_fully_local_roadmap.md`). Today:

- **Live streaming** (BLE pendant via `routers/transcribe.py`, desktop PTT, multi-channel phone audio): hard-wired to Deepgram Nova-3 over the official Deepgram Python SDK WebSocket.
- **Batch / pre-recorded**: already half-migrated. `STT_BATCH_BACKEND=whisper` is set in `.env` and `utils/stt/pre_recorded.py` short-circuits all six production callers to a local `whisper-large-v3` POST against LiteLLM. Cloud Deepgram is still the wired default in the template but is not exercised in this deployment.
- **Cost / privacy**: every word the pendant captures (including ambient conversation in the same room) is streamed to a U.S. SaaS vendor. Even with BYOK gating per-user, the operator's own conversations are billed and logged on Deepgram's side.
- **Capability**: rtx6000 already hosts both `whisper-large-v3` and `voxtral-mini` behind LiteLLM (`http://10.0.60.48:4000/v1/audio/transcriptions`), plus `sortformer` diarization at `10.0.60.48:8094/v1/audio/diarization`. The compute is paid for; we are not using it.

The migration therefore has two distinct difficulty profiles:

1. **Batch is easy** — already migrated, behind a feature flag, callers untouched. We just need to formalize the env var, validate against every caller, and eventually delete the dead Deepgram branch.
2. **Streaming is hard** — Deepgram emits interim transcripts every ~200ms over a long-lived WebSocket and ships native speaker diarization, vocabulary biasing, smart formatting, and a finalize-on-VAD flush. Whisper and Voxtral are batch decoders by nature; neither rtx6000 instance exposes a WS realtime endpoint today. Replacing the streaming path requires either standing up a streaming-shaped server on rtx6000 (separate session, separate agent) or implementing a chunked-batch shim in pusher/transcribe and accepting higher partial-latency.

This spec covers the design choices, the adapter shape, env toggles, rollout phases, and risks. It does **not** start implementation — it ends with a list of decisions Joe needs to make before Phase A code lands.

## Goals

1. Single env var (`STT_STREAMING_PROVIDER`, `STT_BATCH_PROVIDER`) cleanly switches each use case between `deepgram`, `whisper-local`, `voxtral-local` with no code path edits.
2. Default behavior unchanged on a fresh deploy (`deepgram` + `deepgram`) until the operator explicitly opts in.
3. Joe's Jarvis test account can run end-to-end on local STT (streaming + batch) for at least 7 continuous days before any production cutover.
4. BYOK fast-path preserved: users with `x-byok-deepgram` still hit cloud Deepgram and bypass operator credit accounting, regardless of the operator's default provider.
5. Speaker labels remain compatible with the memory-attribution pipeline (`SPEAKER_{n}` ints, see `2026-05-31-memory-attribution-and-triage-design.md`) — diarization continues to come from the Sortformer service on `10.0.60.48:8094`, not from the STT provider.
6. Adapter contract is documented well enough that a future provider (`speaches`, `faster-whisper-server`, NVIDIA Riva, etc.) drops in by implementing one base class.

## Non-goals

- A custom Whisper / Voxtral inference server on rtx6000. That's the rtx6000 agent's job; this spec assumes whatever rtx6000 exposes via LiteLLM and asks for a streaming WS endpoint as a separate work item.
- TTS migration. Kokoro / Orpheus / ElevenLabs is a separate spec (see `~/.claude/projects/-mypool-home-baron-omi/memory/project_elevenlabs_and_local_tts.md`).
- Replacing the Sortformer diarizer. It's already local, already integrated.
- Adding new STT languages beyond what Deepgram Nova-3 supports today. Whisper supports a superset; the existing `deepgram_nova3_multi_languages` set stays as the supported surface in v1.
- Mobile-app UI for switching providers. Operator-only env var.

## Current state — Deepgram integration map

Synthesized from the Phase-1 scout. Detailed function/line references are in the scout JSON; this section is the architectural summary the rest of the spec builds against.

### Streaming path

```
BLE pendant / desktop PTT
        │  (PCM16 LE @ 16kHz, optionally multi-channel)
        ▼
routers/transcribe.py:transcribe_socket  (per-WebSocket connection handler)
        │  process_audio_dg(stream_transcript, lang, sr, ch, model='nova-3', keywords, vad_gate, is_active)
        ▼
utils/stt/streaming.py:process_audio_dg  (async factory)
        │  connect_to_deepgram_with_backoff → connect_to_deepgram
        │      → DeepgramClient(...).listen.websocket.v('1')
        │      → start(LiveOptions(punctuate, diarize, smart_format,
        │                          endpointing=300, interim_results=False,
        │                          encoding='linear16', sample_rate, channels,
        │                          model='nova-3', keyterm|keywords=<vocab>))
        ▼
GatedDeepgramSocket  (VAD wrapper — filters silence, calls finalize() on silence transitions)
        │  └─ SafeDeepgramSocket  (5s keepalive thread, dead-connection latch)
        │      └─ raw DG websocket
        ▼
DG event handlers (on_message)
        │  result.channel.alternatives[0].words = [{word, start, end, speaker:int, punctuated_word, ...}]
        ▼
stream_transcript([{speaker:'SPEAKER_{n}', start, end, text, is_user:False, person_id:None}, ...])
        │
        ▼
realtime_segment_buffers (deque)  →  speaker_identification_task (TitaNet/Sortformer)
                                    →  conversation lifecycle  →  Firestore
```

Critical surfaces the streaming replacement must satisfy:
- `send(bytes)` + `finalize()` + `finish()` socket interface.
- Word-level segment callback shape `{speaker, start, end, text, is_user, person_id}`.
- VAD-driven `finalize()` for early flush on speech→silence transitions.
- Keepalive heartbeat (Deepgram disconnects at 10s idle; we send every 5s).
- Per-request BYOK key swap via `_deepgram_client_for_request()`.
- Optional vocabulary biasing (`keyterm` / `keywords`, capped at 100 entries).

### Batch path (already migrated, behind `STT_BATCH_BACKEND=whisper`)

```
post-conv re-transcribe / sync v2 / voice msg / speech-profile sample
        │
        ▼
utils/stt/pre_recorded.py:deepgram_prerecorded[_from_bytes](audio, lang, model, keywords, ...)
        │
        ├─ if STT_BATCH_BACKEND == 'whisper':
        │       └─→ local_whisper_prerecorded[_from_bytes]
        │              ├─ POST {OPENAI_BASE_URL=http://10.0.60.48:4000/v1}/audio/transcriptions
        │              │       file=audio.wav  model=whisper-large-v3
        │              │       response_format=verbose_json  timestamp_granularities[]=word
        │              │       Authorization: Bearer ${OPENAI_API_KEY}
        │              │       → {text, language, words:[{word,start,end}]}
        │              ├─ POST {VOICE_EXTRAS_URL=http://10.0.60.48:8094}/v1/audio/diarization
        │              │       file=audio.wav  model=sortformer-stream
        │              │       → {segments:[{start,end,speaker:'speaker_N'}]}
        │              └─ _merge_words_with_speakers
        │                       → [{'timestamp':[start,end], 'speaker':'SPEAKER_XX', 'text':word}, ...]
        │
        └─ else: original Deepgram HTTP path (cloud) — currently dormant on this deployment
```

Six production callers all use the public DG-named shim:
- `utils/conversations/postprocess_conversation.py:82`
- `routers/sync.py:957`
- `utils/chat.py:82,86,133,146,197,271`
- `utils/speaker_sample.py:46`

### Pluggable design today

Partial. Only `STT_BATCH_BACKEND` exists as an env-toggled switch. The `STTService` enum in `utils/stt/streaming.py` has exactly one member (`STTService.deepgram`). There is **no** abstract base class, **no** factory beyond `get_stt_service_for_language(lang, multi_lang_enabled)`, and **no** `STT_STREAMING_BACKEND` env var. The `GatedDeepgramSocket` / `SafeDeepgramSocket` interface is the de-facto streaming-backend contract that a Whisper replacement must satisfy.

## Available local providers on rtx6000

From Phase-1 HTTP probes (gpufarm → 10.0.60.48). **The rtx6000 host is reachable on exactly two ports for STT-related work.**

### LiteLLM proxy — `http://10.0.60.48:4000`

OpenAI-shape proxy. Auth: `Authorization: Bearer ${OPENAI_API_KEY}` (the same key already wired in `backend/.env`).

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/v1/models` | GET | Lists `whisper-large-v3`, `voxtral-mini`, `kokoro-tts`, `f5-tts`, plus Qwen/Nemotron LLMs and the GPT/Claude passthroughs |
| `/v1/audio/transcriptions` | POST multipart | Batch STT, OpenAI Whisper API shape. `model`, `file`, `language`, `response_format=verbose_json`, `timestamp_granularities[]=word` |
| `/v1/audio/translations` | — | **404 — not available.** No machine translation. |
| `/v1/realtime` | — | **404 — no WebSocket realtime endpoint.** This is the key constraint forcing the streaming design. |
| `/v1/audio/transcriptions/stream` | — | **404 — no streaming variant of /v1/audio/transcriptions.** |
| `/v1/chat/completions`, `/v1/embeddings` | — | Already used by omi for LLM + text embeddings. |

Smoke tests on a 1-second silent mono 16kHz WAV (`/tmp/silence_1s.wav`, 32,044 bytes):
- `whisper-large-v3`: HTTP 200 in **283ms** wall, returns `{"text":"","language":"nn","duration":1.0,"segments":[]}`. Correct silent-input behavior.
- `voxtral-mini`: HTTP 200 in **1422ms** wall, **hallucinates a Spanish sentence** about a 2013 UN commission. Known Voxtral behavior on silence — this single result is enough to disqualify Voxtral as the default batch STT for production audio.

### voice-extras — `http://10.0.60.48:8094`

TitaNet + Sortformer. Already integrated by omi for diarization and speaker embeddings.

| Endpoint | Notes |
|----------|-------|
| `/health` | `{status:"ok", device:"cuda", models_loaded:{titanet:true, sortformer:true}}` |
| `/v1/embeddings` | TitaNet 192-d speaker embeddings (used by Layer 1 of the memory-attribution voiceprint bank) |
| `/v1/audio/diarization` | Sortformer; consumed by `pre_recorded.py:_post_sortformer` and `_merge_words_with_speakers` |
| `/v1/audio/transcriptions` | **Not present.** This service does not do STT. |

### speaches Realtime API (discovered 2026-06-01, deep-probed same day)

After the rtx6000 agent upgrade landed, a re-probe of LiteLLM found a working OpenAI Realtime API WebSocket. A follow-up deep-probe (5 runs, real 23s pendant audio) confirmed the endpoint is reachable and transcription quality is good — **but the current speaches build is not yet usable as a streaming Deepgram replacement.** See "Realtime API deep-probe (2026-06-01 pm)" below for full details. Summary:

- **WebSocket endpoint**: `ws://10.0.60.48:4000/v1/realtime` (alias also exposed at `ws://10.0.60.48:4000/openai/v1/realtime`)
- **Protocol**: implements the public OpenAI Realtime API spec (`session.created` / `session.update` / `input_audio_buffer.append` / `conversation.item.input_audio_transcription.*` event grammar)
- **Backed by**: `speaches` — open-source Realtime API implementation wrapping faster-whisper for STT and Kokoro for TTS
- **Backing models** (from `session.created` event):
  - `input_audio_transcription.model` = `Systran/faster-distil-whisper-small.en` (English-only default — switch to `Systran/faster-whisper-large-v3` via `session.update` for multi-lang; override IS accepted, verified)
  - `model` = `Systran/faster-whisper-large-v3` (the assistant-side model)
  - `speech_model` (TTS) = `speaches-ai/Kokoro-82M-v1.0-ONNX` (ignored for our STT-only use)
  - `voice` = `af_heart`
  - `turn_detection` = `server_vad`, `threshold=0.9`, `silence_duration_ms=550` — **probe-confirmed unchangeable** (echoed in `session.updated` but server keeps hardcoded defaults regardless of payload)
  - `modalities` = `["audio", "text"]` by default; setting to `["text"]` via `session.update` IS accepted
- **Auth**: `Authorization: Bearer ${OPENAI_API_KEY}` — same key already wired in `backend/.env` for LiteLLM
- **Endpoints registered**: `GET /v1/realtime` (WS upgrade), `POST /v1/realtime/calls`, `POST /v1/realtime/client_secrets`
- **Probe result**: `GET ws://10.0.60.48:4000/v1/realtime` with `Upgrade: websocket` → `101 Switching Protocols` (confirmed reachable from gpufarm)
- **WS URL MUST include `?model=<litellm-known-model>` query param** — without it LiteLLM returns 403. Not in spec docs. The working probe URL was `ws://10.0.60.48:4000/v1/realtime?intent=transcription&model=whisper-large-v3`.
- **`intent=transcription` query param**: accepted but is currently a no-op — the `session.created` event is identical to a default conversational session. speaches does not differentiate yet.
- **Transcription quality (probe-verified)**: faster-whisper-large-v3 on Joe's 23s pendant clip produced the exact transcript `"So our LLM model."` — clean, correctly recognized, no hallucinations. Final-transcript latency `~348ms` after `speech_stopped`. Quality is not the blocker.
- **DOES NOT emit interim partials**: probe confirmed only `conversation.item.input_audio_transcription.completed` is emitted (no `.delta` events). Final-only stream. Hard regression vs Deepgram's ~250ms partials, but recall current omi has `interim_results=False` so the in-app UX impact is zero (we already only ship finals).
- **DOES NOT emit word-level timestamps**: the `.completed` event carries only a flat transcript string. No `word_offsets`, no per-word confidence, no segment boundaries. Speaker attribution must come from Sortformer windowing alone, not word-level alignment.
- **DOES NOT diarize**: speaker IDs still come from our local Sortformer on `:8094` (unchanged). The Realtime API only emits text events; we will continue to run Sortformer in parallel for speaker labels.
- **BLOCKERS preventing immediate adoption** (full list under "Realtime API deep-probe" section):
  - Server VAD fires exactly **once per WS connection**. After the first `speech_started` / `speech_stopped` pair, no further VAD events emit even when more audio is streamed. Single-utterance-per-session is unworkable for Omi's long conversations.
  - `turn_detection.create_response=false` is silently ignored — every transcript turn triggers a failed assistant-response attempt that emits an `error` event ~1s later. This is the likely root cause of the single-VAD bug.
  - `turn_detection: null` (the spec-defined way to disable server VAD) is rejected with a pydantic validation error.
  - Keepalive is broken — server doesn't respond to client pongs; the connection dies with code 1011 after ~30s if you enable client ping_interval. Must disable client pings or implement app-level keepalive via no-op append events.
- **Net assessment**: not yet usable as a streaming Deepgram replacement. Quality is good; control plane is broken. Either (a) patch speaches upstream to honor `create_response=false` and re-arm VAD across turns (estimated 1-2 day fix), (b) reconnect WS per utterance with client-side VAD (high overhead, ~5 reconnects/min/speaker), or (c) defer streaming cutover and keep Deepgram for streaming while batch stays on whisper-local. **Recommendation**: defer; pursue (a) in parallel with Phase A.

### What's **not** reachable

All other STT-candidate ports on rtx6000 are **closed** from gpufarm:
- 9000 (faster-whisper-server default)
- 8000-8099 except 8094 (speaches batch port, etc.)
- 5000-5050, 7860-7862, 11434, 3000-3001

The Realtime API on `:4000/v1/realtime` (LiteLLM-fronted) is the only streaming WebSocket exposed today. The standalone faster-whisper-server / speaches batch ports remain closed; we go through LiteLLM for everything.

## Provider choice per use case

### Batch / pre-recorded → `whisper-large-v3` via LiteLLM

Already implemented, already the default in `.env`. Six callers, all happy. Justification:
- **Latency**: 283ms for a 1s clip; production clips are 15-120s, scaling roughly linearly, well inside the operator's batch budget (post-conversation re-transcribe is async and can take 30s+).
- **Quality**: Whisper-large-v3 ranks higher than Nova-3 on librispeech-clean and on noisy real-world data per public benchmarks; for Joe's pendant audio specifically the prior memory note confirms Whisper-large-v3 is the canonical local STT.
- **Silent-input correctness**: returns empty text, no hallucinations. Voxtral fails this on the smoke test.
- **Language coverage**: superset of Nova-3 multi-language set; no language regressions.
- **Word-level timestamps**: `response_format=verbose_json, timestamp_granularities[]=word` returns the exact shape needed by `_merge_words_with_speakers`.

Voxtral-mini stays in the codebase as a selectable alternative for **future audio-Q&A features** (Voxtral is an audio-LLM that can answer questions about audio in one shot — different product surface than transcription). It is **not** the recommended batch STT provider today because of the silence-hallucination behavior.

### Streaming (live BLE / desktop PTT / multi-channel) → **Keep Deepgram; defer streaming cutover**

The 2026-06-01 morning discovery suggested the speaches Realtime API endpoint would unblock a clean streaming cutover. The same-day deep-probe (see "Realtime API deep-probe (2026-06-01 pm)") revealed that the current speaches build has four hard blockers that make it unusable as a Deepgram replacement today. Three paths now exist:

**Path A — Implement OpenAI Realtime API client targeting speaches on rtx6000 (BLOCKED until upstream fix).**
- Plan: implement `RealtimeApiStreamingProvider` mapping our existing `streaming.py` contract onto Realtime API events. Connect to `ws://10.0.60.48:4000/v1/realtime?model=whisper-large-v3`. Send `session.update` to force STT-only mode and large-v3 model.
- Pros: True streaming cutover with real latency parity to Deepgram. Cloud-zero. Reuses the existing OpenAI Bearer key. Transcription quality probe-confirmed clean (348ms final-transcript latency, accurate text).
- Cons: **Hard-blocked by speaches behavior today**: (1) server VAD fires exactly once per WS connection then dies; (2) `create_response=false` is silently ignored and every turn triggers an `error` event that corrupts the session; (3) `turn_detection: null` is rejected; (4) WS keepalive is broken. None of these are tunable client-side — they require patches to speaches.
- **Recommendation**: do **not** ship this in Phase B. Track an upstream issue with the rtx6000 agent to patch speaches (estimated 1-2 day fix). Once patched, Path A becomes the Phase B target. The adapter code can still be written speculatively in Phase A as `RealtimeApiStreamingProvider`, gated behind `STT_STREAMING_PROVIDER=realtime-local` (default off, will not exit alpha until speaches is fixed).

**Path B — Chunked-batch shim against `/v1/audio/transcriptions` (REINSTATED as contingency).**
- Originally the recommended fallback; obsoleted by the morning Realtime API discovery; reinstated as a contingency because the Realtime endpoint is blocked.
- Pros: relies only on the already-working batch endpoint; no upstream fixes needed; can be implemented and shipped without coordinating with rtx6000 agent.
- Cons: UX-degraded (chunked latency ~3-5s per window vs Deepgram's ~250ms partials); higher complexity than originally hoped because we now have to choose between the chunked shim and waiting for speaches to be patched.
- **Recommendation**: keep this as a documented option but **do not implement** unless Path A's upstream fix slips past 2 weeks. The probe confirmed Deepgram-on-streaming + whisper-local-on-batch (the Phase A configuration) is stable and acceptable as a long-term holding pattern.

**Path C — Reconnect-per-utterance Realtime client (workaround for the single-VAD bug).**
- Reconnect a fresh WS for every utterance (client-side VAD drives reconnect cadence). Each WS handles exactly one transcript before being discarded.
- Pros: works around the speaches single-VAD bug without upstream changes.
- Cons: ~5 WS reconnects/min per active speaker; reconnect cost (~23ms probe-measured + auth roundtrip) accumulates; session.update must be replayed each time; high error-surface area; defeats the point of a "long-lived streaming WS".
- **Recommendation**: not pursued. Too brittle for production.

The revised phased approach: **A (adapter + env toggle, default=deepgram, speculative Realtime provider lands but stays off) → B (Jarvis stays on `deepgram` for streaming + `whisper-local` for batch; soak measures stability of the local batch path under real workload; upstream speaches fix tracked in parallel) → B' (once speaches is patched, flip Jarvis to `realtime-local` with the 7-day side-by-side soak originally planned for B) → C (Joe's personal account) → D (drop Deepgram entirely)**.

## Discovery v1.1 (2026-06-01 am)

**When/how**: re-probed LiteLLM on `10.0.60.48:4000` after the rtx6000 agent posted an upgrade notice. The earlier scout reported `/v1/realtime` as 404; the re-probe found the endpoint live and serving the OpenAI Realtime API event stream over WebSocket (`101 Switching Protocols` on upgrade).

**What's new**:
- A working OpenAI Realtime API WebSocket at `ws://10.0.60.48:4000/v1/realtime` (alias `/openai/v1/realtime`), backed by `speaches` (open-source Realtime impl) with `faster-whisper-large-v3` as the transcription model after `session.update`.
- `server_vad` is built into the session; our existing `VadGate` may become redundant or shift role to "pre-filter before send" rather than "drive finalize()".
- Auth piggybacks on the existing LiteLLM Bearer key — no new credentials to manage.

**What it appeared to change (before deep-probe)**:
- Dropped the Phase B chunked-batch shim plan. The "no streaming endpoint" risk seemed gone.
- Phase B was reframed as a real streaming cutover (`STT_STREAMING_PROVIDER=realtime-local`).

**What the same-day deep-probe then reversed** (see "Realtime API deep-probe" section below): the endpoint is reachable and transcription quality is good, but four hard blockers in the current speaches build make it unusable as a Deepgram streaming replacement today. Phase B reverts to "keep Deepgram for streaming; track upstream speaches fix; flip to `realtime-local` once patched (Phase B')".

## Realtime API deep-probe (2026-06-01 pm)

Streamed real Joe pendant audio (23s clip from conv 02f66cd4, chunked into 230×100ms PCM16 16kHz mono frames, base64-encoded) through the Realtime WS over 5 probe runs. Bearer auth via existing `OPENAI_API_KEY`. WS URL `ws://10.0.60.48:4000/v1/realtime?intent=transcription&model=whisper-large-v3` (probe-confirmed both query params required; without `model=` LiteLLM returns 403).

### Latency (probe-measured)

| Phase | ms |
|-------|----|
| WS upgrade (connect + 101) | 23 |
| Time-to-first-partial | n/a — no `.delta` events are emitted |
| Time-to-final after `speech_stopped` | 348 |
| Time-to-final from WS open (full path) | ~8,770 (dominated by 3.2s of pre-speech audio + 550ms VAD silence threshold + 348ms transcription) |

### Observed event types (full list)

`session.created`, `session.updated`, `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped`, `input_audio_buffer.committed`, `conversation.item.created`, `conversation.item.input_audio_transcription.completed`, `response.created`, `error`.

**Not observed (even though spec implies they should exist)**: `conversation.item.input_audio_transcription.delta`. speaches/faster-whisper emits final-only.

### Sample event shapes

```jsonc
// L8: the final transcript event — only carries flat text + item_id, NO timestamps, NO words, NO speaker, NO confidence
{"content_index": 0,
 "event_id": "event_wUQC75paBZrkJa1o3wE2C",
 "item_id": "item_E7XFQnwY80LkaxDPGVq0C",
 "transcript": "So our LLM model.",
 "type": "conversation.item.input_audio_transcription.completed"}

// L4: speech_started — audio_start_ms is offset within the streamed buffer, NOT wall-clock
{"audio_start_ms": 3240,
 "event_id": "event_2V3J0Nv4jHds293hCljEQ",
 "item_id": "item_E7XFQnwY80LkaxDPGVq0C",
 "type": "input_audio_buffer.speech_started"}

// L11: ~1s after L8, speaches tries to generate an assistant response (despite create_response:false request),
//      fails with no-LLM-wired error. CONNECTION STAYS OPEN but session is corrupted afterwards.
{"error": {"message": "InternalServerError: Internal Server Error",
           "type": "server_error", "code": null, "event_id": null, "param": null},
 "event_id": "event_DgTGdZnUp8xdwfLe5n4nj",
 "type": "error"}
```

### `session.update` — what works and what doesn't

Working payload (probe-verified accepted; behavior matches the echo on these fields):
```json
{"type": "session.update",
 "session": {"modalities": ["text"],
             "input_audio_transcription": {"model": "Systran/faster-whisper-large-v3", "language": "en"},
             "temperature": 0.0}}
```

| Field | Accepted by server | Actually applied | Notes |
|-------|--------------------|------------------|-------|
| `modalities=["text"]` | yes | yes | Forces STT-only echo; default is `["audio","text"]` |
| `input_audio_transcription.model` | yes | yes | Switch from `Systran/faster-distil-whisper-small.en` (English-only) to `Systran/faster-whisper-large-v3` (multi-lang) |
| `input_audio_transcription.language` | yes | yes | `null` lets speaches auto-detect; `"en"` pins |
| `temperature` | yes | yes | `0.0` ok |
| `instructions` | yes | yes (but not used in STT-only path) | Defaults to "helpful witty friendly AI" |
| `turn_detection.threshold` | echoed | **NO** — server keeps hardcoded `0.9` regardless |
| `turn_detection.silence_duration_ms` | echoed | **NO** — server keeps hardcoded `550ms` regardless |
| `turn_detection.create_response` | echoed | **NO** — server always behaves as `true` (this is the root of the `error`-event-after-every-turn bug) |
| `turn_detection.prefix_padding_ms` | rejected | n/a | Explicit `invalid_request_error` |
| `turn_detection.interrupt_response` | rejected | n/a | Explicit `invalid_request_error` |
| `turn_detection: null` | rejected | n/a | pydantic validation error — speaches violates the OpenAI spec which says `null` disables VAD |
| `input_audio_format` | rejected | n/a | Hardcoded to `pcm16` |

### Hard blockers (probe-confirmed, not speculation)

1. **Single VAD turn per WS connection.** After the first `speech_started`/`speech_stopped` pair, the server stops detecting speech even when more audio is streamed. Verified in probe v4 by continuing to stream 18s of additional speech audio at 100ms cadence — zero subsequent `speech_started` events, zero subsequent transcripts. The session is dead after one utterance.
2. **`create_response=false` is silently ignored.** Every transcript turn triggers a `response.created` followed ~1s later by an `InternalServerError` error event (because no LLM is wired in this speaches deployment). This is likely the root cause of #1 — the failed response.created corrupts session state and disarms the VAD.
3. **`turn_detection: null` is rejected.** The OpenAI Realtime spec says `null` disables server VAD so the client can drive commits via `input_audio_buffer.commit`. speaches rejects `null` with a pydantic error, so client-driven commit is also not available.
4. **WS keepalive is broken.** Server doesn't respond to client pongs. With `ping_interval` enabled on the asyncio websockets client, the connection dies with `1011 keepalive ping timeout` after ~30s. Must set `ping_interval=None` and rely on traffic to keep the connection alive, OR implement app-level keepalive (zero-payload `input_audio_buffer.append` every 5s).

Probe also confirmed: `input_audio_buffer.commit` after VAD has already committed does NOT trigger a second transcription (4 manual commits after first VAD turn → 0 additional transcripts).

### What works well (preserved for the eventual fix)

- WS upgrade with `?model=X&intent=transcription` query params + Bearer auth.
- `session.update` of `model` / `language` / `modalities` / `temperature`.
- PCM16 16kHz mono audio streamed via base64-encoded `input_audio_buffer.append`.
- Server VAD on the first turn detects speech accurately (3.2s onset in probe matched real audio onset).
- Final transcript latency excellent (~348ms after `speech_stopped`).
- Transcription quality clean and correct (`"So our LLM model."` matched recognizable speech in clip).

### Net effect on the spec

- Streaming Phase B **deferred** until speaches is patched to honor `create_response=false` and re-arm VAD across turns. Estimated upstream fix: 1-2 days in speaches source; tracked separately with the rtx6000 agent.
- Phase A still ships the `RealtimeApiStreamingProvider` skeleton and integration test infrastructure — they will be ready to flip once speaches is patched.
- Phase A's user-visible behavior is unchanged from today: Deepgram for streaming + whisper-local for batch.
- Phase B reverts to "soak Phase A under real workload; do not flip streaming yet". Phase B' is the streaming cutover, scheduled after upstream fix lands.
- New risks added: speaches control-plane stability, keepalive workaround, single-VAD bug regression on speaches upgrades.
- Resolved open questions: `intent=transcription` is a no-op on the current build; word-level timestamps are not emitted; interim partials are not emitted; concurrent-session behavior is moot until single-VAD bug is fixed.
- New open questions: do we wait for the upstream speaches fix, or pursue the chunked-batch shim (Path B in "Provider choice per use case") as a parallel path?

## Adapter design

### `STTProvider` abstract base class

New file `backend/utils/stt/providers/base.py`:

```python
class STTSegment(TypedDict):
    speaker: str         # 'SPEAKER_0', 'SPEAKER_1', ... — must match memory-attribution Layer 2 expectations
    start: float         # seconds from session start (wall-clock-aligned by GatedDeepgramSocket equivalent)
    end: float
    text: str
    is_user: bool
    person_id: Optional[str]

class STTStreamingProvider(ABC):
    """Long-lived push-bytes / pull-segments interface. Replaces GatedDeepgramSocket as the
    de-facto streaming contract."""

    @abstractmethod
    async def start(self, *, language: str, sample_rate: int, channels: int,
                    keywords: list[str], on_segment: Callable[[list[STTSegment]], Awaitable[None]],
                    vad_gate: VadGate) -> None: ...

    @abstractmethod
    async def send(self, pcm16le_bytes: bytes) -> bool:
        """Returns False on dead connection. VAD pre-filtering handled internally if the provider
        wants; the wrapper passes raw bytes through."""

    @abstractmethod
    async def finalize(self) -> None:
        """Force-flush any pending partials. Called by VadGate on speech→silence transitions."""

    @abstractmethod
    async def finish(self) -> None:
        """Close the underlying transport. Idempotent."""

    @property
    @abstractmethod
    def is_dead(self) -> bool: ...

class STTBatchProvider(ABC):
    """One-shot transcribe-from-bytes interface. Replaces deepgram_prerecorded_from_bytes
    as the de-facto batch contract."""

    @abstractmethod
    async def transcribe(self, *, audio_bytes: bytes, encoding: str, sample_rate: int,
                         channels: int, language: str, keywords: list[str],
                         return_language: bool) -> dict:
        """Returns the legacy DG-shaped response:
            {'results': {'channels': [{'alternatives': [{'words': [...]}]}]},
             'detected_language': '<iso>'} (optional)
        The DG-shaped envelope is preserved so callers don't change."""
```

### Concrete providers

| Provider | Streaming class | Batch class |
|----------|-----------------|-------------|
| `deepgram` | `DeepgramStreamingProvider` (today's `GatedDeepgramSocket` + `connect_to_deepgram_with_backoff`, refactored) | `DeepgramBatchProvider` (today's DG branch of `deepgram_prerecorded`) |
| `realtime-local` | `RealtimeApiStreamingProvider` (NEW — connects to `ws://10.0.60.48:4000/v1/realtime`; see "Realtime API event-model translation" below) | (n/a — batch uses `whisper-local`) |
| `whisper-local` | (OBSOLETED chunked-batch shim removed from plan) | `WhisperBatchProvider` (today's `local_whisper_prerecorded_from_bytes`, plus `_post_sortformer` merge) |
| `voxtral-local` | not implemented — Voxtral is batch-only on LiteLLM | `VoxtralBatchProvider` (alternative selectable but disabled by default; reserved for audio-Q&A features) |

### Realtime API event-model translation

The OpenAI Realtime API event model is fundamentally different from Deepgram's word-level callbacks. The adapter must translate between the two. Event names + behavior below are probe-verified (see "Realtime API deep-probe (2026-06-01 pm)"):

| Aspect | Deepgram (today) | Realtime API (speaches, as observed in probe) |
|--------|------------------|-----------------------------------------------|
| Transcript granularity | Word-level callbacks (`result.channel.alternatives[0].words = [{word, start, end, speaker:int, ...}]`) | Utterance-level only: `conversation.item.input_audio_transcription.completed` with a flat `transcript` string. **NO `.delta` events emitted by speaches** (probe-verified). |
| Speaker IDs | Baked into each word | Not provided — we inject from Sortformer running in parallel |
| Word timings | Per-word `start`/`end` (sub-second precision) | **None** — probe confirmed the `.completed` event carries only flat text. No `word_offsets`, no per-word confidence. Word timings must be reconstructed from Sortformer segment boundaries + `speech_started.audio_start_ms` / `speech_stopped.audio_end_ms` interpolation. |
| VAD | Server-side endpointing (`endpointing=300`) emits a finalize event | Server-side `turn_detection: server_vad`, **`threshold=0.9` and `silence_duration_ms=550` are hardcoded** (probe-verified — payload values are echoed in `session.updated` but not actually applied). Emits `speech_started` / `speech_stopped` / `input_audio_buffer.committed` for the **first** turn only (single-VAD bug). |
| Audio ingest | `socket.send(pcm16_bytes)` raw frames | `input_audio_buffer.append` event with base64-encoded PCM16. 100ms chunks worked cleanly in probe. `input_audio_format=pcm16` is hardcoded. |
| Keepalive | DG SDK + our 5s `SafeDeepgramSocket` thread | **Broken on speaches** — server doesn't respond to client pongs; must disable `ping_interval` and rely on traffic or app-level no-op appends. |

**`RealtimeApiStreamingProvider` responsibilities** (skeleton lands in Phase A; do not flip on until speaches is patched):
- **Async WS lifecycle**: connect to `ws://10.0.60.48:4000/v1/realtime?intent=transcription&model=whisper-large-v3` (probe-verified that `?model=` query param is **mandatory** — without it LiteLLM returns 403, regardless of `session.update`). Send Bearer auth in the Upgrade request. Listen for `session.created`, then send `session.update` with the probe-verified working payload:
  ```json
  {"type":"session.update",
   "session":{"modalities":["text"],
              "input_audio_transcription":{"model":"Systran/faster-whisper-large-v3","language":null},
              "temperature":0.0}}
  ```
  Do **not** include `turn_detection`, `input_audio_format`, or `instructions` overrides — probe showed these are either rejected outright or silently ignored. Document this in code with a link back to this spec.
- **Disable client pings**: pass `ping_interval=None` to the `websockets` client to avoid the probe-observed 30s keepalive timeout. If idle-disconnect issues emerge in Phase B', add app-level no-op `input_audio_buffer.append` events as keepalives.
- **Reconnect-on-error**: exponential backoff with jitter (mirror `connect_to_deepgram_with_backoff` shape: 3 retries, jittered 1s/2s/4s). On reconnect, replay `session.update` with the same payload.
- **Single-VAD-turn workaround (until upstream fix)**: when the provider detects the post-`speech_stopped` `error` event (probe-verified to fire ~1s after every transcript), close the WS and reconnect for the next utterance. This is brittle (~5 reconnects/min/speaker, 23ms upgrade + auth roundtrip per reconnect) and is the reason streaming Phase B is deferred. Code path is implemented but gated behind `STT_STREAMING_PROVIDER=realtime-local` which stays off by default.
- **PCM ingest**: receive raw PCM16LE bytes from the pusher socket, base64-encode, chunk into ≤100ms slices (probe used 100ms with no issues), emit `input_audio_buffer.append` events. Keep an internal frame counter so we can map Sortformer wall-clock back to audio offsets.
- **Event-loop translator** — Realtime events → backend's `stream_transcript(segments)` callback shape:
  - `session.created` / `session.updated` → log; trigger `session.update` after `session.created`.
  - `input_audio_buffer.speech_started` → start a "current utterance" record; remember `audio_start_ms`.
  - `input_audio_buffer.speech_stopped` → remember `audio_end_ms`.
  - `input_audio_buffer.committed` → server-side commit fired; await `.completed` next.
  - `conversation.item.created` → placeholder; carries `item_id` linking back to forthcoming `.completed`.
  - `conversation.item.input_audio_transcription.completed` → final transcript. Reconcile with Sortformer window (covering same `audio_start_ms`-`audio_end_ms` slice) to pick a speaker label. Emit ONE `STTSegment` covering the full utterance, stamped from `audio_start_ms` to `audio_end_ms` plus our WS-open wall-clock anchor.
  - `response.created` → ignored (would be the assistant-response side; STT-only mode means it stays `incomplete`).
  - `error` (with message `InternalServerError`) → log warn; mark utterance done; trigger reconnect to work around single-VAD bug.
  - **No `.delta` events** — partial-transcript code path is intentionally absent; this matches current omi behavior (`interim_results=False`).
- **Speaker injection**: Sortformer runs in a parallel windowed POST loop (independent of the WS event stream). Per-window Sortformer outputs are reconciled against a per-session running TitaNet embedding bank so speaker IDs are stable across the session. The `.completed` event's `audio_start_ms`/`audio_end_ms` (from the matching `speech_started`/`speech_stopped` events) identifies which Sortformer window's labels to use.

### Factory + env-var dispatch

```python
# backend/utils/stt/providers/__init__.py

STT_STREAMING_PROVIDER = os.getenv('STT_STREAMING_PROVIDER', 'deepgram')
STT_BATCH_PROVIDER     = os.getenv('STT_BATCH_PROVIDER',     'deepgram')

def get_streaming_provider(byok_key: Optional[str] = None) -> STTStreamingProvider:
    # BYOK fast-path: any user with a per-request DG key bypasses the operator default
    if byok_key:
        return DeepgramStreamingProvider(api_key=byok_key)
    name = STT_STREAMING_PROVIDER
    if name == 'deepgram':         return DeepgramStreamingProvider()
    if name == 'realtime-local':   return RealtimeApiStreamingProvider()
    raise ValueError(f"Unknown STT_STREAMING_PROVIDER={name}")

def get_batch_provider(byok_key: Optional[str] = None) -> STTBatchProvider:
    if byok_key:
        return DeepgramBatchProvider(api_key=byok_key)
    name = STT_BATCH_PROVIDER
    if name == 'deepgram':       return DeepgramBatchProvider()
    if name == 'whisper-local':  return WhisperBatchProvider()
    if name == 'voxtral-local':  return VoxtralBatchProvider()
    raise ValueError(f"Unknown STT_BATCH_PROVIDER={name}")
```

### File-by-file change list

| File | Change |
|------|--------|
| `backend/utils/stt/providers/base.py` (new) | Abstract base classes + `STTSegment` typed dict |
| `backend/utils/stt/providers/__init__.py` (new) | Factory + env-var dispatch |
| `backend/utils/stt/providers/deepgram.py` (new) | Extract today's DG streaming/batch code into provider classes |
| `backend/utils/stt/providers/realtime_local.py` (new) | `RealtimeApiStreamingProvider` — OpenAI Realtime API WS client targeting `ws://10.0.60.48:4000/v1/realtime` (speaches). Streaming-only; uses `whisper_local` for batch. |
| `backend/utils/stt/providers/whisper_local.py` (new) | `WhisperBatchProvider` (move `local_whisper_prerecorded*` here). No `WhisperStreamingProvider` — chunked-batch shim plan obsoleted by 2026-06-01 Realtime API discovery. |
| `backend/utils/stt/providers/voxtral_local.py` (new) | `VoxtralBatchProvider`; no streaming class |
| `backend/utils/stt/streaming.py` | Replace `process_audio_dg` with `process_audio_streaming(...)` factory that gets a provider from the factory and wraps it in the VAD gate. Keep `deepgram_nova3_multi_languages` + `deepgram_nova3_languages` sets in place but rename module-level constants to `stt_multi_languages` / `stt_languages` with deprecation aliases. |
| `backend/utils/stt/pre_recorded.py` | Replace `deepgram_prerecorded[_from_bytes]` body with a thin dispatcher to `get_batch_provider().transcribe(...)`. Keep the public function names as compatibility shims so the six callers don't have to change. |
| `backend/utils/stt/safe_socket.py` | Rename `SafeDeepgramSocket` → `STTSocketKeepalive`. Make `keep_alive()` optional (Whisper has no analog — provider implements as no-op). Deprecation alias preserved. |
| `backend/utils/stt/vad_gate.py` | Rename `GatedDeepgramSocket` → `VadGatedSTTSocket`. Deprecation alias preserved. |
| `backend/routers/transcribe.py` | Rename local `deepgram_socket` variable → `stt_socket`. Update import sites for `process_audio_dg`. |
| `backend/routers/sync.py`, `backend/routers/users.py`, `backend/routers/speech_profile.py`, `backend/utils/chat.py`, `backend/utils/speaker_sample.py`, `backend/utils/conversations/postprocess_conversation.py` | Update imports from `deepgram_prerecorded[_from_bytes]` → `stt_prerecorded[_from_bytes]` (alias kept for one release). |
| `backend/utils/byok.py`, `backend/utils/subscription.py` | Keep `x-byok-deepgram` header as-is. BYOK still implies "the user pays Deepgram directly" — the header name is fine; only the operator-default changes. |
| `backend/migrations/006_auto_set_transcription_mode.py` | Update import alias to `stt_multi_languages`. |
| `backend/.env.template` | Add `STT_STREAMING_PROVIDER=deepgram` and `STT_BATCH_PROVIDER=deepgram` (template defaults stay safe; ops flip via `.env`). Add `WHISPER_BASE_URL`, `WHISPER_MODEL`, `VOXTRAL_MODEL`, `VOICE_EXTRAS_URL`, `SORTFORMER_MODEL` defaults. Add `REALTIME_WS_URL=ws://10.0.60.48:4000/v1/realtime`, `REALTIME_WS_QUERY=intent=transcription&model=whisper-large-v3` (the `model=` query param is probe-verified mandatory for LiteLLM-fronted speaches), `REALTIME_TRANSCRIPTION_MODEL=Systran/faster-whisper-large-v3`. Do **not** ship `REALTIME_VAD_THRESHOLD` / `REALTIME_VAD_SILENCE_MS` env vars — probe confirmed these are silently ignored by speaches; documenting them as tunable would mislead operators. Once speaches upstream honors these, add the env vars then. |
| `backend/pusher/requirements.txt` | Leave `deepgram-sdk==4.8.1` in place through Phase A. Drop in Phase D after we are certain pusher has no remaining DG code paths. |
| `backend/scripts/lint_async_blockers.py` | Add the new whisper-local streaming provider to the file allowlist (it issues async POSTs via `httpx.AsyncClient` from `utils/http_client.py` — already compliant). |

Estimated total: **~14 backend files** touched (matches Phase-1 scout estimate). Largest single change is `utils/stt/streaming.py`.

### Fallback strategy

Two levels of fallback:

1. **Provider-internal retry**: each provider implements its own backoff (DeepgramStreamingProvider keeps today's 3-retry + exponential-backoff-with-jitter; WhisperStreamingProvider retries each chunk POST up to 3 times before dropping the chunk and emitting a warning segment).
2. **Cross-provider fallback (configurable, default OFF)**: new env var `STT_STREAMING_FALLBACK=deepgram` (only consulted when primary provider declares `is_dead`). Disabled by default to honor cloud-zero; explicitly enabled by operators who want a safety net during the Phase B-C transition. **Joe's call** — see open questions.

Hard-cutover vs. graceful fallback is the key open question. The recommendation is to **ship Phase A with no cross-provider fallback** (the env var is `deepgram` for streaming, so primary is Deepgram, fallback is moot), and **decide at Phase B-start** whether the chunked-Whisper shim gets a Deepgram safety net.

## Rollout phases

### Phase A — Adapter + batch cutover + speculative Realtime client (3-4 days of focused work)

- Implement `STTProvider` ABCs, factory, env vars.
- Move existing DG streaming code into `DeepgramStreamingProvider` (refactor, no behavior change).
- Implement `RealtimeApiStreamingProvider` against `ws://10.0.60.48:4000/v1/realtime?intent=transcription&model=whisper-large-v3`, with the probe-verified `session.update` payload (modalities/text + model + temperature + language only — do not include the silently-ignored or rejected fields). Lands in the codebase but **not** wired as the default and **not** user-facing yet. Gate the integration test for it on speaches version (env var `SPEACHES_VERSION_AT_LEAST=...`) so a stale build doesn't pretend it works.
- Move existing whisper-local batch code into `WhisperBatchProvider` + `VoxtralBatchProvider`.
- Move existing DG batch code into `DeepgramBatchProvider`.
- Set `STT_BATCH_PROVIDER=whisper-local` in `.env` (preserves today's `STT_BATCH_BACKEND=whisper` semantics).
- Set `STT_STREAMING_PROVIDER=deepgram` in `.env` (NO behavior change for streaming — the new Realtime provider ships in code but stays off by default and is blocked by upstream speaches bugs).
- Ship unit tests for adapter contract (`test_stt_provider_contract.py`) — all providers must satisfy the same interface and pass the same fixture WAVs.
- Ship integration tests against rtx6000 (Whisper batch round-trip on a 30s test clip + Realtime WS connect + `session.update` + 5s PCM round-trip + single-utterance transcript-completed assertion). The Realtime test asserts only what the current speaches build supports; it will be extended once speaches is patched.
- Validate every batch caller (six) still works end-to-end on Joe's deployment.
- **Exit criteria**: 24h on Joe's listen pipeline with `STT_BATCH_PROVIDER=whisper-local` and zero regressions in the post-conversation re-transcribe path. Memory-attribution Layer 2 keeps working. Realtime provider passes the (limited) integration test but is not user-facing yet.

### Phase B — Soak Phase A under real workload + track speaches upstream fix (open-ended)

**Streaming cutover deferred.** The deep-probe (2026-06-01 pm) showed the current speaches build is blocked by four issues (see "Realtime API deep-probe"). Phase B is reframed:

- Keep `STT_STREAMING_PROVIDER=deepgram` on Jarvis. **Do not flip to `realtime-local` yet.**
- Soak `STT_BATCH_PROVIDER=whisper-local` for 7 continuous days on Jarvis. Validate batch-path stability under real workload (post-conversation re-transcribe, sync v2, voice messages, speech-profile samples). Capture per-caller latency and error rate.
- **In parallel**, file a fix request with the rtx6000 agent for the speaches blockers: (1) honor `create_response=false`, (2) re-arm VAD across turns after a transcript-completed event, (3) accept `turn_detection: null` per OpenAI spec, (4) respond to client WS pongs. Reference this spec section for the exact probe-verified failure modes.
- **Decision gate at +7 days**: if speaches has been patched, proceed to Phase B' below. If not, decide between (i) wait longer, or (ii) implement the chunked-batch shim (Path B in "Provider choice per use case") as a non-realtime workaround. Recommendation: wait if the upstream fix has a credible ETA; implement chunked-batch only if speaches is stuck >2 weeks.
- **Exit criteria**: 7 days clean on Jarvis batch + a green light from one of (a) speaches patched, (b) Joe approves chunked-batch shim, (c) Joe approves indefinite "Deepgram-for-streaming forever" stance.

### Phase B' — Flip Jarvis to Realtime (7-day side-by-side soak; gated on speaches fix)

- Only run this phase **after** speaches is patched and the integration test (the one gated by `SPEACHES_VERSION_AT_LEAST`) starts passing.
- Flip Jarvis to `STT_STREAMING_PROVIDER=realtime-local`. Real streaming cutover.
- **Side-by-side validation** for 7 continuous days: run a parallel shadow Deepgram session (read-only, no segments written to Firestore) for every Jarvis conversation; diff word error rate, latency-to-first-word, latency-to-final-word, speaker-attribution accuracy.
- Validate against the 2026-06-01 19:48 "Gesture" conversation and other recent Jarvis sessions as benchmark fixtures.
- If speaches starts honoring `turn_detection.threshold` / `silence_duration_ms` in the patched build, tune those (Deepgram's `endpointing=300` is the latency target). Until then, leave them out of the `session.update` payload to avoid sending no-op fields.
- Decision gate: if Jarvis WER and latency are within 15% of Deepgram for English, proceed to Phase C. If not, debug or temporarily fall back to `STT_STREAMING_PROVIDER=deepgram` and escalate (speaches version pin, model swap, threshold tune).
- **Exit criteria**: 7 days clean on Jarvis with no regressions in conversation extraction quality (measured against the memory-critic LLM scoring pipeline).

### Phase C — Cut Joe's personal account (1-2 days; gated on Phase B' clean exit)

- Joe's personal omi account (`jlportman3@gmail.com`) is **not yet on self-host** per memory note `user_google_accounts.md`. This phase is forward-looking: it assumes Joe has migrated his personal account to the self-host backend before flipping the provider. **Separate work item.**
- Once on self-host: flip `STT_STREAMING_PROVIDER=realtime-local` for Joe's account-scoped env (or globally if Jarvis-as-second-account is on the same backend instance).
- 7-day soak period mirroring Phase B'.

### Phase D — Remove Deepgram code (1 day, once stable)

- Drop `deepgram-sdk` from `pusher/requirements.txt` and `backend/requirements.txt`.
- Delete `DeepgramStreamingProvider`, `DeepgramBatchProvider`.
- Delete `DEEPGRAM_API_KEY`, `DEEPGRAM_SELF_HOSTED_*` env vars from template (keep in `.env.template` as a deprecated comment for one release).
- **BYOK gate**: this is the one place we cannot fully delete. Users on BYOK pay DG directly and must continue to work. Either (a) keep the DeepgramStreamingProvider behind a BYOK-only code path, or (b) drop BYOK-DG entirely and only accept BYOK-OpenAI-for-Whisper. **Joe's call** — see open questions.

## Risks and mitigations

### Latency budget for streaming

Numbers below for the Realtime API column are probe-measured (2026-06-01 pm deep-probe, 23s pendant audio, 100ms chunks, faster-whisper-large-v3). All numbers are on a quiet rtx6000 with no concurrent omi load.

| Metric | Deepgram today | Phase B' (Realtime API / speaches, probe-measured) | Mitigation |
|--------|---------------|----------------------------------------------------|------------|
| Time-to-first-partial | ~200ms (Deepgram interim) | **n/a** — no `.delta` events emitted by speaches | UX impact zero (omi already runs `interim_results=False`). Do not surface partials. |
| Time-to-final after VAD silence | ~300ms (DG `endpointing=300`) | **348ms** (probe-measured) | Acceptable parity once speaches is patched. |
| End-to-end first-final from speech onset | ~speech_duration + 300ms | speech_duration + 550ms (hardcoded `silence_duration_ms`) + 348ms transcription | ~250ms regression vs Deepgram. Mitigated only once speaches honors a lower `silence_duration_ms`. |
| Concurrent sessions per LiteLLM endpoint | Unlimited (Deepgram cloud) | Untested under load — single-VAD bug blocks meaningful concurrency testing today | Re-measure once speaches is patched. Test concurrency against rtx6000 agent; if hit, add session queueing in `RealtimeApiStreamingProvider`. |

### Language detection

- **Deepgram**: returns BCP-47 (`en-US`); we already split on `-`. Streaming has no live detection (we feed `language='multi'` or static code).
- **Whisper**: returns ISO short codes (`en`). `local_whisper_prerecorded_from_bytes` already normalizes. For streaming chunked-batch, each window can return its own `language` — we pin to the first window's result for the session to avoid mid-conversation language flapping.
- **Risk**: Whisper's auto-detect is less accurate than Deepgram on the first 1-2s of audio. **Mitigation**: if the user has a non-`multi` language pinned in settings, pass `language=<iso>` and skip Whisper's detection. For `multi` users, accept ~5-10% lower first-window language accuracy as a known regression and document it.

### Vocabulary biasing (`keyterm` / `keywords`)

- Deepgram supports up to 100 keyterm/keywords entries with measurable WER improvement for in-vocab tokens.
- Whisper supports `initial_prompt` (a text hint, ~250 token cap) but no real keyterm boosting. `local_whisper_prerecorded_from_bytes` accepts a `keywords` kwarg today and ignores it (`# Whisper does not support keyterm boost`).
- **Mitigation**: convert the keywords list to a comma-separated `initial_prompt` ("Omi, Jarvis, jlportman3, ...") and pass it as `prompt=` on the multipart POST. Test WER impact on in-vocab tokens; if insufficient, Voxtral-mini's audio-LLM prompt channel is the next-best option for batch — Voxtral can absorb a vocab hint in its text prompt with measurable effect. **For streaming**, Voxtral is not an option (no streaming endpoint). Joe accepts that streaming will lose keyterm boosting in Phase B.

### Speaker labels (diarization provenance)

- Deepgram streaming bakes diarization into the same WebSocket message (`words[i].speaker:int`).
- The Realtime API has no diarization. Sortformer runs in parallel.
- **Risk in streaming**: Sortformer is batch-only today (POST `/v1/audio/diarization`). The Realtime provider runs Sortformer in a parallel windowing loop (independent of the WS event stream). Per-window Sortformer calls produce **per-window-local speaker IDs** (window A's `speaker_0` and window B's `speaker_0` may be different people). The memory-attribution pipeline's Layer 2 cluster-vote algorithm assumes stable IDs across the session.
- **Mitigation**: post-process each window's Sortformer output by clustering its speaker centroids against a per-session running embedding bank (one TitaNet embedding per speaker label). New speakers get a new global ID; existing speakers get their stable ID. This is online speaker clustering — implemented in `RealtimeApiStreamingProvider._reconcile_speakers()`. The TitaNet service is already hot.

### Interim transcripts UX

- Deepgram emits interim partials every ~200ms. Today the omi backend has `interim_results=False`, so we don't actually surface partials to the app — every segment we push is final. **This means the UX impact of any partials policy is zero**: omi already only ships final segments.
- Probe confirmed speaches **does not emit `.delta` events at all** — only `conversation.item.input_audio_transcription.completed`. Since current omi already runs final-only, the missing partials are not a regression for omi. If a future omi feature wants live-typing UX, it will have to wait for speaches to add `.delta` (or use a different provider).

### Error recovery

- Deepgram WS reconnect: today's `connect_to_deepgram_with_backoff` retries 3× with exponential jitter.
- Realtime API WS reconnect: `RealtimeApiStreamingProvider` mirrors the same backoff shape. On reconnect, a new `session.update` re-pins STT-only mode + transcription model + language. Any audio buffered during the reconnect gap is replayed (bounded by 5s buffer; older audio is dropped with a warning log).
- **Per-utterance reconnect (single-VAD workaround, until upstream fix)**: until speaches re-arms VAD across turns, the provider must reconnect a fresh WS for each utterance. Probe-measured upgrade time is 23ms; per-reconnect cost is ~23ms WS + auth + `session.update` round-trip. At ~5 utterances/min/speaker this is 5 reconnects/min of provider churn. This is the primary reason streaming Phase B is deferred behind the upstream fix.
- **Risk**: rtx6000 LiteLLM goes down mid-conversation. Without cross-provider fallback, the conversation transcription stalls until LiteLLM is back. **Mitigation**: optional `STT_STREAMING_FALLBACK=deepgram` env var (default OFF for cloud-zero; ON during the Phase B'/C transition if Joe wants the safety net).

### speaches single-VAD-turn bug (BLOCKER, probe-confirmed 2026-06-01 pm)

- After the first `speech_started`/`speech_stopped` pair on a WS connection, the server stops detecting speech even when more audio is streamed. Verified by probe v4 with 18s of additional speech audio post-first-turn producing zero subsequent events.
- **Root cause hypothesis**: `turn_detection.create_response=true` (silently unchangeable) triggers a failed assistant-response generation flow that emits an `error` event ~1s after every transcript. This appears to corrupt session state and disarm VAD.
- **Mitigation**: (preferred) file upstream patch with rtx6000 agent — speaches must (a) honor `create_response=false` in `session.update`, and (b) re-arm VAD after each transcript even when the response.created path errors out. (workaround) reconnect per utterance — implemented in `RealtimeApiStreamingProvider` but production-discouraged due to reconnect churn. **This is the single reason streaming Phase B is deferred.**

### speaches `turn_detection` knobs are unchangeable (probe-confirmed 2026-06-01 pm)

- `turn_detection.threshold` / `silence_duration_ms` / `create_response` are echoed in `session.updated` but the server keeps hardcoded `0.9` / `550ms` / `true` regardless. `prefix_padding_ms` and `interrupt_response` are explicitly rejected. `turn_detection: null` (the OpenAI-spec-compliant way to disable server VAD) is rejected with a pydantic validation error.
- **Mitigation**: do not ship `REALTIME_VAD_THRESHOLD` / `REALTIME_VAD_SILENCE_MS` env vars in Phase A — they would mislead operators into thinking they have a knob they don't. Once speaches honors these, add the env vars in the same PR as the speaches version bump.

### speaches WS keepalive is broken (probe-confirmed 2026-06-01 pm)

- Server doesn't respond to client pongs. With the `websockets` Python client's default `ping_interval=20s`, the connection dies with `1011 keepalive ping timeout` after ~30s of idle.
- **Mitigation**: `RealtimeApiStreamingProvider` must construct the WS client with `ping_interval=None`. If silent-idle periods longer than ~30s become an issue in production (rare for an active pendant), emit a zero-payload `input_audio_buffer.append` event every 5s as an application-level keepalive.

### faster-whisper-large-v3 multi-lang accuracy vs Deepgram nova-3-multi (probe-partially-measured 2026-06-01)

- Probe transcribed 23s of Joe's pendant audio cleanly with the override to `Systran/faster-whisper-large-v3` (`"So our LLM model."` — accurate). This is a single sample, single language (English), not a WER measurement.
- **Mitigation**: Phase B' 7-day side-by-side soak captures per-conversation WER deltas across the multi-language set. If accuracy regresses >15% on any well-represented language, debug (model swap to a larger faster-whisper variant if available on rtx6000, or pin language explicitly per user) before proceeding to Phase C.

### Realtime API spec evolution + speaches divergence (probe-confirmed 2026-06-01)

- The OpenAI Realtime API is pre-1.0; speaches diverges from the spec in several places (probe-confirmed): `turn_detection: null` rejected when spec says it should disable VAD; `intent=transcription` query param accepted but no-op; `?model=` query param required when spec does not document this; `create_response=false` silently ignored.
- **Mitigation**: pin speaches version in deployment notes. Document the probe-verified working `session.update` payload in code comments + this spec. On any speaches upgrade, re-run the integration test suite (Phase A) before promoting. Maintain the divergence list in this spec; update on each rtx6000 agent speaches upgrade.

### BYOK preservation

- BYOK users pay Deepgram directly. The migration must not break this.
- The factory's `byok_key` parameter overrides the env-var default. `_deepgram_client_for_request` semantics are preserved by `DeepgramStreamingProvider(api_key=byok_key)`.
- **No risk** if implemented correctly. Test coverage explicit (`test_byok_security.py` already exists; extend it).

### Module-import side-effects

- Today's `streaming.py` constructs `DeepgramClient(...)` at module import time. If `DEEPGRAM_API_KEY` is removed, the import will fail.
- **Mitigation**: lazy-construct in `DeepgramStreamingProvider.__init__` (per-instance, not module-level). Same for `pre_recorded.py`. This is a small refactor but mandatory for Phase D.

## Testing strategy

### Unit tests (Phase A)

- `tests/unit/test_stt_provider_contract.py` (new): every concrete `STTBatchProvider` must accept the same fixture WAV (5s English, 5s Spanish, 1s silence) and return the same DG-shaped envelope. Every `STTStreamingProvider` must accept the same fixture PCM stream and emit comparable segments (same start/end within 100ms tolerance, same speaker count, text WER < 0.20 vs reference).
- `tests/unit/test_whisper_batch.py` (extend): exercise the new provider class directly; current test only hits the legacy function shim.
- `tests/unit/test_stt_factory.py` (new): env-var dispatch, BYOK override, invalid name raises `ValueError`.
- `tests/unit/test_byok_security.py` (extend): assert `STT_STREAMING_PROVIDER=realtime-local` + user-supplied `x-byok-deepgram` still routes the user through DG, not through realtime-local.

### Integration tests (Phase A — gated on rtx6000 reachability)

- `tests/integration/test_whisper_against_rtx6000.py` (new): synthetic 30s WAV with three speakers (TTS-generated, scripted), POST to LiteLLM, validate word count, language detection, diarization merge. Skipped if `RTX6000_REACHABLE=0`.
- `tests/integration/test_realtime_api_against_rtx6000.py` (new): WS connect to `ws://10.0.60.48:4000/v1/realtime`, send `session.update` to STT-only, stream a 5s WAV via `input_audio_buffer.append`, assert `conversation.item.input_audio_transcription.completed` arrives with non-empty text. Skipped if `RTX6000_REACHABLE=0`.
- `tests/integration/test_stt_concurrent_sessions.py` (new): 4 parallel sessions × 30s WAVs (mix of batch `whisper-large-v3` and Realtime WS). Validates rtx6000 doesn't choke on omi's typical concurrent load. Skipped if `RTX6000_REACHABLE=0`.
- Existing `test_dg_start_guard.py`, `test_streaming_deepgram_backoff.py` keep passing against `STT_STREAMING_PROVIDER=deepgram`.

### End-to-end (Phase B)

- Run Jarvis listen session for a 30-minute conversation under `STT_STREAMING_PROVIDER=realtime-local`. Validate:
  - All segments persisted to Firestore.
  - Memory-attribution Layer 2 produces non-empty `is_user` attribution.
  - Memory extractor produces ≥1 fact with provenance pointing back to a Phase-B-captured segment.
  - Memory-critic background sweep doesn't flag the Phase-B segments at a higher rate than Phase-A baselines.
- Capture WER + latency-to-final-word side-by-side with a shadow Deepgram session (read-only DG, no Firestore writes).
- Record the shadow-comparison metrics into AMS so future sessions can `/recall` them.

## Open questions for the user

These are concrete decisions Joe needs to make. Each blocks specific code paths. The 2026-06-01 deep-probe resolved several earlier questions and surfaced new ones; resolved items are listed at the end.

1. **Phase B path forward — wait for upstream speaches fix, or implement chunked-batch shim as a parallel track?** The probe-confirmed single-VAD bug + ignored `create_response=false` mean `realtime-local` cannot ship today. Two options: **(a)** wait for the rtx6000 agent to patch speaches (estimated 1-2 day fix, but real ETA unknown) and keep `STT_STREAMING_PROVIDER=deepgram` indefinitely until then; **(b)** also build the chunked-batch shim (Path B in "Provider choice per use case") as a non-realtime intermediate so streaming users can move off Deepgram even before speaches is fixed. Recommendation: (a) — accept Deepgram-for-streaming as the holding pattern, since the probe showed it's stable and our batch path is already local. Only fall to (b) if upstream slips >2 weeks.

2. **Cross-provider fallback — implement `STT_STREAMING_FALLBACK=deepgram` (default OFF), or skip it entirely?** Implementing it costs ~half a day and preserves a Deepgram safety net during the Phase B'/C transition. Skipping it keeps the codebase simpler and honors cloud-zero. Recommendation: implement it, leave it OFF in the template, flip ON during Phase B'/C if speaches stability proves shaky.

3. **Vocabulary biasing in batch — accept `initial_prompt` as a degraded substitute, or selectively route batch with vocab through Voxtral-mini?** Voxtral's audio-LLM prompt channel can absorb the vocab list with measurable effect. The trade-off is Voxtral's ~5× higher latency and silence-hallucination behavior (manageable in batch where audio is non-silent). Recommendation: ship with `initial_prompt` first, add Voxtral-as-vocab-aware-batch-fallback as a Phase B+ stretch goal.

4. **BYOK in Phase D — keep `x-byok-deepgram` support after we delete `DeepgramStreamingProvider` for the operator default, or drop BYOK-DG entirely?** Recommendation: keep the BYOK code path through Phase D — it's a per-request DG client construction, no module-level dependencies — and only drop it if it becomes a maintenance burden. **Alternative**: drop BYOK-DG, document that BYOK users now bring their OpenAI key (used against LiteLLM passthrough or a separate openai.com Whisper endpoint).

5. **Language scope** — keep the existing `deepgram_nova3_multi_languages` set as the only supported `language=multi` whitelist, or expand to Whisper's full 99-language list? Recommendation: keep today's set in v1 (no UI/UX changes, no user expectations to manage). Expand in a separate spec once Phase D is stable.

6. **Pusher service** — the `pusher/main.py` itself has no remaining DG references (Phase-1 scout confirmed), but `pusher/requirements.txt` still pins `deepgram-sdk==4.8.1`. Confirm: do we drop the dep in Phase A (lower attack surface, fewer build artifacts) or wait until Phase D (parallel removal with the rest of DG code)? Recommendation: Phase D — keep one consistent removal point.

7. **`?model=` query param on the Realtime WS URL (NEW 2026-06-01 pm)** — probe confirmed the param is **mandatory** (LiteLLM returns 403 without it) and not documented in the OpenAI spec. Do we hardcode it in `RealtimeApiStreamingProvider`, or make it an env var (`REALTIME_WS_QUERY=intent=transcription&model=whisper-large-v3`)? Recommendation: env var (defaults to the working probe value), so operators with a different LiteLLM model alias can override without code changes.

8. **Upstream speaches patch coordination** — who files the issue with the rtx6000 agent: this spec author, or Joe? Recommendation: this spec author files a concise issue referencing this spec's "Realtime API deep-probe" section as the failing-behavior repro. The four blockers (single VAD, ignored `create_response=false`, rejected `turn_detection: null`, broken keepalive pong) are described concretely enough to act on.

### Resolved by the 2026-06-01 deep-probe (no longer open)

- ~~Whether `intent=transcription` query param meaningfully changes the event stream.~~ **No** — accepted but no-op on the current speaches build.
- ~~Whether `session.update` to set `modalities=["text"]` + `create_response=false` cleanly disables the assistant-response side.~~ **Partially** — `modalities=["text"]` applies; `create_response=false` is silently ignored.
- ~~Whether speaches exposes word-level timestamps in `.completed` events.~~ **No** — only flat transcript text. Word timings must come from Sortformer.
- ~~Whether speaches emits `.delta` partials.~~ **No** — final-only.
- ~~Whether `turn_detection.threshold` / `silence_duration_ms` are tunable.~~ **No** — silently ignored despite being echoed in `session.updated`.
- ~~End-to-end time-to-final-word latency.~~ **~348ms after `speech_stopped`** — competitive with Deepgram once single-VAD bug is fixed.

---

*End of spec. Implementation plan to follow once questions 1, 2, 4, 6, 7 are answered. Question 8 (upstream coordination) is independent and can proceed immediately.*
