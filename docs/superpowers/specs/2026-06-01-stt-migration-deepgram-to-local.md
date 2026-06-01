# STT Migration: Deepgram → rtx6000 local (Whisper / Voxtral)

**Date:** 2026-06-01
**Status:** Design draft. Scout passes complete (rtx6000 HTTP probe + backend integration map). Awaiting Joe's answers to the open questions at the bottom before Phase A implementation kicks off.
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

### What's **not** reachable

All other STT-candidate ports on rtx6000 are **closed** from gpufarm:
- 9000 (faster-whisper-server default)
- 8000-8099 except 8094 (speaches, etc.)
- 5000-5050, 7860-7862, 11434, 3000-3001

There is **no standalone faster-whisper-server / speaches / Riva instance directly exposed** today. Anything we want as a streaming endpoint must either be exposed by the rtx6000 agent in a future session, or wrapped on top of the LiteLLM batch endpoint via chunked POSTs.

## Provider choice per use case

### Batch / pre-recorded → `whisper-large-v3` via LiteLLM

Already implemented, already the default in `.env`. Six callers, all happy. Justification:
- **Latency**: 283ms for a 1s clip; production clips are 15-120s, scaling roughly linearly, well inside the operator's batch budget (post-conversation re-transcribe is async and can take 30s+).
- **Quality**: Whisper-large-v3 ranks higher than Nova-3 on librispeech-clean and on noisy real-world data per public benchmarks; for Joe's pendant audio specifically the prior memory note confirms Whisper-large-v3 is the canonical local STT.
- **Silent-input correctness**: returns empty text, no hallucinations. Voxtral fails this on the smoke test.
- **Language coverage**: superset of Nova-3 multi-language set; no language regressions.
- **Word-level timestamps**: `response_format=verbose_json, timestamp_granularities[]=word` returns the exact shape needed by `_merge_words_with_speakers`.

Voxtral-mini stays in the codebase as a selectable alternative for **future audio-Q&A features** (Voxtral is an audio-LLM that can answer questions about audio in one shot — different product surface than transcription). It is **not** the recommended batch STT provider today because of the silence-hallucination behavior.

### Streaming (live BLE / desktop PTT / multi-channel) → **dual-track**

This is the contentious decision. Three viable paths, each with trade-offs:

**Path A — Keep Deepgram for streaming, only migrate batch (recommended for Phase A).**
- Pros: Zero UX regression. Deepgram's interim latency (~200ms partial words on screen) is something neither Whisper nor Voxtral can match without a real streaming server. Speaker diarization is baked in; vocabulary biasing works as today. Implementation cost: ship the env-var plumbing + adapter shape, default everything to `deepgram` for streaming, validate batch is end-to-end local. This is the **lowest-risk first cut** and unblocks the cloud-zero goal for the larger of the two surfaces (batch re-transcribe and sync v2 generate the bulk of cloud DG minutes on Joe's account).
- Cons: Doesn't get to cloud-zero. Still bills Deepgram for live audio.
- **Recommendation**: this is the Phase A target.

**Path B — Chunked-batch shim against `whisper-large-v3` (Phase B target).**
- Pusher / `process_audio_dg` replacement that buffers PCM into 3-5s rolling windows, POSTs each window to `/v1/audio/transcriptions`, and emits the words as a "final" segment per window. No interim transcripts. VAD-aware: instead of fixed 3s windows, flush a window every time `vad_gate` detects speech→silence (matches the existing `finalize()` semantics — the silence transition becomes the natural chunk boundary).
- Pros: Zero new infra on rtx6000. Cloud-zero achievable today. Uses the existing Sortformer service for streaming diarization via parallel windowed POSTs.
- Cons: No interim transcripts on the app — the live transcript view will tick once per VAD utterance (~1-3s) instead of every ~200ms. Latency budget per window: 283ms whisper + ~150ms diarization round-trip + chunk-assembly overhead. For 5s utterances total end-to-end is ~500-700ms vs. Deepgram's ~200ms.
- **Recommendation**: this is the Phase B target. Acceptable for Joe's solo Jarvis test account because the desktop app's transcript view is glanced at, not read live. **Not acceptable for the BasedHardware fork's UX target without further work**.

**Path C — Streaming WS endpoint on rtx6000 (Phase C target, requires separate agent work).**
- Stand up `faster-whisper-server` or `speaches` or NVIDIA Riva ASR on rtx6000 with a `/v1/audio/transcriptions/stream` WebSocket. Implement a true streaming adapter against that.
- Pros: True interim-transcript parity with Deepgram. Cloud-zero. Production-quality UX.
- Cons: Not our session to do — rtx6000 agent has to expose the port, and Joe has to confirm there's VRAM headroom (memory note: `~5GB VRAM free on rtx6000`, Whisper-large needs ~3-5GB depending on quantization). Adapter implementation is significantly more complex (back-pressure, partial-result reconciliation, reconnect semantics).
- **Recommendation**: keep on the roadmap but **do not block Phase A / B on it**.

The phased approach below picks **A → B → C → drop Deepgram entirely**.

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
| `whisper-local` | `WhisperStreamingProvider` (chunked-batch shim — see Phase B section) | `WhisperBatchProvider` (today's `local_whisper_prerecorded_from_bytes`, plus `_post_sortformer` merge) |
| `voxtral-local` | not implemented — Voxtral is batch-only on LiteLLM | `VoxtralBatchProvider` (alternative selectable but disabled by default; reserved for audio-Q&A features) |

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
    if name == 'deepgram':       return DeepgramStreamingProvider()
    if name == 'whisper-local':  return WhisperStreamingProvider()
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
| `backend/utils/stt/providers/whisper_local.py` (new) | `WhisperBatchProvider` (move `local_whisper_prerecorded*` here); `WhisperStreamingProvider` (Phase B chunked shim) |
| `backend/utils/stt/providers/voxtral_local.py` (new) | `VoxtralBatchProvider`; no streaming class |
| `backend/utils/stt/streaming.py` | Replace `process_audio_dg` with `process_audio_streaming(...)` factory that gets a provider from the factory and wraps it in the VAD gate. Keep `deepgram_nova3_multi_languages` + `deepgram_nova3_languages` sets in place but rename module-level constants to `stt_multi_languages` / `stt_languages` with deprecation aliases. |
| `backend/utils/stt/pre_recorded.py` | Replace `deepgram_prerecorded[_from_bytes]` body with a thin dispatcher to `get_batch_provider().transcribe(...)`. Keep the public function names as compatibility shims so the six callers don't have to change. |
| `backend/utils/stt/safe_socket.py` | Rename `SafeDeepgramSocket` → `STTSocketKeepalive`. Make `keep_alive()` optional (Whisper has no analog — provider implements as no-op). Deprecation alias preserved. |
| `backend/utils/stt/vad_gate.py` | Rename `GatedDeepgramSocket` → `VadGatedSTTSocket`. Deprecation alias preserved. |
| `backend/routers/transcribe.py` | Rename local `deepgram_socket` variable → `stt_socket`. Update import sites for `process_audio_dg`. |
| `backend/routers/sync.py`, `backend/routers/users.py`, `backend/routers/speech_profile.py`, `backend/utils/chat.py`, `backend/utils/speaker_sample.py`, `backend/utils/conversations/postprocess_conversation.py` | Update imports from `deepgram_prerecorded[_from_bytes]` → `stt_prerecorded[_from_bytes]` (alias kept for one release). |
| `backend/utils/byok.py`, `backend/utils/subscription.py` | Keep `x-byok-deepgram` header as-is. BYOK still implies "the user pays Deepgram directly" — the header name is fine; only the operator-default changes. |
| `backend/migrations/006_auto_set_transcription_mode.py` | Update import alias to `stt_multi_languages`. |
| `backend/.env.template` | Add `STT_STREAMING_PROVIDER=deepgram` and `STT_BATCH_PROVIDER=deepgram` (template defaults stay safe; ops flip via `.env`). Add `WHISPER_BASE_URL`, `WHISPER_MODEL`, `VOXTRAL_MODEL`, `VOICE_EXTRAS_URL`, `SORTFORMER_MODEL` defaults. |
| `backend/pusher/requirements.txt` | Leave `deepgram-sdk==4.8.1` in place through Phase A. Drop in Phase D after we are certain pusher has no remaining DG code paths. |
| `backend/scripts/lint_async_blockers.py` | Add the new whisper-local streaming provider to the file allowlist (it issues async POSTs via `httpx.AsyncClient` from `utils/http_client.py` — already compliant). |

Estimated total: **~14 backend files** touched (matches Phase-1 scout estimate). Largest single change is `utils/stt/streaming.py`.

### Fallback strategy

Two levels of fallback:

1. **Provider-internal retry**: each provider implements its own backoff (DeepgramStreamingProvider keeps today's 3-retry + exponential-backoff-with-jitter; WhisperStreamingProvider retries each chunk POST up to 3 times before dropping the chunk and emitting a warning segment).
2. **Cross-provider fallback (configurable, default OFF)**: new env var `STT_STREAMING_FALLBACK=deepgram` (only consulted when primary provider declares `is_dead`). Disabled by default to honor cloud-zero; explicitly enabled by operators who want a safety net during the Phase B-C transition. **Joe's call** — see open questions.

Hard-cutover vs. graceful fallback is the key open question. The recommendation is to **ship Phase A with no cross-provider fallback** (the env var is `deepgram` for streaming, so primary is Deepgram, fallback is moot), and **decide at Phase B-start** whether the chunked-Whisper shim gets a Deepgram safety net.

## Rollout phases

### Phase A — Adapter + batch cutover (2-3 days of focused work)

- Implement `STTProvider` ABCs, factory, env vars.
- Move existing DG streaming code into `DeepgramStreamingProvider` (refactor, no behavior change).
- Move existing whisper-local batch code into `WhisperBatchProvider` + `VoxtralBatchProvider`.
- Move existing DG batch code into `DeepgramBatchProvider`.
- Set `STT_BATCH_PROVIDER=whisper-local` in `.env` (preserves today's `STT_BATCH_BACKEND=whisper` semantics).
- Set `STT_STREAMING_PROVIDER=deepgram` in `.env` (no behavior change).
- Ship unit tests for adapter contract (`test_stt_provider_contract.py`) — both providers must satisfy the same interface and pass the same fixture WAVs.
- Ship integration tests against rtx6000 (Whisper batch round-trip on a 30s test clip, language-detect round-trip, diarization-merge round-trip).
- Validate every batch caller (six) still works end-to-end on Joe's deployment.
- **Exit criteria**: 24h on Joe's listen pipeline with `STT_BATCH_PROVIDER=whisper-local` and zero regressions in the post-conversation re-transcribe path. Memory-attribution Layer 2 keeps working (it depends on segment shape, not provider).

### Phase B — Streaming chunked-batch shim (4-7 days, Jarvis only)

- Implement `WhisperStreamingProvider` as a VAD-driven chunked-batch shim:
  - Buffer PCM in memory until `vad_gate` flags a speech→silence transition or 5s elapsed (whichever first).
  - On flush: POST the buffered window to `/v1/audio/transcriptions` (Whisper) and `/v1/audio/diarization` (Sortformer) in parallel via `asyncio.gather`.
  - Merge words+speakers via `_merge_words_with_speakers` (reuse the batch code).
  - Emit segments via `on_segment` callback, timestamps remapped to wall-clock by the same `DgWallMapper` (renamed `STTWallMapper`).
  - `finalize()` = force-flush any pending window even before VAD silence.
  - `is_dead` = LiteLLM has returned ≥3 consecutive non-2xx responses.
- Flip Jarvis test account to `STT_STREAMING_PROVIDER=whisper-local`.
- **Side-by-side validation** for 7 continuous days: run a parallel shadow Deepgram session (read-only, no segments written to Firestore) for every Jarvis conversation; diff word error rate, latency-to-first-word, latency-to-final-word, speaker-attribution accuracy.
- Decision gate: if Jarvis WER and latency are within 15% of Deepgram for English, proceed to Phase C. If not, debug or fall back to Phase A defaults and escalate to the rtx6000 agent for Path C work.
- **Exit criteria**: 7 days clean on Jarvis with no regressions in conversation extraction quality (measured against the memory-critic LLM scoring pipeline).

### Phase C — Cut Joe's personal account (1-2 days)

- Joe's personal omi account (`jlportman3@gmail.com`) is **not yet on self-host** per memory note `user_google_accounts.md`. This phase is forward-looking: it assumes Joe has migrated his personal account to the self-host backend before flipping the provider.
- Once on self-host: flip `STT_STREAMING_PROVIDER=whisper-local` for Joe's account-scoped env (or globally if Jarvis-as-second-account is on the same backend instance).
- 7-day soak period mirroring Phase B.
- If a true streaming WS endpoint has landed on rtx6000 by this point (Path C from the provider-choice section), the streaming provider name becomes `whisper-stream-ws` (new concrete class). The adapter shape is unchanged.

### Phase D — Remove Deepgram code (1 day, once stable)

- Drop `deepgram-sdk` from `pusher/requirements.txt` and `backend/requirements.txt`.
- Delete `DeepgramStreamingProvider`, `DeepgramBatchProvider`.
- Delete `DEEPGRAM_API_KEY`, `DEEPGRAM_SELF_HOSTED_*` env vars from template (keep in `.env.template` as a deprecated comment for one release).
- **BYOK gate**: this is the one place we cannot fully delete. Users on BYOK pay DG directly and must continue to work. Either (a) keep the DeepgramStreamingProvider behind a BYOK-only code path, or (b) drop BYOK-DG entirely and only accept BYOK-OpenAI-for-Whisper. **Joe's call** — see open questions.

## Risks and mitigations

### Latency budget for streaming

| Metric | Deepgram today | Phase B (chunked Whisper) | Mitigation |
|--------|---------------|---------------------------|------------|
| Time-to-first-word | ~200ms | First word lands at end of first VAD utterance, typically 1-3s | Surface a "transcribing…" placeholder during the chunk gap. Already implicit in the BLE pendant UX (no live captions). |
| End-to-end word latency | ~300ms (final after VAD endpointing=300ms) | 283ms whisper + ~150ms diarization + chunk-assembly = ~500-700ms per 5s window | Cap window at 3-5s. Run whisper + sortformer POSTs in parallel via `asyncio.gather`. |
| Concurrent sessions per LiteLLM endpoint | Unlimited (Deepgram cloud) | Limited by rtx6000 GPU concurrency on whisper-large | Test concurrency against rtx6000 agent. If hit, add request queueing in `WhisperStreamingProvider`. |

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
- Whisper has no diarization. Sortformer runs in parallel.
- **Risk in streaming**: Sortformer is batch-only today (POST `/v1/audio/diarization`). Per-window Sortformer calls produce **per-window-local speaker IDs** (window A's `speaker_0` and window B's `speaker_0` may be different people). The memory-attribution pipeline's Layer 2 cluster-vote algorithm assumes stable IDs across the session.
- **Mitigation**: post-process each window's Sortformer output by clustering its speaker centroids against a per-session running embedding bank (one TitaNet embedding per speaker label). New speakers get a new global ID; existing speakers get their stable ID. This is essentially online speaker clustering — small extra work in `WhisperStreamingProvider._reconcile_speakers()`. The TitaNet service is already hot.

### Interim transcripts UX

- Deepgram emits interim partials every ~200ms. Today the omi backend has `interim_results=False`, so we don't actually surface partials to the app — every segment we push is final. **This means the UX impact of dropping interim transcripts is zero**: omi already only ships final segments. Phase B's "no interims" property is not a regression against current shipped behavior.
- This was the single biggest concern going in; on inspection it turns out to be a non-issue.

### Error recovery

- Deepgram WS reconnect: today's `connect_to_deepgram_with_backoff` retries 3× with exponential jitter.
- Whisper chunked-batch: each window POST is independently retryable. On 3 consecutive failures the provider declares `is_dead`, the pusher socket closes, the client reconnects, and a new provider instance is built. Same UX as Deepgram cold-restart.
- **Risk**: rtx6000 LiteLLM goes down mid-conversation. Without cross-provider fallback, the conversation transcription stalls until LiteLLM is back. **Mitigation**: optional `STT_STREAMING_FALLBACK=deepgram` env var (default OFF for cloud-zero; ON during the Phase B-C transition if Joe wants the safety net).

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
- `tests/unit/test_byok_security.py` (extend): assert `STT_STREAMING_PROVIDER=whisper-local` + user-supplied `x-byok-deepgram` still routes the user through DG, not through whisper-local.

### Integration tests (Phase A — gated on rtx6000 reachability)

- `tests/integration/test_whisper_against_rtx6000.py` (new): synthetic 30s WAV with three speakers (TTS-generated, scripted), POST to LiteLLM, validate word count, language detection, diarization merge. Skipped if `RTX6000_REACHABLE=0`.
- `tests/integration/test_stt_concurrent_sessions.py` (new): 4 parallel sessions × 30s WAVs against `whisper-large-v3`. Validates rtx6000 doesn't choke on omi's typical concurrent load. Skipped if `RTX6000_REACHABLE=0`.
- Existing `test_dg_start_guard.py`, `test_streaming_deepgram_backoff.py` keep passing against `STT_STREAMING_PROVIDER=deepgram`.

### End-to-end (Phase B)

- Run Jarvis listen session for a 30-minute conversation under `STT_STREAMING_PROVIDER=whisper-local`. Validate:
  - All segments persisted to Firestore.
  - Memory-attribution Layer 2 produces non-empty `is_user` attribution.
  - Memory extractor produces ≥1 fact with provenance pointing back to a Phase-B-captured segment.
  - Memory-critic background sweep doesn't flag the Phase-B segments at a higher rate than Phase-A baselines.
- Capture WER + latency-to-final-word side-by-side with a shadow Deepgram session (read-only DG, no Firestore writes).
- Record the shadow-comparison metrics into AMS so future sessions can `/recall` them.

## Open questions for the user

These are concrete decisions Joe needs to make before Phase A implementation kicks off. Each blocks specific code paths.

1. **Phase A streaming default — keep `deepgram` or hard-cut to `whisper-local` chunked-batch?** Recommendation: keep `deepgram` for Phase A, flip to `whisper-local` in Phase B with the 7-day Jarvis side-by-side soak. **Alternative**: hard-cut now for the cloud-zero principle, accept the ~1-3s "transcribing…" gap as the new UX. Pick one.

2. **Cross-provider fallback — implement `STT_STREAMING_FALLBACK=deepgram` (default OFF), or skip it entirely?** Implementing it costs ~half a day and preserves a Deepgram safety net during the Phase B-C transition. Skipping it keeps the codebase simpler and honors cloud-zero. Recommendation: implement it, leave it OFF in the template, flip ON during Phase B-C if needed.

3. **Vocabulary biasing in batch — accept `initial_prompt` as a degraded substitute, or selectively route batch with vocab through Voxtral-mini?** Voxtral's audio-LLM prompt channel can absorb the vocab list with measurable effect. The trade-off is Voxtral's ~5× higher latency and silence-hallucination behavior (manageable in batch where audio is non-silent). Recommendation: ship with `initial_prompt` first, add Voxtral-as-vocab-aware-batch-fallback as a Phase B+ stretch goal.

4. **BYOK in Phase D — keep `x-byok-deepgram` support after we delete `DeepgramStreamingProvider` for the operator default, or drop BYOK-DG entirely?** Recommendation: keep the BYOK code path through Phase D — it's a per-request DG client construction, no module-level dependencies — and only drop it if it becomes a maintenance burden. **Alternative**: drop BYOK-DG, document that BYOK users now bring their OpenAI key (used against LiteLLM passthrough or a separate openai.com Whisper endpoint).

5. **Streaming WS endpoint on rtx6000 — request from the rtx6000 agent now, or wait until Phase B chunked-batch lands and we can measure the actual UX gap?** Recommendation: file the request now (separate task to the rtx6000 agent) but do not block Phase A or B on its delivery. If `faster-whisper-server` or `speaches` is up by the time Phase B finishes, swap the streaming provider class; otherwise keep the chunked-batch shim.

6. **Language scope** — keep the existing `deepgram_nova3_multi_languages` set as the only supported `language=multi` whitelist, or expand to Whisper's full 99-language list? Recommendation: keep today's set in v1 (no UI/UX changes, no user expectations to manage). Expand in a separate spec once Phase D is stable.

7. **Pusher service** — the `pusher/main.py` itself has no remaining DG references (Phase-1 scout confirmed), but `pusher/requirements.txt` still pins `deepgram-sdk==4.8.1`. Confirm: do we drop the dep in Phase A (lower attack surface, fewer build artifacts) or wait until Phase D (parallel removal with the rest of DG code)? Recommendation: Phase D — keep one consistent removal point.

---

*End of spec. Implementation plan to follow once questions 1, 2, 4, 5, 7 are answered.*
