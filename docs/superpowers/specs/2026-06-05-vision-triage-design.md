# Vision Triage Service — `POST /v1/vision/triage` design

**Date:** 2026-06-05
**Status:** Design proposed. No code written yet. Awaiting rtx6000 agent confirmation of the deployment brief (last section) before Phase A standup.
**Owner:** Joe Portman (@jlportman3)
**Scope:** A new rtx6000-hosted FastAPI service that fronts a YOLO open-vocab detector and exposes a single `POST /v1/vision/triage` endpoint. Phone (and future Seeed XIAO) sends every motion-passing frame; service answers "is this interesting?" in <100 ms, and conditionally fans interesting frames asynchronously into the existing `openglass.describe_image()` Qwen-VL pipeline. Touches: **rtx6000 deployment surface** (new service), **omi-backend** (new `backend/utils/vision/triage.py` client + call-site in the photo handler / WS image-chunk path), **YAML rule file** (hot-reloadable decision rules). Out of scope: face DB (deferred — separate cluster), audio integration, on-device wake-word, CLIP retrieval index.

---

## Problem

Today the only "vision intelligence" in the omi stack is `backend/utils/llm/openglass.py:describe_image()` — a single Qwen-VL call against the LiteLLM `openglass` slot. It runs once per uploaded photo. It is comparatively expensive (~1-2 s per call on rtx6000) and it tells you the rich description of every frame whether or not the frame is novel.

The Phone-Cam Glass and Voice-Commanded-Continuous-Multimodal entries in `project_omi_pain_points_to_fix.md` both call for **continuous capture at ~1 fps while a conversation is active**. At that rate, naively calling Qwen-VL on every frame is:

- **Wasteful** — most consecutive frames at 1 fps are visually near-identical. Describing the same scene 60 times in a minute produces 60 nearly-identical "a man at a desk with a coffee mug" descriptions.
- **Slow on the user-visible path** — 1-2 s per call means a 1 fps stream piles up behind the LLM.
- **Vague on classes** — the rich description can't be quickly indexed for "show me the frame where a dog appeared." For that we need detection classes too.
- **GPU-greedy** — Qwen-VL is large enough that doing it for every frame on rtx6000 starves other vision/voice work running on the same GPU.

The 2026-06-03 architecture decision in the memory file (Server-side YOLO triage section) was: insert a **fast YOLO classifier** in front of Qwen-VL. Frames pass through YOLO first (cheap — milliseconds). A hot-reloadable YAML rule decides whether the frame is **interesting enough** to fan onward to Qwen-VL for description. Phone sends every motion-gate-passing frame; rtx6000 decides what gets the expensive treatment.

This spec defines the contract of that service.

---

## Goals

1. **Local-only.** No cloud. Aligns with `project_fully_local_roadmap.md`. Model lives on rtx6000 alongside Qwen-VL, Whisper, voice-extras, Kokoro.
2. **Triage latency target <100 ms** at the wire (POST → YOLO → decision → JSON response). Includes JPEG decode + inference + rule eval.
3. **Async describe.** When the rule says "describe," YOLO returns a `describe_job_id` immediately. The Qwen-VL call runs out-of-band (1-3 s) and the result is polled via `GET /v1/vision/describe/{job_id}` or pushed via the existing photo-described event channel.
4. **Per-user vocabulary.** Each call can override the detection vocabulary (e.g. Joe wants "license plate" classes but a different user wants "guitar"). YOLO-World makes this zero-shot.
5. **Hot-reloadable rules.** YAML on disk; SIGHUP or filesystem-watch reloads without restarting the service.
6. **Single endpoint for two use-cases.** Same `POST /v1/vision/triage` handles both passive 1 fps capture (no `force_describe`) and voice-commanded "Hey Jarvis what is that?" (`force_describe=true`).
7. **Source-agnostic.** Phone today, XIAO ESP32-S3-Sense tomorrow. Whatever produces a JPEG calls the same endpoint.
8. **Backend integration is a thin client.** `backend/utils/vision/triage.py` is the only omi-backend module that needs to know about the rtx6000 service. Existing `/v1/conversations/{id}/photos` and the WS image-chunk path call into it.

## Non-goals

- **Face DB / InsightFace ArcFace.** Person identification is its own cluster (Phase 3 of Voice-Commanded Continuous Multimodal). Not in this spec — but the service is structured so a future `faces` field in the response is additive, not breaking.
- **Audio integration.** The triage service never sees audio. Joining frames to conversation transcripts happens on the backend side, same way the existing photo pipeline already does it.
- **On-device wake-word.** That belongs to the phone app (see "Wake-word visual Q&A" in the pain-points memo).
- **Multi-tenant prioritization.** Single-user deployment for now. Queue + per-uid QoS only becomes interesting once XIAO + phone both stream simultaneously, or a second user is added.
- **A bespoke training pipeline.** YOLO12-nano stock weights and YOLO-World stock weights. No custom training in v1.
- **CLIP retrieval index.** "Show me the frame where I held a coffee mug" is Phase 5 of the continuous-multimodal feature, downstream of this service.

---

## Architecture

```
                                                  rtx6000 (10.0.60.48)
                                              ┌────────────────────────────────┐
phone camera ─┐                               │  /v1/vision/triage  (FastAPI)  │
              │                               │     ├─ JPEG decode             │
   XIAO cam ──┼── 1 fps (configurable) ──┐    │     ├─ YOLO12-nano OR          │
              │                          │    │     │   YOLO-World inference   │
              │                          │    │     ├─ decision rule (YAML)    │
              │                          ▼    │     ├─ enqueue async describe  │
              │            ┌─────────────────────┐  │     (if interesting)     │
              │            │ device-side motion  │  │     └─ return JSON       │
              │            │ gate (no ML,        │  │                          │
              │            │ ~5 ms RGB delta)    │  │  /v1/vision/describe/{id}│
              │            └──────┬──────────────┘  │     ├─ lookup async job  │
              │                   │ pass            │     └─ return desc or    │
              │                   ▼                 │        202-still-running │
              │            POST /v1/vision/triage   │                          │
              │            (multipart jpeg+meta)    │  /v1/vision/rules (GET)  │
              │                                     │     └─ live YAML dump    │
              │                                     │                          │
              │            ┌─────  JSON resp ◄──────┤  /v1/vision/health       │
              │            │       <100 ms          │     └─ model + version   │
              │            ▼                        └────────────────────────────┘
              │      device-side caches             │
              │      next_detection_state                ▲
              │                                          │  internal call
              │                                          │  (loopback or in-process)
              │                                          ▼
              │                                  Qwen-VL via LiteLLM
              │                                  http://10.0.60.48:4000
              │                                  (openglass slot)
              │                                          │
              │                                          ▼
              │                                  description text
              │                                          │
              │                                          ▼
              │                                  cached in describe_job
              │
              ▼  (separately, when describe completes)
       omi-backend (10.0.60.??:8080)
       └─ existing /v1/conversations/{conv_id}/photos
          + new utils/vision/triage.py client
             ├─ triage_frame(jpeg, vocab, force_describe, conv_id)
             └─ describe_result(job_id)
```

### Sequence — passive 1 fps capture

1. Phone app captures a frame, runs a cheap RGB-delta check vs the previous frame. If delta < threshold, drop. (No ML, no network call.)
2. Phone POSTs the frame to `https://omi-backend/v1/conversations/{conv_id}/photos` (existing endpoint). Backend stores the raw JPEG via existing pipeline, then **calls `triage.triage_frame(...)`** before optionally calling `openglass.describe_image()`.

   Alternative: phone POSTs directly to rtx6000 `/v1/vision/triage`, skipping the backend round-trip. Decision: **route through the backend.** Reasons: (a) auth — backend already validates the Firebase token; rtx6000 doesn't; (b) the backend has the conversation context, the privacy LED state, the user's vocabulary preferences; (c) it's the existing capture surface.

3. Backend `triage.triage_frame(jpeg, vocab=<user_vocab>, force_describe=False, conv_id=<id>, previous_detection_state=<opaque>)` posts to rtx6000.
4. rtx6000 YOLO infers in <30 ms, evaluates the rule, returns `{detections: [...], should_describe: bool, decision_reason: str, describe_job_id: str|None, next_detection_state: <opaque str>}`.
5. If `should_describe=False`: backend stores the photo with `description=None`, `objects=detections`, marks `discarded=False`. Done.
6. If `should_describe=True`: backend kicks off a fire-and-forget poll for `/v1/vision/describe/{job_id}`. When it returns (1-3 s later), backend updates the `ConversationPhoto` with the description and emits a `PhotoDescribedEvent` over the WS — same event shape the existing pipeline uses.
7. Backend echoes the YOLO detections to the WS as a new `PhotoTriagedEvent` so the app can show "I see: person, laptop, coffee mug" immediately without waiting for the rich description.

### Sequence — `force_describe=True` (voice command "what is that?")

1. Phone wake-word fires "Hey Jarvis what is that?". Phone snaps a frame.
2. Phone POSTs to backend with `force_describe=true` (probably as a new field on the existing photos endpoint or a new `/v1/conversations/{conv_id}/photos:visual-query` action).
3. Backend calls `triage.triage_frame(force_describe=True)`. rtx6000 runs YOLO **and** enqueues describe regardless of rule outcome — but `decision_reason` says `"force_describe"`.
4. Backend awaits the describe result synchronously (within a 3-4 s budget), returns it to the phone, phone TTSes it.

YOLO output is still useful even when force_describe is set — it gives the LLM a hint for the prompt ("the user is asking about a scene containing: dog, frisbee, grass") which steers Qwen-VL toward the right level of specificity.

---

## Model choice — YOLO12-nano vs YOLO-World

Joe's preference (recorded 2026-06-03 in memory): "look at YOLO12-nano and World, pick one." Here is the side-by-side and the pick.

| Axis | YOLO12-nano (Ultralytics, 2025) | YOLO-World (Tencent AILab, 2024) |
|------|--------------------------------|----------------------------------|
| **License** | AGPL-3.0 (Ultralytics) — fine for Joe's self-hosted single-user deployment, but legally fraught for any future commercial fork | Apache-2.0 (Tencent) — safe everywhere |
| **Vocabulary** | Fixed: 80 COCO classes (person, chair, laptop, cup, dog, cat, car, bottle, ...). No zero-shot. Custom classes require fine-tune. | **Open vocabulary** — accept arbitrary class names at inference time. "license plate", "guitar", "name badge" all work without retraining. |
| **Speed (rtx6000)** | ~5-10 ms per 640px frame, batch=1, FP16 | ~25-40 ms per 640px frame (text encoder + visual encoder + detection head). 2-4× slower than nano. |
| **VRAM** | ~50-80 MB resident (model is <10 MB on disk; activations dominate) | ~400-600 MB resident (CLIP-style text encoder included) |
| **Accuracy on COCO-80** | mAP-val ~40 (nano tier). Better than v8/v11-nano at comparable size per Ultralytics' 2025 release notes — *unverified, benchmark before trust*. | mAP on COCO zero-shot ~35-37 (medium variant). Slightly lower than a class-trained YOLO but acceptable. |
| **Accuracy on Joe-realistic frames (clothing, books, screens, signs, license plates, electronics SKUs)** | Will whiff on anything outside the COCO-80 list. Books = "book"; license plates don't exist as a class; SKU labels don't exist. | Handles open-vocab natively — Joe can say "vocab=['book', 'license plate', 'esp32 board', 'laptop screen']" and get reasonable detections on all four. |
| **Per-user vocabulary** | Impossible without per-user fine-tune. Vocab list in the API contract is decorative. | First-class — the API contract's `vocab` field actually controls inference. |
| **Hot-reload of vocabulary** | Restart with new weights | Just pass a new list per request |
| **Deployment complexity** | Trivial — `pip install ultralytics; YOLO('yolo12n.pt')`. One model file. | A bit heavier — `ultralytics` also ships YOLO-World (`YOLOWorld('yolov8s-worldv2.pt')`). Text encoder warmup ~1 s on first request. |
| **Decision-rule expressiveness** | Limited to COCO-80 classes in `new_class_appearing` rules. Joe's interesting-classes list has to fit inside COCO. | Rules can mention any noun the user cares about. `new_class_appearing: [license_plate, name_badge, ESP32_board]` works. |
| **Future-proofing** | Locked into Ultralytics' release cadence + license | Open weights, multiple maintained forks |

### Recommendation: **YOLO-World (small or medium variant)**

Reasons, in priority order:

1. **Open vocabulary is the load-bearing feature for Joe's use case.** Joe's "interesting" set is not COCO-80. It includes books (which COCO has but doesn't distinguish), license plates (COCO doesn't have), SKU labels, ESP32 boards, name badges, license cards, etc. YOLO12-nano can detect "book" but not "the book Sarah is reading" or "license plate ABC-1234". YOLO-World accepts arbitrary noun phrases per request and that flexibility translates directly into the YAML rule expressiveness.
2. **License sanity.** Apache-2.0 is a non-issue forever. AGPL-3.0 on Ultralytics is fine while everything is self-hosted single-user, but it would block any commercial fork that uses the same inference path. Easier to make the right choice now than rip it out later.
3. **The latency budget tolerates 30 ms.** Target was <100 ms wire-to-wire. JPEG decode is ~5 ms, FastAPI overhead is ~5 ms, network is ~5 ms. That leaves ~85 ms for inference. YOLO-World at 25-40 ms fits comfortably. YOLO12-nano at 5-10 ms is faster but the savings don't unlock anything — there's no downstream stage that benefits from sub-50 ms triage.
4. **VRAM headroom on rtx6000 is sufficient.** Per the memory file, rtx6000 has ~12-16 GB free after Qwen-VL + Whisper + voice-extras + Kokoro. YOLO-World's ~500 MB is rounding error. (Confirm with rtx6000 agent — see Open Questions.)
5. **Per-request vocab eliminates the "what if a new user wants different classes" problem.** No per-user fine-tune nightmare. Just store the vocab list on the user doc.

Use the **small** variant (`yolov8s-worldv2.pt` ≈ 30 ms) as the default. Drop to **medium** if accuracy demands; jump to **large** only if we ever need to run on hand-zoom-cropped patches.

If the rtx6000 agent reports the text-encoder warmup is consistently >300 ms cold, **prewarm** the model with a synthetic dummy request at service startup so the first real request hits a hot encoder.

### Fallback / hybrid option (if YOLO-World benchmarks badly on real frames)

Run **both**. YOLO12-nano on every frame for the cheap "did anything at all change in the COCO-80 dimensions" gate; YOLO-World only on the frames that pass nano's filter. That doubles the inference work but is still cheaper than calling YOLO-World blind on every frame, and it preserves the open-vocab benefit. Cost ~35-50 ms total per frame instead of ~30 ms. This is the **insurance** — only deploy it if Phase A benching shows YOLO-World alone misses too much.

---

## Endpoint contract

### `POST /v1/vision/triage`

**Request** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | binary (JPEG) | yes | Frame bytes. Max 5 MB. Server resizes to model input (640×640 default). |
| `vocab` | JSON string (list of str) | no | Detection vocabulary for YOLO-World. If omitted, server falls back to its configured default vocab (COCO-80 + user-specific additions). Max 200 entries. |
| `force_describe` | bool (string `"true"` / `"false"`) | no, default `false` | If true, bypasses the YAML rule and always returns `should_describe=true`. Voice-command use case. |
| `conversation_id` | str | no | Opaque conversation ID. Returned in the response and used as the storage key for `next_detection_state` deduping. |
| `previous_detection_state` | str (opaque) | no | Server-returned opaque state from the previous `/v1/vision/triage` response for the same `conversation_id`. The rule engine uses this to compute IoU drops, frames-since-last-describe, etc. without keeping per-conversation state server-side. |
| `user_id` | str | no | Optional. Used only for per-user vocab defaulting and per-user rule overlays. Not required when vocab is passed explicitly. |
| `frame_timestamp_ms` | int | no | Capture time in epoch ms. If omitted, server uses receive time. Used by `frames_since_last_describe_gt` rule logic. |

**Response** (200, JSON):

```jsonc
{
  "detections": [
    {"class": "person", "score": 0.91, "bbox": [120, 80, 360, 540]},
    {"class": "laptop", "score": 0.78, "bbox": [400, 280, 600, 460]},
    {"class": "coffee mug", "score": 0.62, "bbox": [310, 320, 380, 410]}
  ],
  "should_describe": true,
  "decision_reason": "new_class_appearing: laptop",
  "describe_job_id": "j_abc123def456",
  "next_detection_state": "<base64 opaque>",
  "frame_id": "f_91827364",
  "conversation_id": "<echoed>",
  "elapsed_ms": {
    "decode": 4,
    "inference": 28,
    "rule_eval": 1,
    "total": 38
  }
}
```

- `detections` — list of all YOLO detections above the configured score threshold (default 0.25). `bbox` is `[x1, y1, x2, y2]` in original frame pixels.
- `should_describe` — boolean from the rule engine. `True` when YAML rule said so or `force_describe=true` was passed.
- `decision_reason` — string. One of: `new_class_appearing: <class>`, `bbox_iou_drop_below: <iou>`, `frames_since_last_describe_gt: <n>`, `heartbeat`, `force_describe`, `no_trigger`. Useful for debugging + telemetry.
- `describe_job_id` — string. Present iff `should_describe=true`. The caller polls `GET /v1/vision/describe/{job_id}` for the Qwen-VL result.
- `next_detection_state` — opaque base64 string. Caller stores it and passes it back as `previous_detection_state` on the next call for the same `conversation_id`. Keeps the rule engine stateless.
- `frame_id` — server-assigned. Useful as a debug correlation handle.
- `elapsed_ms` — observability. Helps tune.

**Error responses:**

- 400 — malformed body (no JPEG, vocab too long, etc.)
- 413 — frame >5 MB
- 503 — YOLO model not loaded / warming up
- 504 — internal queue backed up; client should drop the frame and try again next tick

**Latency SLO:** p50 <50 ms, p95 <100 ms, p99 <200 ms (for triage only — does not include describe). If a request exceeds 200 ms, server logs a slow-frame warning with `elapsed_ms` broken out.

### `GET /v1/vision/describe/{job_id}`

**Response** (200, JSON) — describe complete:

```jsonc
{
  "job_id": "j_abc123def456",
  "status": "complete",
  "description": "A wooden desk with a closed silver laptop and a half-full coffee mug, viewed from a first-person perspective. The mood is calm and focused.",
  "model": "qwen-vl",
  "elapsed_ms": 1830
}
```

**Response** (202, JSON) — still running:

```jsonc
{
  "job_id": "j_abc123def456",
  "status": "pending",
  "queued_ms": 12,
  "eta_ms": 1500
}
```

**Response** (404, JSON) — unknown job_id or expired (TTL 60 s):

```jsonc
{"error": "job not found or expired"}
```

The caller is expected to long-poll or short-poll with exponential backoff. 250 ms initial poll, doubling to a 2 s cap, with a hard timeout of 5 s. (Backend's `triage.describe_result()` client encapsulates this.)

### `GET /v1/vision/rules`

Returns the currently-loaded YAML as JSON. Useful for the backend to surface "current triage policy" in admin UIs and for the rtx6000 agent to sanity-check what's live.

### `GET /v1/vision/health`

Returns `{model: 'yolov8s-worldv2', model_loaded: true, vocab_default_size: 80, rules_loaded_at: <ts>, gpu: 'cuda:0', vram_mb: 487}`. Health probe target for the omi-backend client's circuit breaker.

### `POST /v1/vision/reload-rules`

Force a YAML reload. Optional — filesystem-watch reload is the primary mechanism. Useful from a sysadmin shell.

---

## YAML rule schema

Rules live in `~rtx6000/vision-triage/rules.yaml`. Filesystem-watched (inotify or polling at 5 s) — edits go live with no restart. Schema:

```yaml
# /opt/vision-triage/rules.yaml
version: 1                                  # bumps on breaking schema change

# Default vocabulary when caller doesn't pass `vocab`.
default_vocab:
  - person
  - laptop
  - coffee mug
  - book
  - dog
  - cat
  - license plate
  - name badge
  - phone
  - keyboard

# A trigger fires whenever ANY of the listed conditions evaluate true for
# the current frame (vs previous_detection_state). Evaluation is OR-of-rules,
# AND-of-conditions-inside-a-rule.
triggers:
  - name: new_interesting_class
    when:
      new_class_appearing:
        - person
        - vehicle
        - dog
        - cat
        - license plate
        - name badge
    # When this rule fires, the JSON response carries `decision_reason: "new_class_appearing: <class>"`.

  - name: scene_change
    when:
      bbox_iou_drop_below: 0.3              # if ANY detection's bbox IoU vs the previous frame drops below this, fire
    # Catches "the person moved across the room", "the laptop closed", etc.

  - name: heartbeat
    when:
      frames_since_last_describe_gt: 60     # if 60 frames (at 1 fps = 60 s) have passed since last describe, fire
    # Ensures the LLM gets at least one describe per minute even if nothing changes.

  - name: scheduled_heartbeat
    when:
      heartbeat_describe_every_sec: 300     # absolute wall-clock heartbeat — fire if >300 s since last describe regardless of frame count
    # Useful when fps is variable (e.g. user switched the capture rate).

# Score threshold for YOLO detections — anything below is filtered before rule eval.
yolo_score_threshold: 0.25

# Per-user overlays. Looked up by `user_id` field on the request.
# Overlays are deep-merged onto the defaults.
user_overlays:
  jlportman3:
    default_vocab:
      additions:
        - ESP32 board
        - raspberry pi
        - oscilloscope
        - solder iron
    triggers:
      - name: new_interesting_class
        when:
          new_class_appearing:
            additions:
              - ESP32 board
              - oscilloscope

  jarvis_test:
    default_vocab:
      additions:
        - bookshelf
        - amplifier rack

# Hot-reload knobs.
hot_reload:
  watch_filesystem: true                    # inotify on rules.yaml
  poll_interval_sec: 5                      # fallback polling if inotify unavailable
  signal: SIGHUP                            # also reload on signal

# Observability.
observability:
  log_decision_reasons: true                # log every triage's decision_reason at INFO
  log_slow_frames_above_ms: 150             # log a warning if total elapsed > 150ms
  metrics_endpoint: /metrics                # Prometheus-compatible /metrics
```

### Worked example — a typical 5-frame sequence

Assume `vocab = default_vocab`, `previous_detection_state` carries the last detections + the "frames since last describe" counter.

| Frame | Detections | `should_describe`? | `decision_reason` |
|------:|------------|-------------------:|-------------------|
| 1 | [person] | yes | `new_class_appearing: person` (first frame after a long silence — no prev state) |
| 2 | [person] | no | `no_trigger` (same class, low IoU drop, recent describe) |
| 3 | [person, laptop] | yes | `new_class_appearing: laptop` (laptop wasn't in frame 2) |
| 4 | [person, laptop] | no | `no_trigger` |
| ... | ... | ... | ... |
| 63 | [person, laptop] | yes | `frames_since_last_describe_gt: 60` (heartbeat) |

### Hot-reload mechanism

Primary: inotify watch on `rules.yaml`. On `IN_CLOSE_WRITE` or `IN_MOVED_TO`, re-parse and atomically swap the loaded rule set. Failure to parse logs an error and **keeps the previous rules live** — never goes naked.

Fallback (containers without inotify, e.g. some Docker on overlayfs): poll the file mtime every 5 s. Same atomic-swap on change.

Manual: send the service `SIGHUP` (handled in the FastAPI startup) or `POST /v1/vision/reload-rules`.

Validation: a CI job (or pre-commit hook on the rtx6000 host) runs `python -m vision_triage.validate_rules rules.yaml` before any change is saved.

---

## omi-backend integration surface

### New module: `backend/utils/vision/triage.py`

```python
# backend/utils/vision/triage.py
import base64
import os
from typing import Optional

import httpx

from utils.http_client import get_default_async_client  # existing shared pool

VISION_TRIAGE_BASE_URL = os.environ.get('VISION_TRIAGE_BASE_URL', 'http://10.0.60.48:8095')
VISION_TRIAGE_TIMEOUT_SEC = float(os.environ.get('VISION_TRIAGE_TIMEOUT_SEC', '2.0'))
VISION_TRIAGE_ENABLED = os.environ.get('VISION_TRIAGE_ENABLED', 'false').lower() == 'true'


async def triage_frame(
    jpeg_bytes: bytes,
    *,
    vocab: Optional[list[str]] = None,
    force_describe: bool = False,
    conversation_id: Optional[str] = None,
    previous_detection_state: Optional[str] = None,
    user_id: Optional[str] = None,
    frame_timestamp_ms: Optional[int] = None,
) -> dict:
    """POST a frame to rtx6000 /v1/vision/triage.

    Returns the parsed JSON response. Raises httpx.HTTPError on transport failure;
    caller should wrap in a try and fall through to the legacy describe-everything
    behavior on error (graceful degradation).
    """
    ...


async def describe_result(job_id: str, *, max_wait_sec: float = 5.0) -> Optional[str]:
    """Poll /v1/vision/describe/{job_id} until complete or max_wait_sec elapses.

    Returns the description string on success, None on timeout / failure.
    Uses 250 ms initial poll, doubling to 2 s, capped at max_wait_sec total.
    """
    ...


def get_user_vision_vocab(uid: str) -> Optional[list[str]]:
    """Look up the user's configured detection vocabulary, if any. Cached.

    Returns None when the user has no overlay — server falls back to default.
    """
    ...
```

Constraints:

- **No in-function imports** (CLAUDE.md rule).
- **Async-safe** — uses `httpx.AsyncClient` from `utils/http_client.py`, with a semaphore-bounded vision-triage pool. Add a `get_vision_triage_client()` and `get_vision_triage_semaphore()` in `utils/http_client.py`.
- **Circuit breaker** — wrap the call in `get_vision_triage_circuit_breaker()`. On open circuit, `triage_frame()` returns a synthetic "always describe" response so the existing `openglass.describe_image()` path still runs. Triage is an optimization; failure must degrade to current behavior.
- **Black formatting**, line length 120.

### Call site: `backend/routers/transcribe.py::process_photo`

Current code (verified above, line 2351-2367):

```python
async def process_photo(uid: str, image_b64: str, temp_id: str, send_event_func, photo_buffer):
    from utils.llm.openglass import describe_image
    photo_id = str(uuid.uuid4())
    await send_event_func(PhotoProcessingEvent(temp_id=temp_id, photo_id=photo_id))
    try:
        description = await describe_image(uid, image_b64)
        discarded = not description or not description.strip()
    except Exception as e:
        logger.error(f"Error describing image: {e} {uid} {session_id}")
        description = "Could not generate description."
        discarded = True
    final_photo = ConversationPhoto(id=photo_id, base64=image_b64, description=description, discarded=discarded)
    photo_buffer.append(final_photo)
    await send_event_func(PhotoDescribedEvent(photo_id=photo_id, description=description, discarded=discarded))
```

After integration (top-level imports per CLAUDE.md):

```python
# at top of file:
from utils.vision import triage

# inside transcribe_socket scope, alongside the other realtime buffers:
last_triage_state_by_conv: dict[str, str] = {}  # conv_id → opaque state

async def process_photo(uid: str, image_b64: str, temp_id: str, send_event_func, photo_buffer, conv_id: str | None):
    photo_id = str(uuid.uuid4())
    await send_event_func(PhotoProcessingEvent(temp_id=temp_id, photo_id=photo_id))

    description: str | None = None
    discarded = True
    detections: list[dict] = []
    decision_reason = "triage_disabled"

    if triage.VISION_TRIAGE_ENABLED:
        try:
            jpeg_bytes = base64.b64decode(image_b64)
            user_vocab = triage.get_user_vision_vocab(uid)
            triage_result = await triage.triage_frame(
                jpeg_bytes,
                vocab=user_vocab,
                conversation_id=conv_id,
                previous_detection_state=last_triage_state_by_conv.get(conv_id) if conv_id else None,
                user_id=uid,
            )
            detections = triage_result.get('detections', [])
            decision_reason = triage_result.get('decision_reason', 'unknown')
            if conv_id:
                last_triage_state_by_conv[conv_id] = triage_result.get('next_detection_state', '')

            await send_event_func(PhotoTriagedEvent(photo_id=photo_id, detections=detections, decision_reason=decision_reason))

            if triage_result.get('should_describe'):
                job_id = triage_result.get('describe_job_id')
                description = await triage.describe_result(job_id) if job_id else None
                discarded = not description or not description.strip()
            else:
                # YOLO said "boring frame" — store it, no LLM call
                description = None
                discarded = False  # not discarded; we just didn't describe
        except Exception as e:
            logger.error(f"Vision triage failed, falling back to direct describe: {e} {uid} {session_id}")
            # graceful fall-through to legacy path
            description = await describe_image(uid, image_b64)
            discarded = not description or not description.strip()
    else:
        # Triage disabled — original behavior
        try:
            description = await describe_image(uid, image_b64)
            discarded = not description or not description.strip()
        except Exception as e:
            logger.error(f"Error describing image: {e} {uid} {session_id}")
            description = "Could not generate description."
            discarded = True

    final_photo = ConversationPhoto(
        id=photo_id,
        base64=image_b64,
        description=description,
        discarded=discarded,
        detections=detections,
        decision_reason=decision_reason,
    )
    photo_buffer.append(final_photo)
    await send_event_func(PhotoDescribedEvent(photo_id=photo_id, description=description or "", discarded=discarded))
```

### Model changes: `backend/models/conversation_photo.py`

Add (optional, default `None`/`[]`) fields to `ConversationPhoto`:

- `detections: list[dict] | None = None` — YOLO output, list of `{class, score, bbox}` dicts.
- `decision_reason: str | None = None` — why describe ran (or didn't). Audit trail.
- `triaged_at_ms: int | None = None` — server receive time for the triage call.

Backward compatible — old clients ignore unknown fields; old DB docs don't have these set.

### New WS event: `PhotoTriagedEvent`

Fired before `PhotoDescribedEvent` so the app can show "I see: person, laptop" immediately while waiting (or never waiting if `should_describe=False`) for the full description. Shape:

```python
class PhotoTriagedEvent(BaseModel):
    event: Literal['photo_triaged'] = 'photo_triaged'
    photo_id: str
    detections: list[dict]
    decision_reason: str
```

### Env vars

Add to `backend/.env.template`:

```bash
# Vision triage (rtx6000)
VISION_TRIAGE_ENABLED=false
VISION_TRIAGE_BASE_URL=http://10.0.60.48:8095
VISION_TRIAGE_TIMEOUT_SEC=2.0
```

`VISION_TRIAGE_ENABLED=false` by default keeps existing behavior untouched. Flip to `true` after Phase A+B+C land.

### `utils/http_client.py` additions

- `get_vision_triage_client()` — `httpx.AsyncClient` with `timeout=httpx.Timeout(2.0, connect=0.5)`.
- `get_vision_triage_semaphore()` — `asyncio.Semaphore(8)` keyed per event loop (existing pattern).
- `get_vision_triage_circuit_breaker()` — open after 5 consecutive failures, half-open after 30 s.

### `scripts/lint_async_blockers.py`

No changes — this is an async-only path.

### Tests

Add `backend/tests/unit/test_vision_triage_client.py`:

- Mock `httpx.AsyncClient.post` → assert URL, multipart fields, headers.
- Mock 200 response → assert parsed dict.
- Mock 503 → assert raises and circuit breaker opens after 5.
- Mock describe poll: 202, 202, 200 sequence → assert description returned and elapsed time within bound.
- Mock describe timeout → assert returns `None` after `max_wait_sec`.

Add to `backend/test.sh` runner list (CI source of truth).

---

## rtx6000 deployment brief (for the rtx6000 agent)

> **This section is the standalone copy-pasteable brief for the rtx6000 agent. Hand the whole section over verbatim.**

### What you're standing up

A new FastAPI service on rtx6000 exposing `POST /v1/vision/triage` and `GET /v1/vision/describe/{job_id}`. Triage = fast YOLO classification + YAML rule eval. Describe = async fan-out to the existing Qwen-VL via LiteLLM. The omi-backend (10.0.60.xx) is the only caller.

### Suggested code layout on rtx6000

```
/opt/vision-triage/
  pyproject.toml
  rules.yaml                       # the hot-reloadable decision rules
  src/vision_triage/
    __init__.py
    main.py                        # FastAPI app, startup, lifespan
    config.py                      # env-var loading
    yolo_engine.py                 # ultralytics YOLOWorld wrapper
    rules.py                       # YAML parse + evaluation against detection state
    state.py                       # opaque next_detection_state codec (msgpack+base64)
    describe.py                    # async Qwen-VL job manager (in-memory dict, 60s TTL)
    routes/
      triage.py                    # POST /v1/vision/triage
      describe.py                  # GET /v1/vision/describe/{job_id}
      health.py                    # /health, /metrics
      rules.py                     # GET /v1/vision/rules + POST /v1/vision/reload-rules
    schemas.py                     # pydantic request/response models
    watcher.py                     # filesystem watcher for rules.yaml
    metrics.py                     # prometheus_client gauges/histograms
  tests/
    test_rule_eval.py
    test_triage_roundtrip.py
```

### Install

```bash
# rtx6000 python env (confirm which interpreter — there's at least one venv at /opt/voice-extras-venv;
# either reuse or create /opt/vision-triage-venv if isolation is preferred)
python -m venv /opt/vision-triage-venv
source /opt/vision-triage-venv/bin/activate

pip install -U pip
pip install fastapi uvicorn[standard] httpx pydantic pyyaml python-multipart \
            ultralytics torch torchvision Pillow \
            prometheus-client watchdog msgpack

# YOLO-World weight — small variant
python -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8s-worldv2.pt')"
# (this downloads ~75 MB to ~/.cache/ultralytics/, or however your hosts cache models)
```

### Configure

`/opt/vision-triage/rules.yaml` — start with the example schema in the "YAML rule schema" section above.

Env (`/etc/default/vision-triage` or systemd EnvironmentFile):

```bash
VISION_TRIAGE_BIND_HOST=0.0.0.0
VISION_TRIAGE_BIND_PORT=8095
VISION_TRIAGE_MODEL=yolov8s-worldv2.pt
VISION_TRIAGE_DEVICE=cuda:0
VISION_TRIAGE_RULES_PATH=/opt/vision-triage/rules.yaml
VISION_TRIAGE_DESCRIBE_BACKEND=litellm
LITELLM_BASE_URL=http://127.0.0.1:4000
LITELLM_API_KEY=<reuse the existing local key>
VISION_TRIAGE_DESCRIBE_MODEL=qwen-vl     # whatever the openglass slot resolves to via LiteLLM
VISION_TRIAGE_DESCRIBE_JOB_TTL_SEC=60
VISION_TRIAGE_DESCRIBE_MAX_TOKENS=150     # mirror openglass.describe_image default
```

### systemd unit (suggested)

```ini
# /etc/systemd/system/vision-triage.service
[Unit]
Description=Vision Triage (YOLO-World + Qwen-VL fanout)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/default/vision-triage
WorkingDirectory=/opt/vision-triage
ExecStart=/opt/vision-triage-venv/bin/uvicorn vision_triage.main:app \
          --host ${VISION_TRIAGE_BIND_HOST} --port ${VISION_TRIAGE_BIND_PORT} \
          --workers 1 --no-access-log
Restart=on-failure
RestartSec=5
# GPU access:
SupplementaryGroups=video render

[Install]
WantedBy=multi-user.target
```

(One worker is sufficient — model is a singleton, async dispatch handles concurrency, batching is intentionally out of scope for v1.)

### Port

**8095** unless you already host something there. Tell the omi-backend agent the chosen port so `VISION_TRIAGE_BASE_URL` can be set.

### Expected footprint

| Resource | Steady-state | Burst |
|----------|-------------:|------:|
| VRAM (YOLO-World small) | ~500 MB | ~700 MB on first call (encoder warmup) |
| VRAM (Qwen-VL describe) | already live in the openglass LiteLLM slot — no additional draw | — |
| CPU | ~0.5 core idle, ~1 core at 1 fps | ~2 cores at 5 fps |
| RAM | ~600 MB | up to ~1 GB with 60 in-flight describe jobs |
| Disk | ~100 MB (model weights + venv) | — |
| Network | ingress only (no egress except loopback LiteLLM) | — |

### Validation checklist

1. `curl http://localhost:8095/v1/vision/health` returns `{model_loaded: true, ...}` within 10 s of service start.
2. `curl -F 'file=@frame.jpg' -F 'vocab=["person","laptop"]' http://localhost:8095/v1/vision/triage` returns detections within 100 ms.
3. With `force_describe=true`, the response carries a `describe_job_id`; `GET /v1/vision/describe/{job_id}` eventually returns a Qwen-VL description.
4. Edit `rules.yaml`, save; within 5 s the next triage call reflects the new rule (verify via `decision_reason` field).
5. Send 100 frames at 1 fps from a script — verify the heartbeat rule fires at the configured interval and no rule eval takes >2 ms.

### What to report back

- Confirmed bound port.
- Steady-state VRAM after warmup, and total free VRAM remaining for Qwen-VL + Whisper + voice-extras + Kokoro.
- Cold-start latency from systemd start → `model_loaded: true`.
- Any deviations from the spec (e.g. forced to use YOLO12-nano because YOLO-World benched poorly on real frames).

---

## Open questions

1. **Does rtx6000 already have `ultralytics` in some venv, or do we provision a new one?** If a clean install is required, confirm Python version (>=3.10 for ultralytics 8.3+) and CUDA toolkit availability.
2. **VRAM after current load.** Memory file estimates 12-16 GB free after Qwen-VL + Whisper + voice-extras + Kokoro. Need a hard number from the rtx6000 agent before committing YOLO-World vs nano. If <2 GB free, fall back to YOLO12-nano.
3. **FastAPI vs Triton.** v1 is single-user, FastAPI is fine. If/when XIAO + phone stream concurrently from the same uid (or a second user is added), Triton's dynamic batching becomes attractive. Decide at Phase D, not now.
4. **Co-locate on LiteLLM host (port 4000) or its own port (8095).** Recommendation: own port. LiteLLM is a proxy with its own auth model and request schema; bolting a custom YOLO service into the same process complicates both. Separate service, separate port, internal loopback for the Qwen-VL fanout.
5. **JPEG decode on the YOLO server or pre-decoded by the backend?** v1: server decodes. Saves backend memory and keeps the wire format simple. If backend ever sends pre-decoded tensors (e.g. for batch fan-in), add a `Content-Type: application/x-vision-tensor` codepath later.
6. **YOLO-World vocab caching.** Calling `model.set_classes(['a', 'b', 'c'])` recomputes text-encoder embeddings each time. For per-request vocab changes that hurts. Cache by `tuple(sorted(vocab))` → embedding tensor in an LRU of size ~64. Reset on model reload.
7. **Describe queue depth ceiling.** When the rule fires harder than Qwen-VL can keep up (e.g. user opens a busy storefront), describe jobs pile up. Bound the in-memory job dict at 60 entries; drop oldest on overflow, set `decision_reason = "describe_dropped_queue_full"` on subsequent triage responses. Surfaces to operator that triage is too liberal.
8. **Privacy LED / consent state.** Backend, not rtx6000, holds privacy policy. If consent is revoked mid-conversation, backend simply stops calling triage. rtx6000 has no PII because it discards the JPEG after inference + describe (does NOT persist frames).
9. **Telemetry export.** `/metrics` Prometheus endpoint — what exists for scraping on rtx6000? If nothing yet, log-only is fine for v1; revisit when broader rtx6000 observability lands.
10. **YOLO-World benchmark on real Joe-style frames.** Joe should hand the rtx6000 agent 10-20 sample frames (BLE pendant or phone capture) to bench YOLO-World small vs medium vs YOLO12-nano on for accuracy in his actual environment **before** the recommendation in the "Model choice" section is locked in.

---

## Phasing

### Phase A — rtx6000 service skeleton (1-2 days; rtx6000 agent owns)

1. Stand up `/opt/vision-triage` skeleton with FastAPI, ultralytics YOLO-World load at startup.
2. `/v1/vision/triage` with stubbed rules (single hard-coded "always describe" rule) — proves the YOLO-World inference path end-to-end.
3. `/v1/vision/describe/{job_id}` calling LiteLLM `openglass` slot — proves the async fanout works.
4. `/v1/vision/health` returning model + version.
5. systemd unit + manual `curl` validation per checklist above.

**Exit:** rtx6000 agent reports steady-state VRAM and a successful 1-frame round trip in <100 ms triage + <3 s describe.

### Phase B — YAML rule engine (1-2 days; rtx6000 agent owns)

1. `rules.py` — parse, validate, evaluate against `previous_detection_state`.
2. `state.py` — opaque codec (msgpack + base64) so backend stays stateless.
3. `watcher.py` — inotify + polling fallback. Atomic swap with parse-failure rollback.
4. Per-user overlays.
5. Per-vocab LRU on the YOLO-World text encoder.
6. Unit tests for each trigger type.

**Exit:** all four trigger types fire correctly in unit tests. Edit-and-reload round-trip <5 s.

### Phase C — backend client + tests (1 day; omi-backend agent owns)

1. `backend/utils/vision/triage.py` with `triage_frame`, `describe_result`, `get_user_vision_vocab`.
2. `backend/utils/http_client.py` additions (client + semaphore + circuit breaker).
3. `backend/models/conversation_photo.py` field additions.
4. `backend/tests/unit/test_vision_triage_client.py` + add to `test.sh`.
5. `.env.template` additions.

**Exit:** unit tests green; `pytest` clean; black-formatted.

### Phase D — wire into photo handler (1 day; omi-backend agent owns)

1. Update `routers/transcribe.py::process_photo` to call triage when `VISION_TRIAGE_ENABLED=true`.
2. New `PhotoTriagedEvent` WS event.
3. Graceful degradation: triage failure → fall through to legacy `describe_image()`.
4. Per-conv `last_triage_state_by_conv` cache in the WS scope.
5. End-to-end smoke: send a single photo, confirm triage call + describe poll + WS events.

**Exit:** Joe captures 5 minutes of phone camera at 1 fps; confirms describe runs only on the interesting frames (verifiable via decision_reason); BLE conversation timeline shows correct interleaving with detections + descriptions.

### Phase E — observability + tuning (ongoing)

1. Log `decision_reason` aggregates per session — pareto-curve which rules fire most.
2. Track describe queue depth, slow-frame counts.
3. Tune `rules.yaml` based on real usage. (E.g. "heartbeat every 60 s is too chatty — bump to 180 s.")
4. Iterate the per-user vocab overlay for jlportman3 as Joe surfaces missing classes.

### Phase F (deferred) — face DB integration

When InsightFace work lands (separate cluster), extend triage response with a `faces: [{embed, bbox, identity_guess}]` field. Backend resolves identities via Qdrant lookup. No breaking change to the v1 endpoint contract — `faces` is just a new optional field.

### Phase G (deferred) — XIAO ESP32-S3-Sense as additional camera source

When the XIAO build lands, point its capture HTTP POST at the same backend endpoint. Backend's `process_photo` is camera-source-agnostic — XIAO frames flow through the identical triage path. The only XIAO-specific concern is the smaller frame size (the OV2640 caps at ~1600×1200 but XIAO typically streams ~640×480) — well below the 5 MB cap.

---

## Risks

- **YOLO-World accuracy on real frames is unverified.** Mitigation: bench before locking in (Open Question 10). Fallback path: YOLO12-nano + hybrid two-stage gating.
- **Cold-start latency on first request after deploy.** Mitigation: prewarm with a synthetic dummy frame in the startup hook.
- **Describe queue blowup if rules are too liberal.** Mitigation: bounded in-memory job dict (Open Question 7) + telemetry surfaces the problem before it OOMs.
- **Vocabulary mismatch between rule classes and YOLO output classes.** YOLO-World produces exactly the classes you asked for, but the YAML rule's `new_class_appearing` list must reference the SAME strings. Mitigation: validate-rules step rejects rules that reference classes not in any known default vocab or per-user overlay.
- **Backend graceful-degradation correctness.** If triage fails, falling back to legacy describe is correct only as long as the legacy path itself is healthy. Mitigation: existing describe error-handling is preserved verbatim; triage only adds a fast-path, never removes the slow-path fallback.
- **Privacy.** rtx6000 must NOT persist JPEGs. Confirm: `describe.py` discards bytes after Qwen-VL returns; no on-disk frame cache. Audit the systemd unit's `ReadWritePaths` to be empty except for `/var/log/vision-triage`.

---

## Related specs / memories

- `~/.claude/projects/-mypool-home-baron-omi/memory/project_omi_pain_points_to_fix.md` — Forward Features → Phone-Cam Glass, Wake-word Visual Q&A, Voice-Commanded Continuous Multimodal, **Server-side YOLO triage** (the seed of this design).
- `~/.claude/projects/-mypool-home-baron-omi/memory/project_fully_local_roadmap.md` — local-only mandate.
- `~/.claude/projects/-mypool-home-baron-omi/memory/feedback_qwen_vision_capable.md` — confirms Qwen-VL is the local vision LLM and lives behind the openglass LiteLLM slot.
- `docs/superpowers/specs/2026-06-01-stt-migration-deepgram-to-local.md` — sibling rtx6000-migration spec; same adapter-and-graceful-degradation pattern.
- `docs/superpowers/specs/2026-05-31-memory-attribution-and-triage-design.md` — provenance pattern the new `detections` + `decision_reason` fields on `ConversationPhoto` follow.
