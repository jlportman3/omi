# Memory Attribution and Autonomous Triage — Design

**Date:** 2026-05-31
**Status:** Design approved (brainstormed 2026-05-31). Layer 1 (voiceprint bank) deployed live to Jarvis on 2026-05-31 — 141 embeddings, calibrated_threshold=0.5430, version `bank-20260531-141828-11min-jarvis-enrollment`. Layers 2-6 pending implementation plan.
**Owner:** Joe Portman (@jlportman3)
**Scope:** Backend changes spanning `utils/stt/speaker_embedding.py`, `routers/transcribe.py`, `utils/conversations/process_conversation.py`, `database/memories.py`, `utils/prompts.py`, plus new `utils/memory_critic.py` and `scripts/rediarize.py`

---

## Problem

User-reported 2026-05-31: *"Jarvis has been asking some questions and getting some really bizarre answers from the memories. Anything Omi hears it sometimes attributes to the user which is completely incorrect, so a bunch of false memories / phony memories / incorrect memories get added."*

In practice: omi captures audio that includes TV, ambient speakers, other people in the room. The live speaker-identification pipeline (~70-80% accurate on BLE-pendant audio per cluster #3) sometimes mis-tags non-user audio as `is_user=True`. The memory extractor then faithfully turns those segments into "facts about the user," which downstream chat / RAG / persona surfaces treat as authoritative — producing bizarre answers.

The fix has to be **layered**: improve attribution at the source, gate extraction more tightly, track provenance so phantoms are surgically removable, run autonomous quality control, and retroactively clean up the existing pollution (11,287 conversations + 8,530 memories already imported under the old pipeline on the Jarvis test account).

## Goals

1. Phantom-memory generation rate drops to near-zero on new conversations.
2. Existing memories can be re-evaluated retroactively when the voiceprint or attribution algorithm improves.
3. Memory health is maintained **autonomously** — the user spot-checks, doesn't manually triage each memory.
4. All automated actions are auditable and reversible.
5. The new attribution pipeline works against Deepgram-clustered audio TODAY (no dependency on Phase B Whisper-live-swap landing first).

## Non-goals

- Phase B (replacing Deepgram in the live BLE streaming WebSocket) is a separate track. This design improves attribution that runs against whatever transcription source is in place.
- A mobile-app triage UI. The daemon makes one unnecessary for the test deployment; mobile UI is a Phase 2 ship when production needs warrant.
- Per-fact LLM debate / argumentation. The critic is a single LLM scoring pass, not an adversarial system.

## Background — current pipeline

`routers/transcribe.py:speaker_identification_task` runs against Deepgram-clustered speaker IDs:

1. Deepgram (live) emits transcript words with `speaker_id` integer clusters
2. For each cluster: extract audio segment, POST to voice-extras `/v1/embeddings` for a TitaNet 192-d embedding
3. Cosine-compare against the user's single enrolled `speaker_embedding` (a ~17s recording averaged into one vector)
4. If distance < fixed `SPEAKER_MATCH_THRESHOLD=0.45` → entire cluster tagged `is_user=True`
5. Same flow but against enrolled people's embeddings for person_id assignment
6. Memory extractor in `utils/conversations/process_conversation.py` filters to `is_user=True` segments only, sends them to LLM with the `extract_memories_prompt` in `utils/prompts.py`, writes extracted facts to `users/{uid}/memories/{memory_id}` in Firestore

Failure modes producing phantom memories:
- **Single-sample voiceprint variance**: 17s enrollment doesn't cover vocal range; novel audio conditions cross the threshold by chance
- **Cluster-merge propagation**: Deepgram merges two real speakers into one cluster; whole merged cluster gets the user's tag if the merged embedding lands close enough
- **Fixed binary threshold**: no confidence weighting; one match decides the fate of every segment in the cluster
- **No first-person verification**: extractor LLM trusts what segments say even when there's no explicit user-asserted statement
- **No corroboration requirement**: one mention → one memory, even from a single low-confidence segment

## Design overview

Six layers, each composable, each with its own implementation surface:

| Layer | Responsibility | Touches |
|-------|---------------|---------|
| 1. Voiceprint | Multi-sample TitaNet bank with calibrated threshold | New fields on user doc; new enrollment flow |
| 2. Per-segment algorithm | Per-segment attribution with corroboration + cluster split | `routers/transcribe.py:speaker_identification_task`, `utils/stt/speaker_embedding.py` |
| 3. Provenance schema | Source-segment + confidence tracking on every memory | `models/memory.py`, `database/memories.py` |
| 4. Extraction guardrails | First-person + corroboration check; reject hallucinated sources | `utils/prompts.py`, `utils/conversations/process_conversation.py` |
| 5. Memory health daemon | Heuristic inline + LLM critic background sweep; autonomous actions | New `utils/memory_critic.py`; existing `modal/` cron pattern |
| 6. Re-diarize batch | Tiered priority sweep with parallel workers | New `scripts/rediarize.py`; new fields on conversation doc |

---

## Layer 1: Voiceprint — embedding bank with calibrated threshold

### Enrollment

User records **10-15 minutes of continuous speech in a quiet room** via the existing app conversation flow. Audio chunks land in `omi-mine-private-cloud-sync/chunks/{uid}/{conv_id}/*.batch.enc` as normal.

A one-shot enrollment script identifies the long quiet-room conversation, decrypts the chunks, concatenates them into one audio stream, and extracts ~120 embeddings via a **10s sliding window with 5s overlap** (each window POSTed to voice-extras `/v1/embeddings` for a TitaNet 192-d vector).

### Voiceprint structure

Stored on the user doc as a new field (does not disturb the legacy `speaker_embedding` field):

```python
voiceprint_bank: {
    'version': 'bank-2026-05-31-141828-11min-jarvis-enrollment',  # human-readable identifier
    'created_at': <timestamp>,
    'source_conversation_ids': [<conv_id>, ...],    # list — enrollment can be split across multiple recordings
    'window_seconds': 10,
    'window_overlap_seconds': 5,
    'sample_rate': 16000,
    'embeddings': [                                  # NOTE: array of maps, not array of arrays
        {'v': [192 floats]},                         # Firestore disallows arrays directly nested in arrays;
        {'v': [192 floats]},                         # wrapping each embedding in a {'v': ...} map dodges
        ...                                          # the restriction without splitting into a subcollection
    ],
    'calibrated_threshold': 0.54,                   # 95th percentile of intra-bank pairwise cosine distance + 0.05 margin
    'continual_samples': [],                        # same {'v': [192 floats]} shape when populated
    'baseline_centroid': [192 floats],              # flat single-level array — Firestore allows this
    'stats': {                                       # observability snapshot at build time
        'n_embeddings': 141,
        'total_duration_s': 710.9,
        'intra_pairwise_min': 0.032,
        'intra_pairwise_mean': 0.234,
        'intra_pairwise_p95': 0.493,
        'intra_pairwise_max': 0.642,
    },
}
```

**Firestore nesting note:** Firestore rejects arrays-directly-inside-arrays (`[[…], […]]`). All embedding lists are wrapped in `{'v': [...]}` maps so the outer container is `array of maps`, which Firestore allows. The verified live bank for Jarvis (built 2026-05-31, 141 embeddings) uses this shape successfully — do NOT "simplify" back to `[[…], […]]` in implementations.

Same schema applies to enrolled people (`users/{uid}/people/{person_id}.voiceprint_bank`), populated when sufficient speech samples accumulate.

### Matching

For each query embedding (a segment to be attributed):

```python
def match_user(query_embedding, voiceprint_bank) -> tuple[float, bool]:
    # Unwrap the map-wrap on read
    bank_vectors = [e['v'] for e in voiceprint_bank['embeddings'] + voiceprint_bank.get('continual_samples', [])]
    distances = [cosine(query_embedding, v) for v in bank_vectors]
    nearest_distance = min(distances)
    is_match = nearest_distance < voiceprint_bank['calibrated_threshold']
    return nearest_distance, is_match
```

K-NN against ~120-200 vectors is microseconds in numpy. Threshold is **per-user, data-driven** — anyone outside your natural intra-voice variance is rejected.

### Continual learning + drift detection

Segments tagged is_user=True with **very high confidence** (distance < 0.2 AND corroboration_count >= 3 AND critic-confirmed) get their embedding appended to `continual_samples` (capped at 50 most-recent).

Periodically, compute the centroid of `continual_samples` and compare to `baseline_centroid`. If divergence > threshold → emit drift alert ("your voiceprint has drifted; consider re-enrolling"). This catches both natural vocal change AND contamination.

---

## Layer 2: Per-segment attribution algorithm

Replaces the cluster-level decision in `routers/transcribe.py:_match_speaker_embedding` with per-segment + corroboration + cluster split.

### Algorithm

For each speaker-clustered group from Deepgram (or future Sortformer):

1. **Sub-window embed**: split the cluster into 2-3s sub-windows; embed each via voice-extras
2. **Per-window match**: each sub-window gets `(nearest_distance, is_user_match)` against the user's voiceprint bank
3. **Corroboration window**: with default **N=2 of M=3** consecutive sub-windows required to match before locking `is_user=True`. Single-window matches don't trigger memory extraction (they may still be tagged for display, but `corroboration_count = 1` flags them downstream)
4. **Cluster split refinement**: if a Deepgram cluster has mixed sub-window matches (some pass corroboration, some fail), split the cluster:
   - Passing sub-windows form a new "user" cluster
   - Failing sub-windows form a demoted "unattributed" cluster
   - Both retain audio range references for traceability
5. **Per-segment metadata written to the segment**: `attribution_distance`, `corroboration_count`, `voiceprint_version`

### Defaults (env-overridable)

- `CORROBORATION_N = 2`
- `CORROBORATION_M = 3`
- `SUB_WINDOW_SECONDS = 2.5`
- `USE_CLUSTER_SPLIT = true`

### What this fixes

| Failure mode | Result after Layer 2 |
|--------------|---------------------|
| Single-segment false positive on user match | Filtered (corroboration N=2 minimum) |
| Deepgram cluster-merge → wrong segments tagged user | Split: only the actually-matching sub-windows stay user |
| Fixed threshold inflating false positives | Replaced by per-user calibrated threshold from voiceprint bank |

---

## Layer 3: Provenance schema

Every memory doc gains a nested `provenance` object + a top-level `user_status`. Memory content stays encrypted via the existing `@prepare_for_write` decorator; provenance fields stay plaintext (no sensitive content; queryable).

### Schema additions

```python
{
    # ... existing fields (id, content, category, conversation_id, created_at, etc.) ...

    'provenance': {
        'source_segment_ids': ['seg_001', 'seg_005', 'seg_006'],
        'attribution_distance': 0.32,           # min across source segments; lower is better
        'corroboration_count': 3,                # number of source segments
        'extractor_version': 'memory-extractor-v2-strict',
        'voiceprint_version': 'bank-2026-05-31-15min-quiet',
        'extracted_at': <timestamp>,
    },

    'user_status': 'pending',                    # 'pending' | 'confirmed' | 'dismissed' | 'auto_suppressed'
    'dismissed_by': None,                         # 'heuristic' | 'critic' | 'user' | 're_diarize_replacement'
}
```

### Backward compatibility

Reads from memories without provenance: treat as `{extractor_version: 'legacy-unknown', voiceprint_version: 'legacy-unknown', source_segment_ids: [], attribution_distance: None, corroboration_count: 0, extracted_at: <use created_at>}`. The legacy-unknown marker lets the re-diarize batch identify all pre-fix memories that need re-attribution.

### Encryption note

The `@prepare_for_write` decorator that encrypts `content` must learn to skip the `provenance` sub-object and `user_status` / `dismissed_by` top-level fields. Implementation detail.

---

## Layer 4: Extraction guardrails

Changes `extract_memories_prompt` in `utils/prompts.py` and the surrounding extraction code in `utils/conversations/process_conversation.py`.

### Input gating (before LLM call)

Only segments where:
- `is_user == True`
- `attribution_distance < calibrated_threshold` (from the voiceprint bank)
- Segment is in a corroboration window (corroboration_count >= 2)

… get sent to the extractor.

### Prompt changes

Add to `extract_memories_prompt`:
- "Only extract facts where the source segments contain first-person assertion (`I`, `I'm`, `my`, `me`, `mine`). Do not extract facts from second- or third-person statements."
- "Each extracted fact MUST cite the segment ID(s) it came from in a `source_segment_ids` field."
- "Do not synthesize facts that aren't explicitly stated. Behavioral inference (e.g., user said 'getting coffee' → extract 'user likes coffee') is forbidden — facts must be explicit assertions."

Update the output Pydantic schema:

```python
class ExtractedFact(BaseModel):
    content: str
    category: Literal['system', 'interesting']
    source_segment_ids: List[str]  # MUST be non-empty
```

### Post-extraction validation

```python
for fact in extracted_facts:
    sources = [s for s in input_segments if s.id in fact.source_segment_ids]

    if not sources:
        log_phantom_attempt(fact, reason='hallucinated_sources')
        continue

    if fact.category == 'system' and not any(FIRST_PERSON_RE.search(s.text) for s in sources):
        log_phantom_attempt(fact, reason='no_first_person_in_sources')
        continue

    fact.provenance = Provenance(
        source_segment_ids=fact.source_segment_ids,
        attribution_distance=min(s.attribution_distance for s in sources),
        corroboration_count=len(sources),
        extractor_version='memory-extractor-v2-strict',
        voiceprint_version=current_voiceprint_version(),
        extracted_at=now(),
    )

    if fact.category == 'system' and len(sources) < CORROBORATION_MIN:
        fact.user_status = 'auto_suppressed'

    save_memory(fact)
```

### First-person detection

Simple regex: `\b(I|I'm|I've|I'd|I'll|my|me|mine|myself)\b` — case-insensitive. Fast, deterministic, no LLM dependency. False positives are tolerable (we let through a fact we could have rejected); false negatives are not (we drop a real fact).

### Dropped-attempt logging

Every rejected extraction writes an audit record with `conversation_id`, `reason`, snapshot of the fact. After weeks of production data, review whether guardrails are too aggressive or too loose.

---

## Layer 5: Memory health daemon (hybrid)

### Stage 1 — Inline heuristic (at memory write time)

Runs synchronously inside `save_memory()`. Pure Python, no LLM. Computes a `fast_score` from provenance + cited-segment properties:

```python
def heuristic_score(memory, source_segments) -> float:
    score = 0.5
    if memory.provenance.attribution_distance < 0.3: score += 0.25
    elif memory.provenance.attribution_distance > 0.5: score -= 0.25
    if memory.provenance.corroboration_count >= 3: score += 0.15
    elif memory.provenance.corroboration_count == 1: score -= 0.15
    if memory.category == 'system' and not any(FIRST_PERSON_RE.search(s.text) for s in source_segments):
        score -= 0.3
    if all(s.duration_ms < 1000 for s in source_segments): score -= 0.15
    return max(0.0, min(1.0, score))
```

Decision (test mode — binary):
- `fast_score >= 0.5` → `user_status='confirmed'`, skip critic queue
- `fast_score < 0.5` → `user_status='pending'`, **queue for critic**

In test mode the heuristic catches ~60-70% of obvious cases without any LLM cost.

### Stage 2 — LLM critic (background sweep)

Scheduled cron via the existing `modal/` job pattern. Hourly default.

Walks `users/*/memories` for `user_status='pending'`. For each:

```
You are a memory-quality auditor for an AI wearable device.

USER NAME: {user_name}
MEMORY CONTENT: {memory.content}
MEMORY CATEGORY: {memory.category}
SOURCE CONVERSATION CONTEXT: {5 segments before + after the cited sources}
CITED SOURCE SEGMENTS:
  - [is_user={s.is_user}, distance={s.attribution_distance:.2f}]: "{s.text}"
  - ...

Score (0-1 each):
  - first_person_assertion: does the user actually assert this in the cited segments?
  - context_coherence: do source segments fit the conversation context, or look like overheard TV/podcast?
  - content_plausibility: real fact about a person, or scripted dialogue?
  - overall_reliability: composite

Flags:
  - looks_like_tv: true if patterns match TV/movie/podcast
  - contradicts_known_user_traits: true if conflicts with established user facts

Output strict JSON.
```

Action engine (test mode — binary):
- `looks_like_tv == true` OR `first_person_assertion < 0.3` → dismiss
- `overall_reliability >= 0.5` → confirmed
- otherwise → dismiss

Audit log entry written to `users/{uid}/memory_actions/` with full critic result + memory snapshot for rollback.

### Stage 3 — Re-evaluation triggers

Memories get re-queued for critic on:
- Voiceprint bank version change (re-queue everything with `voiceprint_version != current`)
- Source conversation re-diarized
- N days elapsed since last critique (config: 30 days)
- New high-confidence corroborating evidence appears (cross-conversation match)

User-set `user_status` (manual override) freezes the memory from further auto-actions.

### Production mode (future, when jlportman3 migrates)

Same daemon, different config:
```python
MEMORY_DAEMON_MODE='production'
AUTO_DISMISS_THRESHOLD=0.3          # more conservative
AUTO_CONFIRM_THRESHOLD=0.7          # genuine 'needs_review' middle ground
GENERATE_DIGEST=true                # weekly rollup email
```

### Audit log schema

`users/{uid}/memory_actions/{action_id}`:

```python
{
    'memory_id': ...,
    'action': 'dismissed' | 'confirmed',
    'taken_by': 'heuristic' | 'critic' | 'user' | 're_diarize',
    'reason': '...',
    'critic_scores': {...} | None,
    'memory_snapshot': {content, provenance},
    'timestamp': ...,
}
```

Rollback is a Firestore query against this collection — no UI required.

---

## Layer 6: Re-diarize batch — tiered priority sweep

Retroactively re-attributes historical conversations using the current voiceprint bank. Produces new memories that the daemon then processes normally.

### Per-conversation flow

```
For each conv:
  1. Load conversation doc (transcript_segments + audio_files refs)
  2. Check audio availability in omi-mine-private-cloud-sync
  3. If yes (full re-diarize path):
     a. Download + decrypt chunks
     b. Run Sortformer clustering (replaces Deepgram/cloud-omi speaker_ids)
     c. Per-cluster TitaNet sub-window embeddings via voice-extras
     d. Per-segment match against current voiceprint bank
     e. Apply Layer 2 algorithm (corroboration + cluster split)
     f. Update transcript_segments with new speaker_id, is_user, per-segment metadata
  4. If no audio available (text-only path — applies to most Wookie imports where cloud-omi didn't include audio in the export):
     a. Skip re-attribution entirely (no embeddings to compare against)
     b. Trust existing is_user tags (limitation: these came from cloud-omi's pipeline with no quality control)
     c. Memories produced from this path get extractor_version='v2-strict' but voiceprint_version='no-audio'
     d. The daemon's heuristic will be less able to score these (provenance.attribution_distance will be None) — they'll lean more on the LLM critic for quality decisions
  5. Soft-dismiss all existing memories for this conversation:
     - user_status='dismissed'
     - dismissed_by='re_diarize_replacement'
     - audit log entry with snapshot
  6. Re-run memory extractor (with Layer 4 guardrails)
  7. New memories land with current provenance (extractor_version + voiceprint_version)
  8. Daemon processes new memories normally
  9. Mark conv: re_diarized_at = now, re_diarized_voiceprint_version = current
```

### Scheduling: tiered priority sweep with parallel workers

Three priority tiers drained in order:

1. **Recent (last 30 days)** — most likely to be queried; freshest data
2. **Low-confidence (any age, `provenance.attribution_distance > 0.4` OR `corroboration_count == 1`)** — highest phantom potential
3. **Backlog (everything else)** — slow background drain

Multiple workers in parallel, respecting rtx6000 voice-extras capacity (configurable concurrent request limit, default 4).

### Per-conversation state

Added to conversation doc:

```python
{
    # ... existing fields ...
    're_diarize_status': 'pending' | 'in_progress' | 'completed' | 'failed' | 'skipped_no_audio',
    're_diarize_voiceprint_version': 'bank-2026-05-31',
    're_diarized_at': <timestamp>,
    're_diarize_error': <str or None>,
}
```

Worker CAS-updates `re_diarize_status` from `pending` → `in_progress` to prevent two workers picking the same conv. On completion, sets `completed` or `failed`. Resumable across backend restarts.

### Voiceprint version change → automatic propagation

When the voiceprint bank version updates:
1. Compute hash/version of new bank
2. Update `users/{uid}.voiceprint_bank.version`
3. Scan: mark all conversations where `re_diarize_voiceprint_version != current_version` as `re_diarize_status='pending'`
4. Workers naturally drain the refreshed backlog

This means the re-diarize pipeline is also the voiceprint-update propagation mechanism.

### CLI interface

```bash
omi-rediarize --uid <UID> --conversation-id <conv_id>
omi-rediarize --uid <UID> --tier recent | low-confidence | backlog
omi-rediarize --uid <UID> --all                 # bulk reset to pending
omi-rediarize status --uid <UID>                # backlog per tier, recent activity, failures
omi-rediarize workers start --concurrency 4
omi-rediarize workers stop
```

---

## Migration / rollout plan

### Phase 1 — Foundations (Week 1)
1. Implement Layer 1 voiceprint bank schema + enrollment script
2. User records the 15-min quiet-room baseline on Jarvis
3. Run enrollment script → voiceprint bank written
4. Implement Layer 3 provenance schema (add fields, encryption decorator update)
5. Tests for both

### Phase 2 — Live attribution improvements (Week 2)
1. Implement Layer 2 per-segment algorithm in `speaker_identification_task`
2. Implement Layer 4 extraction guardrails (prompt + validation)
3. Deploy to Jarvis backend
4. Validate: new conversations show meaningfully lower phantom rate

### Phase 3 — Autonomous daemon (Week 3)
1. Implement Layer 5 inline heuristic
2. Implement Layer 5 LLM critic + cron schedule
3. Implement audit log + rollback queries
4. Deploy; observe daemon behavior on new memories

### Phase 4 — Retroactive cleanup (Weeks 4-6)
1. Implement Layer 6 re-diarize batch + worker pool
2. Run Tier 1 (recent) sweep on Jarvis — validate output quality
3. Run Tier 2 (low-confidence) sweep
4. Run Tier 3 (backlog) sweep — drain over weeks
5. Periodic re-evaluation as continual-learning samples accumulate

### Phase 5 — Production-ready (when jlportman3 migrates)
1. Flip `MEMORY_DAEMON_MODE='production'`
2. Add digest generator + override interface
3. Build mobile UI for triage (separate design)

## Test plan

- Unit tests for every helper: voiceprint matching, corroboration windowing, cluster split, heuristic scorer, first-person regex
- Mocked integration tests for the full attribution pipeline with synthetic embedded segments
- Live integration tests against real conversation chunks from Jarvis's imported Wookie data — measure phantom rate before/after
- Daemon tests using known phantom + known real memories
- Re-diarize batch tested against a small subset before full sweep

## Success metrics

- Phantom memory rate on Jarvis (manually-confirmed phantoms / total new memories) drops from current (unknown but high) to < 5%
- Re-diarize backlog drains completely within 30 days of voiceprint enrollment
- Memory critic decisions correlate with manual spot-checks at > 90% agreement
- Zero data loss from automated actions (every dismissal is reversible via audit log)
- Daemon LLM critic compute stays within rtx6000's headroom (no impact on chat / RAG latency)

## Open questions deferred to implementation

- Optimal corroboration N-of-M values (defaults 2-of-3 are tunable; production tuning happens after observing real data)
- Whether `continual_samples` cap of 50 is right (could be more or less depending on drift behavior)
- LLM critic temperature + prompt wording — needs iteration after first production data
- Whether action_items also need provenance + critic treatment (current scope is memories only; action_items follows same pattern if needed)
- Mobile UI design — deferred to Phase 5

## Future work

- Multi-signal voting (Layer 2 Approach 3): add first-person regex, mic-amplitude, temporal-adjacency signals as weighted votes alongside TitaNet
- Cross-user memory federation: if Joe has multiple test accounts (Jarvis, Wookie, future jlportman3), share critic experience without sharing memories
- Per-person voiceprint banks populated from `speech_samples` (currently only Joe-the-user gets a bank; enrolled people get one when sample volume warrants)
- Sortformer-based clustering replacing Deepgram (Phase B dependency) — when Phase B lands, Sortformer's cluster output feeds directly into Layer 2

---

**Approved during brainstorming session 2026-05-31. Next: implementation plan via writing-plans skill.**
