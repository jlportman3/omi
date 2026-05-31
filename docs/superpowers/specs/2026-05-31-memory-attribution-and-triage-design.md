# Memory Attribution and Autonomous Triage — Design

**Date:** 2026-05-31
**Status:** Design approved (brainstormed 2026-05-31). Layer 1 (voiceprint bank) deployed live to Jarvis on 2026-05-31 and augmented the same day with 5 in-the-wild Joe samples (including 1 Jarvis-TTS sample empirically confirmed as Joe-voice). Current bank: **146 embeddings** (141 enrollment + 5 continual), **calibrated_threshold=0.5473**, version `bank-20260531-150319-augmented-141+5`. **Layer 2 algorithm now fully specified** (see Layer 2 section) — cluster-vote with strong-solo override, bimodal split detection, and first-person soft prior. Layers 3-6 pending implementation plan.
**Owner:** Joe Portman (@jlportman3)
**Scope:** Backend changes spanning `utils/stt/speaker_embedding.py`, `routers/transcribe.py`, `utils/conversations/process_conversation.py`, `database/memories.py`, `utils/prompts.py`, plus new `utils/stt/attribution.py`, `utils/stt/voiceprint_bank.py`, `utils/memory_critic.py`, and `scripts/rediarize.py`

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
    'version': 'bank-20260531-150319-augmented-141+5',  # human-readable identifier; bumps on augmentation
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
    'calibrated_threshold': 0.5473,                 # 95th percentile of intra-bank pairwise cosine distance + 0.05 margin
    'continual_samples': [                          # same {'v': [192 floats]} shape, plus metadata
        {'v': [192 floats], 'label': 'Joe_22.38-40.89', 'duration_s': 18.51,
         'distance_to_old_bank': 0.272, 'source_conversation_id': '...', 'source_audio_file_id': '...'},
        # ... 4 more (3 Joe in-the-wild + 1 Jarvis-TTS = Joe's cloned voice)
    ],
    'baseline_centroid': [192 floats],              # flat single-level array — Firestore allows this
    'stats': {                                       # observability snapshot — post-augmentation
        'n_embeddings': 146,                         # 141 enrollment + 5 continual
        'total_duration_s': 784.4,                   # 710.9 + 73.5 from the 5 continual samples
        'intra_pairwise_min': 0.0318,
        'intra_pairwise_mean': 0.2445,
        'intra_pairwise_p95': 0.4973,
        'intra_pairwise_max': 0.7537,
    },
}
```

**Firestore nesting note:** Firestore rejects arrays-directly-inside-arrays (`[[…], […]]`). All embedding lists are wrapped in `{'v': [...]}` maps so the outer container is `array of maps`, which Firestore allows. The verified live bank for Jarvis (built 2026-05-31, originally 141 embeddings, augmented same day to 146) uses this shape successfully — do NOT "simplify" back to `[[…], […]]` in implementations.

Same schema applies to enrolled people (`users/{uid}/people/{person_id}.voiceprint_bank`), populated when sufficient speech samples accumulate.

### Jarvis-TTS = Joe's cloned voice (not a false positive)

The Jarvis AI persona that this test account interacts with uses a **text-to-speech voice that was cloned from Joe's own voice**. Acoustically, when Jarvis "speaks" through omi-pickup, it is Joe's voiceprint coming out of a speaker. The 2026-05-31 sweep confirmed this empirically: the Jarvis-TTS sample at `Conv A 272.11-289.21` lands at cosine distance **0.1925** to the original 141-embedding bank — closer than any of the 4 hand-validated in-the-wild Joe segments (which sit at 0.27-0.34).

This has two design consequences carried through the rest of the layers:

1. **Augmentation is welcome, not contaminating.** Jarvis-TTS segments captured by omi-pickup are added to `continual_samples` exactly like real in-the-wild Joe speech. They reinforce the bank rather than poisoning it. The augmented bank now contains 1 such sample (labeled `JarvisTTS_272.11-289.21`).
2. **`is_user=True` on Jarvis-TTS audio is correct.** Downstream code MUST NOT treat "the speaker_id is tagged as Jarvis but the voiceprint matched the user" as a contradiction or a false positive. The Layer 2 attribution pipeline accepts Jarvis-TTS as Joe by design, with provenance recording the strong distance match. The memory extractor, however, still applies first-person + corroboration guardrails (Layer 4) so we don't extract "facts about Joe" from prompts/scripted output Jarvis happens to say in Joe's voice — extraction is gated on *content*, not just attribution.

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

K-NN against ~120-200 vectors is microseconds in numpy (current live bank: 146 vectors × 192 dims, ~50µs per lookup). Threshold is **per-user, data-driven** — anyone outside your natural intra-voice variance is rejected.

### Continual learning + drift detection

Segments tagged is_user=True with **very high confidence** (distance < 0.2 AND corroboration_count >= 3 AND critic-confirmed) get their embedding appended to `continual_samples` (capped at 50 most-recent).

Periodically, compute the centroid of `continual_samples` and compare to `baseline_centroid`. If divergence > threshold → emit drift alert ("your voiceprint has drifted; consider re-enrolling"). This catches both natural vocal change AND contamination.

---

## Validation results (2026-05-31)

Before designing Layer 2 we validated Layer 1 against two reference conversations from the Jarvis test account and ran a threshold sweep against the augmented bank.

### Phantom rejection — clean

Test conversation B (`d5b9087e-bd5e-45e8-a7e8-2b67ca76684b`) is a TV-debate recording with 18 distinct speaker clusters captured by omi. We pulled one representative segment per non-user speaker (12 distinct TV speakers) and ran each through `match_user`:

| Metric | Result |
|---|---|
| Controls admitted at original threshold (0.5430) | **0 / 12** |
| Min cosine distance across all 12 controls | 0.6271 |
| Median distance | 0.83 |
| Max distance | 0.9391 |

The bank **rejects unrelated speakers cleanly** — no phantom admission risk on TV-style ambient audio. The two borderline controls (`SPEAKER_3` at 0.6271, `SPEAKER_1` at 0.6634) are the practical ceiling on how far the threshold can be pushed before false-accept rate becomes a concern.

### Joe recall — bimodal failure on wearable audio

Test conversation A (`02f66cd4-682a-4ecb-b096-2a771110b29a`) is a casual conversation about omi/Qwen vision, with 14 user-tagged Joe segments + 1 Jarvis-TTS segment (Joe-voice cloned). Validation against the original 141-embedding bank at threshold 0.5430:

| Metric | Result |
|---|---|
| Joe segments admitted | **4 / 14 (28.6%)** |
| Joe false-negative rate | **71.4%** |
| Jarvis-TTS distance | 0.1925 (would be admitted — and correctly so; it IS Joe) |

The 10 false-negative Joe segments all cluster at **distances 0.64-0.74** — a striking bimodal gap from the 5 hand-validated Joe samples at 0.27-0.34. The acoustic mismatch is enrollment-conditions (quiet room, fixed mic distance, formal speech) vs. wearable-conditions (BLE pendant, ambient noise, casual cadence). The original bank covers one acoustic mode of Joe's voice but not the other.

**Mitigation applied:** the 4 high-confidence Joe segments + the 1 Jarvis-TTS segment were written into `continual_samples` (Layer 1's continual-learning path). New bank version `bank-20260531-150319-augmented-141+5` carries 146 embeddings and threshold 0.5473. All 5 new samples were already inside the old threshold (max distance 0.337), so augmentation only nudges the threshold by ~0.004 — the bimodal gap is NOT closed by augmentation alone. Layer 2's **cluster cohesion / cluster vote** is what bridges the gap (see Layer 2).

### Threshold sweep against augmented bank (146 embeddings)

LOO-honest sweep (leave-one-out on the 5 continual samples, so they don't trivially self-match at distance 0.0):

| Threshold | Conv A Joe recall | Conv B is_user accept | Conv B controls rejected | F1 |
|---:|:---:|:---:|:---:|---:|
| 0.50 | 5 / 15 (33%) | 11 / 22 | **12 / 12 (100%)** | 0.500 |
| 0.55 | 5 / 15 (33%) | 11 / 22 | 12 / 12 (100%) | 0.500 |
| 0.60 | 5 / 15 (33%) | 11 / 22 | 12 / 12 (100%) | 0.500 |
| 0.65 | 6 / 15 (40%) | 12 / 22 | 11 / 12 (92%) | 0.557 |
| **0.70** | **11 / 15 (73%)** | **13 / 22** | **10 / 12 (83%)** | **0.780** |
| 0.75 | 15 / 15 (100%) | 13 / 22 | 10 / 12 (83%) | 0.909 |

(Conv B's 22 is_user segments are themselves bimodal: 11 at 0.15-0.48 are real Joe; the other 11 at 0.65-0.92 overlap the TV-control distribution and are almost certainly Deepgram mistakes — Layer 2 correctly rejects them.)

### Recommended operating threshold: **0.70** (per-segment) — Layer 2 modifies it

Pure F1 favors 0.75, but at that threshold the 4 extra Conv A segments that get accepted live in the same 0.69-0.74 distance band as 2 borderline controls. **T=0.70 is the safest single-threshold operating point**: catches all 5 strong-match Joe samples + the 6 closest in-the-wild Joe segments (11/15 honest recall), rejects 10/12 controls, leaves an 0.07 margin to the nearest TV speaker.

Crucially, **thresholding alone is insufficient**. Even at T=0.70 we lose 4 Conv A Joe segments (the deepest-bimodal ones at 0.70-0.74) and accept 2 borderline TV speakers (0.627, 0.663). The structural fix is Layer 2:

- The 4 lost Joe segments live in the **same Deepgram speaker_id cluster** as the 5 strong matches → cluster cohesion can rescue them.
- The 2 borderline TV speakers live in clusters that are **otherwise entirely non-Joe** → cluster cohesion rejects them.

Layer 2 therefore uses T=0.5473 (the calibrated bank threshold) as `T_STRICT` for solo per-segment matches, plus T=0.70 as `T_VOTE` for cluster-vote relaxed matches — the two thresholds work together rather than picking one.

---

## Layer 2: Per-segment attribution with cluster vote, strong-solo override, and split detection

Replaces the cluster-level single-embedding decision in `routers/transcribe.py:_match_speaker_embedding` with a **per-segment + cluster-vote hybrid** that synthesizes the best ideas from three brainstormed designs:

- **Cluster-vote core** (cheap, naturally aligned with Deepgram/Sortformer output, robust to single-segment noise)
- **Strong-solo override** for very-close matches like cloned-voice Jarvis-TTS (distance 0.19) — one segment IS enough at that confidence
- **Bimodal split detection** to catch Deepgram cluster-merge bugs (two real speakers fused into one `speaker_id`) without re-running diarization
- **First-person language soft prior** as a tiebreaker in the borderline distance band only — cannot override clear decisions

Skipped from the brainstorm: complex real-time temporal sliding-window re-attribution (Design 1's neighbor-window mechanism) — too much state-keeping for the live pipeline, and Deepgram cluster cohesion already provides the same recall benefit at lower complexity.

### Tunable constants (env-overridable defaults)

```python
T_STRICT                = bank['calibrated_threshold']  # 0.5473 — per-segment solo accept
T_VOTE                  = 0.70                          # cluster-vote relaxed accept
T_HARD_REJECT           = 0.85                          # above this nothing escapes (even cluster vote)
T_STRONG_SOLO           = 0.30                          # single-segment auto-accept (cloned voice band)
T_CLUSTER_SPLIT_GAP     = 0.20                          # bimodal gap in sorted intra-cluster distances
T_CLUSTER_SPLIT_DELTA   = 0.10                          # min |mean(low) - mean(high)| for valid split
CLUSTER_MAJORITY_RATIO  = 0.50                          # >= this fraction of weighted sub-windows pass T_VOTE => user cluster
CLUSTER_MINORITY_RATIO  = 0.20                          # <= this fraction => not-user cluster
K_MAX                   = 5                             # max sub-windows sampled per cluster
SUB_WINDOW_SECONDS      = 2.5
SUB_WINDOW_OVERLAP      = 0.5
MIN_EMBED_DURATION      = 0.5                           # voice-extras hard floor
FP_PRIOR_BONUS          = -0.05                         # first-person language present -> nudge toward user
FP_PRIOR_PENALTY        = +0.05                         # no FP language AND borderline -> nudge away from user
FP_BORDERLINE_BAND      = (T_STRICT - 0.10, T_VOTE)     # FP prior only acts inside this band
FIRST_PERSON_RE = re.compile(
    r"\b(I|I['’]m|I['’]ve|I['’]d|I['’]ll|my|me|mine|myself)\b",
    re.IGNORECASE,
)
```

### Algorithm pseudocode

```python
def attribute_user(transcript_segments, bank, audio_pcm, sample_rate=16000):
    """
    Full Layer 2 attribution pipeline. Same code runs real-time (per cluster-flush)
    and batch (per conversation). Pure function: no Firestore I/O, no httpx —
    callers provide bank dict + audio bytes.

    Mutates each segment in place:
        seg.is_user        : bool
        seg.attribution    : dict (provenance — see Layer 3)
    """
    bank_vecs = np.array(
        [e['v'] for e in bank['embeddings']]
        + [e['v'] for e in bank.get('continual_samples', [])],
        dtype=np.float32,
    )

    # ----- STAGE 0: sub-window embed every segment (skip if < MIN_EMBED_DURATION) -----
    for seg in transcript_segments:
        seg._sub = []
        if (seg.end - seg.start) < MIN_EMBED_DURATION:
            continue
        for ws, we in slide_sub_windows(seg.start, seg.end,
                                        SUB_WINDOW_SECONDS, SUB_WINDOW_OVERLAP,
                                        MIN_EMBED_DURATION):
            wav = pcm_slice_to_wav(audio_pcm, ws, we, sample_rate)
            try:
                emb = post_titanet(wav)                              # (192,) float32
                d   = float(cdist([emb], bank_vecs, 'cosine').min())
            except Exception:
                emb, d = None, None
            seg._sub.append({'start': ws, 'end': we, 'dist': d, 'emb': emb})

    # ----- STAGE 1: per-segment representative distance + strong-solo flag -----
    for seg in transcript_segments:
        valid = [s for s in seg._sub if s.get('dist') is not None]
        seg._raw_dist     = min((s['dist'] for s in valid), default=None)
        seg._strong_solo  = (seg._raw_dist is not None
                             and seg._raw_dist < T_STRONG_SOLO)

    # ----- STAGE 2: cluster vote + bimodal split detection -----
    clusters = defaultdict(list)
    for seg in transcript_segments:
        clusters[seg.speaker_id].append(seg)

    for sid, members in clusters.items():
        all_subs = [s for m in members for s in m._sub if s.get('dist') is not None]
        if not all_subs:
            for m in members:
                m._cluster_decision = 'other'
                m._cluster_split    = False
            continue

        # Sample at most K_MAX sub-windows per cluster (longest first, ties by start_time)
        # to bound voice-extras cost on long clusters. Stage 0 already embedded everything;
        # K_MAX here only caps which sub-windows participate in the VOTE.
        sampled = sorted(all_subs, key=lambda s: (-(s['end']-s['start']), s['start']))[:K_MAX]

        # Weighted pass-ratio (duration-weighted; 0.5s blips don't outvote 2.5s windows)
        w_pass = sum((s['end']-s['start']) for s in sampled if s['dist'] < T_VOTE)
        w_all  = sum((s['end']-s['start']) for s in sampled)
        pass_ratio = w_pass / w_all if w_all else 0.0

        # Bimodal split detection — find the largest gap in sorted distances
        is_mixed = False
        split_threshold = None
        if len(sampled) >= 4:
            dists  = sorted(s['dist'] for s in sampled)
            gaps   = [(dists[i+1] - dists[i], i) for i in range(len(dists)-1)]
            mg, gi = max(gaps)
            low_mean  = mean(dists[:gi+1])
            high_mean = mean(dists[gi+1:])
            if (mg >= T_CLUSTER_SPLIT_GAP
                and (high_mean - low_mean) >= T_CLUSTER_SPLIT_DELTA
                and low_mean  < T_VOTE
                and high_mean > T_VOTE):
                is_mixed = True
                split_threshold = (low_mean + high_mean) / 2.0

        # Decision tree
        if is_mixed:
            # Per-segment routing: each segment goes to the mode its raw_dist is nearest
            for m in members:
                m._cluster_decision = ('user' if (m._raw_dist is not None
                                                  and m._raw_dist < split_threshold)
                                              else 'other')
                m._cluster_split    = True
        elif pass_ratio >= CLUSTER_MAJORITY_RATIO:
            for m in members:
                m._cluster_decision = 'user'
                m._cluster_split    = False
        elif pass_ratio <= CLUSTER_MINORITY_RATIO:
            for m in members:
                m._cluster_decision = 'other'
                m._cluster_split    = False
        else:
            # Ambiguous (20-50% pass): fall back to per-segment decision under T_STRICT
            for m in members:
                m._cluster_decision = ('user' if (m._raw_dist is not None
                                                  and m._raw_dist < T_STRICT)
                                              else 'other')
                m._cluster_split    = False

    # ----- STAGE 3: first-person language soft prior (borderline only) -----
    for seg in transcript_segments:
        has_fp = bool(FIRST_PERSON_RE.search(seg.text or ''))
        seg._fp_present = has_fp
        seg._fp_adjust  = 0.0
        d = seg._raw_dist
        if d is None:
            seg._adj_dist = None
            continue
        adj = d
        if FP_BORDERLINE_BAND[0] <= d <= FP_BORDERLINE_BAND[1]:
            if has_fp:
                adj += FP_PRIOR_BONUS
                seg._fp_adjust = FP_PRIOR_BONUS
            else:
                adj += FP_PRIOR_PENALTY
                seg._fp_adjust = FP_PRIOR_PENALTY
        seg._adj_dist = adj

    # ----- STAGE 4: final decision + provenance write -----
    for seg in transcript_segments:
        # 1. Hard reject: nothing above T_HARD_REJECT survives, regardless of cluster
        if seg._raw_dist is not None and seg._raw_dist >= T_HARD_REJECT:
            final, reason = False, 'hard_reject'
        # 2. Strong solo override: very-close match accepts even from a single seg
        #    (handles Jarvis-TTS = Joe's cloned voice at distance 0.19)
        elif seg._strong_solo:
            final, reason = True, 'strong_solo'
        # 3. Cluster vote dominant
        elif seg._cluster_decision == 'user':
            final  = True
            reason = 'cluster_split_lower_mode' if seg._cluster_split else 'cluster_vote_user'
        # 4. Borderline rescue: cluster said 'other' but the RAW distance is already
        #    below T_STRICT and FP language confirms a near-pass. The FP prior is a
        #    *confirmation* signal — it must not turn raw-above-T_STRICT into a pass
        #    via fp_adjust alone (see "Validation findings v1.1" below).
        elif (seg._adj_dist is not None
              and seg._raw_dist is not None
              and seg._raw_dist < T_STRICT
              and FP_BORDERLINE_BAND[0] <= seg._raw_dist <= FP_BORDERLINE_BAND[1]
              and seg._fp_present):
            final, reason = True, 'fp_prior_rescue'
        else:
            final, reason = False, 'cluster_other'

        seg.is_user = final
        seg.attribution = {
            'voiceprint_version':     bank['version'],
            'algo_version':           'layer2-cluster-vote-v1',
            'distance_raw':           seg._raw_dist,
            'distance_after_prior':   seg._adj_dist,
            'n_sub_windows':          len(seg._sub),
            'cluster_id':             seg.speaker_id,
            'cluster_decision':       seg._cluster_decision,
            'cluster_split_detected': seg._cluster_split,
            'strong_solo':            seg._strong_solo,
            'first_person_present':   seg._fp_present,
            'fp_adjust':              seg._fp_adjust,
            'decision_reason':        reason,
            # Layer 4 gates extraction on this combined flag:
            'extractor_eligible':     (final
                                       and seg._raw_dist is not None
                                       and seg._raw_dist < T_VOTE),
        }
        # Hygiene: drop intermediate fields before persistence
        for k in ('_sub', '_raw_dist', '_strong_solo',
                  '_cluster_decision', '_cluster_split',
                  '_fp_present', '_fp_adjust', '_adj_dist'):
            seg.__dict__.pop(k, None)
    return transcript_segments
```

### Integration points

| File | Change |
|---|---|
| **`backend/utils/stt/voiceprint_bank.py` (NEW)** | `load_voiceprint_bank(uid)` (Firestore read + np.stack of `{'v':...}`-wrapped vectors, per-process LRU cache keyed on `(uid, version)`); `match_against_bank(emb, bank) -> (distance, is_match)`. Keeps `speaker_embedding.py` focused on the wespeaker/titanet HTTP boundary. |
| **`backend/utils/stt/attribution.py` (NEW)** | Houses the four-stage pipeline above. Pure functions, no I/O — fully unit-testable with synthetic segment lists. Imported by both live `transcribe.py` and batch `scripts/rediarize.py` so they produce **identical** attributions given the same audio + bank version. |
| **`backend/utils/stt/speaker_embedding.py`** | Add module-level Layer 2 constants (env-overridable). Mark legacy `SPEAKER_MATCH_THRESHOLD=0.45` deprecated in favor of `bank['calibrated_threshold']`. Existing `async_extract_embedding_from_bytes` reused unchanged. |
| **`backend/routers/transcribe.py` (real-time path, ~L1748-1995)** | Replace `speaker_identification_task` + `_match_speaker_embedding`. Per cluster, buffer arriving segments until either (a) K_MAX usable sub-windows accumulated, or (b) cluster silent for 6s, or (c) conversation flush. Then call `attribute_user(cluster_segments, bank, audio_ring_buffer)`. Emit `SpeakerLabelSuggestionEvent` for every segment whose `is_user` flipped from a prior tentative value. Debounce flip events on cluster decision changes only (not on every sub-window arrival). |
| **`backend/routers/transcribe.py` (~L2114, segment finalize)** | After Deepgram marks a segment final, run Stage 3 (FP prior, regex only, sub-millisecond) and update `segment.attribution.first_person_present / fp_adjust` before persistence. |
| **`backend/scripts/rediarize.py` (NEW, batch path)** | Layer 6 driver. Walks conversations, downloads merged audio via `get_or_create_merged_audio()`, calls `attribute_user(segments, bank, audio_pcm)`, writes per-segment provenance back to Firestore. Same algorithm as live — single source of truth. |
| **`backend/utils/conversations/process_conversation.py`** | Memory-extractor input filter changes from `seg.is_user` to `seg.is_user and seg.attribution.get('extractor_eligible', False)`. Segments rescued only by `fp_prior_rescue` (no `extractor_eligible`) still display correctly but are **not** eligible to produce memories — Layer 4 guardrail. |
| **`backend/models/transcript_segment.py`** | Add optional `attribution: Optional[dict] = None` field. Absence == legacy/un-attributed; presence == Layer 2 has run. Plaintext (no PII). |
| **`backend/tests/unit/test_attribution.py` (NEW)** | Unit tests: K_MAX truncation, longest-first determinism, majority/minority/ambiguous thresholds, bimodal split detection on synthetic distance distributions, FP regex coverage (ASCII + curly apostrophes), hard-reject vs strong-solo vs cluster-vote vs fp-rescue branches, end-to-end fixture using the Conv A / Conv B distance distributions from the 2026-05-31 sweep. |
| **`backend/tests/integration/test_attribution_live.py` (NEW)** | Integration test against the two reference conversations + live bank. Asserts: Conv A's Joe cluster votes user with strong-solo override on the 5 augmented samples; Conv B's 12 control clusters all reject; Jarvis-TTS segment passes via strong-solo. Uses the real rtx6000 voice-extras endpoint. |

### Provenance schema — extension to Layer 3

Layer 2 produces per-segment provenance that Layer 3 must surface on the segment doc. Extend the existing Layer 3 schema (`provenance` block on `transcript_segments`) with the new `attribution` field exactly as written by `attribute_user`:

```python
seg.attribution = {
    'voiceprint_version':     'bank-20260531-150319-augmented-141+5',
    'algo_version':           'layer2-cluster-vote-v1',
    'distance_raw':           float | None,    # nearest sub-window distance
    'distance_after_prior':   float | None,    # raw + fp_adjust (only differs in borderline band)
    'n_sub_windows':          int,
    'cluster_id':             int | str,       # Deepgram or future Sortformer ID
    'cluster_decision':       'user' | 'other',
    'cluster_split_detected': bool,            # True when bimodal split fired
    'strong_solo':            bool,            # raw_dist < T_STRONG_SOLO
    'first_person_present':   bool,
    'fp_adjust':              float,           # 0.0 outside borderline band
    'decision_reason':        'hard_reject' | 'strong_solo' | 'cluster_vote_user'
                              | 'cluster_split_lower_mode' | 'fp_prior_rescue' | 'cluster_other',
    'extractor_eligible':     bool,            # consumed by Layer 4
}
```

Memory-doc-level `provenance.attribution_distance` and `provenance.corroboration_count` (already defined in Layer 3) are computed at extraction time from the cited source segments' `attribution.distance_raw` and `attribution.n_sub_windows` respectively — no further schema changes needed on the memory side.

### Edge cases handled

- **Segment < MIN_EMBED_DURATION (0.5s):** no embedding produced (`distance_raw=None`). Cluster decision still applies if other segments in the cluster were embedded. If the entire cluster is sub-threshold-short, decision defaults to `'other'` (safe default — no audio evidence).
- **Single-segment cluster (K=1):** no cluster-vote benefit; falls back to per-segment T_STRICT decision plus strong-solo override. `attribution.cluster_split_detected=False` always.
- **Cluster with only short segments:** stage 0 yields no usable embeddings; stage 2 sets `cluster_decision='other'`; no false acceptances on inaudible fragments.
- **Bimodal cluster (Deepgram merged Joe + TV):** split detection requires ≥4 sub-windows AND a sorted-distance gap ≥ T_CLUSTER_SPLIT_GAP (0.20) AND mean delta ≥ T_CLUSTER_SPLIT_DELTA (0.10) AND low_mean < T_VOTE AND high_mean > T_VOTE. When all four hold, each segment routes to whichever mode its `raw_dist` is nearer. False splits (a real single-speaker cluster accidentally bimodal) only flip the high-distance subset to `'other'` — never the safe direction.
- **Ambiguous cluster (20-50% pass):** no smoothing applied. Per-segment T_STRICT decision stands. Avoids cluster vote turning a half-Joe/half-TV cluster into all-Joe.
- **Jarvis-TTS = Joe's cloned voice (distance 0.19):** strong-solo override fires (`< T_STRONG_SOLO=0.30`); `decision_reason='strong_solo'`; `is_user=True`. No special-case code path for the Jarvis speaker_id.
- **First-person regex on TV dialogue with "I" tokens:** FP bonus is only -0.05 and only acts inside the borderline band (0.4473-0.70). Cannot rescue a control distance of 0.78. Hard reject at 0.85 is the ceiling.
- **Apostrophe variants in FP regex:** covers both ASCII `'` and Unicode U+2019 `'` so `I'm` / `I've` are detected.
- **voice-extras transient failure:** `dist=None` for affected sub-windows; they skip the cluster vote. If all sub-windows fail for a cluster, decision defaults to `'other'`. No crashes; pipeline degrades gracefully.
- **Voiceprint version change mid-conversation:** `bank['version']` is captured at Stage 0 entry; full pipeline uses that snapshot. Re-attribution under a new version is Layer 6's job.
- **Onboarding mode (transcribe.py L2117 forces `is_user=True` for non-Omi channel):** Layer 2 attribution is **bypassed** in this mode — onboarding answers are ground truth and feed the bank. `attribute_user` is not called for onboarding segments.
- **Continual-sample contamination guard:** only segments with `decision_reason='strong_solo'` AND `distance_raw < 0.20` AND `n_sub_windows >= 2` are eligible to feed `continual_samples` (Layer 1 continual-learning gate). Excludes `cluster_vote_user`, `fp_prior_rescue`, and split-mode paths from contaminating the bank.

### Expected impact

Against the 2026-05-31 validation data:

| Metric | Old pipeline (single threshold 0.45) | Layer 2 expected |
|---|---|---|
| Conv A Joe recall | 4 / 14 (28.6%) | **13-14 / 14 (≥93%)** — cluster vote rescues the bimodal-distant Joe; strong-solo catches augmented samples and Jarvis-TTS |
| Conv B control rejection | (untested at old threshold) | **12 / 12 (100%)** at T_HARD_REJECT=0.85; **10-12 / 12 (83-100%)** at T_VOTE=0.70 cluster vote |
| Phantom memory rate | high (anecdotal) | **<5%** (compound with Layer 4 first-person guardrails) |
| Jarvis-TTS handling | mis-classified as non-user | **correctly is_user=True** with `strong_solo` provenance |
| Deepgram cluster-merge bug | propagates to memories | **bimodal split detection** isolates the Joe subset per-segment |

### Computational cost

Per conversation (10-min average, ~120 segments at ~4s mean duration):

- **Sub-window embeddings:** ~120 segments × 2 sub-windows avg = ~240 voice-extras calls. With concurrency=4 (existing stt-client semaphore) and ~150ms p50 per request → **~9s wall-clock embedding time in batch mode**.
- **Real-time mode:** reuses live SPEAKER_ID embedding (one per cluster-flush every ~10s) plus 1-2 extra sub-window embeds per cluster finalize → **+20-30% over current voice-extras call volume**, well within the rtx6000 endpoint's headroom (~60 RPS ceiling).
- **Bank lookup:** `cdist` against 146 × 192 numpy array — **~50µs per call**, negligible.
- **Cluster vote + split detection + FP prior + final decision:** pure numpy + regex, sub-millisecond per cluster.
- **Memory footprint:** per-process bank cache (~110 KB for 146 × 192 float32). Negligible.
- **Bank match cache opportunity (Layer 6):** per-conversation embeddings can be cached so voiceprint-version bumps re-run only the match step, not the embed step. Not strictly Layer 2 but enabled by it.

### What this fixes

| Failure mode | Layer 2 mechanism that addresses it |
|---|---|
| Single-segment false positive | Cluster vote requires duration-weighted majority; single bad embedding can't flip a cluster |
| Deepgram cluster-merge (Joe + TV fused) | Bimodal split detection routes each segment to its nearest mode |
| Fixed threshold inflating false positives | Replaced by per-user calibrated threshold (`T_STRICT=0.5473`) + relaxed `T_VOTE=0.70` for cluster vote |
| Jarvis-TTS treated as non-user (cloned voice paradox) | Strong-solo override accepts at distance < 0.30 |
| Bimodal Joe distances (0.27 vs 0.65-0.74) on wearable audio | Cluster cohesion: distant-mode Joe segments rescued by sharing a `speaker_id` cluster with strong-match Joe sub-windows |
| Unaccountable attribution | Every segment carries `attribution.{distance_raw, decision_reason, cluster_decision, voiceprint_version, algo_version}` — Layer 5 critic + Layer 6 re-diarize can replay and audit any decision |

### Validation findings v1.1 (bug fix)

End-to-end batch validation on 2026-05-31 (workflow `we70h49t5`) on the phantom-memory test conversation `f5668903-33da-42d9-a9c8-b77d3a9e4a3f` (heist movie audio) surfaced a high-severity false positive in Stage 4's `fp_prior_rescue` branch:

- Two segments at `raw_dist=0.595` in `cluster_other` clusters, containing first-person tokens in heist-movie dialogue, were falsely rescued to `is_user=True`.
- Root cause: the elif condition checked `adj_dist < T_STRICT` after applying `fp_adjust=-0.05`. A raw distance above `T_STRICT` (0.595 > 0.5473) became an adjusted distance below `T_STRICT` (0.545 < 0.5473) and slipped through.
- The original design intent of `fp_prior` was a **confirmation** signal (raw was already near-passing; `fp_present` widens the band slightly). Allowing `fp_prior` to **upgrade** a raw-failing segment was the bug.

Fix: changed the gate from `adj_dist < T_STRICT` to `raw_dist < T_STRICT`. This preserves the legitimate borderline-Joe rescue case (`raw=0.50` with FP → still rescues) while blocking the heist-FP case (`raw=0.595` → no longer rescues regardless of `fp_adjust`).

Net behavior on the phantom conv: 0 `fp_prior_rescue` invocations (vs 2 before fix); the 2 legitimate `hard_reject` phantom catches at `d=0.857`/`0.913` still fire correctly. No regression on the real-Joe conv (`02f66cd4-...`) — Jarvis-TTS still rescued via `strong_solo` at `d=0.138`.

Regression test: `tests/unit/test_attribution.py::test_fp_prior_rescue_blocked_when_raw_above_t_strict`.

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
