"""Layer 2 — per-segment user attribution with cluster vote, strong-solo
override, bimodal split detection, and first-person soft prior.

Pure functions only. No Firestore I/O, no httpx. The caller is responsible
for:

1. Loading the voiceprint bank (`utils.stt.voiceprint_bank.load_voiceprint_bank`)
2. Providing the merged PCM audio buffer for the conversation
3. Injecting an `embed_fn(wav_bytes) -> np.ndarray | None` that performs the
   voice-extras call. For production this is a thin wrapper around
   `utils.stt.speaker_embedding.extract_embedding_from_bytes`. For tests it
   can be a deterministic stub that maps WAV bytes to known distances.

The same algorithm runs in both live (per cluster-flush) and batch
(per conversation) paths — single source of truth.

See the design spec at:
  docs/superpowers/specs/2026-05-31-memory-attribution-and-triage-design.md
  section "Layer 2: Per-segment attribution with cluster vote, strong-solo
  override, and split detection".
"""

import io
import logging
import os
import re
import wave
from collections import defaultdict
from statistics import mean
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from utils.stt.speaker_embedding import extract_embedding_from_bytes as _prod_extract_embedding

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Tunable constants — all env-overridable
# ----------------------------------------------------------------------

# Per-user calibrated threshold default. Real T_STRICT comes from
# `bank['calibrated_threshold']`; this is the fallback used when the bank
# carries no calibration (uncalibrated user) so the algorithm still works
# end to end.
T_STRICT_DEFAULT: float = float(os.getenv("L2_T_STRICT_DEFAULT", "0.5473"))

T_VOTE: float = float(os.getenv("L2_T_VOTE", "0.70"))
T_HARD_REJECT: float = float(os.getenv("L2_T_HARD_REJECT", "0.85"))
T_STRONG_SOLO: float = float(os.getenv("L2_T_STRONG_SOLO", "0.30"))

T_CLUSTER_SPLIT_GAP: float = float(os.getenv("L2_T_CLUSTER_SPLIT_GAP", "0.20"))
T_CLUSTER_SPLIT_DELTA: float = float(os.getenv("L2_T_CLUSTER_SPLIT_DELTA", "0.10"))

CLUSTER_MAJORITY_RATIO: float = float(os.getenv("L2_CLUSTER_MAJORITY_RATIO", "0.50"))
CLUSTER_MINORITY_RATIO: float = float(os.getenv("L2_CLUSTER_MINORITY_RATIO", "0.20"))

K_MAX: int = int(os.getenv("L2_K_MAX", "5"))
SUB_WINDOW_SECONDS: float = float(os.getenv("L2_SUB_WINDOW_SECONDS", "2.5"))
SUB_WINDOW_OVERLAP: float = float(os.getenv("L2_SUB_WINDOW_OVERLAP", "0.5"))
MIN_EMBED_DURATION: float = float(os.getenv("L2_MIN_EMBED_DURATION", "0.5"))

FP_PRIOR_BONUS: float = float(os.getenv("L2_FP_PRIOR_BONUS", "-0.05"))
FP_PRIOR_PENALTY: float = float(os.getenv("L2_FP_PRIOR_PENALTY", "0.05"))

# How far below T_STRICT (exclusive) the FP borderline band begins. Default
# matches the spec: borderline = (T_STRICT - 0.10, T_VOTE).
FP_BORDERLINE_BAND_OFFSET: float = float(os.getenv("L2_FP_BORDERLINE_BAND_OFFSET", "0.10"))

# First-person regex — covers both ASCII apostrophe (') and Unicode U+2019 (’)
# so contractions match in either form (`I'm` and `I’m`).
FIRST_PERSON_RE = re.compile(
    r"\b(I|I['’]m|I['’]ve|I['’]d|I['’]ll|my|me|mine|myself)\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def slide_sub_windows(
    start_s: float,
    end_s: float,
    window_s: float = SUB_WINDOW_SECONDS,
    overlap_s: float = SUB_WINDOW_OVERLAP,
    min_duration: float = MIN_EMBED_DURATION,
) -> Iterator[Tuple[float, float]]:
    """Yield `(window_start, window_end)` tuples that cover `[start_s, end_s]`.

    Behaviour:
    - For segments shorter than `window_s` but longer than `min_duration`,
      yield a single `(start_s, end_s)` window.
    - For longer segments, slide by `window_s - overlap_s` and yield each
      window. The final window is clamped to `end_s`. If the clamped final
      window is shorter than `min_duration`, it is dropped.
    """
    if end_s - start_s < min_duration:
        return
    if end_s - start_s <= window_s:
        yield (start_s, end_s)
        return
    step = max(window_s - overlap_s, 0.1)
    cur = start_s
    while cur < end_s:
        we = min(cur + window_s, end_s)
        if we - cur >= min_duration:
            yield (cur, we)
        if we >= end_s:
            break
        cur += step


def pcm_slice_to_wav(pcm: bytes, start_s: float, end_s: float, sample_rate: int = 16000) -> bytes:
    """Return a WAV slice of `pcm` between `start_s` and `end_s`.

    Assumes 16-bit mono PCM at `sample_rate`. Clamps to the buffer bounds.
    Pure function — no filesystem, no httpx.
    """
    sample_width = 2  # PCM16
    channels = 1
    bytes_per_sec = sample_rate * sample_width * channels
    start_byte = max(int(start_s * bytes_per_sec), 0)
    end_byte = min(int(end_s * bytes_per_sec), len(pcm))
    if end_byte <= start_byte:
        return b''
    # Align to sample boundaries (sample_width * channels = 2 bytes).
    align = sample_width * channels
    start_byte -= start_byte % align
    end_byte -= end_byte % align
    pcm_slice = pcm[start_byte:end_byte]

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_slice)
    return buf.getvalue()


def _seg_get(seg, name, default=None):
    """Read an attribute or dict key from a segment, transparently."""
    if isinstance(seg, dict):
        return seg.get(name, default)
    return getattr(seg, name, default)


def _seg_set(seg, name, value):
    """Set an attribute or dict key on a segment, transparently."""
    if isinstance(seg, dict):
        seg[name] = value
    else:
        setattr(seg, name, value)


def _seg_pop(seg, name) -> None:
    """Remove an attribute or dict key from a segment, transparently."""
    if isinstance(seg, dict):
        seg.pop(name, None)
    else:
        seg.__dict__.pop(name, None)


def _stack_bank_vectors(bank: dict) -> np.ndarray:
    """Local copy of voiceprint_bank._stack_bank_vectors — duplicated here so
    `attribution.py` has no upward dependency on `voiceprint_bank.py`. Both
    sides MUST agree on the shape: `embeddings` first, then `continual_samples`.
    """
    rows = []
    for e in bank.get('embeddings') or []:
        v = e.get('v') if isinstance(e, dict) else None
        if v:
            rows.append(v)
    for e in bank.get('continual_samples') or []:
        v = e.get('v') if isinstance(e, dict) else None
        if v:
            rows.append(v)
    if not rows:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------


def attribute_user(
    transcript_segments: List,
    bank: dict,
    audio_pcm: bytes,
    sample_rate: int = 16000,
    embed_fn: Optional[Callable[[bytes], Optional[np.ndarray]]] = None,
) -> List:
    """Run the 4-stage Layer 2 attribution pipeline.

    Mutates each segment in place with `is_user: bool` and `attribution: dict`
    (per spec Layer 3 schema). Returns the same `transcript_segments` list for
    chainability.

    `embed_fn(wav_bytes)` should return an `(D,)` or `(1, D)` numpy array, or
    `None` / raise on transient failure. If `embed_fn` is `None`, we lazily
    import the production sync wrapper from `utils.stt.speaker_embedding`.
    """
    if embed_fn is None:
        embed_fn = _default_embed_fn

    if not transcript_segments:
        return transcript_segments

    bank_vecs = _stack_bank_vectors(bank)
    if bank_vecs.size == 0:
        # No bank — every segment is non-user with a defensible reason.
        for seg in transcript_segments:
            _seg_set(seg, 'is_user', False)
            _seg_set(
                seg,
                'attribution',
                _empty_attribution(bank, _seg_get(seg, 'speaker_id'), reason='no_bank'),
            )
        return transcript_segments

    t_strict = float(bank.get('calibrated_threshold') or T_STRICT_DEFAULT)
    fp_band_low = t_strict - FP_BORDERLINE_BAND_OFFSET
    fp_band_high = T_VOTE

    # ----- STAGE 0: sub-window embed every segment ------------------
    for seg in transcript_segments:
        sub_records = []
        seg_start = float(_seg_get(seg, 'start') or 0.0)
        seg_end = float(_seg_get(seg, 'end') or 0.0)
        duration = seg_end - seg_start
        if duration >= MIN_EMBED_DURATION:
            for ws, we in slide_sub_windows(
                seg_start, seg_end, SUB_WINDOW_SECONDS, SUB_WINDOW_OVERLAP, MIN_EMBED_DURATION
            ):
                wav = pcm_slice_to_wav(audio_pcm, ws, we, sample_rate)
                emb_arr = None
                dist_val: Optional[float] = None
                if wav:
                    try:
                        result = embed_fn(wav)
                        if result is not None:
                            arr = np.asarray(result, dtype=np.float32)
                            if arr.ndim == 1:
                                arr = arr.reshape(1, -1)
                            if arr.shape[1] == bank_vecs.shape[1]:
                                dist_val = float(cdist(arr, bank_vecs, metric='cosine').min())
                                emb_arr = arr
                            else:
                                logger.warning(
                                    "attribute_user: embedding dim mismatch %s vs bank %s",
                                    arr.shape,
                                    bank_vecs.shape,
                                )
                    except Exception as exc:
                        logger.warning("attribute_user: embed_fn raised, sub-window skipped: %s", exc)
                sub_records.append({'start': ws, 'end': we, 'dist': dist_val, 'emb': emb_arr})
        _seg_set(seg, '_sub', sub_records)

    # ----- STAGE 1: per-segment representative distance + strong-solo -----
    for seg in transcript_segments:
        sub = _seg_get(seg, '_sub') or []
        valid = [s for s in sub if s.get('dist') is not None]
        if valid:
            raw_dist = min(s['dist'] for s in valid)
        else:
            raw_dist = None
        _seg_set(seg, '_raw_dist', raw_dist)
        _seg_set(
            seg,
            '_strong_solo',
            raw_dist is not None and raw_dist < T_STRONG_SOLO,
        )

    # ----- STAGE 2: cluster vote + bimodal split detection -----
    clusters: dict = defaultdict(list)
    for seg in transcript_segments:
        clusters[_seg_get(seg, 'speaker_id')].append(seg)

    for sid, members in clusters.items():
        all_subs = []
        for m in members:
            for s in _seg_get(m, '_sub') or []:
                if s.get('dist') is not None:
                    all_subs.append(s)
        if not all_subs:
            for m in members:
                _seg_set(m, '_cluster_decision', 'other')
                _seg_set(m, '_cluster_split', False)
            continue

        # K_MAX cap — longest sub-windows first, ties broken by start_time.
        sampled = sorted(
            all_subs,
            key=lambda s: (-(s['end'] - s['start']), s['start']),
        )[:K_MAX]

        # Duration-weighted pass ratio.
        w_pass = sum((s['end'] - s['start']) for s in sampled if s['dist'] < T_VOTE)
        w_all = sum((s['end'] - s['start']) for s in sampled)
        pass_ratio = (w_pass / w_all) if w_all else 0.0

        # Bimodal split detection.
        is_mixed = False
        split_threshold: Optional[float] = None
        if len(sampled) >= 4:
            dists = sorted(s['dist'] for s in sampled)
            gaps = [(dists[i + 1] - dists[i], i) for i in range(len(dists) - 1)]
            mg, gi = max(gaps)
            low_mean = mean(dists[: gi + 1])
            high_mean = mean(dists[gi + 1 :])
            if (
                mg >= T_CLUSTER_SPLIT_GAP
                and (high_mean - low_mean) >= T_CLUSTER_SPLIT_DELTA
                and low_mean < T_VOTE
                and high_mean > T_VOTE
            ):
                is_mixed = True
                split_threshold = (low_mean + high_mean) / 2.0

        if is_mixed:
            for m in members:
                raw = _seg_get(m, '_raw_dist')
                decision = 'user' if (raw is not None and raw < split_threshold) else 'other'
                _seg_set(m, '_cluster_decision', decision)
                _seg_set(m, '_cluster_split', True)
        elif pass_ratio >= CLUSTER_MAJORITY_RATIO:
            for m in members:
                _seg_set(m, '_cluster_decision', 'user')
                _seg_set(m, '_cluster_split', False)
        elif pass_ratio <= CLUSTER_MINORITY_RATIO:
            for m in members:
                _seg_set(m, '_cluster_decision', 'other')
                _seg_set(m, '_cluster_split', False)
        else:
            # Ambiguous band — per-segment T_STRICT fallback.
            for m in members:
                raw = _seg_get(m, '_raw_dist')
                decision = 'user' if (raw is not None and raw < t_strict) else 'other'
                _seg_set(m, '_cluster_decision', decision)
                _seg_set(m, '_cluster_split', False)

    # ----- STAGE 3: first-person language soft prior (borderline only) -----
    for seg in transcript_segments:
        text = _seg_get(seg, 'text') or ''
        has_fp = bool(FIRST_PERSON_RE.search(text))
        _seg_set(seg, '_fp_present', has_fp)
        _seg_set(seg, '_fp_adjust', 0.0)
        d = _seg_get(seg, '_raw_dist')
        if d is None:
            _seg_set(seg, '_adj_dist', None)
            continue
        adj = d
        if fp_band_low <= d <= fp_band_high:
            if has_fp:
                adj += FP_PRIOR_BONUS
                _seg_set(seg, '_fp_adjust', FP_PRIOR_BONUS)
            else:
                adj += FP_PRIOR_PENALTY
                _seg_set(seg, '_fp_adjust', FP_PRIOR_PENALTY)
        _seg_set(seg, '_adj_dist', adj)

    # ----- STAGE 4: final decision + provenance write -----
    version = bank.get('version') or 'unknown'
    for seg in transcript_segments:
        raw = _seg_get(seg, '_raw_dist')
        adj = _seg_get(seg, '_adj_dist')
        cluster_decision = _seg_get(seg, '_cluster_decision')
        cluster_split = bool(_seg_get(seg, '_cluster_split'))
        strong_solo = bool(_seg_get(seg, '_strong_solo'))
        fp_present = bool(_seg_get(seg, '_fp_present'))
        fp_adjust = float(_seg_get(seg, '_fp_adjust') or 0.0)

        if raw is not None and raw >= T_HARD_REJECT:
            final, reason = False, 'hard_reject'
        elif strong_solo:
            final, reason = True, 'strong_solo'
        elif cluster_decision == 'user':
            final = True
            reason = 'cluster_split_lower_mode' if cluster_split else 'cluster_vote_user'
        elif (
            adj is not None
            and adj < t_strict
            and fp_band_low <= (raw if raw is not None else 1.0) <= fp_band_high
            and fp_present
        ):
            final, reason = True, 'fp_prior_rescue'
        else:
            final, reason = False, 'cluster_other'

        _seg_set(seg, 'is_user', final)
        sub = _seg_get(seg, '_sub') or []
        attribution = {
            'voiceprint_version': version,
            'algo_version': 'layer2-cluster-vote-v1',
            'distance_raw': raw,
            'distance_after_prior': adj,
            'n_sub_windows': len(sub),
            'cluster_id': _seg_get(seg, 'speaker_id'),
            'cluster_decision': cluster_decision or 'other',
            'cluster_split_detected': cluster_split,
            'strong_solo': strong_solo,
            'first_person_present': fp_present,
            'fp_adjust': fp_adjust,
            'decision_reason': reason,
            'extractor_eligible': bool(final and raw is not None and raw < T_VOTE),
        }
        _seg_set(seg, 'attribution', attribution)

        for k in (
            '_sub',
            '_raw_dist',
            '_strong_solo',
            '_cluster_decision',
            '_cluster_split',
            '_fp_present',
            '_fp_adjust',
            '_adj_dist',
        ):
            _seg_pop(seg, k)

    return transcript_segments


def _empty_attribution(bank: dict, cluster_id, reason: str) -> dict:
    return {
        'voiceprint_version': bank.get('version') or 'unknown',
        'algo_version': 'layer2-cluster-vote-v1',
        'distance_raw': None,
        'distance_after_prior': None,
        'n_sub_windows': 0,
        'cluster_id': cluster_id,
        'cluster_decision': 'other',
        'cluster_split_detected': False,
        'strong_solo': False,
        'first_person_present': False,
        'fp_adjust': 0.0,
        'decision_reason': reason,
        'extractor_eligible': False,
    }


def _default_embed_fn(wav_bytes: bytes) -> Optional[np.ndarray]:
    """Production fallback when the caller doesn't inject `embed_fn`.

    Forwards to `utils.stt.speaker_embedding.extract_embedding_from_bytes`
    (sync wrapper around the voice-extras HTTP boundary). Tests should
    always pass their own `embed_fn` to avoid touching the network.
    """
    return _prod_extract_embedding(wav_bytes)
