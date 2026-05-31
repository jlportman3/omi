"""Unit tests for Layer 2 attribution (`utils.stt.attribution.attribute_user`).

Strategy:
- The bank is a single 2-D unit vector `[1.0, 0.0]`.
- The fake `embed_fn` recovers a target distance from the WAV bytes and
  returns `[cos(θ), sin(θ)]` where `cos(θ) = 1 - target_dist`. cdist(query,
  bank, 'cosine') then equals `target_dist` exactly.
- We tag each sub-window's target distance by embedding the value as a
  little-endian float64 sentinel at the END of the WAV byte buffer. Real
  WAV decoders ignore trailing data; our fake embed_fn just reads it back.

This keeps every test fully deterministic, with zero network and zero file
I/O, while exercising the real `attribute_user` code path including
`pcm_slice_to_wav` and `cdist`.
"""

import math
import struct
from typing import Optional

import numpy as np
import pytest

from utils.stt import attribution as attr
from utils.stt.attribution import (
    CLUSTER_MAJORITY_RATIO,
    CLUSTER_MINORITY_RATIO,
    FIRST_PERSON_RE,
    FP_PRIOR_BONUS,
    FP_PRIOR_PENALTY,
    K_MAX,
    MIN_EMBED_DURATION,
    SUB_WINDOW_OVERLAP,
    SUB_WINDOW_SECONDS,
    T_HARD_REJECT,
    T_STRICT_DEFAULT,
    T_STRONG_SOLO,
    T_VOTE,
    attribute_user,
)


# ----------------------------------------------------------------------
# Test infrastructure
# ----------------------------------------------------------------------

SAMPLE_RATE = 16000
SENTINEL_TAG = b'\xDEADBEEF__DIST__'  # marker before the float64 distance


# Bank with one 2-D anchor at [1.0, 0.0]. Cosine distance between this anchor
# and a unit-length [cos θ, sin θ] is `1 - cos θ`.
def make_bank(calibrated_threshold: float = 0.5473, version: str = 'bank-test-v1') -> dict:
    return {
        'version': version,
        'calibrated_threshold': calibrated_threshold,
        'embeddings': [{'v': [1.0, 0.0]}],
        'continual_samples': [],
    }


class Segment:
    """Minimal stand-in for `TranscriptSegment` — only the fields Layer 2 reads."""

    def __init__(self, start: float, end: float, speaker_id, text: str = ''):
        self.start = start
        self.end = end
        self.speaker_id = speaker_id
        self.text = text
        # populated by attribute_user
        self.is_user: Optional[bool] = None
        self.attribution: Optional[dict] = None


def _enough_pcm_for(seconds: float) -> bytes:
    """Return a silent PCM16 buffer long enough to cover `seconds`."""
    n_samples = int(math.ceil(seconds * SAMPLE_RATE)) + 1
    return b'\x00\x00' * n_samples


def _tag_distance(target: Optional[float]) -> bytes:
    """Encode a target distance as a trailing sentinel + float64."""
    if target is None:
        # Send NaN — embed_fn returns None for NaN to simulate transient failure.
        target = float('nan')
    return SENTINEL_TAG + struct.pack('<d', target)


def make_fake_embed_fn(distance_for_window):
    """Build an `embed_fn` that reads a per-window target distance from a
    side table. The window key is `(round(ws, 3), round(we, 3))`.

    Real `pcm_slice_to_wav` slices the PCM. To inject distances, we monkey-patch
    `pcm_slice_to_wav` *only inside the dispatch loop* (see `patched_attribute`)
    so it appends our sentinel + float64. This keeps `attribute_user`
    completely unmodified for tests.
    """

    def _embed(wav_bytes: bytes):
        idx = wav_bytes.rfind(SENTINEL_TAG)
        if idx == -1:
            # No tag → simulate failure path.
            return None
        try:
            (target,) = struct.unpack('<d', wav_bytes[idx + len(SENTINEL_TAG) : idx + len(SENTINEL_TAG) + 8])
        except struct.error:
            return None
        if math.isnan(target):
            return None
        # Build [cos θ, sin θ] with cos θ = 1 - target → cosine distance = target.
        cos_t = 1.0 - target
        cos_t = max(-1.0, min(1.0, cos_t))
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        return np.array([cos_t, sin_t], dtype=np.float32)

    _embed.lookup = distance_for_window
    return _embed


@pytest.fixture(autouse=True)
def patch_pcm_slicer(monkeypatch):
    """Replace `pcm_slice_to_wav` so each sub-window WAV carries a sentinel
    encoding the target distance for that window. Tests register the lookup
    via `attr._test_distance_table` (a module-level dict) keyed on
    `(round(ws, 3), round(we, 3))`.
    """
    if not hasattr(attr, '_test_distance_table'):
        attr._test_distance_table = {}

    original = attr.pcm_slice_to_wav

    def _patched(pcm, start_s, end_s, sample_rate=16000):
        wav = original(pcm, start_s, end_s, sample_rate)
        key = (round(start_s, 3), round(end_s, 3))
        target = attr._test_distance_table.get(key)
        return wav + _tag_distance(target)

    monkeypatch.setattr(attr, 'pcm_slice_to_wav', _patched)
    yield
    attr._test_distance_table = {}


def _set_window_distance(seg_start: float, seg_end: float, distances):
    """Populate the per-sub-window distance table for a segment.

    `distances` is a list — one entry per sub-window that `slide_sub_windows`
    will yield for `[seg_start, seg_end]`. `None` entries simulate transient
    failures.
    """
    windows = list(
        attr.slide_sub_windows(seg_start, seg_end, SUB_WINDOW_SECONDS, SUB_WINDOW_OVERLAP, MIN_EMBED_DURATION)
    )
    assert len(distances) == len(
        windows
    ), f"Expected {len(windows)} distances for window [{seg_start},{seg_end}], got {len(distances)}"
    for (ws, we), d in zip(windows, distances):
        attr._test_distance_table[(round(ws, 3), round(we, 3))] = d


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_k_max_truncation():
    """Cluster with 10 segments — sub-windows beyond K_MAX must not vote.

    We construct 10 single-window segments (each 0.8 s, one window each), all
    at dist=0.40 (well below T_VOTE=0.70, well above T_STRONG_SOLO=0.30).

    K_MAX=5 caps the SAMPLE size, but the cluster decision applies to ALL
    members regardless of which sub-windows participated in the vote. With
    all distances at 0.40, sampled = first 5 → pass_ratio = 5/5 = 1.0 →
    cluster_decision='user'. No bimodal split (all distances equal). Every
    member, sampled or not, gets `cluster_vote_user`.
    """
    segments = []
    for i in range(10):
        s = i * 1.0
        segments.append(Segment(s, s + 0.8, 'cluster_A'))
        _set_window_distance(s, s + 0.8, [0.40])

    bank = make_bank()
    pcm = _enough_pcm_for(20.0)
    embed = make_fake_embed_fn({})

    result = attribute_user(segments, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=embed)

    # K_MAX = 5: only the first 5 sub-windows vote, but the cluster decision
    # applies to ALL members — that's the whole point of cluster smoothing.
    for seg in result:
        assert seg.is_user is True
        assert seg.attribution['decision_reason'] == 'cluster_vote_user'
        assert seg.attribution['cluster_decision'] == 'user'
        assert seg.attribution['cluster_split_detected'] is False


def test_majority_vote_user():
    """60% of sub-windows below T_VOTE → cluster votes user.

    Use distances close enough to T_VOTE (0.70) that they don't trigger
    bimodal split detection (which needs gap ≥ T_CLUSTER_SPLIT_GAP=0.20).
    Distances: [0.65, 0.65, 0.65, 0.72, 0.72] → max gap 0.07 < 0.20, no
    split → pass_ratio = 3/5 = 0.60 ≥ CLUSTER_MAJORITY_RATIO → cluster='user'.
    """
    segments = []
    # 3 user-pass segments at 0.65 + 2 fail at 0.72 → pass_ratio = 3/5 = 0.60
    for i in range(3):
        s = i * 1.0
        segments.append(Segment(s, s + 0.8, 'cl'))
        _set_window_distance(s, s + 0.8, [0.65])
    for i in range(3, 5):
        s = i * 1.0
        segments.append(Segment(s, s + 0.8, 'cl'))
        _set_window_distance(s, s + 0.8, [0.72])

    bank = make_bank()
    pcm = _enough_pcm_for(10.0)
    result = attribute_user(segments, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    for seg in result:
        assert seg.is_user is True
        assert seg.attribution['cluster_decision'] == 'user'
        assert seg.attribution['cluster_split_detected'] is False


def test_minority_vote_other():
    """20% pass → cluster firmly other; ambiguous fallback NOT triggered.

    Use distances close enough that bimodal split detection does NOT fire
    (gap < T_CLUSTER_SPLIT_GAP=0.20). Distances: 1 × 0.65 + 4 × 0.72 →
    max gap 0.07 < 0.20, no split. pass_ratio = 1/5 = 0.20 → minority →
    cluster='other'. None of the segments are strong_solo (0.65 > 0.30).
    """
    segments = []
    # First seg at 0.65 (passes T_VOTE), rest at 0.72 (fail T_VOTE).
    for i in range(5):
        s = i * 1.0
        segments.append(Segment(s, s + 0.8, 'cl'))
        _set_window_distance(s, s + 0.8, [0.65 if i == 0 else 0.72])

    bank = make_bank()
    pcm = _enough_pcm_for(20.0)
    result = attribute_user(segments, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    # pass_ratio=0.20 → CLUSTER_MINORITY_RATIO threshold → cluster='other'.
    for seg in result:
        assert seg.attribution['cluster_decision'] == 'other'
        # NOT strong_solo (0.65 > T_STRONG_SOLO=0.30, 0.72 > 0.30)
        assert seg.attribution['strong_solo'] is False
        # Split detection did NOT fire (gap too small).
        assert seg.attribution['cluster_split_detected'] is False


def test_ambiguous_falls_back_to_strict():
    """30% pass → ambiguous; per-segment T_STRICT decides each member.

    With T_STRICT_DEFAULT=0.5473 (the bank default), a 0.40 segment goes user
    and a 0.75 segment goes other.
    """
    segments = []
    # Distances: 0.40, 0.40, 0.75, 0.75, 0.75, 0.75, 0.75 → 2/7 ≈ 0.286 (ambiguous)
    raw = [0.40, 0.40, 0.75, 0.75, 0.75, 0.75, 0.75]
    for i, d in enumerate(raw):
        s = i * 1.0
        segments.append(Segment(s, s + 0.8, 'cl'))
        _set_window_distance(s, s + 0.8, [d])

    bank = make_bank()
    pcm = _enough_pcm_for(15.0)
    result = attribute_user(segments, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    # K_MAX still picks the first 5: pass_ratio = 2/5 = 0.40 → ambiguous band.
    # Each segment falls back to T_STRICT (0.5473): 0.40 < 0.5473 → user, 0.75 > 0.5473 → other.
    for seg in result:
        if seg.attribution['distance_raw'] < T_STRICT_DEFAULT:
            assert seg.is_user is True
        else:
            assert seg.is_user is False


def test_bimodal_split_detection():
    """4 sub-windows with sorted dists [0.10, 0.15, 0.80, 0.85] → split fires.

    Note: 0.10 and 0.15 are below T_STRONG_SOLO=0.30 so those segments would
    also strong-solo to user — that's fine, the test still validates the
    cluster split for the high-mode segments.
    """
    # Construct 4 single-window segments (each 0.8 s, one window each).
    distances = [0.10, 0.15, 0.80, 0.85]
    segments = []
    for i, d in enumerate(distances):
        s = i * 1.0
        segments.append(Segment(s, s + 0.8, 'mixed'))
        _set_window_distance(s, s + 0.8, [d])

    bank = make_bank()
    pcm = _enough_pcm_for(10.0)
    result = attribute_user(segments, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    for seg in result:
        assert seg.attribution['cluster_split_detected'] is True
    # Low-mode segs → user (strong_solo OR cluster_split_lower_mode)
    assert result[0].is_user is True
    assert result[1].is_user is True
    # High-mode segs → other
    assert result[2].is_user is False
    assert result[3].is_user is False


def test_no_split_when_gap_too_small():
    """Gap < T_CLUSTER_SPLIT_GAP=0.20 → no split, normal vote logic."""
    distances = [0.40, 0.50, 0.60, 0.70]  # max gap = 0.10
    segments = []
    for i, d in enumerate(distances):
        s = i * 1.0
        segments.append(Segment(s, s + 0.8, 'cl'))
        _set_window_distance(s, s + 0.8, [d])

    bank = make_bank()
    pcm = _enough_pcm_for(10.0)
    result = attribute_user(segments, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    for seg in result:
        assert seg.attribution['cluster_split_detected'] is False


def test_strong_solo_overrides_cluster():
    """Single-segment cluster at dist=0.19 → is_user=True via strong_solo."""
    seg = Segment(0.0, 0.8, 'solo')
    _set_window_distance(0.0, 0.8, [0.19])

    bank = make_bank()
    pcm = _enough_pcm_for(5.0)
    result = attribute_user([seg], bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    assert result[0].is_user is True
    assert result[0].attribution['decision_reason'] == 'strong_solo'
    assert result[0].attribution['strong_solo'] is True


def test_hard_reject_above_085():
    """Raw distance 0.90 → is_user=False even if cluster votes user."""
    # Build a 6-member cluster: 5 strong (0.30 each — but that's strong_solo, so use 0.40)
    # and 1 hard-reject at 0.90.
    segments = []
    for i in range(5):
        s = i * 1.0
        segments.append(Segment(s, s + 0.8, 'cl'))
        _set_window_distance(s, s + 0.8, [0.40])
    bad = Segment(5.0, 5.8, 'cl')
    segments.append(bad)
    _set_window_distance(5.0, 5.8, [0.90])

    bank = make_bank()
    pcm = _enough_pcm_for(15.0)
    result = attribute_user(segments, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    # K_MAX=5 truncation samples first 5 (all 0.40s) → pass_ratio=1.0 → user.
    # But the 0.90 segment is still hard-rejected.
    for seg in result[:5]:
        assert seg.is_user is True
    assert result[5].is_user is False
    assert result[5].attribution['decision_reason'] == 'hard_reject'


def test_fp_prior_bonus():
    """Borderline seg with FP text → cluster='other' (minority) → adj rescues.

    fp_prior_rescue only fires when cluster_decision is 'other'. Construct a
    5-member cluster that votes minority/other (pass_ratio ≤ 0.20) but with
    a gap small enough to NOT trigger bimodal split:
      [0.58, 0.72, 0.72, 0.72, 0.72]  → gap 0.14 < 0.20, pass=1/5=0.20 → minority.
    Bank calibrated_threshold=0.55 → borderline band = (0.45, 0.70).
    The 0.58 seg with FP text gets adj=0.53 < 0.55 → fp_prior_rescue.
    """
    rescue_seg = Segment(0.0, 0.8, 'cl', text='I am Joe')
    _set_window_distance(0.0, 0.8, [0.58])
    other_segs = []
    for i in range(1, 5):
        s = i * 1.0
        seg = Segment(s, s + 0.8, 'cl')
        other_segs.append(seg)
        _set_window_distance(s, s + 0.8, [0.72])

    bank = make_bank(calibrated_threshold=0.55)
    pcm = _enough_pcm_for(10.0)
    result = attribute_user(
        [rescue_seg] + other_segs, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({})
    )

    # The rescued seg
    rescued = result[0]
    assert rescued.attribution['first_person_present'] is True
    assert rescued.attribution['fp_adjust'] == pytest.approx(FP_PRIOR_BONUS)
    assert rescued.attribution['distance_after_prior'] == pytest.approx(0.53)
    assert rescued.attribution['cluster_decision'] == 'other'  # cluster minority
    assert rescued.attribution['cluster_split_detected'] is False
    assert rescued.is_user is True
    assert rescued.attribution['decision_reason'] == 'fp_prior_rescue'


def test_fp_prior_penalty():
    """Borderline seg with no FP text in minority-vote cluster → adj penalised.

    Same 5-member minority-vote cluster as `test_fp_prior_bonus`, but the
    borderline 0.58 seg carries no first-person tokens. fp_adjust = +0.05 →
    distance_after_prior = 0.63. Cluster='other', no rescue available → is_user
    stays False with decision_reason='cluster_other'.
    """
    target_seg = Segment(0.0, 0.8, 'cl', text='the weather is nice today')
    _set_window_distance(0.0, 0.8, [0.58])
    other_segs = []
    for i in range(1, 5):
        s = i * 1.0
        seg = Segment(s, s + 0.8, 'cl')
        other_segs.append(seg)
        _set_window_distance(s, s + 0.8, [0.72])

    bank = make_bank(calibrated_threshold=0.55)
    pcm = _enough_pcm_for(10.0)
    result = attribute_user(
        [target_seg] + other_segs, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({})
    )

    target = result[0]
    assert target.attribution['first_person_present'] is False
    assert target.attribution['fp_adjust'] == pytest.approx(FP_PRIOR_PENALTY)
    assert target.attribution['distance_after_prior'] == pytest.approx(0.63)
    assert target.attribution['cluster_decision'] == 'other'
    assert target.attribution['cluster_split_detected'] is False
    assert target.is_user is False
    assert target.attribution['decision_reason'] == 'cluster_other'


def test_fp_apostrophe_variants():
    """FIRST_PERSON_RE matches both ASCII apostrophe and U+2019."""
    assert FIRST_PERSON_RE.search("I'm here")
    assert FIRST_PERSON_RE.search("I’m here")  # curly apostrophe
    assert FIRST_PERSON_RE.search("I've seen it")
    assert FIRST_PERSON_RE.search("I’ve seen it")
    assert FIRST_PERSON_RE.search("my dog")
    # Negative: no first-person tokens
    assert not FIRST_PERSON_RE.search("the dog ran")


def test_short_segment_no_embed():
    """Segment shorter than MIN_EMBED_DURATION → distance_raw=None and
    contributes nothing to cluster vote.
    """
    seg = Segment(0.0, 0.3, 'cl')  # 0.3 s < 0.5 s
    bank = make_bank()
    pcm = _enough_pcm_for(2.0)
    result = attribute_user([seg], bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    assert result[0].attribution['distance_raw'] is None
    assert result[0].attribution['n_sub_windows'] == 0
    assert result[0].is_user is False
    assert result[0].attribution['decision_reason'] == 'cluster_other'


def test_provenance_complete():
    """Every segment carries the full Layer 3 attribution dict; intermediate
    `_` fields are stripped from the segment.
    """
    seg = Segment(0.0, 0.8, 'cl', text='hello')
    _set_window_distance(0.0, 0.8, [0.40])

    bank = make_bank()
    pcm = _enough_pcm_for(5.0)
    result = attribute_user([seg], bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    attribution = result[0].attribution
    expected_keys = {
        'voiceprint_version',
        'algo_version',
        'distance_raw',
        'distance_after_prior',
        'n_sub_windows',
        'cluster_id',
        'cluster_decision',
        'cluster_split_detected',
        'strong_solo',
        'first_person_present',
        'fp_adjust',
        'decision_reason',
        'extractor_eligible',
    }
    assert set(attribution.keys()) == expected_keys
    assert attribution['algo_version'] == 'layer2-cluster-vote-v1'
    assert attribution['voiceprint_version'] == 'bank-test-v1'

    # Intermediate hygiene: no `_sub`, `_raw_dist`, etc. left on the segment.
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
        assert not hasattr(result[0], k) or getattr(result[0], k, None) is None


def test_jarvis_tts_strong_solo_path():
    """End-to-end Jarvis-TTS case: cloned voice at dist=0.19 → strong_solo."""
    # Realistic-ish: Jarvis speaks one sub-windowable segment in an otherwise
    # mixed conversation.
    jarvis = Segment(0.0, 1.5, 'jarvis_speaker', text='Online and ready, sir.')
    _set_window_distance(0.0, 1.5, [0.19])

    bank = make_bank()
    pcm = _enough_pcm_for(5.0)
    result = attribute_user([jarvis], bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    assert result[0].is_user is True
    assert result[0].attribution['decision_reason'] == 'strong_solo'
    assert result[0].attribution['strong_solo'] is True
    assert result[0].attribution['extractor_eligible'] is True
    assert result[0].attribution['distance_raw'] == pytest.approx(0.19, abs=1e-3)


def test_decision_reasons_exhaustive():
    """One conversation exercising every decision_reason variant."""
    segments = []
    text_lookup = {}

    # 1. strong_solo: standalone cluster at 0.19
    segments.append(Segment(0.0, 0.8, 'solo_cluster'))
    _set_window_distance(0.0, 0.8, [0.19])

    # 2. hard_reject: standalone at 0.92
    segments.append(Segment(1.0, 1.8, 'reject_cluster'))
    _set_window_distance(1.0, 1.8, [0.92])

    # 3. cluster_vote_user: 5-member cluster at 0.40 → unanimous pass
    for i in range(5):
        s = 3.0 + i * 1.0
        segments.append(Segment(s, s + 0.8, 'vote_cluster'))
        _set_window_distance(s, s + 0.8, [0.40])

    # 4. cluster_split_lower_mode: 4-member cluster with bimodal distances
    bimodal_dists = [0.10, 0.15, 0.80, 0.85]
    for i, d in enumerate(bimodal_dists):
        s = 9.0 + i * 1.0
        segments.append(Segment(s, s + 0.8, 'split_cluster'))
        _set_window_distance(s, s + 0.8, [d])

    # 5. fp_prior_rescue: single seg at dist=0.50 with FP text, bank thresh 0.48
    #    (We'll need a second bank — keep it simple and verify reason just for
    #    a single seg via a dedicated sub-test.)

    # 6. cluster_other: standalone at 0.75 (no FP)
    segments.append(Segment(14.0, 14.8, 'reject_solo', text='just the weather'))
    _set_window_distance(14.0, 14.8, [0.75])

    bank = make_bank()
    pcm = _enough_pcm_for(20.0)
    result = attribute_user(segments, bank, pcm, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))

    reasons = {seg.attribution['decision_reason'] for seg in result}
    assert 'strong_solo' in reasons
    assert 'hard_reject' in reasons
    assert 'cluster_vote_user' in reasons
    # The split cluster's low-mode segs strong-solo (0.10/0.15 < T_STRONG_SOLO)
    # and the high-mode segs route via cluster_split_lower_mode → 'cluster_other'?
    # No — when cluster_split fires and a seg routes to 'other', the final
    # decision falls through to `cluster_other`. So we should see that reason.
    assert 'cluster_other' in reasons

    # Validate cluster_split_lower_mode shows up for any low-mode seg that
    # ISN'T also strong_solo. With our distances both low-mode segs (0.10, 0.15)
    # are below T_STRONG_SOLO=0.30, so they take the strong_solo path before
    # cluster_split_lower_mode. That's expected per the decision tree order.
    # Verify the split path is exercised by checking cluster_split_detected:
    split_segs = [s for s in result if s.attribution['cluster_split_detected']]
    assert len(split_segs) == 4

    # Now exercise fp_prior_rescue separately. Needs a minority-vote cluster
    # so cluster_decision='other' (otherwise cluster_vote_user pre-empts).
    target = Segment(0.0, 0.8, 'cl', text='I am Joe')
    _set_window_distance(0.0, 0.8, [0.58])
    others = []
    for i in range(1, 5):
        s = i * 1.0
        o = Segment(s, s + 0.8, 'cl')
        others.append(o)
        _set_window_distance(s, s + 0.8, [0.72])
    bank2 = make_bank(calibrated_threshold=0.55)
    pcm2 = _enough_pcm_for(10.0)
    r2 = attribute_user([target] + others, bank2, pcm2, sample_rate=SAMPLE_RATE, embed_fn=make_fake_embed_fn({}))
    assert r2[0].attribution['decision_reason'] == 'fp_prior_rescue'


def test_constants_within_expected_ranges():
    """Sanity check on tunable constants — guards against env-override drift."""
    assert 0.0 < T_STRONG_SOLO < T_STRICT_DEFAULT < T_VOTE < T_HARD_REJECT <= 1.0
    assert 0.0 < CLUSTER_MINORITY_RATIO < CLUSTER_MAJORITY_RATIO <= 1.0
    assert K_MAX >= 1
    assert SUB_WINDOW_SECONDS > 0
    assert MIN_EMBED_DURATION > 0
    assert FP_PRIOR_BONUS < 0  # bonus pushes distance DOWN toward user
    assert FP_PRIOR_PENALTY > 0
