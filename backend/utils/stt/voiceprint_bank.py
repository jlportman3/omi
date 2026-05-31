"""Per-process voiceprint bank loader and matcher (Layer 2 helper).

Splits responsibility away from `speaker_embedding.py`:
- This module is for **bank** I/O (Firestore read + cache) and **matching** an
  embedding against the bank.
- `speaker_embedding.py` stays focused on the wespeaker/titanet HTTP boundary.

The bank document lives at `users/{uid}.voiceprint_bank` and has the shape:

    {
        "version":               "bank-YYYYMMDD-HHMMSS-...",
        "calibrated_threshold":  0.5473,
        "embeddings":            [{"v": [192 floats]}, ...],
        "continual_samples":     [{"v": [192 floats]}, ...],
        # optional, plus other provenance fields
    }

`load_voiceprint_bank` returns the dict **as stored** — it does not unwrap the
embeddings. Callers that want to use the vectors should hand the dict to
`match_against_bank`, which extracts both `embeddings` and `continual_samples`
into a single `(N, D)` numpy matrix and runs a cosine-distance match.

A per-process LRU cache keyed on `(uid, version)` avoids re-reading Firestore
for every conversation; when the version changes (Layer 1 augments the bank or
a continual sample lands) the next call naturally repopulates because the key
changes.
"""

import logging
from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from database._client import db

logger = logging.getLogger(__name__)


# A tiny version-cache layer: keep the latest version string per uid so we can
# invalidate transparently when the bank rolls over.
_VERSION_CACHE: dict = {}


def _read_bank_doc(uid: str) -> dict:
    """Single Firestore read of `users/{uid}.voiceprint_bank`. No caching."""
    snap = db.collection('users').document(uid).get(field_paths=['voiceprint_bank'])
    if not snap.exists:
        return {}
    data = snap.to_dict() or {}
    bank = data.get('voiceprint_bank') or {}
    return bank


@lru_cache(maxsize=64)
def _cached_load(uid: str, version: str) -> dict:
    """LRU cache keyed on (uid, version). Re-reads when version changes.

    `version` is captured by `load_voiceprint_bank` BEFORE this call so a
    bumped bank doesn't keep returning the stale snapshot.
    """
    # version is part of the cache key only — we still need to read the doc.
    return _read_bank_doc(uid)


def load_voiceprint_bank(uid: str) -> dict:
    """Return the user's voiceprint bank dict, cached per `(uid, version)`.

    The dict is returned **as stored** — embeddings stay wrapped in `{'v': [...]}`
    because that's what the matcher unwraps. Callers that don't intend to call
    `match_against_bank` can read other fields (`calibrated_threshold`,
    `version`, etc.) directly.

    Returns `{}` if the user has no bank yet.
    """
    # Read once to discover current version, then dispatch through the LRU
    # so that subsequent calls within the same version skip Firestore.
    current = _read_bank_doc(uid)
    version = (current or {}).get('version') or ''
    if not current:
        # Nothing to cache; just return empty.
        return {}
    cached_version = _VERSION_CACHE.get(uid)
    if cached_version != version:
        # Bank rolled over — drop the LRU entry for the prior version of this uid.
        _cached_load.cache_clear()
        _VERSION_CACHE[uid] = version
    return _cached_load(uid, version) or current


def _stack_bank_vectors(bank: dict) -> np.ndarray:
    """Concatenate `embeddings` + `continual_samples` into a (N, D) float32 matrix.

    Both fields are stored as Firestore arrays-of-maps, each entry shaped
    `{'v': [192 floats]}`. Missing fields are tolerated and contribute zero
    rows. If both are empty the returned array has shape `(0,)`.
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


def match_against_bank(emb: np.ndarray, bank: dict) -> Tuple[float, bool]:
    """Return `(nearest_cosine_distance, is_match)` for `emb` against the bank.

    `emb` is `(D,)` or `(1, D)`. The bank is unwrapped on every call (the
    stacked matrix is tiny — ~110 KB for 146x192 — and avoiding a side cache
    keeps the lifecycle simple).

    `is_match` is judged against `bank['calibrated_threshold']`; if absent we
    fall back to the legacy `SPEAKER_MATCH_THRESHOLD` (0.45) so callers can
    still get a binary answer when a user hasn't been calibrated yet.

    Returns `(2.0, False)` when the bank is empty or shapes mismatch.
    """
    bank_vecs = _stack_bank_vectors(bank)
    if bank_vecs.size == 0:
        return 2.0, False

    query = np.asarray(emb, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)

    if query.shape[1] != bank_vecs.shape[1]:
        logger.warning(
            "voiceprint_bank match: dim mismatch query=%s bank=%s",
            query.shape,
            bank_vecs.shape,
        )
        return 2.0, False

    distances = cdist(query, bank_vecs, metric='cosine')
    nearest = float(distances.min())
    threshold = float(bank.get('calibrated_threshold') or 0.45)
    return nearest, (nearest < threshold)


def clear_cache() -> None:
    """Test/admin helper — drop the per-process bank cache."""
    _cached_load.cache_clear()
    _VERSION_CACHE.clear()
