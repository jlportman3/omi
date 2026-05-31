"""Batch re-diarization / re-attribution driver (Layer 6 source).

Walks a single conversation (or every conversation for a user), downloads the
merged audio via `get_or_create_merged_audio`, then runs the Layer 2
`attribute_user` pipeline against the user's voiceprint bank. The same
algorithm runs in the live path, so this batch script is the canonical way
to "fix" past conversations after a bank version bump.

Usage:
    python -m scripts.rediarize --uid <uid> --conversation-id <cid> [--dry-run]
    python -m scripts.rediarize --uid <uid> [--limit 50] [--dry-run]

Notes:
- `--dry-run` prints a per-segment before/after table and does NOT write
  Firestore. Use it first to inspect impact.
- Live mode writes only `transcript_segments` (each segment's `is_user` and
  `attribution`), via the existing `update_conversation_segments` helper. No
  new Firestore helpers are introduced.
- All logging is PII-sanitized; raw transcript text is masked with
  `sanitize_pii` and uids/conversation ids are passed straight through (they
  are id strings, not PII).
"""

import argparse
import logging
import sys
from typing import List, Optional

from database import conversations as conversations_db
from routers.sync import pcm_to_wav
from utils.log_sanitizer import sanitize_pii
from utils.other.storage import get_or_create_merged_audio
from utils.stt.attribution import attribute_user
from utils.stt.speaker_embedding import extract_embedding_from_bytes
from utils.stt.voiceprint_bank import load_voiceprint_bank

logger = logging.getLogger(__name__)


AUDIO_SAMPLE_RATE = 16000


def _build_embed_fn():
    """Sync embed_fn for the batch path. Calls voice-extras directly.

    Returns None on failure so the attribution pipeline can skip the
    sub-window cleanly instead of aborting the whole conversation.
    """

    def _embed(wav_bytes: bytes):
        try:
            return extract_embedding_from_bytes(wav_bytes)
        except Exception as exc:  # noqa: BLE001 — best-effort batch path
            logger.warning("rediarize: voice-extras embedding failed: %s", exc)
            return None

    return _embed


def _load_audio_pcm(uid: str, conversation: dict) -> Optional[bytes]:
    """Build a single merged WAV buffer for the conversation and return its
    raw PCM payload (16-bit mono @ 16 kHz). Returns None when no audio file
    is attached.
    """
    audio_files = conversation.get('audio_files') or []
    if not audio_files:
        return None

    # Concatenate every audio file's PCM (most convs have only one).
    out = bytearray()
    for af in audio_files:
        af_id = af.get('id')
        timestamps = af.get('chunk_timestamps') or []
        if not af_id or not timestamps:
            continue
        try:
            wav_bytes, _ = get_or_create_merged_audio(
                uid=uid,
                conversation_id=conversation['id'],
                audio_file_id=af_id,
                timestamps=timestamps,
                pcm_to_wav_func=pcm_to_wav,
                fill_gaps=True,
                sample_rate=AUDIO_SAMPLE_RATE,
            )
        except Exception as exc:  # noqa: BLE001 — surface but continue
            logger.warning(
                "rediarize: get_or_create_merged_audio failed conv=%s af=%s: %s",
                conversation['id'],
                af_id,
                exc,
            )
            continue
        # Strip the 44-byte WAV header — `attribute_user` works on raw PCM.
        if len(wav_bytes) > 44:
            out.extend(wav_bytes[44:])
    return bytes(out) if out else None


def _print_diff_table(segments_before: List[dict], segments_after: List) -> None:
    """Pretty-print before/after attribution for `--dry-run`."""
    print(f"{'#':>3}  {'cluster':>8}  {'before':>6} -> {'after':>6}  {'dist':>6}  {'reason':<26}  {'text(masked)'}")
    for idx, (before, after) in enumerate(zip(segments_before, segments_after)):
        bef = bool(before.get('is_user'))
        aft = bool(getattr(after, 'is_user', None) if not isinstance(after, dict) else after.get('is_user'))
        attribution = after.get('attribution') if isinstance(after, dict) else getattr(after, 'attribution', {}) or {}
        d = attribution.get('distance_raw')
        d_str = f"{d:.3f}" if isinstance(d, (int, float)) else '  N/A'
        reason = attribution.get('decision_reason', '?')
        cluster = attribution.get('cluster_id', '?')
        text = before.get('text', '')
        print(
            f"{idx:>3}  {str(cluster):>8}  {str(bef):>6} -> {str(aft):>6}  "
            f"{d_str:>6}  {reason:<26}  {sanitize_pii(text)[:60]}"
        )


def _attribute_conversation(uid: str, conversation: dict, bank: dict, embed_fn, dry_run: bool) -> bool:
    """Run attribute_user on a single conversation. Returns True if written."""
    cid = conversation.get('id')
    segments_raw = conversation.get('transcript_segments') or []
    if not segments_raw:
        logger.info("rediarize: conv=%s has no segments — skipping", cid)
        return False

    audio_pcm = _load_audio_pcm(uid, conversation)
    if not audio_pcm:
        logger.info("rediarize: conv=%s has no audio — skipping", cid)
        return False

    logger.info(
        "rediarize: conv=%s segments=%d pcm_bytes=%d bank_version=%s",
        cid,
        len(segments_raw),
        len(audio_pcm),
        bank.get('version', 'unknown'),
    )

    # We want to operate on dicts so we can write them straight back.
    segments_copy = [dict(s) for s in segments_raw]
    before_snapshot = [dict(s) for s in segments_raw]

    attribute_user(segments_copy, bank, audio_pcm, sample_rate=AUDIO_SAMPLE_RATE, embed_fn=embed_fn)

    if dry_run:
        _print_diff_table(before_snapshot, segments_copy)
        return False

    conversations_db.update_conversation_segments(uid=uid, conversation_id=cid, segments=segments_copy)
    logger.info("rediarize: conv=%s wrote %d updated segments", cid, len(segments_copy))
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Re-attribute transcript segments using Layer 2 cluster vote.")
    parser.add_argument('--uid', required=True, help='Firestore user id (NOT email).')
    parser.add_argument('--conversation-id', help='Single conversation id (omit to walk all).')
    parser.add_argument('--limit', type=int, default=None, help='Max conversations to process when walking.')
    parser.add_argument('--dry-run', action='store_true', help='Print diff and do NOT write Firestore.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    bank = load_voiceprint_bank(args.uid)
    if not bank:
        logger.error("rediarize: no voiceprint bank for uid=%s — aborting", args.uid)
        return 2
    logger.info(
        "rediarize: loaded bank version=%s embeddings=%d continual=%d threshold=%s",
        bank.get('version', 'unknown'),
        len(bank.get('embeddings') or []),
        len(bank.get('continual_samples') or []),
        bank.get('calibrated_threshold'),
    )

    embed_fn = _build_embed_fn()

    if args.conversation_id:
        conv = conversations_db.get_conversation(args.uid, args.conversation_id)
        if not conv:
            logger.error("rediarize: conversation not found uid=%s cid=%s", args.uid, args.conversation_id)
            return 3
        _attribute_conversation(args.uid, conv, bank, embed_fn, args.dry_run)
        return 0

    n = 0
    for conv in conversations_db.iter_all_conversations(args.uid):
        if args.limit is not None and n >= args.limit:
            break
        try:
            _attribute_conversation(args.uid, conv, bank, embed_fn, args.dry_run)
        except Exception as exc:  # noqa: BLE001 — keep walking on per-conv failures
            logger.exception("rediarize: failed conv=%s: %s", conv.get('id'), exc)
        n += 1
    logger.info("rediarize: walked %d conversations", n)
    return 0


if __name__ == '__main__':
    sys.exit(main())
