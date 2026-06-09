"""Batch memory re-extraction across recent conversations.

Companion to the ``rediarize.py`` and ``trigger_wal_drain`` utilities. Use
this to back-fill memories on conversations that completed before the
QwenChatOpenAI / QwenPydanticOutputParser fixes landed (see commits
8672aedfa, 53cac328a, 10b48b33b, 18e03d2db, 86a282700, bd49a178b), or any
time the memory extractor was silently failing under
``MODEL_QOS=local_only``.

Calls ``_extract_memories_inner`` SYNCHRONOUSLY so the script can verify
each save. The production code path schedules it on
``postprocess_executor``; that returns before the save completes, which
hides extraction failures from naive callers (see today's session debug
log for the rabbit hole this avoids).

Anchors ``GOOGLE_APPLICATION_CREDENTIALS`` relative to ``backend/`` the
same way ``rediarize.py`` does — lets the script run cleanly from any
checkout / worktree.

CLI:
    python scripts/batch_reextract_memories.py --uid <uid> [--lookback-days 30]
                                                [--limit 50] [--dry-run]
                                                [--include-with-memories]

Defaults reflect the safe path: only re-extract on conversations that
currently have ZERO memories. Pass ``--include-with-memories`` to
re-extract everything in the lookback window (rare; resets any memories
already saved for a conv as part of normal _extract_memories_inner
behaviour, then re-creates them).
"""

import argparse
import logging
import os
import pathlib
import sys
from datetime import datetime, timezone, timedelta

# Anchor relative GOOGLE_APPLICATION_CREDENTIALS to backend/ — same trick
# rediarize.py uses so this script runs from any checkout/worktree.
creds_env = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if creds_env and not os.path.isabs(creds_env) and not os.path.exists(creds_env):
    backend_root = pathlib.Path(__file__).resolve().parent.parent
    anchored = backend_root / creds_env
    if anchored.exists():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(anchored)

from database import memories as memories_db
from database.conversations import get_conversation, get_conversations
from database.memories import get_memories
from models.conversation import Conversation
from utils.conversations.process_conversation import _extract_memories_inner
from utils.log_sanitizer import sanitize_pii

logger = logging.getLogger('batch_reextract_memories')


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--uid', required=True, help='Firebase UID of the user whose convs to scan')
    p.add_argument(
        '--lookback-days',
        type=int,
        default=30,
        help='Only scan conversations whose started_at is within this many days (default: 30)',
    )
    p.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Max number of recent conversations to fetch + scan (default: 50)',
    )
    p.add_argument(
        '--include-with-memories',
        action='store_true',
        help='Re-extract on conversations that already have ≥1 memory. Default: skip them.',
    )
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='Print candidates but do not invoke _extract_memories_inner.',
    )
    return p


def _list_candidates(uid: str, lookback_days: int, limit: int, include_with_memories: bool):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    rows = get_conversations(uid, limit=limit, offset=0, statuses=[], include_discarded=False)

    # Index existing memories by conv id so we can identify "zero memories" candidates without
    # making N round-trips.
    all_mems = get_memories(uid, limit=max(limit * 8, 400))
    mems_by_conv: dict = {}
    for m in all_mems:
        cid = m.get('conversation_id')
        if cid:
            mems_by_conv.setdefault(cid, []).append(m)

    candidates = []
    for c in rows:
        if c.get('status') != 'completed':
            continue
        started = c.get('started_at')
        if not started or started < cutoff:
            continue
        segs = c.get('transcript_segments') or []
        eligible = [
            s
            for s in segs
            if s.get('is_user')
            and (not s.get('attribution') or (s.get('attribution') or {}).get('extractor_eligible', False))
        ]
        if not eligible:
            continue
        cur_mems = len(mems_by_conv.get(c['id'], []))
        if cur_mems > 0 and not include_with_memories:
            continue
        title = (c.get('structured') or {}).get('title') or '(no title)'
        candidates.append((c['id'], title, len(eligible), cur_mems))
    return candidates


def _extract_one(uid: str, cid: str) -> int:
    """Run synchronous _extract_memories_inner; return final memory count."""
    c = get_conversation(uid, cid)
    if not c:
        return 0
    conv = Conversation(**c)
    _extract_memories_inner(uid, conv)
    fresh = [m for m in get_memories(uid, limit=400) if m.get('conversation_id') == cid]
    return len(fresh)


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    uid = args.uid
    logger.info('Scanning uid=%s lookback_days=%d limit=%d', sanitize_pii(uid), args.lookback_days, args.limit)

    candidates = _list_candidates(uid, args.lookback_days, args.limit, args.include_with_memories)
    logger.info(
        'Found %d candidates (status=completed, eligible_segs>0%s)',
        len(candidates),
        '' if args.include_with_memories else ', existing_memories=0',
    )
    for cid, title, nseg, cur in candidates:
        logger.info('  candidate %s  nseg=%d  cur_mems=%d  %r', cid[:8], nseg, cur, title[:60])

    if args.dry_run:
        logger.info('--dry-run: not invoking _extract_memories_inner')
        return 0

    results = []
    for cid, title, nseg, cur in candidates:
        try:
            new_count = _extract_one(uid, cid)
            results.append((cid, title, new_count))
            logger.info('  %s  %r → %d memories', cid[:8], title[:50], new_count)
        except Exception as e:
            logger.exception('  %s  ERROR during re-extract: %s', cid[:8], e)
            results.append((cid, title, -1))

    n_ok = sum(1 for _, _, n in results if n > 0)
    n_zero = sum(1 for _, _, n in results if n == 0)
    n_err = sum(1 for _, _, n in results if n < 0)
    total_new = sum(n for _, _, n in results if n > 0)
    logger.info(
        'Summary: candidates=%d  with_memories_now=%d  still_empty=%d  errors=%d  total_new_memories=%d',
        len(candidates),
        n_ok,
        n_zero,
        n_err,
        total_new,
    )
    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
