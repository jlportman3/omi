#!/usr/bin/env python3
"""
backfill_vector_db.py — Backfill Firestore entities into Qdrant (omi_*) collections.

Walks Firestore for a single UID and re-embeds + upserts:
  - conversations       → omi_conversations
  - memories            → omi_memories
  - action_items        → omi_action_items
  - screen_activity     → omi_screen_activity   (skipped — embeddings live on the
                                                 desktop client; we don't have
                                                 raw OCR text to re-embed here)

Embeddings are produced via the same `utils.llm.clients.embeddings` proxy the
runtime uses (OpenAI text-embedding-3-large over the LiteLLM proxy).

Usage:
  python scripts/backfill_vector_db.py --uid <UID> [--collection all] [--limit N] [--dry-run]
  python scripts/backfill_vector_db.py --uid <UID> --collection memories
  python scripts/backfill_vector_db.py --uid <UID> --collection conversations --limit 100

Environment (already loaded from backend/.env in normal runs):
  GOOGLE_APPLICATION_CREDENTIALS   Firebase service account key
  OPENAI_API_KEY / OPENAI_BASE_URL LiteLLM proxy for embeddings
  QDRANT_URL                       Defaults to http://localhost:6333
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List

import firebase_admin
from firebase_admin import credentials, firestore

# Make `database`, `utils`, `models`, etc. importable regardless of cwd.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import vector_db  # noqa: E402
from utils.llm.clients import embeddings as _embeddings  # noqa: E402
from utils.log_sanitizer import sanitize, sanitize_pii  # noqa: E402

logger = logging.getLogger('backfill_vector_db')


# ---------------------------------------------------------------------------
# Firebase / Firestore bootstrap (mirrors migration 005 pattern)
# ---------------------------------------------------------------------------
try:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)
except ValueError:
    pass  # app already initialized
except Exception as e:
    logger.error('Firebase init failed — check GOOGLE_APPLICATION_CREDENTIALS')
    logger.error(sanitize(str(e)))
    sys.exit(1)

db = firestore.client()


CONVERSATIONS_COLLECTION = 'conversations'
MEMORIES_COLLECTION = 'memories'
ACTION_ITEMS_COLLECTION = 'action_items'

ALL_TARGETS = ['conversations', 'memories', 'action_items']
VALID_TARGETS = ALL_TARGETS + ['screen_activity', 'all']

PROGRESS_EVERY = 500
BATCH_SIZE = 100  # embed + upsert this many per LiteLLM/Qdrant round-trip
PAGE_SIZE = 1000  # Firestore docs fetched per cursor page


# ---------------------------------------------------------------------------
# Firestore reader
# ---------------------------------------------------------------------------


def _read_all_docs(uid: str, name: str, limit: int = None) -> List[Dict]:
    """Materialize all docs for a user collection, paginating by document id.

    CRITICAL: we read each page into memory and CLOSE the cursor before doing
    any slow per-item work (embedding/upsert). The earlier implementation held
    a single ``col.stream()`` generator open across the whole collection while
    embedding inline — Firestore's server-side stream deadline (~60s) then
    fired a ``504 Deadline Exceeded`` partway through large collections. Paging
    keeps every cursor short-lived.
    """
    col = db.collection('users').document(uid).collection(name)
    out: List[Dict] = []
    cursor = None
    while True:
        q = col.order_by('__name__').limit(PAGE_SIZE)
        if cursor is not None:
            q = q.start_after(cursor)
        snaps = list(q.stream())  # cursor open only for this fast read
        if not snaps:
            break
        for snap in snaps:
            data = snap.to_dict() or {}
            data['id'] = snap.id
            out.append(data)
            if limit and len(out) >= limit:
                return out
        if len(snaps) < PAGE_SIZE:
            break
        cursor = snaps[-1]
    return out


# ---------------------------------------------------------------------------
# Per-collection backfillers
# ---------------------------------------------------------------------------


def backfill_memories(uid: str, limit: int = None, dry_run: bool = False) -> Dict[str, int]:
    stats = {'total': 0, 'written': 0, 'skipped': 0, 'errors': 0}
    buf: List[Dict] = []
    for memory in _read_all_docs(uid, MEMORIES_COLLECTION, limit=limit):
        stats['total'] += 1
        content = memory.get('content') or ''
        if not content.strip():
            stats['skipped'] += 1
            continue
        buf.append(
            {
                'memory_id': memory['id'],
                'content': content,
                'category': memory.get('category', 'system'),
            }
        )
        if len(buf) >= BATCH_SIZE:
            stats['written'] += _flush_memories(uid, buf, dry_run)
            buf = []
        if stats['total'] % PROGRESS_EVERY == 0:
            logger.info(f'memories progress uid={sanitize_pii(uid)} processed={stats["total"]}')
    if buf:
        stats['written'] += _flush_memories(uid, buf, dry_run)
    return stats


def _flush_memories(uid: str, buf: List[Dict], dry_run: bool) -> int:
    if dry_run:
        return len(buf)
    try:
        return vector_db.upsert_memory_vectors_batch(uid, buf)
    except Exception as e:
        logger.error(f'memories flush failed uid={sanitize_pii(uid)} batch_size={len(buf)}: {sanitize(str(e))}')
        return 0


def backfill_action_items(uid: str, limit: int = None, dry_run: bool = False) -> Dict[str, int]:
    stats = {'total': 0, 'written': 0, 'skipped': 0, 'errors': 0}
    buf: List[Dict] = []
    for ai in _read_all_docs(uid, ACTION_ITEMS_COLLECTION, limit=limit):
        stats['total'] += 1
        desc = ai.get('description') or ai.get('text') or ''
        if not desc.strip():
            stats['skipped'] += 1
            continue
        buf.append({'action_item_id': ai['id'], 'description': desc})
        if len(buf) >= BATCH_SIZE:
            stats['written'] += _flush_action_items(uid, buf, dry_run)
            buf = []
        if stats['total'] % PROGRESS_EVERY == 0:
            logger.info(f'action_items progress uid={sanitize_pii(uid)} processed={stats["total"]}')
    if buf:
        stats['written'] += _flush_action_items(uid, buf, dry_run)
    return stats


def _flush_action_items(uid: str, buf: List[Dict], dry_run: bool) -> int:
    if dry_run:
        return len(buf)
    try:
        return vector_db.upsert_action_item_vectors_batch(uid, buf)
    except Exception as e:
        logger.error(f'action_items flush failed uid={sanitize_pii(uid)} batch_size={len(buf)}: {sanitize(str(e))}')
        return 0


def _conversation_text(conv: Dict) -> str:
    """
    Reconstruct the text to embed for a conversation. Mirrors what the
    Pinecone-era upsert path did: title + overview + transcript-derived
    structured fields. Errors on the side of including content; embeddings
    tolerate long inputs and trailing junk.
    """
    structured = conv.get('structured') or {}
    parts: List[str] = []
    title = structured.get('title')
    overview = structured.get('overview')
    if title:
        parts.append(str(title))
    if overview:
        parts.append(str(overview))
    # Some conversations only have transcript_segments; concatenate text fields.
    segments = conv.get('transcript_segments') or []
    if segments:
        seg_texts = [s.get('text', '') for s in segments if isinstance(s, dict)]
        parts.extend([t for t in seg_texts if t])
    return '\n'.join(parts).strip()


def backfill_conversations(uid: str, limit: int = None, dry_run: bool = False) -> Dict[str, int]:
    stats = {'total': 0, 'written': 0, 'skipped': 0, 'errors': 0}

    # 1. Build the work list from a fully-materialized read (no Firestore cursor
    #    held open during the slow embed step that follows).
    work: List[Dict] = []  # [{id, text, metadata}, ...]
    for conv in _read_all_docs(uid, CONVERSATIONS_COLLECTION, limit=limit):
        stats['total'] += 1
        text = _conversation_text(conv)
        if not text:
            stats['skipped'] += 1
            continue
        metadata: Dict = {}
        structured = conv.get('structured') or {}
        for key in ('topics', 'entities', 'people', 'dates'):
            val = structured.get(key)
            if val:
                metadata[key] = val
        work.append({'id': conv['id'], 'text': text, 'metadata': metadata})

    # 2. Batch-embed + upsert. One LiteLLM call per BATCH_SIZE texts instead of
    #    one call per conversation — turns ~6k sequential round-trips into ~60.
    for i in range(0, len(work), BATCH_SIZE):
        batch = work[i : i + BATCH_SIZE]
        if dry_run:
            stats['written'] += len(batch)
        else:
            try:
                vectors = _embeddings.embed_documents([w['text'] for w in batch])
                for w, vec in zip(batch, vectors):
                    vector_db.upsert_vector2(uid, w['id'], vec, w['metadata'])
                    stats['written'] += 1
            except Exception as e:
                stats['errors'] += len(batch)
                logger.error(
                    f'conversation batch embed/upsert failed uid={sanitize_pii(uid)} '
                    f'batch_start={i} size={len(batch)}: {sanitize(str(e))}'
                )
        if (i // BATCH_SIZE) % 5 == 0:
            logger.info(
                f'conversations progress uid={sanitize_pii(uid)} '
                f'processed={min(i + BATCH_SIZE, len(work))}/{len(work)}'
            )
    return stats


def backfill_screen_activity(uid: str, limit: int = None, dry_run: bool = False) -> Dict[str, int]:
    """
    Screen activity embeddings are produced client-side (Gemini embedding-001 on
    the desktop app) and pushed via the existing /v1/screen-activity upsert path.
    Server has no raw OCR text to re-embed. This stub is here so --collection all
    doesn't fail; integrators who need a true backfill must replay desktop sync.
    """
    logger.warning(
        f'screen_activity backfill is a no-op uid={sanitize_pii(uid)} — '
        f'desktop client is the source of truth for these embeddings.'
    )
    return {'total': 0, 'written': 0, 'skipped': 0, 'errors': 0}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BACKFILLERS = {
    'conversations': backfill_conversations,
    'memories': backfill_memories,
    'action_items': backfill_action_items,
    'screen_activity': backfill_screen_activity,
}


def parse_args():
    p = argparse.ArgumentParser(description='Backfill Firestore entities into Qdrant')
    p.add_argument('--uid', required=True, help='User UID to backfill')
    p.add_argument(
        '--collection',
        choices=VALID_TARGETS,
        default='all',
        help='Which collection to backfill (default: all)',
    )
    p.add_argument('--limit', type=int, default=None, help='Per-collection cap on rows processed')
    p.add_argument('--dry-run', action='store_true', help='Skip embed+upsert; just walk Firestore')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    if vector_db.client is None:
        logger.error('Qdrant client failed to initialize — refusing to backfill.')
        sys.exit(2)

    targets = ALL_TARGETS if args.collection == 'all' else [args.collection]
    logger.info(f'Backfill start uid={sanitize_pii(args.uid)} targets={targets} dry_run={args.dry_run}')

    overall = {}
    t0 = time.time()
    for t in targets:
        fn = BACKFILLERS[t]
        s0 = time.time()
        stats = fn(args.uid, limit=args.limit, dry_run=args.dry_run)
        elapsed = time.time() - s0
        logger.info(
            f'[{t}] uid={sanitize_pii(args.uid)} '
            f'total={stats["total"]} written={stats["written"]} skipped={stats["skipped"]} '
            f'errors={stats["errors"]} elapsed={elapsed:.1f}s'
        )
        overall[t] = stats

    elapsed = time.time() - t0
    logger.info(f'Backfill complete uid={sanitize_pii(args.uid)} elapsed={elapsed:.1f}s summary={overall}')


if __name__ == '__main__':
    main()
