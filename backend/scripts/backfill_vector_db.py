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
from typing import Dict, Iterable, List

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

PROGRESS_EVERY = 50
BATCH_SIZE = 50  # for batch upserts where supported


# ---------------------------------------------------------------------------
# Firestore iterators
# ---------------------------------------------------------------------------


def _iter_user_collection(uid: str, name: str, limit: int = None) -> Iterable[Dict]:
    col = db.collection('users').document(uid).collection(name)
    if limit:
        col = col.limit(limit)
    for doc in col.stream():
        data = doc.to_dict() or {}
        data['id'] = doc.id
        yield data


# ---------------------------------------------------------------------------
# Per-collection backfillers
# ---------------------------------------------------------------------------


def _embed_text(text: str) -> List[float]:
    # The runtime embeddings proxy already routes to LiteLLM via OPENAI_BASE_URL.
    return _embeddings.embed_query(text)


def backfill_memories(uid: str, limit: int = None, dry_run: bool = False) -> Dict[str, int]:
    stats = {'total': 0, 'written': 0, 'skipped': 0, 'errors': 0}
    buf: List[Dict] = []
    for memory in _iter_user_collection(uid, MEMORIES_COLLECTION, limit=limit):
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
    for ai in _iter_user_collection(uid, ACTION_ITEMS_COLLECTION, limit=limit):
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
    for conv in _iter_user_collection(uid, CONVERSATIONS_COLLECTION, limit=limit):
        stats['total'] += 1
        text = _conversation_text(conv)
        if not text:
            stats['skipped'] += 1
            continue
        try:
            if dry_run:
                stats['written'] += 1
            else:
                vec = _embed_text(text)
                metadata = {}
                structured = conv.get('structured') or {}
                for key in ('topics', 'entities', 'people', 'dates'):
                    val = structured.get(key)
                    if val:
                        metadata[key] = val
                vector_db.upsert_vector2(uid, conv['id'], vec, metadata)
                stats['written'] += 1
        except Exception as e:
            stats['errors'] += 1
            logger.error(f'conversation embed/upsert failed cid={conv["id"]}: {sanitize(str(e))}')
        if stats['total'] % PROGRESS_EVERY == 0:
            logger.info(f'conversations progress uid={sanitize_pii(uid)} processed={stats["total"]}')
            # Throttle a touch so we don't pin LiteLLM at 100% during big backfills.
            time.sleep(0.1)
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
