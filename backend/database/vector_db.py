# =============================================================================
# vector_db.py — Qdrant-backed vector index for omi
# =============================================================================
#
# Migration history:
#   - Originally Pinecone (single index, 4 namespaces ns1..ns4) — went offline,
#     leaving every wrapper degrading to "index not initialized" no-ops and
#     killing RAG / memory dedup / action item search / screen activity recall.
#   - Now Qdrant on the local AMS instance (http://localhost:6333) with one
#     collection per index type. Per-user scoping moves from Pinecone metadata
#     filters to Qdrant payload filters on the `uid` field.
#
# Collections (vector_size=3072, Cosine — text-embedding-3-large via LiteLLM):
#   - omi_conversations      (was ns1)
#   - omi_memories           (was ns2)
#   - omi_screen_activity    (was ns3)
#   - omi_action_items       (was ns4)
#
# Public API surface is preserved verbatim — every function below has the
# SAME signature and SAME return shape as the Pinecone implementation. Callers
# in routers/ and utils/ are not updated; they don't need to be.
#
# ID conventions (unchanged; used as Qdrant point IDs via _to_point_id):
#   - conversation: '{uid}-{conversation_id}'
#   - memory:       '{uid}-{memory_id}'
#   - screen:       '{uid}-sa-{screenshot_id}'
#   - action item:  '{uid}-ai-{action_item_id}'
#   Qdrant point IDs are UUIDs derived deterministically from these strings,
#   so re-upserts overwrite the same point and deletes are O(1).
#
# Graceful degradation:
#   If Qdrant is unreachable at import time, `client` stays None and every
#   public function early-returns its empty / None equivalent and logs a
#   warning. This mirrors the previous "index is None" Pinecone behavior so
#   dev / test environments without Qdrant don't crash.
# =============================================================================

import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from utils.llm.clients import embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '') or None
QDRANT_TIMEOUT = float(os.getenv('QDRANT_TIMEOUT', '10'))

VECTOR_SIZE = 3072  # text-embedding-3-large
DISTANCE = qmodels.Distance.COSINE

# Collection names (also act as the public "namespace" identifiers).
CONVERSATIONS_COLLECTION = 'omi_conversations'
MEMORIES_COLLECTION = 'omi_memories'
SCREEN_ACTIVITY_COLLECTION = 'omi_screen_activity'
ACTION_ITEMS_COLLECTION = 'omi_action_items'

# Backwards-compatible constants — callers (and migration 005) import these
# by name. Pinecone-era values "ns1".."ns4" are preserved as string aliases
# so any test that asserts on them stays green. They are NOT used as Qdrant
# collection names; the *_COLLECTION constants are.
CONVERSATIONS_NAMESPACE = "ns1"
MEMORIES_NAMESPACE = "ns2"
SCREEN_ACTIVITY_NAMESPACE = "ns3"
ACTION_ITEMS_NAMESPACE = "ns4"

# Pinecone single-upsert limit was 100. Qdrant tolerates much larger batches,
# but we chunk at 256 to keep payload sizes sane for 3072-d vectors.
UPSERT_CHUNK_SIZE = 256


# ---------------------------------------------------------------------------
# Client init + idempotent collection bootstrap
# ---------------------------------------------------------------------------


def _build_client() -> Optional[QdrantClient]:
    try:
        c = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
        # cheap reachability probe
        c.get_collections()
        return c
    except Exception as e:
        logger.warning(f'Qdrant unreachable at {QDRANT_URL}: {type(e).__name__}')
        return None


def _ensure_collection(c: QdrantClient, name: str) -> None:
    """Create the collection if it doesn't exist. Idempotent."""
    try:
        c.get_collection(collection_name=name)
        return
    except (UnexpectedResponse, ValueError):
        pass
    except Exception as e:
        # Older qdrant-client versions raise different exception types when a
        # collection is missing. Treat any "get_collection failed" as "needs
        # create" and let recreate_collection's own error bubble up if real.
        logger.info(f'_ensure_collection {name}: get_collection raised {type(e).__name__}, will attempt create')

    try:
        c.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )
        logger.info(f'Qdrant: created collection {name} (size={VECTOR_SIZE}, distance={DISTANCE})')
    except Exception as e:
        # Race: another worker created it between the get and create. Tolerate.
        logger.info(f'_ensure_collection {name}: create raised {type(e).__name__} (likely already exists)')


def _bootstrap(c: Optional[QdrantClient]) -> None:
    if c is None:
        return
    for name in (
        CONVERSATIONS_COLLECTION,
        MEMORIES_COLLECTION,
        SCREEN_ACTIVITY_COLLECTION,
        ACTION_ITEMS_COLLECTION,
    ):
        _ensure_collection(c, name)
    # Indexed payload fields make uid / created_at / timestamp filters fast.
    _ensure_payload_indexes(c)


def _ensure_payload_indexes(c: QdrantClient) -> None:
    """Create payload indexes the filters depend on. Idempotent."""
    plans = [
        (CONVERSATIONS_COLLECTION, 'uid', qmodels.PayloadSchemaType.KEYWORD),
        (CONVERSATIONS_COLLECTION, 'created_at', qmodels.PayloadSchemaType.INTEGER),
        (MEMORIES_COLLECTION, 'uid', qmodels.PayloadSchemaType.KEYWORD),
        (MEMORIES_COLLECTION, 'created_at', qmodels.PayloadSchemaType.INTEGER),
        (SCREEN_ACTIVITY_COLLECTION, 'uid', qmodels.PayloadSchemaType.KEYWORD),
        (SCREEN_ACTIVITY_COLLECTION, 'timestamp', qmodels.PayloadSchemaType.INTEGER),
        (SCREEN_ACTIVITY_COLLECTION, 'appName', qmodels.PayloadSchemaType.KEYWORD),
        (ACTION_ITEMS_COLLECTION, 'uid', qmodels.PayloadSchemaType.KEYWORD),
        (ACTION_ITEMS_COLLECTION, 'created_at', qmodels.PayloadSchemaType.INTEGER),
    ]
    for collection, field, schema in plans:
        try:
            c.create_payload_index(collection_name=collection, field_name=field, field_schema=schema)
        except Exception:
            # Already exists, or transient error — non-fatal.
            pass


client: Optional[QdrantClient] = _build_client()
_bootstrap(client)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Stable namespace UUID for omi point-ID derivation. Any UUID works as long as
# it never changes; we use one derived from the string "omi-vector-db" so it's
# reproducible from source.
_POINT_ID_NS = uuid.uuid5(uuid.NAMESPACE_DNS, 'omi-vector-db')


def _to_point_id(raw_id: str) -> str:
    """Deterministically map a Pinecone-style string ID to a Qdrant UUID."""
    return str(uuid.uuid5(_POINT_ID_NS, raw_id))


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _uid_filter(uid: str) -> qmodels.Filter:
    return qmodels.Filter(must=[qmodels.FieldCondition(key='uid', match=qmodels.MatchValue(value=uid))])


def _range_condition(field: str, gte: Optional[int] = None, lte: Optional[int] = None) -> qmodels.FieldCondition:
    return qmodels.FieldCondition(key=field, range=qmodels.Range(gte=gte, lte=lte))


def _exact_condition(field: str, value) -> qmodels.FieldCondition:
    return qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value))


def _any_condition(field: str, values: List[str]) -> qmodels.FieldCondition:
    return qmodels.FieldCondition(key=field, match=qmodels.MatchAny(any=list(values)))


# ---------------------------------------------------------------------------
# Conversation Vector Functions (was ns1)
# ---------------------------------------------------------------------------


def _conv_payload(uid: str, conversation_id: str, extra: Optional[dict] = None) -> dict:
    payload = {
        'uid': uid,
        'memory_id': conversation_id,
        'created_at': _now_ts(),
    }
    if extra:
        payload.update(extra)
    return payload


def upsert_vector(uid: str, conversation_id: str, vector: List[float]):
    if client is None:
        logger.warning('Qdrant client not initialized, skipping upsert_vector')
        return
    raw_id = f'{uid}-{conversation_id}'
    point = qmodels.PointStruct(id=_to_point_id(raw_id), vector=vector, payload=_conv_payload(uid, conversation_id))
    res = client.upsert(collection_name=CONVERSATIONS_COLLECTION, points=[point], wait=True)
    logger.info(f'upsert_vector {res}')


def upsert_vector2(uid: str, conversation_id: str, vector: List[float], metadata: dict):
    if client is None:
        logger.warning('Qdrant client not initialized, skipping upsert_vector2')
        return
    raw_id = f'{uid}-{conversation_id}'
    point = qmodels.PointStruct(
        id=_to_point_id(raw_id),
        vector=vector,
        payload=_conv_payload(uid, conversation_id, extra=metadata),
    )
    res = client.upsert(collection_name=CONVERSATIONS_COLLECTION, points=[point], wait=True)
    logger.info(f'upsert_vector {res}')


def update_vector_metadata(uid: str, conversation_id: str, metadata: dict):
    """
    Update only the metadata of an existing conversation vector.

    Pinecone's index.update() let you set_metadata without touching values.
    Qdrant's equivalent is set_payload, which merges into the existing payload.
    We also force uid/memory_id to stay correct.
    """
    if client is None:
        logger.warning('Qdrant client not initialized, skipping update_vector_metadata')
        return {}
    payload = dict(metadata)
    payload['uid'] = uid
    payload['memory_id'] = conversation_id
    raw_id = f'{uid}-{conversation_id}'
    res = client.set_payload(
        collection_name=CONVERSATIONS_COLLECTION,
        payload=payload,
        points=[_to_point_id(raw_id)],
        wait=True,
    )
    return res


def upsert_vectors(uid: str, vectors: List[List[float]], conversation_ids: List[str]):
    if client is None:
        logger.warning('Qdrant client not initialized, skipping upsert_vectors')
        return
    points = []
    for cid, vec in zip(conversation_ids, vectors):
        raw_id = f'{uid}-{cid}'
        points.append(qmodels.PointStruct(id=_to_point_id(raw_id), vector=vec, payload=_conv_payload(uid, cid)))
    for i in range(0, len(points), UPSERT_CHUNK_SIZE):
        res = client.upsert(
            collection_name=CONVERSATIONS_COLLECTION, points=points[i : i + UPSERT_CHUNK_SIZE], wait=True
        )
        logger.info(f'upsert_vectors {res}')


def query_vectors(query: str, uid: str, starts_at: int = None, ends_at: int = None, k: int = 5) -> List[str]:
    if client is None:
        logger.warning('Qdrant client not initialized, skipping query_vectors')
        return []

    must = [_exact_condition('uid', uid)]
    if starts_at is not None:
        must.append(_range_condition('created_at', gte=starts_at, lte=ends_at))
    filt = qmodels.Filter(must=must)

    xq = embeddings.embed_query(query)
    hits = client.search(
        collection_name=CONVERSATIONS_COLLECTION,
        query_vector=xq,
        query_filter=filt,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    out = []
    for h in hits:
        payload = h.payload or {}
        conv_id = payload.get('memory_id')
        if conv_id:
            out.append(conv_id)
    return out


def query_vectors_by_metadata(
    uid: str,
    vector: List[float],
    dates_filter: List[datetime],
    people: List[str],
    topics: List[str],
    entities: List[str],
    dates: List[str],
    limit: int = 5,
):
    """
    Hybrid vector + structured-metadata search of the conversations collection.

    Mirrors the Pinecone behavior: try with people/topics/entities $or filter
    first, retry with vector-only if empty, then rank by structured-field
    overlap count and truncate to `limit`.
    """
    if client is None:
        logger.warning('Qdrant client not initialized, skipping query_vectors_by_metadata')
        return []

    must = [_exact_condition('uid', uid)]
    structured_should = []
    if people:
        structured_should.append(_any_condition('people', people))
    if topics:
        structured_should.append(_any_condition('topics', topics))
    if entities:
        structured_should.append(_any_condition('entities', entities))
    # NOTE: dates is collected for parity with the original signature but not
    # filtered on (matches Pinecone path which also commented out dates).

    used_structured = bool(structured_should)
    if used_structured:
        must.append(qmodels.Filter(should=structured_should))

    if dates_filter and len(dates_filter) == 2 and dates_filter[0] and dates_filter[1]:
        logger.info(f'dates_filter {dates_filter}')
        must.append(
            _range_condition(
                'created_at',
                gte=int(dates_filter[0].timestamp()),
                lte=int(dates_filter[1].timestamp()),
            )
        )

    filt = qmodels.Filter(must=must)
    hits = client.search(
        collection_name=CONVERSATIONS_COLLECTION,
        query_vector=vector,
        query_filter=filt,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    if not hits:
        # Mirror Pinecone retry: drop the structured-fields filter and re-query.
        if used_structured:
            must_retry = [c for c in must if not isinstance(c, qmodels.Filter)]
            logger.warning(
                f'query_vectors_by_metadata retrying without structured filters: '
                f'{json.dumps({"uid": uid, "limit": 20})}'
            )
            hits = client.search(
                collection_name=CONVERSATIONS_COLLECTION,
                query_vector=vector,
                query_filter=qmodels.Filter(must=must_retry),
                limit=20,
                with_payload=True,
                with_vectors=False,
            )
            if not hits:
                return []
        else:
            return []

    conversation_id_to_matches = defaultdict(int)
    conversation_ids: List[str] = []
    for hit in hits:
        payload = hit.payload or {}
        conv_id = payload.get('memory_id')
        if not conv_id:
            continue
        conversation_ids.append(conv_id)
        for topic in topics:
            if topic in payload.get('topics', []):
                conversation_id_to_matches[conv_id] += 1
        for entity in entities:
            if entity in payload.get('entities', []):
                conversation_id_to_matches[conv_id] += 1
        for person in people:
            if person in payload.get('people_mentioned', []):
                conversation_id_to_matches[conv_id] += 1

    conversation_ids.sort(key=lambda x: conversation_id_to_matches[x], reverse=True)
    return conversation_ids[:limit] if len(conversation_ids) > limit else conversation_ids


def delete_vector(uid: str, conversation_id: str):
    """Delete a conversation vector by composite '{uid}-{conversation_id}'."""
    if client is None:
        logger.warning('Qdrant client not initialized, skipping delete_vector')
        return
    raw_id = f'{uid}-{conversation_id}'
    result = client.delete(
        collection_name=CONVERSATIONS_COLLECTION,
        points_selector=qmodels.PointIdsList(points=[_to_point_id(raw_id)]),
        wait=True,
    )
    logger.info(f'delete_vector {raw_id} {result}')


# ---------------------------------------------------------------------------
# Memory Vector Functions (was ns2)
# ---------------------------------------------------------------------------


def upsert_memory_vector(uid: str, memory_id: str, content: str, category: str):
    if client is None:
        logger.warning('Qdrant client not initialized, skipping memory vector upsert')
        return None

    vector = embeddings.embed_query(content)
    payload = {
        'uid': uid,
        'memory_id': memory_id,
        'category': category,
        'created_at': _now_ts(),
    }
    raw_id = f'{uid}-{memory_id}'
    point = qmodels.PointStruct(id=_to_point_id(raw_id), vector=vector, payload=payload)
    res = client.upsert(collection_name=MEMORIES_COLLECTION, points=[point], wait=True)
    logger.info(f'upsert_memory_vector {memory_id} {res}')
    return vector


def upsert_memory_vectors_batch(uid: str, items: List[dict]) -> int:
    if client is None:
        logger.warning('Qdrant client not initialized, skipping memory vector batch upsert')
        return 0
    if not items:
        return 0

    contents = [item['content'] for item in items]
    vectors = embeddings.embed_documents(contents)

    now_ts = _now_ts()
    points = []
    for i, item in enumerate(items):
        raw_id = f"{uid}-{item['memory_id']}"
        points.append(
            qmodels.PointStruct(
                id=_to_point_id(raw_id),
                vector=vectors[i],
                payload={
                    'uid': uid,
                    'memory_id': item['memory_id'],
                    'category': item['category'],
                    'created_at': now_ts,
                },
            )
        )

    upserted = 0
    for i in range(0, len(points), UPSERT_CHUNK_SIZE):
        chunk = points[i : i + UPSERT_CHUNK_SIZE]
        res = client.upsert(collection_name=MEMORIES_COLLECTION, points=chunk, wait=True)
        upserted += len(chunk)
        logger.info(f'upsert_memory_vectors_batch chunk={len(chunk)} {res}')

    logger.info(f'upsert_memory_vectors_batch total={upserted}')
    return upserted


def find_similar_memories(uid: str, content: str, threshold: float = 0.85, limit: int = 5) -> List[dict]:
    if client is None:
        logger.warning('Qdrant client not initialized, skipping similarity search')
        return []

    vector = embeddings.embed_query(content)
    # Qdrant supports score_threshold natively. We pass it through except when
    # callers explicitly want threshold=0.0 (pure ranked retrieval).
    hits = client.search(
        collection_name=MEMORIES_COLLECTION,
        query_vector=vector,
        query_filter=_uid_filter(uid),
        limit=limit,
        with_payload=True,
        with_vectors=False,
        score_threshold=threshold if threshold > 0 else None,
    )

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            {
                'memory_id': payload.get('memory_id'),
                'category': payload.get('category'),
                'score': hit.score,
            }
        )
    return results


def check_memory_duplicate(uid: str, content: str, threshold: float = 0.85) -> Optional[dict]:
    similar = find_similar_memories(uid, content, threshold=threshold, limit=1)
    if similar:
        logger.warning(f'Found duplicate memory: {similar[0]}')
        return similar[0]
    return None


def search_memories_by_vector(uid: str, query: str, limit: int = 10) -> List[str]:
    if client is None:
        logger.warning('Qdrant client not initialized, skipping memory search')
        return []

    vector = embeddings.embed_query(query)
    hits = client.search(
        collection_name=MEMORIES_COLLECTION,
        query_vector=vector,
        query_filter=_uid_filter(uid),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [(h.payload or {}).get('memory_id') for h in hits]


def delete_memory_vector(uid: str, memory_id: str):
    if client is None:
        logger.warning('Qdrant client not initialized, skipping memory vector delete')
        return
    raw_id = f'{uid}-{memory_id}'
    result = client.delete(
        collection_name=MEMORIES_COLLECTION,
        points_selector=qmodels.PointIdsList(points=[_to_point_id(raw_id)]),
        wait=True,
    )
    logger.info(f'delete_memory_vector {raw_id} {result}')


# ---------------------------------------------------------------------------
# Screen Activity Vector Functions (was ns3)
# Pre-embedded by caller (Gemini embedding-001, 3072-dim).
# ---------------------------------------------------------------------------


def upsert_screen_activity_vectors(uid: str, rows: List[dict]) -> int:
    if client is None:
        logger.warning('Qdrant client not initialized, skipping screen activity vector upsert')
        return 0

    points = []
    for row in rows:
        embedding = row.get('embedding')
        if not embedding:
            continue
        if isinstance(row['timestamp'], str):
            ts = int(datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')).timestamp())
        else:
            ts = int(row['timestamp'])
        raw_id = f'{uid}-sa-{row["id"]}'
        points.append(
            qmodels.PointStruct(
                id=_to_point_id(raw_id),
                vector=embedding,
                payload={
                    'uid': uid,
                    'screenshot_id': str(row['id']),
                    'timestamp': ts,
                    'appName': row.get('appName', ''),
                },
            )
        )

    if not points:
        return 0

    upserted = 0
    for i in range(0, len(points), UPSERT_CHUNK_SIZE):
        chunk = points[i : i + UPSERT_CHUNK_SIZE]
        client.upsert(collection_name=SCREEN_ACTIVITY_COLLECTION, points=chunk, wait=True)
        upserted += len(chunk)

    logger.info(f'upsert_screen_activity_vectors uid={uid} count={upserted}')
    return upserted


def search_screen_activity_vectors(
    uid: str,
    query_vector: List[float],
    start_date: int = None,
    end_date: int = None,
    app_filter: str = None,
    k: int = 10,
) -> List[dict]:
    if client is None:
        logger.warning('Qdrant client not initialized, skipping screen activity search')
        return []

    must = [_exact_condition('uid', uid)]
    if start_date or end_date:
        must.append(_range_condition('timestamp', gte=start_date, lte=end_date))
    if app_filter:
        must.append(_exact_condition('appName', app_filter))

    hits = client.search(
        collection_name=SCREEN_ACTIVITY_COLLECTION,
        query_vector=query_vector,
        query_filter=qmodels.Filter(must=must),
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            'screenshot_id': (h.payload or {}).get('screenshot_id'),
            'timestamp': (h.payload or {}).get('timestamp'),
            'appName': (h.payload or {}).get('appName'),
            'score': h.score,
        }
        for h in hits
    ]


def delete_screen_activity_vectors(uid: str, ids: List[int]):
    if client is None:
        return
    if not ids:
        return
    point_ids = [_to_point_id(f'{uid}-sa-{sid}') for sid in ids]
    client.delete(
        collection_name=SCREEN_ACTIVITY_COLLECTION,
        points_selector=qmodels.PointIdsList(points=point_ids),
        wait=True,
    )


# ---------------------------------------------------------------------------
# Action Item Vector Functions (was ns4)
# ---------------------------------------------------------------------------


def upsert_action_item_vector(uid: str, action_item_id: str, description: str):
    if client is None:
        logger.warning('Qdrant client not initialized, skipping action item vector upsert')
        return None

    vector = embeddings.embed_query(description)
    payload = {
        'uid': uid,
        'action_item_id': action_item_id,
        'created_at': _now_ts(),
    }
    raw_id = f'{uid}-ai-{action_item_id}'
    point = qmodels.PointStruct(id=_to_point_id(raw_id), vector=vector, payload=payload)
    res = client.upsert(collection_name=ACTION_ITEMS_COLLECTION, points=[point], wait=True)
    logger.info(f'upsert_action_item_vector {action_item_id} {res}')
    return vector


def upsert_action_item_vectors_batch(uid: str, items: List[dict]) -> int:
    if client is None:
        logger.warning('Qdrant client not initialized, skipping action item vector batch upsert')
        return 0
    if not items:
        return 0

    descriptions = [item['description'] for item in items]
    vectors = embeddings.embed_documents(descriptions)

    now_ts = _now_ts()
    points = []
    for i, item in enumerate(items):
        raw_id = f"{uid}-ai-{item['action_item_id']}"
        points.append(
            qmodels.PointStruct(
                id=_to_point_id(raw_id),
                vector=vectors[i],
                payload={
                    'uid': uid,
                    'action_item_id': item['action_item_id'],
                    'created_at': now_ts,
                },
            )
        )

    upserted = 0
    for i in range(0, len(points), UPSERT_CHUNK_SIZE):
        chunk = points[i : i + UPSERT_CHUNK_SIZE]
        res = client.upsert(collection_name=ACTION_ITEMS_COLLECTION, points=chunk, wait=True)
        upserted += len(chunk)
        logger.info(f'upsert_action_item_vectors_batch chunk={len(chunk)} {res}')
    return upserted


def search_action_items_by_vector(uid: str, query: str, limit: int = 10, min_score: float = 0.3) -> List[str]:
    if client is None:
        logger.warning('Qdrant client not initialized, skipping action item search')
        return []

    vector = embeddings.embed_query(query)
    hits = client.search(
        collection_name=ACTION_ITEMS_COLLECTION,
        query_vector=vector,
        query_filter=_uid_filter(uid),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    top_score = hits[0].score if hits else None
    kept = [h for h in hits if (h.score or 0.0) >= min_score]
    logger.info(
        f'search_action_items_by_vector uid={uid} matches={len(hits)} kept={len(kept)} '
        f'top_score={top_score} min_score={min_score}'
    )
    return [(h.payload or {}).get('action_item_id') for h in kept]


def find_similar_action_items(uid: str, query: str, threshold: float = 0.6, limit: int = 10) -> List[dict]:
    """
    Find action items semantically similar to query for dedup-during-extraction.
    Pinecone and Qdrant failures degrade silently to an empty list — the caller
    treats "no candidates" as "user has nothing relevant", which is the same
    behavior as a brand-new user.
    """
    if client is None:
        return []

    try:
        vector = embeddings.embed_query(query)
        hits = client.search(
            collection_name=ACTION_ITEMS_COLLECTION,
            query_vector=vector,
            query_filter=_uid_filter(uid),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        kept = []
        dropped_no_id = 0
        for h in hits:
            score = h.score or 0.0
            if score < threshold:
                continue
            aid = (h.payload or {}).get('action_item_id')
            if not aid:
                dropped_no_id += 1
                continue
            kept.append({'action_item_id': aid, 'score': score})
        top_score = hits[0].score if hits else None
        logger.info(
            f'find_similar_action_items uid={uid} matches={len(hits)} '
            f'kept={len(kept)} dropped_no_id={dropped_no_id} '
            f'top_score={top_score} threshold={threshold}'
        )
        return kept
    except Exception as e:
        logger.exception(f'find_similar_action_items failed uid={uid}: {e}')
        return []


def delete_action_item_vector(uid: str, action_item_id: str):
    if client is None:
        logger.warning('Qdrant client not initialized, skipping action item vector delete')
        return
    raw_id = f'{uid}-ai-{action_item_id}'
    result = client.delete(
        collection_name=ACTION_ITEMS_COLLECTION,
        points_selector=qmodels.PointIdsList(points=[_to_point_id(raw_id)]),
        wait=True,
    )
    logger.info(f'delete_action_item_vector {raw_id} {result}')


def delete_action_item_vectors_batch(uid: str, action_item_ids: List[str]):
    if client is None:
        return
    if not action_item_ids:
        return
    point_ids = [_to_point_id(f'{uid}-ai-{aid}') for aid in action_item_ids]
    client.delete(
        collection_name=ACTION_ITEMS_COLLECTION,
        points_selector=qmodels.PointIdsList(points=point_ids),
        wait=True,
    )
    logger.info(f'delete_action_item_vectors_batch count={len(point_ids)}')
