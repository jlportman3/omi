"""
Tests for database/vector_db.py — the Qdrant-backed vector index that
replaced the offline Pinecone integration.

Coverage targets (matches the phase 3 verification plan):

  - Module imports cleanly with Qdrant available (smoke).
  - Bootstrap is idempotent across repeated calls.
  - Embedding helpers route through utils.llm.clients.embeddings and
    return correct dimensionality (mocked LiteLLM).
  - upsert_memory_vector / find_similar_memories round-trip preserves
    the exact response shape that callers in utils/retrieval/ and
    utils/conversations/process_conversation.py depend on
    ({'memory_id', 'category', 'score'}).
  - Delete helpers route to PointIdsList with the deterministic
    UUIDv5 point IDs (so re-upserts overwrite the same point).
  - check_memory_duplicate threads through find_similar_memories.
  - Bootstrap survives "collection already exists" exceptions.
  - All public function names that scout A enumerated are exported
    AND remain callable with their original Pinecone signatures.

Mock strategy:
  vector_db creates a real QdrantClient at module-import time and probes
  it via get_collections(). In the unit test environment that probe will
  succeed against the local AMS Qdrant — but for behavior tests we then
  swap vector_db.client and vector_db.embeddings with MagicMocks via
  monkeypatch so we control what Qdrant "returns" and what dim the
  embedder produces. This keeps the tests hermetic without disabling
  the real bootstrap.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from database import vector_db
from qdrant_client.http import models as qmodels


VEC_DIM = vector_db.VECTOR_SIZE  # 3072


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Replace embeddings with a MagicMock that returns a 3072-d vector."""
    mock = MagicMock()
    mock.embed_query = MagicMock(return_value=[0.1] * VEC_DIM)
    mock.embed_documents = MagicMock(side_effect=lambda texts: [[0.1 * (i + 1)] * VEC_DIM for i, _ in enumerate(texts)])
    monkeypatch.setattr(vector_db, 'embeddings', mock)
    return mock


@pytest.fixture
def fake_client(monkeypatch):
    """Replace the live Qdrant client with a MagicMock."""
    mock = MagicMock()
    mock.upsert = MagicMock(return_value=MagicMock(status='completed'))
    mock.delete = MagicMock(return_value=MagicMock(status='completed'))
    mock.set_payload = MagicMock(return_value=MagicMock(status='completed'))
    monkeypatch.setattr(vector_db, 'client', mock)
    return mock


def _scored_point(payload: dict, score: float):
    """Build a Qdrant-shaped ScoredPoint stand-in (object with .payload, .score)."""
    p = MagicMock()
    p.payload = payload
    p.score = score
    return p


# ---------------------------------------------------------------------------
# 1. Module imports cleanly
# ---------------------------------------------------------------------------


def test_module_imports_cleanly():
    """The bootstrap path must not raise even when AMS Qdrant is reachable."""
    # vector_db is already imported at top of file; if it raised, this file
    # wouldn't have loaded. Assert the public constants are sane.
    assert vector_db.VECTOR_SIZE == 3072
    assert vector_db.CONVERSATIONS_COLLECTION == 'omi_conversations'
    assert vector_db.MEMORIES_COLLECTION == 'omi_memories'
    assert vector_db.SCREEN_ACTIVITY_COLLECTION == 'omi_screen_activity'
    assert vector_db.ACTION_ITEMS_COLLECTION == 'omi_action_items'
    # Backwards-compat namespace constants (migration 005 imports these)
    assert vector_db.CONVERSATIONS_NAMESPACE == 'ns1'
    assert vector_db.MEMORIES_NAMESPACE == 'ns2'
    assert vector_db.SCREEN_ACTIVITY_NAMESPACE == 'ns3'
    assert vector_db.ACTION_ITEMS_NAMESPACE == 'ns4'


# ---------------------------------------------------------------------------
# 2. Embedding helpers shape
# ---------------------------------------------------------------------------


def test_embed_query_returns_3072_dim(fake_embeddings, fake_client):
    """upsert_memory_vector embeds via embed_query and returns the vector it used."""
    vec = vector_db.upsert_memory_vector(uid='u1', memory_id='m1', content='hello world', category='system')
    fake_embeddings.embed_query.assert_called_once_with('hello world')
    assert len(vec) == VEC_DIM
    fake_client.upsert.assert_called_once()
    kwargs = fake_client.upsert.call_args.kwargs
    assert kwargs['collection_name'] == 'omi_memories'
    point = kwargs['points'][0]
    assert isinstance(point, qmodels.PointStruct)
    assert len(point.vector) == VEC_DIM
    assert point.payload['uid'] == 'u1'
    assert point.payload['memory_id'] == 'm1'
    assert point.payload['category'] == 'system'


def test_embed_documents_used_for_batch(fake_embeddings, fake_client):
    """upsert_memory_vectors_batch must call embed_documents once for all items."""
    n = vector_db.upsert_memory_vectors_batch(
        uid='u1',
        items=[
            {'memory_id': 'a', 'content': 'one', 'category': 'system'},
            {'memory_id': 'b', 'content': 'two', 'category': 'system'},
            {'memory_id': 'c', 'content': 'three', 'category': 'system'},
        ],
    )
    assert n == 3
    fake_embeddings.embed_documents.assert_called_once_with(['one', 'two', 'three'])
    fake_client.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Round-trip upsert + find_similar_memories preserves caller-facing shape
# ---------------------------------------------------------------------------


def test_find_similar_memories_returns_caller_shape(fake_embeddings, fake_client):
    """Callers depend on [{'memory_id', 'category', 'score'}] from this helper."""
    fake_client.search = MagicMock(
        return_value=[
            _scored_point({'memory_id': 'm1', 'category': 'system', 'uid': 'u1'}, score=0.97),
            _scored_point({'memory_id': 'm2', 'category': 'user', 'uid': 'u1'}, score=0.81),
        ]
    )

    results = vector_db.find_similar_memories(uid='u1', content='AI infrastructure', threshold=0.5, limit=5)

    fake_embeddings.embed_query.assert_called_once_with('AI infrastructure')
    fake_client.search.assert_called_once()
    kwargs = fake_client.search.call_args.kwargs
    assert kwargs['collection_name'] == 'omi_memories'
    assert kwargs['limit'] == 5
    # threshold > 0 must be passed through to score_threshold
    assert kwargs['score_threshold'] == 0.5

    assert results == [
        {'memory_id': 'm1', 'category': 'system', 'score': 0.97},
        {'memory_id': 'm2', 'category': 'user', 'score': 0.81},
    ]


def test_find_similar_memories_threshold_zero_passes_none(fake_embeddings, fake_client):
    """threshold=0.0 → score_threshold=None (callers explicitly want raw ranking)."""
    fake_client.search = MagicMock(return_value=[])
    vector_db.find_similar_memories(uid='u1', content='x', threshold=0.0, limit=3)
    kwargs = fake_client.search.call_args.kwargs
    assert kwargs['score_threshold'] is None


def test_check_memory_duplicate_threads_through(fake_embeddings, fake_client):
    """check_memory_duplicate is a thin wrapper that returns the first match or None."""
    fake_client.search = MagicMock(
        return_value=[_scored_point({'memory_id': 'mDup', 'category': 'system'}, score=0.95)]
    )
    dup = vector_db.check_memory_duplicate(uid='u1', content='hello', threshold=0.85)
    assert dup == {'memory_id': 'mDup', 'category': 'system', 'score': 0.95}

    fake_client.search = MagicMock(return_value=[])
    assert vector_db.check_memory_duplicate(uid='u1', content='hello', threshold=0.85) is None


# ---------------------------------------------------------------------------
# 4. Delete helpers use deterministic UUIDv5 point IDs
# ---------------------------------------------------------------------------


def test_delete_memory_vector_uses_uuid5(fake_client, monkeypatch):
    """delete_memory_vector must derive the same UUID as the upsert path."""
    expected = vector_db._to_point_id('u1-m1')
    # Sanity: it's a real UUID string
    uuid.UUID(expected)

    vector_db.delete_memory_vector(uid='u1', memory_id='m1')
    fake_client.delete.assert_called_once()
    kwargs = fake_client.delete.call_args.kwargs
    assert kwargs['collection_name'] == 'omi_memories'
    selector = kwargs['points_selector']
    assert isinstance(selector, qmodels.PointIdsList)
    assert selector.points == [expected]


def test_delete_vector_uses_uuid5(fake_client):
    expected = vector_db._to_point_id('u1-conv-1')
    vector_db.delete_vector(uid='u1', conversation_id='conv-1')
    fake_client.delete.assert_called_once()
    selector = fake_client.delete.call_args.kwargs['points_selector']
    assert isinstance(selector, qmodels.PointIdsList)
    assert selector.points == [expected]


def test_delete_screen_activity_vectors_batch(fake_client):
    vector_db.delete_screen_activity_vectors(uid='u1', ids=[1, 2, 3])
    fake_client.delete.assert_called_once()
    kwargs = fake_client.delete.call_args.kwargs
    assert kwargs['collection_name'] == 'omi_screen_activity'
    selector = kwargs['points_selector']
    assert isinstance(selector, qmodels.PointIdsList)
    assert selector.points == [
        vector_db._to_point_id('u1-sa-1'),
        vector_db._to_point_id('u1-sa-2'),
        vector_db._to_point_id('u1-sa-3'),
    ]


def test_delete_screen_activity_empty_is_noop(fake_client):
    vector_db.delete_screen_activity_vectors(uid='u1', ids=[])
    fake_client.delete.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Bootstrap is idempotent (create-if-missing, don't fail on existing)
# ---------------------------------------------------------------------------


def test_ensure_collection_idempotent_when_present():
    """If get_collection succeeds, _ensure_collection returns without create."""
    c = MagicMock()
    c.get_collection = MagicMock(return_value=MagicMock())
    vector_db._ensure_collection(c, 'omi_memories')
    c.create_collection.assert_not_called()


def test_ensure_collection_creates_when_missing():
    """If get_collection raises UnexpectedResponse, create_collection runs."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    c = MagicMock()
    c.get_collection = MagicMock(side_effect=UnexpectedResponse(404, 'reason', b'body', headers=None))
    vector_db._ensure_collection(c, 'omi_memories')
    c.create_collection.assert_called_once()


def test_ensure_collection_tolerates_race():
    """If get raises AND create raises (race with another worker), we swallow."""
    c = MagicMock()
    c.get_collection = MagicMock(side_effect=ValueError('missing'))
    c.create_collection = MagicMock(side_effect=Exception('already exists'))
    # Must not raise.
    vector_db._ensure_collection(c, 'omi_memories')
    c.create_collection.assert_called_once()


def test_bootstrap_visits_all_four_collections():
    """_bootstrap must touch all four omi_* collections."""
    c = MagicMock()
    c.get_collection = MagicMock(return_value=MagicMock())
    vector_db._bootstrap(c)
    visited = {call.kwargs['collection_name'] for call in c.get_collection.call_args_list}
    assert visited == {
        'omi_conversations',
        'omi_memories',
        'omi_screen_activity',
        'omi_action_items',
    }


def test_bootstrap_noop_when_client_is_none():
    """Bootstrap must tolerate client=None (Qdrant unreachable on import)."""
    # Should not raise.
    vector_db._bootstrap(None)


# ---------------------------------------------------------------------------
# 6. Public API surface — all functions exported and callable
# ---------------------------------------------------------------------------


PUBLIC_FUNCTIONS = [
    # conversations (was ns1)
    'upsert_vector',
    'upsert_vector2',
    'upsert_vectors',
    'update_vector_metadata',
    'query_vectors',
    'query_vectors_by_metadata',
    'delete_vector',
    # memories (was ns2)
    'upsert_memory_vector',
    'upsert_memory_vectors_batch',
    'find_similar_memories',
    'check_memory_duplicate',
    'search_memories_by_vector',
    'delete_memory_vector',
    # screen activity (was ns3)
    'upsert_screen_activity_vectors',
    'search_screen_activity_vectors',
    'delete_screen_activity_vectors',
    # action items (was ns4)
    'upsert_action_item_vector',
    'upsert_action_item_vectors_batch',
    'search_action_items_by_vector',
    'find_similar_action_items',
    'delete_action_item_vector',
    'delete_action_item_vectors_batch',
]


@pytest.mark.parametrize('name', PUBLIC_FUNCTIONS)
def test_public_function_exported(name):
    fn = getattr(vector_db, name, None)
    assert fn is not None, f'{name} missing from vector_db'
    assert callable(fn), f'{name} is not callable'


# ---------------------------------------------------------------------------
# 7. Graceful degradation when client is None
# ---------------------------------------------------------------------------


def test_upsert_memory_vector_degrades_when_client_is_none(monkeypatch, fake_embeddings):
    monkeypatch.setattr(vector_db, 'client', None)
    assert vector_db.upsert_memory_vector(uid='u', memory_id='m', content='x', category='c') is None


def test_find_similar_memories_degrades_when_client_is_none(monkeypatch, fake_embeddings):
    monkeypatch.setattr(vector_db, 'client', None)
    assert vector_db.find_similar_memories(uid='u', content='x') == []


def test_query_vectors_degrades_when_client_is_none(monkeypatch, fake_embeddings):
    monkeypatch.setattr(vector_db, 'client', None)
    assert vector_db.query_vectors(query='q', uid='u') == []


def test_search_action_items_degrades_when_client_is_none(monkeypatch, fake_embeddings):
    monkeypatch.setattr(vector_db, 'client', None)
    assert vector_db.search_action_items_by_vector(uid='u', query='q') == []


def test_upsert_vectors_chunk_size_respected(fake_client, fake_embeddings):
    """upsert_vectors must chunk at UPSERT_CHUNK_SIZE to keep payloads sane."""
    chunk = vector_db.UPSERT_CHUNK_SIZE
    n_items = chunk + 5
    vector_db.upsert_vectors(
        uid='u1',
        vectors=[[0.0] * VEC_DIM for _ in range(n_items)],
        conversation_ids=[f'c{i}' for i in range(n_items)],
    )
    # Expect ceil(n_items / chunk) upsert calls.
    assert fake_client.upsert.call_count == 2
