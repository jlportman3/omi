"""Regression tests for the trends ``ExpectedOutput`` Pydantic validation fix.

Background
----------
``utils/llm/trends.py`` defines a nested-list Pydantic schema::

    class Item(BaseModel):
        category: TrendEnum
        type: TrendType
        topic: str

    class ExpectedOutput(BaseModel):
        items: List[Item]

Before the fix, ``QwenChatOpenAI.with_structured_output(ExpectedOutput)``
sent Qwen a schema block that only showed the top-level ``items`` field
as ``"List[any]"`` — Qwen had no information about the inner ``Item``
shape and guessed Title-Case keys (``Category``/``Type``/``Topic``)
based on the prompt's English example. Pydantic then rejected every
item with::

    items.0.category  Field required [type=missing, ...]
    items.0.type      Field required [type=missing, ...]
    items.0.topic     Field required [type=missing, ...]

The error was swallowed by trends_extractor's broad ``except Exception``
and logged as ``"Error determining memory discard: ..."``, so every
conversation processed under MODEL_QOS=local_only silently produced zero
trends.

Fix (utils/llm/qwen_structured.py)
----------------------------------
1. ``_schema_description`` now recurses through ``$ref``/``$defs`` and
   emits one line per nested field path (``items[*].category``,
   ``items[*].type``, ``items[*].topic``) with enum value lists.
2. ``_parse_qwen_structured_response`` retries with case-insensitive
   key normalization on ValidationError as defense-in-depth.

These tests exercise both layers without any network calls.
"""

import sys
from enum import Enum
from typing import List
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Pre-mock heavy deps before importing the LLM module under test.
# Mirrors test_chat_extraction.py to keep tests hermetic.
# ---------------------------------------------------------------------------
_HEAVY_MOCKS = {
    'firebase_admin': MagicMock(),
    'firebase_admin.firestore': MagicMock(),
    'google.cloud.firestore': MagicMock(),
    'google.cloud.firestore_v1': MagicMock(),
    'google.cloud.firestore_v1.base_query': MagicMock(),
    'database': MagicMock(),
    'database._client': MagicMock(),
    'database.llm_usage': MagicMock(),
}
for _mod, _mock in _HEAVY_MOCKS.items():
    sys.modules.setdefault(_mod, _mock)

import os  # noqa: E402

os.environ.setdefault('OPENAI_API_KEY', 'sk-test-fake-key-for-unit-tests')
os.environ.setdefault('ANTHROPIC_API_KEY', 'sk-ant-test-fake-key')
os.environ.setdefault('ENCRYPTION_SECRET', 'test-secret-' + 'x' * 50)

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from utils.llm.qwen_structured import (  # noqa: E402
    QwenChatOpenAI,
    _all_canonical_keys,
    _augment_prompt_with_schema,
    _build_key_map,
    _normalize_keys,
    _parse_qwen_structured_response,
    _schema_description,
)


# ---------------------------------------------------------------------------
# Local replicas of the trends models (avoid the trends.py import graph,
# which pulls in database.users -> Firestore client).
# ---------------------------------------------------------------------------
class TrendEnum(str, Enum):
    ceo = "ceo"
    company = "company"
    software_product = "software_product"
    hardware_product = "hardware_product"
    ai_product = "ai_product"


class TrendType(str, Enum):
    best = "best"
    worst = "worst"


class Item(BaseModel):
    category: TrendEnum = Field(description="The category identified")
    type: TrendType = Field(description="The sentiment identified")
    topic: str = Field(description="The specific topic corresponding the category")


class ExpectedOutput(BaseModel):
    items: List[Item] = Field(default=[], description="List of items.")


# Raw Qwen outputs captured during diagnosis. The Title-Case shape is the
# one that broke production memory extraction under MODEL_QOS=local_only.
_QWEN_TITLE_CASE_ITEMS = (
    '{\n'
    '  "items": [\n'
    '    {"Category": "company", "Type": "best", "Topic": "Tesla"},\n'
    '    {"Category": "ceo", "Type": "best", "Topic": "Elon Musk"},\n'
    '    {"Category": "ai_product", "Type": "best", "Topic": "OpenAI GPT-5"},\n'
    '    {"Category": "software_product", "Type": "worst", "Topic": "Microsoft Copilot"},\n'
    '    {"Category": "hardware_product", "Type": "worst", "Topic": "Apple Vision Pro"}\n'
    '  ]\n'
    '}'
)

_QWEN_LOWERCASE_ITEMS = (
    '{"items":['
    '{"category":"company","type":"best","topic":"Tesla"},'
    '{"category":"ceo","type":"best","topic":"Elon Musk"}'
    ']}'
)

_QWEN_LOWERCASE_FENCED = (
    '\n```json\n' '{"items":[{"category":"hardware_product","type":"worst","topic":"Apple Vision Pro"}]}\n' '```\n'
)


class TestSchemaDescriptionRecursion:
    """The recursive schema description is THE fix — without it, Qwen guesses keys.

    These tests assert the exact strings sent to Qwen so future refactors
    can't silently regress to the broken ``List[any]`` form.
    """

    def test_emits_nested_paths_for_list_of_objects(self):
        block = _schema_description(ExpectedOutput)
        # Top-level field present
        assert '"items"' in block
        # And — critically — each inner field with the [*] path syntax
        assert '"items[*].category"' in block
        assert '"items[*].type"' in block
        assert '"items[*].topic"' in block

    def test_inner_object_label_is_list_object(self):
        # Outer "items" should be labeled List[object] (NOT List[any])
        # so Qwen knows each element is a structured object, not a scalar.
        block = _schema_description(ExpectedOutput)
        assert 'List[object]' in block
        assert 'List[any]' not in block

    def test_enum_values_inlined_for_enum_fields(self):
        # The enum value list must be in-context — Qwen produces invalid
        # outputs when it has to guess the allowed enum values.
        block = _schema_description(ExpectedOutput)
        assert 'enum(ceo|company|software_product|hardware_product|ai_product)' in block
        assert 'enum(best|worst)' in block

    def test_descriptions_attached_to_nested_fields(self):
        block = _schema_description(ExpectedOutput)
        assert 'The category identified' in block
        assert 'The sentiment identified' in block
        assert 'The specific topic corresponding the category' in block

    def test_flat_schema_still_works(self):
        # Don't break the existing chat_extraction path — flat schemas must
        # still emit a top-level field-per-line block.
        class Flat(BaseModel):
            people: List[str] = Field(default=[], description='Names of people.')
            topics: List[str] = Field(default=[], description='Conversation topics.')

        block = _schema_description(Flat)
        assert '"people"' in block
        assert '"topics"' in block
        assert 'List[str]' in block
        assert 'Names of people.' in block

    def test_handles_unparseable_schema(self):
        # Defensive: a class whose model_json_schema raises must not crash.
        class Broken(BaseModel):
            pass

        with patch.object(Broken, 'model_json_schema', side_effect=RuntimeError('boom')):
            assert _schema_description(Broken) == ''


class TestAugmentedPromptIncludesSchema:
    """The augmented prompt must show Qwen the recursive schema block."""

    def test_string_prompt_appends_nested_schema(self):
        out = _augment_prompt_with_schema('Find trends in: ...', ExpectedOutput)
        assert isinstance(out, str)
        assert 'items[*].category' in out
        assert 'enum(best|worst)' in out
        assert 'JSON' in out  # instruction block present


class TestKeyMapBuilding:
    """``_build_key_map`` indexes every nested model so normalization can repair Qwen output."""

    def test_indexes_top_level_and_nested_models(self):
        key_map = _build_key_map(ExpectedOutput)
        # Both ExpectedOutput and Item should be indexed
        assert 'ExpectedOutput' in key_map
        assert 'Item' in key_map
        assert key_map['Item'] == {'category': 'category', 'type': 'type', 'topic': 'topic'}

    def test_all_canonical_keys_flat_map(self):
        key_map = _build_key_map(ExpectedOutput)
        flat = _all_canonical_keys(key_map)
        assert flat['category'] == 'category'
        assert flat['type'] == 'type'
        assert flat['topic'] == 'topic'
        assert flat['items'] == 'items'


class TestKeyNormalization:
    """Case-insensitive key normalization is the safety net for nested-object Qwen quirks."""

    def test_normalizes_title_case_keys_in_nested_list(self):
        payload = {
            'items': [
                {'Category': 'company', 'Type': 'best', 'Topic': 'Tesla'},
                {'Category': 'ceo', 'Type': 'best', 'Topic': 'Elon Musk'},
            ]
        }
        canon = _all_canonical_keys(_build_key_map(ExpectedOutput))
        out = _normalize_keys(payload, canon)
        assert out['items'][0] == {'category': 'company', 'type': 'best', 'topic': 'Tesla'}
        assert out['items'][1]['topic'] == 'Elon Musk'

    def test_preserves_already_canonical_keys(self):
        payload = {'items': [{'category': 'company', 'type': 'best', 'topic': 'Tesla'}]}
        canon = _all_canonical_keys(_build_key_map(ExpectedOutput))
        assert _normalize_keys(payload, canon) == payload

    def test_passes_unknown_keys_through(self):
        payload = {'items': [{'category': 'company', 'type': 'best', 'topic': 'X', 'extra': 1}]}
        canon = _all_canonical_keys(_build_key_map(ExpectedOutput))
        out = _normalize_keys(payload, canon)
        assert out['items'][0]['extra'] == 1


class TestParseQwenStructuredResponseRecovers:
    """End-to-end: the parser must recover from Title-Case Qwen output."""

    def test_recovers_from_title_case_keys(self):
        # This exact shape was rejected with 15 'Field required' errors
        # before the fix. Now it must produce a valid ExpectedOutput.
        obj = _parse_qwen_structured_response(_QWEN_TITLE_CASE_ITEMS, ExpectedOutput)
        assert isinstance(obj, ExpectedOutput)
        assert len(obj.items) == 5
        topics = {item.topic for item in obj.items}
        assert 'Tesla' in topics
        assert 'Apple Vision Pro' in topics
        # Enum coercion preserved
        assert obj.items[0].category == TrendEnum.company
        assert obj.items[0].type == TrendType.best

    def test_parses_canonical_lowercase(self):
        obj = _parse_qwen_structured_response(_QWEN_LOWERCASE_ITEMS, ExpectedOutput)
        assert isinstance(obj, ExpectedOutput)
        assert len(obj.items) == 2
        assert obj.items[0].topic == 'Tesla'

    def test_parses_fenced_lowercase(self):
        obj = _parse_qwen_structured_response(_QWEN_LOWERCASE_FENCED, ExpectedOutput)
        assert isinstance(obj, ExpectedOutput)
        assert obj.items[0].category == TrendEnum.hardware_product
        assert obj.items[0].topic == 'Apple Vision Pro'


class TestExpectedOutputViaChain:
    """Hermetic end-to-end: mocked Qwen → ``.with_structured_output(ExpectedOutput).invoke()``.

    Mirrors test_chat_extraction.py::TestQwenChatOpenAIStructuredOutput so
    failures here are immediately recognisable as the same fix regressing.
    """

    def _make_llm(self) -> QwenChatOpenAI:
        return QwenChatOpenAI(model='qwen3.6-35b-a3b', api_key='sk-test-fake', base_url='http://10.0.60.48:4000/v1')

    def test_title_case_chain_returns_valid_expected_output(self):
        llm = self._make_llm()
        chain = llm.with_structured_output(ExpectedOutput)
        with patch.object(type(llm), 'invoke', return_value=AIMessage(content=_QWEN_TITLE_CASE_ITEMS)):
            out = chain.invoke('Extract trends from: ...')
        assert isinstance(out, ExpectedOutput)
        assert len(out.items) == 5
        assert out.items[2].topic == 'OpenAI GPT-5'

    def test_lowercase_chain_returns_valid_expected_output(self):
        llm = self._make_llm()
        chain = llm.with_structured_output(ExpectedOutput)
        with patch.object(type(llm), 'invoke', return_value=AIMessage(content=_QWEN_LOWERCASE_ITEMS)):
            out = chain.invoke('Extract trends from: ...')
        assert isinstance(out, ExpectedOutput)
        assert len(out.items) == 2
        assert out.items[1].topic == 'Elon Musk'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
