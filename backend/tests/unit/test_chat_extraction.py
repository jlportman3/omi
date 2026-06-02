"""Regression tests for ``ExtractedInformation`` parsing under MODEL_QOS=local_only.

Qwen3.6-35b-a3b served via the local LiteLLM proxy emits markdown-fenced
JSON with prose preambles and sometimes invents its own keys, so the stock
``ChatOpenAI.with_structured_output(Pydantic)`` path fails with::

    pydantic_core._pydantic_core.ValidationError: 1 validation error for
    ExtractedInformation
      Invalid JSON: expected value at line 3 column 1 [type=json_invalid,
      input_value='\\n\\n```json\\n{...}\\n```']

The fix is ``QwenChatOpenAI`` (``utils/llm/qwen_structured.py``) — a
ChatOpenAI subclass installed for qwen-prefixed models in
``_get_or_create_openai_llm`` that:

  1. Appends an explicit schema/keys block to the user prompt.
  2. Sends ``response_format={'type': 'json_object'}`` instead of
     ``json_schema`` (Qwen ignores ``strict:true`` via LiteLLM).
  3. Strips markdown fences and leading prose from the response.
  4. Validates via Pydantic.

These tests exercise the parsing layer with the EXACT raw Qwen responses
captured during diagnosis (see /tmp/qwen_extracted_raw.json) — no network
calls, fully deterministic.
"""

import sys
from typing import List
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Pre-mock heavy deps before importing the LLM module under test.
# Mirrors test_llm_qos_profiles.py to keep tests hermetic.
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
    _parse_qwen_structured_response,
    _strip_markdown_fences,
)


# Mirrors ExtractedInformation in utils/llm/chat.py — duplicated here so the
# test doesn't drag the whole chat.py import graph (Firestore, Redis, etc.).
class ExtractedInformation(BaseModel):
    people: List[str] = Field(default=[], description='Identify all the people names mentioned.')
    topics: List[str] = Field(default=[], description='List all the main topics discussed.')
    entities: List[str] = Field(default=[], description='List any products, technologies, places.')
    dates: List[str] = Field(default=[], description='Extract any dates in YYYY-MM-DD format.')


# Raw Qwen outputs captured during diagnosis of conversation
# e6bf5be3-3c2e-4c2f-940b-6f7454a29b62 (see /tmp/qwen_extracted_raw.json).
# Every shape below has previously caused a ValidationError in production.
_QWEN_RAW_FENCED_JSON_VALID_KEYS = (
    '\n\n```json\n'
    '{\n'
    '  "people": [],\n'
    '  "topics": ["OMI backend", "transcription speed", "AI hallucination"],\n'
    '  "entities": ["OMI backend", "transcription system"],\n'
    '  "dates": ["2026-06-02"]\n'
    '}\n'
    '```'
)

_QWEN_RAW_FENCED_WITH_PROSE_PREAMBLE = (
    '\n\n**Step 1: Corrected transcript**\n\n'
    'Here is the extracted information:\n\n'
    '```json\n'
    '{\n'
    '  "people": ["Joe"],\n'
    '  "topics": ["machine learning"],\n'
    '  "entities": ["OpenAI"],\n'
    '  "dates": []\n'
    '}\n'
    '```\n'
)

_QWEN_RAW_RAW_JSON_NO_FENCES = (
    '\n\n'
    '{\n'
    '  "people": [],\n'
    '  "topics": ["OMI testing"],\n'
    '  "entities": ["OMI backend"],\n'
    '  "dates": []\n'
    '}'
)


class TestStripMarkdownFences:
    """The fence stripper is the heart of the fix — exercise every observed shape."""

    def test_fenced_json_with_leading_whitespace(self):
        out = _strip_markdown_fences(_QWEN_RAW_FENCED_JSON_VALID_KEYS)
        assert out.startswith('{')
        assert out.endswith('}')
        # No markdown fences left over
        assert '```' not in out

    def test_fenced_json_with_prose_preamble(self):
        out = _strip_markdown_fences(_QWEN_RAW_FENCED_WITH_PROSE_PREAMBLE)
        assert out.startswith('{')
        assert out.endswith('}')
        assert '```' not in out
        assert 'Step 1' not in out
        assert 'Here is' not in out

    def test_raw_json_no_fences(self):
        out = _strip_markdown_fences(_QWEN_RAW_RAW_JSON_NO_FENCES)
        assert out.startswith('{')
        assert out.endswith('}')

    def test_empty_input(self):
        assert _strip_markdown_fences('') == ''

    def test_no_json_at_all(self):
        # If there's nothing JSON-ish, return the original string and let
        # downstream json.loads surface the error.
        out = _strip_markdown_fences('I am sorry but I cannot help with that.')
        assert 'I am sorry' in out


class TestParseQwenStructuredResponse:
    """Pydantic-validation path — must accept all observed Qwen shapes."""

    def test_parses_fenced_valid_keys(self):
        obj = _parse_qwen_structured_response(_QWEN_RAW_FENCED_JSON_VALID_KEYS, ExtractedInformation)
        assert isinstance(obj, ExtractedInformation)
        assert obj.people == []
        assert 'OMI backend' in obj.topics
        assert 'transcription system' in obj.entities
        assert obj.dates == ['2026-06-02']

    def test_parses_fenced_with_prose_preamble(self):
        obj = _parse_qwen_structured_response(_QWEN_RAW_FENCED_WITH_PROSE_PREAMBLE, ExtractedInformation)
        assert isinstance(obj, ExtractedInformation)
        assert obj.people == ['Joe']
        assert obj.topics == ['machine learning']
        assert obj.entities == ['OpenAI']
        assert obj.dates == []

    def test_parses_raw_json_no_fences(self):
        obj = _parse_qwen_structured_response(_QWEN_RAW_RAW_JSON_NO_FENCES, ExtractedInformation)
        assert isinstance(obj, ExtractedInformation)
        assert obj.entities == ['OMI backend']

    def test_invalid_json_raises_validation_error(self):
        from pydantic import ValidationError as _VE

        with pytest.raises(_VE):
            _parse_qwen_structured_response('I am sorry but I cannot help with that.', ExtractedInformation)


class TestQwenChatOpenAIStructuredOutput:
    """End-to-end: ``llm.with_structured_output(Pydantic).invoke(prompt)`` returns a Pydantic instance.

    Mocks the underlying ChatOpenAI HTTP call so the test is hermetic.
    """

    def _build_qwen(self) -> QwenChatOpenAI:
        # base_url + api_key are required-ish for ChatOpenAI to instantiate; values are
        # not used because we monkey-patch the bound LLM's .invoke()
        return QwenChatOpenAI(model='qwen3.6-35b-a3b', api_key='sk-fake', base_url='http://localhost:4000/v1')

    def test_structured_output_unmarshals_fenced_qwen_response(self):
        """The exact failure mode from production: Qwen returns markdown-fenced JSON,
        Pydantic validation must succeed end-to-end."""
        llm = self._build_qwen()
        chain = llm.with_structured_output(ExtractedInformation)

        # Patch the bound LLM's .invoke() to return the captured Qwen AIMessage.
        # The chain's _run closure calls bound_llm.invoke(augmented_prompt).
        fake_msg = AIMessage(content=_QWEN_RAW_FENCED_JSON_VALID_KEYS)

        with patch('langchain_openai.ChatOpenAI.invoke', return_value=fake_msg):
            result = chain.invoke('any prompt')

        assert isinstance(result, ExtractedInformation)
        assert result.dates == ['2026-06-02']
        assert 'OMI backend' in result.topics

    def test_structured_output_unmarshals_prose_preamble(self):
        llm = self._build_qwen()
        chain = llm.with_structured_output(ExtractedInformation)
        fake_msg = AIMessage(content=_QWEN_RAW_FENCED_WITH_PROSE_PREAMBLE)

        with patch('langchain_openai.ChatOpenAI.invoke', return_value=fake_msg):
            result = chain.invoke('any prompt')

        assert isinstance(result, ExtractedInformation)
        assert result.people == ['Joe']
        assert result.entities == ['OpenAI']

    def test_structured_output_unmarshals_raw_json(self):
        llm = self._build_qwen()
        chain = llm.with_structured_output(ExtractedInformation)
        fake_msg = AIMessage(content=_QWEN_RAW_RAW_JSON_NO_FENCES)

        with patch('langchain_openai.ChatOpenAI.invoke', return_value=fake_msg):
            result = chain.invoke('any prompt')

        assert isinstance(result, ExtractedInformation)
        assert result.entities == ['OMI backend']


class TestQwenClientWiring:
    """Verify the local Qwen client picks up the subclass via clients.py wiring."""

    def test_local_qwen_client_is_subclass(self):
        # Re-import the cached factory; relies on a real env where LiteLLM is reachable
        # is NOT needed — instantiation alone is enough.
        from utils.llm.clients import _get_or_create_openai_llm

        client = _get_or_create_openai_llm('qwen3.6-35b-a3b', streaming=False)
        assert isinstance(client, QwenChatOpenAI), (
            f"local qwen client must be QwenChatOpenAI for the structured-output fix to apply, "
            f"got {type(client).__name__}"
        )

    def test_openai_client_is_stock_chatopenai(self):
        """The fix must NOT affect non-Qwen OpenAI models — they handle structured output natively."""
        from langchain_openai import ChatOpenAI

        from utils.llm.clients import _get_or_create_openai_llm

        client = _get_or_create_openai_llm('gpt-4.1-mini', streaming=False)
        assert isinstance(client, ChatOpenAI)
        assert not isinstance(
            client, QwenChatOpenAI
        ), f"non-qwen OpenAI clients must keep stock ChatOpenAI behavior, got {type(client).__name__}"
