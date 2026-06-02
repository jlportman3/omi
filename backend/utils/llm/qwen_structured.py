"""Qwen-on-LiteLLM structured-output workaround.

Qwen3.6-35b-a3b served via the local LiteLLM proxy does not honor OpenAI's
`response_format={'type': 'json_schema', 'strict': True}` enforcement: it
wraps its JSON in markdown ```json fences with a prose preamble, and even
ignores the schema's required keys (inventing its own keys like
``test_context`` / ``observations`` instead of the requested ``people`` /
``topics`` / ``entities`` / ``dates``).

This module provides ``QwenChatOpenAI``, a thin subclass of
``langchain_openai.ChatOpenAI`` whose ``with_structured_output`` is
re-implemented for Qwen-via-LiteLLM:

  1. Build a compact JSON-schema description from the Pydantic model.
  2. Append it to the user prompt with explicit "respond with ONLY a JSON
     object, no markdown fences, no prose" instructions — Qwen reliably
     emits the requested keys when shown the schema in-context.
  3. Send ``response_format={'type': 'json_object'}`` to force valid JSON
     (LiteLLM passes this through and Qwen DOES honor it).
  4. Strip any residual markdown fences / leading prose from the response.
  5. Validate via Pydantic and return the model instance.

Identification of the Qwen client happens at construction time in
``clients.py`` — only the local-profile OpenAI client gets the subclass;
GPT-4.x / Gemini / OpenRouter clients keep the stock LangChain behavior
(which works correctly on those endpoints).

Background and trace logs in
``backend/CLAUDE.md`` and the phase-2 diagnose output for conversation
``e6bf5be3-3c2e-4c2f-940b-6f7454a29b62``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Type, Union

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError


# Strip markdown ```json``` fences and any leading/trailing prose that Qwen
# sometimes wraps around the JSON payload. Falls back to greedy first-{ /
# last-} extraction when no fence is present.
_FENCE_RE = re.compile(r'```(?:json|JSON)?\s*\n?(.+?)\n?```', re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Return the JSON payload from a possibly-fenced/prefaced LLM response.

    Handles all observed Qwen shapes:
      - '\\n\\n```json\\n{...}\\n```\\n'         (fenced, no preamble)
      - 'Here is the JSON:\\n```json\\n{...}\\n```' (fenced + preamble)
      - '**Step 1: ...**\\n\\n```json\\n{...}\\n```' (markdown-prefaced)
      - '\\n\\n{...}'                              (raw, leading whitespace)

    Never raises — if no JSON-looking substring is found, returns the
    original string and lets the downstream json.loads surface the error.
    """
    if not text:
        return text
    stripped = text.strip()

    match = _FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()

    first = stripped.find('{')
    last = stripped.rfind('}')
    if first >= 0 and last > first:
        return stripped[first : last + 1].strip()

    return stripped


def _schema_description(schema: Type[BaseModel]) -> str:
    """Build a compact human-readable schema block from a Pydantic class.

    Output looks like::

        {
          "people": "List[str] — Identify all the people names ..."
          "topics": "List[str] — List all the main topics ..."
          ...
        }

    Qwen reliably mirrors this exact key set when this block is appended
    to the prompt with explicit "match these keys" instructions.
    """
    try:
        json_schema = schema.model_json_schema()
    except Exception:
        # Best-effort fallback — never block the call on schema inspection
        return ''

    properties = json_schema.get('properties', {})
    lines = ['{']
    for name, spec in properties.items():
        py_type = spec.get('type', 'any')
        if py_type == 'array':
            items = spec.get('items', {}).get('type', 'any')
            py_type = f'List[{items}]'
        desc = (spec.get('description') or '').replace('\n', ' ').strip()
        if desc:
            lines.append(f'  "{name}": "{py_type} — {desc}",')
        else:
            lines.append(f'  "{name}": "{py_type}",')
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(',')
    lines.append('}')
    return '\n'.join(lines)


def _augment_prompt_with_schema(prompt: Union[str, list], schema: Type[BaseModel]) -> Union[str, list]:
    """Append an explicit schema/keys block to the user-visible prompt.

    For a list-of-messages input we append the schema instructions to the
    last HumanMessage (creating one if absent). For a plain string we
    return prompt + schema block.
    """
    schema_block = _schema_description(schema)
    if not schema_block:
        return prompt

    instructions = (
        '\n\nRespond with ONLY a single JSON object matching this exact schema. '
        'Use exactly the keys shown below — do not invent new keys, do not omit any. '
        'Do NOT wrap the JSON in markdown code fences. Do NOT include any prose, '
        'preamble, or commentary before or after the JSON. Empty arrays should be '
        '`[]`, not `null` or `"None"`.\n\n'
        f'Schema:\n{schema_block}'
    )

    if isinstance(prompt, str):
        return prompt + instructions

    if isinstance(prompt, list):
        # Append to last HumanMessage; if none, add a new one
        new_prompt = list(prompt)
        for i in range(len(new_prompt) - 1, -1, -1):
            msg = new_prompt[i]
            if isinstance(msg, HumanMessage):
                new_prompt[i] = HumanMessage(content=str(msg.content) + instructions)
                return new_prompt
            # Some callers pass dicts
            if isinstance(msg, dict) and msg.get('role') == 'user':
                new_msg = dict(msg)
                new_msg['content'] = str(new_msg.get('content', '')) + instructions
                new_prompt[i] = new_msg
                return new_prompt
        new_prompt.append(HumanMessage(content=instructions.lstrip()))
        return new_prompt

    return prompt


def _parse_qwen_structured_response(content: str, schema: Type[BaseModel]) -> BaseModel:
    """Strip fences, json.loads, validate via Pydantic. Raises ValidationError on failure."""
    cleaned = _strip_markdown_fences(content or '')
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Surface as ValidationError so existing try/except blocks in callers
        # (which catch generic Exception) keep working without changes.
        raise ValidationError.from_exception_data(
            title=schema.__name__,
            line_errors=[
                {
                    'type': 'json_invalid',
                    'loc': (),
                    'input': cleaned,
                    'ctx': {'error': str(exc)},
                }
            ],
        ) from exc
    return schema.model_validate(payload)


class QwenChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that fixes ``with_structured_output`` for Qwen-via-LiteLLM.

    All other behavior (streaming, tool use, plain ``.invoke``) is inherited
    unchanged from the parent class. Only structured output is overridden.

    When a non-Pydantic schema is passed, falls back to the parent
    implementation — the workaround only applies to Pydantic v2 BaseModel
    classes, which is the shape used everywhere in this codebase.
    """

    def with_structured_output(  # type: ignore[override]
        self,
        schema: Any = None,
        *,
        method: str = 'json_schema',
        include_raw: bool = False,
        strict: Any = None,
        **kwargs: Any,
    ):
        # Pass through when the caller asked for a non-Pydantic shape or for
        # raw output (tool-use codepath). Pydantic v2 BaseModel is the only
        # shape we need to fix in this codebase.
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return super().with_structured_output(
                schema, method=method, include_raw=include_raw, strict=strict, **kwargs
            )

        # Build a chain that:
        #   1. augments the prompt with schema instructions
        #   2. binds response_format=json_object on the LLM
        #   3. extracts the .content string
        #   4. strips fences + parses via Pydantic
        bound_llm = self.bind(response_format={'type': 'json_object'})

        def _run(prompt_input: Any) -> BaseModel:
            augmented = _augment_prompt_with_schema(prompt_input, schema)
            ai_msg = bound_llm.invoke(augmented)
            content = getattr(ai_msg, 'content', None)
            if isinstance(content, list):
                # Some message shapes return content blocks
                content = ''.join(block.get('text', '') if isinstance(block, dict) else str(block) for block in content)
            return _parse_qwen_structured_response(content or '', schema)

        async def _arun(prompt_input: Any) -> BaseModel:
            augmented = _augment_prompt_with_schema(prompt_input, schema)
            ai_msg = await bound_llm.ainvoke(augmented)
            content = getattr(ai_msg, 'content', None)
            if isinstance(content, list):
                content = ''.join(block.get('text', '') if isinstance(block, dict) else str(block) for block in content)
            return _parse_qwen_structured_response(content or '', schema)

        return RunnableLambda(_run, afunc=_arun)
