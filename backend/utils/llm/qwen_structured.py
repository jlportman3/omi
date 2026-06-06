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
from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
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


def _resolve_ref(spec: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
    """Follow a $ref like '#/$defs/Item' to its actual definition.

    Pydantic v2 emits $refs for every nested BaseModel / Enum, so we have
    to chase them to inspect the real shape. Returns the original spec
    untouched when no $ref is present or the target is missing.
    """
    ref = spec.get('$ref')
    if not ref or not ref.startswith('#/$defs/'):
        return spec
    name = ref.split('/', 2)[-1]
    target = defs.get(name)
    if target is None:
        return spec
    # Preserve outer description/title if the $ref entry had one
    merged = dict(target)
    if 'description' in spec and 'description' not in merged:
        merged['description'] = spec['description']
    return merged


def _type_label(spec: Dict[str, Any], defs: Dict[str, Any]) -> str:
    """Render a compact Python-style type label for a JSON-schema property.

    Examples::

        {"type": "string"}                      -> "str"
        {"type": "array", "items": {"type":"string"}} -> "List[str]"
        {"enum": ["a","b"]}                     -> "enum(a|b)"
        {"type": "object"}                      -> "object"
    """
    spec = _resolve_ref(spec, defs)

    if 'enum' in spec:
        vals = '|'.join(str(v) for v in spec['enum'])
        return f'enum({vals})'

    py_type = spec.get('type', 'any')
    if py_type == 'array':
        item_spec = _resolve_ref(spec.get('items', {}) or {}, defs)
        inner = _type_label(item_spec, defs)
        return f'List[{inner}]'
    if py_type == 'integer':
        return 'int'
    if py_type == 'number':
        return 'float'
    if py_type == 'string':
        return 'str'
    if py_type == 'boolean':
        return 'bool'
    if py_type == 'object':
        return 'object'
    return py_type


def _emit_object_fields(
    spec: Dict[str, Any],
    defs: Dict[str, Any],
    path_prefix: str,
    lines: List[str],
    visited: set,
) -> None:
    """Append "  \"<path>\": \"<type> — <desc>\"," lines for every field of an object spec.

    Recurses into nested objects (whether referenced via $ref or inlined)
    and into the element type of arrays-of-objects, emitting paths like
    ``items[*].category`` so Qwen sees the exact key names it must use at
    every level. ``visited`` guards against $ref cycles in self-referential
    schemas.
    """
    spec = _resolve_ref(spec, defs)
    title = spec.get('title')
    if title:
        if title in visited:
            return
        visited = visited | {title}

    properties: Dict[str, Any] = spec.get('properties', {}) or {}
    for name, raw_child in properties.items():
        child = _resolve_ref(raw_child, defs)
        label = _type_label(raw_child, defs)
        # Prefer the outer (field-level) description over the referenced
        # model's own title-block description.
        desc = (raw_child.get('description') or child.get('description') or '').replace('\n', ' ').strip()

        key_path = f'{path_prefix}.{name}' if path_prefix else name
        if desc:
            lines.append(f'  "{key_path}": "{label} — {desc}",')
        else:
            lines.append(f'  "{key_path}": "{label}",')

        # Recurse into nested objects so Qwen sees the inner field names.
        if child.get('type') == 'object' and child.get('properties'):
            _emit_object_fields(child, defs, key_path, lines, visited)
        elif child.get('type') == 'array':
            item_spec = _resolve_ref(child.get('items', {}) or {}, defs)
            if item_spec.get('type') == 'object' and item_spec.get('properties'):
                _emit_object_fields(item_spec, defs, f'{key_path}[*]', lines, visited)


def _schema_description(schema: Type[BaseModel]) -> str:
    """Build a compact human-readable schema block from a Pydantic class.

    Output looks like::

        {
          "items": "List[object] — List of items.",
          "items[*].category": "enum(ceo|company|...) — The category identified",
          "items[*].type": "enum(best|worst) — The sentiment identified",
          "items[*].topic": "str — The specific topic corresponding the category"
        }

    The recursive form is required for Qwen to emit the right keys at every
    level: when only the top-level properties are shown, Qwen invents its
    own keys for the inner objects (e.g. ``Category`` / ``Type`` / ``Topic``
    in title case) which then fail Pydantic validation. By giving it the
    exact nested key paths and enum value lists, it reliably mirrors the
    expected shape.
    """
    try:
        json_schema = schema.model_json_schema()
    except Exception:
        # Best-effort fallback — never block the call on schema inspection
        return ''

    defs: Dict[str, Any] = json_schema.get('$defs', {}) or {}
    lines: List[str] = ['{']
    _emit_object_fields(json_schema, defs, '', lines, set())
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(',')
    lines.append('}')
    return '\n'.join(lines)


# Pydantic field names are case-sensitive but Qwen sometimes outputs Title-Case
# keys (Category/Type/Topic) even when shown the exact lowercase schema. This
# defense-in-depth pass normalizes object keys recursively so the resulting
# JSON parses cleanly even when Qwen does not perfectly mirror the schema.
def _build_key_map(schema: Type[BaseModel]) -> Dict[str, Dict[str, str]]:
    """Return ``{model_title: {lowercase_key: canonical_key}}`` for every nested model.

    Used by ``_normalize_keys`` to repair Qwen outputs where keys differ only
    in case from the canonical Pydantic field names.
    """
    try:
        json_schema = schema.model_json_schema()
    except Exception:
        return {}
    defs: Dict[str, Any] = json_schema.get('$defs', {}) or {}
    key_map: Dict[str, Dict[str, str]] = {}

    def _walk(spec: Dict[str, Any]) -> None:
        spec = _resolve_ref(spec, defs)
        title = spec.get('title')
        properties = spec.get('properties', {}) or {}
        if title and properties and title not in key_map:
            key_map[title] = {k.lower(): k for k in properties.keys()}
        for child in properties.values():
            resolved = _resolve_ref(child, defs)
            if resolved.get('type') == 'object':
                _walk(resolved)
            elif resolved.get('type') == 'array':
                _walk(resolved.get('items', {}) or {})

    _walk(json_schema)
    return key_map


def _all_canonical_keys(key_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Flatten every nested model's lowercase→canonical map into one dict.

    Different nested models in the same schema rarely use the same field
    name with different casing, so a single flat map is enough and avoids
    having to thread the model identity through every recursive call.
    """
    flat: Dict[str, str] = {}
    for model_keys in key_map.values():
        for low, canon in model_keys.items():
            flat[low] = canon
    return flat


def _normalize_keys(payload: Any, canonical_keys: Dict[str, str]) -> Any:
    """Recursively rename object keys to their canonical form (case-insensitive).

    Walks dicts and lists. Unknown keys are passed through unchanged so the
    Pydantic validator surfaces them as ``extra_forbidden`` only when the
    model explicitly forbids extras — most omi models accept them.
    """
    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for k, v in payload.items():
            canon = canonical_keys.get(k.lower(), k) if isinstance(k, str) else k
            out[canon] = _normalize_keys(v, canonical_keys)
        return out
    if isinstance(payload, list):
        return [_normalize_keys(item, canonical_keys) for item in payload]
    return payload


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
    """Strip fences, json.loads, normalize key casing, validate via Pydantic.

    Two-stage validation:
      1. First attempt uses the raw payload — fastest path when Qwen mirrors
         the schema exactly (most calls).
      2. On ValidationError, we rebuild the payload with case-insensitive key
         normalization against every nested model in the schema and re-validate.

    This recovers gracefully from the observed Qwen failure mode where, even
    when shown the exact schema, it emits Title-Case keys for nested objects
    (``{"Category": "company", "Type": "best", "Topic": "Tesla"}`` instead of
    the required ``{"category", "type", "topic"}``).
    """
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
    try:
        return schema.model_validate(payload)
    except ValidationError:
        key_map = _build_key_map(schema)
        if not key_map:
            raise
        normalized = _normalize_keys(payload, _all_canonical_keys(key_map))
        return schema.model_validate(normalized)


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


class QwenPydanticOutputParser(PydanticOutputParser):
    """``PydanticOutputParser`` subclass that tolerates Qwen-on-LiteLLM output.

    The stock parser feeds the LLM's raw text directly to ``json.loads`` →
    Pydantic. Qwen wraps its JSON in markdown ``` fences with a prose preamble
    (same shape ``QwenChatOpenAI.with_structured_output`` strips above), and
    also occasionally emits Title-Case keys for nested object fields. This
    subclass mirrors both defenses for callers that use the legacy
    ``prompt | llm | parser`` chain pattern (PydanticOutputParser bypasses
    ``with_structured_output`` entirely).

    Drop-in replacement: ``PydanticOutputParser(pydantic_object=Foo)`` →
    ``QwenPydanticOutputParser(pydantic_object=Foo)``. No call-site changes
    beyond the constructor name.

    Affected call sites:
      utils/llm/memories.py             (memories, learnings, memory_conflict)
      utils/llm/knowledge_graph.py      (KnowledgeGraphExtraction)
      utils/llm/conversation_processing.py (folder, discard, action_items, structure)
      utils/llm/external_integrations.py  (external_structure)
    """

    def parse(self, text: str) -> BaseModel:  # type: ignore[override]
        # 1. Strip markdown fences + prose preamble.
        stripped = _strip_markdown_fences(text or '')
        # 2. Defense-in-depth: try the stock parser first; on ValidationError,
        #    retry with case-insensitive key normalization (same approach as
        #    _parse_qwen_structured_response above).
        try:
            return super().parse(stripped)
        except (ValidationError, json.JSONDecodeError):
            # Re-parse the JSON payload, normalize keys via the schema's
            # canonical key map, then validate. Failures bubble up to the
            # caller exactly as before — no behavior change on success.
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                raise  # surface the original error
            key_map = _build_key_map(self.pydantic_object)
            canonical = _all_canonical_keys(key_map)
            normalized = _normalize_keys(payload, canonical)
            return self.pydantic_object.model_validate(normalized)
