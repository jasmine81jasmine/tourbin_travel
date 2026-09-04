"""OpenAI-compatible local adapter for Ferdowsi University's LLM and
embedding gateway (fumllm.um.ac.ir / ai-gateway.um.ac.ir).

Why this exists
----------------
Every OpenAI-compatible-proxy provider in this project (AvalAI, and now the
university) is wired in through `LLM_BASE_URL` / `EMBEDDING_BASE_URL`, and
consumed unmodified by pydantic-ai's `OpenAIChatModel` and
`neo4j-agent-memory`'s OpenAI adapter. That only works if the thing living
at `base_url` actually speaks the OpenAI wire format -- and the university's
APIs don't:

- LLM (`POST /llm/query_unified`): takes a single `query` string + an
  optional `system_prompt` + a `server_id`. No message array, no tool
  calling, no streaming.
- Embeddings (`POST /ai-embedding/api/embeddings`): takes `{model, input}`
  with `input` as a single string, and returns a response shape that's
  *almost* OpenAI's `data: [{index, embedding}]`, but missing the
  `object`/`usage` fields the OpenAI SDK expects.

This module is a small FastAPI router, mounted in `main.py`, that exposes
`/v1/chat/completions` and `/v1/embeddings` in real OpenAI shape and
translates each call to the university's actual API underneath. Point
`LLM_BASE_URL` / `EMBEDDING_BASE_URL` at this router (see `.env.example`)
and nothing else in the codebase needs to know the university isn't OpenAI.

Tool-calling caveat
--------------------
The university LLM has no native function calling and no multi-turn
conversation state. To let the pydantic-ai agent still call its `tool_*`
functions, this adapter does *prompted* tool calling:

1. On every request, the full OpenAI `messages` array (system + user +
   assistant + prior tool calls/results) is re-serialized into one text
   transcript, because the upstream API is stateless / single-turn.
2. If `tools` were supplied, their name/description/JSON-schema are listed
   in the system prompt along with an instruction to reply with exactly one
   JSON object: either `{"tool_call": {"name": ..., "arguments": {...}}}`
   or `{"final_answer": "..."}`.
3. The raw text reply is parsed back into an OpenAI-shaped
   `choices[0].message` with either `tool_calls` or plain `content`.

This is best-effort. Small/instruction-light models can ignore the format
and just answer in prose -- if that happens, the adapter falls back to
treating the whole reply as the final answer rather than erroring out, so
the agent degrades to "no tool use this turn" instead of crashing.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from src.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

TOOL_CALL_INSTRUCTIONS = (
    "\n\nWhen you need information you don't have, call exactly one tool by "
    "replying with ONLY a JSON object of this form and nothing else "
    '(no markdown fences, no extra text):\n'
    '{"tool_call": {"name": "<tool name>", "arguments": {<matching JSON schema>}}}\n'
    "When you have enough information to answer the user directly, reply "
    "with ONLY:\n"
    '{"final_answer": "<your reply to the user>"}\n'
    "Always reply with exactly one of those two JSON shapes, never both, "
    "never plain prose outside the JSON."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def _content_to_text(content: Any) -> str:
    """OpenAI message content can be a plain string or a list of content
    parts (e.g. [{"type": "text", "text": "..."}]); normalize to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return "" if content is None else str(content)


def _render_transcript(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Split an OpenAI `messages` array into (system_prompt, transcript).

    The university API has no notion of conversation history, so the whole
    exchange -- including prior tool calls and their results -- is replayed
    as plain text in `transcript` on every call.
    """
    system_parts: list[str] = []
    turns: list[str] = []

    for msg in messages:
        role = msg.get("role")
        text = _content_to_text(msg.get("content"))

        if role == "system":
            if text:
                system_parts.append(text)
        elif role == "user":
            turns.append(f"[User]: {text}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                calls_desc = "; ".join(
                    f"{tc.get('function', {}).get('name')}"
                    f"({tc.get('function', {}).get('arguments')})"
                    for tc in tool_calls
                )
                turns.append(f"[Assistant called tool(s)]: {calls_desc}")
            if text:
                turns.append(f"[Assistant]: {text}")
        elif role == "tool":
            name = msg.get("name", "tool")
            turns.append(f"[Result of {name}]: {text}")

    return "\n".join(system_parts), "\n".join(turns)


def _tool_schema_block(tools: list[dict[str, Any]] | None) -> str:
    if not tools:
        return ""
    lines = ["You have the following tools available:"]
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        description = fn.get("description", "")
        params = fn.get("parameters")
        lines.append(f"- {name}: {description}")
        if params:
            lines.append(f"  arguments JSON schema: {json.dumps(params, ensure_ascii=False)}")
    lines.append(TOOL_CALL_INSTRUCTIONS)
    return "\n".join(lines)


def _parse_model_reply(text: str) -> tuple[str | None, dict[str, Any] | None]:
    """Return (final_text, tool_call). Exactly one is non-None on success;
    if the model didn't follow the JSON convention, treat the raw text as
    the final answer so the caller still gets a usable reply."""
    text = (text or "").strip()
    candidate = text
    if not (candidate.startswith("{") and candidate.endswith("}")):
        match = _JSON_OBJECT_RE.search(text)
        if match:
            candidate = match.group(0)

    try:
        obj = json.loads(candidate)
    except (ValueError, TypeError):
        return text, None

    if isinstance(obj, dict) and isinstance(obj.get("tool_call"), dict):
        return None, obj["tool_call"]
    if isinstance(obj, dict) and "final_answer" in obj:
        return str(obj["final_answer"]), None
    return text, None


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
    settings = get_settings()
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")

    system_prompt, transcript = _render_transcript(messages)
    tool_block = _tool_schema_block(tools)
    full_system_prompt = "\n\n".join(part for part in (system_prompt, tool_block) if part)

    api_key = _extract_bearer(authorization) or settings.llm_api_key.get_secret_value()
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "not-needed":
        headers["X-API-Key"] = api_key

    payload = {
        "query": transcript or "(no user message)",
        "system_prompt": full_system_prompt,
        "server_id": settings.university_llm_server_id,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(settings.university_llm_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    reply_text = data.get("response", "")
    usage = data.get("usage") or {}
    final_text, tool_call = _parse_model_reply(reply_text)

    message: dict[str, Any] = {"role": "assistant"}
    finish_reason = "stop"

    if tool_call and tools:
        args = tool_call.get("arguments", tool_call.get("args", {}))
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": tool_call.get("name", ""), "arguments": args},
            }
        ]
        finish_reason = "tool_calls"
    else:
        message["content"] = final_text if final_text is not None else reply_text

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model") or body.get("model") or "university-llm",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
    )


@router.post("/v1/embeddings")
async def embeddings(request: Request, authorization: str | None = Header(default=None)):
    settings = get_settings()
    body = await request.json()
    raw_input = body.get("input", "")
    texts = raw_input if isinstance(raw_input, list) else [raw_input]

    # settings.embedding_model is "openai/mpnet-multilingual"-style; the
    # university API wants the bare model id.
    configured_model = settings.embedding_model.split("/", 1)[-1]
    model_name = body.get("model") or configured_model

    api_key = _extract_bearer(authorization)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"

    data_out: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for index, text in enumerate(texts):
            if not text or not str(text).strip():
                # Deliberately fail loud and fast here rather than inventing
                # a placeholder embedding: a zero vector "solves" the
                # university gateway's 422 but just relocates the failure --
                # Neo4j's vector index requires a positive, finite L2-norm,
                # so a zero vector gets rejected downstream instead, with a
                # much more confusing error three layers deeper. If you hit
                # this, the actual fix is at the caller: don't generate an
                # embedding for empty text in the first place (e.g. use a
                # non-semantic "list everything" query instead of embedding
                # an empty search string).
                raise HTTPException(
                    status_code=400,
                    detail="Cannot embed empty/whitespace-only input.",
                )
            resp = await client.post(
                settings.university_embedding_url,
                headers=headers,
                json={"model": model_name, "input": text},
            )
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("data") or [{}]
            embedding = items[0].get("embedding", [])
            data_out.append({"object": "embedding", "index": index, "embedding": embedding})

    return JSONResponse(
        {
            "object": "list",
            "data": data_out,
            "model": model_name,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    )
