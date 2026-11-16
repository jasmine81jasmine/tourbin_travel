"""LLM proposes preference evidence; the backend alone validates and writes it."""

import json
import logging
import re

from pydantic_ai import Agent

from src.agent.agent import _build_model
from src.config import get_settings
from src.memory.preference_graph import (
    PreferenceBatch, apply_updates, available_features, session_messages, user_preferences,
)

logger = logging.getLogger(__name__)

PROMPT = """You are a Travel Preference Extraction Engine. Return JSON only:
{"updates":[{"node_id":"...","positive_increment":0,"negative_increment":1,
"positive_evidence_message_ids":[],"negative_evidence_message_ids":["..."]}]}
Only choose node IDs from AVAILABLE PREFERENCE NODES. Never create nodes.
Use only statements by the USER as evidence; assistant turns provide context only.
Existing preferences are context, NOT new evidence. Neutral questions, other
people's tastes and one-trip-only constraints are NOT durable preferences.
Distinguish likes from dislikes; do not infer physical ability from a travel
preference. Increments must each be 0 or 1, with at least one valid user
message ID for every nonzero increment. Count a feature at most once per
polarity per conversation. Do not calculate cumulative scores. If unsure,
return {"updates":[]}. Do not invent, generalize or over-interpret evidence."""


async def propose_updates(messages: list[dict], current: list[dict],
                          features: list[dict]) -> PreferenceBatch:
    agent = Agent(_build_model(get_settings()), system_prompt=PROMPT)
    result = await agent.run(json.dumps({
        "AVAILABLE PREFERENCE NODES": features,
        "EXISTING USER PREFERENCES": current,
        "LATEST CONVERSATION": messages,
    }, ensure_ascii=False, default=str))
    output = result.output
    if not isinstance(output, str):
        raise ValueError("Preference extractor did not return JSON")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.IGNORECASE)
    return PreferenceBatch.model_validate(json.loads(text))


async def read_revision(driver, user_id: str, session_id: str) -> int:
    async with driver.session() as session:
        result = await session.run(
            "MATCH (u:User {identifier: $user_id})-[:OWNS_CHAT]->"
            "(c:ChatSession {session_id: $session_id}) "
            "RETURN coalesce(c.preference_revision, 0) AS revision",
            user_id=user_id, session_id=session_id)
        row = await result.single()
    if not row:
        raise ValueError("Conversation is not owned by this user")
    return row["revision"]


async def mark_checked(driver, user_id: str, session_id: str, revision: int) -> None:
    async with driver.session() as session:
        result = await session.run(
            "MATCH (u:User {identifier: $user_id})-[:OWNS_CHAT]->"
            "(c:ChatSession {session_id: $session_id}) "
            "SET c.preference_checked_revision = "
            "CASE WHEN coalesce(c.preference_checked_revision, 0) < $revision "
            "THEN $revision ELSE coalesce(c.preference_checked_revision, 0) END, "
            "c.next_preference_retry_at = 0",
            user_id=user_id, session_id=session_id, revision=revision)
        await result.consume()


async def finalize_preferences(driver, user_id: str, session_id: str) -> int:
    revision = await read_revision(driver, user_id, session_id)
    messages = await session_messages(driver, user_id, session_id)
    updated = 0
    if any(m["role"] == "user" for m in messages):
        features = await available_features(driver)
        if not features:
            raise RuntimeError("Preference vocabulary is unavailable")
        current = await user_preferences(driver, user_id)
        batch = await propose_updates(messages, current, features)
        updated = await apply_updates(driver, user_id, session_id, batch)
    await mark_checked(driver, user_id, session_id, revision)
    return updated
