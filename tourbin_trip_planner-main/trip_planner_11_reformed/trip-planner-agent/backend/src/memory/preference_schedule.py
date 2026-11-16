"""Durable inactivity fallback for preference extraction after a tab disappears."""

import asyncio
import logging
import time

from src.memory.client import get_memory_client
from src.memory.preference_extractor import finalize_preferences

logger = logging.getLogger(__name__)
INACTIVITY_SECONDS = 10 * 60
SWEEP_SECONDS = 60
RETRY_SECONDS = 5 * 60


async def mark_chat_activity(driver, user_id: str, session_id: str) -> None:
    """Mark a *completed* turn for extraction; revisions survive restarts."""
    async with driver.session() as session:
        result = await session.run(
            "MATCH (u:User {identifier: $user_id})-[:OWNS_CHAT]->"
            "(c:ChatSession {session_id: $session_id}) "
            "SET c.last_activity_at = $now, "
            "c.preference_revision = coalesce(c.preference_revision, 0) + 1",
            user_id=user_id, session_id=session_id, now=int(time.time()))
        await result.consume()


async def pending_chats(driver, *, now: int | None = None,
                        inactivity_seconds: int = INACTIVITY_SECONDS) -> list[dict]:
    now = int(time.time()) if now is None else now
    async with driver.session() as session:
        result = await session.run(
            "MATCH (u:User)-[:OWNS_CHAT]->(c:ChatSession) "
            "WHERE c.last_activity_at IS NOT NULL AND c.last_activity_at <= $cutoff "
            "AND coalesce(c.preference_revision, 0) > coalesce(c.preference_checked_revision, 0) "
            "AND coalesce(c.next_preference_retry_at, 0) <= $now "
            "RETURN u.identifier AS user_id, c.session_id AS session_id "
            "ORDER BY c.last_activity_at LIMIT 20",
            cutoff=now - inactivity_seconds, now=now)
        return [r.data() async for r in result]


async def defer_retry(driver, user_id: str, session_id: str) -> None:
    async with driver.session() as session:
        result = await session.run(
            "MATCH (u:User {identifier: $user_id})-[:OWNS_CHAT]->"
            "(c:ChatSession {session_id: $session_id}) "
            "SET c.next_preference_retry_at = $retry_at",
            user_id=user_id, session_id=session_id, retry_at=int(time.time()) + RETRY_SECONDS)
        await result.consume()


async def sweep_preferences(driver, *, now: int | None = None,
                            inactivity_seconds: int = INACTIVITY_SECONDS) -> int:
    processed = 0
    for chat in await pending_chats(driver, now=now, inactivity_seconds=inactivity_seconds):
        try:
            await finalize_preferences(driver, chat["user_id"], chat["session_id"])
            processed += 1
        except Exception:
            logger.warning("Preference extraction failed for %s; retry scheduled",
                           chat["session_id"], exc_info=True)
            await defer_retry(driver, chat["user_id"], chat["session_id"])
    return processed


async def preference_worker() -> None:
    """Restart-safe periodic scan; do not rely on pagehide/sendBeacon delivery."""
    while True:
        await asyncio.sleep(SWEEP_SECONDS)
        memory = get_memory_client()
        if memory is None:
            continue
        try:
            await sweep_preferences(memory._client._ensure_connected())
        except Exception:
            logger.exception("Preference inactivity sweep failed")
