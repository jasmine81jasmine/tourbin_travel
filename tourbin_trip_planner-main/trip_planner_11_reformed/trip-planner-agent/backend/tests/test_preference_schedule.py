"""Tab-close fallback: pending turns survive restarts, retry safely, and stay scoped."""

import os
from uuid import uuid4

import pytest

from src.memory import preference_extractor, preference_schedule as schedule


@pytest.mark.asyncio
async def test_inactivity_sweep_finalizes_and_defers_failures(monkeypatch):
    pending = [{"user_id": "u1", "session_id": "s1"},
               {"user_id": "u2", "session_id": "s2"}]
    finalized, deferred = [], []

    async def find(driver, **kwargs):
        assert kwargs["inactivity_seconds"] == 600
        return pending

    async def finalize(driver, user_id, session_id):
        finalized.append((user_id, session_id))
        if session_id == "s2":
            raise RuntimeError("provider unavailable")

    async def defer(driver, user_id, session_id):
        deferred.append((user_id, session_id))

    monkeypatch.setattr(schedule, "pending_chats", find)
    monkeypatch.setattr(schedule, "finalize_preferences", finalize)
    monkeypatch.setattr(schedule, "defer_retry", defer)
    assert await schedule.sweep_preferences(object(), now=1234) == 1
    assert finalized == [("u1", "s1"), ("u2", "s2")]
    assert deferred == [("u2", "s2")]


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_AUTH_INTEGRATION") != "1", reason="Requires local Neo4j")
async def test_pending_chat_survives_independent_driver_and_is_cleared_after_processing():
    from neo4j import AsyncGraphDatabase
    from src.config import get_settings

    settings = get_settings()
    credentials = (settings.neo4j_username, settings.neo4j_password.get_secret_value())
    user = f"account_schedule_test_{uuid4().hex}"
    session_id = f"session_schedule_test_{uuid4().hex}"
    db = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=credentials)
    try:
        async with db.session() as neo:
            result = await neo.run(
                "CREATE (u:User {identifier: $user})-[:OWNS_CHAT]->"
                "(:ChatSession {session_id: $session_id})",
                user=user, session_id=session_id)
            await result.consume()
        await schedule.mark_chat_activity(db, user, session_id)
        async with db.session() as neo:
            row = await (await neo.run(
                "MATCH (:User {identifier: $user})-[:OWNS_CHAT]->(c:ChatSession "
                "{session_id: $session_id}) RETURN c.last_activity_at AS last_activity",
                user=user, session_id=session_id)).single()
            last_activity = row["last_activity"]
        future = last_activity + schedule.INACTIVITY_SECONDS + 1
        # A separate driver simulates an API process restart.
        other_driver = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=credentials)
        try:
            assert not any(row["session_id"] == session_id for row in
                           await schedule.pending_chats(other_driver, now=last_activity))
            assert any(row["session_id"] == session_id for row in
                       await schedule.pending_chats(other_driver, now=future))
            assert await preference_extractor.finalize_preferences(other_driver, user, session_id) == 0
            assert not any(row["session_id"] == session_id for row in
                           await schedule.pending_chats(other_driver, now=future))
            await schedule.mark_chat_activity(other_driver, user, session_id)
            assert any(row["session_id"] == session_id for row in
                       await schedule.pending_chats(other_driver, now=future + 1))
        finally:
            await other_driver.close()
    finally:
        async with db.session() as neo:
            result = await neo.run(
                "MATCH (u:User {identifier: $user})-[:OWNS_CHAT]->"
                "(c:ChatSession {session_id: $session_id}) DETACH DELETE c, u",
                user=user, session_id=session_id)
            await result.consume()
        await db.close()
