"""Account identity and ownership tests, independent of LLM/map providers."""

import os
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

from src.api.routes import auth


def test_password_hash_is_salted_and_rejects_wrong_password():
    first = auth.password_hash("a secure password")
    second = auth.password_hash("a secure password")
    assert first != second
    assert auth.verify_password("a secure password", first)
    assert not auth.verify_password("wrong password", first)
    assert not auth.verify_password("a secure password", "invalid")


def test_email_and_phone_normalization_and_validation():
    assert auth.normalize("  USER@Example.org ") == ("email", "user@example.org")
    assert auth.normalize("+98 (912) 123-4567") == ("phone", "+989121234567")
    for value in ("someone@", "123", "bad input", "a@b"):
        with pytest.raises(HTTPException):
            auth.normalize(value)


def test_persian_conversation_title_uses_request_not_greeting():
    assert auth.conversation_title("سلام") == "گفتگوی سفر"
    assert auth.conversation_title("سلام، یه سفر دو روزه اطراف تهران پیشنهاد بده") == (
        "یه سفر دو روزه اطراف تهران پیشنهاد بده")
    assert len(auth.conversation_title("برای سفر طبیعت گردی " * 10)) <= 53


@pytest.mark.asyncio
async def test_http_chat_and_session_require_login_and_ignore_old_cookie():
    from src.main import app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test", trust_env=False) as client:
        for endpoint in ("/api/session", "/api/chat"):
            response = await client.post(endpoint, json={"message": "hello", "user_id": "victim"},
                                         cookies={"tourbin_user_id": "victim"})
            assert response.status_code == 401
        assert (await client.get("/api/auth/history")).status_code == 401


@pytest.mark.asyncio
async def test_owned_session_is_checked_before_chat_memory_is_read(monkeypatch):
    from src.main import app
    from src.api.routes import chat

    async def identity():
        return auth.Identity("account_alice", "tokenhash")

    async def claim(user_id, session_id, *, create=False):
        assert user_id == "account_alice"
        if session_id == "session_bob":
            raise HTTPException(403, "This conversation is not yours")

    def no_memory():
        raise AssertionError("Memory must not be opened for a forbidden session")

    app.dependency_overrides[auth.require_user] = identity
    monkeypatch.setattr(chat, "claim_session", claim)
    monkeypatch.setattr(chat, "get_memory_client", no_memory)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test", trust_env=False) as client:
            response = await client.post("/api/chat", json={
                "message": "hello", "session_id": "session_bob", "user_id": "account_alice"})
            assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_AUTH_INTEGRATION") != "1",
                    reason="Opt-in live Neo4j integration test")
async def test_live_register_login_logout_and_session_isolation(monkeypatch):
    from neo4j import AsyncGraphDatabase
    from src.config import get_settings
    from src.main import app
    from src.api.routes import chat

    settings = get_settings()
    db = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=(
        settings.neo4j_username, settings.neo4j_password.get_secret_value()))
    monkeypatch.setattr(auth, "driver", lambda: db)
    monkeypatch.setattr(chat, "get_memory_client", lambda: None)
    email = f"auth-test-{uuid4().hex}@example.org"
    password = "an excellent test password"
    user_id = None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test", trust_env=False) as alice:
            registered = await alice.post("/api/auth/register", json={
                "identifier": email, "password": password})
            assert registered.status_code == 201, registered.text
            user_id = registered.json()["user_id"]
            assert (await alice.get("/api/auth/me")).json()["user_id"] == user_id
            duplicate = await alice.post("/api/auth/register", json={
                "identifier": email.upper(), "password": password})
            assert duplicate.status_code == 409
            session = (await alice.post("/api/session")).json()["session_id"]
            assert (await alice.get("/api/auth/history")).json() == []  # no empty chats
            response = await alice.post("/api/chat", json={
                "session_id": session, "user_id": "account_forged", "message": "quantum mechanics"})
            assert response.status_code == 200
            assert response.json()["user_id"] == user_id
            async with db.session() as neo:
                result = await neo.run(
                    "MATCH (u:User {identifier: $user_id}) "
                    "CREATE (u)-[:HAS_CONVERSATION]->"
                    "(c:Conversation {id: $conversation_id, session_id: $session_id}) "
                    "WITH c UNWIND $turns AS turn "
                    "CREATE (c)-[:HAS_MESSAGE]->(m:Message {id: turn.id, role: turn.role, "
                    "content: turn.content, timestamp: datetime(turn.timestamp)})",
                    user_id=user_id, session_id=session, conversation_id=uuid4().hex,
                    turns=[
                        {"id": uuid4().hex, "role": role, "content": text,
                         "timestamp": f"2026-01-01T00:00:0{index}Z"}
                        for index, (role, text) in enumerate([
                            ("user", "سلام"), ("assistant", "سلام!"),
                            ("user", "سفر دو روزه در طبیعت اطراف تهران"),
                            ("assistant", "برای سفر دو روزه ..."),
                        ])
                    ])
                await result.consume()
            listings = (await alice.get("/api/auth/history")).json()
            assert listings == [{"session_id": session,
                                 "title": "سفر دو روزه در طبیعت اطراف تهران"}]
            transcript = (await alice.get(f"/api/auth/history/{session}")).json()
            assert [turn["role"] for turn in transcript["messages"]] == [
                "user", "assistant", "user", "assistant"]
            assert transcript["messages"][-1]["content"] == "برای سفر دو روزه ..."

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test", trust_env=False) as stranger:
                assert (await stranger.post("/api/auth/login", json={
                    "identifier": email, "password": "incorrect"})).status_code == 401
                assert (await stranger.post("/api/chat", json={
                    "session_id": session, "message": "hello"})).status_code == 401
                # A second account cannot open the first account's chat.
                other_email = f"auth-test-{uuid4().hex}@example.org"
                other = await stranger.post("/api/auth/register", json={
                    "identifier": other_email, "password": password})
                assert other.status_code == 201
                other_id = other.json()["user_id"]
                try:
                    assert (await stranger.get("/api/auth/history")).json() == []
                    assert (await stranger.get(f"/api/auth/history/{session}")).status_code == 403
                    assert (await stranger.post("/api/chat", json={
                        "session_id": session, "message": "hello"})).status_code == 403
                finally:
                    async with db.session() as neo:
                        await neo.run("MATCH (u:User {identifier: $id}) OPTIONAL MATCH "
                                      "(s:AuthSession)-[:AUTHENTICATES]->(u) "
                                      "DETACH DELETE s, u", id=other_id)

            assert (await alice.post("/api/auth/logout")).status_code == 200
            assert (await alice.get("/api/auth/me")).status_code == 401
            signed_in = await alice.post("/api/auth/login", json={
                "identifier": email.upper(), "password": password})
            assert signed_in.status_code == 200
            assert signed_in.json()["user_id"] == user_id
            assert (await alice.get(f"/api/auth/history/{session}")).json() == transcript
    finally:
        if user_id:
            async with db.session() as neo:
                result = await neo.run(
                    "MATCH (u:User {identifier: $id})-[:HAS_CONVERSATION]->(c:Conversation)"
                    "-[:HAS_MESSAGE]->(m:Message) DETACH DELETE m", id=user_id)
                await result.consume()
                result = await neo.run(
                    "MATCH (u:User {identifier: $id})-[:HAS_CONVERSATION]->(c:Conversation) "
                    "DETACH DELETE c", id=user_id)
                await result.consume()
                await neo.run(
                    "MATCH (u:User {identifier: $id}) "
                    "OPTIONAL MATCH (s:AuthSession)-[:AUTHENTICATES]->(u) DETACH DELETE s "
                    "WITH u OPTIONAL MATCH (u)-[:OWNS_CHAT]->(c:ChatSession) "
                    "DETACH DELETE c, u", id=user_id)
        await db.close()
