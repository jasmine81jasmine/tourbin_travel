"""Feature-vocabulary, evidence and recommendation regression tests."""

import os
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from neo4j.time import DateTime
from pydantic import ValidationError

from src.agent.variety import vary_candidates
from src.memory import preference_extractor as extractor
from src.memory import preference_graph as graph


def test_controlled_bins_and_stable_normalized_ids():
    assert graph.feature_id("Category", "طبيعت") == graph.feature_id("Category", "طبیعت")
    assert graph.feature_id("Category", "طبیعت") != graph.feature_id("Season", "طبیعت")
    assert graph.travel_time_bucket(2) == "کوتاه"
    assert graph.travel_time_bucket(2.1) == "متوسط"
    assert graph.travel_time_bucket(5.1) == "طولانی"
    assert graph.travel_time_bucket(None) is None
    assert graph.distance_bucket(101) == "متوسط"
    assert graph.facilities_bucket("دارد") is None
    assert graph.facilities_bucket("دارد(متوسط)") == "متوسط"
    assert graph.road_bucket("آِسان") == "آسان"
    assert graph.road_bucket("سخت") == "دشوار"


def test_llm_output_rejects_invalid_evidence_and_duplicate_nodes():
    good = {"node_id": "category:one", "positive_increment": 1,
            "negative_increment": 0, "positive_evidence_message_ids": ["m1"],
            "negative_evidence_message_ids": []}
    assert graph.PreferenceBatch.model_validate({"updates": [good]}).updates[0].positive_increment == 1
    for bad in ({**good, "positive_increment": 2},
                {**good, "positive_evidence_message_ids": []},
                {**good, "negative_increment": 1, "negative_evidence_message_ids": ["m1"]},
                {**good, "unexpected": "from LLM"}):
        with pytest.raises(ValidationError):
            graph.PreferenceBatch.model_validate({"updates": [bad]})
    with pytest.raises(ValidationError):
        graph.PreferenceBatch.model_validate({"updates": [good, good]})


def test_explicit_filters_and_tiers_remain_prioritized_over_soft_preferences():
    rows = [
        {"id": "no", "name": "unliked", "source_is_osm": False,
         "user_rating_score": 2, "preference_score": -2},
        {"id": "yes", "name": "liked", "source_is_osm": False,
         "user_rating_score": 2, "preference_score": 3},
        {"id": "osm", "name": "osm", "source_is_osm": True,
         "user_rating_score": 3, "preference_score": 100},
    ]
    result = vary_candidates(rows, {"ids": set(), "names": set()}, "u", "s", 3)
    assert result[0]["id"] == "yes"
    assert result[-1]["id"] == "osm"
    assert len(result) == 3  # no hard exclusion based on preferences


@pytest.mark.asyncio
async def test_llm_gets_catalog_and_returns_only_json_increments(monkeypatch):
    captured = {}

    class MockAgent:
        def __init__(self, model, system_prompt):
            assert "other" in system_prompt and "USER" in system_prompt

        async def run(self, prompt):
            captured.update(json.loads(prompt))
            return SimpleNamespace(output='```json\n{"updates":[]}\n```')

    monkeypatch.setattr(extractor, "Agent", MockAgent)
    monkeypatch.setattr(extractor, "_build_model", lambda settings: object())
    batch = await extractor.propose_updates(
        [{"message_id": "m1", "role": "user", "content": "طبیعت دوست دارم"}],
        [{"id": "existing", "score": 1}],
        [{"id": "category:nature", "name": "طبیعت", "label": "Category"}])
    assert not batch.updates
    assert captured["LATEST CONVERSATION"][0]["message_id"] == "m1"
    assert captured["EXISTING USER PREFERENCES"][0]["score"] == 1
    assert captured["AVAILABLE PREFERENCE NODES"][0]["name"] == "طبیعت"


@pytest.mark.asyncio
async def test_extractor_serializes_neo4j_preference_timestamps(monkeypatch):
    captured = {}

    class MockAgent:
        def __init__(self, model, system_prompt):
            pass

        async def run(self, prompt):
            captured.update(json.loads(prompt))
            return SimpleNamespace(output='{"updates":[]}')

    monkeypatch.setattr(extractor, "Agent", MockAgent)
    monkeypatch.setattr(extractor, "_build_model", lambda settings: object())
    updated_at = DateTime(2026, 11, 14, 12, 30, 0)
    batch = await extractor.propose_updates(
        [{"message_id": "m1", "role": "user", "content": "طبیعت دوست دارم"}],
        [{"id": "category:nature", "score": 1, "updated_at": updated_at}],
        [{"id": "category:nature", "name": "طبیعت", "label": "Category"}])
    assert not batch.updates
    assert captured["EXISTING USER PREFERENCES"][0]["updated_at"] == str(updated_at)


@pytest.mark.asyncio
async def test_finalize_api_requires_ownership_and_returns_update_count(monkeypatch):
    from src.main import app
    from src.api.routes import auth

    async def identity():
        return auth.Identity("account_a", "tokenhash")

    async def claim(user_id, session_id, *, create=False):
        if session_id != "owned":
            raise HTTPException(403, "Not your chat")

    async def finalize(db, user_id, session_id):
        assert user_id == "account_a" and session_id == "owned"
        return 2

    app.dependency_overrides[auth.require_user] = identity
    monkeypatch.setattr(auth, "claim_session", claim)
    monkeypatch.setattr(auth, "driver", lambda: object())
    monkeypatch.setattr(auth, "finalize_preferences", finalize)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test", trust_env=False) as client:
            assert (await client.post("/api/auth/history/foreign/finalize")).status_code == 403
            assert (await client.post("/api/auth/history/owned/finalize")).json() == {"updated": 2}
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_extractor_passes_only_owned_messages_and_controlled_nodes(monkeypatch):
    async def messages(driver, user_id, session_id):
        assert user_id == "account_a" and session_id == "session_a"
        return [{"message_id": "m1", "role": "user", "content": "من عاشق طبیعتم"}]

    async def features(driver):
        return [{"id": graph.feature_id("Category", "طبیعت"),
                 "label": "Category", "name": "طبیعت"}]

    async def current(driver, user_id):
        return []

    async def llm(messages, current, features):
        assert messages[0]["message_id"] == "m1" and features[0]["name"] == "طبیعت"
        return graph.PreferenceBatch(updates=[])

    async def apply(driver, user_id, session_id, batch):
        return len(batch.updates)

    async def read_revision(driver, user_id, session_id):
        return 3

    async def mark_checked(driver, user_id, session_id, revision):
        assert (user_id, session_id, revision) == ("account_a", "session_a", 3)

    monkeypatch.setattr(extractor, "session_messages", messages)
    monkeypatch.setattr(extractor, "available_features", features)
    monkeypatch.setattr(extractor, "user_preferences", current)
    monkeypatch.setattr(extractor, "propose_updates", llm)
    monkeypatch.setattr(extractor, "apply_updates", apply)
    monkeypatch.setattr(extractor, "read_revision", read_revision)
    monkeypatch.setattr(extractor, "mark_checked", mark_checked)
    assert await extractor.finalize_preferences(object(), "account_a", "session_a") == 0


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_AUTH_INTEGRATION") != "1", reason="Requires local Neo4j")
async def test_live_ontology_and_atomic_idempotent_evidence():
    from neo4j import AsyncGraphDatabase
    from src.config import get_settings

    settings = get_settings()
    db = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=(
        settings.neo4j_username, settings.neo4j_password.get_secret_value()))
    user = f"account_pref_test_{uuid4().hex}"
    other = f"account_pref_test_{uuid4().hex}"
    session_id = f"session_pref_test_{uuid4().hex}"
    next_session = f"session_pref_test_{uuid4().hex}"
    conversation = str(uuid4())
    next_conversation = str(uuid4())
    m1, m2, m3 = str(uuid4()), str(uuid4()), str(uuid4())
    try:
        await graph.ensure_preference_ontology(db)
        await graph.ensure_preference_ontology(db)
        catalog = await graph.available_features(db)
        nature = next(n for n in catalog if n["label"] == "Category" and n["name"] == "طبیعت")
        road = next(n for n in catalog if n["label"] == "RoadType" and n["name"] == "دشوار")
        async with db.session() as neo:
            result = await neo.run(
                "CREATE (u:User {identifier: $user}), (other:User {identifier: $other}) "
                "CREATE (u)-[:OWNS_CHAT]->(chat:ChatSession {session_id: $sid}) "
                "CREATE (u)-[:HAS_CONVERSATION]->"
                "(conv:Conversation {id: $cid, session_id: $sid}) "
                "CREATE (conv)-[:HAS_MESSAGE]->(m1:Message {id: $m1, role: 'user', "
                "content: 'من عاشق طبیعت هستم'}) "
                "CREATE (conv)-[:HAS_MESSAGE]->(m2:Message {id: $m2, role: 'user', "
                "content: 'از مسیرهای سخت خوشم نمیاد'})",
                user=user, other=other, sid=session_id, cid=conversation, m1=m1, m2=m2)
            await result.consume()
        batch = graph.PreferenceBatch(updates=[
            graph.PreferenceUpdate(node_id=nature["id"], positive_increment=1,
                negative_increment=0, positive_evidence_message_ids=[m1]),
            graph.PreferenceUpdate(node_id=road["id"], positive_increment=0,
                negative_increment=1, negative_evidence_message_ids=[m2]),
        ])
        assert await graph.apply_updates(db, user, session_id, batch) == 2
        assert await graph.apply_updates(db, user, session_id, batch) == 0
        prefs = {p["id"]: p for p in await graph.user_preferences(db, user)}
        assert prefs[nature["id"]]["score"] == 1
        assert prefs[road["id"]]["score"] == -1
        assert isinstance(prefs[nature["id"]]["updated_at"], str)
        json.dumps(prefs, ensure_ascii=False)
        assert await graph.user_preferences(db, other) == []
        async with db.session() as neo:
            result = await neo.run(
                "MATCH (u:User {identifier: $user}) "
                "CREATE (u)-[:OWNS_CHAT]->(:ChatSession {session_id: $sid}) "
                "CREATE (u)-[:HAS_CONVERSATION]->"
                "(c:Conversation {id: $cid, session_id: $sid}) "
                "CREATE (c)-[:HAS_MESSAGE]->(:Message {id: $mid, role: 'user', "
                "content: 'هنوز عاشق طبیعت هستم'})",
                user=user, sid=next_session, cid=next_conversation, mid=m3)
            await result.consume()
        second = graph.PreferenceBatch(updates=[
            graph.PreferenceUpdate(node_id=nature["id"], positive_increment=1,
                negative_increment=0, positive_evidence_message_ids=[m3])])
        assert await graph.apply_updates(db, user, next_session, second) == 1
        assert await graph.apply_updates(db, user, next_session, second) == 0
        nature_pref = next(p for p in await graph.user_preferences(db, user) if p["id"] == nature["id"])
        assert (nature_pref["positive_count"], nature_pref["negative_count"], nature_pref["score"]) == (2, 0, 2)
        async with db.session() as neo:
            row = await (await neo.run(
                "MATCH (n {preference_id: $id})-[e:HAS_EVIDENCE]->(m:Message {id: $m}) "
                "RETURN count(e) AS count", id=nature["id"], m=m1)).single()
            assert row["count"] == 1
        invalid = graph.PreferenceBatch(updates=[
            batch.updates[0],
            graph.PreferenceUpdate(node_id=road["id"], positive_increment=1,
                negative_increment=0, positive_evidence_message_ids=["someone-elses-message"]),
        ])
        with pytest.raises(ValueError):
            await graph.apply_updates(db, user, session_id, invalid)
        assert next(p for p in await graph.user_preferences(db, user)
                    if p["id"] == nature["id"])["score"] == 2
        assert await graph.session_messages(db, other, session_id) == []
        # Migration did not remove the original scalar properties.
        async with db.session() as neo:
            row = await (await neo.run(
                "MATCH (d:Destination)-[:HAS_TRAVEL_TIME]->(:TravelTime) "
                "WHERE d.travel_time_hours IS NOT NULL "
                "RETURN d.travel_time_hours AS hours LIMIT 1")).single()
            assert row is not None and row["hours"] is not None
    finally:
        async with db.session() as neo:
            for query in (
                "MATCH (m:Message) WHERE m.id IN $messages DETACH DELETE m",
                "MATCH (c:Conversation {id: $cid}) DETACH DELETE c",
                "MATCH (c:ChatSession {session_id: $sid}) DETACH DELETE c",
                "MATCH (u:User) WHERE u.identifier IN $users DETACH DELETE u",
            ):
                result = await neo.run(query, messages=[m1, m2, m3], cid=conversation,
                                       sid=session_id, users=[user, other])
                await result.consume()
            result = await neo.run(
                "MATCH (c:Conversation {id: $cid}) DETACH DELETE c",
                cid=next_conversation)
            await result.consume()
            result = await neo.run(
                "MATCH (c:ChatSession {session_id: $sid}) DETACH DELETE c",
                sid=next_session)
            await result.consume()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_PREFERENCE_LLM") != "1",
                    reason="Opt-in real provider request and local Neo4j")
async def test_real_llm_extracts_explicit_preference():
    from neo4j import AsyncGraphDatabase
    from src.config import get_settings

    settings = get_settings()
    db = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=(
        settings.neo4j_username, settings.neo4j_password.get_secret_value()))
    user = f"account_pref_llm_test_{uuid4().hex}"
    session_id = f"session_pref_llm_test_{uuid4().hex}"
    conversation = str(uuid4())
    message = str(uuid4())
    try:
        await graph.ensure_preference_ontology(db)
        async with db.session() as neo:
            result = await neo.run(
                "CREATE (u:User {identifier: $user}) "
                "CREATE (u)-[:OWNS_CHAT]->(:ChatSession {session_id: $sid}) "
                "CREATE (u)-[:HAS_CONVERSATION]->"
                "(c:Conversation {id: $cid, session_id: $sid}) "
                "CREATE (c)-[:HAS_MESSAGE]->(:Message {id: $mid, role: 'user', "
                "content: 'من عاشق طبیعت‌گردی هستم و معمولاً سفرهای طبیعت‌گردی را دوست دارم.'})",
                user=user, sid=session_id, cid=conversation, mid=message)
            await result.consume()
        assert await extractor.finalize_preferences(db, user, session_id) >= 1
        assert await extractor.finalize_preferences(db, user, session_id) == 0
        prefs = await graph.user_preferences(db, user)
        nature_ids = {node["id"] for node in await graph.available_features(db)
                      if node["label"] == "Category" and "طبیعت" in node["name"]}
        assert any(p["id"] in nature_ids and p["score"] == 1 for p in prefs), prefs
    finally:
        async with db.session() as neo:
            for query in (
                "MATCH (m:Message {id: $mid}) DETACH DELETE m",
                "MATCH (c:Conversation {id: $cid}) DETACH DELETE c",
                "MATCH (c:ChatSession {session_id: $sid}) DETACH DELETE c",
                "MATCH (u:User {identifier: $user}) DETACH DELETE u",
            ):
                result = await neo.run(query, mid=message, cid=conversation,
                                       sid=session_id, user=user)
                await result.consume()
        await db.close()
