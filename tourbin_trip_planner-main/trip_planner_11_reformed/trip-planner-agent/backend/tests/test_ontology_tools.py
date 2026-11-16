"""Ontology-edge search and evidence must never substitute vocabulary for facts."""

import asyncio
import json
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.agent import goal_manager
from src.agent.goal_manager import GoalDesign, design_goals, evaluate_plan
from src.agent.goal_progress import observed_result
from src.agent.goals import PlanningSubgoal, TravelGoal
from src.agent.tools import search_destinations_by_features
from src.memory.graph import TripGraphRepository
from src.memory.preference_graph import DESTINATION_FEATURE_RELATIONS


@pytest.mark.asyncio
async def test_feature_query_uses_bound_ontology_ids_and_only_known_relationships():
    repo = TripGraphRepository(None)
    seen = []

    async def read(query, **params):
        seen.append((query, params))
        return [{"id": "d1", "name": "جنگل", "matched_feature_ids": ["facilities:1"]}]

    repo._read = read
    rows = await repo.search_destinations_by_features(
        ["facilities:1", "facilities:1"], location=["گیلان"], match_all=True)
    assert rows[0]["matched_feature_ids"] == ["facilities:1"]
    query, params = seen[0]
    assert params["feature_ids"] == ["facilities:1"]
    assert params["allowed_pairs"] == [
        f"{relation}:{label}" for label, relation
        in DESTINATION_FEATURE_RELATIONS.items()]
    assert params["location"] == ["گیلان"] and params["match_all"]
    assert "f.preference_id IN $feature_ids" in query
    assert "size(matched_feature_ids) = size($feature_ids)" in query
    assert "facilities:1" not in query  # never interpolate model-supplied IDs


@pytest.mark.asyncio
async def test_feature_tool_expands_region_and_never_treats_unlinked_node_as_match():
    class Graph:
        async def search_destinations_by_features(self, ids, location, match_all, limit):
            assert ids == ["f1", "f2"]
            assert location == ["گیلان", "مازندران", "گلستان"]
            assert match_all and limit == 12
            return [{"id": "eligible", "name": "جنگل",
                     "matched_feature_ids": ["f1", "f2"]}]

        async def get_avoided_destination_ids(self, user):
            return []

    ctx = SimpleNamespace(deps=SimpleNamespace(
        graph=Graph(), user_id="user", session_id="session",
        travel_goal=TravelGoal(region="شمال"), current_query="شمال",
        previous_recommendations={"ids": set(), "names": set()}))
    result = await search_destinations_by_features(ctx, ["f1", "f2"], limit=4)
    assert result["candidates"][0]["matched_feature_ids"] == ["f1", "f2"]
    assert result["match_all"] is True
    assert (await search_destinations_by_features(ctx, []))["error"]


@pytest.mark.asyncio
async def test_designer_accepts_only_destination_linked_feature_ids(monkeypatch):
    class Designer:
        async def run(self, prompt):
            nodes = json.loads(prompt)["available_feature_nodes"]
            assert nodes[0]["destination_linked"]
            assert not nodes[1]["destination_linked"]
            return SimpleNamespace(output=GoalDesign(subgoals=[
                PlanningSubgoal(id="G1", name="امکانات کلی", evidence_type="feature",
                                feature_id="facilities:high"),
                PlanningSubgoal(id="G2", name="خلوت", evidence_type="feature",
                                feature_id="environment:quiet"),
            ]))

    monkeypatch.setattr(goal_manager, "_designer", lambda: Designer())
    designed = await design_goals(TravelGoal(), "سفر آرام", "", [], [
        {"id": "facilities:high", "label": "FacilitiesLevel", "name": "زیاد"},
        {"id": "environment:quiet", "label": "Environment", "name": "خلوت"}])
    assert designed.subgoals[0].feature_id == "facilities:high"
    assert designed.subgoals[1].evidence_type == "unverified"
    assert designed.subgoals[1].feature_id is None


@pytest.mark.asyncio
async def test_feature_goal_requires_graph_edge_on_final_route_not_candidate_only():
    class Graph:
        async def get_destination_details(self, name):
            return {"name": name, "categories": ["طبیعت"]}

        async def get_destination_features(self, name):
            return ([{"id": "road:easy"}] if name == "جنگل" else [])

    goal = TravelGoal(subgoals=[PlanningSubgoal(
        id="G1", name="جاده آسان", evidence_type="feature",
        feature_id="road:easy", priority="high")])
    ctx = SimpleNamespace(deps=SimpleNamespace(
        travel_goal=goal, turn_action="plan", graph=None,
        user_id="user", session_id="session", observed_tools=[],
        progress_lock=asyncio.Lock()))
    await observed_result(ctx, "search_destinations_by_features", {
        "candidates": [{"name": "جنگل", "matched_feature_ids": ["road:easy"]}]})
    assert goal.goal_coverage[0].status == "partial"
    rejected = await evaluate_plan(goal, [{"stops": [{"name": "شهر"}]}], Graph())
    assert rejected[0].status == "pending"
    accepted = await evaluate_plan(goal, [{"stops": [{"name": "جنگل"}]}], Graph())
    assert accepted[0].status == "satisfied"
    assert (await evaluate_plan(goal, [{"stops": [{"name": "جنگل"}]}]))[0].status == "pending"


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_AUTH_INTEGRATION") != "1", reason="Requires local Neo4j")
async def test_ontology_edge_tool_against_local_neo4j():
    from neo4j import AsyncGraphDatabase
    from src.config import get_settings

    settings = get_settings()
    db = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))
    marker = f"ontology_test_{uuid4().hex}"
    repo = TripGraphRepository(db)
    try:
        async with db.session() as session:
            await (await session.run(
                "CREATE (d:Destination {id:$id, name:$id})"
                "-[:HAS_ROAD_TYPE]->(:RoadType {preference_key:$id, "
                "preference_id:$feature, name:'آسان'})",
                id=marker, feature=f"roadtype:{marker}")).consume()
        rows = await repo.search_destinations_by_features([f"roadtype:{marker}"])
        assert any(row["id"] == marker for row in rows)
        assert any(item["id"] == f"roadtype:{marker}"
                   for item in await repo.get_destination_features(marker))
        assert await repo.search_destinations_by_features(
            [f"roadtype:{marker}", f"environment:{marker}"], match_all=True) == []
        goal = TravelGoal(subgoals=[PlanningSubgoal(
            id="G1", name="جادهٔ آسان", evidence_type="feature",
            feature_id=f"roadtype:{marker}",
            related_features=[f"roadtype:{marker}"])])
        await repo.upsert_travel_goal(marker, marker, goal.model_dump())
        async with db.session() as session:
            link = await (await session.run(
                "MATCH (:User {identifier:$id})-[:HAS_GOAL]->"
                "(:TravelGoal)-[:HAS_GOAL_STATE]->(s:GoalState)"
                "-[:USES_FEATURE]->(f:RoadType {preference_id:$feature}) "
                "RETURN s.feature_id AS goal_feature",
                id=marker, feature=f"roadtype:{marker}")).single()
        assert link["goal_feature"] == f"roadtype:{marker}"
    finally:
        async with db.session() as session:
            await (await session.run(
                "MATCH (u:User {identifier:$id}) "
                "OPTIONAL MATCH (u)-[:HAS_GOAL]->(g:TravelGoal) "
                "OPTIONAL MATCH (g)-[:HAS_GOAL_STATE]->(s:GoalState) "
                "DETACH DELETE s,g,u", id=marker)).consume()
            await (await session.run(
                "MATCH (d:Destination {id:$id})-[r:HAS_ROAD_TYPE]->(f:RoadType) "
                "DELETE r,d,f", id=marker)).consume()
        await db.close()
