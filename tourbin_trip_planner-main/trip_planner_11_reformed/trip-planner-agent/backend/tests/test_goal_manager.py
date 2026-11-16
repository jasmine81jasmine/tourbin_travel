import os
import json
from uuid import uuid4

import pytest

from src.agent.goal_manager import GoalDesign, design_goals, evaluate_plan, replan_feedback
from src.agent.goals import GoalCoverage, PlanningConstraint, PlanningSubgoal, TravelGoal
from src.memory.graph import TripGraphRepository


class Graph:
    async def get_destination_details(self, name):
        return {"categories": ["تاریخی"]} if name == "چهلستون" else None


@pytest.mark.asyncio
async def test_coverage_requires_final_stops_graph_categories_and_real_neshan_route():
    goal = TravelGoal(subgoals=[
        PlanningSubgoal(id="G1", name="تاریخ", priority="high", evidence_type="category",
                        category="تاریخی", minimum=2, source="preference"),
        PlanningSubgoal(id="G2", name="غذا", evidence_type="unverified", source="assumption"),
        PlanningSubgoal(id="G3", name="مسیر", evidence_type="route", source="explicit"),
        PlanningSubgoal(id="G4", name="چهلستون", evidence_type="destination",
                        target="چهلستون", source="explicit"),
    ])
    route = {"stops": [{"name": "چهلستون"}], "used_real_routing": False}
    coverage = await evaluate_plan(goal, [route], Graph())
    assert [c.status for c in coverage] == ["partial", "pending", "pending", "satisfied"]
    assert "غذا" not in coverage[1].activities
    assert replan_feedback(goal, coverage) is not None
    coverage = await evaluate_plan(goal, [route | {"used_real_routing": True}], Graph())
    assert coverage[2].status == "satisfied"


@pytest.mark.asyncio
async def test_independent_alternatives_do_not_combine_into_one_visit_plan():
    goal = TravelGoal(subgoals=[PlanningSubgoal(
        id="G1", name="تاریخ", evidence_type="category", category="تاریخی", minimum=2)])

    class HistoricalGraph:
        async def get_destination_details(self, name):
            return {"categories": ["تاریخی"]}

    coverage = await evaluate_plan(goal, [
        {"stops": [{"name": "الف"}], "used_real_routing": True},
        {"stops": [{"name": "ب"}], "used_real_routing": True}], HistoricalGraph())
    assert coverage[0].status == "partial"


@pytest.mark.asyncio
async def test_goal_designer_never_accepts_uncheckable_destination_or_assumed_origin_as_explicit(monkeypatch):
    from src.agent import goal_manager

    class Designer:
        async def run(self, message):
            class Result:
                output = GoalDesign(subgoals=[
                    PlanningSubgoal(id="G1", name="چهلستون", source="explicit",
                                    evidence_type="destination"),
                    PlanningSubgoal(id="G2", name="تهران", source="explicit",
                                    evidence_type="route")])
            return Result()

    monkeypatch.setattr(goal_manager, "_designer", lambda: Designer())
    designed = await design_goals(TravelGoal(), "برنامه اصفهان", "", [])
    assert designed.subgoals[0].evidence_type == "unverified"
    assert designed.subgoals[1].source == "assumption"


@pytest.mark.asyncio
async def test_goal_revision_preserves_existing_subgoals_and_considers_full_ontology(monkeypatch):
    from src.agent import goal_manager
    from types import SimpleNamespace

    ontology = [{"id": f"category:{i}", "label": "Category", "name": f"نوع {i}",
                 "description": "نوع مقصد"} for i in range(68)]
    ontology.append({"id": "facilities:high", "label": "FacilitiesLevel",
                     "name": "زیاد", "description": "سطح امکانات"})
    seen = []

    class Designer:
        async def run(self, message):
            seen.append(json.loads(message))
            return SimpleNamespace(output=GoalDesign(
                subgoals=[PlanningSubgoal(
                    id="G2", name="امکانات برای خانواده", source="explicit",
                    evidence_type="unverified",
                    related_features=["facilities:high", "fabricated:unknown"])],
                constraints=[PlanningConstraint(description="پیاده‌روی کم", kind="hard",
                                                source="explicit")]))

    monkeypatch.setattr(goal_manager, "_designer", lambda: Designer())
    original = TravelGoal(
        subgoals=[PlanningSubgoal(id="G1", name="طبیعت پاییزی", priority="high",
                                  evidence_type="category", category="طبیعت")],
        planning_constraints=[PlanningConstraint(description="پاییز", kind="hard",
                                                 source="explicit")])
    updated = await design_goals(original, "با خانواده میریم، پیاده‌روی کم", "", [],
                                 ontology)
    assert len(seen[0]["available_feature_nodes"]) == len(ontology)
    assert [s.id for s in updated.subgoals] == ["G1", "G2"]
    assert updated.subgoals[1].related_features == ["facilities:high"]
    assert {c.description for c in updated.planning_constraints} == {"پاییز", "پیاده‌روی کم"}


@pytest.mark.asyncio
async def test_explicit_trip_value_replaces_conflicting_old_feature_preference(monkeypatch):
    from src.agent import goal_manager
    from types import SimpleNamespace

    ontology = [
        {"id": "season:old", "label": "Season", "name": "پاییز"},
        {"id": "season:new", "label": "Season", "name": "بهار"},
        {"id": "triptype:camp", "label": "TripType", "name": "کمپ"},
    ]
    seen = {}

    class Designer:
        async def run(self, message):
            seen.update(json.loads(message))
            return SimpleNamespace(output=GoalDesign())

    monkeypatch.setattr(goal_manager, "_designer", lambda: Designer())
    old = TravelGoal(season="بهار", subgoals=[
        PlanningSubgoal(id="G1", name="پاییز", source="preference",
                        evidence_type="feature", feature_id="season:old"),
        PlanningSubgoal(id="G2", name="کمپ", source="preference",
                        evidence_type="feature", feature_id="triptype:camp"),
        PlanningSubgoal(id="G3", name="بهار", source="explicit",
                        evidence_type="feature", feature_id="season:new")])
    result = await design_goals(
        old, "این بار بهار بریم", "", [
            {"id": "season:old", "score": 2},
            {"id": "triptype:camp", "score": 2}], ontology)
    assert seen["user_preferences"] == [{"id": "triptype:camp", "score": 2}]
    assert seen["current_trip_feature_ids"]["Season"] == ["season:new"]
    assert [item.id for item in result.subgoals] == ["G2", "G3"]
    assert old.subgoals[0].id == "G1"  # earlier snapshots are untouched
    rejected = old.model_copy(update={"avoid": ["کمپ"]})
    without_camp = await design_goals(
        rejected, "کمپ نمی‌خواهم", "", [{"id": "triptype:camp", "score": 2}], ontology)
    assert "G2" not in [item.id for item in without_camp.subgoals]
    assert seen["user_preferences"] == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_AUTH_INTEGRATION") != "1", reason="Requires local Neo4j")
async def test_goals_are_immutable_session_scoped_revisions():
    from neo4j import AsyncGraphDatabase
    from src.config import get_settings

    settings = get_settings()
    db = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))
    user, other, session_id = (f"goal_test_{uuid4().hex}" for _ in range(3))
    repo = TripGraphRepository(db)
    try:
        await repo.ensure_goal_revisions()
        async with db.session() as session:
            await (await session.run(
                "CREATE (u:User {identifier:$user})-[:OWNS_CHAT]->"
                "(:ChatSession {session_id:$sid}) CREATE (:User {identifier:$other})",
                user=user, other=other, sid=session_id)).consume()
        assert await repo.upsert_travel_goal(session_id, user, {"objective": "اصفهان"}) == 1
        assert await repo.upsert_travel_goal(session_id, user, {"objective": "شیراز"}) == 2
        assert (await repo.get_travel_goal(session_id, user))["objective"] == "شیراز"
        assert await repo.get_travel_goal(session_id, other) is None
        with pytest.raises(ValueError):
            await repo.upsert_travel_goal(session_id, other, {"objective": "غیرمجاز"})
        async with db.session() as session:
            row = await (await session.run(
                "MATCH (u:User {identifier:$user})-[:HAS_GOAL]->(a:TravelGoal)"
                "-[:REVISED_TO]->(b:TravelGoal) RETURN a.state_json AS before,"
                " b.state_json AS after, a.revision AS r1, b.revision AS r2",
                user=user)).single()
        assert row["r1"] == 1 and row["r2"] == 2
        assert "اصفهان" in row["before"] and "شیراز" in row["after"]
        assert json.loads(row["before"])["revision"] == 1
        assert json.loads(row["after"])["revision"] == 2
        async with db.session() as session:
            count = await (await session.run(
                "MATCH (:ChatSession {session_id:$sid})-[:HAS_GOAL_REVISION]->"
                "(g:TravelGoal) RETURN count(g) AS count", sid=session_id)).single()
        assert count["count"] == 2
    finally:
        async with db.session() as session:
            await (await session.run(
                "MATCH (u:User) WHERE u.identifier IN [$user,$other] "
                "OPTIONAL MATCH (u)-[:HAS_GOAL|OWNS_CHAT]->(n) "
                "DETACH DELETE u,n", user=user, other=other)).consume()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_AUTH_INTEGRATION") != "1", reason="Requires local Neo4j")
async def test_neo4j_goal_state_nodes_track_each_revision_and_owner():
    from neo4j import AsyncGraphDatabase
    from src.config import get_settings

    settings = get_settings()
    db = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))
    user, session_id = (f"progress_test_{uuid4().hex}" for _ in range(2))
    repo = TripGraphRepository(db)
    try:
        await repo.ensure_goal_revisions()
        base = TravelGoal(subgoals=[PlanningSubgoal(id="G1", name="طبیعت")])
        assert await repo.upsert_travel_goal(session_id, user, base.model_dump()) == 1
        base.goal_coverage = [GoalCoverage(goal_id="G1", status="partial", activities=["جنگل"])]
        base.last_observation_tool = "search_destinations"
        assert await repo.upsert_travel_goal(session_id, user, base.model_dump()) == 2
        async with db.session() as session:
            row = await (await session.run(
                "MATCH (:User {identifier:$user})-[:HAS_GOAL]->"
                "(:TravelGoal)-[:HAS_GOAL_STATE]->(a:GoalState {goal_id:'G1'})"
                "-[:REVISED_TO]->(b:GoalState {goal_id:'G1'}) "
                "RETURN a.status AS before, b.status AS after, "
                "b.observed_tool AS tool, b.evidence_json AS evidence",
                user=user)).single()
        assert row["before"] == "pending" and row["after"] == "partial"
        assert row["tool"] == "search_destinations"
        assert json.loads(row["evidence"]) == ["جنگل"]
    finally:
        async with db.session() as session:
            await (await session.run(
                "MATCH (u:User {identifier:$user}) "
                "OPTIONAL MATCH (u)-[:HAS_GOAL]->(g:TravelGoal) "
                "OPTIONAL MATCH (g)-[:HAS_GOAL_STATE]->(s:GoalState) "
                "DETACH DELETE s,g,u", user=user)).consume()
        await db.close()
