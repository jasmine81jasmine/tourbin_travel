"""OnGoal-style conversational goals remain private and session-scoped."""

import json
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.agent import conversation_pipeline
from src.agent.conversation_pipeline import evaluate_response, infer_and_merge
from src.agent.goal_manager import retire_replaced_subgoals
from src.agent.goals import ConversationalGoal, PlanningSubgoal, TravelGoal
from src.agent.turns import TurnDecision, apply_goal_updates, current_trip_query
from src.agent.feasibility import (_discovery_layers, _multiday_candidates,
                                   _render_multiday_plans)
from src.memory.graph import TripGraphRepository
from src.memory.preference_graph import feature_id


def intent(message, *, action="plan", updates=None, scope="trip", kind="request",
           merge_operations=None):
    updates = updates or {}
    return TurnDecision(
        action=action, updates=updates,
        inferred_clauses=[{"clause": message, "summary": message, "kind": kind,
                           "scope": scope, "dimensions": list(updates)}],
        merge_operations=merge_operations or [])


def test_infer_merge_keeps_unrelated_goals_replaces_contradictions_and_composes_compatible_ones():
    first = "سفر دو نفره پاییزی می‌خوام"
    goal = infer_and_merge(
        TravelGoal(), first, intent(first, updates={"season": "پاییز", "group_size": 2}))
    goal = apply_goal_updates(goal, {"season": "پاییز", "group_size": 2})
    assert [(g.id, g.clause, g.scope) for g in goal.conversational_goals] == [
        ("C1", first, "trip")]
    assert goal.goal_merge_events[0].operation == "add"

    walk = "پیاده‌روی هم می‌خوایم"
    walking = intent(walk, updates={"activities": ["پیاده‌روی"]})
    updated = infer_and_merge(goal, walk, walking)
    updated = apply_goal_updates(updated, walking.updates)
    assert updated.season == "پاییز" and updated.group_size == 2
    assert updated.conversational_goals[0].status == "active"
    assert any(e.operation == "keep" and e.old_ids == ["C1"]
               for e in updated.goal_merge_events)
    assert goal.activities == []  # snapshots are not mutated

    spring = "این بار بهار می‌ریم"
    decision = intent(spring, updates={"season": "بهار"},
                      merge_operations=[{"operation": "replace", "old_ids": ["C1"],
                                         "new_index": 0}])
    changed = apply_goal_updates(infer_and_merge(updated, spring, decision), decision.updates)
    assert changed.season == "بهار" and changed.group_size == 2
    assert changed.activities == ["پیاده‌روی"]
    assert changed.conversational_goals[0].status == "superseded"
    assert changed.conversational_goals[-2].parents == ["C1"]
    assert changed.conversational_goals[-1].dimensions == ["group_size"]
    assert changed.goal_merge_events[0].operation == "replace"
    assert updated.conversational_goals[0].status == "active"

    similar = "بهار و پیاده‌روی را نگه دار"
    same = intent(similar, updates={"season": "بهار"})
    combined = infer_and_merge(changed, similar, same)
    assert combined.goal_merge_events[0].operation == "combine"
    assert combined.conversational_goals[-1].parents == [changed.conversational_goals[-2].id]


def test_questions_do_not_turn_assistant_suggestions_into_trip_constraints():
    original = TravelGoal(season="پاییز", destination_names=["جنگل الف"])
    question = "جنگل ب چطوره؟"
    next_goal = infer_and_merge(
        original, question, intent(question, action="answer", scope="turn", kind="question"))
    assert next_goal.conversational_goals[-1].scope == "turn"
    assert next_goal.destination_names == ["جنگل الف"]
    assert next_goal.season == "پاییز"
    assert next_goal.conversational_goals[-1].clause == question

    # The model cannot invent a "verbatim" user clause or retire an unrelated
    # goal just by supplying a bogus merge instruction.
    attempted = TurnDecision(
        action="answer", inferred_clauses=[{
            "clause": "کمپ می‌خواهم", "kind": "request", "scope": "trip",
            "summary": "کمپ", "dimensions": ["trip_types"]}],
        merge_operations=[{"operation": "replace", "old_ids": ["C1"], "new_index": 0}])
    safe = infer_and_merge(next_goal, "چطوری برم؟", attempted)
    assert safe.conversational_goals[-1].clause == "چطوری برم؟"
    assert safe.conversational_goals[0].status == "active"


def test_combine_retains_independent_parts_of_a_compound_goal():
    original = "سفر دو نفره بهاری به جنگل الف می‌خواهم"
    prior_updates = {"season": "بهار", "group_size": 2,
                     "destination_names": ["جنگل الف"]}
    prior = apply_goal_updates(
        infer_and_merge(TravelGoal(), original, intent(original, updates=prior_updates)),
        prior_updates)
    followup = "همان سفر را بدون کمپ می‌خواهم"
    decision = intent(followup, updates={"avoid": ["کمپ"], "season": "بهار"},
                      merge_operations=[{"operation": "combine", "old_ids": ["C1"],
                                         "new_index": 0}])
    changed = apply_goal_updates(infer_and_merge(prior, followup, decision), decision.updates)
    assert changed.goal_merge_events[0].operation == "combine"
    active_dimensions = {dimension for item in changed.conversational_goals
                         if item.status == "active" for dimension in item.dimensions}
    assert {"avoid", "season", "group_size", "destination_names"} <= active_dimensions
    assert changed.group_size == 2 and changed.destination_names == ["جنگل الف"]
    assert changed.avoid == ["کمپ"]


@pytest.mark.asyncio
async def test_evaluation_quotes_must_exist_and_ignored_goals_remain_visible_in_private_history(monkeypatch):
    class Evaluator:
        async def run(self, payload):
            data = json.loads(payload)
            assert {g["id"] for g in data["active_goals"]} == {"C1", "C2"}
            return SimpleNamespace(output=json.dumps({"evaluations": [
                {"goal_id": "C1", "category": "confirm", "explanation": "answered",
                 "examples": ["جنگل الف در دامنه کوه است"]},
                {"goal_id": "C2", "category": "confirm", "explanation": "invented quote",
                 "examples": ["جاده واقعی تأیید شده است"]}]}, ensure_ascii=False))

    monkeypatch.setattr(conversation_pipeline, "_evaluator", lambda: Evaluator())
    goal = TravelGoal(destination_names=["جنگل الف"], conversational_goals=[
        ConversationalGoal(id="C1", clause="جنگل الف چطوره؟", summary="جنگل",
                           kind="question", scope="turn"),
        ConversationalGoal(id="C2", clause="برای جنگل الف برنامه بده", summary="برنامه",
                           kind="request", scope="trip", dimensions=["destination_names"]),
    ])
    checked = await evaluate_response(goal, "جنگل الف در دامنه کوه است.")
    assert [(e.goal_id, e.category) for e in checked.conversational_evaluations] == [
        ("C1", "confirm"), ("C2", "ignore")]
    assert checked.conversational_goals[0].status == "addressed"
    assert checked.conversational_goals[1].status == "active"
    assert "جاده واقعی" not in checked.model_dump_json()


def test_active_query_replaces_stale_request_instead_of_searching_old_goal():
    current = TravelGoal(objective="می‌خوام پاییز کمپ کنم", season="بهار",
                         trip_types=[], activities=["پیاده‌روی"], group_size=2,
                         destination_names=["جنگل ب"], avoid=["کمپ"])
    query = current_trip_query(current, "این بار جنگل ب می‌ریم")
    assert "پاییز" not in query and "نوع سفر: کمپ" not in query
    assert "فصل: بهار" in query and "تعداد نفرات: 2" in query
    assert "پرهیز: کمپ" in query


def test_numeric_duration_does_not_discard_other_live_turn_constraints():
    goal = apply_goal_updates(
        TravelGoal(), {"duration": 1, "season": "پاییز", "group_size": 2,
                       "destination_names": ["جنگل دوهزار", "جنگل سه هزار"]})
    assert goal.duration == "1 روز"
    assert goal.season == "پاییز" and goal.group_size == 2
    assert goal.destination_names == ["جنگل دوهزار", "جنگل سه هزار"]
    with_invalid = apply_goal_updates(goal, {"duration": {"invalid": True},
                                             "season": "بهار"})
    assert with_invalid.duration == "1 روز" and with_invalid.season == "بهار"


def test_explicit_new_trip_destination_retires_old_subgoal_without_deleting_snapshot():
    old = TravelGoal(destination_names=["روستای الف"], season="پاییز", subgoals=[
        PlanningSubgoal(id="G1", name="دیدن روستای الف", source="explicit",
                        evidence_type="destination", target="روستای الف"),
        PlanningSubgoal(id="G2", name="پیاده‌روی", source="explicit",
                        evidence_type="unverified")])
    new = old.model_copy(deep=True)
    new.destination_names = ["روستای ب"]
    new = retire_replaced_subgoals(old, new, {"destination_names": ["روستای ب"]}, [])
    assert [g.id for g in new.subgoals] == ["G2"]
    assert [g.id for g in old.subgoals] == ["G1", "G2"]


def test_explicit_avoid_replaces_prior_camping_requirement_and_search_context():
    initial = "می‌خوام کمپ کنم"
    old = infer_and_merge(TravelGoal(subgoals=[
        PlanningSubgoal(id="G1", name="تجربه کمپ", source="explicit",
                        evidence_type="unverified")]),
        initial, intent(initial, updates={"trip_types": ["کمپ"]}))
    old = apply_goal_updates(old, {"trip_types": ["کمپ"]})
    correction = "کمپ نمی‌خوام"
    decision = intent(correction, updates={"avoid": ["کمپ"]},
                      merge_operations=[{"operation": "replace", "old_ids": ["C1"], "new_index": 0}])
    changed = apply_goal_updates(infer_and_merge(old, correction, decision), decision.updates)
    changed.objective = current_trip_query(changed, correction)
    assert changed.conversational_goals[0].status == "superseded"
    assert changed.trip_types == [] and changed.avoid == ["کمپ"]
    assert "campground" not in _discovery_layers(changed)
    pruned = retire_replaced_subgoals(
        old, changed, decision.updates, features=[])
    assert pruned.subgoals == [] and old.subgoals[0].name == "تجربه کمپ"


@pytest.mark.asyncio
async def test_verified_drive_time_cannot_override_explicit_vehicle_constraint():
    plan = {"origin": {"name": "تهران"}, "stops": [
        {"name": "روستای سواری", "leg_distance_km_from_previous": 40,
         "leg_duration_hours_from_previous": 1},
        {"name": "دریاچه آفرود", "leg_distance_km_from_previous": 20,
         "leg_duration_hours_from_previous": 1}],
        "return_leg": {"leg_distance_km_from_previous": 50,
                       "leg_duration_hours_from_previous": 1},
        "total_distance_km": 110, "total_duration_hours": 3}

    class Graph:
        async def get_destination_details(self, name):
            return {"name": name, "vehicles": ["سواری"] if name == "روستای سواری"
                    else ["آفرود"], "road_type": "آسان" if name == "روستای سواری"
                    else "سخت"}

    maps = []
    reply, itinerary = await _render_multiday_plans(
        TravelGoal(duration="2 روز", transport="ماشین معمولی"),
        [plan], [], Graph(), None, maps)
    assert "تأیید نشد" in reply and itinerary is None
    assert maps == []


@pytest.mark.asyncio
async def test_explicit_vehicle_limits_discovery_to_graph_verified_destinations():
    offroad = {"id": "offroad", "name": "دره الف", "latitude": 35.8, "longitude": 51.3}
    ordinary = {"id": "ordinary", "name": "پارک ب", "latitude": 35.9, "longitude": 51.4}

    class Graph:
        async def search_destinations_by_features(self, ids, **kwargs):
            assert ids == [feature_id("Vehicle", "آفرود")]
            return [offroad]

        async def search_destinations(self, **kwargs):
            return [ordinary, offroad]

    goal = TravelGoal(duration="3 روز", transport="آفرود",
                      objective="سفر جوانانه با خودروی آفرود")
    candidates = await _multiday_candidates(goal, Graph(), [], [ordinary])
    assert [row["id"] for row in candidates] == ["offroad"]


@pytest.mark.asyncio
async def test_conversation_goal_states_and_merge_lineage_are_written_with_snapshot():
    queries = []

    class Result:
        def __init__(self, row=None):
            self.row = row

        async def single(self):
            return self.row

        async def consume(self):
            return None

    class Transaction:
        def __init__(self):
            self.revision = 0

        async def run(self, query, **kwargs):
            queries.append((query, kwargs))
            if "RETURN u.identifier AS identifier" in query:
                return Result({"identifier": "u"})
            if "RETURN g.id AS id, g.revision AS revision" in query:
                return Result({"id": "old", "revision": self.revision} if self.revision else None)
            if "CREATE (g:TravelGoal" in query:
                self.revision += 1
            return Result()

    class Session:
        def __init__(self):
            self.tx = Transaction()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute_write(self, fn):
            return await fn(self.tx)

    class Driver:
        def __init__(self):
            self.session_obj = Session()

        def session(self):
            return self.session_obj

    graph = TripGraphRepository(Driver())
    original = infer_and_merge(TravelGoal(), "پاییز می‌خوام",
                               intent("پاییز می‌خوام", updates={"season": "پاییز"}))
    assert await graph.upsert_travel_goal("s", "u", original.model_dump()) == 1
    revised = apply_goal_updates(original, {"season": "پاییز"})
    revised = infer_and_merge(revised, "بهار می‌خوام",
                              intent("بهار می‌خوام", updates={"season": "بهار"}))
    assert await graph.upsert_travel_goal("s", "u", revised.model_dump()) == 2
    writes = [(query, args) for query, args in queries if "ConversationalGoalState" in query]
    assert len(writes) == 2
    assert "TRACKS_CONVERSATIONAL_GOAL" in writes[-1][0]
    assert "DERIVED_FROM_GOAL" in writes[-1][0]
    assert writes[-1][1]["items"][-1]["parents"] == ["C1"]
    assert writes[0][1]["items"][0]["clause"] == "پاییز می‌خوام"
    events = [(query, args) for query, args in queries
              if "CREATE (event:GoalRevisionEvent" in query]
    assert len(events) == 2
    assert all("CREATED_REVISION" in query for query, _ in events)


@pytest.mark.asyncio
async def test_trace_link_uses_exact_observation_marker_not_tool_name_or_time(monkeypatch):
    repo = TripGraphRepository(None)
    writes = []

    async def capture(query, **kwargs):
        writes.append((query, kwargs))
        return []

    monkeypatch.setattr(repo, "_write", capture)
    await repo.link_goal_revision_trace("session", "turn", "trace")
    assert len(writes) == 2
    assert "HAS_GOAL_REVISION" in writes[0][0]
    assert "call.result CONTAINS ('Goal observation id: ' + event.observation_id)" in writes[1][0]
    assert "observation_source:'agent_tool'" in writes[1][0]
    writes.clear()
    await repo.link_goal_revision_message("session", "turn", "message")
    assert len(writes) == 1
    assert "TRIGGERED_REVISION" in writes[0][0]
    assert writes[0][1]["message_id"] == "message"


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_AUTH_INTEGRATION") != "1", reason="Requires configured Neo4j")
async def test_neo4j_conversation_states_keep_revision_and_derivation_edges():
    from neo4j import AsyncGraphDatabase
    from src.config import get_settings

    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))
    user, session_id = f"dialogue_{uuid4().hex}", f"dialogue_{uuid4().hex}"
    repo = TripGraphRepository(driver)
    try:
        await repo.ensure_goal_revisions()
        original = infer_and_merge(
            TravelGoal(), "پاییز می‌خوام",
            intent("پاییز می‌خوام", updates={"season": "پاییز"}))
        original = apply_goal_updates(original, {"season": "پاییز"})
        assert await repo.upsert_travel_goal(session_id, user, original.model_dump()) == 1
        changed = infer_and_merge(
            original, "بهار می‌خوام",
            intent("بهار می‌خوام", updates={"season": "بهار"}))
        changed = apply_goal_updates(changed, {"season": "بهار"})
        assert await repo.upsert_travel_goal(session_id, user, changed.model_dump()) == 2
        async with driver.session() as session:
            result = await session.run(
                "MATCH (a:ConversationalGoalState {session_id:$sid, revision:1, goal_id:'C1'})"
                "-[:REVISED_TO]->(b:ConversationalGoalState {revision:2, goal_id:'C1'}) "
                "MATCH (a)-[link:DERIVED_FROM_GOAL]->"
                "(c:ConversationalGoalState {revision:2, goal_id:'C2'}) "
                "RETURN a.status AS prior, b.status AS revised, "
                "c.clause AS clause, link.operation AS operation",
                sid=session_id)
            row = await result.single()
        assert row is not None
        assert row["prior"] == "active" and row["revised"] == "superseded"
        assert row["clause"] == "بهار می‌خوام" and row["operation"] == "replace"
    finally:
        async with driver.session() as session:
            await (await session.run(
                "MATCH (u:User {identifier:$user})-[:HAS_GOAL]->(g:TravelGoal) "
                "OPTIONAL MATCH (g)-[:TRACKS_CONVERSATIONAL_GOAL]->(s:ConversationalGoalState) "
                "DETACH DELETE s,g,u", user=user)).consume()
        await driver.close()
