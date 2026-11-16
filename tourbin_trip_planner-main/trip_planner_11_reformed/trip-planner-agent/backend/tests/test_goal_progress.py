"""Progress, bounded decomposition, and non-planning regression checks."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.agent.goal_manager import GoalDecomposition, decompose_unmet_goal, evaluate_plan
from src.agent.goal_progress import observed_result
from src.agent.goals import GoalCoverage, PlanningConstraint, PlanningSubgoal, TravelGoal


class Recorder:
    def __init__(self):
        self.snapshots = []

    async def upsert_travel_goal(self, session_id, user_id, state):
        assert (session_id, user_id) == ("s", "u")
        self.snapshots.append(json.loads(json.dumps(state)))
        return len(self.snapshots)


def context(goal, graph=None, action="plan"):
    return SimpleNamespace(deps=SimpleNamespace(
        travel_goal=goal, graph=graph, turn_action=action,
        session_id="s", user_id="u", observed_tools=[], progress_lock=asyncio.Lock()))


@pytest.mark.asyncio
async def test_each_tool_step_snapshots_goals_and_returns_progress_to_next_model():
    graph = Recorder()
    goal = TravelGoal(subgoals=[
        PlanningSubgoal(id="G1", name="طبیعت", evidence_type="category", category="طبیعت"),
        PlanningSubgoal(id="G2", name="مسیر", evidence_type="route", priority="high"),
        PlanningSubgoal(id="G3", name="خلوت", evidence_type="unverified")])
    ctx = context(goal, graph)
    first = await observed_result(ctx, "search_destinations", [
        {"name": "جنگل", "categories": ["طبیعت"]}])
    assert "Goal progress" in first
    assert [entry["status"] for entry in graph.snapshots[0]["goal_coverage"]] == [
        "partial", "pending", "pending"]
    assert goal.goal_coverage[0].status == "partial"  # never satisfied by search
    await observed_result(ctx, "build_trip_map", {
        "stops": [{"name": "جنگل"}], "used_real_routing": True})
    assert graph.snapshots[1]["goal_coverage"][1]["status"] == "partial"
    assert graph.snapshots[0]["goal_coverage"][1]["status"] == "pending"  # immutable
    assert graph.snapshots[1]["last_observation_tool"] == "build_trip_map"
    assert ctx.deps.observed_tools == ["search_destinations", "build_trip_map"]
    assert goal.revision == 2
    assert graph.snapshots[0]["revision_cause"] == "tool_observation"
    assert graph.snapshots[0]["observation_source"] == "agent_tool"
    assert graph.snapshots[0]["observation_summary"] == ["جنگل"]
    assert "Goal observation id: " + graph.snapshots[0]["observation_id"] in first
    assert graph.snapshots[1]["observation_id"] != graph.snapshots[0]["observation_id"]


@pytest.mark.asyncio
async def test_preload_observation_is_not_misidentified_as_an_agent_tool_call():
    graph = Recorder()
    ctx = context(TravelGoal(), graph)
    output = await observed_result(ctx, "semantic_search_destinations", [],
                                   source="preload")
    assert graph.snapshots[0]["observation_source"] == "preload"
    assert "Goal observation id: " not in output


@pytest.mark.asyncio
async def test_parallel_tool_calls_keep_monotonic_goal_snapshots():
    graph = Recorder()
    goal = TravelGoal(subgoals=[
        PlanningSubgoal(id="G1", name="طبیعت", evidence_type="category", category="طبیعت"),
        PlanningSubgoal(id="G2", name="جاده", evidence_type="feature",
                        feature_id="roadtype:easy")])
    ctx = context(goal, graph)
    await asyncio.gather(
        observed_result(ctx, "search_destinations", [
            {"name": "جنگل", "categories": ["طبیعت"]}]),
        observed_result(ctx, "search_destinations_by_features", {
            "candidates": [{"name": "جنگل", "matched_feature_ids": ["roadtype:easy"]}]}))
    assert len(graph.snapshots) == 2
    assert [entry["status"] for entry in graph.snapshots[-1]["goal_coverage"]] == [
        "partial", "partial"]


@pytest.mark.asyncio
async def test_unverified_or_failed_tools_cannot_claim_coverage_and_answers_do_not_write():
    graph = Recorder()
    goal = TravelGoal(subgoals=[PlanningSubgoal(
        id="G1", name="غذا", evidence_type="unverified")])
    ctx = context(goal, graph)
    await observed_result(ctx, "search_destinations", [{"error": "offline"}])
    assert graph.snapshots[0]["goal_coverage"] == []
    assert goal.goal_coverage == []
    await observed_result(context(goal, graph, "answer"), "search_destinations", [{"name": "کافه"}])
    assert len(graph.snapshots) == 1
    assert (await evaluate_plan(goal, [{"stops": [{"name": "کافه"}]}]))[0].status == "pending"


@pytest.mark.asyncio
async def test_decomposition_only_on_unmet_goal_and_preserves_explicit_constraints(monkeypatch):
    from src.agent import goal_manager
    seen = []

    class Designer:
        async def run(self, prompt):
            seen.append(json.loads(prompt))
            return SimpleNamespace(output=GoalDecomposition(children=[
                PlanningSubgoal(id="G2.1", parent_id="G2", name="جست‌وجوی مسیر",
                                evidence_type="route"),
                PlanningSubgoal(id="G2.2", parent_id="G2", name="بررسی مقصد",
                                evidence_type="destination", target="جنگل"),
                PlanningSubgoal(id="G9", parent_id="G2", name="نامعتبر")]))

    monkeypatch.setattr(goal_manager, "_decomposer", lambda: Designer())
    goal = TravelGoal(
        subgoals=[PlanningSubgoal(id="G1", name="طبیعت", priority="high"),
                  PlanningSubgoal(id="G2", name="مسیر", priority="high",
                                  evidence_type="route")],
        planning_constraints=[PlanningConstraint(description="دو روز", source="explicit", kind="hard")])
    coverage = [GoalCoverage(goal_id="G1", status="satisfied"),
                GoalCoverage(goal_id="G2", status="pending", reason="نشان پاسخ نداد")]
    revised = await decompose_unmet_goal(goal, coverage, ["search_destinations"])
    assert [g.id for g in revised.subgoals] == ["G1", "G2", "G2.1", "G2.2"]
    assert revised.planning_constraints == goal.planning_constraints
    assert seen[0]["failed_goal"]["id"] == "G2"
    assert await decompose_unmet_goal(revised, coverage, []) is revised  # depth bounded


@pytest.mark.asyncio
async def test_no_decomposition_when_goal_satisfied_or_unverifiable(monkeypatch):
    from src.agent import goal_manager

    def unexpected():
        raise AssertionError("Unnecessary LLM call")

    monkeypatch.setattr(goal_manager, "_decomposer", unexpected)
    goal = TravelGoal(subgoals=[PlanningSubgoal(
        id="G1", name="ساعت", priority="high", evidence_type="unverified")])
    assert await decompose_unmet_goal(goal, [GoalCoverage(goal_id="G1", status="pending")], []) is goal
