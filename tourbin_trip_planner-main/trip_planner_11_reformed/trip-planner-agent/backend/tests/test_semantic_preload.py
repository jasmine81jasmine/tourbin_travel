"""Semantic retrieval complements every planning turn, without claiming verification."""

import importlib

import pytest

from src.agent.dependencies import AgentDeps
from src.agent.goals import PlanningSubgoal, TravelGoal

chat_route = importlib.import_module("src.api.routes.chat")


class Graph:
    def __init__(self):
        self.queries = []
        self.snapshots = []

    async def semantic_search_destinations(self, query, limit):
        self.queries.append((query, limit))
        return [{"id": "d1", "name": "جنگل", "categories": ["طبیعت"],
                 "description": "مسیر جنگلی آرام", "semantic_score": 0.82}]

    async def get_avoided_destination_ids(self, user_id):
        return []

    async def upsert_travel_goal(self, session_id, user_id, state):
        self.snapshots.append(state)
        return len(self.snapshots)


@pytest.mark.asyncio
async def test_plan_preloads_semantic_matches_and_tracks_only_candidate_evidence():
    graph = Graph()
    goal = TravelGoal(region="شمال", semantic_query="سفر طبیعت با دوستان",
                      subgoals=[PlanningSubgoal(id="G1", name="طبیعت",
                                                evidence_type="category", category="طبیعت")])
    deps = AgentDeps.create(memory=None, graph=graph, session_id="s",
                            user_id="u", current_query="شمال", travel_goal=goal)
    await chat_route._preload_semantic_candidates(deps)
    assert graph.queries == [(goal.semantic_query, 18)]
    assert deps.semantic_candidates[0]["name"] == "جنگل"
    assert deps.observed_tools == ["semantic_search_destinations"]
    assert goal.goal_coverage[0].status == "partial"
    assert graph.snapshots[0]["goal_coverage"][0]["status"] == "partial"


@pytest.mark.asyncio
async def test_answer_turn_skips_semantic_preload():
    graph = Graph()
    deps = AgentDeps.create(memory=None, graph=graph, session_id="s",
                            turn_action="answer", travel_goal=TravelGoal())
    await chat_route._preload_semantic_candidates(deps)
    assert graph.queries == [] and deps.semantic_candidates is None


@pytest.mark.asyncio
async def test_failed_semantic_preload_does_not_block_planning():
    class Unavailable(Graph):
        async def semantic_search_destinations(self, query, limit):
            raise RuntimeError("Index unavailable")

    graph = Unavailable()
    deps = AgentDeps.create(memory=None, graph=graph, session_id="s",
                            travel_goal=TravelGoal(semantic_query="طبیعت"))
    await chat_route._preload_semantic_candidates(deps)
    assert deps.semantic_candidates is None and graph.snapshots == []
