"""Follow-up questions must bypass itinerary templates and preserve session goals."""

import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi import Response

from src.agent.goals import TravelGoal
from src.agent import turns
from src.api.schemas import ChatRequest


@pytest.mark.asyncio
async def test_llm_decides_followup_is_an_answer_and_keeps_duration(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            data = json.loads(prompt)
            assert data["current_goal"]["duration"] == "یک روز"
            assert "درکه" in data["recent_conversation"]
            return SimpleNamespace(output='{"action":"answer","updates":{}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    goal = TravelGoal(duration="یک روز", origin="تهران")
    decision = await turns.understand_turn("کدومشون خلوت تره؟", goal, "توربین: درکه و دربند")
    assert decision.action == "answer"
    assert turns.apply_goal_updates(goal, decision.updates).duration == "یک روز"


@pytest.mark.asyncio
async def test_llm_can_select_destination_without_erasing_existing_trip(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"action":"answer","updates":{"destination_names":["روستای ایگل"]}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    goal = TravelGoal(duration="دو روز", origin="تهران", accommodation=["کمپ"])
    decision = await turns.understand_turn("کجا کمپ کنم ایگل؟", goal, "توربین: روستای ایگل")
    updated = turns.apply_goal_updates(goal, decision.updates)
    assert decision.action == "answer"
    assert updated.destination_names == ["روستای ایگل"]
    assert updated.duration == "دو روز" and updated.accommodation == ["کمپ"]


@pytest.mark.asyncio
async def test_revised_constraints_preserve_unchanged_session_preferences(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"action":"plan","updates":{"duration":"سه روز","region":"شمال"}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    goal = TravelGoal(duration="دو روز", origin="تهران", companions=["کودک"], accommodation=["کمپ"])
    decision = await turns.understand_turn("سه روز وقت دارم، شمال رو ترجیح میدم", goal, "توربین: برنامهٔ قبلی")
    updated = turns.apply_goal_updates(goal, decision.updates)
    assert decision.action == "plan"
    assert updated.duration == "سه روز" and updated.region == "شمال"
    assert updated.origin == "تهران" and updated.companions == ["کودک"]
    assert updated.accommodation == ["کمپ"]


def test_classifier_failure_favors_answering_questions():
    assert turns._fallback_action("کدوم گزینه خلوت‌تره؟", True) == "answer"
    assert turns._fallback_action("گلابدره آب و هواش چطوره؟", True) == "answer"
    assert turns._fallback_action("کجا کمپ کنم ایگل؟", True) == "answer"
    assert turns._fallback_action("چند گزینه نزدیک تر پیشنهاد بده", True) == "plan"


@pytest.mark.asyncio
async def test_chat_followup_does_not_run_plan_screen_or_rewrite_reply(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")

    class FakeAgent:
        async def run(self, message, deps, message_history):
            assert deps.turn_action == "answer"
            return SimpleNamespace(output="آب‌وهوای گلابدره در بهار معمولاً معتدل است؛ برای وضعیت امروز پیش‌بینی زنده را ببینید.",
                                   all_messages=lambda: [], new_messages=lambda: [])

    async def answer(*args):
        return turns.TurnDecision("answer")

    async def forbidden(*args, **kwargs):
        raise AssertionError("A destination question must not trigger the trip-plan formatter")

    monkeypatch.setattr(route, "get_memory_client", lambda: None)
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: FakeAgent())
    monkeypatch.setattr(route, "understand_turn", answer)
    monkeypatch.setattr(route, "screen_short_trip", forbidden)

    response = await route.chat(ChatRequest(message="گلابدره آب و هواش چطوره؟"), Response())
    assert "گلابدره" in response.reply
    assert "برای پیشنهاد سفر کوتاه" not in response.reply
    assert response.itinerary is None


@pytest.mark.asyncio
async def test_chat_exposes_independent_map_routes_for_alternative_plans(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")

    class FakeAgent:
        async def run(self, message, deps, message_history):
            return SimpleNamespace(output="چند برنامه", all_messages=lambda: [], new_messages=lambda: [])

    async def plan(*args):
        return turns.TurnDecision("plan")

    async def goal(*args):
        return TravelGoal(duration="یک روز")

    async def screen(*args, map_itineraries, **kwargs):
        for index in (1, 2):
            map_itineraries.append({
                "origin": {"name": "تهران", "latitude": 35.6892, "longitude": 51.389},
                "stops": [{"order": 1, "name": f"مقصد {index}", "latitude": 35.8 + index * .01,
                           "longitude": 51.4}],
                "return_leg": {"leg_distance_km_from_previous": 20, "leg_duration_hours_from_previous": .5},
                "round_trip": True,
            })
        return "### 🌿 پیشنهادها\n\n#### گزینهٔ 1: مقصد 1\n\n#### گزینهٔ 2: مقصد 2", None

    monkeypatch.setattr(route, "get_memory_client", lambda: None)
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: FakeAgent())
    monkeypatch.setattr(route, "understand_turn", plan)
    monkeypatch.setattr(route, "update_travel_goal", goal)
    monkeypatch.setattr(route, "screen_short_trip", screen)

    response = await route.chat(ChatRequest(message="برای یک روز دو مقصد پیشنهاد بده"), Response())
    assert response.itinerary is None
    assert [plan.stops[0].name for plan in response.itineraries] == ["مقصد 1", "مقصد 2"]
    assert response.session_id and response.user_id
