"""Follow-up questions must bypass itinerary templates and preserve session goals."""

import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi import Response

from src.agent.goals import TravelGoal
from src.agent import turns
from src.agent.agent import update_travel_goal
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
async def test_explicit_plan_for_previously_suggested_place_is_not_an_answer(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"action":"answer","updates":{}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    decision = await turns.understand_turn("برای آبشار لاتون برنامه بده", TravelGoal(), "توربین: آبشار لاتون")
    assert decision.action == "plan"


@pytest.mark.asyncio
async def test_first_travel_request_cannot_be_misclassified_as_chat(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"action":"chat","updates":{}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    decision = await turns.understand_turn("میخوایم آخر هفته بریم کمپ", None, "")
    assert decision.action == "plan"


@pytest.mark.asyncio
async def test_young_group_is_friends_not_children():
    goal = await update_travel_goal(None, "میخوایم آخر هفته با بچه ها جوونی بریم کمپ", "")
    assert goal.duration == "آخر هفته"
    assert goal.accommodation == ["کمپ"]
    assert "دوستان" in goal.companions and "کودک" not in goal.companions


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

    response = await route.chat(ChatRequest(message="گلابدره آب و هواش چطوره؟"), Response(), browser_user_id=None)
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

    async def explanation(*args):
        return "### چرا این مقصدها برای تو؟\n- **مقصد 1:** چون طبیعت دوست داری."

    monkeypatch.setattr(route, "explain_selected_plans", explanation)

    response = await route.chat(ChatRequest(message="برای یک روز دو مقصد پیشنهاد بده"), Response(), browser_user_id=None)
    assert response.reply.startswith("### چرا این مقصدها برای تو؟")
    assert "#### گزینهٔ 2: مقصد 2" in response.reply
    assert response.itinerary is None
    assert [plan.stops[0].name for plan in response.itineraries] == ["مقصد 1", "مقصد 2"]
    assert response.session_id and response.user_id


@pytest.mark.asyncio
async def test_chat_explains_named_alternatives_without_map_results(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")

    class Graph:
        synced_calls = []

        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return None

        async def upsert_travel_goal(self, *args):
            pass

        async def sync_session_features(self, user_id, session_id, state):
            assert user_id.startswith("user_") and session_id.startswith("session_")
            assert state["trip_types"] == [] or "کمپ" in state["accommodation"]
            Graph.synced_calls.append((user_id, session_id))

        async def find_destinations_mentioned_in_text(self, reply):
            assert "آبشار لاتون" in reply and "دریاچه نئور" in reply
            return ["آبشار لاتون", "دریاچه نئور"]

        async def get_destination_details(self, name):
            return {"name": name, "categories": ["طبیعت"], "trip_types": ["کمپ"]}

        async def get_user_history(self, user_id):
            return {"liked": []}

    class Agent:
        async def run(self, *args, **kwargs):
            return SimpleNamespace(output="برای کمپ: آبشار لاتون یا دریاچه نئور را پیشنهاد می‌کنم.",
                                   all_messages=lambda: [], new_messages=lambda: [])

    class Memory:
        _client = SimpleNamespace(_ensure_connected=lambda: object())
        short_term = SimpleNamespace(
            get_conversation=None,
            add_message=None,
        )
        long_term = SimpleNamespace(get_preferences_for=None)
        reasoning = SimpleNamespace()

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=[])

    async def add_message(*args, **kwargs):
        return SimpleNamespace(id="m1")

    async def preferences(*args):
        return []

    async def plan(*args):
        return turns.TurnDecision("plan")

    async def goal(*args):
        return TravelGoal(objective="آخر هفته کمپ بریم", accommodation=["کمپ"])

    async def no_screen(*args, **kwargs):
        return None

    Memory.short_term.get_conversation = conversation
    Memory.short_term.add_message = add_message
    Memory.long_term.get_preferences_for = preferences
    monkeypatch.setattr(route, "get_memory_client", Memory)
    monkeypatch.setattr(route, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(route, "TripGraphRepository", Graph)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: Agent())
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "understand_turn", plan)
    monkeypatch.setattr(route, "update_travel_goal", goal)
    monkeypatch.setattr(route, "screen_short_trip", no_screen)

    response = await route.chat(ChatRequest(message="میخوایم آخر هفته با بچه‌ها بریم کمپ"),
                                Response(), browser_user_id=None)
    assert response.reply.startswith("### چرا این مقصدها برای تو؟")
    assert "**آبشار لاتون:**" in response.reply and "**دریاچه نئور:**" in response.reply
    assert response.itinerary is None and response.itineraries == []
    assert Graph.synced_calls == [(response.user_id, response.session_id)]
