"""Returning-user context is soft; old suggestions are not visits."""

import json
import importlib
from types import SimpleNamespace

import pytest
from fastapi import Response

from src.agent.goals import TravelGoal, insights_from_previous_goal, with_prior_insights
from src.agent.variety import vary_candidates
from src.agent import turns
from src.api.schemas import ChatRequest
from src.memory.graph import TripGraphRepository


def test_previous_trip_hints_fill_only_missing_dimensions():
    prior = {"categories": ["طبیعت"], "moods": ["ماجراجویانه"],
             "trip_types": ["کمپ"], "companions": ["دوستان"],
             "seasons": ["بهار"], "vehicles": ["آفرود"]}
    inferred = with_prior_insights(TravelGoal(objective="یه سفر پیشنهاد بده", semantic_query="سفر"), prior)
    assert inferred.prior_insights == {
        "categories": ["طبیعت"], "trip_types": ["کمپ"], "moods": ["ماجراجویانه"]
    }
    assert "طبیعت" in inferred.semantic_query
    assert inferred.categories == [] and inferred.moods == []  # never re-label old requests as new ones
    assert inferred.companions == [] and inferred.season is None and inferred.transport is None

    explicit = with_prior_insights(
        TravelGoal(categories=["تاریخی"], moods=["آرام"], trip_types=["بازدید"],
                   avoid=["کمپ"], semantic_query="سفر تاریخی آرام"), prior
    )
    assert explicit.prior_insights == {}
    assert "طبیعت" not in explicit.semantic_query and "ماجراجویانه" not in explicit.semantic_query
    no_camping = with_prior_insights(TravelGoal(avoid=["کمپ"], semantic_query="بدون کمپ"), prior)
    assert "trip_types" not in no_camping.prior_insights


def test_older_session_goal_can_supply_soft_hints_without_reusing_trip_logistics():
    old = TravelGoal(categories=["طبیعت"], moods=["ماجراجویانه"], accommodation=["کمپ"],
                     companions=["دوستان"], duration="دو روز", transport="آفرود", origin="شیراز")
    hints = insights_from_previous_goal(old.model_dump_json())
    assert hints == {"categories": ["طبیعت"], "moods": ["ماجراجویانه"],
                     "trip_types": ["کمپ"], "companions": ["دوستان"]}
    assert insights_from_previous_goal("not json") == {}


def test_recommendations_prefer_new_similarly_qualified_places_and_stay_user_scoped():
    rows = [{"id": f"d{i}", "name": f"مقصد {i}", "user_rating_score": 3,
             "source_is_osm": False} for i in range(8)]
    first = vary_candidates(rows, {"ids": set(), "names": set()}, "user-a", "session-1", 3)
    repeat = vary_candidates(rows, {"ids": {r["id"] for r in first}, "names": set()},
                             "user-a", "session-2", 3)
    assert len(first) == len(repeat) == 3
    assert not {r["id"] for r in first} & {r["id"] for r in repeat}
    assert vary_candidates(rows, {"ids": set(), "names": set()}, "user-a", "session-1", 3) == first
    assert vary_candidates(rows, {"ids": set(), "names": set()}, "user-b", "session-1", 8) != vary_candidates(
        rows, {"ids": set(), "names": set()}, "user-a", "session-1", 8
    )


def test_no_bad_semantic_match_promoted_just_to_avoid_a_repeat():
    rows = [{"id": "familiar", "name": "الف", "semantic_score": .91},
            {"id": "fresh", "name": "ب", "semantic_score": .35}]
    ranked = vary_candidates(rows, {"ids": {"familiar"}, "names": set()},
                             "user-a", "session-2", 1, semantic=True)
    assert ranked[0]["id"] == "familiar"


def test_variety_does_not_promote_low_rated_fallback_above_credible_result():
    rows = [{"id": "familiar", "name": "الف", "user_rating_score": 3},
            {"id": "poor", "name": "ب", "user_rating_score": 0}]
    result = vary_candidates(rows, {"ids": {"familiar"}, "names": set()},
                             "user-a", "session-2", 1)
    assert result[0]["id"] == "familiar"


@pytest.mark.asyncio
async def test_previous_features_only_read_for_same_user_and_other_session():
    repo = TripGraphRepository(None)

    async def read(query, **params):
        assert "REQUESTED_FOR_TRIP" in query and "r.session_id <> $exclude_session_id" in query
        assert params == {"user_id": "user-a", "exclude_session_id": "session-2"}
        return [{"labels": ["RequestedStyle"], "name": "ماجراجویانه"},
                {"labels": ["RequestedCompanion"], "name": "دوستان"},
                {"labels": ["TripType"], "name": "کمپ"}]

    repo._read = read
    assert await repo.get_recent_session_features("user-a", "session-2") == {
        "moods": ["ماجراجویانه"], "companions": ["دوستان"], "trip_types": ["کمپ"]
    }


@pytest.mark.asyncio
async def test_shown_is_separate_from_visited_and_recovers_old_goal_routes():
    repo = TripGraphRepository(None)
    state = json.dumps({"recommended_routes": [["جنگل قدیمی"]]}, ensure_ascii=False)

    async def read(query, **params):
        assert "r.session_id <> $exclude_session_id" in query
        assert "PLANNED" in query and "SHOWN_DESTINATION" in query
        assert params["user_id"] == "user-a" and params["exclude_session_id"] == "session-new"
        return [{"shown": [{"id": "one", "name": "آبشار الف"}],
                 "planned": [{"id": "two", "name": "کوه ب"}], "recent_goals": [state]}]

    repo._read = read
    assert await repo.get_previous_recommendations("user-a", "session-new") == {
        "ids": {"one", "two"}, "names": {"آبشار الف", "کوه ب", "جنگل قدیمی"}
    }

    async def write(query, **params):
        assert "SHOWN_DESTINATION" in query and "VISITED" not in query
        assert params == {"user_id": "user-a", "session_id": "session-new", "names": ["آبشار الف"]}
        return []

    repo._write = write
    await repo.record_shown_destinations("user-a", "session-new", ["آبشار الف", "آبشار الف"])


@pytest.mark.asyncio
async def test_returning_user_greeting_mentions_last_request_not_permanent_habit(monkeypatch):
    chat = importlib.import_module("src.api.routes.chat")

    class Graph:
        def __init__(self, driver):
            pass

        async def get_user_history(self, user_id, exclude_session_id):
            assert user_id == "user-a" and exclude_session_id.startswith("session_")
            return {"recent_goals": [], "trips": []}

        async def get_recent_session_features(self, user_id, exclude_session_id):
            return {"moods": ["ماجراجویانه"], "companions": ["دوستان"]}

    async def preferences(user_id):
        return []

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             long_term=SimpleNamespace(get_preferences_for=preferences))
    monkeypatch.setattr(chat, "get_memory_client", lambda: memory)
    monkeypatch.setattr(chat, "TripGraphRepository", Graph)

    session = await chat.start_session(Response(), browser_user_id="user-a")
    assert session.user_id == "user-a" and session.session_id.startswith("session_")
    assert "آخرین بار" in session.greeting and "ماجراجویانه" in session.greeting
    assert "دوستان" in session.greeting and "همیشه" not in session.greeting


@pytest.mark.asyncio
async def test_first_time_user_gets_new_identity_and_no_claimed_memories(monkeypatch):
    chat = importlib.import_module("src.api.routes.chat")
    monkeypatch.setattr(chat, "get_memory_client", lambda: None)
    first = await chat.start_session(Response(), browser_user_id=None)
    second = await chat.start_session(Response(), browser_user_id=None)
    assert first.user_id != second.user_id
    assert "خوش برگشتی" not in first.greeting


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_message", "expected"),
    [("یه سفر پیشنهاد بده", {"categories": ["طبیعت"], "moods": ["ماجراجویانه"]}),
     ("یه سفر تاریخی و آرام پیشنهاد بده", {})],
)
async def test_new_session_plan_uses_only_unspecified_prior_insights(monkeypatch, user_message, expected):
    chat = importlib.import_module("src.api.routes.chat")

    class Graph:
        _destination_lookup_keys = staticmethod(TripGraphRepository._destination_lookup_keys)

        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return None

        async def get_previous_recommendations(self, *args):
            return {"ids": {"old"}, "names": {"قبلی"}}

        async def get_recent_session_features(self, *args):
            return {"categories": ["طبیعت"], "moods": ["ماجراجویانه"], "companions": ["دوستان"]}

        async def upsert_travel_goal(self, session_id, user_id, state):
            assert state["prior_insights"] == expected
            assert state["companions"] == []

        async def sync_session_features(self, user_id, session_id, state):
            assert state["categories"] == (["تاریخی"] if "تاریخی" in user_message else [])
            assert state["prior_insights"] == expected

        async def find_destinations_mentioned_in_text(self, reply):
            return []

    class Agent:
        async def run(self, message, deps, message_history):
            assert deps.travel_goal.prior_insights == expected
            assert deps.previous_recommendations["names"] == {"قبلی"}
            return SimpleNamespace(output="در حال بررسی سفر هستم.",
                                   all_messages=lambda: [], new_messages=lambda: [])

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=[])

    async def message(*args, **kwargs):
        return SimpleNamespace(id="m1")

    async def preferences(*args):
        return []

    async def plan(*args):
        return turns.TurnDecision("plan")

    async def no_screen(*args, **kwargs):
        return None

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             short_term=SimpleNamespace(get_conversation=conversation, add_message=message),
                             long_term=SimpleNamespace(get_preferences_for=preferences),
                             reasoning=SimpleNamespace())
    monkeypatch.setattr(chat, "get_memory_client", lambda: memory)
    monkeypatch.setattr(chat, "TripGraphRepository", Graph)
    monkeypatch.setattr(chat, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(chat, "get_neshan_client", lambda: None)
    monkeypatch.setattr(chat, "get_trip_planner_agent", lambda: Agent())
    monkeypatch.setattr(chat, "understand_turn", plan)
    monkeypatch.setattr(chat, "screen_short_trip", no_screen)

    result = await chat.chat(ChatRequest(message=user_message), Response(), browser_user_id="user-a")
    assert result.user_id == "user-a" and result.itinerary is None
