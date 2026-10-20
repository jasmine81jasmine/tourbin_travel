"""Memory questions must report stored facts, not an LLM's inconsistent guess."""

import importlib
from types import SimpleNamespace

import pytest
from fastapi import Response

from src.agent.goals import TravelGoal
from src.agent.memory_summary import summarize_user_memory
from src.agent.tools import _default_tehran_area, search_destinations, semantic_search_destinations
from src.api.schemas import ChatRequest
from src.agent import turns
from src.agent.agent import update_travel_goal
from src.agent.feasibility import _intent_score


class Graph:
    async def get_user_history(self, user_id, exclude_session_id):
        assert user_id == "returning-user" and exclude_session_id == "current-session"
        return {"liked": ["روستای الف"], "disliked": [], "visited": [],
                "recent_goals": [], "trips": [{"destinations": ["آبشار ب"]}]}

    async def get_recent_session_features(self, user_id, session_id):
        assert user_id == "returning-user" and session_id == "current-session"
        return {"moods": ["ماجراجویانه"], "companions": ["دوستان"], "trip_types": ["کمپ"]}


@pytest.mark.asyncio
async def test_memory_answer_distinguishes_past_requests_from_actual_feedback():
    async def preferences(user_id):
        assert user_id == "returning-user"
        return [SimpleNamespace(preference="همیشه طبیعت‌گردی را دوست دارم")]

    memory = SimpleNamespace(long_term=SimpleNamespace(get_preferences_for=preferences))
    reply = await summarize_user_memory(memory, Graph(), "returning-user", "current-session", None)
    assert "ماجراجویانه" in reply and "دوستان" in reply and "کمپ" in reply
    assert "ترجیح همیشگی‌ات فرض نمی‌کنم" in reply
    assert "همیشه طبیعت‌گردی" in reply and "روستای الف" in reply
    assert "آبشار ب" in reply and "نه لزوماً سفری که رفته‌ای" in reply
    assert "اطلاعات خاصی" not in reply


@pytest.mark.asyncio
async def test_memory_answer_can_recall_prior_trip_details_without_assuming_current_duration():
    class GraphWithGoal(Graph):
        async def get_user_history(self, user_id, exclude_session_id):
            history = await super().get_user_history(user_id, exclude_session_id)
            return {**history, "recent_goals": [TravelGoal(duration="دو روز", region="شمال").model_dump_json()]}

    async def preferences(user_id):
        return []

    memory = SimpleNamespace(long_term=SimpleNamespace(get_preferences_for=preferences))
    reply = await summarize_user_memory(memory, GraphWithGoal(), "returning-user", "current-session", None)
    assert "**جزئیات همان برنامهٔ قبلی:** دو روز، شمال" in reply


@pytest.mark.asyncio
async def test_memory_question_skips_model_and_uses_same_browser_identity(monkeypatch):
    chat = importlib.import_module("src.api.routes.chat")
    messages = []

    class CurrentGraph(Graph):
        def __init__(self, *args):
            pass

        async def get_travel_goal(self, session_id, user_id):
            assert session_id == "current-session" and user_id == "returning-user"
            return TravelGoal(prior_insights={"moods": ["ماجراجویانه"]}).model_dump()

    async def add_message(**kwargs):
        messages.append((kwargs["role"], kwargs["content"], kwargs["user_identifier"]))

    async def preferences(user_id):
        return []

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             short_term=SimpleNamespace(get_conversation=None, add_message=add_message),
                             long_term=SimpleNamespace(get_preferences_for=preferences))
    monkeypatch.setattr(chat, "get_memory_client", lambda: memory)
    monkeypatch.setattr(chat, "TripGraphRepository", CurrentGraph)
    monkeypatch.setattr(chat, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(chat, "_load_session_history", lambda *args: _empty_history())
    monkeypatch.setattr(chat, "get_trip_planner_agent", lambda: pytest.fail("memory answer must not call model"))

    response = await chat.chat(ChatRequest(message="تو چیا از من میدونی ؟", session_id="current-session"),
                               Response(), browser_user_id="returning-user")
    assert "ماجراجویانه" in response.reply and "روستای الف" in response.reply
    assert response.user_id == "returning-user"
    assert [role for role, _, _ in messages] == ["user", "assistant"]
    assert all(user == "returning-user" for _, _, user in messages)


async def _empty_history():
    return [], ""


@pytest.mark.asyncio
async def test_memory_unavailable_is_not_misreported_as_no_preferences():
    reply = await summarize_user_memory(None, None, "user", "session", None)
    assert "دسترسی ندارم" in reply and "چیزی درباره‌ات" not in reply

    class BrokenGraph:
        async def get_user_history(self, *args, **kwargs):
            raise RuntimeError("temporary Neo4j failure")

        async def get_recent_session_features(self, *args):
            raise RuntimeError("temporary Neo4j failure")

    async def preferences(*args):
        return []

    memory = SimpleNamespace(long_term=SimpleNamespace(get_preferences_for=preferences))
    reply = await summarize_user_memory(memory, BrokenGraph(), "user", "session", None)
    assert "نتونستم" in reply and "اطلاعات تأییدشده‌ای" not in reply


def test_memory_based_plan_is_not_mistaken_for_what_do_you_know_question():
    chat = importlib.import_module("src.api.routes.chat")
    assert chat._is_memory_question("تو چیا از من میدونی ؟")
    assert chat._is_memory_question("تو درباره برنامه سفرهام چی میدونی؟")
    assert chat._is_memory_question("چی درباره من میدونی؟")
    assert not chat._is_memory_question("با توجه به چیزایی ک از من میدونی یه برنامه سفر برام بریز")
    assert chat._is_memory_based_plan("با توجه به چیزایی ک از من میدونی یه برنامه سفر برام بریز")


@pytest.mark.asyncio
async def test_from_me_in_memory_request_does_not_change_origin_from_tehran():
    goal = await update_travel_goal(None, "با توجه به چیزایی که از من میدونی یه برنامه سفر برام بریز", "")
    assert goal.origin == "تهران"
    damaged = await update_travel_goal(TravelGoal(origin="من"), "یه برنامه دیگه برام بریز", "")
    assert damaged.origin == "تهران"
    assert turns.apply_goal_updates(goal, {"origin": "من"}).origin == "تهران"


@pytest.mark.asyncio
async def test_memory_based_plan_replaces_far_draft_with_assumed_nearby_day_trip(monkeypatch):
    chat = importlib.import_module("src.api.routes.chat")

    class PlannerGraph:
        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return None

        async def get_previous_recommendations(self, *args):
            return {"ids": set(), "names": set()}

        async def get_recent_session_features(self, *args):
            return {"moods": ["ماجراجویانه"]}

        async def upsert_travel_goal(self, session, user, state):
            assert state["duration"] is None
            assert state["prior_insights"] == {"moods": ["ماجراجویانه"]}

        async def sync_session_features(self, user, session, state):
            assert state["duration"] is None

        async def find_destinations_mentioned_in_text(self, reply):
            return []

    class Agent:
        async def run(self, message, deps, message_history):
            return SimpleNamespace(output="دریاچه دربندیخان در عراق را پیشنهاد می‌کنم.",
                                   all_messages=lambda: [], new_messages=lambda: [])

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=[])

    async def add_message(*args, **kwargs):
        return SimpleNamespace(id="m1")

    async def preferences(*args):
        return []

    async def plan(*args):
        return turns.TurnDecision("plan")

    async def screen(goal, *args, **kwargs):
        assert goal.duration == "یک روز"
        return "### 🌿 برنامهٔ یک‌روزه نزدیک تهران: لواسان", None

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             short_term=SimpleNamespace(get_conversation=conversation, add_message=add_message),
                             long_term=SimpleNamespace(get_preferences_for=preferences),
                             reasoning=SimpleNamespace())
    monkeypatch.setattr(chat, "get_memory_client", lambda: memory)
    monkeypatch.setattr(chat, "TripGraphRepository", PlannerGraph)
    monkeypatch.setattr(chat, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(chat, "get_neshan_client", lambda: None)
    monkeypatch.setattr(chat, "get_trip_planner_agent", lambda: Agent())
    monkeypatch.setattr(chat, "understand_turn", plan)
    monkeypatch.setattr(chat, "screen_short_trip", screen)
    async def no_travel_memory(*args):
        return {"preferences": [], "mentions": []}
    monkeypatch.setattr(chat, "extract_travel_memory", no_travel_memory)

    reply = await chat.chat(ChatRequest(message="با توجه به چیزایی ک از من میدونی یه برنامه سفر برام بریز"),
                            Response(), browser_user_id="returning-user")
    assert "یک‌روزه" in reply.reply and "لواسان" in reply.reply
    assert "مدت سفر را مشخص نکردی" in reply.reply
    assert "درخواست سفر قبلی‌ات ماجراجویانه" in reply.reply
    assert "دربندیخان" not in reply.reply


@pytest.mark.asyncio
async def test_vague_tehran_plan_filters_faraway_semantic_places_and_searches_locally():
    class GraphSearch:
        async def get_avoided_destination_ids(self, user_id):
            return []

        async def search_destinations(self, **kwargs):
            assert kwargs["location"] == ["تهران", "البرز", "قزوین"]
            return [{"id": "near", "name": "لواسان", "latitude": 35.8, "longitude": 51.5}]

        async def semantic_search_destinations(self, goal, limit):
            return [{"id": "far", "name": "دریاچه دربندیخان", "latitude": 35.1, "longitude": 45.9,
                     "semantic_score": .9},
                    {"id": "near", "name": "لواسان", "latitude": 35.8, "longitude": 51.5,
                     "semantic_score": .6}]

    ctx = SimpleNamespace(deps=SimpleNamespace(
        graph=GraphSearch(), user_id="returning-user", session_id="new-session",
        travel_goal=TravelGoal(prior_insights={"moods": ["ماجراجویانه"]}),
        current_query="با توجه به چیزایی که از من میدونی یه برنامه سفر برام بریز",
        previous_recommendations={"ids": set(), "names": set()},
    ))
    assert [row["name"] for row in await search_destinations(ctx)] == ["لواسان"]
    assert [row["name"] for row in await semantic_search_destinations(ctx)] == ["لواسان"]
    assert not _default_tehran_area(ctx, "شیراز")
    ctx.deps.travel_goal = TravelGoal(duration="سه روز")
    assert not _default_tehran_area(ctx)


def test_verifier_uses_prior_adventure_only_when_this_trip_did_not_override_it():
    adventure = {"name": "کوه", "categories": ["کوهنوردی"]}
    relaxed = {"name": "باغ", "categories": ["تفریح خانوادگی"]}
    goal = TravelGoal(prior_insights={"moods": ["ماجراجویانه"]})
    assert _intent_score(adventure, goal) > _intent_score(relaxed, goal)
    changed = goal.model_copy(update={"moods": ["آرام"]})
    assert _intent_score(adventure, changed) == _intent_score(relaxed, changed)
