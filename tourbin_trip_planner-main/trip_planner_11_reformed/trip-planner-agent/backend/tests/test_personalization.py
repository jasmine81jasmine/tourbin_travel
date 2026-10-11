"""The visible reason must be supported by the selected place and this user."""

from types import SimpleNamespace

import pytest

from src.agent.goals import TravelGoal
from src.agent.personalization import explain_selected_plans
from src.agent.tools import search_destinations, semantic_search_destinations


def route(name):
    return [{"stops": [{"name": name}]}]


class Graph:
    async def get_destination_details(self, name):
        return {"name": name, "categories": ["طبیعت", "جنگل"], "trip_types": []}

    async def get_user_history(self, user_id):
        return {"liked": []}


@pytest.mark.asyncio
async def test_explains_current_goal_and_remembered_preference():
    memory = SimpleNamespace(long_term=SimpleNamespace(get_preferences_for=async_preferences))
    text = await explain_selected_plans(
        TravelGoal(objective="یک سفر طبیعت‌گردی"), route("جنگل الف"), Graph(), memory, "user-a"
    )
    assert "جنگل الف" in text and "در این سفر" in text and "گفتگوهای قبلی" in text
    assert "جنگل ب" not in text


async def async_preferences(user_id):
    assert user_id == "user-a"
    return [SimpleNamespace(category="interest", preference="طبیعت‌گردی را دوست دارم")]


@pytest.mark.asyncio
async def test_no_unsupported_memory_or_unverified_feature():
    class NoTags(Graph):
        async def get_destination_details(self, name):
            return {"name": name, "description": "جنگلی خلوت", "categories": []}

    assert await explain_selected_plans(TravelGoal(objective="جنگل خلوت"), route("جنگل الف"), NoTags(), None, "u") == ""
    assert await explain_selected_plans(TravelGoal(objective="طبیعت", avoid=["طبیعت"]), route("جنگل الف"), Graph(), None, "u") == ""
    assert await explain_selected_plans(
        TravelGoal(objective="طبیعت؛ اصلاح جدید: به‌جایش سفر تاریخی می‌خواهم"),
        route("جنگل الف"), Graph(), None, "u"
    ) == ""


@pytest.mark.asyncio
async def test_preference_not_loaded_from_another_user_or_from_dislikes():
    async def prefs(user_id):
        assert user_id == "correct-user"
        return [SimpleNamespace(category="avoid", preference="از طبیعت بدم می‌آید")]

    memory = SimpleNamespace(long_term=SimpleNamespace(get_preferences_for=prefs))
    assert await explain_selected_plans(TravelGoal(), route("جنگل الف"), Graph(), memory, "correct-user") == ""


@pytest.mark.asyncio
async def test_new_explicit_interest_overrides_old_preference():
    memory = SimpleNamespace(long_term=SimpleNamespace(get_preferences_for=async_preferences))
    text = await explain_selected_plans(
        TravelGoal(objective="طبیعت؛ اصلاح جدید: به‌جایش سفر تاریخی می‌خواهم"),
        route("جنگل الف"), Graph(), memory, "user-a"
    )
    assert text == ""


@pytest.mark.asyncio
async def test_liked_place_is_not_confused_with_merely_proposed_trips():
    class LikedGraph(Graph):
        async def get_user_history(self, user_id):
            assert user_id == "user-a"
            return {"liked": ["جنگل الف"], "trips": [{"destinations": ["جنگل ب"]}]}

    text = await explain_selected_plans(TravelGoal(), route("جنگل الف"), LikedGraph(), None, "user-a")
    assert "قبلاً هم پسندیده بودی" in text
    assert "جنگل ب" not in text


@pytest.mark.asyncio
async def test_search_excludes_feedback_ids_and_semantic_results():
    class SearchGraph:
        async def get_avoided_destination_ids(self, user_id):
            assert user_id == "user-a"
            return ["disliked-id"]

        async def search_destinations(self, **kwargs):
            assert kwargs["exclude_destination_ids"] == ["disliked-id"]
            return [{"id": "safe-id"}]

        async def semantic_search_destinations(self, goal, limit):
            return [{"id": "disliked-id"}, {"id": "safe-id"}]

    deps = SimpleNamespace(graph=SearchGraph(), user_id="user-a", travel_goal=None, current_query="طبیعت")
    ctx = SimpleNamespace(deps=deps)
    assert await search_destinations(ctx) == [{"id": "safe-id"}]
    assert await semantic_search_destinations(ctx) == [{"id": "safe-id"}]
