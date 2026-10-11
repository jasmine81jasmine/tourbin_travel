"""The visible reason must be supported by the selected place and this user."""

import json
from types import SimpleNamespace

import pytest

from src.agent.goals import TravelGoal
from src.agent.agent import update_travel_goal
from src.agent import personalization
from src.agent.personalization import explain_selected_plans
from src.agent.tools import search_destinations, semantic_search_destinations
from src.memory.graph import TripGraphRepository


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
    assert "جنگل الف" in text and "از قبل یادم هست" in text
    assert "جنگل ب" not in text


async def async_preferences(user_id):
    assert user_id == "user-a"
    return [SimpleNamespace(category="interest", preference="طبیعت‌گردی را دوست دارم")]


@pytest.mark.asyncio
async def test_no_unsupported_memory_or_unverified_feature():
    class NoTags(Graph):
        async def get_destination_details(self, name):
            return {"name": name, "description": "جنگلی خلوت", "categories": []}

    assert "اطلاعی در مورد هیجان ندارم" not in await explain_selected_plans(
        TravelGoal(objective="جنگل خلوت"), route("جنگل الف"), NoTags(), None, "u")
    assert "به خواسته‌ات نزدیک است" not in await explain_selected_plans(
        TravelGoal(objective="طبیعت", avoid=["طبیعت"]), route("جنگل الف"), Graph(), None, "u")
    changed = await explain_selected_plans(
        TravelGoal(objective="طبیعت؛ اصلاح جدید: به‌جایش سفر تاریخی می‌خواهم"),
        route("جنگل الف"), Graph(), None, "u"
    )
    assert "در این سفر به طبیعت گفتی" not in changed


@pytest.mark.asyncio
async def test_preference_not_loaded_from_another_user_or_from_dislikes():
    async def prefs(user_id):
        assert user_id == "correct-user"
        return [SimpleNamespace(category="avoid", preference="از طبیعت بدم می‌آید")]

    memory = SimpleNamespace(long_term=SimpleNamespace(get_preferences_for=prefs))
    text = await explain_selected_plans(TravelGoal(), route("جنگل الف"), Graph(), memory, "correct-user")
    assert "از قبل یادم هست" not in text and "بدم می‌آید" not in text


@pytest.mark.asyncio
async def test_new_explicit_interest_overrides_old_preference():
    memory = SimpleNamespace(long_term=SimpleNamespace(get_preferences_for=async_preferences))
    text = await explain_selected_plans(
        TravelGoal(objective="طبیعت؛ اصلاح جدید: به‌جایش سفر تاریخی می‌خواهم"),
        route("جنگل الف"), Graph(), memory, "user-a"
    )
    assert "از قبل یادم هست طبیعت" not in text


@pytest.mark.asyncio
async def test_liked_place_is_not_confused_with_merely_proposed_trips():
    class LikedGraph(Graph):
        async def get_user_history(self, user_id):
            assert user_id == "user-a"
            return {"liked": ["جنگل الف"], "trips": [{"destinations": ["جنگل ب"]}]}

    text = await explain_selected_plans(TravelGoal(), route("جنگل الف"), LikedGraph(), None, "user-a")
    assert "قبلاً پسندیده بودی" in text
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

    deps = SimpleNamespace(graph=SearchGraph(), user_id="user-a", session_id="session-a",
                           previous_recommendations={"ids": set(), "names": set()},
                           travel_goal=None, current_query="طبیعت")
    ctx = SimpleNamespace(deps=deps)
    assert await search_destinations(ctx) == [{"id": "safe-id"}]
    assert await semantic_search_destinations(ctx) == [{"id": "safe-id"}]


@pytest.mark.asyncio
async def test_camping_evidence_and_prior_preference_in_followup():
    class LatonGraph(Graph):
        async def get_destination_details(self, name):
            return {"name": "آبشار لاتون", "categories": ["طبیعت"], "trip_types": ["کمپ", "بازدید"]}

    memory = SimpleNamespace(long_term=SimpleNamespace(get_preferences_for=async_preferences))
    text = await explain_selected_plans(
        TravelGoal(objective="میخوایم آخر هفته بریم کمپ؛ اصلاح جدید: برای آبشار لاتون برنامه بده",
                   accommodation=["کمپ"]), route("آبشار لاتون"), LatonGraph(), memory, "user-a"
    )
    assert "آبشار لاتون" in text and "کمپ" in text
    assert "از قبل یادم هست" in text and "طبیعت" in text
    assert "هفته پیش" not in text


@pytest.mark.asyncio
async def test_named_place_can_be_explained_without_map_or_tag_match():
    text = await explain_selected_plans(
        TravelGoal(objective="برای مقصد الف برنامه بده"), route("مقصد الف"), Graph(), None, "u"
    )
    assert "مقصد الف" in text and "طبیعت" in text


@pytest.mark.asyncio
async def test_text_lookup_uses_only_names_in_reply_in_display_order():
    repo = TripGraphRepository(None)

    async def read(query, **params):
        assert "NOT d:OsmFeature" in query
        assert "دریاچه نئور" in params["excerpt"]
        return [{"name": "دریاچه نئور"}, {"name": "آبشار لاتون"}, {"name": "سوباتان"}]

    repo._read = read
    names = await repo.find_destinations_mentioned_in_text("آبشار لاتون، سوباتان و دریاچه نئور")
    assert names == ["آبشار لاتون", "سوباتان", "دریاچه نئور"]


@pytest.mark.asyncio
async def test_destination_details_fetches_trip_type_relationship_for_camping_reason():
    repo = TripGraphRepository(None)

    async def read(query, **params):
        assert "[:SUITABLE_FOR]->(tt:TripType)" in query
        assert "trip_types: trip_types" in query
        return [{"destination": {"name": "آبشار لاتون", "trip_types": ["کمپ"]}}]

    repo._read = read
    assert (await repo.get_destination_details("آبشار لاتون"))["trip_types"] == ["کمپ"]


@pytest.mark.asyncio
async def test_session_insights_link_existing_ontology_and_shared_style_nodes():
    calls = []

    class Result:
        async def consume(self):
            pass

    class Transaction:
        async def run(self, query, **params):
            calls.append((query, params))
            return Result()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute_write(self, callback):
            await callback(Transaction())

    class Driver:
        def session(self):
            return Session()

    repo = TripGraphRepository(Driver())
    goal = await update_travel_goal(None, "بهار با ماشین ماجراجویی و هیجان در طبیعت، با دوستام کمپ بریم", "")
    assert goal.moods == ["ماجراجویانه"] and goal.companions == ["دوستان"]
    await repo.sync_session_features("user-a", "session-1", goal.model_dump())

    assert "DELETE r" in calls[0][0] and calls[0][1] == {"user_id": "user-a", "session_id": "session-1"}
    features = {params["names"][0]: query for query, params in calls[1:]}
    assert "MATCH (feature:Category" in features["طبیعت"]
    assert "MATCH (feature:TripType" in features["کمپ"]
    assert "MATCH (feature:Season" in features["بهار"]
    assert "MATCH (feature:Vehicle" in features["سواری"]
    assert "MERGE (feature:RequestedStyle" in features["ماجراجویانه"]
    assert "MERGE (feature:RequestedCompanion" in features["دوستان"]
    assert all("REQUESTED_FOR_TRIP {session_id: $session_id}" in query for query in features.values())
    assert all(params["user_id"] == "user-a" and params["session_id"] == "session-1" for _, params in calls)
    assert not any("HAS_PREFERENCE" in query for query, _ in calls)

    calls.clear()
    revised = goal.model_copy(update={"moods": ["آرام"], "trip_types": [], "accommodation": []})
    await repo.sync_session_features("user-a", "session-1", revised.model_dump())
    assert "DELETE r" in calls[0][0]
    assert not any("ماجراجویانه" in str(params) or "کمپ" in str(params) for _, params in calls)


@pytest.mark.asyncio
async def test_changed_trip_interest_removes_old_session_feature():
    goal = await update_travel_goal(None, "ماجراجویی در طبیعت", "")
    revised = await update_travel_goal(goal, "هیجان نه، طبیعت نه؛ سفر تاریخی و آرام میخوام", "")
    assert "ماجراجویانه" not in revised.moods
    assert "طبیعت" not in revised.categories
    assert "تاریخی" in revised.categories and "آرام" in revised.moods
    assert "ماجراجویانه" in revised.avoid and "طبیعت" in revised.avoid


@pytest.mark.asyncio
async def test_partner_and_friends_are_distinct_from_spouse_or_child():
    partner = await update_travel_goal(None, "با پارتنرم سفر هیجان‌انگیز می‌خوام", "")
    friends = await update_travel_goal(None, "با رفقا سفر می‌ریم", "")
    assert partner.companions == ["پارتنر"]
    assert friends.companions == ["دوستان"]


@pytest.mark.asyncio
async def test_explanation_shows_extracted_trip_insights_and_verified_destination_matches():
    class AdventureGraph(Graph):
        async def get_destination_details(self, name):
            return {"name": name, "categories": ["کوهنوردی", "طبیعت"],
                    "trip_types": ["کمپ"], "seasons": ["بهار"], "vehicles": ["آفرود"]}

    goal = TravelGoal(
        objective="سفر هیجانی با دوستان می‌خواهم",
        categories=["طبیعت"], trip_types=["کمپ"], moods=["ماجراجویانه"],
        companions=["دوستان"], season="بهار", transport="آفرود",
    )
    text = await explain_selected_plans(goal, route("مقصد الف"), AdventureGraph(), None, "user-a")
    assert "### چرا این سفر را پیشنهاد کردم؟" in text
    assert "ماجراجویانه" in text and "دوستان" in text
    assert "کوهنوردی" in text
    assert "جادهٔ آسان" not in text
    assert "گفتگوهای قبلی" not in text
    assert "مناسب دوستان" not in text  # companions are known; group suitability isn't verified


@pytest.mark.asyncio
async def test_adventure_is_not_inferred_from_nature_tag_alone():
    goal = TravelGoal(moods=["ماجراجویانه"], companions=["پارتنر"])
    text = await explain_selected_plans(goal, route("جنگل الف"), Graph(), None, "user-a")
    assert "پارتنر" in text
    assert "دربارهٔ میزان هیجان" not in text
    assert "ماجراجویی" not in text or "به حال‌وهوای پرتحرکی" not in text


@pytest.mark.asyncio
async def test_explanation_credits_prior_request_without_calling_it_current_or_durable():
    class AdventureGraph(Graph):
        async def get_destination_details(self, name):
            return {"name": name, "categories": ["طبیعت", "کوهنوردی"], "trip_types": []}

    goal = TravelGoal(objective="یه سفر پیشنهاد بده", prior_insights={
        "categories": ["طبیعت"], "moods": ["ماجراجویانه"]
    })
    text = await explain_selected_plans(goal, route("کوه الف"), AdventureGraph(), None, "user-a")
    assert "سفر قبلی‌ات" in text and "ماجراجویانه" in text
    assert "در این سفر به طبیعت گفتی" not in text and "همیشه" not in text


@pytest.mark.asyncio
async def test_new_spouse_trip_acknowledges_previous_mood_without_claiming_nature_is_thrilling():
    goal = TravelGoal(duration="سه روز", companions=["همسر"],
                      prior_insights={"moods": ["ماجراجویانه"]})
    text = await explain_selected_plans(goal, route("جنگل الف"), Graph(), None, "user-a")
    intro, why = text.split("**مسیر 1:**")
    assert "قبلی‌ات" in intro and "همسر" in intro
    assert "هیجان" not in why and "دربارهٔ میزان هیجان" not in text


@pytest.mark.asyncio
async def test_one_natural_reason_per_final_route_with_writer_receiving_verified_facts():
    class RoutesGraph(Graph):
        async def get_destination_details(self, name):
            return {"name": name, "categories": ["طبیعت"], "trip_types": [],
                    "description": f"{name} در دامنهٔ البرز قرار دارد."}

    async def writer(payload):
        assert payload["this_trip"]["duration"] == "سه روز"
        assert payload["previous_trip_requests_not_durable"] == {"moods": ["ماجراجویانه"]}
        assert [p["stops"][0]["name"] for p in payload["plans"]] == ["آبشار الف", "جنگل ب"]
        return {"intro": "از گفت‌وگوی قبلی‌مان یادم هست سفر پرتحرک دوست داشتی.",
                "plans": [{"index": 1, "why": "آبشار الف را برای دیدن طبیعت دامنهٔ البرز گذاشتم."},
                          {"index": 2, "why": "جنگل ب را برای گشت‌وگذار در همان محدوده در نظر گرفتم."}]}

    text = await explain_selected_plans(
        TravelGoal(duration="سه روز", prior_insights={"moods": ["ماجراجویانه"]}),
        [*route("آبشار الف"), *route("جنگل ب")], RoutesGraph(), None, "user-a", writer=writer,
    )
    assert text.count("**مسیر 1:**") == 1 and text.count("**مسیر 2:**") == 1
    assert "دیدن طبیعت دامنهٔ البرز" in text
    assert "ویژگی «طبیعت»" not in text and "اطلاعات تأییدشدهٔ کافی ندارم" not in text


@pytest.mark.asyncio
async def test_reason_writer_rejects_invented_memory_and_route_numbers(monkeypatch):
    payload = {"previous_trip_requests_not_durable": {}, "durable_preferences": [],
               "plans": [{"stops": [{"name": "جنگل الف"}]}]}

    class FakeAgent:
        output = ""

        async def run(self, prompt):
            return SimpleNamespace(output=self.output)

    fake = FakeAgent()
    monkeypatch.setattr(personalization, "_reason_writer", lambda: fake)
    fake.output = json.dumps({"intro": "یادم هست همیشه جنگل را دوست داشتی.",
                              "plans": [{"index": 1, "why": "جنگل الف برای تو خوب است."}]}, ensure_ascii=False)
    assert await personalization.write_explanation(payload) is None
    fake.output = json.dumps({"intro": "این بار مسیر کوتاهی پیشنهاد می‌کنم.",
                              "plans": [{"index": 1, "why": "راه ۲ ساعت رانندگی دارد."}]}, ensure_ascii=False)
    assert await personalization.write_explanation(payload) is None
    fake.output = json.dumps({"intro": "این بار طبیعت را انتخاب کردم.",
                              "plans": [{"index": 1, "why": "جنگل الف برای هیجان‌طلبی عالی است."}]}, ensure_ascii=False)
    assert await personalization.write_explanation(payload) is None
    fake.output = json.dumps({"intro": "این بار مسیر طبیعت‌گردی پیشنهاد می‌کنم.",
                              "plans": [{"index": 1, "why": "جنگل الف جاذبهٔ طبیعی مسیر است."}]}, ensure_ascii=False)
    assert (await personalization.write_explanation(payload))["plans"][0]["why"].startswith("جنگل الف")
