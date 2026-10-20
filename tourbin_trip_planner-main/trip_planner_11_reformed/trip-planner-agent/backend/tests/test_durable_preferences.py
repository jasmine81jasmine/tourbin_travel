"""Cross-session preference precedence and graph-backed recommendation signals."""

from types import SimpleNamespace
import json

import pytest

from src.agent.goals import TravelGoal
from src.agent.preference_context import effective_preferences
from src.agent import preference_extraction
from src.agent.variety import vary_candidates
from src.memory.graph import TripGraphRepository


def test_current_trip_overrides_only_conflicting_memory_dimensions():
    stored = [
        {"category": "style", "preference": "سفرهای آرام", "feature_type": "RequestedStyle", "feature_name": "آرام"},
        {"category": "category", "preference": "جنگل", "feature_type": "Category", "feature_name": "جنگل"},
        {"category": "season", "preference": "پاییز", "feature_type": "Season", "feature_name": "پاییز"},
    ]
    assert effective_preferences(stored, TravelGoal()) == stored
    assert effective_preferences(stored, TravelGoal(moods=["ماجراجویانه"], avoid=["جنگل"])) == stored[2:]
    assert effective_preferences(stored, TravelGoal(categories=["تاریخی"])) == [stored[0], stored[2]]


@pytest.mark.asyncio
async def test_extraction_validates_llm_output(monkeypatch):
    class Extractor:
        async def run(self, message):
            payload = json.loads(message)
            assert payload["available_graph_features"]["Category"] == ["جنگل"]
            return SimpleNamespace(output='{"preferences":['
                '{"category":"category","preference":"جنگل را دوست دارم",'
                '"feature_type":"Category","feature_name":"جنگل","polarity":"like"},'
                '{"category":"evil label","preference":"bad","polarity":"like"}]}' )

    monkeypatch.setattr(preference_extraction, "_extractor", lambda: Extractor())
    result = await preference_extraction.extract_travel_memory(
        "من همیشه جنگل را دوست دارم", {"Category": ["جنگل"]},
    )
    assert result["preferences"] == [
        {"category": "category", "preference": "جنگل را دوست دارم", "feature_type": "Category",
         "feature_name": "جنگل", "polarity": "like"}
    ]


@pytest.mark.asyncio
async def test_llm_semantic_mapping_requires_real_feature_and_literal_user_phrase(monkeypatch):
    class Extractor:
        async def run(self, message):
            assert "Category" in json.loads(message)["available_graph_features"]
            return SimpleNamespace(output=json.dumps({
                "preferences": [{"category": "category", "preference": "عاشق طبیعت هستم",
                                 "polarity": "like", "feature_type": "Category", "feature_name": "طبیعت"},
                                {"category": "trip_type", "preference": "زیارت",
                                 "polarity": "like", "feature_type": "TripType", "feature_name": "زیارت"}],
                "mentions": [{"text": "جاهای سرسبز", "feature_type": "Category", "feature_name": "طبیعت"},
                             {"text": "ساحل", "feature_type": "Category", "feature_name": "طبیعت"},
                             {"text": "جاهای سرسبز", "feature_type": "Category", "feature_name": "ناشناخته"}],
            }, ensure_ascii=False))

    monkeypatch.setattr(preference_extraction, "_extractor", lambda: Extractor())
    result = await preference_extraction.extract_travel_memory(
        "من عاشق طبیعت هستم، جاهای سرسبز می‌خوام", {"Category": ["طبیعت"]},
    )
    assert result["preferences"][0]["feature_name"] == "طبیعت"
    assert result["preferences"][1]["feature_name"] == ""  # hallucinated node cannot link
    assert result["mentions"] == [{"text": "جاهای سرسبز", "feature_type": "Category",
                                   "feature_name": "طبیعت"}]


@pytest.mark.asyncio
async def test_preference_write_links_user_message_and_existing_ontology():
    calls = []

    class Tx:
        async def run(self, query, **params):
            calls.append((query, params))
            return SimpleNamespace(consume=consume)

    async def consume():
        return None

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute_write(self, callback):
            await callback(Tx())

    repo = TripGraphRepository(SimpleNamespace(session=lambda: Session()))
    await repo.save_durable_preferences("user-a", [{
        "category": "category", "preference": "دوستدار طبیعت", "polarity": "like",
        "feature_type": "Category", "feature_name": "طبیعت",
    }], "message-a")
    assert any("HAS_PREFERENCE" in query and params["user_id"] == "user-a" for query, params in calls)
    assert any("LEARNED_FROM" in query and params["message_id"] == "message-a" for query, params in calls)
    assert any("MATCH (f:Category {name: $name})" in query and params["name"] == "طبیعت"
               for query, params in calls)
    assert any("SUPERSEDED_BY" in query for query, _ in calls)


@pytest.mark.asyncio
async def test_ranking_only_uses_this_users_active_graph_paths():
    repo = TripGraphRepository(None)

    async def read(query, **params):
        assert "User {identifier: $user_id}" in query
        assert "HAS_FEATURE" in query and "HAS_CATEGORY" in query
        assert params == {"user_id": "user-a", "ids": ["forest", "city"],
                          "values": ["دوستدار طبیعت"]}
        return [{"id": "forest", "score": 1}]

    repo._read = read
    assert await repo.rank_destinations_by_preferences(
        "user-a", ["forest", "city"], [{"preference": "دوستدار طبیعت"}]
    ) == {"forest": 1}


def test_graph_preference_ranking_survives_candidate_variety():
    rows = [{"id": "other", "user_rating_score": 4},
            {"id": "forest", "user_rating_score": 4},
            {"id": "osm", "source_is_osm": True, "user_rating_score": 5}]
    ranked = vary_candidates(rows, {"ids": set(), "names": set()}, "u", "s", 3,
                             preference_scores={"forest": 2, "osm": 9})
    assert [row["id"] for row in ranked] == ["forest", "other", "osm"]


@pytest.mark.asyncio
async def test_grounded_mention_is_user_scoped_and_points_to_real_feature():
    repo = TripGraphRepository(None)
    calls = []

    async def write(query, **params):
        calls.append((query, params))
        return []

    repo._write = write
    await repo.link_grounded_mentions("user-a", "message-a", [
        {"text": "جاهای سرسبز", "feature_type": "Category", "feature_name": "طبیعت"}
    ])
    query, params = calls[0]
    assert "HAS_CONVERSATION" in query and "MENTIONS" in query and "REFERS_TO" in query
    assert "MATCH (f:Category {name: $feature_name})" in query
    assert params == {"user_id": "user-a", "message_id": "message-a", "text": "جاهای سرسبز",
                      "entity_type": "TRIP_CATEGORY", "feature_name": "طبیعت"}


@pytest.mark.asyncio
async def test_llm_receives_only_destination_backed_feature_vocabulary():
    repo = TripGraphRepository(None)
    queries = []

    async def read(query, **params):
        queries.append(query)
        return [{"name": "طبیعت"}] if ":Category" in query else []

    repo._read = read
    vocabulary = await repo.list_preference_features()
    assert vocabulary["Category"] == ["طبیعت"]
    assert vocabulary["TripType"] == []
    assert "خلوت" in vocabulary["RequestedStyle"]
    assert all("(:Destination)-[:" in query for query in queries)


@pytest.mark.asyncio
async def test_unfamiliar_phrasing_still_reaches_llm_intent(monkeypatch):
    class Classifier:
        async def run(self, message):
            assert message == "جاهای سرسبز برای گشت دوست دارم"
            return SimpleNamespace(output='{"travel_related":true}')

    monkeypatch.setattr(preference_extraction, "_intent_agent", lambda: Classifier())
    assert await preference_extraction.is_travel_message("جاهای سرسبز برای گشت دوست دارم")
