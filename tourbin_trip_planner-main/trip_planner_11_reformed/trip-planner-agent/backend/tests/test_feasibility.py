"""Regression checks for the day-trip / route-text mismatch seen in chat."""

import json

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from src.agent.feasibility import _intent_score, _wants_one_place, driving_budget, ground_route_text, screen_short_trip
from src.agent.goals import TravelGoal, recall_recommendations, remember_recommended_routes
from src.api.schemas import Itinerary


def _details(name="دریاچه سقالکسار"):
    return [ModelRequest(parts=[ToolReturnPart(
        tool_name="tool_get_destination_details",
        content=json.dumps({"name": name, "latitude": 37.15, "longitude": 49.52}),
        tool_call_id="t",
    )])]


def test_new_multiday_request_overrides_old_single_place_request():
    assert _wants_one_place(TravelGoal(objective="یه جای نزدیک پیشنهاد بده"))
    assert not _wants_one_place(TravelGoal(
        objective="یه جای نزدیک پیشنهاد بده؛ اصلاح جدید: برای دو روز چند مقصد نزدیک برنامه‌ریزی کن"
    ))
    assert _wants_one_place(TravelGoal(objective="همهٔ جاهای قبلی را رفتم؛ اصلاح جدید: جای جدید معرفی کن"))


def test_two_and_three_day_driving_budgets_are_per_trip_not_per_day():
    assert driving_budget(TravelGoal(duration="دو روز")) == 12
    assert driving_budget(TravelGoal(duration="سه روز")) == 18
    assert driving_budget(TravelGoal(duration="آخر هفته")) == 12


def test_remembers_only_bounded_distinct_routes_in_the_current_goal():
    goal = TravelGoal(duration="دو روز")
    first = {"stops": [{"name": "امامه"}, {"name": "لواسان"}]}
    goal = remember_recommended_routes(goal, [first, first, {"stops": [{"name": "ایگل"}]}])
    assert goal.recommended_routes == [["امامه", "لواسان"], ["ایگل"]]
    assert TravelGoal.model_validate(goal.model_dump()).recommended_routes == goal.recommended_routes


def test_old_chat_can_recover_previously_proposed_options_from_transcript():
    transcript = ("توربین: **ترتیب بازدید:** تهران ← روستای امامه ← لواسان ← تهران\n"
                  "### 🌿 برنامهٔ 2\nترتیب بازدید: تهران ← اوشان فشم ← روستای ایگل ← تهران")
    goal = recall_recommendations(TravelGoal(duration="دو روز"), transcript)
    assert goal.recommended_routes == [["روستای امامه", "لواسان"], ["اوشان فشم", "روستای ایگل"]]
    assert recall_recommendations(goal, transcript).recommended_routes == goal.recommended_routes


def test_adventurous_graph_features_outrank_relaxed_recommendations():
    goal = TravelGoal(moods=["ماجراجویانه"], objective="برنامهٔ هیجانی برای دوستام")
    exciting = {"name": "تنگه", "categories": ["کوهنوردی"], "description": "رودخانه‌نوردی و صخره‌نوردی هیجان‌انگیز"}
    relaxed = {"name": "پارک", "categories": ["تفریح خانوادگی"], "description": "آرامش در پارک شهری"}
    assert _intent_score(exciting, goal) > _intent_score(relaxed, goal)


@pytest.mark.asyncio
async def test_around_tehran_is_preserved_as_graph_location_filter():
    from src.agent.feasibility import _multiday_candidates

    class Graph:
        calls = []

        async def search_destinations(self, **kwargs):
            self.calls.append(kwargs)
            return [{"name": "تنگه واشی", "latitude": 35.8, "longitude": 52.7}]

    graph = Graph()
    goal = TravelGoal(duration="دو روز", objective="کمپ دو روزه اطراف تهران")
    await _multiday_candidates(goal, graph, [], [])
    assert graph.calls[0]["location"] == ["تهران", "البرز", "قزوین"]


@pytest.mark.asyncio
async def test_long_graph_article_becomes_short_plan_card():
    description = "این مقصد طبیعت زیبایی دارد. " + "این جمله طولانی هم باید کامل بماند " * 20
    messages = [ModelRequest(parts=[ToolReturnPart(
        tool_name="tool_get_destination_details",
        content=json.dumps({"name": "دیزین", "latitude": 36.05, "longitude": 51.42, "description": description}),
        tool_call_id="t",
    )])]

    class FastMaps(_Maps):
        async def route(self, origin, destination):
            return {"distance_km": 76.3, "duration_hours": 1.58}

    reply, _ = await screen_short_trip(TravelGoal(duration="یک روز"), FastMaps(), messages, "دیزین")
    assert description.strip() not in reply
    assert "**ویژگی‌ها:** این مقصد طبیعت زیبایی دارد." in reply
    assert "### 🌿" in reply and "#### گزینهٔ 1: دیزین" in reply
    assert "**رانندگی از تهران:**" in reply and "**📅 الگوی یک‌روزه:**" in reply


@pytest.mark.asyncio
async def test_short_trip_uses_formatter_only_for_description_not_route():
    messages = [ModelRequest(parts=[ToolReturnPart(
        tool_name="tool_get_destination_details",
        content=json.dumps({"name": "درکه", "latitude": 35.8, "longitude": 51.4, "description": "متن کامل مقصد."}),
        tool_call_id="t",
    )])]

    class FastMaps(_Maps):
        async def route(self, origin, destination):
            return {"distance_km": 16.1, "duration_hours": 0.33}

    async def formatter(descriptions):
        assert descriptions["درکه"]["description"] == "متن کامل مقصد."
        return {"درکه": "- **ویژگی‌ها:** رودخانه و کوچه‌باغ دارد.\n"
                "- **امکانات:** کافه‌های اطراف دارد."}

    reply, route = await screen_short_trip(
        TravelGoal(duration="یک روز"), FastMaps(), messages, "درکه",
        description_formatter=formatter,
    )
    assert "**ویژگی‌ها:**" in reply and "**امکانات:**" in reply
    assert "16.1 کیلومتر" in reply
    assert route["stops"][0]["name"] == "درکه"


class _Maps:
    enabled = True

    async def route(self, origin, destination):
        return {"distance_km": 233.1, "duration_hours": 5.51}


@pytest.mark.asyncio
async def test_one_day_rejects_long_round_trip_even_when_draft_calls_it_feasible():
    goal = TravelGoal(duration="یک روز")
    result = await screen_short_trip(goal, _Maps(), _details(), "دریاچه سقالکسار برای یک روز عالی است")
    assert driving_budget(goal) == 4
    assert "سقالکسار" not in result[0]
    assert "رفت‌وبرگشت" in result[0]


@pytest.mark.asyncio
async def test_feasible_alternatives_use_route_figures_not_model_guesses():
    class FastMaps(_Maps):
        async def route(self, origin, destination):
            return {"distance_km": 76.3, "duration_hours": 1.58}

    result = await screen_short_trip(TravelGoal(duration="یک روز"), FastMaps(), _details("دیزین"),
                                     "دیزین ۳۲۴ کیلومتر و ۴ ساعت است")
    assert "76.3" in result[0] and "1.58" in result[0]
    assert "۳۲۴" not in result[0]


@pytest.mark.asyncio
async def test_two_individually_near_stops_can_still_exceed_combined_budget():
    itinerary = Itinerary.model_validate({
        "origin": {"name": "تهران", "latitude": 35.68, "longitude": 51.38},
        "stops": [
            {"order": 1, "name": "مقصد اول", "latitude": 36, "longitude": 52},
            {"order": 2, "name": "مقصد دوم", "latitude": 36.1, "longitude": 52.1},
        ],
    })

    class SlowLegs(_Maps):
        async def route(self, origin, destination):
            return {"distance_km": 100, "duration_hours": 1.5}

    reply, _ = await screen_short_trip(TravelGoal(duration="یک روز"), SlowLegs(), [], "مقصد اول و مقصد دوم", itinerary)
    assert "سازگار نیست" in reply  # three legs: 4.5h > 4h budget


@pytest.mark.asyncio
async def test_one_nearby_place_request_does_not_collapse_four_options_to_route_only():
    names = ["رودخانه ولنجک", "رودخانه گلابدره", "رودخانه دربند", "درکه"]
    itinerary = Itinerary.model_validate({
        "origin": {"name": "تهران", "latitude": 35.6892, "longitude": 51.389},
        "stops": [{"order": i, "name": name, "latitude": 35.8 + i * .01, "longitude": 51.4}
                  for i, name in enumerate(names, 1)],
    })

    class FastMaps(_Maps):
        async def route(self, origin, destination):
            return {"distance_km": 16.1, "duration_hours": .33}

    class Graph:
        async def get_destination_details(self, name):
            return {"name": name, "description": f"شرح کامل و خواندنی {name}. نکات بازدید {name}."}

    async def formatter(descriptions):
        assert set(descriptions) == set(names)
        return {name: f"- **ویژگی‌ها:** رودخانهٔ {name} و فضای باز دارد.\n"
                      "- **امکانات:** امکانات ثبت‌شده را پیش از حرکت بررسی کنید."
                for name in descriptions}

    map_options = []
    reply, route = await screen_short_trip(
        TravelGoal(duration="یک روز", objective="یه جای نزدیک با طبیعت خوب پیشنهاد بده"),
        FastMaps(), [], "پیشنهاد: " + "، ".join(names), itinerary,
        description_formatter=formatter, graph=Graph(), map_itineraries=map_options,
    )
    assert route is None  # four independent alternatives, not one journey
    assert [option["stops"][0]["name"] for option in map_options] == names
    assert all(option["round_trip"] and option["return_leg"] for option in map_options)
    for name in names:
        assert name in reply
        assert f"شرح کامل و خواندنی {name}. نکات بازدید {name}." not in reply
        assert f"رودخانهٔ {name} و فضای باز دارد." in reply
    assert reply.count("\n#### گزینهٔ ") == 4
    assert "تهران ←" not in reply
    assert "این‌ها **برنامه‌های جایگزین**" not in reply
    assert reply.count("الگوی یک‌روزه") == 1


@pytest.mark.asyncio
async def test_real_multistop_trip_keeps_descriptions_and_verified_itinerary():
    itinerary = Itinerary.model_validate({
        "origin": {"name": "تهران", "latitude": 35.6892, "longitude": 51.389},
        "stops": [
            {"order": 1, "name": "درکه", "latitude": 35.8, "longitude": 51.4},
            {"order": 2, "name": "دربند", "latitude": 35.81, "longitude": 51.41},
        ],
    })

    class FastMaps(_Maps):
        async def route(self, origin, destination):
            return {"distance_km": 10.0, "duration_hours": .3}

    class Graph:
        async def get_destination_details(self, name):
            return {"name": name, "description": f"متن کامل {name}."}

    async def formatter(descriptions):
        return {name: f"- **ویژگی‌ها:** فضای سبز {name}.\n"
                      "- **امکانات:** امکانات را پیش از حرکت بررسی کنید."
                for name in descriptions}

    reply, route = await screen_short_trip(
        TravelGoal(duration="یک روز", objective="چند توقف نزدیک تهران برای گردش می‌خواهم"),
        FastMaps(), [], "از درکه و دربند دیدن می‌کنیم", itinerary,
        description_formatter=formatter, graph=Graph(),
    )
    assert "متن کامل درکه." not in reply and "متن کامل دربند." not in reply
    assert "**ویژگی‌ها:**" in reply and "0.9 ساعت" in reply
    assert [s["name"] for s in route["stops"]] == ["درکه", "دربند"]
    assert route["total_duration_hours"] == .9 and route["round_trip"] is True


@pytest.mark.asyncio
async def test_two_day_trip_has_two_day_sections_with_ordered_stops():
    itinerary = Itinerary.model_validate({
        "origin": {"name": "تهران", "latitude": 35.6892, "longitude": 51.389},
        "stops": [
            {"order": 1, "name": "مقصد الف", "latitude": 35.8, "longitude": 51.4},
            {"order": 2, "name": "مقصد ب", "latitude": 35.81, "longitude": 51.41},
        ],
    })

    class FastMaps(_Maps):
        async def route(self, origin, destination):
            return {"distance_km": 15.0, "duration_hours": 1.0}

    async def formatter(destinations):
        return {name: "- **ویژگی‌ها:** طبیعت زیبایی دارد.\n"
                      "- **امکانات:** اطلاعات دقیق در دسترس نیست."
                for name in destinations}

    reply, route = await screen_short_trip(
        TravelGoal(duration="دو روز", objective="برنامهٔ سفر دو روزه با چند توقف بده"),
        FastMaps(), [], "مقصد الف و مقصد ب", itinerary, description_formatter=formatter,
    )
    assert "**📅 روز 1" in reply and "**📅 روز 2" in reply
    assert reply.index("#### مقصد الف") < reply.index("#### مقصد ب")
    assert route["total_duration_hours"] == 3.0
    assert route["return_leg"]["leg_duration_hours_from_previous"] == 1.0


@pytest.mark.asyncio
async def test_tsp_order_is_reflected_in_two_day_text_and_itinerary():
    itinerary = Itinerary.model_validate({
        "origin": {"name": "تهران", "latitude": 35.6892, "longitude": 51.389},
        "stops": [
            {"order": 1, "name": "مقصد الف", "latitude": 36.1, "longitude": 51.4},
            {"order": 2, "name": "مقصد ب", "latitude": 36.12, "longitude": 51.42},
        ],
    })

    class Maps(_Maps):
        async def trip_order(self, waypoints, **kwargs):
            return [0, 2, 1]

        async def route(self, origin, destination):
            return {"distance_km": 15.0, "duration_hours": 1.0}

    goal = TravelGoal(duration="دو روز", objective="برنامهٔ دو روزه با چند مقصد")
    reply, route = await screen_short_trip(goal, Maps(), [], "مقصد الف و مقصد ب", itinerary)
    assert [stop["name"] for stop in route["stops"]] == ["مقصد ب", "مقصد الف"]
    assert reply.index("#### مقصد ب") < reply.index("#### مقصد الف")
    assert route["total_duration_hours"] == 3.0


@pytest.mark.asyncio
async def test_two_day_plan_builds_nearby_multistop_route_when_model_skips_map_tool():
    rows = [
        {"name": "مقصد الف", "id": "a", "latitude": 35.8, "longitude": 51.4, "categories": ["طبیعت"]},
        {"name": "مقصد ب", "id": "b", "latitude": 35.85, "longitude": 51.45, "categories": ["طبیعت"]},
    ]
    messages = [ModelRequest(parts=[ToolReturnPart(
        tool_name="tool_search_destinations", content=json.dumps(rows), tool_call_id="t",
    )])]

    class FastMaps(_Maps):
        async def route(self, origin, destination):
            return {"distance_km": 15.0, "duration_hours": 1.0}

    reply, route = await screen_short_trip(
        TravelGoal(duration="دو روز", objective="برای دو روز برنامهٔ سفر با چند مقصد بده"),
        FastMaps(), messages, "مقصد الف را پیشنهاد می‌کنم",
    )
    assert "**📅 روز 1" in reply and "**📅 روز 2" in reply
    assert "#### مقصد الف" in reply and "#### مقصد ب" in reply
    assert route["round_trip"] is True and len(route["stops"]) == 2


@pytest.mark.asyncio
async def test_two_day_camping_discovers_northern_clusters_without_named_destination():
    places = [
        {"id": "a", "name": "جنگل الف", "latitude": 36.4, "longitude": 51.2, "province": "مازندران", "trip_types": ["کمپ"]},
        {"id": "b", "name": "دریاچه ب", "latitude": 36.43, "longitude": 51.23, "province": "مازندران", "trip_types": ["کمپ"]},
        {"id": "c", "name": "روستای ج", "latitude": 36.47, "longitude": 51.25, "province": "مازندران", "trip_types": ["کمپ"]},
    ]

    class Graph:
        calls = []

        async def search_destinations(self, **kwargs):
            self.calls.append(kwargs)
            return places

        async def get_destination_details(self, name):
            row = next(row for row in places if row["id"] == name or row["name"] == name)
            return {**row, "description": f"طبیعت سرسبز {row['name']} دارد.", "facilities_level": "محدود"}

    class Maps(_Maps):
        tsp_calls = 0

        async def trip_order(self, waypoints, **kwargs):
            self.tsp_calls += 1
            return list(range(len(waypoints)))

        async def route(self, origin, destination):
            hours = 4.5 if 35.6892 in (origin[0], destination[0]) else .5
            return {"distance_km": hours * 50, "duration_hours": hours}

    graph, maps = Graph(), Maps()
    goal = TravelGoal(duration="دو روز", region="شمال", objective="میخوام دو روز آخر هفته برم کمپ؛ اصلاح جدید: پیشنهاد بده")
    map_options = []
    reply, itinerary = await screen_short_trip(goal, maps, [], "جایی پیدا نکردم", graph=graph,
                                               map_itineraries=map_options)
    assert graph.calls[0]["location"] == ["گیلان", "مازندران", "گلستان"]
    assert graph.calls[0]["trip_types"] == ["کمپ"]
    assert "### 🌿 برنامهٔ 1" in reply and "### 🌿 برنامهٔ 2" in reply
    assert "**📅 روز 1" in reply and "**📅 روز 2" in reply
    assert "جنگل الف" in reply and "دریاچه ب" in reply
    assert "مجـاز" not in reply  # no invented campsite confirmation
    assert "مجاز بودن کمپ" in reply
    assert "پیشنهاد بده" not in reply
    assert itinerary is None  # two independent complete routes cannot fit one itinerary field
    assert len(map_options) == 2
    assert all(len(option["stops"]) >= 2 and option["round_trip"] for option in map_options)
    assert maps.tsp_calls >= 2


@pytest.mark.asyncio
async def test_exciting_followup_finds_new_graph_clusters_instead_of_rebranding_old_plans():
    old = [
        {"id": "old1", "name": "روستای امامه", "latitude": 35.8, "longitude": 51.5,
         "categories": ["طبیعت"], "trip_types": ["کمپ"]},
        {"id": "old2", "name": "لواسان", "latitude": 35.81, "longitude": 51.51,
         "categories": ["تفریح خانوادگی"], "trip_types": ["کمپ"]},
        {"id": "old3", "name": "اوشان فشم", "latitude": 35.85, "longitude": 51.53,
         "categories": ["طبیعت"], "trip_types": ["کمپ"]},
        {"id": "old4", "name": "روستای ایگل", "latitude": 35.87, "longitude": 51.55,
         "categories": ["طبیعت"], "trip_types": ["کمپ"]},
    ]
    adventurous = [
        {"id": "new1", "name": "تنگه واشی", "latitude": 35.8, "longitude": 52.7,
         "categories": ["طبیعت"], "trip_types": ["کمپ"],
         "description": "مسیر هیجان‌انگیز رودخانه‌نوردی و صخره‌نوردی دارد."},
        {"id": "new2", "name": "آبشار واشی", "latitude": 35.82, "longitude": 52.72,
         "categories": ["ماجراجویی"], "description": "مسیر کوهنوردی و آبشار دارد."},
        {"id": "new3", "name": "دریاچه تار", "latitude": 35.95, "longitude": 52.1,
         "categories": ["کوهنوردی"], "trip_types": ["کمپ"], "description": "مسیر آفرود دارد."},
        {"id": "new4", "name": "دریاچه هویر", "latitude": 35.98, "longitude": 52.12,
         "categories": ["کوهنوردی"], "description": "سفر ماجراجویانه است."},
    ]

    class Graph:
        semantic_queries = []

        async def semantic_search_destinations(self, query, limit):
            self.semantic_queries.append(query)
            return [{**row, "semantic_score": .9} for row in adventurous]

        async def search_destinations(self, **kwargs):
            return old + adventurous

        async def get_destination_details(self, key):
            return next(row for row in old + adventurous if key in (row["id"], row["name"]))

    class Maps(_Maps):
        async def trip_order(self, waypoints, **kwargs):
            return list(range(len(waypoints)))

        async def route(self, origin, destination):
            hours = 3.0 if 35.6892 in (origin[0], destination[0]) else .5
            return {"distance_km": hours * 50, "duration_hours": hours}

    goal = TravelGoal(duration="دو روز", origin="تهران", moods=["ماجراجویانه"],
                      objective="سلام دو روز کمپ با دوستام؛ اصلاح جدید: برنامه هیجانی دیگه چی داری",
                      semantic_query="دو روز کمپ هیجانی ماجراجویانه",
                      recommended_routes=[["روستای امامه", "لواسان"], ["اوشان فشم", "روستای ایگل"]])
    graph, maps = Graph(), Maps()
    routes = []
    reply, selected = await screen_short_trip(goal, maps, [], "برنامه‌های تازه", graph=graph,
                                               map_itineraries=routes)
    assert graph.semantic_queries == [goal.semantic_query]
    assert routes and len(routes) == 2 and selected is None
    assert "تنگه واشی" in reply or "دریاچه تار" in reply
    assert all(not {stop["name"] for stop in route["stops"]} &
               {"روستای امامه", "لواسان", "اوشان فشم", "روستای ایگل"} for route in routes)
    assert all(route["total_duration_hours"] <= 12 for route in routes)


def test_explicit_place_can_be_revisited_even_if_it_was_suggested_before():
    from src.agent.feasibility import _allow_revisit

    goal = TravelGoal(objective="کمپ دو روزه؛ اصلاح جدید: برای روستای ایگل برنامه بده",
                      recommended_routes=[["اوشان فشم", "روستای ایگل"]])
    assert _allow_revisit(goal)
    assert not _allow_revisit(goal.model_copy(update={"objective": "کمپ؛ اصلاح جدید: برنامه‌های دیگه چی داری"}))


@pytest.mark.asyncio
async def test_repeat_day_trip_request_uses_unseen_graph_destinations():
    old = {"name": "درکه", "latitude": 35.8, "longitude": 51.4, "categories": ["طبیعت"]}
    fresh = {"name": "آبشار کمرد", "latitude": 35.85, "longitude": 51.55, "categories": ["طبیعت"]}
    messages = [ModelRequest(parts=[ToolReturnPart(
        tool_name="tool_get_destination_details", content=json.dumps(old), tool_call_id="t",
    )])]

    class Graph:
        async def search_destinations(self, **kwargs):
            return [old, fresh]

        async def get_destination_details(self, name):
            return {**fresh, "description": "آبشار با مسیر طبیعت‌گردی است."}

    class Maps(_Maps):
        async def route(self, origin, destination):
            return {"distance_km": 25.0, "duration_hours": .6}

    goal = TravelGoal(duration="یک روز", objective="یک روز طبیعت؛ اصلاح جدید: یه جای جدید پیشنهاد بده",
                      recommended_routes=[["درکه"]])
    reply, itinerary = await screen_short_trip(goal, Maps(), messages, "درکه را پیشنهاد می‌کنم", graph=Graph())
    assert "آبشار کمرد" in reply and "#### گزینهٔ 1: درکه" not in reply
    assert itinerary["stops"][0]["name"] == "آبشار کمرد"


@pytest.mark.asyncio
async def test_three_day_plan_accepts_longer_verified_round_trip():
    places = [
        {"id": str(i), "name": f"مقصد {i}", "latitude": 36.3 + i * .02,
         "longitude": 51.2 + i * .02, "categories": ["طبیعت"]} for i in range(3)
    ]

    class Graph:
        async def search_destinations(self, **kwargs):
            return places

        async def get_destination_details(self, key):
            row = next(row for row in places if key in (row["id"], row["name"]))
            return {**row, "description": "این مقصد طبیعت سرسبز دارد."}

    class Maps(_Maps):
        async def trip_order(self, waypoints, **kwargs):
            return list(range(len(waypoints)))

        async def route(self, origin, destination):
            hours = 5.5 if 35.6892 in (origin[0], destination[0]) else .5
            return {"distance_km": hours * 50, "duration_hours": hours}

    goal = TravelGoal(duration="سه روز", objective="سه روز برای یک سفر طبیعت گردی وقت دارم")
    reply, _ = await screen_short_trip(goal, Maps(), [], "پیشنهادی ندارم", graph=Graph())
    assert "**📅 روز 1" in reply and "**📅 روز 2" in reply and "**📅 روز 3" in reply
    assert "مجموع رانندگی" in reply and "12.0 ساعت" in reply


@pytest.mark.asyncio
async def test_discovers_nearby_nature_when_graph_candidate_too_far():
    class NearbyMaps(_Maps):
        async def isochrone(self, lat, lon, minutes=None):
            assert minutes == 100  # less than half of four hours outbound
            ring = [[51.1, 35.4], [51.8, 35.4], [51.8, 36], [51.1, 36], [51.1, 35.4]]
            return {"features": [{"geometry": {"type": "Polygon", "coordinates": [ring]}}]}

        async def nearby(self, lat, lon, layer, radius):
            assert layer in ("natural_feature", "park", "garden")
            assert radius == 7000
            return [{"name": "بوستان نزدیک", "latitude": 35.75, "longitude": 51.45}] if layer == "park" else []

        async def route(self, origin, destination):
            if 37.15 in (origin[0], destination[0]):
                return {"distance_km": 233.1, "duration_hours": 5.51}
            return {"distance_km": 23.0, "duration_hours": 0.7}

    reply, route = await screen_short_trip(
        TravelGoal(duration="یک روز", objective="طبیعت نزدیک تهران"), NearbyMaps(),
        _details(), "دریاچه سقالکسار برای یک روز خوب است",
    )
    assert "دریاچه سقالکسار" not in reply
    assert "بوستان نزدیک" in reply
    assert route["stops"][0]["name"] == "بوستان نزدیک"
    assert route["round_trip"] is True and route["total_duration_hours"] == 1.4


@pytest.mark.asyncio
async def test_nearby_cannot_certify_trip_without_return_routing():
    class NoReturnMaps(_Maps):
        async def isochrone(self, lat, lon, minutes=None):
            return None

        async def nearby(self, lat, lon, layer, radius):
            return [{"name": "پارک", "latitude": 35.75, "longitude": 51.45}]

        async def route(self, origin, destination):
            return None if origin[0] == 35.75 else {"distance_km": 23, "duration_hours": 0.7}

    reply, route = await screen_short_trip(
        TravelGoal(duration="یک روز", objective="طبیعت"), NoReturnMaps(), [], "چند پیشنهاد می‌خواهم"
    )
    assert route is None
    assert "پارک" not in reply


def test_route_text_replaces_inconsistent_numbers_and_preserves_place_details():
    itinerary = Itinerary.model_validate({
        "origin": {"name": "تهران", "latitude": 35.68, "longitude": 51.38},
        "stops": [{"order": 1, "name": "دریاچه سقالکسار", "latitude": 37.15, "longitude": 49.52,
                   "leg_distance_km_from_previous": 233.1, "leg_duration_hours_from_previous": 5.51}],
        "total_distance_km": 233.1, "total_duration_hours": 5.51,
    })
    result = ground_route_text("دریاچه منظره زیبایی دارد. فاصله ۳۲۴ کیلومتر و زمان ۴ ساعت است.", itinerary)
    assert "منظره زیبایی دارد" in result
    assert "۳۲۴" not in result and "۴ ساعت" not in result
    assert "233.1" in result and "5.51" in result
