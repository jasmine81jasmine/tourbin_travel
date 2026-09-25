"""Regression checks for the day-trip / route-text mismatch seen in chat."""

import json

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from src.agent.feasibility import _wants_one_place, driving_budget, ground_route_text, screen_short_trip
from src.agent.goals import TravelGoal
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

    reply, route = await screen_short_trip(
        TravelGoal(duration="یک روز", objective="یه جای نزدیک با طبیعت خوب پیشنهاد بده"),
        FastMaps(), [], "پیشنهاد: " + "، ".join(names), itinerary,
        description_formatter=formatter, graph=Graph(),
    )
    assert route is None  # four independent alternatives, not one journey
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
    reply, itinerary = await screen_short_trip(goal, maps, [], "جایی پیدا نکردم", graph=graph)
    assert graph.calls[0]["location"] == ["گیلان", "مازندران", "گلستان"]
    assert graph.calls[0]["trip_types"] == ["کمپ"]
    assert "### 🌿 برنامهٔ 1" in reply and "### 🌿 برنامهٔ 2" in reply
    assert "**📅 روز 1" in reply and "**📅 روز 2" in reply
    assert "جنگل الف" in reply and "دریاچه ب" in reply
    assert "مجـاز" not in reply  # no invented campsite confirmation
    assert "مجاز بودن کمپ" in reply
    assert "پیشنهاد بده" not in reply
    assert itinerary is None  # two independent complete routes cannot fit one itinerary field
    assert maps.tsp_calls >= 2


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
