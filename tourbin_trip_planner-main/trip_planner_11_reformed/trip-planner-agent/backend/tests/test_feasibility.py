"""Regression checks for the day-trip / route-text mismatch seen in chat."""

import json

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from src.agent.feasibility import driving_budget, ground_route_text, screen_short_trip
from src.agent.goals import TravelGoal
from src.api.schemas import Itinerary


def _details(name="دریاچه سقالکسار"):
    return [ModelRequest(parts=[ToolReturnPart(
        tool_name="tool_get_destination_details",
        content=json.dumps({"name": name, "latitude": 37.15, "longitude": 49.52}),
        tool_call_id="t",
    )])]


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
    assert "تأیید نشد" in reply  # three legs: 4.5h > 4h budget


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
