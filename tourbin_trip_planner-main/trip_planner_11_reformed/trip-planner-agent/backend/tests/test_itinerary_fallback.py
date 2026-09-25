"""Unit tests for:
- linking.extract_finalized_destination_coords (scoped, narrow extraction
  used as the fallback's data source)
- tools.build_itinerary_from_coords (the deterministic, non-TSP fallback
  itinerary builder used in chat.py when the agent skips tool_build_trip_map)
"""

import json

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from src.agent.tools import build_itinerary_from_coords
from src.memory import linking


def _tool_return(tool_name: str, content) -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name=tool_name, content=json.dumps(content), tool_call_id="x")])


def _model_text(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


# ---------------------------------------------------------------------
# extract_finalized_destination_coords
# ---------------------------------------------------------------------


def test_extract_finalized_coords_reads_details_tool():
    messages = [
        _tool_return(
            "tool_get_destination_details",
            {"id": "d1", "name": "دماوند", "latitude": 35.95, "longitude": 52.1},
        ),
        _model_text("here's the plan"),
    ]
    result = linking.extract_finalized_destination_coords(messages)
    assert result == [{"id": "d1", "name": "دماوند", "latitude": 35.95, "longitude": 52.1}]


def test_extract_finalized_coords_reads_find_near_list():
    messages = [
        _tool_return(
            "tool_find_destinations_near",
            [
                {"id": "d1", "name": "دماوند", "latitude": 35.95, "longitude": 52.1, "distance_km": 0},
                {"id": "d2", "name": "لاسم", "latitude": 36.0, "longitude": 52.2, "distance_km": 12},
            ],
        )
    ]
    result = linking.extract_finalized_destination_coords(messages)
    assert [r["id"] for r in result] == ["d1", "d2"]


def test_extract_finalized_coords_ignores_broad_search_tool():
    # tool_search_destinations must NOT contribute -- it's a candidate list,
    # not a finalized selection, and including it would pollute the map
    # with everything the agent ever considered.
    messages = [
        _tool_return(
            "tool_search_destinations",
            [{"id": "d1", "name": "دماوند", "latitude": 35.95, "longitude": 52.1}],
        )
    ]
    assert linking.extract_finalized_destination_coords(messages) == []


def test_extract_finalized_coords_dedupes_by_id_keeping_first():
    messages = [
        _tool_return("tool_get_destination_details", {"id": "d1", "name": "دماوند", "latitude": 1.0, "longitude": 1.0}),
        _tool_return("tool_get_destination_details", {"id": "d1", "name": "دماوند", "latitude": 9.9, "longitude": 9.9}),
    ]
    result = linking.extract_finalized_destination_coords(messages)
    assert len(result) == 1
    assert result[0]["latitude"] == 1.0  # first occurrence kept


def test_extract_finalized_coords_skips_rows_missing_coords():
    messages = [_tool_return("tool_get_destination_details", {"id": "d1", "name": "جایی بی‌مختصات"})]
    assert linking.extract_finalized_destination_coords(messages) == []


def test_extract_finalized_coords_handles_malformed_json_gracefully():
    messages = [ModelRequest(parts=[ToolReturnPart(tool_name="tool_get_destination_details", content="not json", tool_call_id="x")])]
    assert linking.extract_finalized_destination_coords(messages) == []


# ---------------------------------------------------------------------
# build_itinerary_from_coords
# ---------------------------------------------------------------------


def test_build_itinerary_from_coords_orders_and_sums():
    stops = [
        {"id": "d1", "name": "دماوند", "latitude": 35.95, "longitude": 52.1},
        {"id": "d2", "name": "فیروزکوه", "latitude": 35.75, "longitude": 52.77},
    ]
    result = build_itinerary_from_coords(stops)
    assert [s["name"] for s in result["stops"]] == ["دماوند", "فیروزکوه"]
    assert result["stops"][0]["order"] == 1
    assert result["stops"][1]["order"] == 2
    assert result["used_real_routing"] is False
    assert result["total_distance_km"] > 0
    assert result["total_distance_km"] == pytest.approx(
        result["stops"][0]["leg_distance_km_from_previous"] + result["stops"][1]["leg_distance_km_from_previous"],
        abs=0.2,  # total is rounded from the unrounded running sum, legs are each rounded individually
    )


def test_build_itinerary_from_coords_single_stop():
    stops = [{"id": "d1", "name": "دماوند", "latitude": 35.95, "longitude": 52.1}]
    result = build_itinerary_from_coords(stops)
    assert len(result["stops"]) == 1
    assert result["stops"][0]["leg_distance_km_from_previous"] > 0  # distance from default Tehran origin


def test_build_itinerary_from_coords_empty_stops():
    result = build_itinerary_from_coords([])
    assert result["stops"] == []
    assert result["total_distance_km"] == 0.0
