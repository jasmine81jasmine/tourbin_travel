"""Unit tests for the new map-grounding tools in src/agent/tools.py.

We don't build a real `pydantic_ai.RunContext` here -- the functions under
test only ever read `ctx.deps`, so a tiny stand-in with just that attribute
is enough and keeps these tests independent of the agent-framework's own
construction requirements.
"""

import pytest

from src.adapters.neshan_client import NeshanClient
from src.agent import tools


class _Ctx:
    """Minimal stand-in for pydantic_ai.RunContext -- only `.deps` is used."""

    def __init__(self, deps):
        self.deps = deps


class _Deps:
    def __init__(self, graph=None, maps=None):
        self.graph = graph
        self.maps = maps if maps is not None else NeshanClient(api_key="")  # disabled by default
        self.itinerary_result = None


# ---------------------------------------------------------------------
# _resolve_coordinates: explicit coords > DB by id > DB by name > geocode
# ---------------------------------------------------------------------


class _FakeGraph:
    async def get_coordinates_by_ids(self, ids):
        if "dest1" in ids:
            return {"dest1": {"id": "dest1", "name": "X", "latitude": 36.0, "longitude": 53.0}}
        return {}

    async def get_destination_details(self, name):
        if name == "known-by-name":
            return {"latitude": 37.0, "longitude": 54.0}
        return None


class _FakeMapsGeocode:
    enabled = True

    async def geocode(self, address, city=None, province=None):
        return {"latitude": 40.0, "longitude": 45.0}


@pytest.mark.asyncio
async def test_resolve_coordinates_prefers_explicit_over_everything():
    ctx = _Ctx(_Deps(graph=_FakeGraph(), maps=_FakeMapsGeocode()))
    result = await tools._resolve_coordinates(ctx, "whatever", latitude=1.0, longitude=2.0, destination_id="dest1")
    assert result == {"latitude": 1.0, "longitude": 2.0}


@pytest.mark.asyncio
async def test_resolve_coordinates_id_beats_name_and_geocode():
    ctx = _Ctx(_Deps(graph=_FakeGraph(), maps=_FakeMapsGeocode()))
    result = await tools._resolve_coordinates(ctx, "unrelated-name", destination_id="dest1")
    assert result == {"latitude": 36.0, "longitude": 53.0}


@pytest.mark.asyncio
async def test_resolve_coordinates_name_beats_geocode():
    ctx = _Ctx(_Deps(graph=_FakeGraph(), maps=_FakeMapsGeocode()))
    result = await tools._resolve_coordinates(ctx, "known-by-name")
    assert result == {"latitude": 37.0, "longitude": 54.0}


@pytest.mark.asyncio
async def test_resolve_coordinates_falls_back_to_geocode():
    ctx = _Ctx(_Deps(graph=_FakeGraph(), maps=_FakeMapsGeocode()))
    result = await tools._resolve_coordinates(ctx, "totally-unknown-place")
    assert result == {"latitude": 40.0, "longitude": 45.0}


@pytest.mark.asyncio
async def test_resolve_coordinates_returns_none_when_nothing_matches():
    ctx = _Ctx(_Deps())  # no graph, maps disabled
    result = await tools._resolve_coordinates(ctx, "nowhere")
    assert result is None


# ---------------------------------------------------------------------
# build_trip_map: fallback path (no Neshan) and Neshan-enabled path
# ---------------------------------------------------------------------

_DAMAVAND = {"name": "دماوند", "latitude": 35.95, "longitude": 52.1}
_FIROOZKOOH = {"name": "فیروزکوه", "latitude": 35.75, "longitude": 52.77}


@pytest.mark.asyncio
async def test_build_trip_map_fallback_orders_and_sums_distances():
    ctx = _Ctx(_Deps())  # maps disabled -> haversine fallback
    result = await tools.build_trip_map(ctx, [_DAMAVAND, _FIROOZKOOH])

    assert result["used_real_routing"] is False
    assert [s["name"] for s in result["stops"]] == ["دماوند", "فیروزکوه"]
    assert result["total_distance_km"] == pytest.approx(134.8, abs=1.0)
    assert result["total_duration_hours"] > 0
    # side effect: the tool must persist its result for the API layer to read back
    assert ctx.deps.itinerary_result == result


@pytest.mark.asyncio
async def test_build_trip_map_single_stop():
    ctx = _Ctx(_Deps())
    result = await tools.build_trip_map(ctx, [_DAMAVAND])
    assert len(result["stops"]) == 1
    assert result["stops"][0]["order"] == 1


@pytest.mark.asyncio
async def test_build_trip_map_unresolved_stop_reports_error_not_crash():
    ctx = _Ctx(_Deps())  # no graph, maps disabled -> nothing can resolve a bare name
    result = await tools.build_trip_map(ctx, [{"name": "جایی که مختصات ندارد"}])
    assert "error" in result
    assert "جایی که مختصات ندارد" in result["unresolved"]


class _FakeMapsTSP:
    enabled = True

    async def trip_order(self, waypoints, round_trip=False, source_is_any_point=False, last_is_any_point=True):
        # waypoints[0] is the origin; API decides to visit resolved-index 1 (Firoozkooh) before 0 (Damavand)
        return [0, 2, 1]

    async def route(self, origin, destination):
        return {"distance_km": 111.1, "duration_hours": 2.22}

    async def geocode(self, *a, **k):
        return None


@pytest.mark.asyncio
async def test_build_trip_map_uses_tsp_order_and_real_routing():
    ctx = _Ctx(_Deps(maps=_FakeMapsTSP()))
    result = await tools.build_trip_map(ctx, [_DAMAVAND, _FIROOZKOOH])

    assert result["used_real_routing"] is True
    # TSP said visit index 2 (Firoozkooh, resolved[1]) before index 1 (Damavand, resolved[0])
    assert [s["name"] for s in result["stops"]] == ["فیروزکوه", "دماوند"]
    assert result["total_distance_km"] == pytest.approx(222.2)


# ---------------------------------------------------------------------
# check_reachable_within_time
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_reachable_within_time_fallback_distinguishes_candidates():
    ctx = _Ctx(_Deps())  # maps disabled -> haversine + average-speed estimate
    origin = {"name": "تهران", "latitude": 35.6892, "longitude": 51.389}
    results = await tools.check_reachable_within_time(ctx, origin, minutes=120, candidates=[_DAMAVAND, _FIROOZKOOH])

    by_name = {r["name"]: r for r in results}
    assert by_name["دماوند"]["reachable"] is True
    assert by_name["فیروزکوه"]["reachable"] is False


@pytest.mark.asyncio
async def test_check_reachable_within_time_unknown_origin_returns_none_verdicts():
    ctx = _Ctx(_Deps())  # no graph, maps disabled -> origin with only a name can't be resolved
    results = await tools.check_reachable_within_time(ctx, {"name": "جایی نامعلوم"}, 60, [_DAMAVAND])
    assert results[0]["reachable"] is None


class _FakeMapsIsochrone:
    enabled = True

    async def isochrone(self, lat, lon, minutes=None, distance_km=None, dataset="NO_TRAFFIC"):
        # a square roughly covering Damavand but not Firoozkooh
        ring = [[51.8, 35.8], [52.3, 35.8], [52.3, 36.1], [51.8, 36.1], [51.8, 35.8]]
        return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}


@pytest.mark.asyncio
async def test_check_reachable_within_time_uses_isochrone_polygon():
    ctx = _Ctx(_Deps(maps=_FakeMapsIsochrone()))
    origin = {"name": "تهران", "latitude": 35.6892, "longitude": 51.389}
    results = await tools.check_reachable_within_time(ctx, origin, minutes=90, candidates=[_DAMAVAND, _FIROOZKOOH])
    by_name = {r["name"]: r for r in results}
    assert by_name["دماوند"]["reachable"] is True
    assert by_name["فیروزکوه"]["reachable"] is False


# ---------------------------------------------------------------------
# find_nearby_amenities
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_nearby_amenities_skips_gracefully_when_disabled():
    ctx = _Ctx(_Deps())  # maps disabled
    result = await tools.find_nearby_amenities(ctx, 35.7, 51.4, "restaurant")
    assert result == {"status": "skipped", "reason": "maps API not configured"}
