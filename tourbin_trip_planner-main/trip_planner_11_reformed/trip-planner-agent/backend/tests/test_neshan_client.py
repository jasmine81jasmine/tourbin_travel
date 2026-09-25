"""Unit tests for src/adapters/neshan_client.py.

`_get` (the only method that actually touches the network) is monkeypatched
in every test below, so these never make a real HTTP call -- they check
(a) that a missing API key short-circuits every method to a safe empty
result instead of raising, and (b) that each method's request params and
response parsing are correct, independent of network conditions.
"""

import pytest

from src.adapters.neshan_client import NeshanClient


# ---------------------------------------------------------------------
# No API key configured -> every method must no-op safely, never raise.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_client_never_calls_network(monkeypatch):
    client = NeshanClient(api_key="")
    assert client.enabled is False

    async def _fail_if_called(self, *a, **k):
        raise AssertionError("network should never be reached when disabled")

    # Patch the actual network call (httpx.AsyncClient.get), not `_get` --
    # `_get` itself is what performs the `enabled` short-circuit, so
    # replacing it would bypass the very safeguard this test checks.
    monkeypatch.setattr("httpx.AsyncClient.get", _fail_if_called)

    assert await client.geocode("میدان آزادی") is None
    assert await client.route((35.7, 51.4), (35.6, 51.3)) is None
    assert await client.isochrone(35.7, 51.4, minutes=10) is None
    assert await client.nearby(35.7, 51.4, "park") == []


@pytest.mark.asyncio
async def test_trip_order_with_fewer_than_two_points_skips_network():
    client = NeshanClient(api_key="")  # disabled, but this path never calls _get anyway
    assert await client.trip_order([(35.7, 51.4)]) == [0]
    assert await client.trip_order([]) == []


# ---------------------------------------------------------------------
# Enabled client: verify each method's param building / response parsing.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_geocode_parses_first_item(monkeypatch):
    client = NeshanClient(api_key="fake-key")

    async def fake_get(path, params):
        assert path == "/geocoding/v1"
        assert params["address"] == "میدان آزادی"
        return {"items": [{"location": {"latitude": 35.7, "longitude": 51.3}, "province": "تهران"}]}

    monkeypatch.setattr(client, "_get", fake_get)
    result = await client.geocode("میدان آزادی")
    assert result == {"latitude": 35.7, "longitude": 51.3}


@pytest.mark.asyncio
async def test_geocode_no_items_returns_none(monkeypatch):
    client = NeshanClient(api_key="fake-key")
    monkeypatch.setattr(client, "_get", lambda *a, **k: _async_return({"items": []}))
    assert await client.geocode("جایی که وجود ندارد") is None


@pytest.mark.asyncio
async def test_route_sums_legs(monkeypatch):
    client = NeshanClient(api_key="fake-key")

    async def fake_get(path, params):
        assert path == "/v4/direction/no-traffic"
        assert params["origin"] == "35.7,51.4"
        assert params["destination"] == "35.6,51.3"
        return {
            "routes": [
                {
                    "legs": [
                        {"distance": {"value": 1000.0}, "duration": {"value": 600.0}},
                        {"distance": {"value": 2000.0}, "duration": {"value": 1200.0}},
                    ]
                }
            ]
        }

    monkeypatch.setattr(client, "_get", fake_get)
    result = await client.route((35.7, 51.4), (35.6, 51.3))
    assert result == {"distance_km": 3.0, "duration_hours": 0.5}


@pytest.mark.asyncio
async def test_route_no_routes_returns_none(monkeypatch):
    client = NeshanClient(api_key="fake-key")
    monkeypatch.setattr(client, "_get", lambda *a, **k: _async_return({"routes": []}))
    assert await client.route((35.7, 51.4), (35.6, 51.3)) is None


@pytest.mark.asyncio
async def test_trip_order_parses_indices(monkeypatch):
    client = NeshanClient(api_key="fake-key")

    async def fake_get(path, params):
        assert path == "/v3/trip"
        assert params["waypoints"] == "35.7,51.4|35.6,51.3|35.8,51.5"
        return {"points": [{"index": 2}, {"index": 0}, {"index": 1}]}

    monkeypatch.setattr(client, "_get", fake_get)
    order = await client.trip_order([(35.7, 51.4), (35.6, 51.3), (35.8, 51.5)])
    assert order == [2, 0, 1]


@pytest.mark.asyncio
async def test_isochrone_requires_time_or_distance():
    client = NeshanClient(api_key="fake-key")
    assert await client.isochrone(35.7, 51.4) is None  # neither minutes nor distance_km given


@pytest.mark.asyncio
async def test_nearby_maps_points(monkeypatch):
    client = NeshanClient(api_key="fake-key")

    async def fake_get(path, params):
        assert path == "/v1/nearby"
        assert params["layer"] == "restaurant"
        return {
            "layerPoints": {
                "nearestPoints": [
                    {"name": "رستوران الف", "distance": 500, "duration": 90, "location": {"latitude": 35.7, "longitude": 51.4}}
                ]
            }
        }

    monkeypatch.setattr(client, "_get", fake_get)
    result = await client.nearby(35.7, 51.4, "restaurant")
    assert result == [
        {"name": "رستوران الف", "distance_m": 500, "duration_s": 90, "latitude": 35.7, "longitude": 51.4}
    ]


async def _async_return(value):
    return value
