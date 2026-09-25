"""Ordered map geometry uses server-side Neshan routing, including return legs."""

import importlib

import httpx
import pytest
from fastapi import FastAPI

from src.adapters.neshan_client import NeshanClient, decode_polyline


def test_decodes_encoded_road_polyline():
    assert decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@") == [
        [38.5, -120.2], [40.7, -120.95], [43.252, -126.453]
    ]
    assert decode_polyline("_p~iF") == []  # incomplete coordinate pair


@pytest.mark.asyncio
async def test_neshan_route_geometry_uses_step_polylines(monkeypatch):
    client = NeshanClient("test-key")

    async def fake_get(path, params):
        assert path == "/v4/direction/no-traffic"
        assert params["origin"] == "35.7,51.4"
        return {"routes": [{"legs": [{
            "distance": {"value": 12345}, "duration": {"value": 3600},
            "steps": [{"polyline": "_p~iF~ps|U_ulLnnqC_mqNvxq`@"}],
        }]}]}

    monkeypatch.setattr(client, "_get", fake_get)
    result = await client.route_geometry((35.7, 51.4), (35.8, 51.5))
    assert len(result["coordinates"]) == 3
    assert result["distance_km"] == 12.3
    assert result["duration_hours"] == 1.0


@pytest.mark.asyncio
async def test_map_endpoint_preserves_order_and_routes_back(monkeypatch):
    map_routes = importlib.import_module("src.api.routes.map")

    class FakeMaps:
        enabled = True

        def __init__(self):
            self.calls = []

        async def route_geometry(self, start, end):
            self.calls.append((start, end))
            return {"coordinates": [list(start), list(end)], "distance_km": 10.0, "duration_hours": .25}

    fake = FakeMaps()
    monkeypatch.setattr(map_routes, "get_neshan_client", lambda: fake)
    app = FastAPI()
    app.include_router(map_routes.router, prefix="/api")
    payload = {"origin": {"name": "تهران", "latitude": 35.7, "longitude": 51.4},
               "stops": [{"name": "درکه", "order": 1, "latitude": 35.8, "longitude": 51.5},
                         {"name": "دربند", "order": 2, "latitude": 35.9, "longitude": 51.6}],
               "round_trip": True}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/map/route", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert [(leg["from_name"], leg["to_name"]) for leg in data["legs"]] == [
            ("تهران", "درکه"), ("درکه", "دربند"), ("دربند", "تهران")
        ]
        assert data["total_distance_km"] == 30.0
        assert data["total_duration_hours"] == .75
        assert fake.calls[-1] == ((35.9, 51.6), (35.7, 51.4))

        payload["stops"][1]["order"] = 5
        assert (await client.post("/api/map/route", json=payload)).status_code == 422

        payload["stops"][1]["order"] = 2
        payload["round_trip"] = False
        one_way = (await client.post("/api/map/route", json=payload)).json()
        assert len(one_way["legs"]) == 2


@pytest.mark.asyncio
async def test_map_endpoint_does_not_return_a_partial_route(monkeypatch):
    map_routes = importlib.import_module("src.api.routes.map")

    class FakeMaps:
        enabled = True

        async def route_geometry(self, start, end):
            if start[0] == 35.8:
                return None
            return {"coordinates": [list(start), list(end)], "distance_km": 10, "duration_hours": .2}

    monkeypatch.setattr(map_routes, "get_neshan_client", lambda: FakeMaps())
    app = FastAPI()
    app.include_router(map_routes.router, prefix="/api")
    payload = {"origin": {"name": "تهران", "latitude": 35.7, "longitude": 51.4},
               "stops": [{"name": "درکه", "order": 1, "latitude": 35.8, "longitude": 51.5},
                         {"name": "دربند", "order": 2, "latitude": 35.9, "longitude": 51.6}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/map/route", json=payload)
        assert response.status_code == 502
        assert "legs" not in response.json()
