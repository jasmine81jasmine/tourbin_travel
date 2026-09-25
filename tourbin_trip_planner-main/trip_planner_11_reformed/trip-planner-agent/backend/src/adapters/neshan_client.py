"""Thin async client for the Neshan Map API services used to make trip
plans geographically realistic: geocoding, no-traffic routing (distance/
duration between stops), TSP (optimal visiting order), isochrone
(reachable-in-time area) and nearby-place search.

Design notes:
- Every public method degrades to `None` (or an empty result) on any
  error -- missing API key, network failure, rate limit, bad response --
  and logs a warning instead of raising. The agent must keep working (using
  DB data + its own general knowledge) even if this external API is down
  or not configured; it should never turn a trip-planning reply into a 500.
- Callers should prefer coordinates already stored on Destination nodes
  (`d.location`) over calling `geocode()`; this client only fills gaps.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.neshan.org"
_TIMEOUT = httpx.Timeout(8.0, connect=4.0)


def decode_polyline(encoded: str) -> list[list[float]]:
    """Decode Google's polyline format into Leaflet [latitude, longitude] points."""
    points: list[list[float]] = []
    latitude = longitude = index = 0
    while index < len(encoded) and len(points) < 20000:
        changes = []
        for _ in range(2):
            value = shift = 0
            while True:
                if index >= len(encoded) or shift > 30:
                    return []
                digit = ord(encoded[index]) - 63
                index += 1
                if digit < 0 or digit > 63:
                    return []
                value |= (digit & 0x1F) << shift
                shift += 5
                if digit < 0x20:
                    break
            changes.append(~(value >> 1) if value & 1 else value >> 1)
        latitude += changes[0]
        longitude += changes[1]
        lat, lon = latitude / 1e5, longitude / 1e5
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            return []
        points.append([lat, lon])
    return points if index == len(encoded) else []


class NeshanClient:
    def __init__(self, api_key: str, base_url: str = _BASE):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        return {"Api-Key": self._api_key}

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=params, headers=self._headers())
            if resp.status_code != 200:
                logger.warning("Neshan %s returned %s: %s", path, resp.status_code, resp.text[:300])
                return None
            return resp.json()
        except Exception:
            logger.warning("Neshan %s request failed", path, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Geocoding: text address/place name -> lat/lon. Only called when a
    # destination has no coordinates in the graph.
    # ------------------------------------------------------------------
    async def geocode(
        self, address: str, city: str | None = None, province: str | None = None
    ) -> dict[str, float] | None:
        if not address or not address.strip():
            return None
        data = await self._get(
            "/geocoding/v1",
            {"address": address, **({"city": city} if city else {}), **({"province": province} if province else {})},
        )
        items = (data or {}).get("items") or []
        if not items:
            return None
        loc = items[0].get("location") or {}
        if loc.get("latitude") is None or loc.get("longitude") is None:
            return None
        return {"latitude": float(loc["latitude"]), "longitude": float(loc["longitude"])}

    # ------------------------------------------------------------------
    # No-traffic routing: real road distance/duration for car travel
    # between an origin, an ordered list of waypoints, and a destination.
    # ------------------------------------------------------------------
    async def route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        waypoints: list[tuple[float, float]] | None = None,
    ) -> dict[str, float] | None:
        params: dict[str, Any] = {
            "type": "car",
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
        }
        if waypoints:
            params["waypoints"] = "|".join(f"{lat},{lon}" for lat, lon in waypoints)
        data = await self._get("/v4/direction/no-traffic", params)
        routes = (data or {}).get("routes") or []
        if not routes:
            return None
        legs = routes[0].get("legs") or []
        if not legs or any(
            (leg.get("distance") or {}).get("value") is None
            or (leg.get("duration") or {}).get("value") is None for leg in legs
        ):
            return None
        distance_m = sum((leg.get("distance") or {}).get("value") or 0 for leg in legs)
        duration_s = sum((leg.get("duration") or {}).get("value") or 0 for leg in legs)
        return {"distance_km": round(distance_m / 1000.0, 1), "duration_hours": round(duration_s / 3600.0, 2)}

    async def route_geometry(
        self, origin: tuple[float, float], destination: tuple[float, float]
    ) -> dict[str, Any] | None:
        """Retrieve actual road geometry for the map; never expose API keys."""
        data = await self._get("/v4/direction/no-traffic", {
            "type": "car",
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
        })
        routes = (data or {}).get("routes") or []
        if not routes or not isinstance(routes[0], dict):
            return None
        route = routes[0]
        legs = route.get("legs") or []
        if not legs or any((leg.get("distance") or {}).get("value") is None
                           or (leg.get("duration") or {}).get("value") is None for leg in legs):
            return None
        geometry: list[list[float]] = []
        incomplete_steps = False
        for leg in legs:
            for step in leg.get("steps") or []:
                encoded = step.get("polyline") or ""
                if not encoded:
                    continue
                chunk = decode_polyline(encoded)
                if not chunk:
                    incomplete_steps = True
                    break
                for point in chunk:
                    if not geometry or point != geometry[-1]:
                        geometry.append(point)
            if incomplete_steps:
                break
        if incomplete_steps or len(geometry) < 2:
            geometry = decode_polyline((route.get("overview_polyline") or {}).get("points") or "")
        if len(geometry) < 2:
            return None
        distance_m = sum(leg["distance"]["value"] for leg in legs)
        duration_s = sum(leg["duration"]["value"] for leg in legs)
        return {"coordinates": geometry,
                "distance_km": round(distance_m / 1000, 1),
                "duration_hours": round(duration_s / 3600, 2)}

    # ------------------------------------------------------------------
    # TSP: given a set of stops, return the visiting order that minimizes
    # total travel (does NOT return geometry/distance -- pair with route()
    # per consecutive leg for that).
    # ------------------------------------------------------------------
    async def trip_order(
        self,
        waypoints: list[tuple[float, float]],
        round_trip: bool = False,
        source_is_any_point: bool = False,
        last_is_any_point: bool = True,
    ) -> list[int] | None:
        if len(waypoints) < 2:
            return list(range(len(waypoints)))
        params = {
            "waypoints": "|".join(f"{lat},{lon}" for lat, lon in waypoints),
            "roundTrip": str(round_trip).lower(),
            "sourceIsAnyPoint": str(source_is_any_point).lower(),
            "lastIsAnyPoint": str(last_is_any_point).lower(),
        }
        data = await self._get("/v3/trip", params)
        points = (data or {}).get("points")
        if not points:
            return None
        try:
            return [int(p["index"]) for p in points]
        except (KeyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Isochrone: the area reachable from a point within a time/distance
    # budget, used to check whether candidate destinations are actually
    # feasible given how much time the user has.
    # ------------------------------------------------------------------
    async def isochrone(
        self,
        lat: float,
        lon: float,
        minutes: float | None = None,
        distance_km: float | None = None,
        dataset: str = "NO_TRAFFIC",
    ) -> dict[str, Any] | None:
        if minutes is None and distance_km is None:
            return None
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "dataset": dataset,
            "polygons": "true",
            "denoise": 0.3,
        }
        if minutes is not None:
            params["time"] = minutes
        else:
            params["distance"] = distance_km
        return await self._get("/v2/isochrone", params)

    # ------------------------------------------------------------------
    # Nearby search: amenities/POIs around a point (restaurant, hotel,
    # parking, ...), used to enrich a destination's description.
    # ------------------------------------------------------------------
    async def nearby(self, lat: float, lon: float, layer: str, radius_m: int = 3000) -> list[dict[str, Any]]:
        data = await self._get(
            "/v1/nearby", {"location": f"{lat},{lon}", "layer": layer, "searchRadius": radius_m}
        )
        points = ((data or {}).get("layerPoints") or {}).get("nearestPoints") or []
        return [
            {
                "name": p.get("name"),
                "distance_m": p.get("distance"),
                "duration_s": p.get("duration"),
                "latitude": (p.get("location") or {}).get("latitude"),
                "longitude": (p.get("location") or {}).get("longitude"),
            }
            for p in points
        ]


@lru_cache
def get_neshan_client() -> NeshanClient:
    """Singleton client, built lazily from settings. `enabled` is False
    (every call short-circuits to None) when NESHAN_API_KEY is unset, so
    it's always safe to fetch and pass this around even without a key."""
    from src.config import get_settings

    settings = get_settings()
    return NeshanClient(
        api_key=settings.neshan_api_key.get_secret_value(),
        base_url=settings.neshan_base_url,
    )
