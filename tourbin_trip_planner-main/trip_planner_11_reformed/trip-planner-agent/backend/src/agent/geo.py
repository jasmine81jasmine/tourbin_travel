"""Small, dependency-free geo helpers used as fallbacks when the Neshan
maps API is unavailable/unconfigured, and for local point-in-polygon
checks on isochrone results (no need for shapely for a single test).
"""

from __future__ import annotations

import math

_EARTH_RADIUS_KM = 6371.0
_ASSUMED_ROAD_SPEED_KMH = 55.0  # rough Iranian intercity/mountain-road average


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km. Used only as a fallback estimate when
    the routing API is unreachable -- real road distance is always longer
    than this, so callers should treat it as a lower bound."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def estimate_duration_hours(distance_km: float) -> float:
    """Rough road-time estimate from straight-line distance, only used when
    the real routing API could not be reached."""
    if distance_km <= 0:
        return 0.0
    # Straight-line distance underestimates real road distance; pad it a
    # bit before dividing by an assumed average speed.
    return (distance_km * 1.3) / _ASSUMED_ROAD_SPEED_KMH


def point_in_polygon(lat: float, lon: float, ring: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test.

    `ring` is a list of [longitude, latitude] pairs (GeoJSON coordinate
    order), first and last point identical, exactly as Neshan's isochrone
    endpoint returns a single polygon ring. Good enough for the simple,
    non-self-intersecting polygons an isochrone service returns -- no need
    to pull in a full geometry library (shapely) for one test.
    """
    inside = False
    n = len(ring)
    if n < 3:
        return False
    x, y = lon, lat
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_in_any_polygon(lat: float, lon: float, geojson_feature_collection: dict) -> bool:
    """Check membership against every Polygon feature in an isochrone
    FeatureCollection (handles both Polygon and MultiPolygon geometries)."""
    for feature in (geojson_feature_collection or {}).get("features", []):
        geometry = feature.get("geometry") or {}
        gtype = geometry.get("type")
        coords = geometry.get("coordinates") or []
        if gtype == "Polygon":
            rings = coords
            if rings and point_in_polygon(lat, lon, rings[0]):
                return True
        elif gtype == "MultiPolygon":
            for polygon in coords:
                if polygon and point_in_polygon(lat, lon, polygon[0]):
                    return True
    return False
