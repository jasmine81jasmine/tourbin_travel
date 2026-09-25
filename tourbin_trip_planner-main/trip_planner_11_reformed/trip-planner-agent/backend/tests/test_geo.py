"""Unit tests for src/agent/geo.py -- pure functions, no mocking needed."""

from src.agent.geo import (
    estimate_duration_hours,
    haversine_km,
    point_in_any_polygon,
    point_in_polygon,
)


def test_haversine_km_known_distance():
    # Tehran -> Isfahan, straight-line distance is well documented (~338km)
    d = haversine_km(35.6892, 51.3890, 32.6546, 51.6680)
    assert 330 <= d <= 345


def test_haversine_km_zero_for_same_point():
    assert haversine_km(35.7, 51.4, 35.7, 51.4) == 0


def test_estimate_duration_hours_scales_with_distance():
    assert estimate_duration_hours(0) == 0.0
    short = estimate_duration_hours(50)
    long = estimate_duration_hours(500)
    assert 0 < short < long


def test_point_in_polygon_simple_square():
    ring = [[50, 35], [52, 35], [52, 36], [50, 36], [50, 35]]  # [lon, lat] pairs
    assert point_in_polygon(35.5, 51, ring) is True
    assert point_in_polygon(35.5, 60, ring) is False


def test_point_in_polygon_degenerate_ring_is_false():
    assert point_in_polygon(35.5, 51, [[50, 35], [52, 35]]) is False


def test_point_in_any_polygon_feature_collection():
    ring = [[50, 35], [52, 35], [52, 36], [50, 36], [50, 35]]
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}],
    }
    assert point_in_any_polygon(35.5, 51, fc) is True
    assert point_in_any_polygon(10, 10, fc) is False


def test_point_in_any_polygon_multipolygon():
    ring = [[50, 35], [52, 35], [52, 36], [50, 36], [50, 35]]
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPolygon", "coordinates": [[ring]]}}
        ],
    }
    assert point_in_any_polygon(35.5, 51, fc) is True


def test_point_in_any_polygon_empty_collection():
    assert point_in_any_polygon(35.5, 51, {"type": "FeatureCollection", "features": []}) is False
    assert point_in_any_polygon(35.5, 51, {}) is False
