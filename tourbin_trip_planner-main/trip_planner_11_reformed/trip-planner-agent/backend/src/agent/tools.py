"""Tool implementations for the trip planner agent.

Kept separate from agent.py (which only registers @agent.tool wrappers) so
the query/business logic is easy to unit test independently of pydantic-ai.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import jdatetime
from pydantic_ai import RunContext

from src.agent.dependencies import AgentDeps
from src.agent.geo import estimate_duration_hours, haversine_km, point_in_any_polygon
from src.agent.regions import expand_location

# Tehran city-center coordinates, used as the default trip origin when the
# user hasn't given/updated one (matches SYSTEM_PROMPT's default assumption).
_DEFAULT_ORIGIN = {"name": "تهران", "latitude": 35.6892, "longitude": 51.3890}

# Jalali (Persian solar calendar) month -> season. Farvardin-Khordad (1-3) is
# بهار, Tir-Shahrivar (4-6) تابستان, Mehr-Azar (7-9) پاییز, Dey-Esfand (10-12)
# زمستان. This table indexes by JALALI month number, not Gregorian -- see
# current_season() below for why that distinction matters.
_MONTH_TO_SEASON = {
    1: "بهار", 2: "بهار", 3: "بهار",
    4: "تابستان", 5: "تابستان", 6: "تابستان",
    7: "پاییز", 8: "پاییز", 9: "پاییز",
    10: "زمستان", 11: "زمستان", 12: "زمستان",
}

_TEHRAN_TZ = ZoneInfo("Asia/Tehran")


async def search_destinations(
    ctx: RunContext[AgentDeps],
    categories: list[str] | None = None,
    seasons: list[str] | None = None,
    trip_types: list[str] | None = None,
    location: str | None = None,
    max_distance_km: float | None = None,
    max_travel_hours: float | None = None,
    max_physical_readiness: str | None = None,
    stay_duration: str | None = None,
    ecotourism: bool | None = None,
    exclude_liked_or_visited: bool = True,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not ctx.deps.graph:
        return [{"error": "Graph repository not available"}]

    exclude_ids = None
    if exclude_liked_or_visited:
        history = await ctx.deps.graph.get_user_history(ctx.deps.user_id)
        # We only have names from history; a production version should also
        # store ids on User relationships to exclude precisely by id.
        _ = history  # left in place intentionally; see graph.py TODO note

    # Expand informal/regional phrasing ("شمال", "جنوب", ...) into the
    # concrete city/province names the graph actually has -- see
    # src.agent.regions. A literal city/province name passes through
    # unchanged (expand_location returns [location] in that case).
    location_candidates = expand_location(location)

    return await ctx.deps.graph.search_destinations(
        categories=categories,
        seasons=seasons,
        trip_types=trip_types,
        location=location_candidates or None,
        max_distance_km=max_distance_km,
        max_travel_hours=max_travel_hours,
        max_physical_readiness=max_physical_readiness,
        stay_duration=stay_duration,
        ecotourism=ecotourism,
        exclude_destination_ids=exclude_ids,
        limit=limit,
    )


async def get_destination_details(ctx: RunContext[AgentDeps], name_or_id: str) -> dict[str, Any]:
    if not ctx.deps.graph:
        return {"error": "Graph repository not available"}
    result = await ctx.deps.graph.get_destination_details(name_or_id)
    return result or {"error": f"No destination found matching '{name_or_id}'"}


async def semantic_search_destinations(
    ctx: RunContext[AgentDeps],
    goal: str | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not ctx.deps.graph:
        return [{"error": "Graph repository not available"}]
    semantic_goal = goal
    if not semantic_goal and ctx.deps.travel_goal:
        semantic_goal = ctx.deps.travel_goal.semantic_query
    semantic_goal = semantic_goal or ctx.deps.current_query or ""
    return await ctx.deps.graph.semantic_search_destinations(semantic_goal, limit=limit)


async def find_destinations_near(
    ctx: RunContext[AgentDeps], name_or_id: str, radius_km: float = 50
) -> list[dict[str, Any]]:
    if not ctx.deps.graph:
        return [{"error": "Graph repository not available"}]
    return await ctx.deps.graph.find_destinations_near(name_or_id, radius_km)


async def list_available_filters(ctx: RunContext[AgentDeps]) -> dict[str, list[str]]:
    """Lets the agent see the actual vocabulary in the graph (categories,
    seasons, trip types) instead of guessing values that might not exist."""
    if not ctx.deps.graph:
        return {"error": "Graph repository not available"}
    return await ctx.deps.graph.list_categories_seasons()


async def save_user_preference(
    ctx: RunContext[AgentDeps],
    category: str,
    preference: str,
) -> dict[str, str]:
    """Persist a durable preference inferred from conversation, e.g.
    category="trip_companion", preference="partner (romantic trips)", or
    category="pace", preference="prefers relaxed, low-difficulty trips".
    This uses neo4j-agent-memory's long-term preference store, so it's
    available in future sessions, not just this conversation."""
    if not ctx.deps.client:
        return {"status": "skipped", "reason": "memory client not available"}
    try:
        # NOTE: the installed neo4j-agent-memory's add_preference() takes
        # `user_identifier=`, not `user_id=` -- passing the wrong kwarg name
        # raises TypeError (not AttributeError), so double-check this
        # against whatever version you have installed if it starts failing
        # again after an upgrade.
        await ctx.deps.client.long_term.add_preference(
            category=category,
            preference=preference,
            user_identifier=ctx.deps.user_id,
        )
    except AttributeError:
        return {"status": "error", "reason": "long_term.add_preference not found in installed library"}
    return {"status": "saved", "category": category, "preference": preference}


async def get_user_preferences(ctx: RunContext[AgentDeps]) -> list[dict[str, Any]]:
    if not ctx.deps.client:
        return [{"error": "memory client not available"}]
    # get_preferences_for() is a plain Cypher MATCH on
    # (:User)-[:HAS_PREFERENCE]->(:Preference), scoped to this user --
    # unlike search_preferences(), it needs no embedding at all, so it can't
    # trip over providers (like the university gateway) that reject empty
    # embedding input. It's also the *correct* query here regardless: an
    # unscoped search_preferences("", limit=20) would return every user's
    # preferences, not just this user's.
    prefs = await ctx.deps.client.long_term.get_preferences_for(ctx.deps.user_id)
    return [{"category": p.category, "preference": p.preference} for p in prefs]


async def record_trip_plan(
    ctx: RunContext[AgentDeps],
    destination_ids: list[str],
    trip_summary: str,
    trip_type: str | None = None,
    companions: str | None = None,
) -> dict[str, str]:
    """Save the itinerary the agent just proposed, linked to the user, so
    it can be referenced ("plan something like last time") and so future
    recommendations avoid repeating the same spots by default."""
    if not ctx.deps.graph:
        return {"status": "skipped", "reason": "graph repository not available"}
    trip_id = str(uuid.uuid4())
    await ctx.deps.graph.upsert_user(ctx.deps.user_id)
    await ctx.deps.graph.record_trip(
        user_id=ctx.deps.user_id,
        trip_id=trip_id,
        destination_ids=destination_ids,
        trip_context={
            "summary": trip_summary,
            "trip_type": trip_type,
            "companions": companions,
            "planned_at": datetime.utcnow().isoformat(),
        },
    )
    return {"status": "saved", "trip_id": trip_id}


async def record_destination_feedback(
    ctx: RunContext[AgentDeps],
    destination_id: str,
    sentiment: str,
    note: str | None = None,
) -> dict[str, str]:
    """Record explicit user feedback about a place (e.g. after the trip,
    or when they say "we've already been there and loved it"). sentiment
    is one of: liked, disliked, visited."""
    if not ctx.deps.graph:
        return {"status": "skipped", "reason": "graph repository not available"}
    await ctx.deps.graph.upsert_user(ctx.deps.user_id)
    await ctx.deps.graph.record_feedback(ctx.deps.user_id, destination_id, sentiment, note)
    return {"status": "saved", "destination_id": destination_id, "sentiment": sentiment}


async def get_user_trip_history(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
    if not ctx.deps.graph:
        return {"error": "Graph repository not available"}
    return await ctx.deps.graph.get_user_history(ctx.deps.user_id)


async def get_similar_past_trip_plans(
    ctx: RunContext[AgentDeps], current_request: str, limit: int = 2
) -> list[dict[str, Any]]:
    """Reasoning memory: look at how similar past requests were successfully
    handled (which tool sequence, which destinations ended up recommended)."""
    if not ctx.deps.client:
        return [{"error": "memory client not available"}]
    traces = await ctx.deps.client.reasoning.get_similar_traces(
        task=current_request, limit=limit, success_only=True
    )
    return [{"task": t.task, "outcome": t.outcome} for t in traces]


async def _resolve_coordinates(
    ctx: RunContext[AgentDeps],
    name: str,
    destination_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    province: str | None = None,
) -> dict[str, float] | None:
    """Resolve one place's coordinates, cheapest source first:
    1. Coordinates already supplied by the caller (e.g. from a prior
       search-tool result that already carries `latitude`/`longitude`).
    2. The graph, by id (exact) or by name (fuzzy, via get_destination_details).
    3. The Neshan geocoding API, only as a last resort for a place that has
       a name but no coordinates anywhere in the graph.
    """
    if latitude is not None and longitude is not None:
        return {"latitude": float(latitude), "longitude": float(longitude)}

    if ctx.deps.graph is not None:
        if destination_id:
            found = await ctx.deps.graph.get_coordinates_by_ids([destination_id])
            row = found.get(destination_id)
            if row and row.get("latitude") is not None and row.get("longitude") is not None:
                return {"latitude": row["latitude"], "longitude": row["longitude"]}
        if name:
            details = await ctx.deps.graph.get_destination_details(name)
            if details and details.get("latitude") is not None and details.get("longitude") is not None:
                return {"latitude": details["latitude"], "longitude": details["longitude"]}

    if ctx.deps.maps is not None and ctx.deps.maps.enabled and name:
        geocoded = await ctx.deps.maps.geocode(name, city=city, province=province)
        if geocoded:
            return geocoded

    return None


async def build_trip_map(
    ctx: RunContext[AgentDeps],
    stops: list[dict[str, Any]],
    origin: dict[str, Any] | None = None,
    round_trip: bool = False,
) -> dict[str, Any]:
    """Resolve coordinates for every stop (DB first, geocoding fallback),
    compute the best visiting order (TSP) and real road distance/duration
    between consecutive stops (no-traffic routing), and stash the result on
    `ctx.deps.itinerary_result` so the API layer can attach map-ready
    {name, latitude, longitude, order} data to the response. Falls back to
    straight-line distance estimates when the maps API is unavailable, so
    it always returns something usable.
    """
    origin = origin or _DEFAULT_ORIGIN
    origin_coords = await _resolve_coordinates(
        ctx, origin.get("name", "مبدا"), latitude=origin.get("latitude"), longitude=origin.get("longitude")
    ) or {"latitude": _DEFAULT_ORIGIN["latitude"], "longitude": _DEFAULT_ORIGIN["longitude"]}

    resolved: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for stop in stops:
        coords = await _resolve_coordinates(
            ctx,
            stop.get("name", ""),
            destination_id=stop.get("id"),
            latitude=stop.get("latitude"),
            longitude=stop.get("longitude"),
            city=stop.get("city"),
            province=stop.get("province"),
        )
        if coords is None:
            unresolved.append(stop.get("name", "?"))
            continue
        resolved.append({"name": stop.get("name", "?"), **coords})

    if not resolved:
        return {"error": "no stop coordinates could be resolved", "unresolved": unresolved}

    # --- Visiting order ---
    order: list[int]
    if len(resolved) == 1:
        order = [0]
    elif ctx.deps.maps is not None and ctx.deps.maps.enabled:
        waypoints = [(origin_coords["latitude"], origin_coords["longitude"])] + [
            (s["latitude"], s["longitude"]) for s in resolved
        ]
        tsp_order = await ctx.deps.maps.trip_order(
            waypoints, round_trip=round_trip, source_is_any_point=False, last_is_any_point=True
        )
        if tsp_order:
            # index 0 in `waypoints` is the origin -- drop it, shift the rest back to 0-based `resolved` indices.
            order = [i - 1 for i in tsp_order if i != 0]
        else:
            order = _nearest_neighbor_order(origin_coords, resolved)
    else:
        order = _nearest_neighbor_order(origin_coords, resolved)

    # --- Leg distances/durations, in visiting order, anchored on origin ---
    ordered_stops: list[dict[str, Any]] = []
    prev = origin_coords
    total_distance_km = 0.0
    total_duration_hours = 0.0
    for position, idx in enumerate(order, start=1):
        stop = resolved[idx]
        leg = None
        if ctx.deps.maps is not None and ctx.deps.maps.enabled:
            leg = await ctx.deps.maps.route(
                (prev["latitude"], prev["longitude"]), (stop["latitude"], stop["longitude"])
            )
        if leg is None:
            dist = haversine_km(prev["latitude"], prev["longitude"], stop["latitude"], stop["longitude"])
            leg = {"distance_km": round(dist, 1), "duration_hours": round(estimate_duration_hours(dist), 2)}
        total_distance_km += leg["distance_km"]
        total_duration_hours += leg["duration_hours"]
        ordered_stops.append(
            {
                "order": position,
                "name": stop["name"],
                "latitude": stop["latitude"],
                "longitude": stop["longitude"],
                "leg_distance_km_from_previous": leg["distance_km"],
                "leg_duration_hours_from_previous": leg["duration_hours"],
            }
        )
        prev = stop

    result = {
        "origin": origin_coords | {"name": origin.get("name", "مبدا")},
        "stops": ordered_stops,
        "total_distance_km": round(total_distance_km, 1),
        "total_duration_hours": round(total_duration_hours, 2),
        "round_trip": round_trip,
        "unresolved_stops": unresolved,
        "used_real_routing": bool(ctx.deps.maps is not None and ctx.deps.maps.enabled),
    }
    ctx.deps.itinerary_result = result
    return result


def _nearest_neighbor_order(origin: dict[str, float], stops: list[dict[str, Any]]) -> list[int]:
    """Greedy nearest-neighbor ordering by straight-line distance -- used
    only when the Neshan TSP endpoint is unavailable/unconfigured."""
    remaining = list(range(len(stops)))
    order: list[int] = []
    current = origin
    while remaining:
        best = min(
            remaining,
            key=lambda i: haversine_km(
                current["latitude"], current["longitude"], stops[i]["latitude"], stops[i]["longitude"]
            ),
        )
        order.append(best)
        remaining.remove(best)
        current = stops[best]
    return order


async def check_reachable_within_time(
    ctx: RunContext[AgentDeps],
    origin: dict[str, Any],
    minutes: float,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each candidate destination, decide whether it's realistically
    reachable within `minutes` of driving from `origin` -- use this before
    proposing destinations for a short/time-boxed trip so the plan doesn't
    combine stops that are individually fine but jointly infeasible.

    Uses the isochrone API when available (real road network); falls back
    to a straight-line-distance + average-speed estimate otherwise. Always
    returns a per-candidate verdict, never an error, so it's safe to call
    speculatively.
    """
    origin_coords = await _resolve_coordinates(
        ctx, origin.get("name", "مبدا"), latitude=origin.get("latitude"), longitude=origin.get("longitude")
    )
    if origin_coords is None:
        return [{"name": c.get("name", "?"), "reachable": None, "reason": "origin coordinates unknown"} for c in candidates]

    isochrone_geojson = None
    if ctx.deps.maps is not None and ctx.deps.maps.enabled:
        isochrone_geojson = await ctx.deps.maps.isochrone(
            origin_coords["latitude"], origin_coords["longitude"], minutes=minutes
        )

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        coords = await _resolve_coordinates(
            ctx,
            candidate.get("name", ""),
            destination_id=candidate.get("id"),
            latitude=candidate.get("latitude"),
            longitude=candidate.get("longitude"),
        )
        if coords is None:
            results.append({"name": candidate.get("name", "?"), "reachable": None, "reason": "coordinates unknown"})
            continue
        distance_km = round(
            haversine_km(origin_coords["latitude"], origin_coords["longitude"], coords["latitude"], coords["longitude"]),
            1,
        )
        if isochrone_geojson is not None:
            reachable = point_in_any_polygon(coords["latitude"], coords["longitude"], isochrone_geojson)
        else:
            reachable = estimate_duration_hours(distance_km) * 60 <= minutes
        results.append(
            {
                "name": candidate.get("name", "?"),
                "reachable": reachable,
                "approx_distance_km": distance_km,
            }
        )
    return results


async def find_nearby_amenities(
    ctx: RunContext[AgentDeps],
    latitude: float,
    longitude: float,
    layer: str,
    radius_m: int = 3000,
) -> list[dict[str, Any]] | dict[str, str]:
    """Look up nearby amenities (restaurant, hotel, parking, hospital, ...)
    around a point to enrich a destination's description. `layer` must be
    one of Neshan's nearby-search layer slugs (e.g. "restaurant", "hotel",
    "parking", "cafe", "hospital")."""
    if ctx.deps.maps is None or not ctx.deps.maps.enabled:
        return {"status": "skipped", "reason": "maps API not configured"}
    return await ctx.deps.maps.nearby(latitude, longitude, layer, radius_m)


def current_season(now: datetime | None = None) -> str:
    """Persian season for "now" (or a given datetime), in Tehran local time.

    Previously this indexed _MONTH_TO_SEASON with the *Gregorian* month
    number (datetime.utcnow().month), which is wrong: Gregorian
    January-March is winter/early spring, not بهار. It also used UTC instead
    of Tehran local time. Both are fixed here by converting to Tehran local
    time and then to the Jalali calendar before looking up the season.
    """
    if now is None:
        now = datetime.now(_TEHRAN_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc).astimezone(_TEHRAN_TZ)
    else:
        now = now.astimezone(_TEHRAN_TZ)
    jalali_month = jdatetime.date.fromgregorian(date=now.date()).month
    return _MONTH_TO_SEASON[jalali_month]
