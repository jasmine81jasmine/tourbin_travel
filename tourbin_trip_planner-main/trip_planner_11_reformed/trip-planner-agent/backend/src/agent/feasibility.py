"""Server-side driving checks for time-boxed trip recommendations.

TSP only orders stops. Driving time comes from Neshan routing, including the
return leg; without a routing result we cannot certify a short trip.
"""

import asyncio
import json
import re
from itertools import combinations
from typing import Any, Awaitable, Callable

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from src.agent.description_format import _fallback_card
from src.agent.goals import TravelGoal
from src.agent.geo import haversine_km, point_in_any_polygon
from src.agent.regions import expand_location
from src.agent.tools import DEFAULT_ORIGIN


_DAYS = {"یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "1": 1, "2": 2, "3": 3, "۱": 1, "۲": 2, "۳": 3}
_TOOLS = {"tool_search_destinations", "tool_semantic_search_destinations", "tool_get_destination_details"}


def driving_budget(goal: TravelGoal | None) -> float | None:
    """Reserve most of a short trip for visiting, meals and breaks."""
    if not goal or not goal.duration:
        return None
    match = re.search(r"(یک|دو|سه|چهار|پنج|[1-5۱-۵])\s*روز", goal.duration)
    if not match:
        return 12.0 if goal.duration == "آخر هفته" else None
    days = _DAYS.get(match.group(1))
    return {1: 4.0, 2: 12.0, 3: 18.0}.get(days)


def _trip_days(goal: TravelGoal) -> int:
    match = re.search(r"(یک|دو|سه|[1-3۱-۳])\s*روز", goal.duration or "")
    return _DAYS.get(match.group(1), 2) if match else 2 if goal.duration == "آخر هفته" else 1


def mentioned_destinations(messages: list[Any], reply: str) -> list[dict[str, Any]]:
    """Only screen graph-backed places actually named in this turn's answer."""
    seen: dict[str, dict[str, Any]] = {}
    normalized_reply = reply.replace("ي", "ی").replace("ك", "ک").replace("‌", " ")
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, ToolReturnPart) or part.tool_name not in _TOOLS:
                continue
            try:
                data = json.loads(part.content) if isinstance(part.content, str) else part.content
            except (ValueError, TypeError):
                continue
            for row in data if isinstance(data, list) else [data]:
                if not isinstance(row, dict):
                    continue
                name = row.get("name")
                if not isinstance(name, str) or len(name) < 3:
                    continue
                normalized_name = name.replace("ي", "ی").replace("ك", "ک").replace("‌", " ")
                if normalized_name in normalized_reply:
                    seen[name] = row
    return list(seen.values())


def _unused_search_candidates(messages: list[Any], excluded: set[str]) -> list[dict[str, Any]]:
    """Try other graph search matches when the draft's choices are too far."""
    choices = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, ToolReturnPart) or part.tool_name not in {
                "tool_search_destinations", "tool_semantic_search_destinations"
            }:
                continue
            try:
                rows = json.loads(part.content) if isinstance(part.content, str) else part.content
            except (TypeError, ValueError):
                continue
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("name") and row["name"] not in excluded:
                        choices.append(row)
                        excluded.add(row["name"])
    return choices[:4]


def ground_route_text(reply: str, itinerary: Any) -> str:
    """Replace unverified driving figures in the prose with itinerary figures."""
    if itinerary is None or not itinerary.stops:
        return reply
    numeric_route = re.compile(r"[\d۰-۹٠-٩]+(?:[.,٫][\d۰-۹٠-٩]+)?\s*(?:تا\s*[\d۰-۹٠-٩]+\s*)?(?:کیلومتر|ساعت|دقیقه)")
    # Split at sentence boundaries as well as line boundaries; removing an
    # entire paragraph would also discard useful graph-based place details.
    fragments = re.split(r"(?<=[.؟!\n])", reply)
    clean = "".join(fragment for fragment in fragments if not numeric_route.search(fragment)).strip()
    lines = ["### 🚗 مسیر و زمان رانندگی (بدون ترافیک)"]
    for stop in itinerary.stops:
        lines.append(f"- تا {stop.name}: {stop.leg_distance_km_from_previous} کیلومتر، {stop.leg_duration_hours_from_previous} ساعت رانندگی از توقف قبلی.")
    if itinerary.return_leg:
        lines.append(f"- برگشت: {itinerary.return_leg.leg_distance_km_from_previous} کیلومتر، {itinerary.return_leg.leg_duration_hours_from_previous} ساعت.")
    lines.append(f"مجموع رانندگی: {itinerary.total_distance_km} کیلومتر و {itinerary.total_duration_hours} ساعت؛ زمان بازدید و ترافیک زنده جداست.")
    return clean + "\n\n" + "\n".join(lines)


def _discovery_layers(goal: TravelGoal) -> tuple[str, ...]:
    """Search for places to visit, not arbitrary nearby services."""
    text = (goal.objective + " " + goal.semantic_query).replace("‌", " ").lower()
    if "کمپ" in text or "چادر" in text:
        return ("campground", "natural_feature", "interests")
    if any(word in text for word in ("طبیعت", "جنگل", "کوه", "دریاچه", "آبشار")):
        return ("natural_feature", "park", "garden")
    if any(word in text for word in ("زیارت", "مسجد")):
        return ("mosque", "historical", "interests")
    if any(word in text for word in ("تاریخ", "موزه", "دیدنی")):
        return ("historical", "interests", "garden")
    if "اقامت" in text or "بوم گردی" in text:
        return ("lodging_tourist", "interests", "garden")
    return ("interests", "park", "natural_feature")


def _search_centers(origin: tuple[float, float], area: dict[str, Any] | None) -> list[tuple[float, float]]:
    """Sample a couple of reachable directions; nearby() only searches a circle."""
    centers = [origin]
    rings = []
    for feature in (area or {}).get("features", []):
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if geometry.get("type") == "Polygon" and coords:
            rings.append(coords[0])
        elif geometry.get("type") == "MultiPolygon":
            rings.extend(polygon[0] for polygon in coords if polygon)
    points = [p for ring in rings for p in ring if isinstance(p, list) and len(p) >= 2]
    if not points:
        return centers
    for edge in (max(points, key=lambda p: p[1]), max(points, key=lambda p: p[0]),
                 min(points, key=lambda p: p[0]), min(points, key=lambda p: p[1])):
        # Search within the isochrone, not exactly on its boundary.
        lat = origin[0] + 0.45 * (edge[1] - origin[0])
        lon = origin[1] + 0.45 * (edge[0] - origin[1])
        if point_in_any_polygon(lat, lon, area) and all(abs(lat - a) + abs(lon - b) > 0.02 for a, b in centers):
            centers.append((lat, lon))
    return centers[:4]


async def _discover_nearby(goal: TravelGoal, maps: Any, origin: tuple[float, float],
                           budget: float, excluded: set[str]) -> list[tuple[dict, dict, dict, float]]:
    """Explore Neshan POIs only after graph candidates fail, then verify road time."""
    try:
        # Half the driving budget is the maximum outbound time; leave a
        # margin for an asymmetric return and for time at the destination.
        area = await maps.isochrone(origin[0], origin[1], minutes=min(300, int(budget * 25)))
        if area and not (area.get("features") or []):
            area = None
        centers = _search_centers(origin, area)
        requests = [maps.nearby(lat, lon, layer, 7000) for lat, lon in centers
                    for layer in _discovery_layers(goal)]
        batches = await asyncio.gather(*requests, return_exceptions=True)
        pois: list[dict] = []
        # Take one result per geographic center/layer before second choices;
        # otherwise one crowded origin layer can monopolize the route budget.
        for rank in range(2):
            for batch in batches:
                if isinstance(batch, Exception) or not isinstance(batch, list) or len(batch) <= rank:
                    continue
                poi = batch[rank]
                if not isinstance(poi, dict):
                    continue
                name, lat, lon = poi.get("name"), poi.get("latitude"), poi.get("longitude")
                if not name or lat is None or lon is None or name in excluded:
                    continue
                if area and not point_in_any_polygon(float(lat), float(lon), area):
                    continue
                excluded.add(name)
                pois.append({"name": name, "latitude": float(lat), "longitude": float(lon)})
                if len(pois) >= 8:
                    break
            if len(pois) >= 8:
                break

        feasible = []
        for poi in pois:
            point = (poi["latitude"], poi["longitude"])
            outbound, inbound = await asyncio.gather(maps.route(origin, point), maps.route(point, origin))
            if outbound and inbound:
                duration = outbound["duration_hours"] + inbound["duration_hours"]
                if duration <= budget:
                    feasible.append((poi, outbound, inbound, duration))
                    if len(feasible) == 3:
                        break
        return feasible
    except Exception:
        # Unsupported isochrone/nearby subscription or malformed geometry
        # must not turn an ordinary chat response into a server error.
        return []


def _wants_one_place(goal: TravelGoal) -> bool:
    """Distinguish independent options from a real multi-stop trip request."""
    # goal.objective retains earlier turns; the latest explicit correction wins.
    text = goal.objective.rsplit("اصلاح جدید:", 1)[-1].replace("‌", " ")
    if re.search(r"چند\s*(?:توقف|مقصد)|(?:دو|سه|[۲۳23])\s*(?:توقف|مقصد)", text):
        return False
    return bool(re.search(r"(?:یه|یک)\s+(?:جا(?:یی|ی)?|مقصد)|(?:فقط|تنها)\s+(?:یه|یک)"
                          r"|چند\s*(?:گزینه|پیشنهاد)|جای جدید|گزینه\s*ها", text))


async def _destination_descriptions(
    rows: list[dict], graph: Any,
    formatter: Callable[[dict[str, dict]], Awaitable[dict[str, str]]] | None,
) -> dict[str, str]:
    """Pass graph facts, including facilities, into short trip-plan cards."""
    destinations = {}
    for row in rows:
        name = row["name"]
        data = dict(row)
        if graph is not None:
            try:
                details = await graph.get_destination_details(row.get("id") or name)
                if details and details.get("name") == name:
                    data = {**data, **{key: value for key, value in details.items()
                                       if key not in {"name", "latitude", "longitude"} and value is not None}}
            except Exception:
                pass  # Missing optional details must not discard a verified route.
        destinations[name] = data
    fallback = {name: _fallback_card(row) for name, row in destinations.items()}
    if formatter is not None:
        try:
            return {**fallback, **(await formatter(destinations))}
        except Exception:
            pass
    return fallback


def _place(row: dict) -> tuple[float, float]:
    return float(row["latitude"]), float(row["longitude"])


async def _route_plan(maps: Any, origin: tuple[float, float], origin_name: str,
                      rows: list[dict], budget: float, optimize: bool = True) -> dict | None:
    """TSP determines visit order; no-traffic routing verifies every leg."""
    ordered = list(rows)
    if optimize and len(ordered) > 1 and hasattr(maps, "trip_order"):
        points = [origin] + [_place(row) for row in ordered]
        order = await maps.trip_order(points, round_trip=True, source_is_any_point=False)
        if order and len(order) == len(points) and order[0] == 0 and sorted(order) == list(range(len(points))):
            ordered = [ordered[index - 1] for index in order[1:]]
    points = [origin] + [_place(row) for row in ordered] + [origin]
    legs = await asyncio.gather(*(maps.route(a, b) for a, b in zip(points, points[1:])),
                                return_exceptions=True)
    if any(not isinstance(leg, dict) or leg["duration_hours"] > 6 for leg in legs):
        return None
    stops = []
    total_km = 0.0
    total_hours = 0.0
    for position, (row, leg) in enumerate(zip(ordered, legs), 1):
        total_km += leg["distance_km"]
        total_hours += leg["duration_hours"]
        stops.append({"order": position, "name": row["name"], "latitude": row["latitude"],
                      "longitude": row["longitude"], "leg_distance_km_from_previous": leg["distance_km"],
                      "leg_duration_hours_from_previous": leg["duration_hours"]})
    back = legs[-1]
    if total_hours + back["duration_hours"] > budget:
        return None
    days = 1 if budget <= 4 else 2 if budget <= 12 else 3
    driving_per_day = [0.0] * days
    for index, stop in enumerate(stops):
        day = min(days - 1, index * days // len(stops))
        driving_per_day[day] += stop["leg_duration_hours_from_previous"]
    driving_per_day[-1] += back["duration_hours"]
    if days > 1 and any(hours > 6 for hours in driving_per_day):
        return None
    total_km += back["distance_km"]
    total_hours += back["duration_hours"]
    return {"origin": {"name": origin_name, "latitude": origin[0], "longitude": origin[1]},
            "stops": stops, "return_leg": {"leg_distance_km_from_previous": back["distance_km"],
                                            "leg_duration_hours_from_previous": back["duration_hours"]},
            "total_distance_km": round(total_km, 1), "total_duration_hours": round(total_hours, 2),
            "round_trip": True}


async def _multiday_candidates(goal: TravelGoal, graph: Any, messages: list[Any],
                               mentioned: list[dict]) -> list[dict]:
    """Search the graph even when the agent returned no place names this turn."""
    location = expand_location(goal.region) or None
    rows = []
    if graph is not None:
        try:
            if "کمپ" in goal.objective or "چادر" in goal.objective:
                rows = await graph.search_destinations(trip_types=["کمپ"], location=location, limit=25)
            if len(rows) < 5:
                rows += await graph.search_destinations(location=location, limit=25)
        except Exception:
            pass
    # For an explicitly requested region, trust region-filtered graph matches
    # over the model's potentially out-of-region suggestions.
    if not rows and not goal.region:
        rows = [*mentioned, *_unused_search_candidates(messages, {r.get("name") for r in mentioned})]
    seen = set()
    result = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("name") or row.get("latitude") is None or row.get("longitude") is None:
            continue
        if row["name"] in seen:
            continue
        seen.add(row["name"])
        result.append(row)
    return result[:25]


async def _discover_multiday_plans(goal: TravelGoal, maps: Any, graph: Any, messages: list[Any],
                                   mentioned: list[dict], origin: tuple[float, float],
                                   budget: float) -> list[dict]:
    pool = await _multiday_candidates(goal, graph, messages, mentioned)
    if len(pool) < 2:
        return []
    # A few likely anchors, not every graph node: route requests have quotas.
    camping = "کمپ" in goal.objective or "چادر" in goal.objective
    anchors = sorted(pool, key=lambda row: (
        0 if camping and "کمپ" in (row.get("trip_types") or []) else 1 if camping else 0,
        haversine_km(*origin, *_place(row)),
    ))[:8]
    estimates = await asyncio.gather(*(maps.route(origin, _place(row)) for row in anchors), return_exceptions=True)
    reachable = [(row, leg) for row, leg in zip(anchors, estimates)
                 if isinstance(leg, dict) and leg["duration_hours"] <= 6]
    reachable.sort(key=lambda item: (
        0 if camping and "کمپ" in (item[0].get("trip_types") or []) else 1 if camping else 0,
        item[1]["duration_hours"],
    ))
    plans = []
    seen = set()
    for anchor, _ in reachable[:5]:
        nearby = sorted((row for row in pool if row["name"] != anchor["name"]
                         and haversine_km(*_place(anchor), *_place(row)) <= 80),
                        key=lambda row: haversine_km(*_place(anchor), *_place(row)))
        if not nearby and graph is not None:
            try:
                nearby = [row for row in await graph.find_destinations_near(anchor.get("id") or anchor["name"], 80)
                          if row.get("latitude") is not None and row.get("longitude") is not None]
            except Exception:
                pass
        for count in range(min(_trip_days(goal), len(nearby) + 1, 3), 1, -1):
            chosen = [anchor, *nearby[:count - 1]]
            key = frozenset(row["name"] for row in chosen)
            if len(key) != len(chosen) or key in seen:
                continue
            seen.add(key)
            plan = await _route_plan(maps, origin, goal.origin, chosen, budget)
            if plan:
                plans.append(plan)
                if len(plans) == 2:
                    return plans
    return plans


async def _render_multiday_plans(
    goal: TravelGoal, plans: list[dict], rows: list[dict], graph: Any,
    formatter: Callable[[dict[str, dict]], Awaitable[dict[str, str]]] | None,
) -> tuple[str, dict | None]:
    names = {stop["name"] for plan in plans for stop in plan["stops"]}
    data = [next((row for row in rows if row.get("name") == name), {"name": name}) for name in names]
    cards = await _destination_descriptions(data, graph, formatter)
    lines = []
    for number, plan in enumerate(plans, 1):
        lines.append(f"### 🌿 برنامهٔ {number} — سفر {goal.duration}" if len(plans) > 1
                     else f"### 🌿 برنامهٔ سفر {goal.duration}")
        lines.append("**ترتیب بازدید:** " + " ← ".join(
            [plan["origin"]["name"]] + [stop["name"] for stop in plan["stops"]]
            + [plan["origin"]["name"]]
        ))
        last_day = 0
        days = _trip_days(goal)
        for index, stop in enumerate(plan["stops"]):
            day = min(days, index * days // len(plan["stops"]) + 1)
            if day != last_day:
                lines.append(f"**📅 روز {day} — بازدید و گردش**")
            last_day = day
            lines.append(f"#### {stop['name']}")
            lines.append(cards[stop["name"]])
            lines.append(f"- **رانندگی از توقف قبلی:** {stop['leg_distance_km_from_previous']} کیلومتر، "
                         f"{stop['leg_duration_hours_from_previous']} ساعت")
        if days > last_day:
            lines.append(f"**📅 روز {days} — بازگشت**")
        if "کمپ" in goal.objective or "چادر" in goal.objective:
            lines.append("**شب‌مانی:** پیش از حرکت، مجاز بودن کمپ و امکانات محل شب‌مانی را بررسی کنید.")
        lines.append(f"- **بازگشت به {plan['origin']['name']}:** "
                     f"{plan['return_leg']['leg_distance_km_from_previous']} کیلومتر، "
                     f"{plan['return_leg']['leg_duration_hours_from_previous']} ساعت\n"
                     f"- **مجموع رانندگی:** {plan['total_distance_km']} کیلومتر، "
                     f"{plan['total_duration_hours']} ساعت (بدون ترافیک)")
    lines.append("زمان بازدید و استراحت به زمان رانندگی اضافه می‌شود؛ وضعیت روز سفر را پیش از حرکت بررسی کنید.")
    # A single itinerary field cannot encode several independent alternatives.
    return "\n\n".join(lines), plans[0] if len(plans) == 1 else None


async def screen_short_trip(goal: TravelGoal | None, maps: Any, messages: list[Any], reply: str,
                            itinerary: Any = None,
                            description_formatter: Callable[[dict[str, dict]], Awaitable[dict[str, str]]] | None = None,
                            graph: Any = None,
                            ) -> tuple[str, dict | None] | None:
    """Return a grounded replacement when the model proposes a short trip.

    No inferred coordinates or straight-line estimates are used as road ETA.
    A day trip always includes the drive back, even if the model omitted it.
    """
    budget = driving_budget(goal)
    if budget is None:
        return None
    candidates = mentioned_destinations(messages, reply)
    if itinerary is not None:
        for stop in itinerary.stops:
            if stop.name not in {row.get("name") for row in candidates}:
                candidates.append({"name": stop.name, "latitude": stop.latitude, "longitude": stop.longitude})
    origin_name = goal.origin if goal else DEFAULT_ORIGIN["name"]
    if origin_name == DEFAULT_ORIGIN["name"]:
        origin = (DEFAULT_ORIGIN["latitude"], DEFAULT_ORIGIN["longitude"])
    elif itinerary and itinerary.origin and itinerary.origin.get("name") == origin_name:
        origin = (itinerary.origin["latitude"], itinerary.origin["longitude"])
    elif maps is not None and maps.enabled:
        location = await maps.geocode(origin_name)
        origin = (location["latitude"], location["longitude"]) if location else None
    else:
        origin = None

    if _trip_days(goal) >= 2 and not _wants_one_place(goal) and origin is not None and maps and maps.enabled:
        plans = []
        if itinerary is not None and len(itinerary.stops) > 1:
            rows = [{"name": stop.name, "latitude": stop.latitude, "longitude": stop.longitude}
                    for stop in itinerary.stops]
            verified = await _route_plan(maps, origin, origin_name, rows, budget)
            if verified:
                plans.append(verified)
        if len(plans) < 2:
            discovered_plans = await _discover_multiday_plans(goal, maps, graph, messages, candidates, origin, budget)
            for plan in discovered_plans:
                places = frozenset(stop["name"] for stop in plan["stops"])
                if places not in [frozenset(stop["name"] for stop in current["stops"]) for current in plans]:
                    plans.append(plan)
                if len(plans) == 2:
                    break
        if not plans:
            nearby = await _discover_nearby(goal, maps, origin, budget, set())
            nearby_rows = [row for row, _, _, _ in nearby]
            if goal.region == "شمال":
                nearby_rows = [row for row in nearby_rows if row["latitude"] > origin[0] + 0.3]
            if len(nearby_rows) >= 2:
                plan = await _route_plan(maps, origin, origin_name, nearby_rows[:min(_trip_days(goal), 3)], budget)
                if plan:
                    plans.append(plan)
        if plans:
            return await _render_multiday_plans(goal, plans, candidates, graph, description_formatter)

    if itinerary is not None and len(itinerary.stops) > 1 and not _wants_one_place(goal) and _trip_days(goal) == 1:
        if origin is not None and maps is not None and maps.enabled:
            rows = [{"name": stop.name, "latitude": stop.latitude, "longitude": stop.longitude}
                    for stop in itinerary.stops]
            plan = await _route_plan(maps, origin, origin_name, rows, budget, optimize=False)
            if plan:
                return await _render_multiday_plans(goal, [plan], candidates, graph, description_formatter)
        return ("مسیر ترکیبی این توقف‌ها با فرصت یک‌روزه سازگار نیست؛ گزینه‌های نزدیک‌تر را بررسی می‌کنم.", None)

    feasible = []
    verified = 0
    for row in candidates[:5]:
        lat, lon = row.get("latitude"), row.get("longitude")
        if origin is None or lat is None or lon is None or maps is None or not maps.enabled:
            continue
        destination = (float(lat), float(lon))
        outbound = await maps.route(origin, destination)
        inbound = await maps.route(destination, origin)
        if not outbound or not inbound:
            continue
        verified += 1
        drive = outbound["duration_hours"] + inbound["duration_hours"]
        if drive <= budget:
            feasible.append((row, outbound, inbound, drive))

    if not feasible and origin is not None and maps is not None and maps.enabled:
        for row in _unused_search_candidates(messages, {r.get("name") for r in candidates}):
            lat, lon = row.get("latitude"), row.get("longitude")
            if lat is None or lon is None:
                continue
            point = (float(lat), float(lon))
            outbound = await maps.route(origin, point)
            inbound = await maps.route(point, origin)
            if outbound and inbound:
                drive = outbound["duration_hours"] + inbound["duration_hours"]
                if drive <= budget:
                    feasible.append((row, outbound, inbound, drive))
                    break

    discovered = False
    if not feasible and origin is not None and maps is not None and maps.enabled:
        if not goal.region or goal.region == "شمال":
            feasible = await _discover_nearby(goal, maps, origin, budget, {r.get("name") for r in candidates})
            if goal.region == "شمال":
                feasible = [option for option in feasible if option[0]["latitude"] > origin[0] + 0.3]
        discovered = bool(feasible)

    if not feasible:
        area = f" در {goal.region}" if goal.region else ""
        if verified:
            return (f"مسیر رفت‌وبرگشت مقصدهای پیدا‌شده{area} برای سفر {goal.duration} با فرصت بازدید و استراحت سازگار نبود. "
                    "در میان مقصدهای نزدیک‌تر و مسیرهای قابل تأیید هم گزینهٔ مناسبی پیدا نکردم؛ "
                    "اگر شهر مشخصی مدنظرتان است بگویید تا همان محدوده را بررسی کنم.", None)
        return (f"فعلاً نتوانستم برای سفر {goal.duration}{area} مسیر رفت‌وبرگشت و توقف‌های قابل تأیید پیدا کنم. "
                "اگر شهر یا محدودهٔ دقیق‌تری مدنظرتان است بگویید تا همان‌جا را بررسی کنم.", None)

    # The model turns graph facts into short visit/activity/facility cards;
    # road figures and destination selection remain deterministic.
    formatted = await _destination_descriptions(
        [row for row, _, _, _ in feasible[:4]], graph, description_formatter,
    )

    # List alternatives independently: never fabricate a combined itinerary.
    lines = [f"### 🌿 برنامه‌های پیشنهادی سفر {goal.duration}"]
    for index, (row, outbound, inbound, drive) in enumerate(feasible[:4], 1):
        lines.append(f"#### گزینهٔ {index}: {row['name']}")
        lines.append(formatted[row["name"]])
        lines.append(
            f"- **رانندگی از {origin_name}:** رفت {outbound['distance_km']} کیلومتر، "
            f"{outbound['duration_hours']} ساعت؛ برگشت {inbound['distance_km']} کیلومتر، "
            f"{inbound['duration_hours']} ساعت؛ مجموع {drive:.2f} ساعت."
        )
    if _trip_days(goal) == 1:
        lines.append("**📅 الگوی یک‌روزه:** حرکت صبح، بازدید و استراحت در مقصد انتخابی، بازگشت تا پایان روز.")
    lines.append("**🚗 نکتهٔ مسیر:** زمان‌ها بدون ترافیک و جدا از بازدید و استراحت‌اند؛ شرایط روز سفر را بررسی کنید.")
    if discovered:
        lines.append("دربارهٔ امکانات و آسان‌بودن مسیر پیاده‌روی این مکان‌ها اطلاعات تأییدشده ندارم؛ اگر همراه کودک یا سالمند هستید، پیش از انتخاب بررسی کنید.")
    # A single verified option is a usable map itinerary. Several independent
    # alternatives must never be presented as a combined route.
    route = None
    if len(feasible) == 1:
        row, outbound, inbound, duration = feasible[0]
        route = {
            "origin": {"name": origin_name, "latitude": origin[0], "longitude": origin[1]},
            "stops": [{"order": 1, "name": row["name"], "latitude": row["latitude"], "longitude": row["longitude"],
                       "leg_distance_km_from_previous": outbound["distance_km"],
                       "leg_duration_hours_from_previous": outbound["duration_hours"]}],
            "return_leg": {"leg_distance_km_from_previous": inbound["distance_km"],
                           "leg_duration_hours_from_previous": inbound["duration_hours"]},
            "total_distance_km": round(outbound["distance_km"] + inbound["distance_km"], 1),
            "total_duration_hours": round(duration, 2), "round_trip": True,
        }
    return "\n\n".join(lines), route
