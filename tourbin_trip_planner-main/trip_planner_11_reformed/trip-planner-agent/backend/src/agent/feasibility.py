"""Server-side driving checks for time-boxed trip recommendations.

TSP only orders stops. Driving time comes from Neshan routing, including the
return leg; without a routing result we cannot certify a short trip.
"""

import asyncio
import json
import re
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from src.agent.goals import TravelGoal
from src.agent.geo import point_in_any_polygon
from src.agent.tools import DEFAULT_ORIGIN


_DAYS = {"یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "1": 1, "2": 2, "3": 3, "۱": 1, "۲": 2, "۳": 3}
_TOOLS = {"tool_search_destinations", "tool_semantic_search_destinations", "tool_get_destination_details"}


def driving_budget(goal: TravelGoal | None) -> float | None:
    """Reserve most of a short trip for visiting, meals and breaks."""
    if not goal or not goal.duration:
        return None
    match = re.search(r"(یک|دو|سه|چهار|پنج|[1-5۱-۵])\s*روز", goal.duration)
    if not match:
        return 7.0 if goal.duration == "آخر هفته" else None
    days = _DAYS.get(match.group(1))
    return {1: 4.0, 2: 7.0, 3: 10.0}.get(days)


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
    lines = ["### مسیر و زمان رانندگی (بدون ترافیک)"]
    for stop in itinerary.stops:
        lines.append(f"- تا {stop.name}: {stop.leg_distance_km_from_previous} کیلومتر، {stop.leg_duration_hours_from_previous} ساعت رانندگی از توقف قبلی.")
    if itinerary.return_leg:
        lines.append(f"- برگشت: {itinerary.return_leg.leg_distance_km_from_previous} کیلومتر، {itinerary.return_leg.leg_duration_hours_from_previous} ساعت.")
    lines.append(f"مجموع رانندگی: {itinerary.total_distance_km} کیلومتر و {itinerary.total_duration_hours} ساعت؛ زمان بازدید و ترافیک زنده جداست.")
    return clean + "\n\n" + "\n".join(lines)


def _discovery_layers(goal: TravelGoal) -> tuple[str, ...]:
    """Search for places to visit, not arbitrary nearby services."""
    text = (goal.objective + " " + goal.semantic_query).replace("‌", " ").lower()
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
        area = await maps.isochrone(origin[0], origin[1], minutes=int(budget * 25))
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


async def screen_short_trip(goal: TravelGoal | None, maps: Any, messages: list[Any], reply: str,
                            itinerary: Any = None) -> tuple[str, dict | None] | None:
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

    if itinerary is not None and len(itinerary.stops) > 1:
        legs = []
        previous = origin
        if maps is not None and maps.enabled and previous is not None:
            for stop in itinerary.stops:
                point = (stop.latitude, stop.longitude)
                leg = await maps.route(previous, point)
                if leg is None:
                    break
                legs.append(leg)
                previous = point
            if len(legs) == len(itinerary.stops):
                back = await maps.route(previous, origin)
                if back is not None:
                    total = sum(leg["duration_hours"] for leg in legs) + back["duration_hours"]
                    if total <= budget:
                        route = " ← ".join([origin_name] + [stop.name for stop in itinerary.stops] + [origin_name])
                        return (f"مسیر پیشنهادی: {route}. زمان رانندگی رفت‌وبرگشت بدون ترافیک {total:.2f} ساعت است؛ زمان بازدید و استراحت جداست.", None)
        return ("مسیر ترکیبی این مقصدها با زمان سفر شما تأیید نشد. بهتر است تعداد توقف‌ها را کمتر کنیم یا مقصدهای نزدیک‌تری انتخاب کنیم.", None)

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
        feasible = await _discover_nearby(goal, maps, origin, budget, {r.get("name") for r in candidates})
        discovered = bool(feasible)

    if not feasible:
        if verified:
            return (f"با درنظرگرفتن مسیر رفت‌وبرگشت از {origin_name}، مقصدهای بررسی‌شده برای سفر {goal.duration} زمان زیادی در خودرو می‌گیرند. برای اینکه فرصت کافی برای گردش و استراحت بماند، بهتر است مقصد نزدیک‌تری انتخاب کنیم. دوست دارید چند گزینه نزدیک‌تر پیشنهاد بدهم؟", None)
        if not candidates:
            return ("برای پیشنهاد سفر کوتاه باید اول مسیر رفت‌وبرگشت یک مقصد مشخص را بررسی کنم. مقصد یا محدودهٔ دلخواهتان را بگویید تا گزینه‌ای متناسب با زمانتان پیدا کنیم.", None)
        return ("برای این سفر کوتاه فعلاً زمان مسیر رفت‌وبرگشت را نمی‌توانم با دادهٔ مسیریابی تأیید کنم؛ نمی‌خواهم مسیر دور را یک‌روزه پیشنهاد کنم. مبدأ و مقصد دقیق را بگویید یا کمی بعد دوباره امتحان کنیم.", None)

    # List alternatives independently: never fabricate a combined itinerary.
    lines = [f"برای سفر {goal.duration} از {origin_name}، این گزینه‌ها با زمان رانندگی رفت‌وبرگشت بررسی شدند (بدون ترافیک):"]
    for row, outbound, inbound, drive in feasible[:4]:
        lines.append(
            f"- {row['name']}: رفت {outbound['distance_km']} کیلومتر و {outbound['duration_hours']} ساعت؛ "
            f"برگشت {inbound['distance_km']} کیلومتر و {inbound['duration_hours']} ساعت؛ "
            f"مجموع رانندگی {drive:.2f} ساعت."
        )
        # The graph remains the source for destination characteristics; map
        # search is only used for driving feasibility and nearby POIs.
        description = row.get("description")
        if isinstance(description, str) and description.strip():
            lines.append(f"  {description.strip()}")
    lines.append("این زمان‌ها شامل بازدید، استراحت و ترافیک زنده نیستند؛ پیش از حرکت شرایط مسیر را بررسی کنید.")
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
    return "\n".join(lines), route
