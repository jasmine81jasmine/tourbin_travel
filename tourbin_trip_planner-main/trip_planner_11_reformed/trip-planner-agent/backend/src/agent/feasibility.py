"""Server-side driving checks for time-boxed trip recommendations.

TSP only orders stops. Driving time comes from Neshan routing, including the
return leg; without a routing result we cannot certify a short trip.
"""

import asyncio
import json
import re
from typing import Any, Awaitable, Callable

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from src.agent.description_format import _fallback_card
from src.agent.goals import TravelGoal
from src.agent.geo import haversine_km, point_in_any_polygon
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


async def _multiday_stops(goal: TravelGoal, maps: Any, origin: tuple[float, float] | None,
                          candidates: list[dict], messages: list[Any], budget: float) -> list[dict]:
    """When the model omitted a real itinerary, group nearby graph candidates."""
    if origin is None or maps is None or not maps.enabled or _trip_days(goal) < 2:
        return []
    latest = goal.objective.rsplit("اصلاح جدید:", 1)[-1]
    if _wants_one_place(goal) or "برنامه" not in latest:
        return []
    pool = [*candidates, *_unused_search_candidates(messages, {row.get("name") for row in candidates})]
    pool = [row for row in pool if row.get("name") and row.get("latitude") is not None
            and row.get("longitude") is not None]
    if len(pool) < 2:
        return []
    first = pool[0]
    neighbors = sorted((row for row in pool[1:] if row["name"] != first["name"]),
                       key=lambda row: haversine_km(first["latitude"], first["longitude"],
                                                     row["latitude"], row["longitude"]))
    close = [row for row in neighbors if haversine_km(first["latitude"], first["longitude"],
                                                      row["latitude"], row["longitude"]) <= 80]
    for count in range(min(_trip_days(goal), len(close) + 1, 3), 1, -1):
        chosen = [first, *close[:count - 1]]
        if count > 2 and hasattr(maps, "trip_order"):
            points = [origin] + [(row["latitude"], row["longitude"]) for row in chosen]
            order = await maps.trip_order(points, round_trip=True, source_is_any_point=False)
            if order and len(order) == len(points) and order[0] == 0 and sorted(order) == list(range(len(points))):
                chosen = [chosen[index - 1] for index in order[1:]]
        previous = origin
        duration = 0.0
        for row in chosen:
            destination = (row["latitude"], row["longitude"])
            leg = await maps.route(previous, destination)
            if leg is None:
                break
            duration += leg["duration_hours"]
            previous = destination
        else:
            back = await maps.route(previous, origin)
            if back and duration + back["duration_hours"] <= budget:
                return chosen
    return []


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

    if (itinerary is None or len(itinerary.stops) < 2) and origin is not None:
        from src.api.schemas import Itinerary

        chosen = await _multiday_stops(goal, maps, origin, candidates, messages, budget)
        if chosen:
            for row in chosen:
                if row["name"] not in {item.get("name") for item in candidates}:
                    candidates.append(row)
            itinerary = Itinerary.model_validate({
                "origin": {"name": origin_name, "latitude": origin[0], "longitude": origin[1]},
                "stops": [{"order": i, "name": row["name"], "latitude": row["latitude"],
                           "longitude": row["longitude"]} for i, row in enumerate(chosen, 1)],
            })

    if itinerary is not None and len(itinerary.stops) > 1 and not _wants_one_place(goal):
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
                        rows = [next((row for row in candidates if row.get("name") == stop.name),
                                     {"name": stop.name}) for stop in itinerary.stops]
                        descriptions = await _destination_descriptions(rows, graph, description_formatter)
                        days = _trip_days(goal)
                        lines = [f"### 🌿 برنامهٔ سفر {goal.duration}",
                                 f"**ترتیب بازدید:** {route}"]
                        stops = []
                        last_day = 0
                        for index, (stop, leg) in enumerate(zip(itinerary.stops, legs)):
                            day = min(days, index * days // len(itinerary.stops) + 1)
                            if days > 1 and day != last_day:
                                lines.append(f"### 📅 روز {day} — بازدید و گردش")
                            last_day = day
                            lines.append(f"#### {stop.name}")
                            lines.append(descriptions[stop.name])
                            lines.append(f"- **مسیر از توقف قبلی:** {leg['distance_km']} کیلومتر، "
                                         f"{leg['duration_hours']} ساعت رانندگی")
                            stops.append({"order": stop.order, "name": stop.name,
                                          "latitude": stop.latitude, "longitude": stop.longitude,
                                          "leg_distance_km_from_previous": leg["distance_km"],
                                          "leg_duration_hours_from_previous": leg["duration_hours"]})
                        if days > last_day:
                            lines.append(f"### 📅 روز {days} — بازگشت و استراحت")
                        lines.append(f"### 🚗 جمع‌بندی مسیر\n- **بازگشت به {origin_name}:** "
                                     f"{back['distance_km']} کیلومتر، {back['duration_hours']} ساعت\n"
                                     f"- **مجموع رانندگی رفت‌وبرگشت:** "
                                     f"{round(sum(leg['distance_km'] for leg in legs) + back['distance_km'], 1)} "
                                     f"کیلومتر، {total:.2f} ساعت\nزمان بازدید، استراحت و ترافیک زنده جداست.")
                        verified_route = {
                            "origin": {"name": origin_name, "latitude": origin[0], "longitude": origin[1]},
                            "stops": stops,
                            "return_leg": {"leg_distance_km_from_previous": back["distance_km"],
                                           "leg_duration_hours_from_previous": back["duration_hours"]},
                            "total_distance_km": round(sum(leg["distance_km"] for leg in legs) + back["distance_km"], 1),
                            "total_duration_hours": round(total, 2), "round_trip": True,
                        }
                        return "\n\n".join(lines), verified_route
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

    # The model turns graph facts into short visit/activity/facility cards;
    # road figures and destination selection remain deterministic.
    formatted = await _destination_descriptions(
        [row for row, _, _, _ in feasible[:4]], graph, description_formatter,
    )

    # List alternatives independently: never fabricate a combined itinerary.
    lines = [f"### 🌿 پیشنهادهای سفر {goal.duration}",
             f"این‌ها **برنامه‌های جایگزین** از {origin_name} هستند؛ قرار نیست همه را در یک سفر بروید."]
    for index, (row, outbound, inbound, drive) in enumerate(feasible[:4], 1):
        lines.append(f"#### گزینهٔ {index}: {row['name']}")
        lines.append(formatted[row["name"]])
        if _trip_days(goal) == 1:
            lines.append("- **طرح یک‌روزه:** صبح حرکت، وقت‌گذاشتن برای بازدید و استراحت، و بازگشت در پایان روز.")
        lines.append(
            f"- **رفت:** {outbound['distance_km']} کیلومتر، {outbound['duration_hours']} ساعت\n"
            f"- **برگشت:** {inbound['distance_km']} کیلومتر، {inbound['duration_hours']} ساعت\n"
            f"- **مجموع رانندگی:** {drive:.2f} ساعت"
        )
    lines.append("### 🚗 نکات مسیر\nاین زمان‌ها شامل بازدید، استراحت و ترافیک زنده نیستند؛ پیش از حرکت شرایط مسیر را بررسی کنید.")
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
