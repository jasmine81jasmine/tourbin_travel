"""Server-side driving checks for time-boxed trip recommendations.

TSP only orders stops. Driving time comes from Neshan routing, including the
return leg; without a routing result we cannot certify a short trip.
"""

import json
import re
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart

from src.agent.goals import TravelGoal
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


async def screen_short_trip(goal: TravelGoal | None, maps: Any, messages: list[Any], reply: str,
                            itinerary: Any = None) -> tuple[str, None] | None:
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
            lines.append(f"  {description.strip()[:240]}")
    lines.append("این زمان‌ها شامل بازدید، استراحت و ترافیک زنده نیستند؛ پیش از حرکت شرایط مسیر را بررسی کنید.")
    return "\n".join(lines), None
