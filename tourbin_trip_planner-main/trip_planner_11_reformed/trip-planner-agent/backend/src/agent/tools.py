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
from src.agent.regions import expand_location

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
