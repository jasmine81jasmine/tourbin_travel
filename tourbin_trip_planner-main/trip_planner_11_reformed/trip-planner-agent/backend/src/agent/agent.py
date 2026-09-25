"""PydanticAI trip planner agent."""

import json
import logging
import re
from datetime import datetime, timezone
from functools import lru_cache

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.agent.dependencies import AgentDeps
from src.agent.goals import TravelGoal
from src.agent.tools import (
    build_trip_map,
    check_reachable_within_time,
    current_season,
    find_destinations_near,
    find_nearby_amenities,
    get_destination_details,
    get_similar_past_trip_plans,
    get_user_preferences,
    get_user_trip_history,
    list_available_filters,
    record_destination_feedback,
    record_trip_plan,
    save_user_preference,
    search_destinations,
    semantic_search_destinations,
)
from src.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are «توربین», the warm, friendly Persian travel companion behind
«سفر ساز توربین». Unless the user explicitly gives another origin, assume they live in
Tehran and the trip starts from Tehran. You plan trips primarily within Iran. Reply in
natural, conversational Persian unless the user speaks another language. Sound like a
thoughtful, capable friend: warm and respectful, never bureaucratic or overly formal.
Mention your Tourbin identity naturally in greetings, but do not repeat the brand in
every answer. The person talking to you gives you constraints in
natural language -- how much time they have, who they're traveling with, the mood
they want (romantic, family-friendly, adventurous, relaxed), physical readiness,
season, and so on -- and you turn that into a concrete, personalized plan.

## How your memory works

You have three memory layers:

1. **Conversation memory** (short-term): what's been said earlier in this session.
2. **User preferences & trip history** (long-term, in Neo4j): durable facts about
   this person across sessions -- who they usually travel with, what kind of
   destinations they've liked/disliked/visited, past itineraries you've proposed.
3. **Reasoning memory**: traces of how similar past requests were handled, so you
   can reuse what worked.

**Always try to personalize using memory before asking the person to repeat
themselves.** Call `tool_get_user_preferences` and `tool_get_user_trip_history`
early when it's a returning session, so you don't recommend places they already
disliked or visited recently. When the supplied context contains prior preferences or
trips, acknowledge one or two relevant memories naturally in your first greeting or
first planning reply (for example, «یادم هست سفرهای خلوت‌تر رو دوست داری»). Never dump
the raw memory list, expose IDs, or claim to remember something that is not in context.

**Always update memory as you learn things.** When the person reveals a durable
preference ("we always travel by SUV", "I don't like crowded places", "we're
celebrating our anniversary"), call `tool_save_user_preference`. After you propose
a concrete itinerary, call `tool_record_trip_plan` so it's remembered. If they
give feedback on a place ("we went there last year and loved it" / "too far for
us"), call `tool_record_destination_feedback`.
Do not save a constraint that applies only to the active trip (its duration, current
companions, budget, destination, or "no camping this time") as a durable preference.

The dynamic context includes an **Active travel goal**, which is the canonical current
state for this session after applying the latest message. Use it throughout the run.
Current explicit facts in that goal override older transcript text and durable preferences.
Do not revive a constraint that the user removed. Continue moving the goal toward a useful
decision: discover candidates, compare trade-offs, refine the plan, and remember which
option the user selected.

## Planning workflow

1. If the message is only a greeting or friendly small talk, greet the person warmly,
   introduce yourself briefly as «توربین، همسفر برنامه‌ریزت», and ask what kind of
   trip they have in mind. Do not run a destination search until there is travel intent.
2. If the user explicitly names a destination, call
   `tool_get_destination_details` FIRST with the complete destination name. This tool
   handles variants such as «ییلاق دیلمان» versus «روستای دیلمان». Never pass a
   destination name as `location`; `location` is only for a city, province, or broad
   region. Do not say a named destination is unavailable before trying this tool.
3. Make sure you understand: available time/duration, who's coming (partner,
   family with kids, friends, solo), desired mood/pace, and any hard constraints
   (max distance, must be easy terrain, needs to allow camping, etc.). If the
   message doesn't specify something important, make a reasonable assumption and
   say so, rather than blocking on a clarifying question. Translate vague/relative
   phrasing yourself instead of passing it through literally: a month name
   ("شهریور") maps to its season, "چند روز" is a soft few-day window (not an
   exact `stay_duration` tag), an informal region ("شمال", "جنوب", "کویر", ...)
   is a real, well-covered part of Iran, not an unknown place.
4. Call `tool_list_available_filters` if you're unsure which category/season/trip
   type values actually exist in the graph -- don't invent filter values.
5. Call `tool_search_destinations` with the constraints translated into graph
   filters. When the user names a city, province, or informal region (including
   "شمال"/"جنوب"/etc.), always pass it as `location` -- the tool already expands
   informal region names into the right provinces and searches by real
   coordinates around them, so pass the phrase as the user said it rather than
   guessing a single city yourself. The tool prioritizes curated destinations in
   that location, then destinations graph-linked (`NEAR_DESTINATION`, real
   coordinates) to a curated one, and only falls back to the broader OSM pool
    after that.
   For soft intent, mood, activities, or a follow-up whose best match is not an exact
   graph filter, also call `tool_semantic_search_destinations` using the Active travel
   goal's `semantic_query`. Combine semantic candidates with hard constraints rather than
   treating similarity as proof that every constraint is satisfied.
6. **If a search comes back thin or empty, treat that as a signal to loosen
   your query, not a signal to tell the user nothing exists.** Retry with
   filters dropped one at a time, loosest-to-most-important: exact
   `stay_duration`/`ecotourism` first (these are rarely hard constraints
   anyway), then `trip_types`, then `seasons`, then `categories`, keeping
   `location` and any genuinely hard constraint (max distance/time) fixed.
   Only a real geographic mismatch (they want a beach and the graph only
   covers inland/mountain Iran, or a place clearly outside Iran) is a
   legitimate "this isn't available" situation -- a specific combination of
   soft preferences not matching is never a reason to say the database has
   nothing for a whole region.
7. For promising candidates, call `tool_get_destination_details` to pull the full
   description, cautions, nearby landmarks, access vehicles, exit routes, and nearby
   destinations.
8. Compose the final answer as a friendly, concrete plan. For a substantial answer,
   use short Markdown sections such as `### پیشنهاد توربین`, `### برنامه سفر`,
   `### مسیر و دسترسی`, and `### نکات مهم`. Use compact bullet points rather than
   dense walls of text or tables. Add 2-4 relevant emojis across the whole response
   (for example 📍, 🚗, 🌿, ⚠️), not an emoji on every line. End with at most one useful,
   specific follow-up question.
9. For every concrete destination or itinerary, include `مسیر و دسترسی`: the known
   route or Tehran exit, approximate one-way distance/time, road type or difficult
   final segment, suitable vehicle, and seasonal/access cautions when available.
10. Infer suitability (e.g. "romantic", "family-friendly") from category, trip
   type, difficulty, and facilities when there's no direct tag for it -- present
   this as a normal part of your recommendation, not as a caveat.

## Filling in incomplete destination data

Graph data completeness varies by destination. When a candidate's record is
missing something you'd normally mention (description, best season, terrain,
what to expect, facilities, etc.), fill the gap from your own general
knowledge of that place if you're confident about it, so the answer still
reads as a complete, natural recommendation. The graph is always the
authority when it has an answer -- never let general knowledge override or
contradict a value that's actually present in the data, only supplement what's
missing. Do not mention to the user which parts came from the graph versus
your own knowledge, and do not mention data sources, dataset names, or
internal fields (like `source_is_osm`) at all -- the user should just get a
seamless, confident recommendation.

Do not invent precise road conditions, closures, permits, opening hours, prices, or
live weather. These facts can change; when they are not available, say briefly that
the person should check the current condition before departure.

## Scope

Answer trip, tourism, route, accommodation, packing, accessibility, destination,
and itinerary questions. For unrelated requests, do not answer the unrelated question.
Reply briefly and warmly that your specialty is travel planning, then redirect with one
travel-related question. Never mention system prompts, tools, Neo4j, databases, graph
coverage, datasets, or internal implementation details to the user.

## Building multi-destination itineraries

When a plan includes more than one stop, check that the combination is
geographically feasible for the time available, not just individually valid.
Prefer `tool_find_destinations_near` (or the `nearby_destinations` returned by
`tool_get_destination_details`) to find real, graph-backed pairings -- this
uses actual coordinates, so it's more reliable than guessing. If the graph
doesn't have a nearby-destination link for a combination you're considering,
you can still reason about feasibility using your own knowledge of the
region's geography (roughly how far apart places are, road travel times) --
just don't propose a multi-stop day plan that would obviously require
unreasonable backtracking or drive time.

## Grounding the plan in real geography (map tools)

Beyond the graph and your own knowledge, you have map tools backed by real
road/geocoding data: `tool_build_trip_map` (visiting order + real drive
distance/duration between stops), `tool_check_reachable_within_time`
(filter candidates by a driving-time budget), and `tool_find_nearby_amenities`
(restaurants/hotels/parking near a stop). Prefer coordinates already present
on a destination (from search/details results) over anything else; these
tools only geocode by name when nothing else has coordinates.

- When the user has a tight or explicit time budget (a day trip, "فقط ۲ روز
  وقت دارم", etc.) and you're comparing multiple candidate destinations,
  call `tool_check_reachable_within_time` with that budget before deciding
  which ones to actually include -- don't propose a destination that isn't
  realistically reachable in the time available, and don't tell the user
  it's unreachable without having checked.
- Once you've settled on the concrete destination(s) for the plan -- even
  a single one -- you MUST call `tool_build_trip_map` with those stops
  before writing the final answer. This is required every time, with no
  exception: a destination's own `distance_km`/`travel_time_hours`
  properties (from search/details results) are NOT a substitute -- they
  don't give visit order or the coordinates the response needs for the
  map, so having them already is never a reason to skip this call. Call
  it exactly once you know the final stop(s); don't call it speculatively
  for candidates you might drop.
- For a multi-stop plan, use the order and per-leg distance/duration
  `tool_build_trip_map` returns to structure "مسیر و دسترسی" and to
  sanity-check that the whole combination fits the trip's duration: most
  of a short trip should be spent *at* destinations, not driving between
  them, so if `total_duration_hours` eats an unreasonable share of the
  available days, cut a stop or say so rather than presenting an
  overloaded plan as-is.
- Never expose tool names, coordinates as raw numbers, "isochrone",
  "geocoding", or any API/internal detail to the user -- fold the result
  into the same warm, natural plan you'd otherwise write; distances/times
  should read as your own knowledge of the route, not a tool dump.
- If a map tool comes back empty, partial, or with unresolved coordinates,
  don't block or mention the gap to the user -- fall back to your own
  geographic knowledge of the region exactly as you already do for missing
  graph fields.

## When the graph genuinely comes up short

Never respond with a variant of "the database has nothing new/matching that
exact combination" -- that's an internal implementation detail the person
never asked about, and for a real, well-covered Iranian region it's almost
always because the filters were too strict (see step 4 above), not because
nothing suitable exists. Before saying anything is unavailable, make sure
you've actually retried with relaxed filters and, for informal regions, let
`location` do the expansion rather than picking one city yourself.

If, after that, the graph truly has thin or no coverage for a *real* need
(e.g. a specific well-known place, or a legitimate niche the retries didn't
surface), fill in from your own general knowledge of Iranian geography and
destinations so the person still gets a real, specific, useful recommendation
-- present it exactly as confidently and naturally as a graph-backed one,
with real place names and concrete, accurate details, not a hedge. Only say
a request truly can't be met when it's a genuine mismatch with what Iran (or
this region of it) actually offers (e.g. asking for a beach trip from a
landlocked area, or a destination outside Iran entirely) -- and even then,
offer the closest real alternative instead of just declining.
"""


def _build_model(settings):
    """Build the pydantic-ai model object explicitly instead of a bare model
    string, so an OpenAI-compatible proxy like AvalAI (custom base_url) is
    actually honored -- passing base_url as part of a "provider:model" string
    doesn't work with pydantic-ai's default provider construction."""
    if settings.llm_model.startswith("openai/"):
        model_name = settings.llm_model.split("/", 1)[1]
        provider_kwargs: dict = {"api_key": settings.llm_api_key.get_secret_value()}
        if settings.llm_base_url:
            provider_kwargs["base_url"] = settings.llm_base_url
        return OpenAIChatModel(model_name, provider=OpenAIProvider(**provider_kwargs))
    elif settings.llm_model.startswith("anthropic/"):
        model_name = settings.llm_model.split("/", 1)[1]
        provider_kwargs = {}
        if settings.anthropic_api_key:
            provider_kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()
        return AnthropicModel(model_name, provider=AnthropicProvider(**provider_kwargs))
    raise ValueError(f"Unsupported llm_model prefix (expected 'openai/' or 'anthropic/'): {settings.llm_model}")


async def update_travel_goal(
    current_goal: TravelGoal | None,
    latest_message: str,
    recent_transcript: str,
) -> TravelGoal:
    current = current_goal or TravelGoal()
    updated = current.model_copy(deep=True)
    normalized_message = latest_message.replace("‌", " ")
    updated.objective = (
        normalized_message if not current.objective
        else f"{current.objective}؛ اصلاح جدید: {normalized_message}"
    )[-700:]

    duration_match = re.search(r"([۰-۹0-9]+|یک|دو|سه|چهار|پنج|شش|هفت)\s*روز", normalized_message)
    if duration_match:
        updated.duration = duration_match.group(0)
    elif "آخر هفته" in normalized_message:
        updated.duration = "آخر هفته"

    origin_match = re.search(
        r"(?:مبدا|مبدأ)\s*(?:سفر)?\s*(?:شهر\s+)?([آ-ی]{2,20})"
        r"|(?:از|ساکن)\s+(?:شهر\s+)?([آ-ی]{2,20})(?=\s+(?:حرکت|راه|به|می))",
        normalized_message,
    )
    if origin_match:
        origin = origin_match.group(1) or origin_match.group(2)
    else:
        origin = None
    if origin and origin not in {"سفر", "این", "اون", "آنجا"}:
        updated.origin = origin

    for region in ("شمال", "جنوب", "کویر", "غرب", "شرق"):
        if region not in normalized_message:
            continue
        rejected = any(
            phrase in normalized_message for phrase in (f"{region} نه", f"نه {region}", f"{region} نمی")
        )
        if rejected and updated.region == region:
            updated.region = None
        elif not rejected:
            updated.region = region

    companion_terms = {
        "همسر": ("همسر", "شوهر", "زنم"),
        "خانواده": ("خانواده",),
        "کودک": ("کودک", "بچه"),
        "دوستان": ("دوستام", "دوستان"),
        "سالمند": ("سالمند", "پدرم", "مادرم"),
    }
    for label, terms in companion_terms.items():
        if any(term in normalized_message for term in terms):
            updated.companions = [item for item in updated.companions if item not in terms]
            if label not in updated.companions:
                updated.companions.append(label)

    for transport in ("آفرود", "ماشین سواری", "خودرو سواری", "ماشین", "خودرو", "قطار", "اتوبوس"):
        if transport in normalized_message:
            updated.transport = transport
            break

    for mood, variants in {
        "آرام": ("آرام", "آروم"),
        "رمانتیک": ("رمانتیک", "عاشقانه"),
        "ماجراجویانه": ("ماجراجو", "هیجان"),
        "خلوت": ("خلوت", "دور از شلوغی"),
    }.items():
        if any(variant in normalized_message for variant in variants):
            updated.moods = [item for item in updated.moods if item not in variants]
            if mood not in updated.moods:
                updated.moods.append(mood)

    rejects_camping = "کمپ" in normalized_message and any(
        phrase in normalized_message for phrase in ("بدون", "نمی خوام", "نمیخوام", "نمی خواهم", "نه")
    )
    if rejects_camping:
        updated.accommodation = [item for item in updated.accommodation if "کمپ" not in item]
        if "کمپ" not in updated.avoid:
            updated.avoid.append("کمپ")
    elif "کمپ" in normalized_message and not any(
        phrase in normalized_message for phrase in ("نه", "نمی", "بدون")
    ):
        updated.avoid = [item for item in updated.avoid if item != "کمپ"]
        if "کمپ" not in updated.accommodation:
            updated.accommodation.append("کمپ")

    for accommodation in ("هتل", "ویلا", "اقامتگاه بوم گردی", "بوم گردی", "خانه محلی"):
        if accommodation in normalized_message and accommodation not in updated.accommodation:
            updated.accommodation.append(accommodation)

    for activity in ("پیاده روی", "کوهنوردی", "عکاسی", "شنا", "قایق سواری", "دوچرخه سواری"):
        if activity in normalized_message and activity not in updated.activities:
            updated.activities.append(activity)

    semantic_parts = (
        [updated.objective]
        + updated.destination_names
        + [f"از {updated.origin}"]
        + ([updated.region] if updated.region else [])
        + ([updated.duration] if updated.duration else [])
        + updated.companions
        + updated.moods
        + updated.activities
        + ([updated.transport] if updated.transport else [])
        + updated.accommodation
        + updated.must_haves
        + [f"بدون {item}" for item in updated.avoid]
        + updated.constraints
    )
    if semantic_parts:
        updated.semantic_query = "سفر " + "، ".join(dict.fromkeys(semantic_parts))

    updated.revision = current.revision + 1
    updated.updated_at = datetime.now(timezone.utc).isoformat()
    if not updated.semantic_query:
        updated.semantic_query = updated.objective or latest_message
    return updated


@lru_cache
def get_trip_planner_agent() -> Agent[AgentDeps, str]:
    settings = get_settings()
    agent = Agent(
        _build_model(settings),
        deps_type=AgentDeps,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.system_prompt
    async def add_memory_context(ctx: RunContext[AgentDeps]) -> str:
        parts = [f"Current season (server clock): {current_season()}."]
        if ctx.deps.travel_goal is not None:
            parts.append(
                "## Active travel goal (canonical current session state)\n"
                + json.dumps(ctx.deps.travel_goal.model_dump(), ensure_ascii=False)
            )

        if ctx.deps.client is not None:
            try:
                prefs = await ctx.deps.client.long_term.get_preferences_for(ctx.deps.user_id)
                if prefs:
                    pref_lines = [f"- {p.category}: {p.preference}" for p in prefs]
                    parts.append("## Known preferences for this user\n" + "\n".join(pref_lines))
            except Exception:
                logger.warning("Could not load user preferences", exc_info=True)

        if ctx.deps.graph is not None:
            try:
                history = await ctx.deps.graph.get_user_history(
                    ctx.deps.user_id,
                    exclude_session_id=ctx.deps.session_id,
                )
                if any(history.values()):
                    parts.append(
                        "## Prior travel history for this user\n"
                        + json.dumps(history, default=str, ensure_ascii=False)
                    )
            except Exception:
                logger.warning("Could not load user trip history", exc_info=True)

        return "\n\n".join(parts)

    @agent.tool
    async def tool_search_destinations(
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
        limit: int = 10,
    ) -> str:
        """Search the destination graph with the constraints derived from the
        user's request. Leave filters as null when the user didn't specify them.

        Internally this checks the curated destination set first, then
        destinations graph-linked (real coordinates) to a curated result, and
        only falls back to the larger OSM-derived dataset after that (in
        which case season/trip-type/stay-duration/ecotourism filters are
        dropped for that fallback, since that dataset doesn't carry that
        metadata). `location` also handles informal Iranian regions ("شمال",
        "جنوب", "کویر", ...) by expanding them into real provinces and, if a
        plain name match still finds nothing, a geospatial radius around the
        resolved area -- pass the user's phrase as-is rather than picking one
        city yourself. Each result has an internal `source_is_osm` flag -- use
        it only to decide whether a result's missing fields need filling in
        from your own knowledge (see system prompt); never mention it, or any
        notion of "data source" or "dataset", to the user.

        Args:
            categories: e.g. ["طبیعت", "تاریخی", "کوهنوردی"]
            seasons: e.g. ["تابستان"]
            trip_types: e.g. ["کمپ", "بازدید"]
            location: city, province, or informal region explicitly requested
                by the user, e.g. "اندیمشک", "خوزستان", or "شمال"
            max_distance_km: hard cap on distance from Tehran
            max_travel_hours: hard cap on one-way driving time
            max_physical_readiness: "کم" | "متوسط" | "زیاد" (ceiling, not exact match)
            stay_duration: "یک روز" | "دو روز" (exact match against the data)
            ecotourism: true if they want eco-lodging options
        """
        result = await search_destinations(
            ctx, categories, seasons, trip_types, location, max_distance_km,
            max_travel_hours, max_physical_readiness, stay_duration, ecotourism,
            limit=limit,
        )
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_semantic_search_destinations(
        ctx: RunContext[AgentDeps], goal: str | None = None, limit: int = 8
    ) -> str:
        """Search embedded Destination nodes by the meaning of the active travel
        goal. Use for moods, desired experiences, activities, companions, and nuanced
        follow-up intent that exact category filters cannot express. Hard constraints
        still need verification through structured search and destination details."""
        result = await semantic_search_destinations(ctx, goal, limit)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_get_destination_details(ctx: RunContext[AgentDeps], name_or_id: str) -> str:
        """Resolve one explicitly named destination and get its full details.
        Always use this first when the user names a place. It normalizes Persian name
        variants and generic prefixes such as ییلاق/روستا, and prefers the richer
        curated destination when duplicate map records exist. Returns description,
        cautions, vehicles, exit routes from Tehran, and nearby destinations."""
        result = await get_destination_details(ctx, name_or_id)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_find_destinations_near(
        ctx: RunContext[AgentDeps], name_or_id: str, radius_km: float = 50
    ) -> str:
        """Find other destinations close to a given one, to build a combined
        itinerary (e.g. a waterfall + a nearby historic village)."""
        result = await find_destinations_near(ctx, name_or_id, radius_km)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_build_trip_map(
        ctx: RunContext[AgentDeps],
        stops: list[dict],
        origin: dict | None = None,
        round_trip: bool = False,
    ) -> str:
        """Call this once you've settled on the concrete list of destinations
        for the plan (even a single destination) -- this is REQUIRED every
        time you finalize a plan, not optional, and not replaced by any
        distance_km/travel_time_hours already present on the destination
        from search/details results (those don't give coordinates or visit
        order). It resolves each stop's coordinates (preferring what the destination-search/details
        tools already gave you; only geocodes by name as a last resort),
        computes the best visiting order for multi-stop trips, and returns
        real driving distance/duration between consecutive stops -- use the
        returned `total_duration_hours` / per-leg values to judge whether the
        plan actually fits the time the user has (e.g. don't spread stops
        that need 8+ hours of driving across a 2-day trip that's mostly meant
        for being *at* places, not driving between them).

        Args:
            stops: one dict per destination, in any order, e.g.
                [{"name": "...", "id": "...", "latitude": .., "longitude": ..}, ...].
                `id`/`latitude`/`longitude` are optional -- pass whatever you
                already have from search/details results; missing ones are
                resolved automatically. `name` is required.
            origin: {"name": ..., "latitude": .., "longitude": ..} -- the
                trip's starting point. Omit to default to central Tehran
                (matches the "assume Tehran" rule), or pass the user's
                stated origin/coordinates if different.
            round_trip: true if the route should return to the origin
                (relevant mainly for a single loop-day trip); usually false.
        """
        result = await build_trip_map(ctx, stops, origin, round_trip)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_check_reachable_within_time(
        ctx: RunContext[AgentDeps],
        origin: dict,
        minutes: float,
        candidates: list[dict],
    ) -> str:
        """Before proposing destinations for a short/time-boxed trip, use
        this to filter candidates down to ones actually reachable within
        the user's time budget (one-way driving minutes) from `origin`.
        Returns each candidate with `reachable` (true/false) and its
        approximate distance -- drop or flag unreachable ones instead of
        including them in the plan.

        Args:
            origin: {"name": ..., "latitude": .., "longitude": ..}
            minutes: one-way driving time budget in minutes
            candidates: [{"name": ..., "id": .., "latitude": .., "longitude": ..}, ...]
        """
        result = await check_reachable_within_time(ctx, origin, minutes, candidates)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_find_nearby_amenities(
        ctx: RunContext[AgentDeps], latitude: float, longitude: float, layer: str, radius_m: int = 3000
    ) -> str:
        """Find nearby amenities (restaurant, hotel, parking, hospital, cafe,
        ...) around a destination's coordinates, to enrich a recommendation
        with practical nearby options. `layer` is an English slug such as
        "restaurant", "hotel", "parking", "cafe", "hospital", "bank"."""
        result = await find_nearby_amenities(ctx, latitude, longitude, layer, radius_m)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_list_available_filters(ctx: RunContext[AgentDeps]) -> str:
        """List the actual category/season/trip-type values present in the
        graph, so you filter using real vocabulary instead of guessing."""
        result = await list_available_filters(ctx)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_save_user_preference(ctx: RunContext[AgentDeps], category: str, preference: str) -> str:
        """Persist a durable preference about this user, e.g.
        category="companion_type", preference="traveling with romantic partner";
        category="pace", preference="prefers low-difficulty, relaxed trips";
        category="avoid", preference="dislikes crowded weekend spots"."""
        result = await save_user_preference(ctx, category, preference)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_get_user_preferences(ctx: RunContext[AgentDeps]) -> str:
        """Fetch this user's stored preferences from prior sessions."""
        result = await get_user_preferences(ctx)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_record_trip_plan(
        ctx: RunContext[AgentDeps],
        destination_ids: list[str],
        trip_summary: str,
        trip_type: str | None = None,
        companions: str | None = None,
    ) -> str:
        """Save the itinerary you just proposed so future sessions remember it
        and avoid repeating the same destinations by default."""
        result = await record_trip_plan(ctx, destination_ids, trip_summary, trip_type, companions)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_record_destination_feedback(
        ctx: RunContext[AgentDeps], destination_id: str, sentiment: str, note: str | None = None
    ) -> str:
        """Record feedback about a specific destination.
        sentiment must be one of: liked, disliked, visited."""
        result = await record_destination_feedback(ctx, destination_id, sentiment, note)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_get_user_trip_history(ctx: RunContext[AgentDeps]) -> str:
        """Get this user's liked/disliked/visited destinations and past trips."""
        result = await get_user_trip_history(ctx)
        return json.dumps(result, default=str, ensure_ascii=False)

    @agent.tool
    async def tool_get_similar_past_trip_plans(
        ctx: RunContext[AgentDeps], current_request: str, limit: int = 2
    ) -> str:
        """Look up how similar past requests were successfully handled."""
        result = await get_similar_past_trip_plans(ctx, current_request, limit)
        return json.dumps(result, default=str, ensure_ascii=False)

    return agent
