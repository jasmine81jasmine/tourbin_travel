# What changed: Neshan maps integration for the trip planner agent

## New files
- `backend/src/adapters/neshan_client.py` — async client for Neshan's
  geocoding, no-traffic routing, TSP (visit order), isochrone, and
  nearby-search endpoints. Every call degrades to `None`/`[]` on any error
  (missing key, network failure, bad response) instead of raising, so the
  chat endpoint never breaks because of this service.
- `backend/src/agent/geo.py` — dependency-free haversine distance,
  road-time estimate, and point-in-polygon helpers, used as fallbacks when
  the Neshan API is unavailable/unconfigured.

## Modified files
- `backend/src/config.py`, `env.example` — new optional `NESHAN_API_KEY`
  (and `NESHAN_BASE_URL`, default `https://api.neshan.org`). Leave it
  blank and everything still works, just without real road distances/TSP
  ordering (falls back to straight-line-distance estimates).
- `backend/src/memory/graph.py` — every destination query now also returns
  `latitude`/`longitude` (from the existing `d.location` point property),
  plus a new `get_coordinates_by_ids()` batch helper. No query behavior
  changed, only extra fields added to existing results.
- `backend/src/agent/dependencies.py` — `AgentDeps` gained `maps` (the
  Neshan client) and a mutable `itinerary_result` field that the new tools
  write to when the agent finalizes a concrete plan.
- `backend/src/agent/tools.py` — three new tool implementations:
  - `build_trip_map`: resolves each stop's coordinates (DB first, then
    geocoding as a last resort), computes the best visiting order via TSP,
    and gets real driving distance/duration between consecutive stops via
    routing. Writes the result to `ctx.deps.itinerary_result`.
  - `check_reachable_within_time`: isochrone-based (or haversine-fallback)
    filter for whether candidate destinations are reachable within a
    driving-time budget from an origin.
  - `find_nearby_amenities`: nearby-search wrapper (restaurants, hotels,
    parking, etc. around a point).
- `backend/src/agent/agent.py` — registers `tool_build_trip_map`,
  `tool_check_reachable_within_time`, `tool_find_nearby_amenities`, and
  adds a system-prompt section telling the agent: check reachability
  before proposing destinations for time-boxed trips, and call
  `tool_build_trip_map` once the plan's destinations are settled, using
  its `total_duration_hours` to judge whether a multi-stop plan actually
  fits the trip's duration. The agent never exposes tool names, raw
  coordinates, "isochrone", "geocoding", etc. to the user — it's folded
  into the same natural Persian reply as before.
- `backend/src/api/schemas.py` — new `Itinerary` / `ItineraryStop` models,
  and an **additive, optional** `itinerary` field on `ChatResponse`.
  `reply`, `session_id`, `user_id` are unchanged, so existing frontend
  code that only reads those three keeps working exactly as before.
- `backend/src/api/routes/chat.py` — wires the Neshan client into the
  agent's dependencies, and after `agent.run()`, reads
  `deps.itinerary_result` back into the response as `itinerary`.

## New response shape (backward compatible)

```json
{
  "reply": "... normal Persian plan text ...",
  "session_id": "...",
  "user_id": "...",
  "itinerary": {
    "origin": {"name": "تهران", "latitude": 35.6892, "longitude": 51.389},
    "stops": [
      {
        "order": 1,
        "name": "دماوند",
        "latitude": 35.95,
        "longitude": 52.1,
        "leg_distance_km_from_previous": 70.4,
        "leg_duration_hours_from_previous": 1.66
      }
    ],
    "total_distance_km": 70.4,
    "total_duration_hours": 1.66,
    "round_trip": false
  }
}
```

`itinerary` is `null` whenever the turn didn't reach a concrete plan (small
talk, a clarifying question, etc.) — exactly what you asked for: name,
lat/lon, and visit order for every destination in the final plan, ready
for a map view later, with zero changes required to the current frontend.

## To deploy

1. Copy the changed/added files into `$LAPTOP_PROJECT_DIR` (same paths as
   in this zip), matching Part B's sync step.
2. Optionally add `NESHAN_API_KEY=...` to the server's `.env` (not git) if
   you want real routing/TSP/isochrone; otherwise leave it unset.
3. `git add -A && git commit -m "Add Neshan maps grounding to trip planner" && git push origin main`
   — your existing GitHub Actions pipeline builds and deploys it exactly
   like any other change. No ETL scripts need to be re-run.
4. No `requirements.txt` changes were needed — `httpx` was already a
   dependency.
