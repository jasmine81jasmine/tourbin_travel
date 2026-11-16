# Trip Planner Agent

An LLM trip-planning agent for short trips around Tehran, built the same way as
[Lenny's Memory](https://github.com/neo4j-labs/agent-memory) (FastAPI +
PydanticAI + `neo4j-agent-memory`), but pointed at a tourism knowledge graph
instead of a podcast transcript graph.

Create an account with an email address or international phone number and a password,
then send free text ("یک سفر رمانتیک آخر هفته با همسرم، دو روز وقت داریم")
with an account-owned `session_id`. It replies with a plan in natural language -- while
quietly updating a Neo4j graph with what it learned about you and what it
recommended, so the next conversation is smarter.

```
POST /api/auth/register   {"identifier":"you@example.com","password":"your password"}
POST /api/session         (with the returned HttpOnly cookie)
POST /api/chat            {"message":"...","session_id":"session_..."}
GET  /api/auth/history    (with the returned HttpOnly cookie)
```

## Why Neo4j + a graph, not just the ES index as-is

Your current Elasticsearch index is a flat table of ~20+ destination
documents (see `fields.json` / `mapping.json` / `sample_documents.json`).
That's great for full-text search, but a trip planner agent needs to reason
about *relationships*: "what else is near this waterfall", "what has this
user already seen", "what worked for similar requests before". A graph makes
those first-class, queryable relationships instead of application-side joins.

## Graph schema

```
(:Destination {id, name, description, distance_km, travel_time_hours,
                stay_duration, physical_readiness, road_type, ecotourism,
                facilities_level, facilities_score, user_rating_text,
                user_rating_score, sensitive_notes, location: Point,
                embedding: Vector})

(:Destination)-[:LOCATED_IN]->(:City)-[:PART_OF]->(:Province)
(:Destination)-[:HAS_CATEGORY]->(:Category)          -- طبیعت, تاریخی, کوهنوردی
(:Destination)-[:BEST_SEASON]->(:Season)              -- بهار, تابستان, پاییز, زمستان
(:Destination)-[:SUITABLE_FOR]->(:TripType)           -- کمپ, بازدید
(:Destination)-[:ACCESSIBLE_BY]->(:Vehicle)           -- سواری, آفرود, قایق
(:Destination)-[:NEAR]->(:Landmark)                   -- مکان‌های شاخص نزدیک
(:Destination)-[:VIA_EXIT]->(:ExitRoute)              -- خروجی (بابایی/کرج/...)
(:Destination)-[:NEAR_DESTINATION {distance_km}]-(:Destination)  -- computed geo-proximity

-- Grows through interaction:
(:User {id})-[:LIKED|DISLIKED|VISITED {note, updated_at}]->(:Destination)
(:User)-[:PLANNED]->(:Trip {id, summary, trip_type, companions, planned_at})
(:Trip)-[:INCLUDES]->(:Destination)
```

Plus the three generic memory layers from `neo4j-agent-memory`, which store
their own nodes in the same database:
- **Short-term**: `Message`/`Session` nodes -- this conversation's turns.
- **Long-term preferences**: durable facts like "prefers relaxed trips",
  "usually travels with a partner" -- persist across sessions.
- **Reasoning traces**: records of how past requests were handled, used for
  few-shot-style "learn from a similar past task".

See `backend/etl/constraints.cypher` for the full DDL (uniqueness
constraints, the vector index for semantic search over descriptions, a point
index for geo queries, and a full-text index).

## Switching LLM / embedding providers

`backend/.env.example` has two blocks for each of `LLM_*` and `EMBEDDING_*`:

- **Option A -- AvalAI**, an OpenAI-compatible proxy, used as-is via `LLM_BASE_URL`/`EMBEDDING_BASE_URL`.
- **Option B -- the university gateway** (`fumllm.um.ac.ir` for chat, `ai-gateway.um.ac.ir` for embeddings). Its APIs aren't OpenAI-wire-compatible (single `query` string, no tool calling, a slightly different embeddings response shape), so `backend/src/adapters/university_proxy.py` is a small FastAPI router, mounted at `/adapters/university`, that speaks OpenAI's wire format on one side and the university's real API on the other. Point `LLM_BASE_URL` / `EMBEDDING_BASE_URL` at it (`http://localhost:8000/adapters/university/v1`) and everything else -- pydantic-ai's model, `neo4j-agent-memory`'s embedding provider -- works unmodified, exactly like it already does for AvalAI.

Uncomment one block per section in `.env`, comment out the other, restart the app. You can mix (e.g. AvalAI for chat + university for embeddings).

Since the university LLM has no native tool/function calling, the adapter emulates it by re-serializing the full conversation into text on every call and asking the model to reply in a specific JSON shape (tool call vs. final answer) -- see the docstring in `university_proxy.py` for details and caveats. This works, but is less reliable with a small model like `gemma4` than AvalAI/Anthropic's native tool calling.

The university embedding model (`mpnet-multilingual`) is natively 768-dim, matching this project's `EMBEDDING_DIMENSIONS` default, so no vector-index resizing is needed when switching.

## Loading your existing ES data

```bash
cd backend
uv venv && source .venv/bin/activate  # or your usual Python env
uv pip install neo4j elasticsearch

# from a JSON export (what you gave me, sample_documents.json, or a full scroll dump):
python etl/load_destinations.py --source json --path sample_documents.json

# or straight from the live ES cluster:
python etl/load_destinations.py --source es --es-host http://localhost:9200 --index tehran_destinations
```

The ETL (`backend/etl/load_destinations.py`):
1. Maps each Persian ES field to a normalized property (`FIELD_MAP`).
2. Splits multi-value free-text fields (`مکان_های_شاخص_نزدیک`,
   `وسیله_نقلیه`, `بهترین_فصل`) into proper lists.
3. `MERGE`s dimension nodes (Category, Season, City, ...) so they're shared
   across destinations instead of duplicated per document.
4. Computes a `NEAR_DESTINATION` edge between any two destinations within
   30km, so the agent can propose "combine X with nearby Y" itineraries.
5. Is idempotent: re-running it updates existing nodes (id is a stable hash
   of name + city/province) rather than creating duplicates.

Re-run it whenever your ES index changes, or point `load_es_source()` at a
live cluster and put it on a schedule.

## Loading an Overpass GeoJSON export

The nationwide Overpass export is too large to load with `json.load()`.
`load_osm_geojson.py` uses `ijson` to read one feature at a time and commits
small Neo4j batches, so RAM usage does not grow with the file size.

```bash
cd backend
pip install -r requirements.txt

# Optional: audit every property key before changing the allowlist.
python etl/audit_geojson.py \
  --path ../export.geojson \
  --output geojson-property-audit.json

# Idempotent import. Credentials are read from .env or the shown flags.
python etl/load_osm_geojson.py \
  --path ../export.geojson \
  --batch-size 500 \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-database neo4j
```

Set `NEO4J_PASSWORD` in `backend/.env` or the shell rather than putting it in
command history. The importer creates its required constraints/indexes. Every
valid GeoJSON record is preserved as `:OsmFeature`; only records with a
Persian-script `name:fa` or `name` also receive
`:Destination:OsmAttraction`. Other localized text (`name:en`,
`description:en`, and similar) is excluded. Language-neutral planning data
such as coordinates, elevation, access, fee, opening hours, drinking water,
wheelchair access, seasonal status, website, phone, and protected-area
classification is retained for all features.

City nodes are shared, not duplicated, across the two sources: both this
script and `load_destinations.py` `MERGE (city:City {name: ...})` on the same
label+key, so a city that already exists from the curated import (e.g.
"آمل") gets reused rather than re-created when the same city name shows up
in the GeoJSON, and vice versa if you load the GeoJSON first. This only works
when the two sources spell the city name identically -- see
`etl/load_destinations.py::_parse_city_province` for how the curated side
normalizes its `شهرستان_استان` field.

`load_osm_geojson.py` automatically runs `etl/link_nearby.py` at the end
(city/province geospatial centroids + `NEAR_DESTINATION` across *all*
destinations, curated and OSM). If you ever load the GeoJSON in smaller
pieces, or re-import just the curated JSON afterward, re-run it manually:

```bash
python etl/link_nearby.py \
  --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password '...'
```

## How the agent searches ("near \<city\>", curated vs. OSM priority)

`TripGraphRepository.search_destinations` (`backend/src/memory/graph.py`) is
a three-tier search, in priority order:

1. **Curated** (`sample_documents.json`) destinations matching every filter.
2. If tier 1 doesn't fill the requested count: destinations linked via
   `NEAR_DESTINATION` to a curated result -- these may be OSM nodes, but
   their real-coordinate proximity to a curated destination is itself the
   signal that they're worth recommending.
3. If still short: the broader OSM pool, first by name match on
   `location`, then (if that finds nothing) by a geospatial radius around
   the resolved City/Province centroid set by `etl/link_nearby.py`.

Informal regions ("شمال", "جنوب", "کویر", ...) are expanded to real
provinces by `backend/src/agent/regions.py::expand_location` before
`location` ever reaches the graph query -- a plain city/province name passes
through unchanged.

The two-tier OSM ontology keeps anonymous map features out of recommendations:

```text
(:OsmFeature[:NaturalFeature|ProtectedArea])
(:OsmFeature:Destination:OsmAttraction[:NaturalAttraction])  // Persian name
  -[:HAS_CATEGORY]->(:Category {name})
  -[:HAS_OSM_TYPE]->(:OsmFeatureType {id, key, value, name_fa})
  -[:IN_COUNTRY]->(:Country {code: "IR", name: "ایران"})
```

`description_fa` is also copied to `description`, allowing the existing agent
tools and full-text index to use it without a separate code path. Nationwide
proximity is calculated from Neo4j point properties at query time instead of
creating an unmanageably large all-pairs `NEAR_DESTINATION` graph.

## Your data is enough to get started, but here's what would make
## recommendations meaningfully better

Everything below is inferred indirectly right now (e.g. "romantic" is
guessed from category + difficulty + rating). Adding these as explicit
fields would let the agent filter directly instead of guessing:

**Trip-fit tags (highest impact for your stated use case):**
- `مناسب_برای` (suitability tags): رمانتیک، خانوادگی، دوستانه، تنها، گروه بزرگ.
  This directly answers your romantic-trip / family-trip example.
- `حداقل_سن` / `مناسب_کودک`: whether small children can realistically go.
- `حداکثر_افراد` or group-size guidance for camping spots.

**Planning logistics:**
- `هزینه_تقریبی` (price band or number): cost is almost always a real
  constraint and isn't in the data at all today.
- `ساعات_بازدید` / `روزهای_تعطیل`: opening hours/days, closed seasons.
- `نیاز_به_مجوز`: permits required (some forests/protected areas need one).
- `پارکینگ` and `پوشش_آنتن_دهی` (parking, mobile signal) -- both matter a
  lot for "family trip with kids" planning.
- `اقامتگاه_های_نزدیک`: structured nearby lodging (name, type, distance,
  price band) instead of leaving that to `معرفی` free text.

**Richer descriptors:**
- Numeric `سطح_سختی` (1-5) alongside the current 3-value
  `آمادگی_بدنی`, for finer filtering.
- `برچسب‌ها` free tags (آبشار, چشمه آب‌گرم, دریاچه, غار, ...) -- lets you do
  "show me hot springs" without parsing `معرفی`.
- `تعداد_نظرات` / average numeric rating instead of only a 4-value bucket
  (`نظرات_کاربران`), so ranking isn't so coarse.
- Structured "what to do there" activity list instead of only free text.

**Real-time-ish signals (bigger lift, optional):**
- Seasonal closures/crowding info, live weather integration by coordinates
  (you already have `مختصات`, so this is a pure enrichment job, no schema
  change needed).

None of this blocks getting started -- the agent already gets useful mileage
out of category/season/trip-type/distance/difficulty/rating. Treat the list
above as a prioritized backlog for the ES team.

## How the "memory that updates by interaction" part works

- `tool_save_user_preference` -- called whenever the person reveals a durable
  fact ("I usually prefer quiet places", "I don't like off-road driving").
  Written to `neo4j-agent-memory`'s long-term preference store, available in
  future sessions for the same `user_id`. This depends on the agent calling
  the tool; a one-trip request is not automatically a permanent preference.
- Each planning turn stores a session-scoped `TravelGoal` and links the user
  to requested Category/TripType/Season/Vehicle and requested style/companion
  nodes through `REQUESTED_FOR_TRIP {session_id}`. A later edit replaces that
  session's links. A new session can use the last session's category, trip type
  or mood as a *soft hint* when unspecified, without assuming its companions,
  dates, origin or transport still apply. Explicit new requests win.
- `tool_record_trip_plan` -- called after the agent proposes a concrete
  itinerary. Creates a `(:Trip)` linked to the `(:User)` and the
  `(:Destination)` nodes involved, so "plan something like last time" and
  "don't repeat what we already did" both become graph queries.
- `tool_record_destination_feedback` -- called when the user reacts to a
  specific place (liked/disliked/visited), creating/updating a relationship
  from `(:User)` to `(:Destination)`.
- `tool_get_user_preferences` / `tool_get_user_trip_history` -- called
  proactively at the start of a session so the agent doesn't ask the person
  to repeat context it already has.
- Final graph-backed suggestions are also recorded as `SHOWN_DESTINATION`
  per session (not `VISITED`). Future searches prefer unseen, similarly
  suitable places, while still allowing a named revisit or a repeat if the
  graph has no credible fresh option. Among comparably ranked candidates a
  per-user/per-session tie-breaker varies suggestions without relaxing filters.
- Questions such as «تو چیا از من میدونی؟» read user-scoped stored preferences,
  earlier session requests and explicit destination feedback directly; the
  agent does not guess a memory answer. A broad request to plan from memory
  without a duration first screens a clearly labelled one-day assumption near
  Tehran, rather than presenting unverified faraway destinations as a plan.
- The final route explanation is written *after* feasibility screening from
  the destinations actually offered and user-scoped trip context. A follow-up
  asking why those routes were chosen reads the last proposed routes rather
  than inventing a generic rationale. Choosing one previously offered place
  keeps that destination as the plan's anchor; vehicle questions answer about
  its recorded access instead of starting a new search.

This is the same pattern Lenny's Memory uses for podcast preference
learning; we just added the tourism-specific `Trip`/`LIKED`/`DISLIKED`
relationships since "preferences" alone don't capture "please don't suggest
this exact waterfall again."

## Project layout

```
trip-planner-agent/
├── docker-compose.yml        # local Neo4j (5.x + APOC)
└── backend/
    ├── requirements.txt
    ├── .env.example
    ├── etl/
    │   ├── constraints.cypher      # schema DDL, run once
    │   └── load_destinations.py    # ES/JSON -> Neo4j
    └── src/
        ├── main.py                # FastAPI app
        ├── config.py
        ├── memory/
        │   ├── client.py          # neo4j-agent-memory MemoryClient wrapper
        │   └── graph.py           # TripGraphRepository (domain Cypher)
        ├── agent/
        │   ├── agent.py           # PydanticAI agent + tool registrations
        │   ├── dependencies.py
        │   └── tools.py           # tool business logic
        └── api/
            ├── schemas.py
            └── routes/chat.py     # POST /api/chat
```

## Running it

### Option A: with Docker (Neo4j only)

```bash
docker compose up -d neo4j          # Neo4j on bolt://localhost:7687
cd backend
cp .env.example .env                # fill in your LLM/embedding API keys
uv pip install -r requirements.txt  # or pip install -r requirements.txt
cat etl/constraints.cypher | cypher-shell -u neo4j -p password   # or run in Neo4j Browser
python etl/load_destinations.py --source json --path ../../sample_documents.json
uvicorn src.main:app --reload
```

### Option B: fully local, no Docker at all

**1. Install Neo4j natively (pick one):**

- **Neo4j Desktop** (easiest, any OS, GUI): download from
  https://neo4j.com/download/, create a new local DBMS, set a password,
  hit "Start". It listens on `bolt://localhost:7687` by default.
- **Ubuntu/Debian, command line:**
  ```bash
  wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
  echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
  sudo apt update && sudo apt install neo4j apoc  # or install APOC as a plugin jar, see below
  sudo systemctl enable --now neo4j
  ```
  Set the initial password (first login only accepts changing the default):
  ```bash
  cypher-shell -u neo4j -p neo4j "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'yourpassword';"
  ```
- **macOS:** `brew install neo4j` then `neo4j start`.

Either way, the constraints file uses `CREATE VECTOR INDEX`, which needs
**Neo4j 5.15+**, and `point.distance()` for proximity, which is built-in —
no extra plugin needed for that part. APOC is optional here (nothing in this
project currently calls APOC procedures directly), so you can skip installing
it if it's giving you trouble.

**2. Python environment (no `uv` needed either, plain venv works):**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

**3. Edit `.env`** — this is the part that actually changes for you:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=yourpassword        # <-- whatever you set in step 1, NOT elastic's password

ELASTIC_URL=https://localhost:9200
ELASTIC_USERNAME=elastic
ELASTIC_PASSWORD=F_HjeyJJF-A_qVNXUv-K
INDEX_NAME=tehran_destinations
ELASTIC_VERIFY_CERTS=false         # local ES usually has a self-signed cert

ANTHROPIC_API_KEY=...              # or OPENAI_API_KEY if you switch LLM_MODEL
OPENAI_API_KEY=...                 # needed regardless, for embeddings
```

Two things worth calling out about the credentials you shared:
- `ELASTIC_*` and `INDEX_NAME` are only read by `etl/load_destinations.py`
  (via `os.environ`, loaded through `load_dotenv()`) — the FastAPI app never
  talks to Elasticsearch at runtime, only Neo4j.
- `NEO4J_PASSWORD` has nothing to do with your Elastic password — it's
  whatever you set when you installed Neo4j in step 1. Don't reuse the
  Elastic one there.

**4. Apply the schema and load your data:**

```bash
cypher-shell -u neo4j -p yourpassword -f etl/constraints.cypher
# pull straight from your live, authenticated ES cluster:
python etl/load_destinations.py --source es
# (equivalent to loading from the JSON export you already gave me:)
# python etl/load_destinations.py --source json --path ../../sample_documents.json
```

**5. Run the API:**

```bash
uvicorn src.main:app --reload --port 8000
```

No Docker involved anywhere in Option B — Neo4j runs as a normal local
service/process, and the backend is a normal `uvicorn` process.

Then open `http://127.0.0.1:8000/` to register/sign in, or use the HTTP API:

```bash
curl -c /tmp/tourbin-cookies -X POST http://127.0.0.1:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"identifier":"you@example.com","password":"a-long-password"}'
curl -b /tmp/tourbin-cookies -X POST http://127.0.0.1:8000/api/session
curl -b /tmp/tourbin-cookies -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"دو روز وقت داریم، یه سفر آروم بریم","session_id":"session_FROM_PREVIOUS_RESPONSE"}'
```

Accounts, password hashes and revocable 30-day login sessions live in Neo4j.
The old `tourbin_user_id` cookie and request-body `user_id` no longer grant access.
For security, chats created before accounts existed are **not automatically claimed**
by a new account. Run the app with Neo4j accessible; account endpoints fail closed
if the database is unavailable. Set `AUTH_COOKIE_SECURE=true` on HTTPS deployments
(keep it false for local HTTP). Password login requires transport HTTPS in production.

### Structured preference memory

On startup the backend adds a controlled feature vocabulary to Neo4j. It
retains the original destination properties and existing library `:Preference`
nodes. Existing `Category`/`Season`/`TripType`/`Vehicle` nodes receive stable
`preference_id` values; destination properties are also linked to shared
`FacilitiesLevel`, `TravelTime`, `RoadType`, `PhysicalReadiness`, and
`DistanceRange` nodes. Travel time bins from **Tehran** are <=2h, >2–5h,
>5h; distance bins from Tehran are <=100km, >100–300km, >300km.
Unknown/ambiguous facility or road values are left unlinked.
`TravelStyle`, `Activity`, `Environment`, `FoodPreference` and
`AccommodationType` have a small controlled vocabulary but no invented
destination links.

When the user starts a new chat or logs out, the UI calls
`POST /api/auth/history/{session_id}/finalize` with their login cookie. This
endpoint can be retried: the LLM sees owned conversation messages (with message
IDs), current structured preferences and allowed feature nodes. The backend
rejects unknown features or foreign/non-user message IDs, then atomically
connects the existing user-message evidence to the feature and increments
`positive_count`/`negative_count` **at most once per polarity per conversation**.
`score = positive_count - negative_count` is a net evidence count, not a
probability. Candidate destinations are soft-ranked by matching shared
feature scores *after* explicit filters, without removing valid candidates.
Closing the browser tab sends a best-effort `sendBeacon` finalize request.
If the browser cannot deliver it, a server worker scans durable Neo4j
`ChatSession` revisions once per minute and extracts preferences after
10 minutes of inactivity. Failed attempts retry after five minutes; a
successful zero-update extraction is marked checked too. This is an
inactivity approximation, not proof that a tab closed: a user who pauses
for 10 minutes can have preferences processed while their chat remains
open. New messages mark it pending again. For immediate, explicit control,
call the finalize endpoint; failures leave the conversation intact.

Local regression tests, with the project root as working directory:

```bash
PYTHONPATH=backend NEO4J_URI=bolt://localhost:7687 \
  /home/yasi/tourbin_server_deploy/.venv/bin/python -m pytest backend/tests
# Include isolated account integration tests against the running local Neo4j:
RUN_AUTH_INTEGRATION=1 PYTHONPATH=backend NEO4J_URI=bolt://localhost:7687 \
  /home/yasi/tourbin_server_deploy/.venv/bin/python -m pytest backend/tests/test_auth.py
```

## Things to verify before production use

- `long_term.add_preference` / `search_preferences` and
  `short_term.add_message` method names/signatures in `src/memory/client.py`,
  `src/agent/tools.py`, and `src/api/routes/chat.py` are written against the
  same `neo4j-agent-memory` conventions used in Lenny's Memory's
  `agent/tools.py`, `memory/client.py`. Pin a specific commit/tag of
  `neo4j-labs/agent-memory` and check its current public API before deploying
  -- it's an actively developed Labs project.
- `memory_client._client.driver` (used in `chat.py` to share a driver with
  `TripGraphRepository`) is an internal accessor, same pattern Lenny's Memory
  uses (`ctx.deps.client._client.execute_read(...)`); confirm it still holds
  in whatever version you install.
- `_build_model()` in `src/agent/agent.py` constructs `OpenAIChatModel`/`AnthropicModel`
  explicitly (rather than a bare `"openai:model"` string) specifically so
  `LLM_BASE_URL`/`AVALAI_BASE_URL` is honored -- confirm `OpenAIProvider` and
  `OpenAIChatModel` are importable from those exact module paths in whatever
  pydantic-ai version you pin; the import paths have moved before between
  releases.
- Embeddings default to real OpenAI, separate from the AvalAI key, since
  AvalAI's sample script only demonstrates chat completions. If AvalAI also
  proxies `/v1/embeddings`, set `EMBEDDING_BASE_URL` to test it before relying
  on it in production.
### Goal-driven planning

On a planning turn, the agent decomposes the current request and relevant
structured preferences into session-scoped subgoals, hard/soft constraints and
a short execution outline. Explicit current requests take precedence over
older preferences. The existing planner chooses Neo4j and Neshan tools;
multi-stop ordering and road durations remain Neshan-backed, not manually
calculated. The final itinerary is checked against **final route stops and
graph categories**, with at most one additional planning attempt for uncovered
high-priority verifiable goals. Internal `goal_coverage` means *planned*
coverage, not completed visits; it is not returned in `/api/chat` or displayed
as raw goal state. Unknown hours, prices, activities and missing routing
remain unverified.

The goal designer sees all controlled ontology feature nodes as **candidate
dimensions**, not mandatory stops, together with the user's structured
preference scores. It merges new subgoals with existing session goals, retaining
older constraints unless explicitly retired. Every follow-up in an active trip
session, even a factual question, records a snapshot containing that user turn.
The LLM decides whether new requirements need a revised plan or just an answer
and whether the user actually selected a named destination. A graph match
against a word in the user's sentence is never by itself treated as a selected
destination (for example «برم یه جا کمپ کنم» does not select a place named
«برم»). Answering does not by itself rerun the route planner. Tool calls
remain in the existing reasoning traces, not separate per-step goal nodes.

Each save creates a new `TravelGoal` JSON node linked from the previous node
with `REVISED_TO` and from the owned `ChatSession` with `HAS_GOAL_REVISION`;
the newest revision for the user's session is loaded on
future turns. Startup migrates the previous unique `session_id` constraint to
a unique goal `id` and a non-unique `session_id` index without deleting old
goals. Existing deployments should restart the backend to apply this schema
migration before using the new revision writes.

New revisions also have an immutable `GoalRevisionEvent -[:CREATED_REVISION]-> TravelGoal`
with a cause (`user_goal_update`, `tool_observation`, `subgoal_decomposition`,
`routes_proposed`, `plan_coverage`, `assistant_offer`, `response_evaluation`,
or `direct_answer`). A per-turn ID links the actual user `Message` to all of
its revisions via `TRIGGERED_REVISION` and the completed `ReasoningTrace` via
`HAS_GOAL_REVISION`. Tool observations record a unique `observation_id`,
source (`preload` versus `agent_tool`), tool name and candidate-name summary.
The same ID is appended to the agent tool return; when that exact return
appears in a recorded trace, `ToolCall -[:OBSERVED_IN]-> GoalRevisionEvent`
links the call to its snapshot. No link is guessed from tool name or time:
preloads, missing traces, and tools omitted by the trace recorder will not
have a `ToolCall` edge. Existing historical revisions cannot be retroactively
assigned exact tool-call IDs; this audit trail applies to new turns.

For a timeline in Neo4j, replace the parameter with a session ID:

```cypher
MATCH (event:GoalRevisionEvent {session_id:$session_id})
      -[:CREATED_REVISION]->(goal:TravelGoal)
OPTIONAL MATCH (call:ToolCall)-[:OBSERVED_IN]->(event)
RETURN goal.revision AS revision, event.cause AS cause,
       event.observation_source AS source, event.tool AS tool,
       event.candidate_names AS candidates, call.tool_name AS traced_tool,
       call.arguments AS arguments
ORDER BY revision;
```

### Changing goals within a conversation (OnGoal-inspired)

For each user turn, `understand_turn()` infers verbatim question/request clauses
and structured trip updates in the context of the session and the latest
assistant offer. `infer_and_merge()` compares them with the saved
`ConversationalGoal` records: unrelated requirements are **kept**, explicit
conflicts are **replaced**, and compatible requirements are **combined**.
Independent parts of a compound goal survive a change to just one dimension.
A question about a destination does not select it; a brief acceptance can
select a pending assistant offer. When the assistant offers independent
options or "one of these or a combination", a bare acceptance asks which
option(s) the user wants; ordinal choices refer to the original offered order.
Names missing from the destination graph remain unverified, not silently
replaced with another place. Requests for other plans rerun the grounded
planner rather than returning an unverified free-form list. A car constraint
requires destination vehicle-access evidence in addition to road travel time.
Short follow-ups such as "دیگه چی داری؟" refer to the preceding plan in the
same session; an explicitly owned vehicle, including Persian spelling
variants of a graph `Vehicle` name, changes the trip's `transport` constraint
instead of being searched as a destination name.
Trip-planning subgoals remain separate from
these conversational goals, and search context is rebuilt from current trip
constraints instead of appending obsolete requests.

After the reply, `evaluate_response()` assesses active conversational goals
as confirmed, contradicted or ignored, requiring verbatim reply evidence for
non-ignored assessments. This is distinct from verified final-itinerary
coverage: prose alone does not verify a route or amenity. Each session
`TravelGoal` revision stores the private merge/evaluation history and links
`ConversationalGoalState` nodes across revisions with `REVISED_TO` and
`DERIVED_FROM_GOAL`. Neither the raw goal graph nor its evaluations are sent to
the user. See `backend/tests/test_conversation_pipeline.py` for multi-turn
conflict, acceptance, question and graph-history cases. The standard tests
mock the LLM and graph; real-provider and Neo4j integration require configured
services.
# Goal progress during planning (ADaPT-inspired)

The turn-understanding LLM supplies explicit trip facts; `design_goals()` uses
the current goal, the destination-feature ontology, and prior user preference
links to propose optional, *soft* goals for unspecified dimensions. An explicit
request always takes priority over an older preference. The chat endpoint no
longer invokes the legacy regex `update_travel_goal()` parser.

Every planning-tool response includes the current goal-progress summary for
the agent's next model call. Candidate destinations and preliminary route
evidence are marked **partial**, never satisfied. Final coverage still requires
the existing verified-itinerary evaluation. Each tool step and goal revision
writes a new `TravelGoal` snapshot with distinct `GoalState` nodes, joined to
their preceding states by `REVISED_TO`; children of an unmet goal use
`DECOMPOSED_INTO`. No previous goal snapshot is overwritten.

If an important, checkable goal remains unmet after planning, the LLM may
decompose *that goal only* into up to two checkable children before the
existing single retry. Unverifiable goals remain explicitly unverified.
This is deliberately a **bounded ADaPT-inspired** flow, not the paper's
unbounded recursive controller.

`tool_search_destinations_by_features` and `tool_get_destination_features`
query **actual** destination-to-ontology edges by exact feature ID. Their
results can give a feature goal preliminary evidence; the selected final
route must independently verify the same edge to satisfy it. Supported
destination links include Category, Season, TripType, Vehicle,
FacilitiesLevel, TravelTime, RoadType, PhysicalReadiness and DistanceRange.
Other ontology vocabulary, such as quiet Environment or FoodPreference,
does not yet establish a destination fact. Decomposition names checkable
outcomes; the executor still chooses its next tool based on each result.

Every **planning** turn also attempts one semantic vector search over
destination embeddings before the planning LLM runs. Its candidates are
included in the agent's context alongside the goal and are tracked as
preliminary evidence; the LLM may call `tool_semantic_search_destinations`
again for a more focused search. This adds one embedding/query per plan,
not per answer/chat. If embeddings or the vector index are unavailable,
the plan proceeds with structured graph tools; semantic similarity never
substitutes for checking hard constraints or amenities.

Run the local tests (no Docker required) with:

```bash
source /home/yasi/tourbin_server_deploy/.venv/bin/activate
PYTHONPATH=backend pytest -q backend/tests
```

To exercise the Neo4j revision-history integration tests against a running
local database, set `NEO4J_URI=bolt://localhost:7687 RUN_AUTH_INTEGRATION=1`.
