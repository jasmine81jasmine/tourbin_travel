# Trip Planner Agent

An LLM trip-planning agent for short trips around Tehran, built the same way as
[Lenny's Memory](https://github.com/neo4j-labs/agent-memory) (FastAPI +
PydanticAI + `neo4j-agent-memory`), but pointed at a tourism knowledge graph
instead of a podcast transcript graph.

You send it free text ("یک سفر رمانتیک آخر هفته با همسرم، دو روز وقت داریم")
plus a `session_id`, and it replies with a plan in natural language -- while
quietly updating a Neo4j graph with what it learned about you and what it
recommended, so the next conversation is smarter.

```
POST /api/chat
{ "message": "...", "session_id": "user-123", "user_id": "user-123" }
→ { "reply": "...", "session_id": "user-123", "user_id": "user-123" }
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
  fact ("we're celebrating an anniversary", "I don't like off-road driving").
  Written to `neo4j-agent-memory`'s long-term preference store, available in
  future sessions for the same `user_id`.
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

Then:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "دو روز وقت داریم، می‌خوایم با همسرم یه سفر رمانتیک و آروم بریم، سواری داریم", "session_id": "u1", "user_id": "u1"}'
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
