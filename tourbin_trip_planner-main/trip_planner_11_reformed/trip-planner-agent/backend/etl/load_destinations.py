"""ETL: Elasticsearch tourism index -> Neo4j graph.

Reads documents shaped like your existing `tehran_destinations` ES index
(see fields.json / mapping.json) and loads them into Neo4j as a connected
graph instead of a flat document store.

Usage
-----
# From a JSON export (e.g. sample_documents.json, or a full scroll dump):
uv run python etl/load_destinations.py --source json --path sample_documents.json

# Directly from a live Elasticsearch index. Reads ELASTIC_URL / ELASTIC_USERNAME /
# ELASTIC_PASSWORD / INDEX_NAME from .env if present, or pass them as flags:
uv run python etl/load_destinations.py --source es
uv run python etl/load_destinations.py --source es --es-host https://localhost:9200 \
    --es-username elastic --es-password '...' --index tehran_destinations

Both paths converge on `ingest_destinations()`, which is the part you should
extend when you add new properties to the source data.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from neo4j import AsyncDriver, AsyncGraphDatabase

load_dotenv()  # picks up ../.env when run from backend/, or a local etl/.env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("etl")

# Persian source field -> normalized property name used going forward.
FIELD_MAP = {
    "مقصد": "name",
    "شهرستان_استان": "city_province",
    "معرفی": "description",
    "نکات_حساس": "sensitive_notes",
    "مکان_های_شاخص_نزدیک": "nearby_landmarks",
    "خروجی": "exit_route",
    "آمادگی_بدنی": "physical_readiness",
    "دسته_بندی": "categories",
    "وسیله_نقلیه": "vehicles",
    "امکانات": "facilities_level",
    "نظرات_کاربران": "user_rating_text",
    "نوع_سفر": "trip_types",
    "مدت_اقامت": "stay_duration",
    "فاصله_از_تهران_km": "distance_km",
    "فاصله_از_تهران_توضیح": "distance_description",
    "فاصله_زمانی_ساعت": "travel_time_hours",
    "بهترین_فصل": "best_seasons",
    "نوع_جاده": "road_type",
    "بوم_گردی": "ecotourism",
    "مختصات": "coordinates",
}

# Rating text -> numeric score, used for sorting/scoring in the agent tools.
RATING_SCORE = {"عالی": 3, "خوب": 2, "متوسط": 1, "ضعیف": 0}
LEVEL_SCORE = {"زیاد": 3, "متوسط": 2, "کم": 1, "ندارد": 0}


def _split_multi(value: str, seps: str = "/،,-") -> list[str]:
    """Split ES free-text list-ish fields ('شیرگاه/آبشار اسکلیم/...') into a list."""
    if not value:
        return []
    parts = re.split(f"[{re.escape(seps)}]", value)
    return [p.strip() for p in parts if p.strip()]


def _split_seasons(value: Any) -> list[str]:
    """Normalize 'بهترین_فصل' which appears both as a list and as free text
    like 'بهار و تابستان' / 'بهار تابستان' in the sample data."""
    if isinstance(value, list):
        raw = " ".join(value)
    else:
        raw = value or ""
    raw = raw.replace("و", " ")
    known = ["بهار", "تابستان", "پاییز", "زمستان"]
    return [s for s in known if s in raw]


def _parse_city_province(city_province: str) -> tuple[str, str]:
    """Parse the free-text `شهرستان_استان` field into (province, city).

    Seen in the wild (sample_documents.json):
      "اردبیل-تالش"        -> province="اردبیل", city="تالش"       (dash form)
      "مازندران (آمل)"      -> province="مازندران", city="آمل"      (parenthetical form)
      "مازندران - بابل"     -> province="مازندران", city="بابل"     (spaced dash)
      "رشت"                -> province="رشت", city="رشت"           (single term, ambiguous)

    The parenthetical form used to fall through to the single-term case,
    which set both province AND city to the literal string "مازندران
    (آمل)" -- a different City node than the one the plain "آمل" entries
    elsewhere in the same file created for the same real city. That split
    the same city across two City nodes and made province-level location
    matching (e.g. matching "مازندران") accidental (it happened to work
    only because CONTAINS still matched the noisy string), not correct.
    """
    text = city_province.strip()
    if not text:
        return "نامشخص", "نامشخص"

    paren_match = re.match(r"^(.+?)\s*\((.+?)\)\s*$", text)
    if paren_match:
        province, city = paren_match.group(1).strip(), paren_match.group(2).strip()
        return province or "نامشخص", city or province or "نامشخص"

    if "-" in text:
        province, city = [p.strip() for p in text.split("-", 1)]
        return province or "نامشخص", city or province or "نامشخص"

    return text, text


def _stable_id(name: str, city_province: str) -> str:
    """Deterministic id so re-running the ETL updates rather than duplicates."""
    return hashlib.sha1(f"{name}|{city_province}".encode("utf-8")).hexdigest()[:16]


def normalize_document(source: dict[str, Any]) -> dict[str, Any]:
    """Map a raw ES `_source` document to normalized fields ready for Cypher."""
    doc: dict[str, Any] = {}
    for fa_key, en_key in FIELD_MAP.items():
        doc[en_key] = source.get(fa_key)

    doc["categories"] = doc.get("categories") or []
    doc["vehicles"] = _split_multi(doc.get("vehicles") or "", seps="-")
    doc["trip_types"] = _split_multi(doc.get("trip_types") or "", seps="و,")
    doc["best_seasons"] = _split_seasons(doc.get("best_seasons"))
    doc["nearby_landmarks"] = _split_multi(doc.get("nearby_landmarks") or "")
    doc["ecotourism"] = str(doc.get("ecotourism") or "").strip() == "دارد"

    doc["province"], doc["city"] = _parse_city_province(doc.get("city_province") or "")

    coords = doc.get("coordinates") or {}
    doc["lat"] = coords.get("lat")
    doc["lon"] = coords.get("lon")

    doc["user_rating_score"] = RATING_SCORE.get(doc.get("user_rating_text") or "", None)
    doc["facilities_score"] = LEVEL_SCORE.get(doc.get("facilities_level") or "", None)

    doc["id"] = _stable_id(doc.get("name") or "", doc.get("city_province") or "")
    return doc


INGEST_CYPHER = """
UNWIND $rows AS row

MERGE (d:Destination {id: row.id})
ON CREATE SET d.created_at = datetime()
SET d.name = row.name,
    d.description = row.description,
    d.sensitive_notes = row.sensitive_notes,
    d.exit_route_name = row.exit_route,
    d.physical_readiness = row.physical_readiness,
    d.facilities_level = row.facilities_level,
    d.facilities_score = row.facilities_score,
    d.user_rating_text = row.user_rating_text,
    d.user_rating_score = row.user_rating_score,
    d.stay_duration = row.stay_duration,
    d.distance_km = row.distance_km,
    d.distance_description = row.distance_description,
    d.travel_time_hours = row.travel_time_hours,
    d.road_type = row.road_type,
    d.ecotourism = row.ecotourism,
    d.updated_at = datetime()

WITH d, row
FOREACH (_ IN CASE WHEN row.lat IS NOT NULL AND row.lon IS NOT NULL THEN [1] ELSE [] END |
    SET d.location = point({latitude: row.lat, longitude: row.lon})
)

MERGE (prov:Province {name: row.province})
MERGE (city:City {name: row.city})
MERGE (city)-[:PART_OF]->(prov)
MERGE (d)-[:LOCATED_IN]->(city)

WITH d, row
UNWIND (CASE WHEN size(row.categories) = 0 THEN [null] ELSE row.categories END) AS cat
FOREACH (_ IN CASE WHEN cat IS NOT NULL THEN [1] ELSE [] END |
    MERGE (c:Category {name: cat})
    MERGE (d)-[:HAS_CATEGORY]->(c)
)

WITH d, row
UNWIND (CASE WHEN size(row.best_seasons) = 0 THEN [null] ELSE row.best_seasons END) AS season
FOREACH (_ IN CASE WHEN season IS NOT NULL THEN [1] ELSE [] END |
    MERGE (s:Season {name: season})
    MERGE (d)-[:BEST_SEASON]->(s)
)

WITH d, row
UNWIND (CASE WHEN size(row.trip_types) = 0 THEN [null] ELSE row.trip_types END) AS tt
FOREACH (_ IN CASE WHEN tt IS NOT NULL THEN [1] ELSE [] END |
    MERGE (t:TripType {name: tt})
    MERGE (d)-[:SUITABLE_FOR]->(t)
)

WITH d, row
UNWIND (CASE WHEN size(row.vehicles) = 0 THEN [null] ELSE row.vehicles END) AS veh
FOREACH (_ IN CASE WHEN veh IS NOT NULL THEN [1] ELSE [] END |
    MERGE (v:Vehicle {name: veh})
    MERGE (d)-[:ACCESSIBLE_BY]->(v)
)

WITH d, row
UNWIND (CASE WHEN size(row.nearby_landmarks) = 0 THEN [null] ELSE row.nearby_landmarks END) AS lm
FOREACH (_ IN CASE WHEN lm IS NOT NULL THEN [1] ELSE [] END |
    MERGE (l:Landmark {name: lm})
    MERGE (d)-[:NEAR]->(l)
)

WITH d, row
FOREACH (_ IN CASE WHEN row.exit_route IS NOT NULL AND row.exit_route <> '' THEN [1] ELSE [] END |
    MERGE (e:ExitRoute {name: row.exit_route})
    MERGE (d)-[:VIA_EXIT]->(e)
)

RETURN count(d) AS ingested
"""

# Second pass: connect destinations that are geographically close to each
# other, so the agent can suggest "combine X with nearby Y" itineraries.
LINK_NEARBY_CYPHER = """
MATCH (d1:Destination), (d2:Destination)
WHERE d1.id < d2.id
  AND d1.location IS NOT NULL AND d2.location IS NOT NULL
  AND point.distance(d1.location, d2.location) <= $radius_m
MERGE (d1)-[r:NEAR_DESTINATION]-(d2)
SET r.distance_km = round(point.distance(d1.location, d2.location) / 1000.0, 1)
"""


async def ingest_destinations(driver: AsyncDriver, docs: list[dict[str, Any]]) -> int:
    rows = [normalize_document(doc) for doc in docs]
    async with driver.session() as session:
        result = await session.run(INGEST_CYPHER, rows=rows)
        record = await result.single()
        await session.run(LINK_NEARBY_CYPHER, radius_m=30_000)
    logger.info("Ingested %s destinations", record["ingested"] if record else 0)
    return record["ingested"] if record else 0


def load_json_source(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Supports either raw `_source` docs or full ES hit objects.
    return [d["_source"] if "_source" in d else d for d in data]


def load_es_source(
    es_host: str,
    index: str,
    username: str | None = None,
    password: str | None = None,
    verify_certs: bool = False,
    scroll_size: int = 500,
) -> list[dict[str, Any]]:
    from elasticsearch import Elasticsearch, helpers

    es_kwargs: dict[str, Any] = {}
    if username and password:
        es_kwargs["basic_auth"] = (username, password)
    if es_host.startswith("https://"):
        # Local single-node clusters usually run on a self-signed cert;
        # set ELASTIC_VERIFY_CERTS=true once you have a real CA to trust.
        es_kwargs["verify_certs"] = verify_certs
        if not verify_certs:
            import warnings

            warnings.filterwarnings("ignore", message=".*Unverified HTTPS request.*")

    es = Elasticsearch(es_host, **es_kwargs)
    docs = []
    for hit in helpers.scan(es, index=index, size=scroll_size):
        docs.append(hit["_source"])
    return docs


async def main() -> None:
    parser = argparse.ArgumentParser(description="Load tourism data into Neo4j")
    parser.add_argument("--source", choices=["json", "es"], required=True)
    parser.add_argument("--path", help="Path to JSON export (for --source json)")
    parser.add_argument("--es-host", default=os.environ.get("ELASTIC_URL", "http://localhost:9200"))
    parser.add_argument("--es-username", default=os.environ.get("ELASTIC_USERNAME"))
    parser.add_argument("--es-password", default=os.environ.get("ELASTIC_PASSWORD"))
    parser.add_argument(
        "--es-verify-certs",
        action="store_true",
        default=os.environ.get("ELASTIC_VERIFY_CERTS", "false").lower() == "true",
    )
    parser.add_argument("--index", default=os.environ.get("INDEX_NAME", "tehran_destinations"))
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "neo4j"),
    )
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "password"))
    args = parser.parse_args()

    if args.source == "json":
        if not args.path:
            raise SystemExit("--path is required for --source json")
        docs = load_json_source(args.path)
    else:
        docs = load_es_source(
            args.es_host,
            args.index,
            username=args.es_username,
            password=args.es_password,
            verify_certs=args.es_verify_certs,
        )

    logger.info("Loaded %s raw documents from %s", len(docs), args.source)

    driver = AsyncGraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)
    )
    try:
        await ingest_destinations(driver, docs)
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
