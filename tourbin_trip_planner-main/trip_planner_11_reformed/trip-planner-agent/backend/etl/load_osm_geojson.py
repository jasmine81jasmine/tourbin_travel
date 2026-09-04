"""Stream an Overpass GeoJSON export into the trip-planner Neo4j graph.

Every feature is loaded as ``:OsmFeature``. Features with a Persian-script
name additionally become ``:Destination:OsmAttraction`` for the trip planner.
Free text is retained only when Persian; language-neutral OSM values remain
available as structured data on both kinds of node.

Run from ``backend``:

    python etl/load_osm_geojson.py --path ../export.geojson --batch-size 500
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from geojson_utils import iter_features, persian_value
from link_nearby import link_nearby_destinations, set_city_centroids


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("osm-geojson-etl")


# OSM values are language-neutral ontology codes. The Persian labels are what
# the agent sees as Category and OsmFeatureType names.
TYPE_LABELS_FA = {
    "natural": {
        "bay": "خلیج", "beach": "ساحل", "cape": "دماغه", "cave_entrance": "غار",
        "cliff": "صخره", "dune": "تپه شنی", "fell": "ارتفاعات", "geyser": "آبفشان",
        "glacier": "یخچال طبیعی", "grassland": "علفزار", "heath": "بوته‌زار",
        "hot_spring": "چشمه آب گرم", "island": "جزیره", "peak": "قله",
        "peninsula": "شبه‌جزیره", "ridge": "خط‌الرأس", "rock": "صخره",
        "saddle": "گردنه", "sand": "شنزار", "scree": "سنگ‌ریزه‌زار",
        "spring": "چشمه", "stone": "سنگ", "tree": "درخت", "tree_row": "ردیف درختان",
        "valley": "دره", "water": "پهنه آبی", "wetland": "تالاب", "wood": "جنگل",
    },
    "water": {
        "basin": "حوضچه", "canal": "کانال", "lake": "دریاچه", "lagoon": "مرداب",
        "pond": "برکه", "reservoir": "مخزن سد", "river": "رودخانه",
    },
    "wetland": {
        "bog": "باتلاق", "marsh": "مرداب", "reedbed": "نی‌زار", "saltmarsh": "شورمرداب",
        "swamp": "باتلاق جنگلی", "wet_meadow": "چمنزار مرطوب",
    },
    "tourism": {
        "alpine_hut": "پناهگاه کوهستانی", "attraction": "جاذبه گردشگری",
        "camp_site": "کمپینگ", "caravan_site": "محل کاروان", "museum": "موزه",
        "picnic_site": "تفرجگاه", "viewpoint": "چشم‌انداز",
    },
    "boundary": {"national_park": "پارک ملی", "protected_area": "منطقه حفاظت‌شده"},
    "leisure": {"nature_reserve": "ذخیره‌گاه طبیعی"},
    "geological": {"outcrop": "رخنمون زمین‌شناسی", "palaeontological_site": "محوطه دیرینه‌شناسی"},
    "historic": {
        "archaeological_site": "محوطه باستانی", "castle": "قلعه", "caravanserai": "کاروانسرا",
        "monument": "بنای یادبود", "ruins": "ویرانه تاریخی", "tree": "درخت تاریخی",
    },
    "place": {"island": "جزیره", "islet": "جزیره کوچک", "sea": "دریا"},
}

ROOT_CATEGORIES = {
    "natural": "جاذبه طبیعی",
    "tourism": "جاذبه گردشگری",
    "boundary": "منطقه حفاظت‌شده",
    "leisure": "طبیعت‌گردی",
    "historic": "جاذبه تاریخی",
}

PERSIAN_FIELDS = {
    "alt_name_fa": ("alt_name:fa", "alt_name"),
    "official_name_fa": ("official_name:fa", "official_name"),
    "old_name_fa": ("old_name:fa", "old_name"),
    "short_name_fa": ("short_name:fa", "short_name"),
    "operator_fa": ("operator:fa", "operator"),
    "inscription_fa": ("inscription:fa", "inscription"),
    "address_fa": ("addr:full:fa", "addr:full"),
    "street_fa": ("addr:street:fa", "addr:street"),
}

IRAN_PROVINCES = {
    "آذربایجان شرقی", "آذربایجان غربی", "اردبیل", "اصفهان", "البرز", "ایلام",
    "بوشهر", "تهران", "چهارمحال و بختیاری", "خراسان جنوبی", "خراسان رضوی",
    "خراسان شمالی", "خوزستان", "زنجان", "سمنان", "سیستان و بلوچستان", "فارس",
    "قزوین", "قم", "کردستان", "کرمان", "کرمانشاه", "کهگیلویه و بویراحمد",
    "گلستان", "گیلان", "لرستان", "مازندران", "مرکزی", "هرمزگان", "همدان", "یزد",
}

SCHEMA_QUERIES = (
    "CREATE CONSTRAINT osm_feature_id IF NOT EXISTS FOR (f:OsmFeature) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT destination_id IF NOT EXISTS FOR (d:Destination) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT osm_feature_type_id IF NOT EXISTS FOR (t:OsmFeatureType) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT country_code IF NOT EXISTS FOR (c:Country) REQUIRE c.code IS UNIQUE",
    "CREATE CONSTRAINT province_name IF NOT EXISTS FOR (p:Province) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT city_name IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE",
    "CREATE POINT INDEX destination_location IF NOT EXISTS FOR (d:Destination) ON (d.location)",
    "CREATE FULLTEXT INDEX destination_fulltext IF NOT EXISTS FOR (d:Destination) ON EACH [d.name, d.description]",
)

MIGRATE_EXISTING_CYPHER = """
MATCH (d:OsmAttraction)
WHERE d.source = 'OpenStreetMap'
SET d:OsmFeature
"""

INGEST_CYPHER = """
UNWIND $rows AS row
MERGE (d:OsmFeature {id: row.id})
ON CREATE SET d.created_at = datetime()
SET d.name = row.name,
    d.name_fa = row.name,
    d.description = row.description_fa,
    d.description_fa = row.description_fa,
    d.source = 'OpenStreetMap',
    d.source_id = row.source_id,
    d.osm_type = row.osm_type,
    d.osm_numeric_id = row.osm_numeric_id,
    d.natural_type = row.natural_type,
    d.tourism_type = row.tourism_type,
    d.boundary_type = row.boundary_type,
    d.leisure_type = row.leisure_type,
    d.water_type = row.water_type,
    d.wetland_type = row.wetland_type,
    d.geological_type = row.geological_type,
    d.historic_type = row.historic_type,
    d.place_type = row.place_type,
    d.attraction_type = row.attraction_type,
    d.waterway_type = row.waterway_type,
    d.landuse_type = row.landuse_type,
    d.leaf_type = row.leaf_type,
    d.leaf_cycle = row.leaf_cycle,
    d.denotation = row.denotation,
    d.elevation_m = row.elevation_m,
    d.height_m = row.height_m,
    d.prominence_m = row.prominence_m,
    d.opening_hours = row.opening_hours,
    d.fee = row.fee,
    d.access = row.access,
    d.wheelchair = row.wheelchair,
    d.drinking_water = row.drinking_water,
    d.surface = row.surface,
    d.seasonal = row.seasonal,
    d.salt_water = row.salt_water,
    d.tidal = row.tidal,
    d.lit = row.lit,
    d.toilets = row.toilets,
    d.internet_access = row.internet_access,
    d.bicycle_access = row.bicycle_access,
    d.foot_access = row.foot_access,
    d.motor_vehicle_access = row.motor_vehicle_access,
    d.reservation = row.reservation,
    d.website = row.website,
    d.phone = row.phone,
    d.image = row.image,
    d.wikidata = row.wikidata,
    d.wikipedia_fa = row.wikipedia_fa,
    d.protect_class = row.protect_class,
    d.iucn_class = row.iucn_class,
    d.heritage = row.heritage,
    d.intermittent = row.intermittent,
    d.alt_name_fa = row.alt_name_fa,
    d.official_name_fa = row.official_name_fa,
    d.old_name_fa = row.old_name_fa,
    d.short_name_fa = row.short_name_fa,
    d.operator_fa = row.operator_fa,
    d.inscription_fa = row.inscription_fa,
    d.address_fa = row.address_fa,
    d.street_fa = row.street_fa,
    d.updated_at = datetime()
FOREACH (_ IN CASE WHEN row.is_destination THEN [1] ELSE [] END | SET d:Destination:OsmAttraction)
FOREACH (_ IN CASE WHEN NOT row.is_destination THEN [1] ELSE [] END | REMOVE d:Destination:OsmAttraction)
FOREACH (_ IN CASE WHEN row.natural_type IS NOT NULL THEN [1] ELSE [] END | SET d:NaturalFeature)
FOREACH (_ IN CASE WHEN row.natural_type IS NULL THEN [1] ELSE [] END | REMOVE d:NaturalFeature)
FOREACH (_ IN CASE WHEN row.is_destination AND row.natural_type IS NOT NULL THEN [1] ELSE [] END |
    SET d:NaturalAttraction
)
FOREACH (_ IN CASE WHEN NOT row.is_destination OR row.natural_type IS NULL THEN [1] ELSE [] END |
    REMOVE d:NaturalAttraction
)
FOREACH (_ IN CASE WHEN row.is_protected THEN [1] ELSE [] END | SET d:ProtectedArea)
FOREACH (_ IN CASE WHEN NOT row.is_protected THEN [1] ELSE [] END | REMOVE d:ProtectedArea)
FOREACH (_ IN CASE WHEN row.lon IS NOT NULL AND row.lat IS NOT NULL THEN [1] ELSE [] END |
    SET d.location = point({longitude: row.lon, latitude: row.lat})
)
FOREACH (_ IN CASE WHEN row.lon IS NULL OR row.lat IS NULL THEN [1] ELSE [] END | REMOVE d.location)
WITH d, row
OPTIONAL MATCH (d)-[old:HAS_CATEGORY|HAS_OSM_TYPE]->()
DELETE old
WITH DISTINCT d, row
OPTIONAL MATCH (d)-[old_location:LOCATED_IN]->(old_area)
WHERE old_area:City OR old_area:Province
DELETE old_location
WITH DISTINCT d, row
MERGE (country:Country {code: 'IR'})
ON CREATE SET country.name = 'ایران'
MERGE (d)-[:IN_COUNTRY]->(country)
WITH d, row, country
FOREACH (_ IN CASE WHEN row.province IS NOT NULL THEN [1] ELSE [] END |
    MERGE (province:Province {name: row.province})
    MERGE (province)-[:PART_OF]->(country)
)
FOREACH (_ IN CASE WHEN row.city IS NOT NULL THEN [1] ELSE [] END |
    MERGE (city:City {name: row.city})
    MERGE (d)-[:LOCATED_IN]->(city)
)
FOREACH (_ IN CASE WHEN row.city IS NULL AND row.province IS NOT NULL THEN [1] ELSE [] END |
    MERGE (province:Province {name: row.province})
    MERGE (d)-[:LOCATED_IN]->(province)
)
FOREACH (_ IN CASE WHEN row.city IS NOT NULL AND row.province IS NOT NULL THEN [1] ELSE [] END |
    MERGE (city:City {name: row.city})
    MERGE (province:Province {name: row.province})
    MERGE (city)-[:PART_OF]->(province)
)
WITH d, row
UNWIND row.categories AS category_name
MERGE (category:Category {name: category_name})
MERGE (d)-[:HAS_CATEGORY]->(category)
WITH DISTINCT d, row
UNWIND row.feature_types AS feature_type
MERGE (type:OsmFeatureType {id: feature_type.id})
SET type.key = feature_type.key, type.value = feature_type.value, type.name_fa = feature_type.name_fa
MERGE (d)-[:HAS_OSM_TYPE]->(type)
RETURN count(DISTINCT d) AS ingested,
       count(DISTINCT CASE WHEN row.is_destination THEN d END) AS destinations
"""


def _number(value: Any) -> float | None:
    try:
        return float(str(value).strip()) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _boolean(value: Any) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    return None


def _safe_code(properties: dict[str, Any], key: str) -> str | None:
    value = properties.get(key)
    return str(value).strip() if value not in (None, "") else None


def _normalize_province(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.removeprefix("استان ").removesuffix(" استان").strip()
    return normalized if normalized in IRAN_PROVINCES else None


def normalize_feature(feature: dict[str, Any]) -> dict[str, Any] | None:
    properties = feature.get("properties") or {}
    name = persian_value(properties, "name:fa", "name")
    source_id = str(properties.get("@id") or feature.get("id") or "").strip()
    if not source_id:
        return None

    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") if geometry.get("type") == "Point" else None
    lon = _number(coordinates[0]) if isinstance(coordinates, list) and len(coordinates) >= 2 else None
    lat = _number(coordinates[1]) if isinstance(coordinates, list) and len(coordinates) >= 2 else None

    explicit_province = persian_value(properties, "addr:province:fa", "addr:province", "is_in:province")
    city = persian_value(properties, "addr:city:fa", "addr:city", "is_in:city")
    generic_is_in = persian_value(properties, "is_in:fa", "is_in")
    province = _normalize_province(explicit_province) or explicit_province
    province = province or _normalize_province(generic_is_in)
    if city is None and generic_is_in and _normalize_province(generic_is_in) is None:
        city = generic_is_in

    feature_types = []
    categories = []
    for key in (
        "natural", "water", "wetland", "tourism", "boundary", "leisure",
        "geological", "historic", "place",
    ):
        value = _safe_code(properties, key)
        if not value:
            continue
        label_fa = TYPE_LABELS_FA.get(key, {}).get(value)
        feature_types.append({"id": f"{key}={value}", "key": key, "value": value, "name_fa": label_fa})
        root = ROOT_CATEGORIES.get(key)
        if root:
            categories.append(root)
        if label_fa:
            categories.append(label_fa)

    if not categories:
        categories.append("جاذبه گردشگری")

    osm_type, _, osm_numeric_id = source_id.partition("/")
    row: dict[str, Any] = {
        "id": f"osm:{source_id}",
        "source_id": source_id,
        "osm_type": osm_type or None,
        "osm_numeric_id": osm_numeric_id or None,
        "name": name,
        "is_destination": name is not None,
        "description_fa": persian_value(properties, "description:fa", "description"),
        "lon": lon,
        "lat": lat,
        "categories": list(dict.fromkeys(categories)),
        "feature_types": feature_types,
        "natural_type": _safe_code(properties, "natural"),
        "tourism_type": _safe_code(properties, "tourism"),
        "boundary_type": _safe_code(properties, "boundary"),
        "leisure_type": _safe_code(properties, "leisure"),
        "water_type": _safe_code(properties, "water"),
        "wetland_type": _safe_code(properties, "wetland"),
        "geological_type": _safe_code(properties, "geological"),
        "historic_type": _safe_code(properties, "historic"),
        "place_type": _safe_code(properties, "place"),
        "attraction_type": _safe_code(properties, "attraction"),
        "waterway_type": _safe_code(properties, "waterway"),
        "landuse_type": _safe_code(properties, "landuse"),
        "leaf_type": _safe_code(properties, "leaf_type"),
        "leaf_cycle": _safe_code(properties, "leaf_cycle"),
        "denotation": _safe_code(properties, "denotation"),
        "elevation_m": _number(properties.get("ele")),
        "height_m": _number(properties.get("height")),
        "prominence_m": _number(properties.get("prominence")),
        "opening_hours": _safe_code(properties, "opening_hours"),
        "fee": _boolean(properties.get("fee")),
        "access": _safe_code(properties, "access"),
        "wheelchair": _safe_code(properties, "wheelchair"),
        "drinking_water": _safe_code(properties, "drinking_water"),
        "surface": _safe_code(properties, "surface"),
        "seasonal": _safe_code(properties, "seasonal"),
        "salt_water": _boolean(properties.get("salt")),
        "tidal": _boolean(properties.get("tidal")),
        "lit": _boolean(properties.get("lit")),
        "toilets": _safe_code(properties, "toilets"),
        "internet_access": _safe_code(properties, "internet_access"),
        "bicycle_access": _safe_code(properties, "bicycle"),
        "foot_access": _safe_code(properties, "foot"),
        "motor_vehicle_access": _safe_code(properties, "motor_vehicle"),
        "reservation": _safe_code(properties, "reservation"),
        "website": _safe_code(properties, "website") or _safe_code(properties, "contact:website"),
        "phone": _safe_code(properties, "phone") or _safe_code(properties, "contact:phone"),
        "image": _safe_code(properties, "image"),
        "wikidata": _safe_code(properties, "wikidata"),
        "wikipedia_fa": (value if (value := _safe_code(properties, "wikipedia")) and value.startswith("fa:") else None),
        "protect_class": _safe_code(properties, "protect_class"),
        "iucn_class": _safe_code(properties, "ref:IUCN"),
        "heritage": _safe_code(properties, "heritage"),
        "intermittent": _boolean(properties.get("intermittent")),
        "province": province,
        "city": city,
        "is_protected": properties.get("boundary") in {"national_park", "protected_area"}
        or properties.get("leisure") == "nature_reserve",
    }
    for target, source_keys in PERSIAN_FIELDS.items():
        row[target] = persian_value(properties, *source_keys)
    return row


def batches(rows: Iterable[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Overpass GeoJSON into the trip-planner graph")
    parser.add_argument("--path", required=True, help="Path to the Overpass .geojson export")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    seen = skipped = ingested = destinations = 0

    def normalized_rows() -> Iterable[dict[str, Any]]:
        nonlocal seen, skipped
        for feature in iter_features(args.path):
            seen += 1
            row = normalize_feature(feature)
            if row is None:
                skipped += 1
            else:
                yield row
            if args.progress_every and seen % args.progress_every == 0:
                logger.info("Scanned %s features; valid OSM features %s", f"{seen:,}", f"{seen - skipped:,}")

    with GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=args.neo4j_database) as session:
            session.run(MIGRATE_EXISTING_CYPHER).consume()
            for query in SCHEMA_QUERIES:
                session.run(query).consume()
            for batch in batches(normalized_rows(), args.batch_size):
                record = session.run(INGEST_CYPHER, rows=batch).single()
                ingested += record["ingested"] if record else 0
                destinations += record["destinations"] if record else 0
                logger.info(
                    "Committed %s OSM features; %s trip-planner destinations",
                    f"{ingested:,}",
                    f"{destinations:,}",
                )

            # Runs once, after all batches: gives every City/Province a
            # geospatial centroid, and (re)links NEAR_DESTINATION across
            # *all* Destination nodes -- curated (loaded in the earlier,
            # mandatory step 8) as well as the OSM destinations just loaded
            # here. Without this, OSM destinations stay geographically
            # disconnected from the curated set and from each other, and
            # "near <city>" queries have no coordinates to search around for
            # any City that only OSM data (never the curated set) mentions.
            set_city_centroids(driver, database=args.neo4j_database)
            link_nearby_destinations(driver, radius_km=30.0, database=args.neo4j_database)

    logger.info(
        "Finished: scanned %s, imported/updated %s OSM features, %s destinations, skipped malformed %s",
        f"{seen:,}",
        f"{ingested:,}",
        f"{destinations:,}",
        f"{skipped:,}",
    )


if __name__ == "__main__":
    main()
