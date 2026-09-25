"""Repository for the tourism knowledge graph.

Separate from the generic short-term/long-term/reasoning memory layers
(`src.memory.client`) because this graph has a domain-specific schema
(Destination, City, Category, Season, ...) built by `etl/load_destinations.py`.
Both share the same underlying Neo4j database, so the agent can freely join
across a Destination and, say, a User's preferences or past Trips.

IMPORTANT: every `:User` node here is keyed by `identifier`, not `id`.
`neo4j-agent-memory` ships a first-class `(:User {identifier, id, ...})`
concept (`client.users`, schema documented in
`neo4j_agent_memory.memory.users`) used by `HAS_CONVERSATION`,
`HAS_TRACE`, and `HAS_PREFERENCE`. This module used to MERGE its own
`:User {id: ...}` node instead -- same real person, different node,
because the property key didn't match. Keying on `identifier` here too
makes every Trip/Like/Dislike/Visit land on the exact same User node as
conversations, reasoning traces, and preferences.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


class TripGraphRepository:
    def __init__(self, driver: AsyncDriver, embedding_provider: Any | None = None):
        self._driver = driver
        self._embedding_provider = embedding_provider

    async def _read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            return [r.data() async for r in result]

    async def _write(self, query: str, **params: Any) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            return [r.data() async for r in result]

    @staticmethod
    def _destination_lookup_keys(value: str) -> list[str]:
        """Build Persian-normalized aliases for a destination name."""
        normalized = unicodedata.normalize("NFKC", value).translate(
            str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ـ": "", "\u200c": " "})
        )
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^\w\s]", " ", normalized.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return []

        keys = [normalized]
        core = re.sub(
            r"^(?:ییلاق|روستا|روستای|منطقه|شهرستان|شهر)\s+",
            "",
            normalized,
        ).strip()
        if core and core != normalized:
            keys.append(core)
        return keys

    # ------------------------------------------------------------------
    # Search & recommendation
    # ------------------------------------------------------------------

    async def search_destinations(
        self,
        categories: list[str] | None = None,
        seasons: list[str] | None = None,
        trip_types: list[str] | None = None,
        location: str | list[str] | None = None,
        max_distance_km: float | None = None,
        max_travel_hours: float | None = None,
        max_physical_readiness: str | None = None,  # "کم" | "متوسط" | "زیاد"
        stay_duration: str | None = None,
        ecotourism: bool | None = None,
        min_facilities_score: int | None = None,
        exclude_destination_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Filter destinations by the constraints the user gave the agent.

        All filters are optional and combined with AND. Physical readiness
        is treated as a ceiling: e.g. "متوسط" also allows "کم" destinations.
        `location` accepts either a single city/province string or a list
        of candidates (e.g. an informal region like "شمال" expanded to
        ["گیلان", "مازندران", "گلستان"] by `src.agent.regions.expand_location`
        before this is called) -- a destination matches if it matches ANY
        candidate.

        Three-tier source priority:
          1. Curated destinations from `sample_documents.json`
             (`load_destinations.py`) are tried first, with every filter
             applied as given. This is always the preferred, most trustworthy
             tier when it has enough matches.
          2. If tier 1 doesn't fill `limit`, we add destinations that are
             graph-linked via `NEAR_DESTINATION` to a tier-1 result (computed
             by `LINK_NEARBY_CYPHER` / `etl/link_nearby.py` from real
             coordinates). These are "vouched for" by a curated destination's
             proximity, so they rank above unlinked OSM results even when
             they are themselves OSM nodes.
          3. Only if still short of `limit` do we fall back to the broader
             OSM-derived pool (`:OsmFeature:Destination:OsmAttraction`, from
             `load_osm_geojson.py`). That ETL never creates `BEST_SEASON` /
             `SUITABLE_FOR` relationships and never sets `stay_duration` /
             `ecotourism` -- those are structurally undefined for OSM nodes,
             so requiring them would always yield zero OSM rows. This pass
             therefore drops those four filters (they can't be honored by
             this data) but keeps category / distance / travel-time /
             physical-readiness / exclusion constraints. It also tries a
             geospatial radius around the resolved city/region centroid (see
             `_resolve_location_centroid`) when the plain string match on
             `location` finds nothing, so "near <city>" still works even if
             an OSM feature's own `city` property is missing or spelled
             differently than the City node the curated data uses.
        """
        location_candidates = self._normalize_location(location)
        readiness_order = {"کم": 1, "متوسط": 2, "زیاد": 3}
        max_readiness_rank = readiness_order.get(max_physical_readiness or "", None)

        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _add(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                rid = row.get("id")
                if rid is not None and rid in seen_ids:
                    continue
                if rid is not None:
                    seen_ids.add(rid)
                results.append(row)

        # Tier 1: curated
        primary = await self._search_destinations_query(
            categories=categories,
            seasons=seasons,
            trip_types=trip_types,
            location_candidates=location_candidates,
            max_distance_km=max_distance_km,
            max_travel_hours=max_travel_hours,
            max_readiness_rank=max_readiness_rank,
            stay_duration=stay_duration,
            ecotourism=ecotourism,
            min_facilities_score=min_facilities_score,
            exclude_destination_ids=exclude_destination_ids,
            limit=limit,
            osm_only=False,
        )
        _add(primary)
        if len(results) >= limit:
            return results[:limit]

        # Tier 2: destinations NEAR_DESTINATION-linked to a curated result
        # (may themselves be OSM nodes) -- only worth querying once we know
        # which curated destinations, if any, are in play for this location.
        anchor_ids = [r["id"] for r in primary if r.get("id")]
        if not anchor_ids and location_candidates:
            # No curated match at all yet -- still try to find curated
            # destinations in the requested area (ignoring the non-locational
            # filters) purely to use as NEAR_DESTINATION anchors.
            area_anchors = await self._search_destinations_query(
                categories=None,
                seasons=None,
                trip_types=None,
                location_candidates=location_candidates,
                max_distance_km=None,
                max_travel_hours=None,
                max_readiness_rank=None,
                stay_duration=None,
                ecotourism=None,
                min_facilities_score=None,
                exclude_destination_ids=exclude_destination_ids,
                limit=25,
                osm_only=False,
            )
            anchor_ids = [r["id"] for r in area_anchors if r.get("id")]

        if anchor_ids:
            near_linked = await self._search_near_linked(
                anchor_ids=anchor_ids,
                categories=categories,
                max_readiness_rank=max_readiness_rank,
                exclude_destination_ids=list(seen_ids | set(exclude_destination_ids or [])),
                limit=limit - len(results),
            )
            _add(near_linked)
            if len(results) >= limit:
                return results[:limit]

        # Tier 3: broad OSM fallback, dropping curated-only filters.
        osm_fallback = await self._search_destinations_query(
            categories=categories,
            seasons=None,
            trip_types=None,
            location_candidates=location_candidates,
            max_distance_km=max_distance_km,
            max_travel_hours=max_travel_hours,
            max_readiness_rank=max_readiness_rank,
            stay_duration=None,
            ecotourism=None,
            min_facilities_score=None,
            exclude_destination_ids=list(seen_ids | set(exclude_destination_ids or [])),
            limit=limit - len(results),
            osm_only=True,
        )
        _add(osm_fallback)
        if len(results) >= limit or not location_candidates:
            return results[:limit]

        # Tier 3b: the string match on `location` found nothing at all (e.g.
        # a city that exists as a City node but isn't attached to enough
        # destinations by name, or OSM rows with missing/differently-spelled
        # `city` text). Fall back to a geospatial radius around the
        # resolved City/Province centroid instead of a name match.
        centroid = await self._resolve_location_centroid(location_candidates)
        if centroid:
            geo_fallback = await self._search_by_radius(
                lat=centroid["lat"],
                lon=centroid["lon"],
                radius_km=centroid.get("radius_km", 80.0),
                categories=categories,
                max_readiness_rank=max_readiness_rank,
                exclude_destination_ids=list(seen_ids | set(exclude_destination_ids or [])),
                limit=limit - len(results),
            )
            _add(geo_fallback)

        return results[:limit]

    @staticmethod
    def _normalize_location(location: str | list[str] | None) -> list[str] | None:
        if location is None:
            return None
        if isinstance(location, str):
            candidates = [location]
        else:
            candidates = list(location)
        candidates = [c.strip() for c in candidates if c and c.strip()]
        return candidates or None

    async def _search_destinations_query(
        self,
        categories: list[str] | None,
        seasons: list[str] | None,
        trip_types: list[str] | None,
        location_candidates: list[str] | None,
        max_distance_km: float | None,
        max_travel_hours: float | None,
        max_readiness_rank: int | None,
        stay_duration: str | None,
        ecotourism: bool | None,
        min_facilities_score: int | None,
        exclude_destination_ids: list[str] | None,
        limit: int,
        osm_only: bool,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        query = """
        MATCH (d:Destination)
        WHERE d:OsmFeature = $osm_only
        OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(cat:Category)
        OPTIONAL MATCH (d)-[:BEST_SEASON]->(sea:Season)
        OPTIONAL MATCH (d)-[:SUITABLE_FOR]->(tt:TripType)
        OPTIONAL MATCH (d)-[:LOCATED_IN]->(city:City)
        OPTIONAL MATCH (city)-[:PART_OF]->(city_prov:Province)
        OPTIONAL MATCH (d)-[:LOCATED_IN]->(direct_prov:Province)
        WITH d, collect(DISTINCT cat.name) AS categories,
                collect(DISTINCT sea.name) AS seasons,
                collect(DISTINCT tt.name) AS trip_types,
                collect(DISTINCT city.name) AS cities,
                collect(DISTINCT city_prov.name) + collect(DISTINCT direct_prov.name) AS provinces
        WHERE ($categories IS NULL OR any(c IN $categories WHERE c IN categories))
          AND ($seasons IS NULL OR any(s IN $seasons WHERE s IN seasons))
          AND ($trip_types IS NULL OR any(t IN $trip_types WHERE t IN trip_types))
          AND ($location_candidates IS NULL OR any(loc IN $location_candidates WHERE
               any(place IN cities + provinces
                   WHERE toLower(toString(place)) CONTAINS toLower(loc)
                      OR toLower(loc) CONTAINS toLower(toString(place)))))
          AND ($max_distance_km IS NULL OR d.distance_km IS NULL OR d.distance_km <= $max_distance_km)
          AND ($max_travel_hours IS NULL OR d.travel_time_hours IS NULL OR d.travel_time_hours <= $max_travel_hours)
          AND ($stay_duration IS NULL OR d.stay_duration = $stay_duration)
          AND ($ecotourism IS NULL OR d.ecotourism = $ecotourism)
          AND ($min_facilities_score IS NULL OR d.facilities_score IS NULL OR d.facilities_score >= $min_facilities_score)
          AND ($exclude_ids IS NULL OR NOT d.id IN $exclude_ids)
          AND ($max_readiness_rank IS NULL OR d.physical_readiness IS NULL OR
               CASE d.physical_readiness WHEN 'کم' THEN 1 WHEN 'متوسط' THEN 2 WHEN 'زیاد' THEN 3 ELSE 2 END <= $max_readiness_rank)
        RETURN d.id AS id, d.name AS name, d.description AS description,
               head(cities) AS city, head(provinces) AS province,
               categories, seasons, trip_types,
               d.distance_km AS distance_km, d.travel_time_hours AS travel_time_hours,
               d.stay_duration AS stay_duration, d.physical_readiness AS physical_readiness,
               d.road_type AS road_type, d.ecotourism AS ecotourism,
               d.facilities_level AS facilities_level, d.user_rating_text AS user_rating_text,
               d.user_rating_score AS user_rating_score,
               d.location.latitude AS latitude, d.location.longitude AS longitude,
               $osm_only AS source_is_osm
        ORDER BY coalesce(d.user_rating_score, 0) DESC, coalesce(d.facilities_score, 0) DESC
        LIMIT $limit
        """
        return await self._read(
            query,
            categories=categories,
            seasons=seasons,
            trip_types=trip_types,
            location_candidates=location_candidates,
            max_distance_km=max_distance_km,
            max_travel_hours=max_travel_hours,
            stay_duration=stay_duration,
            ecotourism=ecotourism,
            min_facilities_score=min_facilities_score,
            exclude_ids=exclude_destination_ids,
            max_readiness_rank=max_readiness_rank,
            limit=limit,
            osm_only=osm_only,
        )

    async def _search_near_linked(
        self,
        anchor_ids: list[str],
        categories: list[str] | None,
        max_readiness_rank: int | None,
        exclude_destination_ids: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Tier 2: destinations connected via NEAR_DESTINATION to one of
        `anchor_ids` (curated destinations already known to be in the right
        area) -- these may be OSM nodes, but their proximity to a trusted
        curated destination is itself the signal that they're relevant."""
        if not anchor_ids or limit <= 0:
            return []
        query = """
        UNWIND $anchor_ids AS aid
        MATCH (anchor:Destination {id: aid})-[r:NEAR_DESTINATION]-(other:Destination)
        WHERE ($exclude_ids IS NULL OR NOT other.id IN $exclude_ids)
        OPTIONAL MATCH (other)-[:HAS_CATEGORY]->(cat:Category)
        OPTIONAL MATCH (other)-[:LOCATED_IN]->(city:City)
        OPTIONAL MATCH (city)-[:PART_OF]->(city_prov:Province)
        OPTIONAL MATCH (other)-[:LOCATED_IN]->(direct_prov:Province)
        WITH other, r, anchor, collect(DISTINCT cat.name) AS categories,
             head(collect(DISTINCT city.name)) AS city,
             head(collect(DISTINCT city_prov.name) + collect(DISTINCT direct_prov.name)) AS province
        WHERE ($categories IS NULL OR any(c IN $categories WHERE c IN categories))
          AND ($max_readiness_rank IS NULL OR other.physical_readiness IS NULL OR
               CASE other.physical_readiness WHEN 'کم' THEN 1 WHEN 'متوسط' THEN 2 WHEN 'زیاد' THEN 3 ELSE 2 END <= $max_readiness_rank)
        RETURN DISTINCT other.id AS id, other.name AS name, other.description AS description,
               city, province, categories,
               [] AS seasons, [] AS trip_types,
               other.distance_km AS distance_km, other.travel_time_hours AS travel_time_hours,
               other.stay_duration AS stay_duration, other.physical_readiness AS physical_readiness,
               other.road_type AS road_type, other.ecotourism AS ecotourism,
               other.facilities_level AS facilities_level, other.user_rating_text AS user_rating_text,
               other.user_rating_score AS user_rating_score,
               other.location.latitude AS latitude, other.location.longitude AS longitude,
               anchor.name AS near_anchor, r.distance_km AS distance_to_anchor_km,
               other:OsmFeature AS source_is_osm
        ORDER BY r.distance_km ASC
        LIMIT $limit
        """
        return await self._read(
            query,
            anchor_ids=anchor_ids,
            categories=categories,
            max_readiness_rank=max_readiness_rank,
            exclude_ids=exclude_destination_ids,
            limit=limit,
        )

    async def _resolve_location_centroid(
        self, location_candidates: list[str]
    ) -> dict[str, Any] | None:
        """Resolve a city/province/region candidate to a lat/lon centroid,
        computed from the real destinations attached to that City/Province
        (see `etl/link_nearby.py::set_city_centroids`, which populates
        `City.location`). Used as a geospatial fallback when a plain name
        match on `location` finds nothing (e.g. spelling differences between
        the curated data's `city` field and an OSM feature's `addr:city`)."""
        query = """
        UNWIND $candidates AS cand
        OPTIONAL MATCH (c:City) WHERE c.location IS NOT NULL AND
            (toLower(c.name) CONTAINS toLower(cand) OR toLower(cand) CONTAINS toLower(c.name))
        WITH cand, collect(c)[0] AS city
        OPTIONAL MATCH (p:Province) WHERE city IS NULL AND p.location IS NOT NULL AND
            (toLower(p.name) CONTAINS toLower(cand) OR toLower(cand) CONTAINS toLower(p.name))
        WITH cand, city, collect(p)[0] AS province
        WITH coalesce(city, province) AS area
        WHERE area IS NOT NULL
        RETURN area.location.latitude AS lat, area.location.longitude AS lon,
               coalesce(area.radius_km, 60.0) AS radius_km
        LIMIT 1
        """
        rows = await self._read(query, candidates=location_candidates)
        return rows[0] if rows and rows[0].get("lat") is not None else None

    async def _search_by_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        categories: list[str] | None,
        max_readiness_rank: int | None,
        exclude_destination_ids: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Geospatial fallback: any Destination (curated or OSM) within
        `radius_km` of a resolved city/region centroid, regardless of what
        its own `city`/`province` text properties say."""
        if limit <= 0:
            return []
        query = """
        MATCH (d:Destination)
        WHERE d.location IS NOT NULL
          AND point.distance(d.location, point({latitude: $lat, longitude: $lon})) <= $radius_km * 1000.0
          AND ($exclude_ids IS NULL OR NOT d.id IN $exclude_ids)
        OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(cat:Category)
        OPTIONAL MATCH (d)-[:LOCATED_IN]->(city:City)
        OPTIONAL MATCH (city)-[:PART_OF]->(city_prov:Province)
        OPTIONAL MATCH (d)-[:LOCATED_IN]->(direct_prov:Province)
        WITH d, collect(DISTINCT cat.name) AS categories,
             head(collect(DISTINCT city.name)) AS city,
             head(collect(DISTINCT city_prov.name) + collect(DISTINCT direct_prov.name)) AS province,
             point.distance(d.location, point({latitude: $lat, longitude: $lon})) / 1000.0 AS distance_from_center_km
        WHERE ($categories IS NULL OR any(c IN $categories WHERE c IN categories))
          AND ($max_readiness_rank IS NULL OR d.physical_readiness IS NULL OR
               CASE d.physical_readiness WHEN 'کم' THEN 1 WHEN 'متوسط' THEN 2 WHEN 'زیاد' THEN 3 ELSE 2 END <= $max_readiness_rank)
        RETURN d.id AS id, d.name AS name, d.description AS description,
               city, province, categories, [] AS seasons, [] AS trip_types,
               d.distance_km AS distance_km, d.travel_time_hours AS travel_time_hours,
               d.stay_duration AS stay_duration, d.physical_readiness AS physical_readiness,
               d.road_type AS road_type, d.ecotourism AS ecotourism,
               d.facilities_level AS facilities_level, d.user_rating_text AS user_rating_text,
               d.user_rating_score AS user_rating_score,
               d.location.latitude AS latitude, d.location.longitude AS longitude,
               round(distance_from_center_km, 1) AS distance_from_center_km,
               d:OsmFeature AS source_is_osm
        ORDER BY distance_from_center_km ASC
        LIMIT $limit
        """
        return await self._read(
            query,
            lat=lat,
            lon=lon,
            radius_km=radius_km,
            categories=categories,
            max_readiness_rank=max_readiness_rank,
            exclude_ids=exclude_destination_ids,
            limit=limit,
        )

    async def get_destination_details(self, name_or_id: str) -> dict[str, Any] | None:
        lookup_keys = self._destination_lookup_keys(name_or_id)
        if not lookup_keys:
            return None
        query = """
        MATCH (d:Destination)
        WITH d, [raw IN [d.name, d.name_fa, d.alt_name_fa, d.official_name_fa,
                         d.old_name_fa] WHERE raw IS NOT NULL |
            replace(replace(replace(replace(toLower(trim(raw)), 'ي', 'ی'), 'ى', 'ی'), 'ك', 'ک'), '‌', ' ')
        ] AS normalized_names
        WHERE d.id = $key OR any(name IN normalized_names WHERE
            any(lookup IN $lookup_keys WHERE name = lookup OR name CONTAINS lookup OR lookup CONTAINS name))
        OPTIONAL MATCH (d)-[:LOCATED_IN]->(city:City)
        OPTIONAL MATCH (city)-[:PART_OF]->(city_prov:Province)
        OPTIONAL MATCH (d)-[:LOCATED_IN]->(direct_prov:Province)
        OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(cat:Category)
        OPTIONAL MATCH (d)-[:BEST_SEASON]->(sea:Season)
        OPTIONAL MATCH (d)-[:NEAR]->(lm:Landmark)
        OPTIONAL MATCH (d)-[:VIA_EXIT]->(ex:ExitRoute)
        OPTIONAL MATCH (d)-[:ACCESSIBLE_BY]->(vehicle:Vehicle)
        CALL (d) {
            OPTIONAL MATCH (other:Destination)
            WHERE d.location IS NOT NULL AND other.location IS NOT NULL AND other <> d
              AND point.distance(d.location, other.location) <= 30000
            WITH other, point.distance(d.location, other.location) / 1000.0 AS distance_km,
                 CASE WHEN EXISTS { MATCH (d)-[:NEAR_DESTINATION]-(other) } THEN 0 ELSE 1 END AS near_priority
            ORDER BY near_priority, distance_km
            RETURN [item IN collect(DISTINCT {name: other.name, distance_km: round(distance_km, 1)})
                    WHERE item.name IS NOT NULL][0..5] AS nearby
        }
        WITH d, normalized_names, city, city_prov, direct_prov,
             collect(DISTINCT cat.name) AS categories,
             collect(DISTINCT sea.name) AS seasons,
             collect(DISTINCT lm.name) AS landmarks,
             collect(DISTINCT ex.name) AS exit_routes,
             collect(DISTINCT vehicle.name) AS vehicles, nearby
        RETURN d {.*, city: city.name, province: coalesce(city_prov.name, direct_prov.name), categories: categories,
                   seasons: seasons, landmarks: landmarks, exit_routes: exit_routes,
                   vehicles: vehicles,
                   latitude: d.location.latitude, longitude: d.location.longitude,
                   nearby_destinations: nearby} AS destination
        ORDER BY CASE WHEN d.id = $key THEN 0 ELSE 1 END,
                 CASE WHEN d:OsmFeature THEN 1 ELSE 0 END,
                 CASE WHEN any(name IN normalized_names WHERE name = $normalized_key) THEN 0 ELSE 1 END,
                 size(coalesce(d.description, '')) DESC
        LIMIT 1
        """
        rows = await self._read(
            query,
            key=name_or_id,
            normalized_key=lookup_keys[0],
            lookup_keys=lookup_keys,
        )
        return rows[0]["destination"] if rows else None

    async def find_destinations_near(self, name_or_id: str, radius_km: float = 50) -> list[dict[str, Any]]:
        lookup_keys = self._destination_lookup_keys(name_or_id)
        if not lookup_keys:
            return []
        query = """
        MATCH (d:Destination)
        WITH d, [raw IN [d.name, d.name_fa, d.alt_name_fa, d.official_name_fa,
                         d.old_name_fa] WHERE raw IS NOT NULL |
            replace(replace(replace(replace(toLower(trim(raw)), 'ي', 'ی'), 'ى', 'ی'), 'ك', 'ک'), '‌', ' ')
        ] AS normalized_names
        WHERE d.id = $key OR any(name IN normalized_names WHERE
            any(lookup IN $lookup_keys WHERE name = lookup OR name CONTAINS lookup OR lookup CONTAINS name))
        WITH d,
             CASE WHEN d.id = $key THEN 0 ELSE 1 END AS id_priority,
             CASE WHEN d:OsmFeature THEN 1 ELSE 0 END AS source_priority,
             CASE WHEN any(name IN normalized_names WHERE name = $normalized_key) THEN 0 ELSE 1 END AS name_priority
        ORDER BY id_priority, source_priority, name_priority, size(coalesce(d.description, '')) DESC
        LIMIT 1
        MATCH (other:Destination)
        WHERE d.location IS NOT NULL AND other.location IS NOT NULL AND other <> d
          AND point.distance(d.location, other.location) <= $radius_km * 1000.0
        WITH d, other, point.distance(d.location, other.location) / 1000.0 AS distance_km,
             CASE WHEN EXISTS { MATCH (d)-[:NEAR_DESTINATION]-(other) } THEN 0 ELSE 1 END AS near_priority
        RETURN other.name AS name, other.id AS id, round(distance_km, 1) AS distance_km,
               other.description AS description,
               other.location.latitude AS latitude, other.location.longitude AS longitude
        ORDER BY near_priority, distance_km ASC
        LIMIT 50
        """
        return await self._read(
            query,
            key=name_or_id,
            normalized_key=lookup_keys[0],
            lookup_keys=lookup_keys,
            radius_km=radius_km,
        )

    async def get_coordinates_by_ids(self, destination_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch-resolve `{id: {name, latitude, longitude}}` for a list of
        destination ids. Used to build the map-ready `itinerary` payload in
        the chat response without a per-destination round trip."""
        if not destination_ids:
            return {}
        rows = await self._read(
            """
            MATCH (d:Destination) WHERE d.id IN $ids
            RETURN d.id AS id, d.name AS name,
                   d.location.latitude AS latitude, d.location.longitude AS longitude
            """,
            ids=destination_ids,
        )
        return {r["id"]: r for r in rows}

    async def semantic_search_destinations(
        self,
        goal: str,
        limit: int = 8,
        threshold: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Find destinations whose embedded meaning best matches a travel goal."""
        if not goal.strip() or self._embedding_provider is None or limit <= 0:
            return []
        try:
            embedding = await self._embedding_provider.embed_one(goal.strip())
        except Exception:
            logger.warning("Could not embed semantic destination query", exc_info=True)
            return []

        query = """
        CALL db.index.vector.queryNodes('destination_embedding', $candidate_limit, $embedding)
        YIELD node AS d, score
        WHERE score >= $threshold
        OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(cat:Category)
        OPTIONAL MATCH (d)-[:BEST_SEASON]->(sea:Season)
        OPTIONAL MATCH (d)-[:SUITABLE_FOR]->(tt:TripType)
        OPTIONAL MATCH (d)-[:LOCATED_IN]->(city:City)
        OPTIONAL MATCH (city)-[:PART_OF]->(city_prov:Province)
        OPTIONAL MATCH (d)-[:LOCATED_IN]->(direct_prov:Province)
        WITH d, score, collect(DISTINCT cat.name) AS categories,
             collect(DISTINCT sea.name) AS seasons,
             collect(DISTINCT tt.name) AS trip_types,
             head(collect(DISTINCT city.name)) AS city,
             head(collect(DISTINCT city_prov.name) + collect(DISTINCT direct_prov.name)) AS province
        RETURN d.id AS id, d.name AS name, d.description AS description,
               city, province, categories, seasons, trip_types,
               d.distance_km AS distance_km, d.travel_time_hours AS travel_time_hours,
               d.physical_readiness AS physical_readiness, d.road_type AS road_type,
               d.facilities_level AS facilities_level, d:OsmFeature AS source_is_osm,
               d.location.latitude AS latitude, d.location.longitude AS longitude,
               score AS semantic_score
        ORDER BY score DESC, CASE WHEN d:OsmFeature THEN 1 ELSE 0 END,
                 coalesce(d.user_rating_score, 0) DESC
        LIMIT $limit
        """
        try:
            return await self._read(
                query,
                embedding=embedding,
                threshold=threshold,
                candidate_limit=max(limit * 5, 25),
                limit=limit,
            )
        except Exception:
            # Structured and exact-name search remain available during a partial
            # backfill or if the vector index is temporarily unavailable.
            logger.warning("Semantic destination search unavailable", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # User / trip memory (this is what makes the graph "learn")
    # ------------------------------------------------------------------

    async def get_travel_goal(self, session_id: str, user_id: str) -> dict[str, Any] | None:
        rows = await self._read(
            """
            MATCH (u:User {identifier: $user_id})-[:HAS_GOAL]->(g:TravelGoal {session_id: $session_id})
            RETURN g.state_json AS state_json, g.revision AS revision, g.updated_at AS updated_at
            LIMIT 1
            """,
            session_id=session_id,
            user_id=user_id,
        )
        if not rows or not rows[0].get("state_json"):
            return None
        try:
            state = json.loads(rows[0]["state_json"])
        except (TypeError, json.JSONDecodeError):
            logger.warning("Ignoring invalid TravelGoal JSON for session %s", session_id)
            return None
        state["revision"] = rows[0].get("revision") or state.get("revision", 0)
        state["updated_at"] = str(rows[0].get("updated_at") or state.get("updated_at") or "")
        return state

    async def upsert_travel_goal(
        self,
        session_id: str,
        user_id: str,
        state: dict[str, Any],
    ) -> None:
        await self._write(
            """
            MERGE (u:User {identifier: $user_id})
            ON CREATE SET u.id = randomUUID(), u.created_at = datetime()
            MERGE (g:TravelGoal {session_id: $session_id})
            ON CREATE SET g.created_at = datetime()
            SET g.user_identifier = $user_id,
                g.state_json = $state_json,
                g.revision = $revision,
                g.updated_at = datetime()
            MERGE (u)-[:HAS_GOAL]->(g)
            """,
            session_id=session_id,
            user_id=user_id,
            state_json=json.dumps(state, ensure_ascii=False),
            revision=state.get("revision", 0),
        )

    async def upsert_user(self, user_id: str) -> None:
        await self._write(
            """
            MERGE (u:User {identifier: $user_id})
            ON CREATE SET u.id = randomUUID(), u.created_at = datetime()
            """,
            user_id=user_id,
        )

    async def record_trip(
        self,
        user_id: str,
        trip_id: str,
        destination_ids: list[str],
        trip_context: dict[str, Any],
    ) -> None:
        """Persist a plan the agent produced, linked to the user and the
        destinations involved. Called after the agent proposes an itinerary
        so future recommendations can avoid repeats and learn what worked."""
        query = """
        MERGE (u:User {identifier: $user_id})
        ON CREATE SET u.id = randomUUID(), u.created_at = datetime()
        MERGE (t:Trip {id: $trip_id})
        SET t += $trip_context, t.created_at = coalesce(t.created_at, datetime())
        MERGE (u)-[:PLANNED]->(t)
        WITH t
        UNWIND $destination_ids AS did
        MATCH (d:Destination {id: did})
        MERGE (t)-[:INCLUDES]->(d)
        """
        await self._write(
            query,
            user_id=user_id,
            trip_id=trip_id,
            trip_context=trip_context,
            destination_ids=destination_ids,
        )

    async def record_feedback(
        self,
        user_id: str,
        destination_id: str,
        sentiment: str,  # "liked" | "disliked" | "visited"
        note: str | None = None,
    ) -> None:
        """Store explicit or inferred feedback so the graph gets smarter
        about this user's taste over time."""
        rel = {"liked": "LIKED", "disliked": "DISLIKED", "visited": "VISITED"}.get(sentiment, "VISITED")
        query = f"""
        MERGE (u:User {{identifier: $user_id}})
        ON CREATE SET u.id = randomUUID(), u.created_at = datetime()
        MATCH (d:Destination {{id: $destination_id}})
        MERGE (u)-[r:{rel}]->(d)
        SET r.note = $note, r.updated_at = datetime()
        """
        await self._write(query, user_id=user_id, destination_id=destination_id, note=note)

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 20,
        exclude_session_id: str | None = None,
    ) -> dict[str, Any]:
        # Each relationship type gets its own CALL {} subquery, for two
        # reasons:
        #  1. Chaining multiple OPTIONAL MATCHes off the same `u` in one
        #     query multiplies rows (cartesian product) before the top-level
        #     collect() ever runs, so e.g. 2 liked x 3 visited would silently
        #     inflate/duplicate counts.
        #  2. Neo4j doesn't allow an aggregate function (collect) nested
        #     directly inside another aggregate function -- grouping trip ->
        #     destinations has to happen in its own WITH before the outer
        #     collect(DISTINCT {...}) wraps it.
        query = """
        MATCH (u:User {identifier: $user_id})
        CALL (u) {
            OPTIONAL MATCH (u)-[:LIKED]->(liked:Destination)
            RETURN collect(DISTINCT liked.name)[0..$limit] AS liked
        }
        CALL (u) {
            OPTIONAL MATCH (u)-[:DISLIKED]->(disliked:Destination)
            RETURN collect(DISTINCT disliked.name)[0..$limit] AS disliked
        }
        CALL (u) {
            OPTIONAL MATCH (u)-[:VISITED]->(visited:Destination)
            RETURN collect(DISTINCT visited.name)[0..$limit] AS visited
        }
        CALL (u) {
            OPTIONAL MATCH (u)-[:PLANNED]->(trip:Trip)-[:INCLUDES]->(planned:Destination)
            WITH trip, collect(DISTINCT planned.name) AS destinations
            WHERE trip IS NOT NULL
            RETURN collect(DISTINCT {trip_id: trip.id, summary: trip.summary,
                    trip_type: trip.trip_type, companions: trip.companions,
                    planned_at: trip.planned_at, destinations: destinations})[0..$limit] AS trips
        }
        CALL (u) {
            OPTIONAL MATCH (u)-[:HAS_CONVERSATION]->(conversation:Conversation)-[:HAS_MESSAGE]->(message:Message)
            WHERE message.role = 'user'
              AND ($exclude_session_id IS NULL OR conversation.session_id <> $exclude_session_id)
            WITH message ORDER BY message.timestamp DESC
            RETURN collect(message.content)[0..5] AS recent_topics
        }
        CALL (u) {
            OPTIONAL MATCH (u)-[:HAS_GOAL]->(goal:TravelGoal)
            WHERE $exclude_session_id IS NULL OR goal.session_id <> $exclude_session_id
            WITH goal ORDER BY goal.updated_at DESC
            RETURN collect(goal.state_json)[0..3] AS recent_goals
        }
        RETURN liked, disliked, visited, trips, recent_topics, recent_goals
        """
        rows = await self._read(
            query,
            user_id=user_id,
            limit=limit,
            exclude_session_id=exclude_session_id,
        )
        return rows[0] if rows else {
            "liked": [], "disliked": [], "visited": [], "trips": [],
            "recent_topics": [], "recent_goals": []
        }

    async def list_categories_seasons(self) -> dict[str, list[str]]:
        query = """
        CALL {
            MATCH (c:Category) RETURN collect(DISTINCT c.name) AS categories
        }
        CALL {
            MATCH (s:Season) RETURN collect(DISTINCT s.name) AS seasons
        }
        CALL {
            MATCH (t:TripType) RETURN collect(DISTINCT t.name) AS trip_types
        }
        RETURN categories, seasons, trip_types
        """
        rows = await self._read(query)
        return rows[0] if rows else {"categories": [], "seasons": [], "trip_types": []}
