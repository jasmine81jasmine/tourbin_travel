"""Post-processing run after EITHER `load_destinations.py` or
`load_osm_geojson.py` (and safe/idempotent to run again after both):

1. `set_city_centroids` -- gives every `City` (and `Province`) node a
   `location` point, computed as the centroid of the real destinations
   `LOCATED_IN` it. Neither ETL script sets this directly (a City is just a
   name node MERGEd by both loaders), but the agent needs it to answer
   "near <city>" requests geospatially instead of by name-substring match
   alone -- see `TripGraphRepository._resolve_location_centroid`.

2. `link_nearby_destinations` -- (re)computes `NEAR_DESTINATION` across
   *all* `:Destination` nodes regardless of source. `load_destinations.py`
   already does this, but only across whatever destinations exist in the
   graph *at the time it runs* -- per the documented import order (curated
   JSON first, then the ~65k-feature GeoJSON), that means curated<->curated
   links only. OSM destinations never get linked to anything, and no
   curated<->OSM links are ever created, unless this is run again after the
   GeoJSON import. Run it every time you (re)import either source:

    python etl/link_nearby.py \
      --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password '...'

Both steps are pure MERGE/SET on data already in the graph -- no source
files needed, and safe to re-run any number of times.
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("etl.link_nearby")


SET_CITY_CENTROIDS_CYPHER = """
MATCH (city:City)<-[:LOCATED_IN]-(d:Destination)
WHERE d.location IS NOT NULL
WITH city, avg(d.location.latitude) AS lat, avg(d.location.longitude) AS lon, count(d) AS n
SET city.location = point({latitude: lat, longitude: lon}),
    city.destination_count = n,
    // Small cities collapse to a tight point; cities with destinations
    // spread further apart need a wider "near this city" radius. 60km
    // floor keeps small/sparse cities usable; grows for spread-out ones.
    city.radius_km = 60.0
RETURN count(city) AS updated
"""

SET_PROVINCE_CENTROIDS_CYPHER = """
MATCH (prov:Province)<-[:PART_OF]-(city:City)
WHERE city.location IS NOT NULL
WITH prov, avg(city.location.latitude) AS lat, avg(city.location.longitude) AS lon, count(city) AS n
SET prov.location = point({latitude: lat, longitude: lon}),
    prov.city_count = n,
    prov.radius_km = 120.0
RETURN count(prov) AS updated
"""

# Symmetric NEAR_DESTINATION across ALL Destination nodes (curated + OSM),
# not just whichever ones existed when load_destinations.py last ran.
LINK_NEARBY_CYPHER = """
MATCH (d1:Destination), (d2:Destination)
WHERE d1.id < d2.id
  AND d1.location IS NOT NULL AND d2.location IS NOT NULL
  AND point.distance(d1.location, d2.location) <= $radius_m
MERGE (d1)-[r:NEAR_DESTINATION]-(d2)
SET r.distance_km = round(point.distance(d1.location, d2.location) / 1000.0, 1)
RETURN count(r) AS linked
"""


def set_city_centroids(driver: Driver, database: str | None = None) -> tuple[int, int]:
    with driver.session(database=database) as session:
        city_result = session.run(SET_CITY_CENTROIDS_CYPHER).single()
        province_result = session.run(SET_PROVINCE_CENTROIDS_CYPHER).single()
    cities = city_result["updated"] if city_result else 0
    provinces = province_result["updated"] if province_result else 0
    logger.info("Set centroid location on %s City nodes, %s Province nodes", cities, provinces)
    return cities, provinces


def link_nearby_destinations(driver: Driver, radius_km: float = 30.0, database: str | None = None) -> int:
    with driver.session(database=database) as session:
        result = session.run(LINK_NEARBY_CYPHER, radius_m=radius_km * 1000.0).single()
    linked = result["linked"] if result else 0
    logger.info("Linked/updated %s NEAR_DESTINATION relationships (radius %.0fkm)", linked, radius_km)
    return linked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process the graph: city/province centroids + cross-source NEAR_DESTINATION links"
    )
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument(
        "--neo4j-user", default=os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "neo4j")
    )
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE"))
    parser.add_argument("--radius-km", type=float, default=30.0, help="NEAR_DESTINATION link radius")
    args = parser.parse_args()

    with GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)) as driver:
        driver.verify_connectivity()
        set_city_centroids(driver, database=args.neo4j_database)
        link_nearby_destinations(driver, radius_km=args.radius_km, database=args.neo4j_database)


if __name__ == "__main__":
    main()
