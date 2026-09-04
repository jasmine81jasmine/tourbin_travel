"""Backfill resumable semantic embeddings for Destination nodes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

from neo4j import AsyncGraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.memory.client import build_embedding_provider


DESTINATIONS_QUERY = """
MATCH (d:Destination)
WHERE NOT $curated_only OR NOT d:OsmFeature
OPTIONAL MATCH (d)-[:HAS_CATEGORY]->(cat:Category)
OPTIONAL MATCH (d)-[:BEST_SEASON]->(sea:Season)
OPTIONAL MATCH (d)-[:SUITABLE_FOR]->(tt:TripType)
OPTIONAL MATCH (d)-[:ACCESSIBLE_BY]->(vehicle:Vehicle)
OPTIONAL MATCH (d)-[:LOCATED_IN]->(city:City)
OPTIONAL MATCH (city)-[:PART_OF]->(province:Province)
OPTIONAL MATCH (d)-[:NEAR]->(landmark:Landmark)
RETURN d.id AS id, d.name AS name, d.description AS description,
       d.sensitive_notes AS sensitive_notes, d.physical_readiness AS physical_readiness,
       d.facilities_level AS facilities_level, d.road_type AS road_type,
       d.stay_duration AS stay_duration, d.ecotourism AS ecotourism,
       d.natural_type AS natural_type, d.tourism_type AS tourism_type,
       d.historic_type AS historic_type, d.access AS access,
       d.wheelchair AS wheelchair, d.fee AS fee, d.ele AS elevation,
       head(collect(DISTINCT city.name)) AS city,
       head(collect(DISTINCT province.name)) AS province,
       collect(DISTINCT cat.name) AS categories,
       collect(DISTINCT sea.name) AS seasons,
       collect(DISTINCT tt.name) AS trip_types,
       collect(DISTINCT vehicle.name) AS vehicles,
       collect(DISTINCT landmark.name)[0..20] AS landmarks,
       d.embedding_text_hash AS embedding_text_hash,
       d.embedding_model AS embedding_model,
       d.embedding IS NOT NULL AS has_embedding
ORDER BY d.id
"""

WRITE_QUERY = """
UNWIND $rows AS row
MATCH (d:Destination {id: row.id})
SET d.embedding = row.embedding,
    d.embedding_model = row.model,
    d.embedding_dimensions = row.dimensions,
    d.embedding_text_hash = row.text_hash,
    d.embedding_updated_at = datetime()
"""


def _semantic_text(row: dict[str, Any]) -> str:
    fields = [
        ("نام", row.get("name")),
        ("معرفی", (row.get("description") or "")[:5000]),
        ("استان", row.get("province")),
        ("شهر", row.get("city")),
        ("دسته‌ها", row.get("categories")),
        ("نوع سفر", row.get("trip_types")),
        ("فصل مناسب", row.get("seasons")),
        ("فعالیت و دیدنی نزدیک", row.get("landmarks")),
        ("وسیله دسترسی", row.get("vehicles")),
        ("آمادگی بدنی", row.get("physical_readiness")),
        ("نوع جاده", row.get("road_type")),
        ("امکانات", row.get("facilities_level")),
        ("مدت اقامت", row.get("stay_duration")),
        ("بوم گردی", "دارد" if row.get("ecotourism") else None),
        ("نوع طبیعی", row.get("natural_type")),
        ("نوع گردشگری", row.get("tourism_type")),
        ("نوع تاریخی", row.get("historic_type")),
        ("دسترسی", row.get("access")),
        ("دسترسی ویلچر", row.get("wheelchair")),
        ("ورودی", row.get("fee")),
        ("ارتفاع", row.get("elevation")),
        ("نکات مهم", row.get("sensitive_notes")),
    ]
    lines = []
    for label, value in fields:
        if isinstance(value, list):
            value = "، ".join(str(item) for item in value if item)
        if value not in (None, "", []):
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


async def _embed_with_retry(provider, texts: list[str], attempts: int = 3):
    for attempt in range(attempts):
        try:
            return await provider.embed(texts)
        except Exception:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(2 ** attempt)


async def backfill(curated_only: bool, batch_size: int, limit: int | None) -> None:
    settings = get_settings()
    provider = build_embedding_provider()
    model = str(getattr(provider, "model", settings.embedding_model))
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
    )
    try:
        async with driver.session() as session:
            result = await session.run(DESTINATIONS_QUERY, curated_only=curated_only)
            destinations = [record.data() async for record in result]
        if limit is not None:
            destinations = destinations[:limit]

        pending = []
        skipped = 0
        for row in destinations:
            text = _semantic_text(row)
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if (
                row.get("has_embedding")
                and row.get("embedding_text_hash") == text_hash
                and row.get("embedding_model") == model
            ):
                skipped += 1
                continue
            pending.append((row["id"], text, text_hash))

        written = 0
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            vectors = await _embed_with_retry(provider, [item[1] for item in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding provider returned an unexpected vector count")
            rows = []
            for (destination_id, _, text_hash), vector in zip(batch, vectors, strict=True):
                if len(vector) != settings.embedding_dimensions or not all(math.isfinite(v) for v in vector):
                    raise RuntimeError(f"Invalid embedding for destination {destination_id}")
                rows.append({
                    "id": destination_id,
                    "embedding": vector,
                    "model": model,
                    "dimensions": settings.embedding_dimensions,
                    "text_hash": text_hash,
                })
            async with driver.session() as session:
                await session.run(WRITE_QUERY, rows=rows)
            written += len(rows)
            print(f"Embedded {written}/{len(pending)} destinations")

        print(f"Complete: embedded={written}, unchanged={skipped}, considered={len(destinations)}")
    finally:
        await driver.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(backfill(args.curated_only, args.batch_size, args.limit))


if __name__ == "__main__":
    main()
