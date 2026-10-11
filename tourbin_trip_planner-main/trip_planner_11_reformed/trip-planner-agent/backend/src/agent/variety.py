"""Prefer fresh, equally credible graph candidates without changing hard filters."""

import hashlib
from typing import Any


def vary_candidates(
    rows: list[dict[str, Any]], previous: dict[str, set[str]],
    user_id: str, session_id: str, limit: int, semantic: bool = False,
) -> list[dict[str, Any]]:
    if not rows or limit <= 0:
        return []

    def quality(row: dict[str, Any]) -> tuple[int, int, float, float]:
        # The graph returns curated, nearby-linked, then broad OSM. Never
        # promote a broad fallback over an available curated destination.
        tier = 2 if row.get("source_is_osm") and not row.get("near_anchor") else 1 if row.get("near_anchor") else 0
        score_band = int(float(row.get("semantic_score") or 0) * 20) if semantic else 0
        return ((-score_band, tier) if semantic else (tier, 0)) + (
            -float(row.get("user_rating_score") or 0), -float(row.get("facilities_score") or 0)
        )

    def tie_break(row: dict[str, Any]) -> bytes:
        return hashlib.sha256(
            f"{user_id}:{session_id}:{row.get('id') or row.get('name')}".encode("utf-8")
        ).digest()

    # Variation only breaks ties between otherwise similarly ranked places.
    ordered = sorted(rows, key=lambda row: (quality(row), tie_break(row)))
    best_tier = quality(ordered[0])[0]
    best_rating = float(ordered[0].get("user_rating_score") or 0)
    good = ([row for row in ordered if quality(row)[0] <= best_tier + 1]
            if semantic else [row for row in ordered if quality(row)[0] == best_tier
                              and float(row.get("user_rating_score") or 0) >= best_rating - 1])
    other = [row for row in ordered if row not in good]
    def unseen(row: dict[str, Any]) -> bool:
        return row.get("id") not in previous.get("ids", set()) and row.get("name") not in previous.get("names", set())

    fresh = [row for row in good if unseen(row)]
    if fresh:
        # Return fewer options rather than padding with places this user saw
        # before. When the best tier is exhausted, consider the next tier.
        fresh += [row for row in other if unseen(row)]
        return fresh[:limit]
    return (good + [row for row in other if unseen(row)] + [row for row in other if not unseen(row)])[:limit]
