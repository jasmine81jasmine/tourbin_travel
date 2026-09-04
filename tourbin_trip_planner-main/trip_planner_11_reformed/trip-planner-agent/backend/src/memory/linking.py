"""Cross-island linking.

Four subgraphs get written to the same Neo4j database by two unrelated
codebases that never referenced each other's nodes:

1. short-term conversation graph   (neo4j-agent-memory: Conversation/Message/Entity)
2. reasoning-trace graph           (neo4j-agent-memory: ReasoningTrace/ReasoningStep/ToolCall)
3. tourism knowledge graph         (this project's etl/load_destinations.py: Destination/...)
4. user / trip / preference graph  (this project's TripGraphRepository + neo4j-agent-memory's
                                     long_term.add_preference)

Each function here writes exactly one relationship type that's otherwise
missing, so a Cypher query can actually walk from "what the user said" to
"what got recommended" to "who they are" instead of stopping at the edge
of whichever subgraph it started in. Kept as a separate module (not part
of `graph.py`'s tourism-domain queries, not part of `client.py`'s generic
memory wrapper) because none of these functions "belong" to either side --
their only job is the seam between them.

All functions are safe to call every turn: they only MERGE (idempotent)
and/or filter on "not already linked", so re-running never duplicates
edges or does wasted work on a second pass.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

# Tool names whose JSON return value contains real Destination `id`s worth
# turning into graph edges. Keep in sync with the @agent.tool wrapper names
# in src/agent/agent.py.
_DESTINATION_RETURNING_TOOLS = {
    "tool_search_destinations",
    "tool_get_destination_details",
    "tool_find_destinations_near",
}


async def link_trace_to_user(driver: AsyncDriver, user_identifier: str, trace_id: str) -> None:
    """(:User {identifier})-[:HAS_TRACE]->(:ReasoningTrace).

    `ReasoningMemory.start_trace(user_identifier=...)` writes this same edge
    at creation time, but `record_agent_trace()` -- the neo4j-agent-memory
    convenience helper used to auto-build trace/step/tool-call nodes from a
    pydantic-ai RunResult -- doesn't expose a `user_identifier` parameter, so
    it never gets called. Written after the fact instead, using the same
    MERGE pattern `start_trace` uses internally, so the resulting graph is
    indistinguishable from having passed user_identifier up front.
    """
    async with driver.session() as session:
        await session.run(
            """
            MERGE (u:User {identifier: $user_identifier})
            ON CREATE SET u.id = randomUUID(), u.created_at = datetime()
            WITH u
            MATCH (rt:ReasoningTrace {id: $trace_id})
            MERGE (u)-[:HAS_TRACE]->(rt)
            SET rt.user_identifier = $user_identifier
            """,
            user_identifier=user_identifier,
            trace_id=str(trace_id),
        )


def extract_recommended_destination_ids(all_messages: list[Any]) -> list[str]:
    """Pull every Destination `id` out of this turn's search_destinations /
    get_destination_details / find_destinations_near tool results.

    Those tools return JSON text (a list of dicts, or a single dict for
    get_destination_details) as the ToolReturnPart content. That's the only
    place the real destination ids exist after a tool call -- nothing
    currently reads it back out and turns it into a graph edge, so a
    "recommendation" is just flattened text sitting in a ToolCall.result
    property, one hop away from the real Destination node it refers to.
    """
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    ids: set[str] = set()
    for msg in all_messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolReturnPart) or part.tool_name not in _DESTINATION_RETURNING_TOOLS:
                continue
            content = part.content
            try:
                data = json.loads(content) if isinstance(content, str) else content
            except (TypeError, ValueError):
                continue
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                if isinstance(row, dict) and row.get("id"):
                    ids.add(row["id"])
    return list(ids)


async def link_trace_recommendations(driver: AsyncDriver, trace_id: str, destination_ids: list[str]) -> int:
    """(:ReasoningTrace)-[:RECOMMENDED]->(:Destination) for every destination
    id surfaced by this turn's tool calls, so a recommendation is a real
    graph edge instead of only existing as text inside a ToolCall.result
    property. Returns how many destinations actually matched and got linked
    (0 if the ids don't correspond to real nodes, e.g. a stale id).
    """
    if not destination_ids:
        return 0
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (rt:ReasoningTrace {id: $trace_id})
            UNWIND $destination_ids AS did
            MATCH (d:Destination {id: did})
            MERGE (rt)-[:RECOMMENDED]->(d)
            RETURN count(DISTINCT d) AS linked
            """,
            trace_id=str(trace_id),
            destination_ids=destination_ids,
        )
        record = await result.single()
        return record["linked"] if record else 0


async def resolve_location_entities(driver: AsyncDriver) -> int:
    """(:Entity {type:'LOCATION'})-[:REFERS_TO]->(:Destination) for
    conversation entities whose name matches a real destination.

    Restricted to type='LOCATION' (one of neo4j-agent-memory's POLE+O entity
    types, set by its LLM extractor) specifically to avoid nonsense matches
    -- e.g. a TripType-ish entity like "کمپ" substring-matching an unrelated
    destination name. `WHERE NOT (e)-[:REFERS_TO]->(:Destination)` means
    this only does work on newly-extracted entities each time it's called,
    so it's cheap enough to run every turn rather than as a batch job.
    Returns how many entities got a new link.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity {type: 'LOCATION'})
            WHERE NOT (e)-[:REFERS_TO]->(:Destination)
            MATCH (d:Destination)
            WHERE toLower(d.name) = toLower(e.name)
               OR toLower(d.name) CONTAINS toLower(e.name)
               OR toLower(e.name) CONTAINS toLower(d.name)
            MERGE (e)-[:REFERS_TO]->(d)
            RETURN count(DISTINCT e) AS linked
            """
        )
        record = await result.single()
        return record["linked"] if record else 0
