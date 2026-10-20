"""Agent dependencies extending MemoryDependency with the trip graph repo."""

from dataclasses import dataclass, field
from typing import Any

from neo4j_agent_memory import MemoryClient
from neo4j_agent_memory.integrations.pydantic_ai import MemoryDependency

from src.adapters.neshan_client import NeshanClient
from src.memory.graph import TripGraphRepository
from src.agent.goals import TravelGoal


@dataclass
class AgentDeps(MemoryDependency):
    """Dependencies available to every agent tool call."""

    graph: TripGraphRepository | None = None
    user_id: str = "anonymous"
    current_query: str | None = None
    travel_goal: TravelGoal | None = None
    turn_action: str = "plan"
    maps: NeshanClient | None = None
    previous_recommendations: dict[str, set[str]] = field(default_factory=lambda: {"ids": set(), "names": set()})
    durable_preferences: list[dict[str, Any]] = field(default_factory=list)
    grounded_mentions: list[dict[str, str]] = field(default_factory=list)
    # Written by tool_build_trip_map (see src/agent/tools.py) when the agent
    # finalizes a concrete itinerary. Read back by the /api/chat route after
    # agent.run() to attach map-ready {name, latitude, longitude, order}
    # data to the JSON response, alongside the normal text reply.
    itinerary_result: dict[str, Any] | None = field(default=None)

    @classmethod
    def create(
        cls,
        memory: MemoryClient | None,
        graph: TripGraphRepository | None,
        session_id: str,
        user_id: str = "anonymous",
        current_query: str | None = None,
        travel_goal: TravelGoal | None = None,
        turn_action: str = "plan",
        maps: NeshanClient | None = None,
        previous_recommendations: dict[str, set[str]] | None = None,
        durable_preferences: list[dict[str, Any]] | None = None,
        grounded_mentions: list[dict[str, str]] | None = None,
    ) -> "AgentDeps":
        return cls(
            client=memory,
            session_id=session_id,
            graph=graph,
            user_id=user_id,
            current_query=current_query,
            travel_goal=travel_goal,
            turn_action=turn_action,
            maps=maps,
            previous_recommendations=previous_recommendations or {"ids": set(), "names": set()},
            durable_preferences=durable_preferences or [],
            grounded_mentions=grounded_mentions or [],
        )
