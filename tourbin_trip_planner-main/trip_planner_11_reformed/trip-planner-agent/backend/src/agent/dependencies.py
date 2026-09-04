"""Agent dependencies extending MemoryDependency with the trip graph repo."""

from dataclasses import dataclass

from neo4j_agent_memory import MemoryClient
from neo4j_agent_memory.integrations.pydantic_ai import MemoryDependency

from src.memory.graph import TripGraphRepository
from src.agent.goals import TravelGoal


@dataclass
class AgentDeps(MemoryDependency):
    """Dependencies available to every agent tool call."""

    graph: TripGraphRepository | None = None
    user_id: str = "anonymous"
    current_query: str | None = None
    travel_goal: TravelGoal | None = None

    @classmethod
    def create(
        cls,
        memory: MemoryClient | None,
        graph: TripGraphRepository | None,
        session_id: str,
        user_id: str = "anonymous",
        current_query: str | None = None,
        travel_goal: TravelGoal | None = None,
    ) -> "AgentDeps":
        return cls(
            client=memory,
            session_id=session_id,
            graph=graph,
            user_id=user_id,
            current_query=current_query,
            travel_goal=travel_goal,
        )
