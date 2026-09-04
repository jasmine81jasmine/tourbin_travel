"""Structured, session-scoped travel goal state."""

from typing import Literal

from pydantic import BaseModel, Field


class TravelGoal(BaseModel):
    """The current truth about the trip being planned in one chat session."""

    objective: str = ""
    origin: str = "تهران"
    destination_names: list[str] = Field(default_factory=list)
    region: str | None = None
    duration: str | None = None
    travel_dates: str | None = None
    season: str | None = None
    companions: list[str] = Field(default_factory=list)
    group_size: int | None = None
    moods: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    transport: str | None = None
    accommodation: list[str] = Field(default_factory=list)
    budget: str | None = None
    must_haves: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    semantic_query: str = ""
    status: Literal["collecting", "planning", "proposed", "confirmed"] = "collecting"
    revision: int = 0
    updated_at: str | None = None
