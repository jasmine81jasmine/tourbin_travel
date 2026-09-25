"""Structured, session-scoped travel goal state."""

import re
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
    # Recently proposed routes in THIS session, not places the user visited.
    # Enables new suggestions without persisting a false visited preference.
    recommended_routes: list[list[str]] = Field(default_factory=list)
    semantic_query: str = ""
    status: Literal["collecting", "planning", "proposed", "confirmed"] = "collecting"
    revision: int = 0
    updated_at: str | None = None


def remember_recommended_routes(goal: TravelGoal, routes: list[dict]) -> TravelGoal:
    """Remember a bounded set of proposed alternatives, independent of stop order."""
    seen = {tuple(sorted(names)) for names in goal.recommended_routes if names}
    history = list(goal.recommended_routes)
    for route in routes:
        names = [stop.get("name") for stop in route.get("stops", []) if stop.get("name")]
        key = tuple(sorted(names))
        if not key or key in seen:
            continue
        history.append(names)
        seen.add(key)
    return goal.model_copy(update={"recommended_routes": history[-16:]})


def recall_recommendations(goal: TravelGoal, transcript: str) -> TravelGoal:
    """Recover recommendations from older sessions created before tracking existed."""
    if goal.recommended_routes or not transcript:
        return goal
    history = []
    for match in re.finditer(r"ترتیب بازدید:\*\*\s*([^\n]+)|ترتیب بازدید:\s*([^\n]+)", transcript):
        route = (match.group(1) or match.group(2)).strip().rstrip("*")
        names = [part.strip() for part in route.split("←")]
        if len(names) >= 3 and names[0] == names[-1]:
            history.append(names[1:-1])
    for name in re.findall(r"#### گزینهٔ [0-9۰-۹]+:\s*([^\n]+)", transcript):
        history.append([name.strip()])
    if not history:
        return goal
    return remember_recommended_routes(goal, [{"stops": [{"name": name} for name in names]} for names in history])
