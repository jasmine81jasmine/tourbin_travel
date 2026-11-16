"""Structured, session-scoped travel goal state."""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field


class PlanningSubgoal(BaseModel):
    id: str
    name: str
    priority: Literal["high", "medium", "low"] = "medium"
    source: Literal["explicit", "preference", "assumption"] = "assumption"
    # Only these evidence types may be marked covered by the evaluator.
    evidence_type: Literal["destination", "category", "feature", "route", "unverified"] = "unverified"
    target: str | None = None
    category: str | None = None
    feature_id: str | None = None
    minimum: int = Field(default=1, ge=1, le=10)
    related_features: list[str] = Field(default_factory=list)
    parent_id: str | None = None


class PlanningConstraint(BaseModel):
    description: str
    kind: Literal["hard", "soft"] = "soft"
    source: Literal["explicit", "preference", "assumption"] = "assumption"


class GoalCoverage(BaseModel):
    goal_id: str
    status: Literal["pending", "partial", "satisfied", "blocked"]
    activities: list[str] = Field(default_factory=list)
    reason: str = ""


class PendingOffer(BaseModel):
    """An assistant proposal awaiting a response, not yet a user constraint."""
    action: Literal["plan", "choose"]
    destination_names: list[str] = Field(default_factory=list)
    # Preserve the offered order, including names not found in the graph.
    # Unverified names must never be promoted to a selected trip destination.
    offered_names: list[str] = Field(default_factory=list)
    unverified_names: list[str] = Field(default_factory=list)
    incomplete: bool = False
    # "plan" visits all named destinations together; "choose" presents alternatives.
    source_reply: str = ""


class ConversationalGoal(BaseModel):
    """User-authored dialogue commitment; separate from ontology planning subgoals."""
    id: str
    clause: str
    summary: str
    kind: Literal["question", "request", "offer", "suggestion"]
    scope: Literal["trip", "turn"] = "turn"
    dimensions: list[str] = Field(default_factory=list)
    status: Literal["active", "addressed", "superseded", "archived"] = "active"
    parents: list[str] = Field(default_factory=list)


class GoalMergeEvent(BaseModel):
    operation: Literal["keep", "replace", "combine", "add"]
    old_ids: list[str] = Field(default_factory=list)
    new_id: str | None = None


class ConversationalEvaluation(BaseModel):
    goal_id: str
    category: Literal["confirm", "contradict", "ignore"]
    explanation: str = ""
    examples: list[str] = Field(default_factory=list)


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
    categories: list[str] = Field(default_factory=list)
    trip_types: list[str] = Field(default_factory=list)
    moods: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    transport: str | None = None
    accommodation: list[str] = Field(default_factory=list)
    budget: str | None = None
    must_haves: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    subgoals: list[PlanningSubgoal] = Field(default_factory=list)
    planning_constraints: list[PlanningConstraint] = Field(default_factory=list)
    planning_steps: list[str] = Field(default_factory=list)
    goal_coverage: list[GoalCoverage] = Field(default_factory=list)
    last_observation_tool: str | None = None
    # Audit metadata for the snapshot being written, not a claim of goal coverage.
    turn_id: str | None = None
    revision_cause: str | None = None
    observation_id: str | None = None
    observation_source: str | None = None
    observation_summary: list[str] = Field(default_factory=list)
    # The most recent turn is recorded even if it only requested information.
    last_user_request: str = ""
    last_turn_action: str = ""
    # Final map-ready routes from the most recent plan, not searched candidates.
    last_itineraries: list[dict] = Field(default_factory=list)
    # Soft context from the latest *other* session, never an explicit request
    # for this trip and never saved as a new REQUESTED_FOR_TRIP link.
    prior_insights: dict[str, list[str]] = Field(default_factory=dict)
    # Recently proposed routes in THIS session, not places the user visited.
    # Enables new suggestions without persisting a false visited preference.
    recommended_routes: list[list[str]] = Field(default_factory=list)
    # The alternatives shown in the last planning reply, for "why these?"
    last_proposed_routes: list[list[str]] = Field(default_factory=list)
    pending_offer: PendingOffer | None = None
    # OnGoal-style infer → merge → evaluate, scoped to this session. Historical
    # snapshots are immutable Neo4j TravelGoal revisions, not user preferences.
    conversational_goals: list[ConversationalGoal] = Field(default_factory=list)
    goal_merge_events: list[GoalMergeEvent] = Field(default_factory=list)
    conversational_evaluations: list[ConversationalEvaluation] = Field(default_factory=list)
    next_conversational_goal: int = 1
    semantic_query: str = ""
    status: Literal["collecting", "planning", "proposed", "confirmed"] = "collecting"
    revision: int = 0
    updated_at: str | None = None


def with_prior_insights(goal: TravelGoal, prior: dict[str, list[str]]) -> TravelGoal:
    """Use past *requests* only as optional hints where this trip is silent."""
    hints = {}
    for key in ("categories", "trip_types", "moods"):
        if not getattr(goal, key):
            values = [value for value in prior.get(key, []) if value not in goal.avoid]
            if values:
                hints[key] = values[:3]
    # Companions, season and vehicle from another trip are remembered for a
    # greeting, not assumed to apply to this trip's routing/safety constraints.
    goal = goal.model_copy(update={"prior_insights": hints})
    if hints:
        goal = goal.model_copy(update={
            "semantic_query": goal.semantic_query + "؛ علایق احتمالی از درخواست قبلی: "
            + "، ".join(value for values in hints.values() for value in values)
        })
    return goal


def insights_from_previous_goal(state_json: str) -> dict[str, list[str]]:
    """Backfill soft hints for sessions created before feature links existed."""
    try:
        state = TravelGoal.model_validate(json.loads(state_json))
    except (ValueError, TypeError):
        return {}
    prior = {}
    for key in ("categories", "moods", "trip_types", "companions"):
        values = [value for value in getattr(state, key) if value not in state.avoid]
        if values:
            prior[key] = values
    if "کمپ" in state.accommodation and "کمپ" not in state.avoid:
        prior.setdefault("trip_types", []).append("کمپ")
        prior["trip_types"] = list(dict.fromkeys(prior["trip_types"]))
    return prior


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
    latest = [[stop["name"] for stop in route.get("stops", []) if stop.get("name")]
              for route in routes]
    return goal.model_copy(update={"recommended_routes": history[-16:],
                                   "last_proposed_routes": [names for names in latest if names][:4]})


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
