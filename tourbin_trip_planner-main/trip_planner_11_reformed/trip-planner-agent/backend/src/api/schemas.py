"""API request/response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ItineraryStop(BaseModel):
    """One destination in the finalized plan, in visiting order -- enough
    to plot a route on a map (frontend concern, not built here)."""

    order: int
    name: str
    latitude: float
    longitude: float
    leg_distance_km_from_previous: float | None = None
    leg_duration_hours_from_previous: float | None = None


class ReturnLeg(BaseModel):
    """The last-stop -> origin leg, present only when `round_trip` is true.
    Kept separate from `stops` (rather than appending origin as a fake stop)
    since it has no `name`/`order` of its own -- it's the same origin point
    already given in `origin`. A map view can draw this as the final segment
    of the route back to `origin`'s coordinates."""

    leg_distance_km_from_previous: float
    leg_duration_hours_from_previous: float


class Itinerary(BaseModel):
    """Map-ready summary of the plan's destinations, order, and real drive
    distances/times, when the agent finalized a concrete itinerary this
    turn (see tool_build_trip_map). Absent/omitted otherwise -- e.g. plain
    small talk or a question that didn't reach a concrete plan.

    `total_distance_km`/`total_duration_hours` include the return leg
    whenever `round_trip` is true -- new consumers should treat these two
    fields as "everything driven, out and back" rather than one-way-only."""

    origin: dict[str, Any] | None = None
    stops: list[ItineraryStop] = Field(default_factory=list)
    return_leg: ReturnLeg | None = None
    total_distance_km: float | None = None
    total_duration_hours: float | None = None
    round_trip: bool = False


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message, in any language (Persian expected).")
    session_id: str | None = Field(
        default=None,
        description="Conversation/session identifier. A new one is generated when omitted or empty.",
    )
    user_id: str | None = Field(
        default=None,
        description="Stable identifier for the person, used for cross-session "
        "preferences and trip history. Restored from the browser cookie or generated if omitted.",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    user_id: str
    # New, additive field: present only when the agent finalized a concrete
    # itinerary this turn. Existing consumers that only read `reply`,
    # `session_id`, `user_id` are unaffected -- this is purely additional
    # data for a future map view.
    itinerary: Itinerary | None = None
    # Independent options have their own map routes; a single `itinerary`
    # cannot represent several alternatives without implying one joined trip.
    itineraries: list[Itinerary] = Field(default_factory=list)


class SessionResponse(BaseModel):
    greeting: str
    session_id: str
    user_id: str
