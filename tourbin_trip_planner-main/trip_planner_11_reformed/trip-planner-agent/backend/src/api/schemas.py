"""API request/response schemas."""

from pydantic import BaseModel, Field


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


class SessionResponse(BaseModel):
    greeting: str
    session_id: str
    user_id: str
