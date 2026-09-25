"""Chat endpoint: text in, planned-trip text out, memory updated as a side effect."""

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from neo4j_agent_memory.integrations.pydantic_ai import record_agent_trace

from src.adapters.neshan_client import get_neshan_client
from src.agent.agent import get_trip_planner_agent, update_travel_goal
from src.agent.dependencies import AgentDeps
from src.agent.goals import TravelGoal
from src.agent.tools import build_itinerary_from_coords
from src.api.schemas import ChatRequest, ChatResponse, Itinerary, SessionResponse
from src.memory import linking
from src.memory.client import get_embedding_provider, get_memory_client
from src.memory.graph import TripGraphRepository

logger = logging.getLogger(__name__)
router = APIRouter()
USER_COOKIE = "tourbin_user_id"
USER_COOKIE_MAX_AGE = 60 * 60 * 24 * 365
TRAVEL_TERMS = (
    "سفر", "گردش", "تور", "مقصد", "برنامه", "مسیر", "جاده", "راه", "فاصله", "اقامت",
    "هتل", "کمپ", "طبیعت", "ییلاق", "روستا", "شهر", "استان", "دریا", "دریاچه",
    "جنگل", "آبشار", "کوه", "کویر", "ساحل", "دیدنی", "تفریح", "تعطیلات", "زیارت",
    "پرواز", "قطار", "اتوبوس", "ماشین", "خودرو", "آفرود", "پیاده", "آب و هوا",
    "هواشناسی", "بارندگی", "رزرو", "بوم گردی", "رستوران", "غذا", "چمدان", "وسایل",
    "چی ببر", "چه ببر", "هزینه", "قیمت", "بودجه", "کجا بریم", "همراه", "همسر",
    "خانواده", "بچه", "کودک", "سالمند", "دوستام", "چند روز", "یک روز", "دو روز",
    "سه روز", "آخر هفته", "travel", "trip", "tourism", "destination", "route", "hotel",
)
GREETING_TERMS = ("سلام", "درود", "صبح بخیر", "عصر بخیر", "شب بخیر", "خوبی", "hello", "hi")


def _set_user_cookie(response: Response, user_id: str) -> None:
    response.set_cookie(
        key=USER_COOKIE,
        value=user_id,
        max_age=USER_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )


def _is_travel_related(message: str) -> bool:
    text = " ".join(message.lower().replace("‌", " ").split())
    return any(term in text for term in TRAVEL_TERMS + GREETING_TERMS)


def _is_only_greeting(message: str) -> bool:
    text = " ".join(message.lower().replace("‌", " ").split())
    return len(text) <= 30 and any(term in text for term in GREETING_TERMS) and not any(
        term in text for term in TRAVEL_TERMS
    )


def _explicitly_mentions_destination(message: str, destination_name: str) -> bool:
    message_keys = TripGraphRepository._destination_lookup_keys(message)
    destination_keys = TripGraphRepository._destination_lookup_keys(destination_name)
    if not message_keys:
        return False
    padded_message = f" {message_keys[0]} "
    return any(f" {key} " in padded_message for key in destination_keys)


def _previous_goal_concept(state_json: str) -> str | None:
    try:
        state = json.loads(state_json)
    except (TypeError, json.JSONDecodeError):
        return None
    destinations = state.get("destination_names") or []
    place = " و ".join(destinations[:2]) or state.get("region")
    parts = [state.get("duration")]
    parts.extend((state.get("moods") or [])[:2])
    if place:
        parts.append(f"به مقصد {place}")
    parts = [str(part) for part in parts if part]
    return "سفر " + " ".join(parts) if parts else "برنامه سفر قبلی‌ات"


async def _load_session_history(memory_client, session_id: str, limit: int = 12):
    if memory_client is None:
        return [], ""
    try:
        conversation = await memory_client.short_term.get_conversation(session_id, limit=100)
    except Exception:
        logger.warning("Could not load native conversation history", exc_info=True)
        return [], ""

    messages = [m for m in conversation.messages if m.role.value in {"user", "assistant"}][-limit:]
    model_history = []
    transcript_lines = []
    for message in messages:
        if message.role.value == "user":
            model_history.append(ModelRequest(parts=[UserPromptPart(message.content)]))
            transcript_lines.append(f"کاربر: {message.content}")
        else:
            model_history.append(ModelResponse(parts=[TextPart(message.content)]))
            transcript_lines.append(f"توربین: {message.content}")
    return model_history, "\n".join(transcript_lines)


@router.post("/session", response_model=SessionResponse)
async def start_session(
    response: Response,
    browser_user_id: str | None = Cookie(default=None, alias=USER_COOKIE),
) -> SessionResponse:
    """Start a fresh chat while retaining this browser's user identity."""
    session_id = f"session_{uuid4().hex}"
    user_id = browser_user_id or f"user_{uuid4().hex}"
    _set_user_cookie(response, user_id)

    greeting = (
        "سلام! من توربینم؛ همسفرت برای ساختن یک برنامه سفر خوب 🌿 "
        "بگو برای سفر بعدی چه حال‌وهوایی در نظر داری تا با هم برنامه‌اش را بچینیم."
    )
    memory_client = get_memory_client()
    if memory_client is None:
        return SessionResponse(greeting=greeting, session_id=session_id, user_id=user_id)

    try:
        preferences = await memory_client.long_term.get_preferences_for(user_id)
        driver = memory_client._client._ensure_connected()  # type: ignore[attr-defined]
        history = await TripGraphRepository(driver).get_user_history(
            user_id,
            exclude_session_id=session_id,
        )
        remembered: list[str] = []
        if preferences:
            remembered.append(f"یادم هست که {preferences[0].preference}")
        if history.get("recent_goals"):
            concept = _previous_goal_concept(history["recent_goals"][0])
            if concept:
                remembered.append(f"یادم هست آخرین بار درباره {concept} برنامه‌ریزی می‌کردیم")
        elif history.get("recent_topics"):
            remembered.append("یادم هست قبلاً درباره برنامه سفرت صحبت کرده بودیم")
        elif history.get("trips"):
            destinations = history["trips"][0].get("destinations") or []
            if destinations:
                remembered.append(f"برنامه قبلی‌مون برای {destinations[0]} بود")
        if remembered:
            greeting = (
                "سلام، خوش برگشتی! من توربینم 🌿 "
                + "؛ ".join(remembered[:2])
                + ". دوست داری همان برنامه را ادامه بدهیم یا سفر تازه‌ای بسازیم؟"
            )
    except Exception:
        logger.warning("Could not personalize session greeting", exc_info=True)

    return SessionResponse(greeting=greeting, session_id=session_id, user_id=user_id)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    response: Response,
    browser_user_id: str | None = Cookie(default=None, alias=USER_COOKIE),
) -> ChatResponse:
    """Send a message to the trip planner agent.

    The agent reads/writes four memory layers as it reasons, and this
    endpoint links them together (see src.memory.linking):
    - short-term conversation memory (this session's raw messages)
    - long-term user preferences & the tourism knowledge graph (durable)
    - reasoning traces (how similar requests were handled before)
    - the user/trip/preference graph (this user's identity, likes, trips)
    """
    memory_client = get_memory_client()
    session_id = (request.session_id or "").strip() or f"session_{uuid4().hex}"
    user_id = (
        request.user_id.strip()
        if request.user_id and request.user_id.strip()
        else browser_user_id or f"user_{uuid4().hex}"
    )
    _set_user_cookie(response, user_id)

    graph_repo: TripGraphRepository | None = None
    driver = None
    if memory_client is not None:
        # neo4j-agent-memory's MemoryClient wraps a Neo4j driver internally;
        # `_client` is the same accessor pattern used by Lenny's Memory tools.
        try:
            # `memory_client._client` is a `neo4j_agent_memory.graph.client.Neo4jClient`
            # wrapper, not a raw driver -- and that wrapper has no public
            # `.driver` property (that's why this always raised AttributeError
            # and silently left graph_repo=None, disabling every tourism-graph
            # tool). Its actual driver lives at the private `_driver` attr,
            # accessed via `_ensure_connected()`, which also raises a clear
            # error instead of returning None if `connect()` hasn't run yet.
            # Neither is public API, so re-check this against whatever
            # neo4j-agent-memory version you have installed if it breaks
            # again after an upgrade.
            driver = memory_client._client._ensure_connected()  # type: ignore[attr-defined]
            graph_repo = TripGraphRepository(driver, get_embedding_provider())
        except AttributeError:
            logger.warning("Could not obtain Neo4j driver from memory client for TripGraphRepository")

    message_history, recent_transcript = await _load_session_history(memory_client, session_id)
    current_goal = None
    if graph_repo is not None:
        stored_goal = await graph_repo.get_travel_goal(session_id, user_id)
        if stored_goal:
            try:
                current_goal = TravelGoal.model_validate(stored_goal)
            except Exception:
                logger.warning("Ignoring invalid stored travel goal", exc_info=True)

    if not _is_travel_related(request.message):
        named_destination = None
        if graph_repo is not None:
            named_destination = await graph_repo.get_destination_details(request.message)
        if current_goal is None and not message_history and (named_destination is None or not _explicitly_mentions_destination(
            request.message,
            str(named_destination.get("name", "")),
        )):
            return ChatResponse(
                reply=(
                    "من توربینم و تخصصم ساختن تجربه‌های خوب سفره 🌿 "
                    "برای موضوع‌های خارج از سفر نمی‌تونم پاسخ دقیقی بدم؛ "
                    "اما با کمال میل می‌تونم برای مقصد، مسیر یا برنامه سفرت کمکت کنم. "
                    "دوست داری از چه سفری شروع کنیم؟"
                ),
                session_id=session_id,
                user_id=user_id,
            )

    travel_goal = current_goal
    if not (_is_only_greeting(request.message) and current_goal is None):
        try:
            travel_goal = await update_travel_goal(
                current_goal,
                request.message,
                recent_transcript,
            )
        except Exception:
            logger.warning("Could not update structured travel goal", exc_info=True)
            if travel_goal is None:
                travel_goal = TravelGoal(
                    objective=request.message,
                    semantic_query=request.message,
                    revision=1,
                )
        if graph_repo is not None and travel_goal is not None:
            await graph_repo.upsert_travel_goal(
                session_id,
                user_id,
                travel_goal.model_dump(),
            )

    deps = AgentDeps.create(
        memory=memory_client,
        graph=graph_repo,
        session_id=session_id,
        user_id=user_id,
        current_query=request.message,
        travel_goal=travel_goal,
        maps=get_neshan_client(),
    )

    agent = get_trip_planner_agent()

    try:
        # Store raw turns for native same-session history. Goal state now handles
        # constraints, so expensive per-message LLM extraction/vectorization is
        # intentionally skipped on the response path.
        #
        # `user_identifier=` is what writes (:User {identifier})-[:HAS_CONVERSATION]->
        # (:Conversation) -- passing it here is what puts this conversation on the
        # *same* User node that long_term.add_preference() and (after the fix
        # below) the tourism-graph writes use. Previously this call omitted
        # user_identifier entirely, so the conversation/message/entity subgraph
        # had no User node at all, let alone the same one everything else used.
        user_message = None
        if memory_client is not None:
            user_message = await memory_client.short_term.add_message(
                session_id=session_id,
                role="user",
                content=request.message,
                metadata={"user_id": user_id},
                user_identifier=user_id,
                extract_entities=False,
                extract_relations=False,
                generate_embedding=False,
                extraction_mode="skip",
            )

        result = await agent.run(request.message, deps=deps, message_history=message_history)
        reply_text = result.output if hasattr(result, "output") else str(result.data)

        if memory_client is not None:
            await memory_client.short_term.add_message(
                session_id=session_id,
                role="assistant",
                content=reply_text,
                metadata={"user_id": user_id},
                user_identifier=user_id,
                extract_entities=False,
                extract_relations=False,
                generate_embedding=False,
                extraction_mode="skip",
            )

            # `tool_get_similar_past_trip_plans` reads reasoning traces via
            # `reasoning.get_similar_traces`, but until now nothing ever wrote
            # one -- start_trace/add_step/complete_trace were never called, so
            # that tool always returned empty. `record_agent_trace` closes the
            # loop: it walks `result.all_messages()`, turns each tool call the
            # agent made (search_destinations, get_destination_details, ...)
            # into a ReasoningStep + ToolCall, and completes the trace with the
            # final reply as the outcome. Kept out of the main try/except path
            # (best-effort, non-fatal) so a bookkeeping failure here never
            # turns a good trip plan into a 500 for the user.
            try:
                trace = await record_agent_trace(
                    memory_client.reasoning,
                    session_id=session_id,
                    result=result,
                    task=request.message,
                )

                # Everything below closes the "islands" gap: without these,
                # the conversation graph, the reasoning-trace graph, and the
                # tourism graph are three subgraphs in the same database that
                # never reference each other's nodes.
                if driver is not None:
                    await linking.link_trace_to_user(driver, user_id, str(trace.id))

                if user_message is not None:
                    # Public library method -- exists specifically for this:
                    # linking a trace to its triggering message after the
                    # fact, since record_agent_trace() has no
                    # triggered_by_message_id param to do it at creation time.
                    await memory_client.reasoning.link_trace_to_message(trace.id, user_message.id)

                if graph_repo is not None:
                    destination_ids = linking.extract_recommended_destination_ids(result.all_messages())
                    linked = await linking.link_trace_recommendations(driver, str(trace.id), destination_ids)
                    if destination_ids:
                        logger.info(
                            "Linked %d/%d recommended destinations to trace %s",
                            linked,
                            len(destination_ids),
                            trace.id,
                        )
                    entities_linked = await linking.resolve_location_entities(driver)
                    if entities_linked:
                        logger.info("Resolved %d LOCATION entities to destinations", entities_linked)
            except Exception:
                logger.exception("Failed to record/link reasoning trace (non-fatal)")

        itinerary = None
        if deps.itinerary_result:
            try:
                itinerary = Itinerary.model_validate(deps.itinerary_result)
            except Exception:
                logger.warning("Could not serialize itinerary_result", exc_info=True)
        elif not _is_only_greeting(request.message):
            # Fallback: the agent settled on (a) destination(s) this turn but
            # didn't itself call tool_build_trip_map (prompt says it must,
            # but LLM tool-calling isn't 100% guaranteed) -- reconstruct a
            # simple, straight-line-distance itinerary from whatever
            # tool_get_destination_details/tool_find_destinations_near
            # already returned this turn, so the response still carries map
            # data rather than silently omitting it.
            try:
                fallback_stops = linking.extract_finalized_destination_coords(result.new_messages())
                if fallback_stops:
                    itinerary = Itinerary.model_validate(build_itinerary_from_coords(fallback_stops))
            except Exception:
                logger.warning("Could not build fallback itinerary", exc_info=True)

        return ChatResponse(reply=reply_text, session_id=session_id, user_id=user_id, itinerary=itinerary)
    except ModelAPIError as e:
        logger.warning("LLM provider request failed: %s", e)
        raise HTTPException(
            status_code=504,
            detail="سرویس هوش مصنوعی موقتاً دیر پاسخ می‌دهد. لطفاً چند لحظه دیگر دوباره تلاش کنید.",
        ) from e
    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
