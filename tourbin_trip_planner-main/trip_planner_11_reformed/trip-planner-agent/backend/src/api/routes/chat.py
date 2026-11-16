"""Chat endpoint: text in, planned-trip text out, memory updated as a side effect."""

import json
import logging
import re
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from neo4j_agent_memory.integrations.pydantic_ai import record_agent_trace

from src.adapters.neshan_client import get_neshan_client
from src.agent.agent import get_trip_planner_agent, update_travel_goal
from src.agent.dependencies import AgentDeps
from src.agent.description_format import format_descriptions
from src.agent.feasibility import ground_route_text, screen_short_trip
from src.agent.goals import (PendingOffer, TravelGoal, insights_from_previous_goal, recall_recommendations,
                              remember_recommended_routes, with_prior_insights)
from src.agent.goal_manager import (design_goals, evaluate_plan, replan_feedback,
                                     decompose_unmet_goal, retire_replaced_subgoals)
from src.agent.conversation_pipeline import infer_and_merge, evaluate_response
from src.agent.goal_progress import observed_result
from src.agent.memory_summary import summarize_user_memory
from src.agent.personalization import explain_selected_plans, write_explanation
from src.agent.tools import build_trip_map, semantic_search_destinations
from src.agent.turns import (TurnDecision, apply_goal_updates, current_trip_query,
                             explicit_vehicle_from_vocabulary, infer_pending_offer,
                             understand_turn)
from src.api.schemas import ChatRequest, ChatResponse, Itinerary, SessionResponse
from src.api.routes.auth import Identity, claim_session, require_user
from src.memory import linking
from src.memory.client import get_embedding_provider, get_memory_client
from src.memory.graph import TripGraphRepository
from src.memory.preference_schedule import mark_chat_activity

logger = logging.getLogger(__name__)
router = APIRouter()


def _compose_plan_reply(reply: str, explanation: str, *, first_turn: bool) -> str:
    """Lead with the greeting/introduction, then reasons, then the itinerary."""
    text = reply.strip()
    if not explanation:
        return ("سلام!\n\n" + text if first_turn and not re.match(
            r"^\s*(?:سلام|درود)(?:[!،.\s]|$)", text) else text)
    first, separator, rest = text.partition("\n\n")
    if re.match(r"^\s*(?:سلام|درود)(?:[!،.\s]|$)", first):
        return first.strip() + "\n\n" + explanation + (
            "\n\n" + rest if separator and rest.strip() else "")
    if first_turn:
        if separator and first.startswith(("یادم", "از درخواست سفر قبلی", "از سفرهای قبلی",
                                           "الان حافظه")):
            return "سلام!\n\n" + first + "\n\n" + explanation + "\n\n" + rest
        return "سلام!\n\n" + explanation + "\n\n" + text
    return explanation + "\n\n" + text


async def _preload_semantic_candidates(deps: AgentDeps) -> None:
    """Always attempt complementary retrieval before a planning agent run."""
    if (deps.turn_action != "plan" or deps.graph is None
            or not hasattr(deps.graph, "semantic_search_destinations")):
        return
    try:
        context = SimpleNamespace(deps=deps)
        deps.semantic_candidates = await semantic_search_destinations(context, limit=6)
        await observed_result(
            context, "semantic_search_destinations", deps.semantic_candidates,
            source="preload")
    except Exception:
        logger.warning("Initial semantic destination search unavailable", exc_info=True)


def _is_memory_question(message: str) -> bool:
    text = " ".join(message.lower().replace("‌", " ").replace("ي", "ی").split())
    if _has_plan_request(text):
        return False
    return ((any(term in text for term in ("از من میدونی", "از من می دونی", "درباره من میدونی",
                                        "درباره من می دونی", "از من یادت", "درباره من یادت",
                                        "چه چیزایی از من", "چیا از من", "چی از من",
                                        "چی درباره من", "راجع به من چی", "چقدر منو میشناسی",
                                        "چقدر منو می شناسی", "ازم چی میدونی", "ازم چی می دونی",
                                        "برنامه سفرهام چی", "سفرهام چی میدونی"))
             and any(term in text for term in ("چی", "چه", "میدونی", "می دونی", "یادت")))
            or "what do you know about me" in text or "what do you remember about me" in text)


def _has_plan_request(text: str) -> bool:
    return bool(re.search(r"(?:برنامه|پلن).{0,30}(?:بده|بریز|بساز|بچین|پیشنهاد)"
                          r"|(?:سفر|مقصد).{0,18}(?:پیشنهاد بده|معرفی کن)", text)
                or "plan a trip" in text or "recommend a trip" in text)


def _requests_more_plans(
    message: str, goal: TravelGoal | None = None, last_reply: str = "",
) -> bool:
    """Resolve an elliptical 'what else?' against the immediately prior plan."""
    text = " ".join(message.replace("‌", " ").replace("ي", "ی").split())
    explicit = bool(
        re.search(r"(?:دیگه|دیگر|جدید|بیشتر).{0,35}(?:برنامه|پلن|مسیر|سفر)"
                  r"|(?:برنامه|پلن|مسیر|سفر).{0,35}(?:دیگه|دیگر|جدید)", text)
        and re.search(r"چی داری|چه .{0,20}داری|پیشنهاد|بده|بریز|بساز|معرفی|"
                      r"چه .{0,20}دارید|می ?تونی|می ?توانی", text))
    if explicit:
        return True
    return bool(
        goal and (goal.last_proposed_routes or goal.recommended_routes)
        and last_reply and any(term in last_reply for term in ("برنامه", "مسیر", "گزینه"))
        and re.search(r"(?:دیگه|دیگر|جدید|بیشتر).{0,25}(?:چی|چه).{0,15}"
                      r"(?:داری|دارید|هست|پیشنهاد)", text))


def _is_memory_based_plan(message: str) -> bool:
    text = " ".join(message.lower().replace("‌", " ").replace("ي", "ی").split())
    return (any(term in text for term in ("از من میدونی", "از من می دونی", "ازم میدونی",
                                         "ازم می دونی", "بر اساس شناختت", "چیزایی که میدونی",
                                         "based on what you know"))
            and _has_plan_request(text))


def _asks_why_plan(message: str) -> bool:
    text = " ".join(message.replace("‌", " ").split())
    return bool(re.search(r"چرا\s+(?:این|اون|همین)\s+(?:برنامه|برنامه\s*ها|مقصد|مسیر)|"
                          r"(?:دلیل|علت)\s+(?:انتخاب|پیشنهاد)\s+(?:این|اون)", text))


def _asks_vehicle(message: str) -> bool:
    text = message.replace("‌", " ")
    return (not _has_plan_request(text)
            and bool(re.search(r"(?:با\s+)?چه\s+(?:ماشین|خودرو|وسیله)|"
                               r"چه\s+نوع\s+(?:ماشین|خودرو)|سواری\s+یا\s+آفرود", text)))


async def _save_direct_exchange(memory_client, session_id: str, user_id: str,
                                question: str, reply: str) -> None:
    if memory_client is None:
        return
    try:
        for role, content in (("user", question), ("assistant", reply)):
            await memory_client.short_term.add_message(
                session_id=session_id, role=role, content=content,
                metadata={"user_id": user_id}, user_identifier=user_id,
                extract_entities=False, extract_relations=False,
                generate_embedding=False, extraction_mode="skip",
            )
    except Exception:
        logger.warning("Could not store direct travel answer", exc_info=True)


async def _mark_completed_turn(identity, driver, user_id: str, session_id: str) -> None:
    if not isinstance(identity, Identity) or driver is None:
        return
    try:
        await mark_chat_activity(driver, user_id, session_id)
    except Exception:
        logger.warning("Could not schedule preference extraction", exc_info=True)


async def _save_answer_goal(graph, goal: TravelGoal | None, session_id: str,
                            user_id: str, request: str) -> None:
    """Record a factual follow-up in the same goal history without inventing a new plan."""
    if graph is None or goal is None:
        return
    snapshot = goal.model_copy(deep=True)
    snapshot.last_user_request = request
    snapshot.last_turn_action = "answer"
    snapshot.turn_id = str(uuid4())
    snapshot.revision_cause = "direct_answer"
    snapshot.updated_at = datetime.now(timezone.utc).isoformat()
    try:
        await graph.upsert_travel_goal(session_id, user_id, snapshot.model_dump())
    except Exception:
        logger.warning("Could not record goal context for direct answer", exc_info=True)


async def _verified_access_answer(message: str, goal: TravelGoal | None, graph) -> str | None:
    """Do not equate a destination-wide amenity with an accessible nearby one."""
    if graph is None or goal is None or len(goal.destination_names) != 1:
        return None
    question = message.replace("‌", " ")
    goal_text = " ".join([*goal.must_haves, *goal.constraints,
                          *(subgoal.name for subgoal in goal.subgoals)]).replace("‌", " ")
    asks_toilet = "سرویس بهداشتی" in question or "دستشویی" in question
    needs_short_walk = any(term in goal_text for term in ("پیاده روی کم", "پیاده روی کوتاه", "کم پیاده"))
    asks_access = any(term in question for term in ("نزدیک", "پیاده", "دسترسی", "تایید", "تأیید"))
    if not (asks_toilet and asks_access and needs_short_walk):
        return None
    name = goal.destination_names[0]
    try:
        detail = await graph.get_destination_details(name)
    except Exception:
        logger.warning("Could not verify nearby facilities for selected destination", exc_info=True)
        detail = None
    description = (detail or {}).get("description") or ""
    listed = "سرویس بهداشتی" in description or "سرویس‌های بهداشتی" in description
    if listed:
        return (f"در توضیحات {name} به سرویس بهداشتی اشاره شده، اما محل آن نسبت به "
                "نقطهٔ شروع، فاصلهٔ پیاده‌روی و دسترس‌پذیری آن برای خانواده تأیید نشده است. "
                "بنابراین نمی‌توانم سرویس نزدیک و برنامهٔ کم‌پیاده‌روی را تأیید کنم؛ "
                "پیش از حرکت از محل یا خدمات محلی استعلام بگیرید.")
    return (f"برای {name} سرویس بهداشتی نزدیک نقطهٔ شروع و مسیر کم‌پیاده‌روی "
            "در داده‌های در دسترس تأیید نشده است؛ پیش از حرکت از محل استعلام بگیرید.")


async def _answer_vehicle_question(message: str, goal: TravelGoal | None, graph) -> str | None:
    if graph is None or not _asks_vehicle(message):
        return None
    names = goal.destination_names if goal else []
    if not names and goal and len(goal.last_proposed_routes) == 1 and len(goal.last_proposed_routes[0]) == 1:
        names = goal.last_proposed_routes[0]
    try:
        mentioned = await graph.get_destination_details(message)
        if mentioned and _explicitly_mentions_destination(message, str(mentioned.get("name", ""))):
            details = mentioned
            names = [mentioned["name"]]
        else:
            details = await graph.get_destination_details(names[-1]) if names else None
    except Exception:
        logger.warning("Could not look up chosen destination vehicle", exc_info=True)
        return None
    if not details or (names and details.get("name") != names[-1]):
        return None
    name = details.get("name", "این مقصد")
    vehicles = [v for v in (details.get("vehicles") or []) if isinstance(v, str)]
    if vehicles:
        phrase = " و ".join(vehicles[:2])
        reply = f"برای {name}، در اطلاعات دسترسیِ مقصد {phrase} ذکر شده است. "
    else:
        reply = f"برای {name} نوع خودروی مناسب در اطلاعات مقصد مشخص نشده است. "
    if details.get("physical_readiness"):
        reply += f"بازدید هم آمادگی بدنی {details['physical_readiness']} می‌خواهد؛ مسیر پیاده را با مسیر خودرو اشتباه نگیریم. "
    return reply + "وضعیت بخش پایانی راه را نزدیک روز حرکت بررسی کن."


def _explicitly_mentions_destination(message: str, destination_name: str) -> bool:
    message_keys = TripGraphRepository._destination_lookup_keys(message)
    destination_keys = TripGraphRepository._destination_lookup_keys(destination_name)
    if not message_keys:
        return False
    padded_message = f" {message_keys[0]} "
    return any(f" {key} " in padded_message for key in destination_keys)


def _asks_for_one_of_offered_options(message: str) -> bool:
    normalized = " ".join(message.replace("‌", " ").replace("ي", "ی").split())
    return (not re.search(r"کدوم|کدام|چرا|چطور|آیا", normalized)
            and bool(re.search(r"(?:یکیشو|یکیش|یکی(?:ش)? رو|یکی از (?:این|اون|آن)(?:ها|ا)|"
                               r"(?:اولی|دومی|سومی)(?: رو)?|گزینه (?:اول|دوم|سوم))"
                               r".{0,25}(?:بده|بگو|انتخاب|برنامه|بریز|میخوام|می خواهم|کن)",
                               normalized)))


def _is_bare_confirmation(message: str) -> bool:
    text = " ".join(message.strip().replace("‌", " ").replace("ي", "ی").split())
    return bool(re.fullmatch(r"(?:آره|اره|بله|باشه|اوکی|حتما|حتماً|آری|yes)"
                             r"(?:[!،. ]*(?:لطفا|لطفاً|ممنون))?[!.؟ ]*", text,
                             flags=re.IGNORECASE))


def _last_assistant_offer(transcript: str) -> tuple[str, str]:
    if "توربین:" not in transcript:
        return "", ""
    before, last_reply = transcript.rsplit("توربین:", 1)
    previous_user = before.rsplit("کاربر:", 1)[-1].strip() if "کاربر:" in before else ""
    return previous_user, last_reply.strip()


def _offers_plan(reply: str) -> bool:
    tail = reply[-650:].replace("‌", " ")
    return bool(re.search(
        r"(?:می ?توانم|میتونم|می ?تونم|مایلی|دوست داری|دوست دارید|"
        r"تمایل دارید|می ?خواهید|می ?خواید|اگر بخواهید)"
        r".{0,240}(?:برنامه|سفر|بازدید|ترکیبی|انتخاب)|"
        r"(?:برنامه|سفر|ترکیبی).{0,120}(?:می ?خواهی|می ?خوای|تمایل دارید)",
        tail))


async def _resolve_accepted_plan_offer(transcript: str, graph) -> tuple[list[str], bool]:
    """Recover an old offer from the immediate exchange; never search older turns."""
    previous_user, reply = _last_assistant_offer(transcript)
    if not reply:
        return [], False
    pending = await _capture_pending_offer(previous_user, reply, graph, force=True)
    return (pending.destination_names if pending and pending.action == "plan"
            and not pending.unverified_names and not pending.incomplete else [],
            pending is not None or _offers_plan(reply))


async def _capture_pending_offer(user_message: str, reply: str, graph,
                                 *, force: bool = False) -> PendingOffer | None:
    """LLM interprets the last exchange; graph validates every proposed target."""
    if graph is None or not (force or _offers_plan(reply)):
        return None
    try:
        candidates = await graph.find_destinations_mentioned_in_text(
            user_message + "\n" + reply, limit=16)
        inferred = await infer_pending_offer(user_message, reply, candidates)
        if inferred is None:
            return None
        validated = []
        offered = []
        unverified = []
        incomplete = False
        context = user_message + "\n" + reply
        for name in inferred.destination_names:
            if not _explicitly_mentions_destination(context, name):
                incomplete = True
                continue
            detail = await graph.get_destination_details(name)
            canonical = str((detail or {}).get("name") or "")
            offered.append(canonical or name)
            if canonical:
                validated.append(canonical)
            else:
                unverified.append(name)
        if not offered:
            return None
        return PendingOffer(action=inferred.action,
                            destination_names=list(dict.fromkeys(validated)),
                            offered_names=list(dict.fromkeys(offered)),
                            unverified_names=list(dict.fromkeys(unverified)),
                            incomplete=incomplete,
                            source_reply=reply[-1500:])
    except Exception:
        logger.warning("Could not validate assistant offer against the destination graph", exc_info=True)
        return None


async def _resolve_offered_option(message: str, goal: TravelGoal | None,
                                  transcript: str, graph) -> str | None:
    """Resolve a deictic choice against the last assistant offer, not old searches."""
    if not goal or graph is None or not _asks_for_one_of_offered_options(message):
        return None
    _, last_reply = _last_assistant_offer(transcript)
    if not last_reply:
        return None
    pending = goal.pending_offer
    if pending and pending.offered_names and last_reply.endswith(pending.source_reply):
        if pending.incomplete:
            return None  # ordinal positions cannot be trusted with missing options
        names = pending.offered_names
        normalized = message.replace("‌", " ")
        index = 2 if "سومی" in normalized or "گزینه سوم" in normalized else (
            1 if "دومی" in normalized or "گزینه دوم" in normalized else 0)
        if index >= len(names) or names[index] not in pending.destination_names:
            return None
        if any(_explicitly_mentions_destination(message, name) for name in names):
            return None
        return names[index]
    try:
        # The graph matcher preserves the order of names in the actual reply.
        names = await graph.find_destinations_mentioned_in_text(last_reply, limit=12)
    except Exception:
        logger.warning("Could not resolve offered destinations", exc_info=True)
        names = []
    offered = [route[0] for route in goal.last_proposed_routes if len(route) == 1
               and _explicitly_mentions_destination(last_reply, route[0])]
    if offered:
        # Earlier destinations may appear incidentally in the prose; the
        # recorded last proposal is a better ordering when still in the reply.
        names = offered
    else:
        numbered = re.findall(r"(?m)^\s*[0-9۰-۹]+\s*[.)]\s+\S+", last_reply)
        if numbered and len(numbered) != len(names):
            # A missing graph match shifts ordinal positions; "third" must
            # not silently select the fourth destination in the reply.
            return None
    if not names:
        return None
    normalized = message.replace("‌", " ")
    index = 2 if "سومی" in normalized or "گزینه سوم" in normalized else (
        1 if "دومی" in normalized or "گزینه دوم" in normalized else 0)
    if index >= len(names):
        return None
    # Avoid treating a destination mentioned in a comparison as a selected one.
    if any(_explicitly_mentions_destination(message, name) for name in names):
        return None
    return names[index]


async def _unrouted_selected_plan(goal: TravelGoal, graph) -> str:
    """Describe only the selected graph places, without claiming a mapped route."""
    selected = goal.destination_names
    context = " ".join(filter(None, [
        "دو نفره" if goal.group_size == 2 or "دو نفر" in goal.companions else "",
        goal.season or "",
    ]))
    trip = f"سفر {context}" if context else "سفرت"
    lines = [f"برای {trip}، {' و '.join(selected)} را که انتخاب کردی بررسی می‌کنیم."]
    for chosen in selected:
        try:
            details = await graph.get_destination_details(chosen) if graph is not None else None
        except Exception:
            logger.warning("Could not load selected destination details", exc_info=True)
            details = None
        if not details or not _explicitly_mentions_destination(str(details.get("name", "")), chosen):
            lines.append(f"برای {chosen} فعلاً اطلاعات مقصد در دسترسم نیست؛ "
                         "مقصد دیگری را جایگزین آن نمی‌کنم.")
            continue
        name = details["name"]
        description = " ".join(str(details.get("description") or "").split())
        if description:
            # A database article can contain stale driving figures; those
            # must not be laundered into an unrouted plan.
            sentences = re.split(r"(?<=[.!؟])\s+", description)
            description = " ".join(sentence for sentence in sentences
                                   if not re.search(r"[0-9۰-۹٠-٩]+(?:[.,٫][0-9۰-۹٠-٩]+)?\s*"
                                                    r"(?:کیلومتر|ساعت|دقیقه|km)", sentence)
                                   and not any(term in sentence for term in (
                                       "برای دسترسی", "مسیر رانندگی", "مسیر خودرو")))
            if description:
                excerpt = description[:650]
                lines.append(f"**دربارهٔ {name}:** {excerpt.rsplit(' ', 1)[0] if len(description) > 650 else excerpt}")
        if details.get("categories"):
            lines.append(f"دسته‌بندی ثبت‌شدهٔ {name}: " + "، ".join(details["categories"][:3]))
    if goal.activities and any("پیاده" in activity for activity in goal.activities):
        lines.append("برای پیاده‌روی، طول و دشواری مسیر این مقصد را پیش از حرکت بررسی کنید؛ "
                     "وجود مسیر مناسب با سطح آمادگی شما هنوز تأیید نشده است.")
    lines.append("**طرح پیشنهادی:** برای بازدید از مقصدهای انتخابی زمان گشت‌وگذار "
                 "و استراحت در نظر بگیرید. ترتیب بازدید، مسیر رانندگی و زمان "
                 "رفت‌وبرگشت با نقشه تأیید نشده؛ پیش از حرکت بررسی کنید.")
    return "\n\n".join(lines)


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


def _previous_feature_concept(features: dict[str, list[str]]) -> str | None:
    parts = []
    for key in ("moods", "categories", "trip_types"):
        if features.get(key):
            parts.append(features[key][0])
    if features.get("companions"):
        parts.append("با " + features["companions"][0])
    return "، ".join(parts[:3]) or None


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
    browser_user_id: Identity = Depends(require_user),
) -> SessionResponse:
    """Start a fresh chat belonging to the authenticated account."""
    session_id = f"session_{uuid4().hex}"
    user_id = (browser_user_id.user_id if isinstance(browser_user_id, Identity)
               else browser_user_id or f"user_{uuid4().hex}")
    if isinstance(browser_user_id, Identity):
        await claim_session(user_id, session_id, create=True)

    greeting = (
        "سلام! من توربینم؛ همسفرت برای ساختن یک برنامه سفر خوب 🌿 "
        "بگو برای سفر بعدی چه حال‌وهوایی در نظر داری تا با هم برنامه‌اش را بچینیم."
    )
    memory_client = get_memory_client()
    if memory_client is None:
        return SessionResponse(greeting=greeting, session_id=session_id, user_id=user_id)

    preferences = []
    recent_features = {}
    history = {}
    try:
        preferences = await memory_client.long_term.get_preferences_for(user_id)
    except Exception:
        logger.warning("Could not load durable preferences for greeting", exc_info=True)
    try:
        driver = memory_client._client._ensure_connected()  # type: ignore[attr-defined]
        graph = TripGraphRepository(driver)
        try:
            recent_features = await graph.get_recent_session_features(user_id, session_id)
        except Exception:
            logger.warning("Could not load prior trip features for greeting", exc_info=True)
        try:
            history = await graph.get_user_history(user_id, exclude_session_id=session_id)
        except Exception:
            logger.warning("Could not load trip history for greeting", exc_info=True)
    except Exception:
        logger.warning("Could not initialize trip graph for greeting", exc_info=True)
    if not recent_features and history.get("recent_goals"):
        recent_features = insights_from_previous_goal(history["recent_goals"][0])
    remembered: list[str] = []
    if preferences:
        remembered.append(f"یادم هست که {preferences[0].preference}")
    concept = _previous_feature_concept(recent_features)
    if concept:
        remembered.append(f"آخرین بار برای سفری با حال‌وهوای {concept} برنامه می‌خواستی")
    elif history.get("recent_goals"):
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

    return SessionResponse(greeting=greeting, session_id=session_id, user_id=user_id)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    response: Response,
    browser_user_id: Identity = Depends(require_user),
) -> ChatResponse:
    """Send a message to the trip planner agent.

    The agent reads/writes four memory layers as it reasons, and this
    endpoint links them together (see src.memory.linking):
    - short-term conversation memory (this session's raw messages)
    - long-term user preferences & the tourism knowledge graph (durable)
    - reasoning traces (how similar requests were handled before)
    - the user/trip/preference graph (this user's identity, likes, trips)
    """
    supplied_session = (request.session_id or "").strip()
    session_id = supplied_session or f"session_{uuid4().hex}"
    user_id = (browser_user_id.user_id if isinstance(browser_user_id, Identity)
               else browser_user_id or f"user_{uuid4().hex}")
    if isinstance(browser_user_id, Identity):
        await claim_session(user_id, session_id, create=not bool(supplied_session))
    memory_client = get_memory_client()

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
        try:
            stored_goal = await graph_repo.get_travel_goal(session_id, user_id)
        except Exception:
            logger.warning("Could not load this session's travel goal", exc_info=True)
            stored_goal = None
        if stored_goal:
            try:
                current_goal = TravelGoal.model_validate(stored_goal)
            except Exception:
                logger.warning("Ignoring invalid stored travel goal", exc_info=True)
    if current_goal is not None:
        current_goal = recall_recommendations(current_goal, recent_transcript)

    _, last_reply = _last_assistant_offer(recent_transcript)
    decision = await understand_turn(request.message, current_goal, recent_transcript)
    more_plans = _requests_more_plans(request.message, current_goal, last_reply)
    if more_plans and decision.action != "plan":
        decision = replace(decision, action="plan")
    vehicle = None
    if graph_repo is not None:
        try:
            vehicle = explicit_vehicle_from_vocabulary(
                request.message, await graph_repo.get_vehicle_names())
        except Exception:
            logger.warning("Could not check explicit vehicle against graph vocabulary", exc_info=True)
        if vehicle and decision.updates.get("transport") != vehicle:
            updates = {**decision.updates, "transport": vehicle}
            clauses = decision.inferred_clauses
            if len(clauses) == 1 and clauses[0].get("clause") == request.message:
                clauses = [{**clauses[0], "kind": "request", "scope": "trip",
                            "dimensions": list(dict.fromkeys(
                                [*(clauses[0].get("dimensions") or []), "transport"]))}]
            decision = replace(decision, updates=updates, inferred_clauses=clauses)
    if (vehicle and current_goal and current_goal.last_proposed_routes
            and current_goal.transport != vehicle and decision.action == "chat"):
        decision = replace(decision, action="plan")
    # The turn classifier sees the previous assistant reply. It must not
    # promote names merely mentioned there into user-selected destinations.
    # Implicit/ordinal choices are resolved against the actual last offer
    # below, after this validation.
    proposed_names = decision.updates.get("destination_names")
    if isinstance(proposed_names, list) and proposed_names:
        user_named = [name for name in proposed_names if isinstance(name, str)
                      and _explicitly_mentions_destination(request.message, name)
                      and not (vehicle and
                               TripGraphRepository._destination_lookup_keys(name)
                               == TripGraphRepository._destination_lookup_keys(vehicle))]
        if decision.action != "plan" and not (
                _has_plan_request(request.message) or re.search(
                    r"(?:مقصد.{0,35}(?:باشه|انتخاب)|می ?خوام برم|می ?ریم|بریم)",
                    request.message.replace("‌", " "))):
            user_named = []
        if len(user_named) != len(proposed_names):
            cleaned = dict(decision.updates)
            if user_named or more_plans:
                cleaned["destination_names"] = user_named
            else:
                cleaned.pop("destination_names", None)
            decision = replace(decision, updates=cleaned)
    elif more_plans and current_goal and current_goal.destination_names:
        decision = replace(decision, updates={**decision.updates, "destination_names": []})
    pending = (current_goal.pending_offer if current_goal and
               current_goal.pending_offer and last_reply.endswith(
                   current_goal.pending_offer.source_reply) else None)
    # The classifier's selected_offer_names is a hint, not a reason to skip
    # grounding an ordinal choice in the last assistant reply. The model
    # sometimes sets it even for "the second option".
    offered_choice = await _resolve_offered_option(
        request.message, current_goal, recent_transcript, graph_repo)
    accepted_offer: list[str] = []
    explicit_selection = decision.updates.get("destination_names") or []
    explicitly_named = any(_explicitly_mentions_destination(request.message, name)
                           for name in explicit_selection) if isinstance(explicit_selection, list) else False
    unresolved_choice = bool((pending is not None or _offers_plan(last_reply))
                             and not offered_choice
                             and not explicitly_named
                             and _asks_for_one_of_offered_options(request.message))
    was_offered = False
    if (not explicitly_named and not offered_choice and
            (decision.offer_response == "accept" or _is_bare_confirmation(request.message))):
        if pending is not None:
            was_offered = True
            selected = [name for name in pending.destination_names
                        if name in decision.selected_offer_names
                        and not _is_bare_confirmation(request.message)]
            accepted_offer = (pending.destination_names if pending.action == "plan"
                              and not pending.unverified_names and not pending.incomplete
                              else selected[:1])
        else:
            previous_user, reply = _last_assistant_offer(recent_transcript)
            pending = await _capture_pending_offer(previous_user, reply, graph_repo, force=True)
            if pending is not None and pending.action == "plan" and not (
                    pending.unverified_names or pending.incomplete):
                accepted_offer = pending.destination_names
            was_offered = pending is not None or _offers_plan(reply)
            was_offered = was_offered or decision.offer_response == "accept"
    if unresolved_choice or (was_offered and not accepted_offer):
        if pending is None and unresolved_choice:
            previous_user, reply = _last_assistant_offer(recent_transcript)
            pending = await _capture_pending_offer(previous_user, reply, graph_repo, force=True)
        options = pending.offered_names if pending is not None else []
        reply = ("برای اینکه برنامهٔ مقصد اشتباهی ندهم، "
                 + (f"از میان «{'، '.join(options)}» بگو کدام مقصد را می‌خواهی؛ "
                    "یا صریحاً بگو ترکیبی از کدام‌ها را برنامه‌ریزی کنم."
                    if options else
                    "نام مقصد یا مقصدهایی که می‌خواهی برایشان برنامه بچینم را بگو."))
        if pending is not None and (pending.unverified_names or pending.incomplete):
            reply += (" بعضی از گزینه‌های پیشنهادی قبلی در داده‌های مقصد تأیید نشدند؛ "
                      "برای آن‌ها مسیر یا امکان کمپ را قطعی فرض نمی‌کنم.")
        clarification = infer_and_merge(current_goal or TravelGoal(), request.message,
                                         replace(decision, action="answer"))
        if pending is not None:
            # The next "second option" must still refer to the original
            # offered order, including unverified names omitted by graph
            # search. Link the pending offer to this clarification reply.
            clarification.pending_offer = pending.model_copy(
                update={"source_reply": reply[-1500:]})
        clarification = await evaluate_response(
            clarification, reply, use_llm=bool(decision.inferred_clauses))
        await _save_direct_exchange(memory_client, session_id, user_id, request.message, reply)
        await _save_answer_goal(graph_repo, clarification, session_id, user_id, request.message)
        await _mark_completed_turn(browser_user_id, driver, user_id, session_id)
        return ChatResponse(reply=reply, session_id=session_id, user_id=user_id)
    if offered_choice:
        # A deictic selection is a new plan, even if the classifier called it
        # an informational follow-up or missed the destination update.
        decision = replace(decision, action="plan",
                           updates={**decision.updates, "destination_names": [offered_choice]})
    elif accepted_offer:
        decision = replace(decision, action="plan",
                           updates={**decision.updates, "destination_names": accepted_offer})

    # Infer from this user message, merge with the session's earlier goals,
    # then apply explicit structured changes. Assistant suggestions are not
    # promoted to user requirements until the user accepts them.
    working_goal = (apply_goal_updates(
        infer_and_merge(current_goal or TravelGoal(), request.message, decision),
        decision.updates) if current_goal is not None or (
            decision.action != "chat" and not _is_memory_question(request.message)) else None)

    if decision.action == "answer" and _is_memory_question(request.message):
        reply = await summarize_user_memory(memory_client, graph_repo, user_id, session_id, current_goal)
        await _save_direct_exchange(memory_client, session_id, user_id, request.message, reply)
        if working_goal is not None:
            working_goal = await evaluate_response(
                working_goal, reply, use_llm=bool(decision.inferred_clauses))
        await _save_answer_goal(graph_repo, working_goal, session_id, user_id, request.message)
        await _mark_completed_turn(browser_user_id, driver, user_id, session_id)
        return ChatResponse(reply=reply, session_id=session_id, user_id=user_id)

    vehicle_reply = (await _answer_vehicle_question(request.message, current_goal, graph_repo)
                     if decision.action == "answer" else None)
    if vehicle_reply:
        await _save_direct_exchange(memory_client, session_id, user_id, request.message, vehicle_reply)
        if working_goal is not None:
            working_goal = await evaluate_response(
                working_goal, vehicle_reply, use_llm=bool(decision.inferred_clauses))
        await _save_answer_goal(graph_repo, working_goal, session_id, user_id, request.message)
        await _mark_completed_turn(browser_user_id, driver, user_id, session_id)
        return ChatResponse(reply=vehicle_reply, session_id=session_id, user_id=user_id)

    if (decision.action == "answer" and _asks_why_plan(request.message)
            and working_goal is not None and graph_repo is not None):
        names_by_route = working_goal.last_proposed_routes or working_goal.recommended_routes[-2:]
        if names_by_route:
            routes = [{"stops": [{"name": name} for name in names]} for names in names_by_route]
            reply = await explain_selected_plans(working_goal, routes, graph_repo, memory_client, user_id,
                                                 writer=write_explanation)
            if not reply:
                reply = "برای این مسیرها هنوز جزئیات کافی ندارم که دلیلی دقیق و قابل‌اتکا بگم."
            await _save_direct_exchange(memory_client, session_id, user_id, request.message, reply)
            if working_goal is not None:
                working_goal = await evaluate_response(
                    working_goal, reply, use_llm=bool(decision.inferred_clauses))
            await _save_answer_goal(graph_repo, working_goal, session_id, user_id, request.message)
            await _mark_completed_turn(browser_user_id, driver, user_id, session_id)
            return ChatResponse(reply=reply, session_id=session_id, user_id=user_id)

    # Goal facts come from understand_turn's LLM, not a second regex parser.
    # Existing state is retained when the model supplies no new facts.
    travel_goal = working_goal
    if travel_goal is None and decision.action != "chat":
        travel_goal = TravelGoal(objective=request.message, semantic_query=request.message)
    elif travel_goal is not None and decision.action == "plan":
        resolved_request = (f"برنامه سفر برای {' و '.join(accepted_offer)}"
                            if accepted_offer else request.message)
        travel_goal.objective = current_trip_query(travel_goal, resolved_request)
        travel_goal.semantic_query = travel_goal.objective
        travel_goal.goal_coverage = []
        travel_goal.last_observation_tool = None
    previous_recommendations: dict[str, set[str]] = {"ids": set(), "names": set()}
    if graph_repo is not None and decision.action == "plan":
        try:
            previous_recommendations = await graph_repo.get_previous_recommendations(user_id, session_id)
        except Exception:
            logger.warning("Could not load previous recommendations", exc_info=True)
        try:
            prior_features = await graph_repo.get_recent_session_features(user_id, session_id)
            if not prior_features:
                # Older sessions stored TravelGoal before REQUESTED_FOR_TRIP
                # existed. Read only their explicit structured fields.
                older = await graph_repo.get_user_history(user_id, exclude_session_id=session_id)
                if older.get("recent_goals"):
                    prior_features = insights_from_previous_goal(older["recent_goals"][0])
            if travel_goal is not None:
                travel_goal = with_prior_insights(travel_goal, prior_features)
        except Exception:
            logger.warning("Could not load previous trip insights", exc_info=True)
    if previous_recommendations["names"]:
        # A named follow-up is a deliberate revisit, never an excluded repeat.
        named = {name for name in previous_recommendations["names"]
                 if name == offered_choice or name in accepted_offer
                 or _explicitly_mentions_destination(request.message, name)}
        previous_recommendations["names"] -= named
    if decision.action != "chat" and travel_goal is not None:
        preferences = []
        features = []
        if driver is not None:
            try:
                from src.memory.preference_graph import available_features, user_preferences
                preferences = await user_preferences(driver, user_id)
                features = await available_features(driver)
            except Exception:
                logger.warning("Could not load ontology and structured goal preferences", exc_info=True)
        # A factual question alone is not permission to turn the mentioned
        # place into a trip requirement. Only a plan or explicit update
        # decomposes/changes planning subgoals.
        if decision.action == "plan" or decision.updates:
            travel_goal = await design_goals(
                travel_goal, (travel_goal.semantic_query if accepted_offer else request.message),
                recent_transcript, preferences, features)
        travel_goal = retire_replaced_subgoals(
            current_goal, travel_goal, decision.updates, features)
        # Repair a missing model target only with a destination that the
        # user actually named AND Neo4j recognizes; never infer food or rest
        # from a city's name in the generated subgoal.
        if graph_repo is not None and any(
                g.source == "explicit" and g.evidence_type == "unverified"
                and "بازدید" in g.name for g in travel_goal.subgoals):
            try:
                named_places = await graph_repo.find_destinations_mentioned_in_text(request.message)
                for subgoal in travel_goal.subgoals:
                    if (subgoal.source == "explicit" and subgoal.evidence_type == "unverified"
                            and "بازدید" in subgoal.name):
                        matched = next((name for name in named_places if name in subgoal.name), None)
                        if matched:
                            subgoal.evidence_type = "destination"
                            subgoal.target = matched
            except Exception:
                logger.warning("Could not resolve explicitly named goal destination", exc_info=True)
    if travel_goal is not None:
        travel_goal.turn_id = str(uuid4())
        travel_goal.revision_cause = "user_goal_update"
        travel_goal.last_user_request = request.message
        travel_goal.last_turn_action = decision.action
        travel_goal.updated_at = datetime.now(timezone.utc).isoformat()
        if decision.action == "plan":
            travel_goal.last_itineraries = []
        if decision.action != "plan" and travel_goal.last_itineraries:
            travel_goal.goal_coverage = await evaluate_plan(
                travel_goal, travel_goal.last_itineraries, graph_repo)
    if graph_repo is not None and travel_goal is not None:
        revision = await graph_repo.upsert_travel_goal(
            session_id,
            user_id,
            travel_goal.model_dump(),
        )
        if revision is not None:
            travel_goal.revision = revision
        try:
            await graph_repo.sync_session_features(user_id, session_id, travel_goal.model_dump())
        except Exception:
            logger.warning("Could not link session travel insights to user", exc_info=True)

    deps = AgentDeps.create(
        memory=memory_client,
        graph=graph_repo,
        session_id=session_id,
        user_id=user_id,
        current_query=request.message,
        travel_goal=travel_goal,
        turn_action=decision.action,
        maps=get_neshan_client(),
        previous_recommendations=previous_recommendations,
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

        # Complementary semantic retrieval is guaranteed for planning turns:
        # the model can then use structured graph tools to check hard facts,
        # and can call semantic search again with a narrower query if needed.
        await _preload_semantic_candidates(deps)

        result = await agent.run(request.message, deps=deps, message_history=message_history)
        reply_text = result.output if hasattr(result, "output") else str(result.data)
        if decision.action == "answer":
            access_answer = await _verified_access_answer(request.message, travel_goal, graph_repo)
            if access_answer is not None:
                reply_text = access_answer

        itinerary = None
        if deps.itinerary_result and deps.itinerary_result.get("used_real_routing"):
            try:
                itinerary = Itinerary.model_validate(deps.itinerary_result)
            except Exception:
                logger.warning("Could not serialize itinerary_result", exc_info=True)
        elif decision.action == "plan":
            try:
                fallback_stops = linking.extract_finalized_destination_coords(result.new_messages())
                # Use the same road-routing tool as a normal plan, not a
                # straight-line one-way estimate attached after the reply.
                if fallback_stops:
                    fallback = await build_trip_map(
                        type("Context", (), {"deps": deps})(), fallback_stops,
                        origin={"name": travel_goal.origin} if travel_goal else None,
                        round_trip=bool(travel_goal and travel_goal.duration and "روز" in travel_goal.duration),
                    )
                    if fallback.get("used_real_routing"):
                        itinerary = Itinerary.model_validate(fallback)
            except Exception:
                logger.warning("Could not build fallback itinerary", exc_info=True)

        map_itineraries: list[dict] = []
        # A request based only on memory has no time budget. Offer a clearly
        # stated one-day *assumption* for a verified nearby starting plan;
        # don't write that assumed duration back as something the user said.
        assumed_day_trip = bool(decision.action == "plan" and travel_goal is not None
                                and not travel_goal.duration and travel_goal.origin == "تهران"
                                and not travel_goal.region and not travel_goal.destination_names
                                and _is_memory_based_plan(request.message))
        screening_goal = (travel_goal.model_copy(update={"duration": "یک روز"})
                          if assumed_day_trip else travel_goal)
        checked = (await screen_short_trip(
            screening_goal, deps.maps, result.new_messages(), reply_text, itinerary,
            description_formatter=lambda rows: format_descriptions(rows, screening_goal),
            graph=graph_repo,
            map_itineraries=map_itineraries,
            avoid_names=previous_recommendations["names"],
        ) if decision.action == "plan" else None)
        if checked is not None:
            reply_text, verified_route = checked
            if verified_route is not None:
                itinerary = Itinerary.model_validate(verified_route)
            # Alternatives aren't a single combined route. Retain map data
            # only when exactly the verified stop(s) were finalized.
            if (itinerary is None or any(s.name not in reply_text for s in itinerary.stops)
                    or (verified_route is None and reply_text.count("\n#### ") > 1)):
                itinerary = None
            elif itinerary and verified_route is None:
                try:
                    revised = await build_trip_map(
                        type("Context", (), {"deps": deps})(),
                        [{"name": s.name, "latitude": s.latitude, "longitude": s.longitude} for s in itinerary.stops],
                        origin=itinerary.origin, round_trip=True,
                    )
                    itinerary = Itinerary.model_validate(revised) if revised.get("used_real_routing") else None
                except Exception:
                    logger.warning("Could not validate return leg", exc_info=True)
                    itinerary = None
        elif itinerary is not None and decision.action == "plan":
            reply_text = ground_route_text(reply_text, itinerary)
        if (decision.action == "plan" and travel_goal is not None
                and travel_goal.destination_names):
            proposed_routes = map_itineraries or ([itinerary.model_dump()] if itinerary else [])
            requested = set(travel_goal.destination_names)
            wrong_route = bool(map_itineraries or itinerary) and not all(
                requested.issubset({stop.get("name") for stop in route.get("stops", [])})
                for route in proposed_routes)
            missing_named_place = not (map_itineraries or itinerary) and not all(
                _explicitly_mentions_destination(reply_text, name) for name in requested)
            if wrong_route or missing_named_place:
                # An LLM response must not substitute a different itinerary
                # when the user retained a named destination.
                map_itineraries.clear()
                itinerary = None
        if (decision.action == "plan" and travel_goal is not None
                and travel_goal.destination_names and not map_itineraries and itinerary is None):
            # A selected graph place can be described without a verified map
            # route. Never reuse the draft's unverified facilities or a
            # substituted destination in this case.
            reply_text = await _unrouted_selected_plan(travel_goal, graph_repo)
        if decision.action == "plan" and travel_goal is not None:
            routes = map_itineraries or ([itinerary.model_dump() | {"used_real_routing": True}]
                                         if itinerary is not None else [])
            travel_goal.last_itineraries = routes[:4]
            coverage = await evaluate_plan(travel_goal, routes, graph_repo)
            feedback = replan_feedback(travel_goal, coverage)
            if (feedback and deps.maps is not None and graph_repo is not None
                    and not (travel_goal.destination_names and not routes)):
                if deps.observed_tools:
                    refined = await decompose_unmet_goal(travel_goal, coverage, deps.observed_tools)
                    if refined is not travel_goal:
                        travel_goal.subgoals = refined.subgoals
                        travel_goal.goal_coverage = refined.goal_coverage
                        # Preserve the original unsatisfied goal and user constraints.
                        # The revision is visible to the next agent.run and auditable.
                        try:
                            travel_goal.revision_cause = "subgoal_decomposition"
                            travel_goal.revision = await graph_repo.upsert_travel_goal(
                                session_id, user_id, travel_goal.model_dump())
                        except Exception:
                            logger.warning("Could not persist decomposed goals", exc_info=True)
                        feedback += ("\nGoal revision: " + json.dumps(
                            [g.model_dump() for g in travel_goal.subgoals if g.parent_id],
                            ensure_ascii=False))
                # At most one extra planning attempt. Retain the first plan unless
                # the revised, screened route actually improves high-priority coverage.
                try:
                    old_route = deps.itinerary_result
                    deps.itinerary_result = None
                    revision_result = await agent.run(
                        feedback, deps=deps, message_history=result.all_messages())
                    revised_text = revision_result.output
                    revised_route = (Itinerary.model_validate(deps.itinerary_result)
                                     if deps.itinerary_result and deps.itinerary_result.get("used_real_routing")
                                     else None)
                    revised_maps: list[dict] = []
                    revised_check = await screen_short_trip(
                        screening_goal, deps.maps, revision_result.new_messages(), revised_text,
                        revised_route, description_formatter=lambda rows: format_descriptions(rows, screening_goal),
                        graph=graph_repo, map_itineraries=revised_maps,
                        avoid_names=previous_recommendations["names"])
                    if revised_check is not None:
                        revised_text, verified = revised_check
                        revised_route = Itinerary.model_validate(verified) if verified else None
                        if revised_route is not None and any(
                                s.name not in revised_text for s in revised_route.stops):
                            revised_route = None
                    elif revised_route is not None:
                        revised_text = ground_route_text(revised_text, revised_route)
                    revised_routes = revised_maps or (
                        [revised_route.model_dump() | {"used_real_routing": True}] if revised_route else [])
                    revised_coverage = await evaluate_plan(travel_goal, revised_routes, graph_repo)
                    old_score = sum(c.status == "satisfied" for c in coverage)
                    new_score = sum(c.status == "satisfied" for c in revised_coverage)
                    requested = set(travel_goal.destination_names)
                    names_in_revised_route = [
                        {stop.get("name") for stop in route.get("stops", [])}
                        for route in revised_routes]
                    if (new_score > old_score
                            and (not requested or any(requested.issubset(names)
                                                      for names in names_in_revised_route))):
                        result, reply_text, itinerary = revision_result, revised_text, revised_route
                        map_itineraries, coverage = revised_maps, revised_coverage
                        travel_goal.last_itineraries = revised_routes[:4]
                    else:
                        deps.itinerary_result = old_route
                        if len(travel_goal.subgoals) != len(coverage):
                            coverage = await evaluate_plan(travel_goal, routes, graph_repo)
                except Exception:
                    logger.warning("Bounded goal replan failed; retaining screened plan", exc_info=True)
            travel_goal.goal_coverage = coverage
        if assumed_day_trip:
            remembered = _previous_feature_concept(travel_goal.prior_insights)
            if remembered:
                intro = f"از درخواست سفر قبلی‌ات {remembered} را یادم هست؛ این بار هم از آن به عنوان سرنخ استفاده کردم. "
            else:
                saved_prefs = []
                if memory_client is not None:
                    try:
                        saved_prefs = await memory_client.long_term.get_preferences_for(user_id)
                    except Exception:
                        logger.warning("Could not read preferences for planning introduction", exc_info=True)
                if saved_prefs:
                    intro = f"یادم هست که {saved_prefs[0].preference}. "
                elif memory_client is None:
                    intro = "الان حافظهٔ سفرهای قبلی‌ات در دسترسم نیست؛ بر اساس حدس درباره‌ات برنامه نمی‌ریزم. "
                else:
                    intro = "از سفرهای قبلی‌ات هنوز سلیقهٔ تأییدشده‌ای پیدا نکردم؛ این یک پیشنهاد اولیه است. "
            reply_text = (intro + "چون مدت سفر را مشخص نکردی، برای شروع یک سفر یک‌روزه از تهران فرض کردم. "
                          "اگر وقت بیشتری داری، می‌تونیم برنامه را تغییر بدیم.\n\n" + reply_text)

        shown_names: list[str] = []
        if decision.action == "plan" and travel_goal is not None:
            proposed = map_itineraries or ([itinerary.model_dump()] if itinerary is not None else [])
            shown_names = list(dict.fromkeys(stop.get("name") for route in proposed
                                              for stop in route.get("stops", []) if stop.get("name")))
            named_routes: list[dict] = []
            try:
                # A list of independent suggestions need not have a map route.
                # Recover only graph destinations actually named in the final
                # answer, rather than using all candidates the agent searched.
                names = (await graph_repo.find_destinations_mentioned_in_text(reply_text)
                         if graph_repo is not None and not proposed and not (
                             travel_goal.destination_names and
                             "مقصد انتخابی شما" in reply_text) and not any(
                             phrase in reply_text for phrase in (
                                 "نتوانستم", "پیدا نکردم", "سازگار نیست",
                                 "فعلاً اطلاعات مقصد", "مقصدهای انتخابی شما")
                          ) else [])
                if not proposed:
                    shown_names = names
                named_routes = ([{"stops": [{"name": name} for name in accepted_offer]}]
                                if len(accepted_offer) > 1 and all(
                                    name in names for name in accepted_offer)
                                else [{"stops": [{"name": name}]} for name in names])
                explanation_routes = proposed or named_routes
                explanation = await explain_selected_plans(
                    travel_goal, explanation_routes, graph_repo, memory_client, user_id,
                    writer=write_explanation,
                )
                if explanation:
                    reply_text = _compose_plan_reply(
                        reply_text, explanation,
                        first_turn=current_goal is None and not message_history)
                elif current_goal is None and not message_history:
                    reply_text = _compose_plan_reply(reply_text, "", first_turn=True)
            except Exception:
                logger.warning("Could not explain plan personalization", exc_info=True)
            if proposed or shown_names:
                travel_goal = remember_recommended_routes(
                    travel_goal, proposed or named_routes
                    or [{"stops": [{"name": name}]} for name in shown_names]
                )
                if graph_repo is not None:
                    try:
                        travel_goal.revision_cause = "routes_proposed"
                        revision = await graph_repo.upsert_travel_goal(session_id, user_id, travel_goal.model_dump())
                        if revision is not None:
                            travel_goal.revision = revision
                    except Exception:
                        logger.warning("Could not remember proposed routes for this session", exc_info=True)
            elif graph_repo is not None and travel_goal.goal_coverage:
                travel_goal.revision_cause = "plan_coverage"
                revision = await graph_repo.upsert_travel_goal(session_id, user_id, travel_goal.model_dump())
                if revision is not None:
                    travel_goal.revision = revision

        if shown_names and graph_repo is not None:
            try:
                await graph_repo.record_shown_destinations(user_id, session_id, shown_names)
            except Exception:
                logger.warning("Could not remember shown destinations", exc_info=True)

        if travel_goal is not None and graph_repo is not None:
            pending = await _capture_pending_offer(request.message, reply_text, graph_repo)
            travel_goal.pending_offer = pending
            if pending is not None or current_goal and current_goal.pending_offer:
                try:
                    travel_goal.revision_cause = "assistant_offer"
                    revision = await graph_repo.upsert_travel_goal(
                        session_id, user_id, travel_goal.model_dump())
                    if revision is not None:
                        travel_goal.revision = revision
                except Exception:
                    logger.warning("Could not persist pending assistant offer", exc_info=True)

        if travel_goal is not None and decision.action != "chat":
            verified = [stop.name for route in map_itineraries
                        for stop in Itinerary.model_validate(route).stops]
            if itinerary is not None:
                verified.extend(stop.name for stop in itinerary.stops)
            travel_goal = await evaluate_response(
                travel_goal, reply_text, verified_route_stops=list(dict.fromkeys(verified)),
                use_llm=bool(decision.inferred_clauses))
            if graph_repo is not None and decision.inferred_clauses:
                try:
                    travel_goal.revision_cause = "response_evaluation"
                    revision = await graph_repo.upsert_travel_goal(
                        session_id, user_id, travel_goal.model_dump())
                    if revision is not None:
                        travel_goal.revision = revision
                except Exception:
                    logger.warning("Could not persist conversational goal evaluation", exc_info=True)

        if (graph_repo is not None and user_message is not None
                and travel_goal is not None and travel_goal.turn_id
                and hasattr(graph_repo, "link_goal_revision_message")):
            try:
                await graph_repo.link_goal_revision_message(
                    session_id, travel_goal.turn_id, str(user_message.id))
            except Exception:
                logger.warning("Could not link goal revisions to user message", exc_info=True)

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
                if graph_repo is not None and travel_goal is not None and travel_goal.turn_id:
                    try:
                        await graph_repo.link_goal_revision_trace(
                            session_id, travel_goal.turn_id, str(trace.id))
                    except Exception:
                        logger.warning("Could not link goal revisions to trace", exc_info=True)

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

        await _mark_completed_turn(browser_user_id, driver, user_id, session_id)
        return ChatResponse(reply=reply_text, session_id=session_id, user_id=user_id,
                            itinerary=itinerary,
                            itineraries=[Itinerary.model_validate(route) for route in map_itineraries])
    except ModelAPIError as e:
        logger.warning("LLM provider request failed: %s", e)
        raise HTTPException(
            status_code=504,
            detail="سرویس هوش مصنوعی موقتاً دیر پاسخ می‌دهد. لطفاً چند لحظه دیگر دوباره تلاش کنید.",
        ) from e
    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
