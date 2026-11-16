"""Understand the latest turn before applying any itinerary-only safeguards."""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from pydantic_ai import Agent

from src.agent.agent import _build_model
from src.agent.goals import PendingOffer, TravelGoal
from src.config import get_settings

logger = logging.getLogger(__name__)

_UPDATABLE = {
    "origin", "destination_names", "region", "duration", "travel_dates", "season",
    "companions", "group_size", "categories", "trip_types", "moods", "activities", "transport", "accommodation",
    "budget", "must_haves", "avoid", "constraints",
}


@dataclass(frozen=True)
class TurnDecision:
    action: str = "answer"  # plan | answer | chat
    updates: dict[str, Any] = field(default_factory=dict)
    offer_response: Literal["accept", "decline", "none"] = "none"
    selected_offer_names: list[str] = field(default_factory=list)
    inferred_clauses: list[dict[str, Any]] = field(default_factory=list)
    merge_operations: list[dict[str, Any]] = field(default_factory=list)


def apply_goal_updates(goal: TravelGoal | None, updates: dict[str, Any]) -> TravelGoal | None:
    """Apply valid explicit facts without losing other fields to one bad value."""
    if not updates:
        return goal
    state = (goal or TravelGoal()).model_dump()
    for key, value in updates.items():
        if key not in _UPDATABLE or (key == "origin" and isinstance(value, str)
                                    and value in {"من", "ما", "تو", "شما", "خودم"}):
            continue
        if key == "duration" and isinstance(value, int) and not isinstance(value, bool) and value > 0:
            value = f"{value} روز"
        try:
            TravelGoal.model_validate({**state, key: value})
        except Exception:
            logger.warning("Ignoring invalid travel-goal field %s", key)
            continue
        state[key] = value
    if "avoid" in updates and isinstance(state.get("avoid"), list):
        excluded = {str(value).strip() for value in state["avoid"]}
        for key in ("categories", "trip_types", "moods", "activities",
                    "accommodation", "must_haves"):
            state[key] = [value for value in state.get(key, [])
                          if value not in excluded]
    return TravelGoal.model_validate(state)


def current_trip_query(goal: TravelGoal, latest_request: str) -> str:
    """Search/planning context from active constraints, not a concatenated chat log."""
    parts = [latest_request]
    fields = (
        ("مقصد", goal.destination_names), ("منطقه", [goal.region] if goal.region else []),
        ("مبدأ", [goal.origin] if goal.origin != "تهران" else []),
        ("مدت", [goal.duration] if goal.duration else []),
        ("فصل", [goal.season] if goal.season else []),
        ("تاریخ", [goal.travel_dates] if goal.travel_dates else []),
        ("همراهان", goal.companions), ("دسته", goal.categories),
        ("نوع سفر", goal.trip_types), ("سبک", goal.moods),
        ("فعالیت", goal.activities), ("اقامت", goal.accommodation),
        ("وسیله", [goal.transport] if goal.transport else []),
        ("بودجه", [goal.budget] if goal.budget else []),
        ("نیازها", goal.must_haves), ("محدودیت", goal.constraints),
        ("پرهیز", goal.avoid),
    )
    for label, values in fields:
        clean = [value for value in values if isinstance(value, str) and value.strip()
                 and (label == "پرهیز" or value not in goal.avoid)]
        if clean:
            parts.append(f"{label}: {'، '.join(clean[:4])}")
    if goal.group_size:
        parts.append(f"تعداد نفرات: {goal.group_size}")
    return "؛ ".join(parts)[:1200]


def explicit_vehicle_from_vocabulary(message: str, names: list[str]) -> str | None:
    """Recover an explicit possessed vehicle from graph-backed vocabulary.

    Mere mention (e.g. 'is off-road necessary?') is not ownership or selection.
    Normalize Persian spelling variations, but never infer a new vehicle name.
    """
    def clean(value: str) -> str:
        text = unicodedata.normalize("NFKC", value).translate(
            str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک",
                           "آ": "ا", "أ": "ا", "إ": "ا", "\u200c": " "}))
        return " ".join(re.sub(r"[^\w\s]", " ", text).split()).casefold()

    text = clean(message)
    for name in names:
        token = clean(name)
        if not token:
            continue
        for match in re.finditer(r"(?<!\w)" + re.escape(token) + r"(?!\w)", text):
            around = text[max(0, match.start() - 20):match.end() + 22]
            if re.search(r"(?:نمی ?خواه|ندار|نیست|لازم|نیاز|کدوم|کدام|؟)", around):
                continue
            if re.search(r"(?:دارم|داریم|دارند|دارید|با\s+" + re.escape(token)
                         + r"\s+(?:می ?ریم|بریم|سفر))", around):
                return name
    return None


def _fallback_action(message: str, has_history: bool) -> str:
    """When classification fails, preserve conversational follow-ups."""
    text = message.strip().replace("‌", " ")
    if (("؟" in text or "?" in text or re.search(r"^(?:کدوم|کدام|کجا|چطور|چگونه|چیه|چی|آیا)", text))
            and not re.search(r"برنامه\s*(?:بده|بساز|بچین)|پیشنهاد\s*(?:بده|کن)", text)):
        return "answer"
    if re.search(r"(?:برنامه|مسیر|گزینه|پیشنهاد بده|معرفی کن|میخوام|می خواهم|بریم|برویم)", text):
        return "plan"
    return "answer" if has_history else "plan"


@lru_cache
def _turn_agent() -> Agent:
    return Agent(
        _build_model(get_settings()),
        system_prompt=(
            "You interpret the LATEST message in a Persian travel-planning conversation. "
            "Respond with ONLY JSON: {\"action\":\"plan|answer|chat\",\"updates\":{},"
            "\"offer_response\":\"accept|decline|none\",\"selected_offer_names\":[],"
            "\"inferred_clauses\":[{\"clause\":\"verbatim user span\",\"kind\":\"question|request|offer|suggestion\","
            "\"summary\":\"one sentence\",\"scope\":\"trip|turn\","
            "\"dimensions\":[\"an updated goal field\"]}],"
            "\"merge_operations\":[{\"operation\":\"keep|replace|combine\","
            "\"old_ids\":[\"existing conversational goal id\"],\"new_index\":0}]}. "
            "Implement an OnGoal-style dialogue goal pipeline: infer every "
            "distinct user question, request, offer or suggestion from exact "
            "clauses in the LATEST message (not assistant text); a question "
            "about a place is turn-scoped, NOT an accepted trip destination. "
            "An explicit travel preference/constraint or acceptance of an "
            "assistant proposal is trip-scoped. Dimensions must refer ONLY "
            "to fields in updates; otherwise use []. Compare each new clause "
            "to current_goal.conversational_goals: keep unique old goals, "
            "replace an old goal ONLY when the user explicitly contradicts "
            "or removes it, combine similar compatible goals, otherwise add "
            "the new goal. Never treat an assistant suggestion as a user's "
            "goal before the user accepts it. Old unrelated constraints stay. "
            "When current_goal.pending_offer matches the last assistant message, "
            "resolve any acceptance, rejection or selection against THAT pending "
            "offer, not an older conversation or a fresh search. Accepting an offer "
            "to plan a joint trip means ALL its destinations; accepting independent "
            "options requires the selected option, or clarification if none is "
            "specified. selected_offer_names must be a subset of the pending "
            "offer's destinations, in the offered order. Do not mistake a factual "
            "question about options for acceptance. "
            "You decide whether the current user's goal or earlier proposed itinerary "
            "needs to be revised. Use 'plan' when they request a new itinerary, "
            "ask what other plans/itineraries you have (دیگه برنامه چی داری), OR when "
            "they say 'what else do you have?' after an itinerary in THIS session; "
            "that short question refers to other trip plans, not small talk. "
            "new explicit requirements make the previous proposal unsuitable (for example "
            "a hiking itinerary followed by family, toilets and little walking). "
            "Use 'answer' when they only request information about a place, comparison, "
            "weather, facilities or access without asking to change their trip. "
            "If 'answer' includes a NEW constraint for THIS trip, still put it in updates; "
            "the goal graph will be revised even without a new itinerary. "
            "To answer a factual question, prefer the graph destination-details tool if "
            "the graph has evidence; otherwise qualify general knowledge, never fabricate "
            "current opening hours or amenities. "
            "A previous goal duration alone does not imply they requested another plan. "
            "Use 'chat' for greetings, non-planning small talk or off-topic messages; "
            "do not create a travel goal for off-topic chat. "
            "The current_goal contains the session's earlier constraints; do not drop them. "
            "Put ONLY explicit changes in updates, using field names from current_goal; "
            "for list fields give the COMPLETE new effective list, keeping earlier "
            "requirements unless the user explicitly removes them. "
            "For explicit negated travel preferences add the rejected thing "
            "to avoid (including all earlier avoid items), even if the positive "
            "preference was never set; clearing a positive field alone loses "
            "the user's prohibition on future turns. "
            "such as duration, origin, region, destination_names, companions, moods, avoid, "
            "categories, and trip_types. Extract stated travel mood (e.g. هیجان/ماجراجویی "
            "-> moods=['ماجراجویانه']), party (with friends -> companions=['دوستان'], "
            "با پارتنرم -> companions=['پارتنر']), "
            "For an explicit party count use the integer group_size "
            "(e.g. دو نفره -> group_size=2), not just companions=['دو نفر']. "
            "category (طبیعت -> categories=['طبیعت']), or trip type "
            "(camping -> trip_types=['کمپ']) when clearly requested; don't infer it "
            "from a destination name or a previous assistant suggestion. "
            "Likewise extract an explicit season or transport when given. "
            "If the user says they HAVE a vehicle (e.g. افرود داریم), update "
            "transport with the vehicle type; it is NOT a destination name. "
            "A question about which option is quieter is NOT a new mood preference. "
            "If the user clearly selects a destination, update destination_names; if they "
            "ask about a destination without selecting it, do not overwrite the chosen plan. "
            "Resolve short references from the LAST assistant-offered "
            "options, not a searched candidate or another session. Keep earlier "
            "explicit season, group size, companions and newly requested activities "
            "(e.g. پیاده‌روی) unless corrected. If asking for other options instead "
            "of the previously selected place, clear destination_names for that "
            "new search; do not silently retain the old selected place. "
            "A confirmation after an offer accepts that offer, not a newly "
            "searched destination. "
            "'میخوام برم آبشار برجی کلا' selects that single place; never plan other destinations instead. "
            "'برام یه برنامه بریز تو پاییز برم یه جا کمپ کنم' does NOT name a destination: "
            "'برم' means 'I go', and 'یه جا' is an unspecified place. Return no "
            "destination_names update in that case, even if a graph happens to contain "
            "a place with the same spelling as a verb. Base named-place selection on "
            "the meaning of the whole sentence, not a keyword or substring match. "
            "Do not invent places, values or user preferences."
        ),
    )


@lru_cache
def _offer_agent() -> Agent:
    return Agent(
        _build_model(get_settings()),
        system_prompt=(
            "Identify the conversational commitment in the assistant's LAST reply "
            "given the immediately preceding user request. Return ONLY JSON "
            "{\"action\":\"plan|choose|none\",\"destination_names\":[...]}. "
            "Use 'plan' if the assistant offered to plan a trip to ALL the "
            "specific places the user had asked about (even if referred to as "
            "'these places'); 'choose' if the assistant offered separate options "
            "for the user to pick ONE; 'none' if there was no actionable offer. "
            "Do not confuse an earlier plan or incidental places in the reply "
            "with the subject of the last offer. Use names from provided graph "
            "candidates where possible; if spelling differs, supply the actual "
            "names that appeared in this immediate exchange. Do not invent a "
            "destination or interpret a bare acknowledgement as a new preference."
        ),
    )


async def infer_pending_offer(user_message: str, assistant_reply: str,
                              candidates: list[str]) -> PendingOffer | None:
    """Model the assistant's own offer separately from the user's accepted goal."""
    payload = {"user_message": user_message[-1200:], "assistant_reply": assistant_reply[-3000:],
               "graph_candidates": candidates[:16]}
    try:
        result = await _offer_agent().run(json.dumps(payload, ensure_ascii=False))
        text = result.output.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        parsed = json.loads(text)
        if parsed.get("action") not in {"plan", "choose"}:
            return None
        names = parsed.get("destination_names")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            return None
        return PendingOffer(action=parsed["action"], destination_names=names[:8],
                            source_reply=assistant_reply[-1500:])
    except Exception:
        logger.warning("Could not interpret the latest assistant offer", exc_info=True)
        return None


async def understand_turn(message: str, goal: TravelGoal | None, transcript: str) -> TurnDecision:
    """Use the LLM and short-term history; stay useful if that extra call fails."""
    fallback = TurnDecision(_fallback_action(message, bool(transcript)))
    payload = {"latest_message": message, "current_goal": goal.model_dump() if goal else None,
               "recent_conversation": transcript[-4500:]}
    try:
        result = await _turn_agent().run(json.dumps(payload, ensure_ascii=False, default=str))
        output = result.output.strip()
        if output.startswith("```"):
            output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output, flags=re.IGNORECASE)
        parsed = json.loads(output)
        if not isinstance(parsed, dict) or parsed.get("action") not in {"plan", "answer", "chat"}:
            return fallback
        updates = parsed.get("updates") or {}
        if not isinstance(updates, dict):
            updates = {}
        offer_response = parsed.get("offer_response", "none")
        selected = parsed.get("selected_offer_names", [])
        if not isinstance(selected, list) or not all(isinstance(n, str) for n in selected):
            selected = []
        clauses = parsed.get("inferred_clauses")
        operations = parsed.get("merge_operations")
        return TurnDecision(
            parsed["action"], {key: value for key, value in updates.items() if key in _UPDATABLE},
            offer_response if offer_response in {"accept", "decline", "none"} else "none",
            selected[:8],
            [item for item in clauses[:5] if isinstance(item, dict)]
            if isinstance(clauses, list) else [],
            [item for item in operations[:8] if isinstance(item, dict)]
            if isinstance(operations, list) else [])
    except Exception:
        logger.warning("Could not classify chat turn; using conversational fallback", exc_info=True)
        return fallback
