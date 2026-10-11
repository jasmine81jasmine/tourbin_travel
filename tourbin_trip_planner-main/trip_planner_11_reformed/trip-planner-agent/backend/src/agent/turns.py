"""Understand the latest turn before applying any itinerary-only safeguards."""

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic_ai import Agent

from src.agent.agent import _build_model
from src.agent.goals import TravelGoal
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


def apply_goal_updates(goal: TravelGoal | None, updates: dict[str, Any]) -> TravelGoal | None:
    """Explicit new facts win; invalid model patches cannot erase session state."""
    if not updates:
        return goal
    state = (goal or TravelGoal()).model_dump()
    state.update({key: value for key, value in updates.items()
                  if key in _UPDATABLE and not (key == "origin" and isinstance(value, str)
                                              and value in {"من", "ما", "تو", "شما", "خودم"})})
    try:
        return TravelGoal.model_validate(state)
    except Exception:
        logger.warning("Ignoring invalid travel-goal update", exc_info=True)
        return goal


def _fallback_action(message: str, has_history: bool) -> str:
    """When classification fails, preserve conversational follow-ups."""
    text = message.strip().replace("‌", " ")
    if (("؟" in text or "?" in text or re.search(r"^(?:کدوم|کدام|کجا|چطور|چگونه|چیه|چی|آیا)", text))
            and not re.search(r"برنامه\s*(?:بده|بساز|بچین)|پیشنهاد\s*(?:بده|کن)", text)):
        return "answer"
    if re.search(r"(?:برنامه|مسیر|گزینه|پیشنهاد بده|معرفی کن|میخوام|می خواهم|بریم|برویم)", text):
        return "plan"
    return "answer" if has_history else "plan"


def _is_factual_followup(message: str, has_context: bool) -> bool:
    text = message.replace("‌", " ")
    if not has_context or re.search(r"برنامه\s*(?:بده|بریز|بساز|بچین)|برنامه\s+جدید", text):
        return False
    return bool(re.search(r"چرا\s+(?:این|اون|همین)\s+برنامه|چرا\s+این\s+(?:جا|مقصد)|"
                          r"(?:با\s+)?چه\s+(?:ماشین|خودرو|وسیله)|ماشین\s+مناسب|"
                          r"(?:درباره|در مورد|راجع به)\s+.{0,25}(?:اطلاعات|بگو)|"
                          r"بیشتر\s+(?:بگو|توضیح\s+بده)|جزئیات\s+.{0,25}(?:بگو|چیه)|"
                          r"(?:جاده|دسترسی|شرایط|امکانات|پیاده\s*روی|هوا|کمپ).{0,25}(?:چطوره|چیه|چیست|دار[هند]|میشه)",
                          text))


@lru_cache
def _turn_agent() -> Agent:
    return Agent(
        _build_model(get_settings()),
        system_prompt=(
            "You interpret the LATEST message in a Persian travel-planning conversation. "
            "Respond with ONLY JSON: {\"action\":\"plan|answer|chat\",\"updates\":{...}}. "
            "Use 'plan' only when the user wants new/revised destination suggestions, "
            "changes trip constraints and expects a revised itinerary, or asks for a plan. "
            "Use 'answer' for questions about already recommended places: compare them "
            "(e.g. 'which is quieter?'), ask about weather, facilities, camping at a "
            "specific place, safety, details or explain a selected option. Such questions "
            "must NOT produce a new trip plan merely because an older trip goal has a duration. "
            "Use 'chat' for greetings or non-planning small talk. "
            "The current_goal contains the session's earlier constraints; do not drop them. "
            "Put ONLY explicit changes in updates, using field names from current_goal "
            "such as duration, origin, region, destination_names, companions, moods, avoid, "
            "categories, and trip_types. Extract stated travel mood (e.g. هیجان/ماجراجویی "
            "-> moods=['ماجراجویانه']), party (with friends -> companions=['دوستان'], "
            "با پارتنرم -> companions=['پارتنر']), "
            "category (طبیعت -> categories=['طبیعت']), or trip type "
            "(camping -> trip_types=['کمپ']) when clearly requested; don't infer it "
            "from a destination name or a previous assistant suggestion. "
            "Likewise extract an explicit season or transport when given. "
            "A question about which option is quieter is NOT a new mood preference. "
            "If the user clearly selects a destination, update destination_names; if they "
            "ask about a destination without selecting it, do not overwrite the chosen plan. "
            "'میخوام برم آبشار برجی کلا' selects that single place; never plan other destinations instead. "
            "Do not invent places, values or user preferences."
        ),
    )


async def understand_turn(message: str, goal: TravelGoal | None, transcript: str) -> TurnDecision:
    """Use the LLM and short-term history; stay useful if that extra call fails."""
    if _is_factual_followup(message, goal is not None or bool(transcript)):
        return TurnDecision("answer")
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
        action = parsed["action"]
        # A request for a plan is not a question about an existing plan,
        # even when it names a previously suggested destination.
        if action != "plan" and re.search(r"برنامه(?:\s+سفر)?(?:\s+برام)?\s*(?:بده|بساز|بچین|بریز)",
                                          message.replace("‌", " ")):
            action = "plan"
        if action != "plan" and goal is None and _fallback_action(message, False) == "plan":
            action = "plan"
        return TurnDecision(action, {key: value for key, value in updates.items() if key in _UPDATABLE})
    except Exception:
        logger.warning("Could not classify chat turn; using conversational fallback", exc_info=True)
        return fallback
