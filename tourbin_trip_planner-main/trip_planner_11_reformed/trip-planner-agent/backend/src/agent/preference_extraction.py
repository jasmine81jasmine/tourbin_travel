"""Extract only enduring travel tastes from the user's own latest message."""

import json
import logging
import re
from functools import lru_cache

from pydantic_ai import Agent

from src.agent.agent import _build_model
from src.config import get_settings

logger = logging.getLogger(__name__)

FEATURE_TYPES = {"Category", "TripType", "Season", "Vehicle", "RequestedStyle"}


def _normalize(text: str) -> str:
    return " ".join(text.translate(str.maketrans({"ي": "ی", "ك": "ک", "‌": " "})).lower().split())


def _grounded_feature(item: dict, vocabulary: dict[str, list[str]]) -> tuple[str, str]:
    label = item.get("feature_type")
    name = item.get("feature_name")
    if label not in FEATURE_TYPES or not isinstance(name, str):
        return "", ""
    # Model output is only a *proposal*. Resolve it against real node names;
    # never let invented names create graph features or influence ranking.
    matches = [value for value in vocabulary.get(label, []) if _normalize(value) == _normalize(name)]
    return (label, matches[0]) if len(matches) == 1 else ("", "")


@lru_cache
def _extractor() -> Agent:
    return Agent(
        _build_model(get_settings()),
        system_prompt=(
            "Extract enduring travel preferences stated by the USER in the latest message only. "
            "Return ONLY JSON: {\"preferences\":[{\"category\":\"...\","
            "\"preference\":\"...\",\"feature_type\":\"...\","
            "\"feature_name\":\"...\",\"polarity\":\"like|avoid\"}],"
            "\"mentions\":[{\"text\":\"exact phrase from latest message\","
            "\"feature_type\":\"...\",\"feature_name\":\"...\"}]}. "
            "Map travel-relevant phrases in the user's latest message to SEMANTICALLY "
            "equivalent graph features from the supplied vocabulary, even if the user "
            "uses a synonym or different language (e.g. 'سرسبز' -> a suitable nature "
            "category). The text field must be copied verbatim from the user's message. "
            "Only return a feature when it genuinely means the same thing, not merely "
            "a nearby or related concept: a quiet place does not prove any destination "
            "is quiet, a mountain does not mean mountaineering. Use [] when unsure. "
            "Return [] for no durable preferences. An enduring taste is explicitly habitual or "
            "general ('I always prefer forests', 'I hate crowds', 'I love camping'). "
            "A request about THIS trip (including companions, destinations, dates, budget, "
            "mood, constraints, or 'no camping this time'), a question, and an assistant "
            "suggestion are NOT durable preferences. Never infer a habit from a one-off request. "
            "Use a stable category identifying the preference dimension: category, trip_type, "
            "season, vehicle, style, pace, accessibility or another short lowercase slug. "
            "For graph-linked concepts use feature_type Category, TripType, Season, Vehicle "
            "or RequestedStyle and COPY the exact feature_name from the supplied vocabulary; "
            "otherwise use empty strings (never fabricate destination attributes). "
            "Examples: nature -> Category/طبیعت, camping -> TripType/کمپ, "
            "quiet -> RequestedStyle/خلوت, adventurous -> RequestedStyle/ماجراجویانه. "
            "For a specific place preference use empty feature fields; place feedback is "
            "handled separately. Preserve the user's meaning, including negation."
        ),
    )


async def extract_travel_memory(
    message: str, vocabulary: dict[str, list[str]] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """One LLM pass grounds both lasting tastes and this turn's travel mentions."""
    vocabulary = vocabulary or {}
    empty: dict[str, list[dict[str, str]]] = {"preferences": [], "mentions": []}
    try:
        result = await _extractor().run(json.dumps(
            {"latest_message": message, "available_graph_features": vocabulary}, ensure_ascii=False,
        ))
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.output.strip(), flags=re.IGNORECASE)
        payload = json.loads(text)
        items = payload.get("preferences", [])
        if not isinstance(items, list):
            return empty
        valid = []
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "")).strip().lower()
            value = str(item.get("preference", "")).strip()
            polarity = item.get("polarity")
            if not re.fullmatch(r"[a-z_]{2,40}", category) or not value or polarity not in {"like", "avoid"}:
                continue
            feature_type, feature_name = _grounded_feature(item, vocabulary)
            valid.append({"category": category, "preference": value[:300],
                          "feature_type": feature_type, "feature_name": feature_name,
                          "polarity": polarity})
        mentions = []
        for item in (payload.get("mentions") or [])[:12]:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            phrase = item["text"].strip()
            label, name = _grounded_feature(item, vocabulary)
            if phrase and len(phrase) <= 100 and _normalize(phrase) in _normalize(message) and label:
                mentions.append({"text": phrase, "feature_type": label, "feature_name": name})
        return {"preferences": valid, "mentions": mentions}
    except Exception:
        logger.warning("Could not ground travel entities and durable preferences", exc_info=True)
        return empty


async def extract_durable_preferences(message: str) -> list[dict[str, str]]:
    """Compatibility entry point for callers that only need durable facts."""
    return (await extract_travel_memory(message))["preferences"]


@lru_cache
def _intent_agent() -> Agent:
    return Agent(
        _build_model(get_settings()),
        system_prompt=(
            "Classify whether the user's message is about travel, destinations, places "
            "they like or avoid visiting, or planning a trip. Understand paraphrases "
            "in any language; do not require fixed keywords. Reply ONLY with JSON "
            "{\"travel_related\":true} or {\"travel_related\":false}."
        ),
    )


async def is_travel_message(message: str) -> bool:
    try:
        result = await _intent_agent().run(message)
        output = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.output.strip(), flags=re.IGNORECASE)
        return json.loads(output).get("travel_related") is True
    except Exception:
        logger.warning("Could not classify travel relevance", exc_info=True)
        return False
