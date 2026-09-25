"""Turn destination data into short, useful trip-plan cards, not articles."""

import json
import logging
import re
from functools import lru_cache
from typing import Any

from pydantic_ai import Agent

from src.agent.agent import _build_model
from src.config import get_settings

logger = logging.getLogger(__name__)

_FIELDS = ("why", "activity", "facilities", "tip")
_ROUTE_FIGURE = re.compile(r"[0-9۰-۹٠-٩]+\s*(?:کیلومتر|ساعت|دقیقه|km|hour)")


def _extract_sentences(text: str, terms: tuple[str, ...], limit: int = 1) -> str:
    """Pick relevant graph facts for the no-LLM fallback, never whole articles."""
    sentences = re.split(r"(?<=[.!؟])\s*|(?<=[.!؟])(?=[آ-ی])", " ".join(text.split()))
    matches = [s.strip() for s in sentences if any(term in s for term in terms)
               and len(s.strip()) <= 230 and "مقاله" not in s and not _ROUTE_FIGURE.search(s)]
    return " ".join(matches[:limit])


def _fallback_card(row: dict[str, Any]) -> str:
    description = row.get("description") if isinstance(row.get("description"), str) else ""
    categories = row.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]
    why = _extract_sentences(description, ("طبیعت", "رودخانه", "جنگل", "دشت", "منظره", "تاریخی"))
    activity = _extract_sentences(description, ("پیاده", "بازدید", "پیک نیک", "عکاسی", "آبشار"))
    facilities = _extract_sentences(description, ("امکانات", "رستوران", "کافه", "سرویس بهداشتی", "آلاچیق"))
    tip = _extract_sentences(description, ("شیب", "کفش", "دشواری", "احتیاط"))
    lines = []
    if why or categories:
        lines.append(f"- **چرا این مقصد؟** {why or '، '.join(map(str, categories))}")
    if activity:
        lines.append(f"- **پیشنهاد بازدید:** {activity}")
    if facilities or row.get("facilities") or row.get("facilities_level"):
        amenities = row.get("facilities") or row.get("facilities_level")
        if isinstance(amenities, list):
            amenities = "، ".join(map(str, amenities))
        lines.append(f"- **امکانات:** {facilities or amenities}")
    if tip or row.get("physical_readiness"):
        lines.append(f"- **نکتهٔ مسیر:** {tip or 'آمادگی بدنی: ' + str(row['physical_readiness'])}")
    return "\n".join(lines) if lines else "- **پیشنهاد بازدید:** برای دیدن این مقصد و استراحت در آن برنامه بگذارید؛ امکانات و شرایط دسترسی را پیش از حرکت بررسی کنید."


def _render_card(data: Any) -> str | None:
    """Constrain the model to short fields; road figures stay server-authored."""
    if not isinstance(data, dict):
        return None
    fields = []
    for field in _FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or len(value) > 230 or "\n" in value or _ROUTE_FIGURE.search(value):
            return None
        fields.append(value.strip())
    if not fields[0] or not fields[1]:
        return None
    labels = ("چرا این مقصد؟", "پیشنهاد بازدید", "امکانات", "نکتهٔ مسیر")
    return "\n".join(f"- **{label}** {text}" for label, text in zip(labels, fields) if text)


@lru_cache
def _formatter() -> Agent:
    return Agent(
        _build_model(get_settings()),
        system_prompt=(
            "You write concise Persian TRIP-PLAN cards, never encyclopedia articles. "
            "Return ONLY JSON with 'destinations': one object per input, using its exact 'name' "
            "and four short plain-text strings 'why', 'activity', 'facilities', 'tip'. "
            "For each destination, summarize why it fits, a concrete activity for the visit, "
            "actual facilities if supported, and a practical accessibility/gear note. "
            "Use graph properties and description as primary evidence; ignore promotional/article "
            "introductions and repetitions. If fields are missing, use general knowledge only "
            "when confident; never invent specific facilities, opening hours, prices or access. "
            "Leave uncertain facilities blank. Adapt suggestions to the travel goal and companions. "
            "Each string should be ONE short sentence, not a paragraph. "
            "Do NOT provide driving distances, driving durations, origin, a route, or other destinations: "
            "those are supplied separately by verified routing. No Markdown in the JSON fields."
        ),
    )


async def format_descriptions(destinations: dict[str, dict[str, Any]], goal: Any = None) -> dict[str, str]:
    """One LLM call for all places; return concise graph-derived fallback if unavailable."""
    if not destinations:
        return {}
    fallback = {name: _fallback_card(row) for name, row in destinations.items()}
    payload = {"travel_goal": goal.model_dump(exclude={"semantic_query"}) if goal else None,
               "destinations": [{"name": name, "data": {k: v for k, v in row.items()
                                                    if k in {"description", "categories", "trip_types", "physical_readiness",
                                                             "facilities_level", "facilities", "road_type", "landmarks",
                                                             "vehicles", "seasons", "city", "province"}}}
                                for name, row in destinations.items()]}
    try:
        result = await _formatter().run(json.dumps(payload, ensure_ascii=False, default=str))
        output = result.output.strip()
        if output.startswith("```"):
            output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output, flags=re.IGNORECASE)
        choices = json.loads(output).get("destinations")
        if not isinstance(choices, list):
            return fallback
        for choice in choices:
            if isinstance(choice, dict) and choice.get("name") in destinations:
                card = _render_card(choice)
                if card:
                    fallback[choice["name"]] = card
    except Exception:
        logger.warning("Could not write destination plan cards; using graph facts", exc_info=True)
    return fallback
