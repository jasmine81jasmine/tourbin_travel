"""Short, evidence-backed explanations for finalized recommendations."""

import logging

from src.agent.goals import TravelGoal

logger = logging.getLogger(__name__)

# Only claim a match when a destination has an explicit category/trip-type tag.
# A similarity score, a nearby place, or an empty graph field is not evidence.
INTERESTS = (
    ("طبیعت", ("طبیعت", "جنگل", "آبشار", "دریاچه", "رودخانه", "کوه", "ییلاق")),
    ("تاریخی", ("تاریخی", "میراث", "باستان")),
    ("کوهنوردی", ("کوهنوردی", "کوهپیمایی")),
    ("کویر", ("کویر",)),
    ("ساحل", ("ساحل",)),
    ("کمپ", ("کمپ",)),
)


def _matching_interest(text: str, tags: list[str], avoided: str) -> str | None:
    text = text.replace("‌", " ").replace("ي", "ی").replace("ك", "ک").lower()
    tags_text = " ".join(str(tag) for tag in tags).replace("ي", "ی").replace("ك", "ک").lower()
    avoided = avoided.replace("‌", " ").lower()
    for label, terms in INTERESTS:
        if (any(term in text for term in terms)
                and any(term in tags_text for term in terms)
                and not any(term in avoided for term in terms)):
            return label
    return None


async def explain_selected_plans(goal: TravelGoal, routes: list[dict], graph, memory_client, user_id: str) -> str:
    """Explain only stops in the final, map-backed plan(s); fail open if memory is down."""
    if graph is None or not routes:
        return ""
    preferences = []
    if memory_client is not None:
        try:
            preferences = await memory_client.long_term.get_preferences_for(user_id)
        except Exception:
            logger.warning("Could not load preferences for plan explanation", exc_info=True)
    liked = set()
    try:
        history = await graph.get_user_history(user_id)
        liked = set(history.get("liked") or [])
    except Exception:
        logger.warning("Could not load liked destinations for plan explanation", exc_info=True)

    # The objective retains earlier turns; deleted interests must not be
    # resurrected from a previous version of the request.
    current = " ".join([goal.objective.rsplit("اصلاح جدید:", 1)[-1],
                        *goal.moods, *goal.activities, *goal.must_haves, *goal.accommodation])
    avoided = " ".join(goal.avoid)
    names = list(dict.fromkeys(stop.get("name") for route in routes for stop in route.get("stops", []) if stop.get("name")))[:6]
    lines = []
    for name in names:
        try:
            detail = await graph.get_destination_details(name)
        except Exception:
            logger.warning("Could not load destination evidence for %s", name, exc_info=True)
            continue
        if not detail or detail.get("name") != name:
            continue  # fuzzy lookup must not explain a different destination
        tags = (detail.get("categories") or []) + (detail.get("trip_types") or [])
        current_match = _matching_interest(current, tags, avoided)
        remembered_match = next((match for pref in preferences
                                 if str(pref.category).lower() not in {"avoid", "dislike", "restriction"}
                                 and not any(word in str(pref.preference).lower() for word in
                                             ("نمی", "دوست ندار", "اجتناب", "بدون", "dislike", "avoid", "not "))
                                 if (match := _matching_interest(str(pref.preference), tags, avoided))), None)
        if (not current_match and any(term in current for _, terms in INTERESTS for term in terms)):
            remembered_match = None  # a different explicit request wins over old tastes
        reasons = []
        if current_match:
            evidence = next((tag for tag in tags if isinstance(tag, str) and len(tag) <= 35
                             and _matching_interest(current, [tag], avoided) == current_match), current_match)
            reasons.append(f"ویژگی «{evidence}» آن با علاقه‌ای که در این سفر به {current_match} گفتی هماهنگ است")
        if remembered_match and remembered_match != current_match:
            evidence = next((tag for tag in tags if isinstance(tag, str) and len(tag) <= 35
                             and _matching_interest(remembered_match, [tag], avoided)), remembered_match)
            reasons.append(f"ویژگی «{evidence}» آن با علاقه‌ات به {remembered_match} که از گفتگوهای قبلی به یاد دارم جور درمی‌آید")
        elif remembered_match and current_match:
            reasons.append("این علاقه را از گفتگوهای قبلی‌مان هم به یاد دارم")
        if name in liked:
            reasons.append("این مقصد را قبلاً هم پسندیده بودی")
        if reasons:
            lines.append(f"- **{name}:** " + "؛ ".join(reasons) + ".")
    return "### چرا این مقصدها برای تو؟\n" + "\n".join(lines) if lines else ""
