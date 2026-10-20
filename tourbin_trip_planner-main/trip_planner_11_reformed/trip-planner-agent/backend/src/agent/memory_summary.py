"""Answer memory questions from this user's stored facts, not model guesses."""

import json
import logging

from src.agent.goals import TravelGoal, insights_from_previous_goal

logger = logging.getLogger(__name__)


def _prior_request(features: dict[str, list[str]]) -> str:
    parts = []
    for key in ("moods", "categories", "trip_types"):
        parts.extend(features.get(key, [])[:2])
    parts.extend("با " + person for person in features.get("companions", [])[:1])
    return "، ".join(dict.fromkeys(parts))


async def summarize_user_memory(memory, graph, user_id: str, session_id: str,
                                current_goal: TravelGoal | None) -> str:
    """Distinguish durable facts, past requests, and actual destination feedback."""
    if memory is None or graph is None:
        return ("الان به حافظهٔ سفرهای قبلی‌ات دسترسی ندارم و نمی‌خوام دربارهٔ سلیقه‌ات حدس بزنم. "
                "اگر بگی این بار چه جور سفری می‌خوای، بر اساس همون برات برنامه می‌ریزم.")

    history = {}
    preferences = []
    prior = {}
    read_failed = False
    try:
        history = await graph.get_user_history(user_id, exclude_session_id=session_id)
    except Exception:
        read_failed = True
        logger.warning("Could not read user travel history for memory answer", exc_info=True)
    try:
        preferences = (await graph.get_durable_preferences(user_id)
                       if hasattr(graph, "get_durable_preferences")
                       else await memory.long_term.get_preferences_for(user_id))
    except Exception:
        read_failed = True
        logger.warning("Could not read durable preferences for memory answer", exc_info=True)
    try:
        prior = await graph.get_recent_session_features(user_id, session_id)
    except Exception:
        read_failed = True
        logger.warning("Could not read prior session features for memory answer", exc_info=True)
    if not prior and history.get("recent_goals"):
        prior = insights_from_previous_goal(history["recent_goals"][0])
    if not prior and current_goal is not None:
        prior = current_goal.prior_insights

    lines = []
    if preferences:
        lines.append("- **سلیقه‌های ماندگاری که گفته‌ای:** "
                     + "؛ ".join(str(p["preference"] if isinstance(p, dict) else p.preference)
                                   for p in preferences[:2]) + ".")
    if prior:
        description = _prior_request(prior)
        if description:
            lines.append(f"- **درخواست یکی از سفرهای قبلی‌ات:** {description}. "
                         "این را ترجیح همیشگی‌ات فرض نمی‌کنم.")
    if history.get("recent_goals"):
        try:
            previous_goal = TravelGoal.model_validate(json.loads(history["recent_goals"][0]))
            details = [value for value in (previous_goal.duration, previous_goal.region) if value]
            if details:
                lines.append("- **جزئیات همان برنامهٔ قبلی:** " + "، ".join(details) + ".")
        except (ValueError, TypeError):
            pass
    if current_goal is not None:
        current = _prior_request({key: getattr(current_goal, key) for key in
                                  ("moods", "categories", "trip_types", "companions")})
        if current:
            lines.append(f"- **در همین گفتگو گفتی:** {current}.")
    for key, label in (("liked", "پسندیده‌ای"), ("disliked", "نپسندیده‌ای"), ("visited", "رفته‌ای")):
        names = history.get(key) or []
        if names:
            lines.append(f"- **مقصدهایی که {label}:** " + "، ".join(map(str, names[:3])) + ".")
    trips = history.get("trips") or []
    if trips:
        names = trips[0].get("destinations") or []
        if names:
            lines.append("- **برنامه‌ای که قبلاً پیشنهاد دادم:** " + "، ".join(map(str, names[:2]))
                         + " (پیشنهاد بوده، نه لزوماً سفری که رفته‌ای).")
    if not lines:
        if read_failed:
            return "الان نتونستم همهٔ اطلاعات ذخیره‌شدهٔ سفرهات را بخونم؛ نمی‌خوام حدس بزنم یا بگم چیزی ازت نمی‌دونم."
        return ("فعلاً برای این حساب اطلاعات تأییدشده‌ای از سلیقه یا سفرهای قبلی‌ات ثبت نشده؛ "
                "نمی‌خوام چیزی درباره‌ات حدس بزنم. دوست داری چه جور سفری برات بچینم؟")
    return "از سفرهامون این‌ها را درباره‌ات می‌دانم:\n" + "\n".join(lines)
