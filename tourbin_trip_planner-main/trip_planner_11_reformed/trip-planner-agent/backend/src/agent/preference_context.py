"""Current trip requests take precedence over remembered general tastes."""

from typing import Any

from src.agent.goals import TravelGoal


def effective_preferences(preferences: list[dict[str, Any]], goal: TravelGoal | None) -> list[dict[str, Any]]:
    if goal is None:
        return preferences
    explicit = {
        "Category": goal.categories,
        "TripType": goal.trip_types + goal.accommodation,
        "Season": [goal.season] if goal.season else [],
        "Vehicle": [goal.transport] if goal.transport else [],
        "RequestedStyle": goal.moods,
    }
    category_fields = {
        "category": "Category", "trip_type": "TripType", "season": "Season",
        "vehicle": "Vehicle", "transport": "Vehicle", "style": "RequestedStyle",
        "mood": "RequestedStyle", "pace": "RequestedStyle",
    }
    avoids = " ".join(goal.avoid).lower()
    result = []
    for pref in preferences:
        dimension = pref.get("feature_type") or category_fields.get(pref.get("category"))
        if dimension in explicit and explicit[dimension]:
            # Do not bring a habitual taste into a trip explicitly specifying
            # this dimension; in particular do not rank a quiet trip as adventure.
            continue
        name = (pref.get("feature_name") or "").lower()
        if name and name in avoids:
            continue
        result.append(pref)
    return result
