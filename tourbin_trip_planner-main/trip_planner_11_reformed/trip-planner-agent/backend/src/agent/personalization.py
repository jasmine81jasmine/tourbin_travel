"""Short, evidence-backed explanations for selected or proposed destinations."""

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
ADVENTURE_TAGS = ("کوهنوردی", "غارنوردی", "رفتینگ", "صخره‌نوردی", "صخره نوردی", "ماجراجویی")


def _trip_context(goal: TravelGoal) -> str:
    """Acknowledge this trip's extracted insights, without calling them habits."""
    parts = []
    moods = [mood for mood in goal.moods if mood not in goal.avoid]
    if moods:
        parts.append("حال‌وهوا: " + "، ".join(moods[:2]))
    if goal.companions:
        parts.append("همراهان: " + "، ".join(goal.companions[:2]))
    if goal.season:
        parts.append("فصل: " + goal.season)
    if goal.transport:
        parts.append("وسیلهٔ سفر: " + goal.transport)
    return "؛ ".join(parts)


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
    """Explain only places named in the final reply; fail open if memory is down."""
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
                        *goal.categories, *goal.trip_types, *goal.moods,
                        *goal.activities, *goal.must_haves, *goal.accommodation])
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
        # "آبشار" in the selected place's *name* isn't proof the user asked
        # for waterfall/nature trips; remove the named place before matching.
        interest_text = current.replace(name, "")
        current_match = _matching_interest(interest_text, tags, avoided)
        remembered_match = next((match for pref in preferences
                                 if str(pref.category).lower() not in {"avoid", "dislike", "restriction"}
                                 and not any(word in str(pref.preference).lower() for word in
                                             ("نمی", "دوست ندار", "اجتناب", "بدون", "dislike", "avoid", "not "))
                                 if (match := _matching_interest(str(pref.preference), tags, avoided))), None)
        if (not current_match and any(term in interest_text for _, terms in INTERESTS for term in terms)):
            remembered_match = None  # a different explicit request wins over old tastes
        reasons = []
        if current_match:
            evidence = next((tag for tag in tags if isinstance(tag, str) and len(tag) <= 35
                             and _matching_interest(interest_text, [tag], avoided) == current_match), current_match)
            reasons.append(f"ویژگی «{evidence}» آن با علاقه‌ای که در این سفر به {current_match} گفتی هماهنگ است")
            if current_match == "کمپ":
                reasons.append("مجاز بودن محل دقیق کمپ را باید پیش از حرکت بررسی کنی")
        if current_match != "کمپ" and "کمپ" not in avoided and "کمپ" in interest_text and "کمپ" in (detail.get("trip_types") or []):
            reasons.append("نوع سفر «کمپ» برای این مقصد ثبت شده؛ مجاز بودن محل دقیق کمپ را باید بررسی کنی")
        if "ماجراجویانه" in goal.moods and "ماجراجویانه" not in goal.avoid:
            adventure = next((tag for tag in tags if any(term in tag for term in ADVENTURE_TAGS)), None)
            if adventure:
                reasons.append(f"ویژگی «{adventure}» آن با سفر ماجراجویانه‌ای که خواستی هماهنگ است")
            elif tags:
                reasons.append("دربارهٔ میزان هیجان این مقصد اطلاعات تأییدشدهٔ کافی ندارم")
        if goal.season and goal.season in (detail.get("seasons") or []):
            reasons.append(f"{goal.season} هم در فصل‌های مناسب بازدیدش ثبت شده است")
        vehicle = ("سواری" if goal.transport in {"ماشین", "خودرو", "ماشین سواری", "خودرو سواری"}
                   else goal.transport)
        if vehicle and vehicle in (detail.get("vehicles") or []):
            reasons.append(f"دسترسی با «{vehicle}» برای این مقصد ثبت شده است")
        if remembered_match and remembered_match != current_match:
            evidence = next((tag for tag in tags if isinstance(tag, str) and len(tag) <= 35
                             and _matching_interest(remembered_match, [tag], avoided)), remembered_match)
            reasons.append(f"ویژگی «{evidence}» آن با علاقه‌ات به {remembered_match} که از گفتگوهای قبلی به یاد دارم جور درمی‌آید")
        elif remembered_match and current_match:
            reasons.append("این علاقه را از گفتگوهای قبلی‌مان هم به یاد دارم")
        prior = " ".join(value for key in ("categories", "trip_types")
                         for value in goal.prior_insights.get(key, []))
        prior_match = _matching_interest(prior, tags, avoided)
        if prior_match and prior_match != current_match:
            evidence = next((tag for tag in tags if isinstance(tag, str) and len(tag) <= 35
                             and _matching_interest(prior_match, [tag], avoided)), prior_match)
            reasons.append(f"ویژگی «{evidence}» آن به درخواست {prior_match} در سفر قبلی‌ات هم نزدیک است")
        if ("ماجراجویانه" in goal.prior_insights.get("moods", [])
                and "ماجراجویانه" not in goal.moods):
            adventure = next((tag for tag in tags if isinstance(tag, str)
                              and any(term in tag for term in ADVENTURE_TAGS)), None)
            if adventure:
                reasons.append(f"ویژگی «{adventure}» آن با حال‌وهوای ماجراجویانهٔ سفر قبلی‌ات جور است")
        if name in liked:
            reasons.append("این مقصد را قبلاً هم پسندیده بودی")
        if not reasons and name in current and tags:
            # A destination chosen by name still deserves a specific reason,
            # even if the user's latest turn doesn't state an activity.
            tag = next((tag for tag in tags if isinstance(tag, str) and len(tag) <= 35), None)
            if tag:
                reasons.append(f"خودت این مقصد را انتخاب کردی؛ یکی از ویژگی‌های ثبت‌شده‌اش «{tag}» است")
        if reasons:
            lines.append(f"- **{name}:** " + "؛ ".join(reasons) + ".")
    if not lines:
        return ""
    context = _trip_context(goal)
    return ("### چرا این مقصدها برای تو؟\n"
            + (f"- **درخواست این سفر:** {context}.\n" if context else "")
            + "\n".join(lines))
