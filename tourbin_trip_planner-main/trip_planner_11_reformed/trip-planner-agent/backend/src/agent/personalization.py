"""Explain actual selected routes in natural language, with user-scoped evidence."""

import json
import logging
import re
from functools import lru_cache
from typing import Any, Awaitable, Callable

from pydantic_ai import Agent

from src.agent.agent import _build_model
from src.agent.goals import TravelGoal
from src.config import get_settings
from src.memory.preference_graph import feature_id

logger = logging.getLogger(__name__)

ADVENTURE_TAGS = ("کوهنوردی", "غارنوردی", "رفتینگ", "صخره‌نوردی", "صخره نوردی", "ماجراجویی")
_NUMERIC_ROUTE = re.compile(r"[0-9۰-۹]+\s*(?:ساعت|دقیقه|کیلومتر|km)")


@lru_cache
def _reason_writer() -> Agent:
    return Agent(
        _build_model(get_settings()),
        system_prompt=(
            "Write a warm, concise Persian explanation of the ACTUAL proposed travel plans. "
            "Return ONLY JSON: {\"intro\":\"one short sentence\", \"plans\":[{\"index\":1,\"why\":\"one or two sentences\"}]}. "
            "One plan entry per supplied route, in the same order. Write natural prose, not tag listings, "
            "field labels, generic compliments, disclaimers about unavailable 'excitement' data, or a new itinerary. "
            "Connect the user's explicitly stated interests to distinctive destination facts ONLY when supported "
            "by the provided graph evidence. The matched_goal_dimensions identify exactly which "
            "atomic goal aspects each selected destination supports; write retrospective WHY "
            "these places were chosen, not goal status, goal IDs or a list of goals. "
            "A category of nature does not verify camping or excitement; a best-season "
            "autumn link does not verify autumn foliage. Mention grounded details from "
            "each destination's description, facilities or terrain where relevant, "
            "without inventing missing amenities. Prior-session insights are soft hints, never current instructions or "
            "permanent habits; durable preferences and explicit likes are separately marked. If adventure was "
            "mentioned but no adventurous feature is verified, don't claim this route is adventurous and don't "
            "announce a missing tag. A selected destination must never be replaced by another. Do not claim "
            "camping is permitted, a road is easy, a place is uncrowded, or it suits children without "
            "supporting data. Wilderness/green scenery alone is not proof of low crowds. "
            "Use only the supplied route names; no extra place, precise distance, duration, live condition, "
            "source name or internal field. Never claim a suggested plan was actually visited."
        ),
    )


async def write_explanation(payload: dict[str, Any]) -> dict[str, Any] | None:
    """One bounded LLM call for all routes; failed output uses grounded fallback."""
    try:
        result = await _reason_writer().run(json.dumps(payload, ensure_ascii=False, default=str))
        output = result.output.strip()
        if output.startswith("```"):
            output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output, flags=re.IGNORECASE)
        data = json.loads(output)
        if not isinstance(data, dict) or not isinstance(data.get("plans"), list):
            return None
        intro = data.get("intro", "")
        if not isinstance(intro, str) or len(intro) > 250 or "\n" in intro or _NUMERIC_ROUTE.search(intro):
            return None
        if (("یادم" in intro or "قبلی" in intro)
                and not (payload["previous_trip_requests_not_durable"] or payload["durable_preferences"])):
            return None
        if len(data["plans"]) != len(payload["plans"]):
            return None
        for index, (item, plan) in enumerate(zip(data["plans"], payload["plans"]), 1):
            evidence = " ".join(str(value) for stop in plan["stops"]
                                for key in ("categories", "trip_types", "description", "facilities",
                                            "road_type", "physical_readiness")
                                for value in (stop.get(key) if isinstance(stop.get(key), list)
                                              else [stop.get(key) or ""]))
            if (not isinstance(item, dict) or item.get("index") != index
                    or not isinstance(item.get("why"), str) or not item["why"].strip()
                    or len(item["why"]) > 520 or "\n" in item["why"]
                    or _NUMERIC_ROUTE.search(item["why"])
                    or any(facility in item["why"] and facility not in evidence
                           for facility in ("سرویس بهداشتی", "پارکینگ", "رستوران", "کافه", "آلاچیق"))
                    or any(word in item["why"] for word in ("اطلاعی در مورد هیجان ندارم", "دادهٔ کافی ندارم"))
                    or re.search(r"کمپ.{0,20}مجاز|مجاز.{0,20}کمپ", item["why"])
                    or (any(term in item["why"] for term in ("ماجراجو", "هیجان"))
                        and not any(term in evidence for term in ADVENTURE_TAGS + ("صخره", "رودخانه‌نوردی")))
                    or any(name not in " ".join(stop["name"] for stop in plan["stops"])
                           for name in re.findall(r"«([^»]+)»", item["why"]) if name.startswith(("آبشار", "جنگل", "دریاچه")))):
                return None
        return data
    except Exception:
        logger.warning("Could not write grounded plan reasons; using concise fallback", exc_info=True)
        return None


def _context(goal: TravelGoal, preferences: list, liked: set[str]) -> dict[str, Any]:
    latest = goal.objective.rsplit("اصلاح جدید:", 1)[-1]
    # A new explicit trip type/category wins over a different old taste.
    active = set(goal.categories + goal.trip_types)
    active.update(label for label in ("طبیعت", "تاریخی", "کوهنوردی", "کویر", "کمپ") if label in latest)
    conflicting = {"طبیعت", "تاریخی", "کوهنوردی", "کویر", "کمپ"} - active
    durable = [str(p.preference)[:120] for p in preferences[:3]
               if str(p.category).lower() not in {"avoid", "dislike"}]
    if active:
        durable = [preference for preference in durable
                   if not any(term in preference for term in conflicting)]
    prior = {key: [value for value in values if value not in goal.avoid]
             for key, values in goal.prior_insights.items()}
    for key in ("categories", "moods", "trip_types"):
        if getattr(goal, key):
            prior.pop(key, None)
    if goal.accommodation and "کمپ" not in goal.accommodation:
        prior.pop("trip_types", None)
    return {
        "this_trip": {"request": latest[:350], "duration": goal.duration,
                      "companions": goal.companions, "moods": [m for m in goal.moods if m not in goal.avoid],
                      "categories": [c for c in goal.categories if c not in goal.avoid],
                      "trip_types": list(dict.fromkeys(
                          [t for t in goal.trip_types if t not in goal.avoid]
                          + (["کمپ"] if "کمپ" in goal.accommodation and "کمپ" not in goal.avoid else [])
                      )),
                      "transport": goal.transport},
        "previous_trip_requests_not_durable": prior,
        "durable_preferences": durable,
        "previously_liked_destinations": sorted(liked),
    }


def _fallback_intro(context: dict) -> str:
    trip = context["this_trip"]
    prior = context["previous_trip_requests_not_durable"]
    remembered = prior.get("moods") or prior.get("categories") or prior.get("trip_types")
    if remembered:
        extra = (f" برای سفر با {trip['companions'][0]}" if trip["companions"] else "")
        return (f"یادم هست در سفر قبلی‌ات حال‌وهوای {remembered[0]} را می‌خواستی؛ "
                f"این بار{extra} آن را به‌عنوان سرنخ در نظر گرفتم.")
    if context["durable_preferences"]:
        return f"از قبل یادم هست {context['durable_preferences'][0]}؛ خواستهٔ همین سفرت هم در انتخاب مسیر اولویت دارد."
    if trip["moods"]:
        companion = f" با {trip['companions'][0]}" if trip["companions"] else ""
        return f"این بار سفر {trip['moods'][0]}{companion} می‌خواهی؛ مسیرها را با همین خواسته سنجیدم."
    if trip["companions"]:
        return f"این مسیرها را برای سفر با {trip['companions'][0]} چیدم."
    return "این مقصدها را از میان گزینه‌های متناسب با درخواست همین سفرت کنار هم گذاشتم."


def _fallback_plan(plan: dict, context: dict) -> str:
    names = [stop["name"] for stop in plan["stops"]]
    if not names:
        return ""
    trip = context["this_trip"]
    prior = context["previous_trip_requests_not_durable"]
    facts = [tag for stop in plan["stops"] for tag in stop.get("categories", []) + stop.get("trip_types", [])
             if isinstance(tag, str) and len(tag) <= 30]
    traits = " ".join(trip["moods"] or (
        prior.get("moods", []) if not trip["categories"] and not trip["trip_types"] else []))
    adventure = next((tag for tag in facts if any(term in tag for term in ADVENTURE_TAGS)), None)
    if adventure and any(term in traits for term in ("ماجراجو", "هیجان")):
        reason = f"{names[0]} با مسیر {adventure}، به حال‌وهوای پرتحرکی که در نظر داشتی نزدیک است"
    elif "کمپ" in facts and "کمپ" in trip["trip_types"]:
        reason = f"{names[0]} برای نوع سفرِ کمپی که این بار می‌خواهی ثبت شده است"
    elif match := next((item for item in plan.get("matched_goal_dimensions", [])
                        if item["source"] == "explicit"), None):
        reason = (f"{match['destination']} به خاطر {match['supported_aspect']} "
                  "با خواستهٔ همین سفرت هم‌خوانی دارد")
    elif "کمپ" in facts and "کمپ" in prior.get("trip_types", []):
        reason = f"{names[0]} برای نوع سفر کمپیِ موردعلاقه‌ات در درخواست قبلی ثبت شده است"
    elif plan.get("matched_goal_dimensions"):
        match = plan["matched_goal_dimensions"][0]
        reason = (f"{match['destination']} به خاطر {match['supported_aspect']} "
                  "با یکی از علاقه‌های ثبت‌شده‌ات هم‌خوانی دارد؛ خواستهٔ همین سفر اولویت دارد")
    elif facts:
        reason = f"{names[0]} با فضای {facts[0]}، بخش اصلی این مسیر را می‌سازد"
    else:
        reason = f"{names[0]} را به‌عنوان نقطهٔ اصلی این مسیر در نظر گرفتم"
    if len(names) > 1:
        reason += f" و {names[1]} را در ادامهٔ همان سفر گذاشتم"
    if plan["stops"][0]["liked_by_this_user"]:
        reason += f"؛ {names[0]} را هم قبلاً پسندیده بودی"
    return reason + "."


def _matched_goal_dimensions(goal: TravelGoal, plan: dict) -> list[dict[str, str]]:
    """Only graph-backed aspects of selected stops may explain a subgoal."""
    matches = []
    for subgoal in goal.subgoals:
        for stop in plan["stops"]:
            aspect = None
            if subgoal.evidence_type == "destination" and subgoal.target == stop["name"]:
                aspect = f"مقصد درخواستی «{stop['name']}»"
            elif subgoal.evidence_type == "category" and subgoal.category in stop["categories"]:
                aspect = f"دستهٔ «{subgoal.category}»"
            elif subgoal.evidence_type == "feature" and subgoal.feature_id:
                named = stop.get("feature_names", {}).get(subgoal.feature_id)
                if named:
                    aspect = f"ویژگی «{named}»"
            if aspect:
                matches.append({"goal_id": subgoal.id, "source": subgoal.source,
                                "destination": stop["name"], "supported_aspect": aspect})
                break
    return matches[:8]


def _description_detail(plan: dict) -> str:
    """Short graph-description excerpts add place context without fabricating facilities."""
    lines = []
    for stop in plan["stops"][:3]:
        description = str(stop.get("description") or "")
        sentences = re.split(r"(?<=[.؟!])\s+", " ".join(description.split()))
        useful = [sentence.strip() for sentence in sentences
                  if 15 <= len(sentence.strip()) <= 220
                  and not _NUMERIC_ROUTE.search(sentence)
                  and not any(term in sentence for term in ("مقاله", "وب‌سایت", "برای رزرو"))]
        if useful:
            lines.append(f"**دربارهٔ {stop['name']}:** {' '.join(useful[:2])}")
    return "\n\n".join(lines)


async def explain_selected_plans(
    goal: TravelGoal, routes: list[dict], graph, memory_client, user_id: str,
    writer: Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]] | None = None,
) -> str:
    """One human-readable reason per final route, after route screening."""
    if graph is None or not routes:
        return ""
    preferences = []
    liked: set[str] = set()
    if memory_client is not None:
        try:
            preferences = await memory_client.long_term.get_preferences_for(user_id)
        except Exception:
            logger.warning("Could not load preferences for plan explanation", exc_info=True)
    try:
        liked = set((await graph.get_user_history(user_id)).get("liked") or [])
    except Exception:
        logger.warning("Could not load liked destinations for plan explanation", exc_info=True)

    details: dict[str, dict] = {}
    plans = []
    for route in routes[:4]:
        stops = []
        for stop in route.get("stops", [])[:4]:
            name = stop.get("name")
            if not isinstance(name, str) or not name:
                continue
            if name not in details:
                try:
                    found = await graph.get_destination_details(name)
                    details[name] = found if found and found.get("name") == name else {}
                except Exception:
                    logger.warning("Could not load evidence for %s", name, exc_info=True)
                    details[name] = {}
            evidence = details[name]
            feature_names = {}
            if hasattr(graph, "get_destination_features"):
                try:
                    feature_names = {item["id"]: item["name"]
                                     for item in await graph.get_destination_features(name)
                                     if item.get("id") and item.get("name")}
                except Exception:
                    logger.warning("Could not load ontology evidence for %s", name, exc_info=True)
            else:
                for label, values in (("TripType", evidence.get("trip_types") or []),
                                      ("Season", evidence.get("seasons") or []),
                                      ("Vehicle", evidence.get("vehicles") or [])):
                    for value in values:
                        feature_names[feature_id(label, value)] = value
            stops.append({"name": name, "categories": evidence.get("categories") or [],
                          "trip_types": evidence.get("trip_types") or [],
                          "seasons": evidence.get("seasons") or [],
                          "vehicles": evidence.get("vehicles") or [],
                          "facilities": evidence.get("facilities") or [],
                          "road_type": evidence.get("road_type"),
                          "facilities_level": evidence.get("facilities_level"),
                          "physical_readiness": evidence.get("physical_readiness"),
                          "description": str(evidence.get("description") or "")[:650],
                          "feature_names": feature_names,
                          "liked_by_this_user": name in liked})
        if stops:
            plan = {"stops": stops}
            plan["matched_goal_dimensions"] = _matched_goal_dimensions(goal, plan)
            plans.append(plan)
    if not plans or not any(stop["categories"] or stop["trip_types"] or stop["description"]
                            or stop["feature_names"]
                            for plan in plans for stop in plan["stops"]):
        return ""

    context = _context(goal, preferences, liked)
    payload = {**context, "plans": plans}
    written = await writer(payload) if writer is not None else None
    intro = written.get("intro") if written else None
    lines = ["### چرا این سفر را پیشنهاد کردم؟", intro or _fallback_intro(context)]
    for index, plan in enumerate(plans, 1):
        generated = written["plans"][index - 1]["why"] if written else ""
        lines.append(f"**مسیر {index}:** {generated or _fallback_plan(plan, context)}")
        if detail := _description_detail(plan):
            lines.append(detail)
    return "\n\n".join(lines)
