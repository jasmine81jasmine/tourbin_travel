"""Small LLM-guided goal decomposition and evidence-based plan evaluation."""

import json
import logging
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.agent.goals import GoalCoverage, PlanningConstraint, PlanningSubgoal, TravelGoal
from src.memory.preference_graph import (
    DESTINATION_FEATURE_RELATIONS, destination_feature_id, normalized,
)

logger = logging.getLogger(__name__)


class GoalDesign(BaseModel):
    subgoals: list[PlanningSubgoal] = Field(default_factory=list)
    constraints: list[PlanningConstraint] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    retired_goal_ids: list[str] = Field(default_factory=list)
    retired_constraints: list[str] = Field(default_factory=list)


class GoalDecomposition(BaseModel):
    children: list[PlanningSubgoal] = Field(default_factory=list)


def retire_replaced_subgoals(prior: TravelGoal | None, current: TravelGoal,
                             updates: dict, features: list[dict]) -> TravelGoal:
    """A changed explicit field retires its obsolete evidence goal, not the history."""
    if prior is None or not updates or not current.subgoals:
        return current
    by_id = {f["id"]: f for f in features}
    selected = {normalized(name) for name in current.destination_names}
    cats = {normalized(name) for name in current.categories}
    typed = {
        "Season": ("season", {normalized(current.season)} if current.season else set()),
        "TripType": ("trip_types", {normalized(n) for n in current.trip_types}),
        "Category": ("categories", cats),
    }
    previous_positive = {
        normalized(value) for field in ("trip_types", "accommodation",
                                        "categories", "activities", "moods")
        for value in getattr(prior, field)
    }
    original_subgoals = {subgoal.id: subgoal for subgoal in prior.subgoals}

    def replaced(subgoal: PlanningSubgoal) -> bool:
        original = original_subgoals.get(subgoal.id)
        if (subgoal.source != "explicit" or original is None
                or original.name != subgoal.name):
            return False
        avoided = {normalized(value) for value in current.avoid} if "avoid" in updates else set()
        if any(value in previous_positive and value in normalized(subgoal.name)
               for value in avoided):
            return True
        if subgoal.evidence_type == "category" and subgoal.category and (
                normalized(subgoal.category) in avoided):
            return True
        if (subgoal.evidence_type == "feature" and subgoal.feature_id in by_id
                and normalized(by_id[subgoal.feature_id]["name"]) in avoided):
            return True
        if ("destination_names" in updates and subgoal.evidence_type == "destination"
                and subgoal.target and normalized(subgoal.target) not in selected):
            return True
        if ("categories" in updates and subgoal.evidence_type == "category"
                and subgoal.category and normalized(subgoal.category) not in cats):
            return True
        if subgoal.evidence_type == "feature" and subgoal.feature_id in by_id:
            feature = by_id[subgoal.feature_id]
            entry = typed.get(feature["label"])
            if entry and entry[0] in updates and normalized(feature["name"]) not in entry[1]:
                return True
        return False

    remaining = [s for s in current.subgoals if not replaced(s)]
    if len(remaining) == len(current.subgoals):
        return current
    surviving = {s.id for s in remaining}
    return current.model_copy(update={
        "subgoals": remaining,
        "goal_coverage": [c for c in current.goal_coverage if c.goal_id in surviving],
    })


def _effective_preferences(
    goal: TravelGoal, preferences: list[dict], features: list[dict],
) -> tuple[list[dict], dict[str, set[str]]]:
    """Only apply older feature preferences where this trip left that dimension open."""
    available = {f["id"]: f for f in features}
    active = {
        "Category": goal.categories,
        "Season": [goal.season] if goal.season else [],
        "TripType": goal.trip_types + (["کمپ"] if "کمپ" in goal.accommodation else []),
        "Vehicle": (["سواری"] if goal.transport in {"ماشین", "خودرو", "ماشین سواری", "خودرو سواری"}
                    else [goal.transport] if goal.transport else []),
        "TravelStyle": goal.moods,
        "Environment": goal.moods,
        "AccommodationType": goal.accommodation,
    }
    active_ids: dict[str, set[str]] = {
        label: {f["id"] for f in features if f["label"] == label and
                normalized(f["name"]) in {normalized(value) for value in values}}
        for label, values in active.items() if values
    }
    avoided = {normalized(value) for value in goal.avoid}
    usable = [
        pref for pref in preferences if (
            (f := available.get(pref.get("id"))) is not None
            and normalized(f["name"]) not in avoided
            and (f["label"] not in active_ids or pref["id"] in active_ids[f["label"]])
        )
    ]
    return usable, active_ids


@lru_cache
def _decomposer() -> Agent:
    from src.agent.agent import _build_model
    from src.config import get_settings

    return Agent(_build_model(get_settings()), output_type=GoalDecomposition, system_prompt=(
        "Decompose ONLY the unmet subgoal into at most two smaller, independently "
        "checkable tasks, if the available tools and data can actually verify them. "
        "Return no children if the evidence is unavailable or the task cannot be "
        "meaningfully decomposed. Never weaken explicit user requirements or invent "
        "destination facts. Decomposition describes CHECKABLE outcomes, not a "
        "predetermined tool-call sequence: the executor chooses tools in a "
        "result-aware loop. Each child must use an id supplied in allowed_ids, "
        "parent_id equal to the failed goal id, and evidence_type destination "
        "(with exact target), category (with category), feature (with a real "
        "destination-linked feature_id), route, or unverified. "
        "Do not turn assumptions or historical preferences into explicit requests."
    ))


async def decompose_unmet_goal(goal: TravelGoal, coverage: list[GoalCoverage],
                               attempted_tools: list[str]) -> TravelGoal:
    """One bounded ADaPT-style revision after an executor fails a checkable goal."""
    missing = next((c for c in coverage if c.status != "satisfied"
                    and any(g.id == c.goal_id and g.priority == "high"
                            and g.evidence_type != "unverified" for g in goal.subgoals)), None)
    if missing is None or len(goal.subgoals) >= 8 or any(
            g.parent_id == missing.goal_id for g in goal.subgoals):
        return goal
    parent = next(g for g in goal.subgoals if g.id == missing.goal_id)
    allowed = [f"{parent.id}.{i}" for i in (1, 2)
               if f"{parent.id}.{i}" not in {g.id for g in goal.subgoals}]
    try:
        payload = {"failed_goal": parent.model_dump(), "failure_reason": missing.reason,
                   "attempted_tools": attempted_tools[-8:],
                   "available_tools": ["search_destinations", "semantic_search_destinations",
                                       "search_destinations_by_features",
                                       "get_destination_details", "get_destination_features",
                                       "build_trip_map"],
                   "destination_linked_feature_ids": list(dict.fromkeys(
                       [fid for fid in parent.related_features if destination_feature_id(fid)]
                       + ([parent.feature_id] if parent.feature_id
                          and destination_feature_id(parent.feature_id) else []))),
                   "trip_constraints": [c.model_dump() for c in goal.planning_constraints],
                   "allowed_ids": allowed}
        result = await _decomposer().run(json.dumps(payload, ensure_ascii=False))
        proposed = GoalDecomposition.model_validate(result.output)
        children = [child for child in proposed.children if child.id in allowed
                    and child.parent_id == parent.id
                    and (child.evidence_type not in {"destination", "category", "feature"} or
                         (child.target if child.evidence_type == "destination" else
                          child.category if child.evidence_type == "category" else
                          destination_feature_id(child.feature_id) and child.feature_id in (
                              {fid for fid in parent.related_features
                               if destination_feature_id(fid)} |
                              ({parent.feature_id} if parent.feature_id else set()))))]
        if not children:
            return goal
        return goal.model_copy(update={
            "subgoals": goal.subgoals + children[:min(2, 8 - len(goal.subgoals))],
            "goal_coverage": coverage + [GoalCoverage(
                goal_id=child.id, status="pending", reason="نیازمند بررسی پس از تجزیهٔ هدف.")
                for child in children[:min(2, 8 - len(goal.subgoals))]],
        })
    except Exception:
        logger.warning("Adaptive decomposition unavailable; keeping original goals", exc_info=True)
        return goal


@lru_cache
def _designer() -> Agent:
    from src.agent.agent import _build_model
    from src.config import get_settings

    return Agent(_build_model(get_settings()), output_type=GoalDesign, system_prompt=(
        "Revise the existing session goal design, NOT an itinerary. Return structured output. "
        "Use current_request and current_goal as authoritative for THIS trip; older preferences "
        "are soft hints only. Build a small set of ATOMIC goal dimensions using the "
        "destination ontology, including the user's stated dimensions and relevant "
        "unstated ones ONLY when supported by prior preferences or clearly marked "
        "optional assumptions. Do not manufacture a hard requirement because a "
        "vocabulary node exists. All supplied ontology feature nodes are candidate dimensions, "
        "NOT 100+ compulsory goals. Consider every feature type, choose only relevant goals "
        "and add sensible optional dimensions the request omitted; use actual user preference "
        "scores when relevant. Link related_features to exact ontology ids, never invent ids. "
        "If a current explicit request conflicts with an older preference, replace the "
        "preference goal for this trip only, including a previously generated subgoal. "
        "Do not merge several independent requirements into one goal: a Category:nature "
        "edge verifies nature, NOT camping, excitement or facilities. BEST_SEASON:autumn "
        "does not prove autumn-colored scenery. Each goal's name and success criterion "
        "must match what its evidence_type actually proves. Never turn an assumption "
        "into an explicit fact. "
        "Preserve existing subgoal IDs, priorities and user constraints unless the CURRENT "
        "request explicitly replaces or removes them; use retired_goal_ids and "
        "retired_constraints for explicit removals, not omissions. Return only changed or "
        "new subgoals/constraints; the server merges unchanged ones. Include up to 8 "
        "concrete subgoals with stable ids G1..G8, "
        "a measurable minimum where justified, and hard/soft constraints. "
        "Use evidence_type destination for explicitly named places, category for a graph "
        "category (set category), feature ONLY for an actual destination-linked "
        "feature_id (Category/Season/TripType/Vehicle/FacilitiesLevel/TravelTime/"
        "RoadType/PhysicalReadiness/DistanceRange), route for route feasibility, "
        "or unverified for food, quietness, rest, "
        "opening hours, prices and anything not reliably checkable from places/route. "
        "For destination evidence_type you MUST set target to the exact named place; "
        "for category evidence_type you MUST set category; for feature set an "
        "exact id from available_feature_nodes. Vocabulary-only nodes (Environment, "
        "FoodPreference, TravelStyle, Activity, AccommodationType) are NOT "
        "destination evidence. A nature category does not "
        "prove forest or autumn colors; a car route never proves low walking, toilets, "
        "rest facilities or accessibility. Mark those unverified without explicit evidence. "
        "A departure city assumed from defaults "
        "is NOT an explicit user instruction. "
        "Do not invent confirmed hours, prices, visit lengths, or activities. "
        "List at most 4 suggested evidence checks selecting from graph search, "
        "ontology-edge search/details, semantic search, Neshan routing/reachability "
        "and nearby amenities. Subgoals are desired outcomes, not tool calls; "
        "the executor chooses and revises tool calls after seeing each result. "
        "Do not require every tool or a fixed itinerary template."
    ))


async def design_goals(goal: TravelGoal, request: str, transcript: str,
                       preferences: list[dict], features: list[dict] | None = None) -> TravelGoal:
    ontology = features or []
    applicable_preferences, active_ids = _effective_preferences(goal, preferences, ontology)
    payload = {
        "current_request": request,
        "current_goal": goal.model_dump(exclude={"recommended_routes", "last_proposed_routes",
                                                  "last_itineraries"}),
        "conversation_context": transcript[-2500:],
        "user_preferences": applicable_preferences,
        "current_trip_feature_ids": {label: sorted(ids) for label, ids in active_ids.items()},
        "available_feature_nodes": [
            {"id": f["id"], "type": f["label"], "name": f["name"],
             "description": f.get("description", ""),
             "destination_linked": f["label"] in DESTINATION_FEATURE_RELATIONS}
            for f in ontology
        ],
    }
    try:
        result = await _designer().run(json.dumps(payload, ensure_ascii=False, default=str))
        design = GoalDesign.model_validate(result.output)
        known_feature_ids = {f["id"] for f in ontology}
        linked_feature_ids = {f["id"] for f in ontology
                              if f["label"] in DESTINATION_FEATURE_RELATIONS}
        removed = set(design.retired_goal_ids)
        feature_types = {f["id"]: f["label"] for f in ontology}
        feature_names = {f["id"]: normalized(f["name"]) for f in ontology}
        avoided_names = {normalized(name) for name in goal.avoid}
        def conflicts_with_current_trip(item: PlanningSubgoal) -> bool:
            if item.source != "preference":
                return False
            for fid in [item.feature_id, *item.related_features]:
                label = feature_types.get(fid)
                if feature_names.get(fid) in avoided_names:
                    return True
                if label in active_ids and fid not in active_ids[label]:
                    return True
            return False

        subgoals = [s.model_copy(deep=True) for s in goal.subgoals if s.id not in removed]
        ids = {s.id: i for i, s in enumerate(subgoals)}
        for changed in design.subgoals[:8]:
            if changed.id in ids:
                subgoals[ids[changed.id]] = changed
            elif changed.id not in removed and len(subgoals) < 8:
                ids[changed.id] = len(subgoals)
                subgoals.append(changed)
        for subgoal in subgoals:
            subgoal.related_features = [fid for fid in subgoal.related_features
                                        if fid in known_feature_ids]
            if subgoal.evidence_type == "destination" and not subgoal.target:
                # Do not silently claim a named visit just because its label
                # appeared in a generated subgoal.
                subgoal.evidence_type = "unverified"
            if subgoal.evidence_type == "category" and not subgoal.category:
                subgoal.evidence_type = "unverified"
            if subgoal.evidence_type == "feature" and subgoal.feature_id not in linked_feature_ids:
                subgoal.evidence_type = "unverified"
                subgoal.feature_id = None
            if subgoal.source == "explicit" and subgoal.evidence_type == "route":
                if goal.origin == "تهران" and "تهران" not in request and "Tehran" not in request:
                    subgoal.source = "assumption"
            if subgoal.evidence_type == "route" and any(
                    word in subgoal.name for word in ("پیاده", "سرویس بهداشتی", "امکانات", "دسترس‌پذیر")):
                subgoal.evidence_type = "unverified"
        subgoals = [s for s in subgoals if not conflicts_with_current_trip(s)]
        constraints = [c.model_copy(deep=True) for c in goal.planning_constraints
                       if c.description not in set(design.retired_constraints)]
        for constraint in constraints:
            if (constraint.source == "explicit" and "تهران" in constraint.description
                    and "تهران" not in goal.objective and "تهران" not in request):
                constraint.source = "assumption"
        for constraint in design.constraints[:12]:
            if (constraint.source == "explicit" and goal.origin == "تهران"
                    and "تهران" not in request and "Tehran" not in request
                    and "تهران" in constraint.description):
                constraint.source = "assumption"
            if constraint.description not in {c.description for c in constraints}:
                constraints.append(constraint)
        return goal.model_copy(update={
            "subgoals": subgoals,
            "planning_constraints": constraints[:16],
            "planning_steps": design.steps[:4] or goal.planning_steps,
            "goal_coverage": (goal.goal_coverage if subgoals == goal.subgoals
                              and constraints == goal.planning_constraints else []),
        })
    except Exception:
        logger.warning("Goal decomposition unavailable; preserving explicit trip goal", exc_info=True)
        return goal.model_copy(update={"goal_coverage": []})


async def evaluate_plan(goal: TravelGoal, routes: list[dict], graph=None) -> list[GoalCoverage]:
    """Only finalized route stops, never searched candidates or model prose, count as evidence."""
    route_stops = [list(dict.fromkeys(str(s["name"]) for s in route.get("stops", [])
                                     if s.get("name"))) for route in routes]
    stops = list(dict.fromkeys(s for group in route_stops for s in group))
    real_routes = [r for r in routes if r.get("used_real_routing") and not r.get("unresolved_stops")]
    categories: dict[str, set[str]] = {}
    features: dict[str, set[str]] = {}
    if graph is not None:
        for name in stops:
            try:
                details = await graph.get_destination_details(name)
                if details:
                    categories[name] = set(details.get("categories") or [])
                if hasattr(graph, "get_destination_features"):
                    features[name] = {item["id"] for item in await graph.get_destination_features(name)
                                      if item.get("id")}
            except Exception:
                logger.warning("Could not verify category for %s", name, exc_info=True)
    coverage = []
    for subgoal in goal.subgoals:
        matches: list[str] = []
        reason = ""
        if subgoal.evidence_type == "destination" and subgoal.target:
            matches = max(([s for s in group if s.casefold() == subgoal.target.casefold()]
                           for group in route_stops), key=len, default=[])
        elif subgoal.evidence_type == "category" and subgoal.category:
            matches = max(([s for s in group if subgoal.category in categories.get(s, set())]
                           for group in route_stops), key=len, default=[])
            if not categories:
                reason = "دسته‌بندی مقصدها در گراف قابل تأیید نبود."
        elif subgoal.evidence_type == "feature" and subgoal.feature_id:
            matches = max(([s for s in group if subgoal.feature_id in features.get(s, set())]
                           for group in route_stops), key=len, default=[])
            if not matches:
                reason = "رابطهٔ ویژگی با مقصدهای مسیر نهایی در گراف تأیید نشد."
        elif subgoal.evidence_type == "route":
            matches = max(([stop["name"] for stop in route.get("stops", [])
                            if stop.get("name")] for route in real_routes), key=len, default=[])
            if not matches:
                reason = "مسیر واقعی با نشان تأیید نشده است."
        else:
            reason = "دادهٔ معتبر برای تأیید این هدف (مثل ساعت، هزینه یا فعالیت) در دسترس نیست."
        status = ("satisfied" if len(matches) >= subgoal.minimum else
                  "partial" if matches else "pending")
        coverage.append(GoalCoverage(goal_id=subgoal.id, status=status,
                                     activities=matches, reason=reason or (
                                         "فقط در برنامهٔ پیشنهادی پوشش داده شده؛ انجام واقعی تأیید نشده است."
                                         if matches else "در مسیر نهایی شواهدی یافت نشد.")))
    return coverage


def replan_feedback(goal: TravelGoal, coverage: list[GoalCoverage]) -> str | None:
    missing = [c for c in coverage if c.status != "satisfied"
               and any(g.id == c.goal_id and g.priority == "high"
                       and g.evidence_type != "unverified" for g in goal.subgoals)]
    if not missing:
        return None
    return ("یک بار برنامه را اصلاح کن: اهداف اولویت‌دار زیر هنوز در مسیر نهایی پوشش ندارند: "
            + json.dumps([{"goal": next(g.name for g in goal.subgoals if g.id == c.goal_id),
                           "reason": c.reason} for c in missing], ensure_ascii=False)
            + ". اهداف پوشش‌داده‌شده را حفظ کن؛ فقط با ابزارهای گراف و نشان شواهد واقعی بیاور. "
            "اگر ممکن نیست، صریحاً بگو و ادعای پوشش نکن.")
