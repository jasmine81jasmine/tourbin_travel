"""Bounded, evidence-first progress tracking inside the planning tool loop.

Candidate evidence is never a completed visit or a verified final itinerary.
The final route evaluator remains the authority for satisfied goals.
"""

import json
import logging
from uuid import uuid4

from src.agent.goals import GoalCoverage, TravelGoal

logger = logging.getLogger(__name__)


def observe(goal: TravelGoal, tool: str, result: object) -> bool:
    """Record candidate evidence without treating search results as final coverage."""
    if not goal.subgoals:
        return False
    rows = (result if isinstance(result, list) else
            result.get("candidates", [result]) if isinstance(result, dict) else [])
    candidates = [row for row in rows if isinstance(row, dict) and
                  (row.get("name") or row.get("categories")) and not row.get("error")]
    previous = {entry.goal_id: entry for entry in goal.goal_coverage}
    changed = False
    for subgoal in goal.subgoals:
        if subgoal.evidence_type not in {"destination", "category", "feature", "route"}:
            continue
        matches = []
        if tool == "get_destination_features" and isinstance(result, dict):
            if subgoal.evidence_type == "feature" and subgoal.feature_id and any(
                    feature.get("id") == subgoal.feature_id
                    for feature in result.get("features", [])):
                matches = [str(result.get("destination", ""))]
        elif tool in {"search_destinations", "semantic_search_destinations",
                    "get_destination_details", "find_destinations_near"}:
            for row in candidates:
                if subgoal.evidence_type == "destination" and subgoal.target and (
                        str(row.get("name", "")).casefold() == subgoal.target.casefold()):
                    matches.append(str(row["name"]))
                elif subgoal.evidence_type == "category" and subgoal.category and (
                        subgoal.category in (row.get("categories") or [])):
                    matches.append(str(row.get("name", "")))
        elif tool == "search_destinations_by_features":
            if subgoal.evidence_type == "feature" and subgoal.feature_id:
                matches = [str(row["name"]) for row in candidates if
                           subgoal.feature_id in (row.get("matched_feature_ids") or [])]
        elif tool == "build_trip_map" and subgoal.evidence_type == "route" and (
                isinstance(result, dict) and result.get("used_real_routing")
                and not result.get("unresolved_stops")):
            matches = [str(stop["name"]) for stop in result.get("stops", [])
                       if isinstance(stop, dict) and stop.get("name")]
        if not matches:
            continue
        old = previous.get(subgoal.id)
        evidence = list(dict.fromkeys((old.activities if old else []) + matches))[:10]
        if old and (old.status == "satisfied" or
                    (old.status == "partial" and old.activities == evidence)):
            continue
        previous[subgoal.id] = GoalCoverage(
            goal_id=subgoal.id, status="partial", activities=evidence,
            reason=f"شواهد اولیه از {tool}؛ فقط برنامهٔ نهایی می‌تواند هدف را تأیید کند.")
        changed = True
    if changed:
        goal.goal_coverage = [previous.get(g.id, GoalCoverage(
            goal_id=g.id, status="pending", reason="هنوز شواهدی بررسی نشده است."))
            for g in goal.subgoals]
    return changed


async def observed_result(ctx, tool: str, result: object, *, source: str = "agent_tool") -> str:
    """Persist an immutable snapshot before returning evidence to the next model call."""
    payload = json.dumps(result, default=str, ensure_ascii=False)
    deps = ctx.deps
    if deps.turn_action != "plan" or deps.travel_goal is None:
        return payload
    # PydanticAI can run several tool calls in one model turn. Serialize
    # snapshot writes so their revision order matches observed progress.
    async with deps.progress_lock:
        observation_id = str(uuid4())
        deps.observed_tools.append(tool)
        observe(deps.travel_goal, tool, result)
        deps.travel_goal.last_observation_tool = tool
        deps.travel_goal.revision_cause = "tool_observation"
        deps.travel_goal.observation_id = observation_id
        deps.travel_goal.observation_source = source
        rows = result if isinstance(result, list) else (
            result.get("candidates", result.get("stops", [result]))
            if isinstance(result, dict) else [])
        deps.travel_goal.observation_summary = [
            str(row["name"]) for row in rows
            if isinstance(row, dict) and row.get("name")][:10]
        if deps.graph is not None:
            try:
                revision = await deps.graph.upsert_travel_goal(
                    deps.session_id, deps.user_id, deps.travel_goal.model_dump())
                deps.travel_goal.revision = revision
                if source == "agent_tool":
                    deps.observation_ids.append(observation_id)
            except Exception:
                logger.warning("Could not persist goal progress after %s", tool, exc_info=True)
        # Tool output is guaranteed visible on the next model request.
        payload += "\nGoal progress (candidate evidence only, not final coverage): " + json.dumps(
            [entry.model_dump() for entry in deps.travel_goal.goal_coverage],
            ensure_ascii=False)
        # The trace library stores this tool return verbatim. A unique marker
        # links the exact invocation to its revision without guessing by time.
        if source == "agent_tool":
            payload += "\nGoal observation id: " + observation_id
    return payload
