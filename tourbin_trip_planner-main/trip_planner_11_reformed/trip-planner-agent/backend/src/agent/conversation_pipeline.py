"""OnGoal-inspired infer → merge → evaluate for one session's dialogue goals.

These are commitments in the conversation, not ontology features, visited
destinations or verified route coverage. Neo4j revisions retain the timeline.
"""

import json
import logging
import re
from functools import lru_cache

from pydantic_ai import Agent

from src.agent.goals import (ConversationalEvaluation, ConversationalGoal,
                             GoalMergeEvent, TravelGoal)
from src.agent.turns import TurnDecision, _UPDATABLE

logger = logging.getLogger(__name__)


def _clauses(message: str, decision: TurnDecision) -> list[dict]:
    """Accept only verbatim user clauses; a short acceptance keeps its own text."""
    inferred = []
    for row in decision.inferred_clauses[:5]:
        clause = row.get("clause")
        if not isinstance(clause, str) or not clause.strip() or clause.strip() not in message:
            continue
        kind = row.get("kind")
        if kind not in {"question", "request", "offer", "suggestion"}:
            continue
        scope = row.get("scope")
        raw_dimensions = row.get("dimensions")
        dimensions = [key for key in (raw_dimensions if isinstance(raw_dimensions, list) else []) if
                      isinstance(key, str) and key in decision.updates and key in _UPDATABLE]
        inferred.append({"clause": clause.strip(),
                         "summary": str(row.get("summary") or clause).strip()[:240],
                         "kind": kind,
                         "scope": scope if scope in {"trip", "turn"} else "turn",
                         "dimensions": list(dict.fromkeys(dimensions))})
    if not inferred and decision.action != "chat":
        # A failed classifier must not silently drop this turn. Without
        # inferred clauses, don't invent a new lasting trip preference.
        inferred = [{"clause": message.strip(), "summary": message.strip()[:240],
                     "kind": "question" if decision.action == "answer" else "request",
                     "scope": "trip" if decision.action == "plan" else "turn",
                     "dimensions": list(decision.updates)}]
    return inferred


def _compatible(old: ConversationalGoal, new: dict, current: TravelGoal,
                updates: dict, operation: str) -> bool:
    if old.scope != "trip" or new["scope"] != "trip":
        return False
    shared = set(old.dimensions) & set(new["dimensions"])
    if not shared:
        if operation == "replace" and "avoid" in new["dimensions"]:
            avoided = updates.get("avoid")
            if isinstance(avoided, list):
                old_values = [value for key in old.dimensions
                              if key in {"categories", "trip_types", "moods",
                                         "activities", "accommodation", "must_haves"}
                              for value in getattr(current, key)]
                return any(value in avoided and value in old.clause
                           and value in new["clause"] for value in old_values)
        return False
    def extends_list(field: str) -> bool:
        before, after = getattr(current, field), updates[field]
        return (isinstance(before, list) and isinstance(after, list)
                and all(isinstance(item, str) for item in [*before, *after])
                and set(before).issubset(set(after)))

    if operation == "replace":
        return any(
            getattr(current, key) != updates[key] and not extends_list(key)
            for key in shared)
    if operation == "combine":
        return any(
            getattr(current, key) == updates[key] or extends_list(key)
            for key in shared)
    return False


def infer_and_merge(current: TravelGoal, message: str, decision: TurnDecision) -> TravelGoal:
    """Keep old goals unless a validated explicit update replaces or combines one."""
    goal = current.model_copy(deep=True)
    proposals = _clauses(message, decision)
    active = {g.id: g for g in goal.conversational_goals if g.status == "active"}
    events: list[GoalMergeEvent] = []
    touched: set[str] = set()
    for index, proposed in enumerate(proposals):
        new_id = f"C{goal.next_conversational_goal}"
        goal.next_conversational_goal += 1
        operation, old_ids = "add", []
        declared = next((item for item in decision.merge_operations
                         if item.get("new_index") == index
                         and item.get("operation") in {"replace", "combine"}), None)
        if declared and isinstance(declared.get("old_ids"), list):
            candidate = [active[oid] for oid in declared["old_ids"]
                         if isinstance(oid, str) and oid in active and oid not in touched]
            if candidate and all(_compatible(old, proposed, current, decision.updates,
                                             declared["operation"]) for old in candidate):
                operation, old_ids = declared["operation"], [old.id for old in candidate]
        # Safe deterministic merge if the model omitted an operation but
        # explicitly changed a typed field from the prior goal.
        if not old_ids:
            for old in active.values():
                if old.id in touched:
                    continue
                if _compatible(old, proposed, current, decision.updates, "replace"):
                    operation, old_ids = "replace", [old.id]
                    break
                if _compatible(old, proposed, current, decision.updates, "combine"):
                    operation, old_ids = "combine", [old.id]
                    break
        for oid in old_ids:
            active[oid].status = "superseded"
            touched.add(oid)
        goal.conversational_goals.append(ConversationalGoal(
            id=new_id, parents=old_ids, **proposed))
        events.append(GoalMergeEvent(operation=operation, old_ids=old_ids, new_id=new_id))
        if operation in {"replace", "combine"}:
            # A clause may encode several independent requirements ("two
            # people in autumn"). Replacing autumn or combining a new
            # preference must not erase the other dimensions from the graph.
            for oid in old_ids:
                old = active[oid]
                for dimension in old.dimensions:
                    if dimension in proposed["dimensions"]:
                        continue
                    carried_id = f"C{goal.next_conversational_goal}"
                    goal.next_conversational_goal += 1
                    goal.conversational_goals.append(ConversationalGoal(
                        id=carried_id, clause=old.clause,
                        summary=f"{dimension}: {getattr(current, dimension)}",
                        kind=old.kind, scope="trip", dimensions=[dimension],
                        parents=[old.id]))
                    events.append(GoalMergeEvent(
                        operation="keep", old_ids=[old.id], new_id=carried_id))
    events.extend(GoalMergeEvent(operation="keep", old_ids=[old.id])
                  for old in active.values() if old.id not in touched)
    goal.goal_merge_events = events
    # Old unresolved *questions* remain in immutable revisions, but must not
    # crowd out persistent trip requirements in a long planning conversation.
    active_turns = [g for g in goal.conversational_goals
                    if g.scope == "turn" and g.status == "active"]
    for older in active_turns[:-12]:
        older.status = "archived"
    # Prior immutable revisions keep older detail. Never evict active trip
    # constraints just to meet an in-memory size limit.
    if len(goal.conversational_goals) > 48:
        inactive = [g for g in goal.conversational_goals
                    if g.status in {"addressed", "superseded", "archived"}]
        prune = {g.id for g in inactive[:len(goal.conversational_goals) - 48]}
        goal.conversational_goals = [g for g in goal.conversational_goals
                                    if g.id not in prune]
    goal.conversational_evaluations = []
    return goal


@lru_cache
def _evaluator() -> Agent:
    from src.agent.agent import _build_model
    from src.config import get_settings

    return Agent(_build_model(get_settings()), system_prompt=(
        "Independently evaluate how the ACTUAL assistant reply addressed "
        "each ACTIVE user conversational goal. Return ONLY JSON: "
        "{\"evaluations\":[{\"goal_id\":\"C1\",\"category\":"
        "\"confirm|contradict|ignore\",\"explanation\":\"one sentence\","
        "\"examples\":[\"exact verbatim quote from reply\"]}]}. "
        "Confirm only when the answer actually fulfills the goal; a place name "
        "mentioned without a plan does not fulfill a request for a plan. "
        "Contradict only when the reply conflicts with the goal; missing content "
        "is ignore. An itinerary, amenity or driving claim is not verified by "
        "mere model prose: consider verified_route_stops and map evidence. "
        "Evaluate turn-scoped questions from this turn, but do not treat an "
        "assistant-offered destination as a user's accepted trip constraint. "
        "Cite exact phrases from the supplied reply only, never hallucinate quotes."
    ))


async def evaluate_response(goal: TravelGoal, reply: str, *,
                            verified_route_stops: list[str] | None = None,
                            use_llm: bool = True) -> TravelGoal:
    """Evaluate response alignment separately from final itinerary coverage."""
    active = [g for g in goal.conversational_goals if g.status == "active"]
    if not active:
        return goal
    answers: dict[str, ConversationalEvaluation] = {}
    if use_llm:
        payload = {
            "active_goals": [g.model_dump() for g in active[-20:]],
            "reply": reply[:6500],
            "verified_route_stops": verified_route_stops or [],
        }
        try:
            result = await _evaluator().run(json.dumps(payload, ensure_ascii=False))
            text = result.output.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
            rows = json.loads(text).get("evaluations", [])
            allowed = {g.id: g for g in active}
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict) or row.get("goal_id") not in allowed:
                    continue
                examples = row.get("examples")
                examples = ([part for part in examples[:3]
                             if isinstance(part, str) and part.strip() and part in reply]
                            if isinstance(examples, list) else [])
                category = row.get("category")
                if category not in {"confirm", "contradict"} or not examples:
                    category, examples = "ignore", []
                target = allowed[row["goal_id"]]
                if (category == "confirm" and target.scope == "trip"
                        and "destination_names" in target.dimensions
                        and goal.destination_names and not all(
                            name in reply for name in goal.destination_names)):
                    category, examples = "ignore", []
                answers[target.id] = ConversationalEvaluation(
                    goal_id=target.id, category=category,
                    explanation=str(row.get("explanation") or "")[:350],
                    examples=examples)
        except Exception:
            logger.warning("Dialogue goal response evaluation unavailable", exc_info=True)
    goal.conversational_evaluations = [
        answers.get(g.id, ConversationalEvaluation(
            goal_id=g.id, category="ignore", explanation="در پاسخ شاهد کافی یافت نشد."))
        for g in active]
    for g in active:
        if g.scope == "turn" and answers.get(g.id) and answers[g.id].category == "confirm":
            g.status = "addressed"
    return goal
