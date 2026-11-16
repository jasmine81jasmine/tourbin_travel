"""Follow-up questions must bypass itinerary templates and preserve session goals."""

import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi import Response

from src.agent.goals import TravelGoal
from src.agent import turns
from src.agent.agent import update_travel_goal
from src.api.schemas import ChatRequest


def test_plan_reply_greets_before_retrospective_reasons_and_keeps_goal_status_private():
    route = importlib.import_module("src.api.routes.chat")
    from src.api.schemas import ChatResponse

    reply = route._compose_plan_reply(
        "سلام! این مسیر در دل جنگل است.\n\nروز اول راهی جنگل می‌شویم.",
        "### چرا این سفر را پیشنهاد کردم؟\n\nبرای طبیعت‌گردی پیشنهادش دادم.",
        first_turn=True)
    assert reply.index("سلام!") < reply.index("چرا این سفر") < reply.index("روز اول")
    assert "goal_coverage" not in ChatResponse.model_fields
    assert route._compose_plan_reply("گزینه‌ها", "چرایی", first_turn=False) == "چرایی\n\nگزینه‌ها"
    assert route._compose_plan_reply("یک مقصد را بررسی می‌کنم.", "", first_turn=True).startswith("سلام!")


@pytest.mark.asyncio
async def test_llm_decides_followup_is_an_answer_and_keeps_duration(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            data = json.loads(prompt)
            assert data["current_goal"]["duration"] == "یک روز"
            assert "درکه" in data["recent_conversation"]
            return SimpleNamespace(output='{"action":"answer","updates":{}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    goal = TravelGoal(duration="یک روز", origin="تهران")
    decision = await turns.understand_turn("کدومشون خلوت تره؟", goal, "توربین: درکه و دربند")
    assert decision.action == "answer"
    assert turns.apply_goal_updates(goal, decision.updates).duration == "یک روز"


@pytest.mark.asyncio
async def test_llm_can_select_destination_without_erasing_existing_trip(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"action":"answer","updates":{"destination_names":["روستای ایگل"]}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    goal = TravelGoal(duration="دو روز", origin="تهران", accommodation=["کمپ"])
    decision = await turns.understand_turn("کجا کمپ کنم ایگل؟", goal, "توربین: روستای ایگل")
    updated = turns.apply_goal_updates(goal, decision.updates)
    assert decision.action == "answer"
    assert updated.destination_names == ["روستای ایگل"]
    assert updated.duration == "دو روز" and updated.accommodation == ["کمپ"]


@pytest.mark.asyncio
async def test_revised_constraints_preserve_unchanged_session_preferences(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"action":"plan","updates":{"duration":"سه روز","region":"شمال"}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    goal = TravelGoal(duration="دو روز", origin="تهران", companions=["کودک"], accommodation=["کمپ"])
    decision = await turns.understand_turn("سه روز وقت دارم، شمال رو ترجیح میدم", goal, "توربین: برنامهٔ قبلی")
    updated = turns.apply_goal_updates(goal, decision.updates)
    assert decision.action == "plan"
    assert updated.duration == "سه روز" and updated.region == "شمال"
    assert updated.origin == "تهران" and updated.companions == ["کودک"]
    assert updated.accommodation == ["کمپ"]


@pytest.mark.asyncio
async def test_vehicle_or_reason_question_is_classified_by_llm_without_new_plan(monkeypatch):
    prompts = []

    class Classifier:
        async def run(self, prompt):
            prompts.append(json.loads(prompt)["latest_message"])
            return SimpleNamespace(output='{"action":"answer","updates":{}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: Classifier())
    goal = TravelGoal(duration="سه روز", recommended_routes=[["آبشار برجی کلا"]])
    assert (await turns.understand_turn("با چه ماشینی برم اونجا؟", goal, "برنامهٔ آبشار برجی کلا")).action == "answer"
    assert (await turns.understand_turn("چرا این برنامه ها رو بهم دادی؟", goal, "برنامهٔ آبشار برجی کلا")).action == "answer"
    assert (await turns.understand_turn("در مورد آبشار برجی کلا بیشتر بگو", goal, "برنامهٔ آبشار برجی کلا")).action == "answer"
    assert len(prompts) == 3


@pytest.mark.asyncio
async def test_llm_replans_when_new_family_facility_constraints_invalidate_hiking_plan(monkeypatch):
    class Classifier:
        async def run(self, prompt):
            assert "جنگل" in json.loads(prompt)["current_goal"]["last_proposed_routes"][0]
            return SimpleNamespace(output=json.dumps({
                "action": "plan",
                "updates": {"companions": ["خانواده"],
                            "must_haves": ["سرویس بهداشتی"],
                            "constraints": ["پیاده‌روی کم"]}}, ensure_ascii=False))

    monkeypatch.setattr(turns, "_turn_agent", lambda: Classifier())
    goal = TravelGoal(duration="یک روز", categories=["طبیعت"], season="پاییز",
                      last_proposed_routes=[["جنگل"]])
    decision = await turns.understand_turn(
        "با خانواده میرم، سرویس بهداشتی و پیاده‌روی کم میخوام", goal, "توربین: جنگل")
    updated = turns.apply_goal_updates(goal, decision.updates)
    assert decision.action == "plan"
    assert updated.season == "پاییز" and updated.categories == ["طبیعت"]
    assert updated.companions == ["خانواده"]
    assert updated.must_haves == ["سرویس بهداشتی"]


def test_classifier_failure_favors_answering_questions():
    assert turns._fallback_action("کدوم گزینه خلوت‌تره؟", True) == "answer"
    assert turns._fallback_action("گلابدره آب و هواش چطوره؟", True) == "answer"
    assert turns._fallback_action("کجا کمپ کنم ایگل؟", True) == "answer"
    assert turns._fallback_action("چند گزینه نزدیک تر پیشنهاد بده", True) == "plan"


def test_more_plans_is_not_a_factual_question_and_offer_can_be_a_question():
    route = importlib.import_module("src.api.routes.chat")
    assert route._requests_more_plans("دیگه برنامه چی داری ما ماشین معمولی داریم")
    assert route._requests_more_plans("برنامه‌های دیگه چی داری؟")
    assert not route._requests_more_plans("برای همین مقصد امکانات دیگه‌ای هم هست؟")
    previous = TravelGoal(last_proposed_routes=[["جنگل الف"]])
    assert route._requests_more_plans(
        "دیگه چی داری؟ آفرود داریم", previous,
        "### برنامهٔ سفر\nجنگل الف برای سفر مناسب است.")
    assert not route._requests_more_plans(
        "دیگه چی داری؟", previous, "امروز هوا خوب است.")
    assert route._offers_plan(
        "آیا تمایل دارید یکی از این مقصدها را برای برنامه دو روزه انتخاب کنیم "
        "یا ترکیبی از آن‌ها را برای شما برنامه‌ریزی کنم؟")


def test_explicit_vehicle_is_resolved_from_graph_vocabulary_not_a_place_name():
    names = ["آفرود", "سواری"]
    assert turns.explicit_vehicle_from_vocabulary(
        "دیگه چی داری؟ افرود داریم", names) == "آفرود"
    assert turns.explicit_vehicle_from_vocabulary(
        "ماشین آفرود دارم", names) == "آفرود"
    assert turns.explicit_vehicle_from_vocabulary(
        "آفرود لازمه؟", names) is None
    assert turns.explicit_vehicle_from_vocabulary(
        "آفرود نمی‌خواهیم", names) is None
    assert turns.explicit_vehicle_from_vocabulary(
        "دو روزه جوونی می‌خوام", names) is None


@pytest.mark.asyncio
async def test_mixed_verified_offers_are_preserved_but_not_silently_accepted(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    from src.agent.goals import PendingOffer

    reply = ("1. دریاچه الف\n2. دشت ب\n"
             "آیا می‌خواهید یکی از این مقصدها را انتخاب کنید یا ترکیبی از "
             "آن‌ها را برایتان برنامه‌ریزی کنم؟")

    class Graph:
        async def find_destinations_mentioned_in_text(self, text, limit=16):
            return ["دریاچه الف"] if "دریاچه الف" in text else []

        async def get_destination_details(self, name):
            return {"name": name} if name == "دریاچه الف" else None

    async def infer(*args):
        return PendingOffer(action="choose",
                            destination_names=["دریاچه الف", "دشت ب"],
                            source_reply=reply)

    monkeypatch.setattr(route, "infer_pending_offer", infer)
    pending = await route._capture_pending_offer("سفر دو روزه", reply, Graph())
    assert pending is not None
    assert pending.offered_names == ["دریاچه الف", "دشت ب"]
    assert pending.destination_names == ["دریاچه الف"]
    assert pending.unverified_names == ["دشت ب"]
    assert await route._resolve_accepted_plan_offer(
        f"کاربر: سفر دو روزه\nتوربین: {reply}", Graph()) == ([], True)
    assert await route._resolve_offered_option(
        "دومی رو بده", TravelGoal(pending_offer=pending),
        f"کاربر: سفر دو روزه\nتوربین: {reply}", Graph()) is None
    clarification = "از میان «دریاچه الف، دشت ب» بگو کدام را می‌خواهی."
    pending.source_reply = clarification
    assert await route._resolve_offered_option(
        "دومی رو بده", TravelGoal(pending_offer=pending),
        f"کاربر: آره\nتوربین: {clarification}", Graph()) is None
    assert await route._resolve_offered_option(
        "اولی رو بده", TravelGoal(pending_offer=pending),
        f"کاربر: آره\nتوربین: {clarification}", Graph()) == "دریاچه الف"
    # An older snapshot without a pending offer must not map "second" to
    # the wrong destination when graph lookup misses one numbered option.
    assert await route._resolve_offered_option(
        "دومی رو بده", TravelGoal(),
        f"کاربر: سفر دو روزه\nتوربین: {reply}", Graph()) is None


@pytest.mark.asyncio
async def test_three_turn_session_resolves_offered_option_and_keeps_trip_constraints(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    from src.memory.graph import TripGraphRepository as RealGraph
    states = {}
    snapshots = []
    messages = []
    offered = ["روستای آهار", "روستا آبنیک", "روستای لالان", "رودخانه دربند", "رودخانه گلابدره"]

    class Graph:
        _destination_lookup_keys = staticmethod(RealGraph._destination_lookup_keys)

        def __init__(self, *args):
            pass

        async def get_travel_goal(self, session_id, user_id):
            return states.get((session_id, user_id))

        async def upsert_travel_goal(self, session_id, user_id, state):
            states[(session_id, user_id)] = json.loads(json.dumps(state))
            snapshots.append(states[(session_id, user_id)])
            return len(snapshots)

        async def sync_session_features(self, *args):
            pass

        async def get_previous_recommendations(self, *args):
            return {"ids": set(), "names": set()}

        async def get_recent_session_features(self, *args):
            return {}

        async def get_user_history(self, *args, **kwargs):
            return {"liked": []}

        async def record_shown_destinations(self, *args):
            pass

        async def find_destinations_mentioned_in_text(self, text, limit=6):
            return [name for name in sorted(
                ["جنگل کارا", *offered], key=lambda name: text.find(name))
                if name in text][:limit]

        async def get_destination_details(self, name):
            if name in offered or name == "جنگل کارا":
                return {"name": name, "categories": ["طبیعت"],
                        "description": f"{name} در دامنه کوه قرار دارد و منظره‌ای طبیعی دارد."}
            return None

        async def get_vehicle_names(self):
            return ["آفرود", "سواری"]

    class Planner:
        async def run(self, prompt, *, deps, message_history):
            if "دو نفره" in prompt:
                text = "برنامهٔ سفر: جنگل کارا برای سفر پاییزی"
            elif "دیگه" in prompt:
                assert deps.travel_goal.transport == "آفرود"
                text = "\n".join(f"{index}. {name}" for index, name in enumerate(offered, 1))
            else:
                assert deps.travel_goal.destination_names == ["روستای آهار"]
                assert deps.travel_goal.group_size == 2
                assert deps.travel_goal.season == "پاییز"
                assert "پیاده‌روی" in deps.travel_goal.activities
                assert deps.travel_goal.transport == "آفرود"
                text = "به‌جایش به روستای لالان بروید؛ سرویس بهداشتی و مسیر آسان دارد."
            return SimpleNamespace(output=text, all_messages=lambda: [], new_messages=lambda: [])

    async def classify(message, goal, transcript):
        if "دو نفره" in message:
            return turns.TurnDecision("plan", {"group_size": 2, "season": "پاییز"})
        if "دیگه" in message:
            assert goal.group_size == 2 and goal.season == "پاییز"
            # Even if the classifier mistakes "other plans" for a factual
            # question and promotes a prior assistant suggestion, the
            # tool-using planner must search new options, not select it.
            return turns.TurnDecision("chat", {
                "activities": ["پیاده‌روی"], "destination_names": ["روستای لالان"]})
        # Even a misclassified choice must be interpreted from the last offer.
        return turns.TurnDecision("answer", selected_offer_names=["روستای آهار"])

    async def conversation(session_id, **kwargs):
        return SimpleNamespace(messages=messages)

    async def add_message(*, role, content, **kwargs):
        messages.append(SimpleNamespace(role=SimpleNamespace(value=role), content=content))
        return None

    async def design(goal, *args):
        return goal

    async def no_explanation(*args, **kwargs):
        return ""

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             short_term=SimpleNamespace(get_conversation=conversation,
                                                        add_message=add_message),
                             reasoning=SimpleNamespace())
    monkeypatch.setattr(route, "get_memory_client", lambda: memory)
    monkeypatch.setattr(route, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(route, "TripGraphRepository", Graph)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: Planner())
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "understand_turn", classify)
    monkeypatch.setattr(route, "design_goals", design)
    monkeypatch.setattr(route, "explain_selected_plans", no_explanation)
    async def no_offer(*args):
        return None

    monkeypatch.setattr(route, "infer_pending_offer", no_offer)
    async def empty_features(*args):
        return []

    async def trace(*args, **kwargs):
        return SimpleNamespace(id="trace")

    async def no_link(*args, **kwargs):
        return 0

    from src.memory import preference_graph
    monkeypatch.setattr(preference_graph, "available_features", empty_features)
    monkeypatch.setattr(preference_graph, "user_preferences", empty_features)
    monkeypatch.setattr(route, "record_agent_trace", trace)
    monkeypatch.setattr(route.linking, "link_trace_to_user", no_link)
    monkeypatch.setattr(route.linking, "link_trace_recommendations", no_link)
    monkeypatch.setattr(route.linking, "resolve_location_entities", no_link)

    for text in ("برام یه سفر دو نفره پاییزی برنامه‌ریزی کن",
                 "دیگه چی داری؟ افرود داریم و می‌خوایم پیاده‌روی کنیم",
                 "آره یکیشو بده"):
        response = await route.chat(ChatRequest(message=text, session_id="s-1"),
                                    Response(), browser_user_id="u-1")
    assert "روستای آهار" in response.reply
    assert "روستای لالان" not in response.reply
    assert "پیاده‌روی" in response.reply and "پاییز" in response.reply
    assert "تأیید نشده" in response.reply and "سرویس بهداشتی" not in response.reply
    assert response.itinerary is None and response.itineraries == []
    final = TravelGoal.model_validate(states["s-1", "u-1"])
    assert final.destination_names == ["روستای آهار"]
    assert final.last_proposed_routes == [["روستای آهار"]]
    assert final.season == "پاییز" and final.group_size == 2
    assert final.activities == ["پیاده‌روی"]
    assert final.transport == "آفرود"
    assert any(state["last_proposed_routes"][:2] == [[offered[0]], [offered[1]]]
               for state in snapshots)
    assert not any(state["destination_names"] == ["روستای لالان"] for state in snapshots)


@pytest.mark.asyncio
async def test_selection_only_uses_latest_assistant_offer_not_older_recommendations():
    route = importlib.import_module("src.api.routes.chat")
    from src.memory.graph import TripGraphRepository as RealGraph

    class Graph:
        _destination_lookup_keys = staticmethod(RealGraph._destination_lookup_keys)

        async def find_destinations_mentioned_in_text(self, text, limit=12):
            return [name for name in ["آبشار قدیم", "روستای آهار", "روستای لالان"] if name in text]

    goal = TravelGoal(last_proposed_routes=[["آبشار قدیم"]])
    transcript = "توربین: آبشار قدیم\nکاربر: گزینه‌های دیگه؟\nتوربین: روستای آهار و روستای لالان"
    assert await route._resolve_offered_option("یکیشو بده", goal, transcript, Graph()) == "روستای آهار"
    assert await route._resolve_offered_option("دومی رو بده", goal, transcript, Graph()) == "روستای لالان"
    assert await route._resolve_offered_option("برنامهٔ گزینهٔ دومی رو بده", goal, transcript, Graph()) == "روستای لالان"
    assert await route._resolve_offered_option("آره", goal, transcript, Graph()) is None
    assert await route._resolve_offered_option("کدوم یکی از اینها بهتره؟", goal, transcript, Graph()) is None


@pytest.mark.asyncio
async def test_bare_yes_accepts_both_forests_only_when_last_assistant_offered_them(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    from src.agent.goals import PendingOffer

    class Graph:
        async def find_destinations_mentioned_in_text(self, text, limit=12):
            return [name for name in ("جنگل دوهزار", "جنگل سه هزار")
                    if name.replace("‌", " ") in text.replace("‌", " ")][:limit]

        async def get_destination_details(self, name):
            return {"name": name} if name in ("جنگل دوهزار", "جنگل سه هزار") else None

    transcript = ("کاربر: جنگل‌های دوهزار و سه‌هزار چی؟\n"
                  "توربین: جنگل دوهزار و جنگل سه‌هزار زیبا هستند. "
                  "اگر مایلید، می‌توانم برنامه سفری برای بازدید از این جنگل‌ها بدهم.")
    async def infer(user, reply, candidates):
        if "برنامه" not in reply:
            return None
        assert "دوهزار" in user and "سه‌هزار" in reply
        return PendingOffer(action="plan", destination_names=["جنگل دوهزار", "جنگل سه هزار"],
                            source_reply=reply)

    monkeypatch.setattr(route, "infer_pending_offer", infer)
    assert route._is_bare_confirmation("اره")
    assert await route._resolve_accepted_plan_offer(transcript, Graph()) == (
        ["جنگل دوهزار", "جنگل سه هزار"], True)
    assert await route._resolve_accepted_plan_offer(
        "کاربر: جنگل‌ها چی؟\nتوربین: طبیعت زیبایی دارند.", Graph()) == ([], False)
    assert not route._is_bare_confirmation("آره برای آبشار ناران")


@pytest.mark.asyncio
async def test_pending_offer_is_generic_and_rejects_other_graph_destinations(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    from src.agent.goals import PendingOffer

    class Graph:
        async def find_destinations_mentioned_in_text(self, text, limit=16):
            return [name for name in ("دریاچه الف", "دریاچه ب")
                    if name in text][:limit]

        async def get_destination_details(self, name):
            return {"name": name} if name in {"دریاچه الف", "دریاچه ب", "آبشار ج"} else None

    prompt = "دریاچه الف و دریاچه ب چطورند؟"
    reply = "دریاچه الف و دریاچه ب نزدیک‌اند. اگر مایلید می‌توانم برنامه بازدید از این دو بدهم."

    async def joint(*args):
        return PendingOffer(action="plan", destination_names=["دریاچه الف", "دریاچه ب"],
                            source_reply=reply)

    monkeypatch.setattr(route, "infer_pending_offer", joint)
    pending = await route._capture_pending_offer(prompt, reply, Graph())
    assert pending.destination_names == ["دریاچه الف", "دریاچه ب"]
    assert (await route._resolve_accepted_plan_offer(
        f"کاربر: {prompt}\nتوربین: {reply}", Graph())) == (pending.destination_names, True)

    async def hallucination(*args):
        return PendingOffer(action="plan", destination_names=["آبشار ج"], source_reply=reply)

    monkeypatch.setattr(route, "infer_pending_offer", hallucination)
    assert await route._capture_pending_offer(prompt, reply, Graph()) is None


@pytest.mark.asyncio
async def test_turn_interpreter_reports_acceptance_without_erasing_goal(monkeypatch):
    class Classifier:
        async def run(self, prompt):
            data = json.loads(prompt)
            assert data["current_goal"]["pending_offer"]["destination_names"] == ["دریاچه الف", "دریاچه ب"]
            return SimpleNamespace(output=json.dumps({
                "action": "plan", "updates": {}, "offer_response": "accept",
                "selected_offer_names": []}, ensure_ascii=False))

    from src.agent.goals import PendingOffer
    monkeypatch.setattr(turns, "_turn_agent", lambda: Classifier())
    goal = TravelGoal(season="پاییز", pending_offer=PendingOffer(
        action="plan", destination_names=["دریاچه الف", "دریاچه ب"], source_reply="برنامه؟"))
    decision = await turns.understand_turn("باشه انجامش بده", goal, "توربین: برنامه؟")
    assert decision.offer_response == "accept"
    assert turns.apply_goal_updates(goal, decision.updates).season == "پاییز"


@pytest.mark.asyncio
async def test_yes_after_forest_offer_updates_goal_and_blocks_unrelated_routes(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    from src.memory.graph import TripGraphRepository as RealGraph
    from src.agent.goals import PendingOffer

    names = ["جنگل دوهزار", "جنگل سه هزار"]
    goal = TravelGoal(objective="سفر خانوادگی", companions=["خانواده"], season="پاییز",
                      destination_names=["آبشار قدیمی"])
    snapshots = []
    designed = []
    offer = ("جنگل دوهزار و جنگل سه‌هزار برای طبیعت‌گردی هستند. "
             "اگر مایلید، می‌توانم برنامه سفر برای بازدید از این جنگل‌ها بدهم.")

    class Graph:
        _destination_lookup_keys = staticmethod(RealGraph._destination_lookup_keys)

        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return snapshots[-1] if snapshots else goal.model_dump()

        async def upsert_travel_goal(self, session, user, state):
            snapshots.append(state)
            return len(snapshots)

        async def sync_session_features(self, *args):
            pass

        async def get_previous_recommendations(self, *args):
            return {"ids": set(), "names": set()}

        async def get_recent_session_features(self, *args):
            return {}

        async def get_user_history(self, *args, **kwargs):
            return {}

        async def get_destination_details(self, name):
            if name in names:
                return {"name": name, "categories": ["طبیعت"],
                        "description": f"در {name} منظرهٔ جنگلی دیده می‌شود."}
            return None

        async def find_destinations_mentioned_in_text(self, text, limit=12):
            return [name for name in [*names, "آبشار ناران", "لواسان", "روستای کن"]
                    if name.replace("‌", " ") in text.replace("‌", " ")][:limit]

        async def record_shown_destinations(self, *args):
            pass

    history = []

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=history)

    async def add_message(*, role, content, **kwargs):
        history.append(SimpleNamespace(role=SimpleNamespace(value=role), content=content))
        return None

    class Planner:
        async def run(self, prompt, *, deps, message_history):
            if "چی؟" in prompt:
                assert deps.travel_goal.destination_names == ["آبشار قدیمی"]
                return SimpleNamespace(output=offer, all_messages=lambda: [], new_messages=lambda: [])
            assert deps.travel_goal.destination_names == names
            assert deps.travel_goal.season == "پاییز"
            assert deps.travel_goal.companions == ["خانواده"]
            assert "جنگل دوهزار" in deps.travel_goal.semantic_query
            return SimpleNamespace(output="به آبشار ناران و لواسان و روستای کن بروید.",
                                   all_messages=lambda: [], new_messages=lambda: [])

    async def classify(*args):
        if "چی؟" in args[0]:
            return turns.TurnDecision("answer", {})
        return turns.TurnDecision("plan", {}, offer_response="accept")

    async def design(goal, *args):
        assert goal.destination_names == names
        designed.append(goal.destination_names)
        return goal

    async def no_explanation(*args, **kwargs):
        return ""

    async def infer(user, reply, candidates):
        return PendingOffer(action="plan", destination_names=names, source_reply=reply)

    async def screen(*args, map_itineraries, **kwargs):
        # Even a fully routed unrelated set of options cannot replace the
        # accepted two-forest offer.
        for name in ("آبشار ناران", "لواسان", "روستای کن"):
            map_itineraries.append({
                "origin": {"name": "تهران", "latitude": 35.69, "longitude": 51.39},
                "stops": [{"order": 1, "name": name, "latitude": 35.8, "longitude": 51.4}],
                "return_leg": {"leg_distance_km_from_previous": 10,
                               "leg_duration_hours_from_previous": .4},
                "round_trip": True,
            })
        return "سه گزینه: آبشار ناران، لواسان، روستای کن", None

    async def empty(*args):
        return []

    async def trace(*args, **kwargs):
        return SimpleNamespace(id="trace")

    async def no_link(*args, **kwargs):
        return 0

    from src.memory import preference_graph
    memory = SimpleNamespace(
        _client=SimpleNamespace(_ensure_connected=lambda: object()),
        short_term=SimpleNamespace(get_conversation=conversation, add_message=add_message),
        reasoning=SimpleNamespace())
    monkeypatch.setattr(route, "get_memory_client", lambda: memory)
    monkeypatch.setattr(route, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(route, "TripGraphRepository", Graph)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: Planner())
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "understand_turn", classify)
    monkeypatch.setattr(route, "design_goals", design)
    monkeypatch.setattr(route, "screen_short_trip", screen)
    monkeypatch.setattr(route, "explain_selected_plans", no_explanation)
    monkeypatch.setattr(route, "infer_pending_offer", infer)
    monkeypatch.setattr(preference_graph, "available_features", empty)
    monkeypatch.setattr(preference_graph, "user_preferences", empty)
    monkeypatch.setattr(route, "record_agent_trace", trace)
    monkeypatch.setattr(route.linking, "link_trace_to_user", no_link)
    monkeypatch.setattr(route.linking, "link_trace_recommendations", no_link)
    monkeypatch.setattr(route.linking, "resolve_location_entities", no_link)

    first = await route.chat(ChatRequest(message="جنگل‌های دوهزار و سه‌هزار چی؟",
                                        session_id="forest-session"),
                             Response(), browser_user_id="forest-user")
    assert first.reply == offer
    assert snapshots[-1]["pending_offer"]["destination_names"] == names
    assert snapshots[-1]["destination_names"] == ["آبشار قدیمی"]
    assert designed == []  # asking about places is not selecting them
    response = await route.chat(ChatRequest(message="باشه انجامش بده", session_id="forest-session"),
                                Response(), browser_user_id="forest-user")
    assert all(name in response.reply for name in names)
    assert all(name not in response.reply for name in ("آبشار ناران", "لواسان", "روستای کن"))
    assert "تأیید نشده" in response.reply
    assert response.itinerary is None and response.itineraries == []
    assert snapshots[-1]["destination_names"] == names
    assert designed == [names]
    assert snapshots[-1]["last_proposed_routes"] == [names]
    assert snapshots[-1]["season"] == "پاییز"


@pytest.mark.asyncio
async def test_explicit_plan_for_previously_suggested_place_is_not_an_answer(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"action":"plan","updates":{}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    decision = await turns.understand_turn("برای آبشار لاتون برنامه بده", TravelGoal(), "توربین: آبشار لاتون")
    assert decision.action == "plan"


@pytest.mark.asyncio
async def test_memory_based_plan_request_uses_llm_classification(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"action":"plan","updates":{}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    decision = await turns.understand_turn(
        "با توجه به چیزایی ک از من میدونی یه برنامه سفر برام بریز", TravelGoal(), ""
    )
    assert decision.action == "plan"


@pytest.mark.asyncio
async def test_first_travel_request_respects_llm_action(monkeypatch):
    class FakeAgent:
        async def run(self, prompt):
            return SimpleNamespace(output='{"action":"chat","updates":{}}')

    monkeypatch.setattr(turns, "_turn_agent", lambda: FakeAgent())
    decision = await turns.understand_turn("میخوایم آخر هفته بریم کمپ", None, "")
    assert decision.action == "chat"


@pytest.mark.asyncio
async def test_young_group_is_friends_not_children():
    goal = await update_travel_goal(None, "میخوایم آخر هفته با بچه ها جوونی بریم کمپ", "")
    assert goal.duration == "آخر هفته"
    assert goal.accommodation == ["کمپ"]
    assert "دوستان" in goal.companions and "کودک" not in goal.companions


@pytest.mark.asyncio
async def test_choice_of_previously_proposed_waterfall_resolves_persian_spelling(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    assert route._explicitly_mentions_destination("میخوام برم ابشار برجی کلا", "آبشار برجی کلا")

    class Classifier:
        async def run(self, prompt):
            assert "میخوام برم ابشار برجی کلا" in json.loads(prompt)["latest_message"]
            return SimpleNamespace(output=json.dumps({
                "action": "plan", "updates": {"destination_names": ["آبشار برجی کلا"]}
            }, ensure_ascii=False))

    monkeypatch.setattr(turns, "_turn_agent", lambda: Classifier())
    decision = await turns.understand_turn(
        "میخوام برم ابشار برجی کلا", TravelGoal(), "توربین: آبشار برجی کلا")
    assert turns.apply_goal_updates(TravelGoal(), decision.updates).destination_names == ["آبشار برجی کلا"]

    from src.memory.graph import TripGraphRepository

    repo = TripGraphRepository(None)

    async def read(query, **params):
        assert "'آ', 'ا'" in query
        assert "ابشار برجی کلا" in params["lookup_keys"][0]
        return [{"destination": {"name": "آبشار برجی کلا"}}]

    repo._read = read
    assert (await repo.get_destination_details("ابشار برجی کلا"))["name"] == "آبشار برجی کلا"


@pytest.mark.asyncio
async def test_open_ended_camping_request_does_not_choose_place_named_like_verb(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    saved = []

    class Graph:
        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return None

        async def get_destination_details(self, query):
            # The graph really can contain a place called "برم". Looking up
            # the whole sentence must not select it as the user's destination.
            return {"name": "برم"}

        async def upsert_travel_goal(self, session_id, user_id, goal):
            saved.append(goal)
            return len(saved)

        async def sync_session_features(self, *args):
            pass

        async def get_previous_recommendations(self, *args):
            return {"ids": set(), "names": set()}

        async def get_recent_session_features(self, *args):
            return {}

        async def get_user_history(self, *args, **kwargs):
            return {}

        async def find_destinations_mentioned_in_text(self, *args):
            return []

    class Classifier:
        async def run(self, prompt):
            assert "برم یه جا کمپ کنم" in json.loads(prompt)["latest_message"]
            return SimpleNamespace(output=json.dumps({
                "action": "plan", "updates": {"season": "پاییز", "trip_types": ["کمپ"]}
            }, ensure_ascii=False))

    class Planner:
        async def run(self, prompt, *, deps, message_history):
            assert deps.travel_goal.destination_names == []
            assert deps.travel_goal.season == "پاییز"
            return SimpleNamespace(output="برای کمپ پاییزی چند مقصد را بررسی می‌کنم.",
                                   all_messages=lambda: [], new_messages=lambda: [])

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=[])

    async def add_message(*args, **kwargs):
        return None

    async def design(goal, *args):
        return goal

    async def no_screen(*args, **kwargs):
        return None

    async def no_link(*args, **kwargs):
        return 0

    async def trace(*args, **kwargs):
        return SimpleNamespace(id="trace")

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             short_term=SimpleNamespace(get_conversation=conversation,
                                                        add_message=add_message),
                             reasoning=SimpleNamespace())
    monkeypatch.setattr(route, "get_memory_client", lambda: memory)
    monkeypatch.setattr(route, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(route, "TripGraphRepository", Graph)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: Planner())
    async def no_regex_parser(*args):
        raise AssertionError("The chat flow must use the LLM's structured updates")
    monkeypatch.setattr(route, "update_travel_goal", no_regex_parser)
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "screen_short_trip", no_screen)
    monkeypatch.setattr(route, "design_goals", design)
    monkeypatch.setattr(route, "record_agent_trace", trace)
    monkeypatch.setattr(route.linking, "link_trace_to_user", no_link)
    monkeypatch.setattr(route.linking, "link_trace_recommendations", no_link)
    monkeypatch.setattr(route.linking, "resolve_location_entities", no_link)
    monkeypatch.setattr(turns, "_turn_agent", lambda: Classifier())
    monkeypatch.setattr(route, "understand_turn", turns.understand_turn)

    result = await route.chat(
        ChatRequest(message="برام یه برنامه بریز تو پاییز برم یه جا کمپ کنم", session_id="session1"),
        Response(), browser_user_id="account1")
    assert "مقصد انتخابی شما (برم)" not in result.reply
    assert saved and all(not state["destination_names"] for state in saved)
    assert saved[0]["season"] == "پاییز" and "کمپ" in saved[0]["trip_types"]


@pytest.mark.asyncio
async def test_chat_followup_does_not_run_plan_screen_or_rewrite_reply(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")

    class FakeAgent:
        async def run(self, message, deps, message_history):
            assert deps.turn_action == "answer"
            return SimpleNamespace(output="آب‌وهوای گلابدره در بهار معمولاً معتدل است؛ برای وضعیت امروز پیش‌بینی زنده را ببینید.",
                                   all_messages=lambda: [], new_messages=lambda: [])

    async def answer(*args):
        return turns.TurnDecision("answer")

    async def forbidden(*args, **kwargs):
        raise AssertionError("A destination question must not trigger the trip-plan formatter")

    monkeypatch.setattr(route, "get_memory_client", lambda: None)
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: FakeAgent())
    monkeypatch.setattr(route, "understand_turn", answer)
    monkeypatch.setattr(route, "screen_short_trip", forbidden)

    response = await route.chat(ChatRequest(message="گلابدره آب و هواش چطوره؟"), Response(), browser_user_id=None)
    assert "گلابدره" in response.reply
    assert "برای پیشنهاد سفر کوتاه" not in response.reply
    assert response.itinerary is None


@pytest.mark.asyncio
async def test_each_followup_saves_a_goal_snapshot_even_without_another_itinerary(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    from src.agent.goals import PlanningSubgoal
    from src.memory import preference_graph

    state = TravelGoal(objective="طبیعت پاییزی", season="پاییز",
                       subgoals=[PlanningSubgoal(id="G1", name="طبیعت")]).model_dump()
    snapshots = []
    seen_features = []

    class Graph:
        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return state

        async def upsert_travel_goal(self, session_id, user_id, saved):
            state.update(saved)
            snapshots.append(json.loads(json.dumps(saved)))
            return len(snapshots)

        async def sync_session_features(self, *args):
            pass

    class Planner:
        async def run(self, prompt, *, deps, message_history):
            assert deps.turn_action == "answer"
            return SimpleNamespace(output="اطلاعات مقصد", all_messages=lambda: [],
                                   new_messages=lambda: [])

    async def understand(message, goal, transcript):
        if "سرویس" in message:
            return turns.TurnDecision("answer", {
                "companions": ["خانواده"], "must_haves": ["سرویس بهداشتی"],
                "constraints": ["پیاده‌روی کم"]})
        return turns.TurnDecision("answer")

    async def designer(goal, message, transcript, preferences, features):
        seen_features.extend(features)
        return goal.model_copy(update={"subgoals": [
            *goal.subgoals, PlanningSubgoal(id="G2", name="امکانات", source="explicit")]})

    async def features(driver):
        return [{"id": "facilities:high", "label": "FacilitiesLevel",
                 "name": "زیاد", "description": "امکانات"}]

    async def prefs(driver, user):
        return []

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=[])

    async def add_message(*args, **kwargs):
        return None

    async def trace(*args, **kwargs):
        return SimpleNamespace(id="trace")

    async def no_link(*args, **kwargs):
        return 0

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             short_term=SimpleNamespace(get_conversation=conversation,
                                                        add_message=add_message),
                             reasoning=SimpleNamespace())
    monkeypatch.setattr(route, "get_memory_client", lambda: memory)
    monkeypatch.setattr(route, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(route, "TripGraphRepository", Graph)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: Planner())
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "understand_turn", understand)
    monkeypatch.setattr(route, "design_goals", designer)
    monkeypatch.setattr(route, "record_agent_trace", trace)
    monkeypatch.setattr(route.linking, "link_trace_to_user", no_link)
    monkeypatch.setattr(route.linking, "link_trace_recommendations", no_link)
    monkeypatch.setattr(route.linking, "resolve_location_entities", no_link)
    monkeypatch.setattr(preference_graph, "available_features", features)
    monkeypatch.setattr(preference_graph, "user_preferences", prefs)

    for text in ("با خانواده میرم، سرویس بهداشتی و پیاده‌روی کم میخوام",
                 "در مورد این سفر بیشتر بگو"):
        result = await route.chat(ChatRequest(message=text, session_id="session1"),
                                  Response(), browser_user_id="account1")
        assert result.itinerary is None
    assert len(snapshots) == 2
    assert snapshots[0]["season"] == snapshots[1]["season"] == "پاییز"
    assert snapshots[0]["must_haves"] == snapshots[1]["must_haves"] == ["سرویس بهداشتی"]
    assert [goal["id"] for goal in snapshots[1]["subgoals"]] == ["G1", "G2"]
    assert snapshots[1]["last_user_request"] == "در مورد این سفر بیشتر بگو"
    assert seen_features[0]["id"] == "facilities:high"


@pytest.mark.asyncio
async def test_chat_exposes_independent_map_routes_for_alternative_plans(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")

    class FakeAgent:
        async def run(self, message, deps, message_history):
            return SimpleNamespace(output="چند برنامه", all_messages=lambda: [], new_messages=lambda: [])

    async def plan(*args):
        return turns.TurnDecision("plan")

    async def goal(*args):
        return TravelGoal(duration="یک روز")

    async def screen(*args, map_itineraries, **kwargs):
        for index in (1, 2):
            map_itineraries.append({
                "origin": {"name": "تهران", "latitude": 35.6892, "longitude": 51.389},
                "stops": [{"order": 1, "name": f"مقصد {index}", "latitude": 35.8 + index * .01,
                           "longitude": 51.4}],
                "return_leg": {"leg_distance_km_from_previous": 20, "leg_duration_hours_from_previous": .5},
                "round_trip": True,
            })
        return "### 🌿 پیشنهادها\n\n#### گزینهٔ 1: مقصد 1\n\n#### گزینهٔ 2: مقصد 2", None

    monkeypatch.setattr(route, "get_memory_client", lambda: None)
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: FakeAgent())
    monkeypatch.setattr(route, "understand_turn", plan)
    monkeypatch.setattr(route, "update_travel_goal", goal)
    monkeypatch.setattr(route, "screen_short_trip", screen)

    async def explanation(*args, **kwargs):
        return "### چرا این سفر را پیشنهاد کردم؟\n\n**مسیر 1:** مقصد 1 با درخواست این سفر هماهنگ است."

    monkeypatch.setattr(route, "explain_selected_plans", explanation)

    response = await route.chat(ChatRequest(message="برای یک روز دو مقصد پیشنهاد بده"), Response(), browser_user_id=None)
    assert response.reply.startswith("سلام!\n\n### چرا این سفر را پیشنهاد کردم؟")
    assert "#### گزینهٔ 2: مقصد 2" in response.reply
    assert response.itinerary is None
    assert [plan.stops[0].name for plan in response.itineraries] == ["مقصد 1", "مقصد 2"]
    assert response.session_id and response.user_id


@pytest.mark.asyncio
async def test_named_destination_survives_family_replan_and_rejects_substitute_routes(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    original = TravelGoal(objective="برنامه یک روزه برای درکه", duration="یک روز",
                          destination_names=["درکه"])
    saved = []

    class Graph:
        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return original.model_dump()

        async def upsert_travel_goal(self, session_id, user_id, state):
            saved.append(state)
            return len(saved)

        async def sync_session_features(self, *args):
            pass

        async def get_destination_details(self, text):
            return None

        async def get_previous_recommendations(self, *args):
            return {"ids": set(), "names": set()}

        async def get_recent_session_features(self, *args):
            return {}

        async def get_user_history(self, *args, **kwargs):
            return {}

        async def find_destinations_mentioned_in_text(self, text):
            raise AssertionError("A rejected plan must not be recorded as a recommendation")

    class Planner:
        async def run(self, *args, **kwargs):
            return SimpleNamespace(output="به جای درکه به روستای امامه بروید",
                                   all_messages=lambda: [], new_messages=lambda: [])

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=[])

    async def add_message(*args, **kwargs):
        return None

    async def classify(*args):
        return turns.TurnDecision("plan", {"companions": ["خانواده"],
                                           "must_haves": ["سرویس بهداشتی"],
                                           "constraints": ["پیاده‌روی کم"]})

    async def screen(*args, map_itineraries, **kwargs):
        map_itineraries.append({
            "origin": {"name": "تهران", "latitude": 35.6892, "longitude": 51.389},
            "stops": [{"order": 1, "name": "روستای امامه", "latitude": 35.8,
                       "longitude": 51.4}],
            "return_leg": {"leg_distance_km_from_previous": 20,
                           "leg_duration_hours_from_previous": .5},
            "round_trip": True,
        })
        return "به جای درکه به روستای امامه بروید", None

    async def design(goal, *args):
        return goal

    async def coverage(*args):
        return []

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             short_term=SimpleNamespace(get_conversation=conversation,
                                                        add_message=add_message),
                             reasoning=SimpleNamespace())
    monkeypatch.setattr(route, "get_memory_client", lambda: memory)
    monkeypatch.setattr(route, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(route, "TripGraphRepository", Graph)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: Planner())
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "understand_turn", classify)
    monkeypatch.setattr(route, "screen_short_trip", screen)
    monkeypatch.setattr(route, "design_goals", design)
    monkeypatch.setattr(route, "evaluate_plan", coverage)
    monkeypatch.setattr(route, "record_agent_trace", lambda *a, **k: trace())

    async def trace():
        return SimpleNamespace(id="trace")

    async def no_link(*args, **kwargs):
        return 0

    monkeypatch.setattr(route.linking, "link_trace_to_user", no_link)
    monkeypatch.setattr(route.linking, "link_trace_recommendations", no_link)
    monkeypatch.setattr(route.linking, "resolve_location_entities", no_link)
    response = await route.chat(
        ChatRequest(message="با خانواده میرم و سرویس بهداشتی و پیاده‌روی کم می‌خوام",
                    session_id="session1"),
        Response(), browser_user_id="account1")
    assert "درکه" in response.reply
    assert "روستای امامه" not in response.reply
    assert response.itinerary is None and response.itineraries == []
    assert saved and all(state["destination_names"] == ["درکه"] for state in saved)
    assert saved[0]["companions"] == ["خانواده"]
    assert saved[0]["must_haves"] == ["سرویس بهداشتی"]


@pytest.mark.asyncio
async def test_toilet_at_remote_shelter_does_not_verify_low_walking_access():
    route = importlib.import_module("src.api.routes.chat")

    class Graph:
        async def get_destination_details(self, name):
            assert name == "درکه"
            return {"name": name, "description": "سرویس‌های بهداشتی در پناهگاه کوهستانی"}

    goal = TravelGoal(destination_names=["درکه"], must_haves=["سرویس بهداشتی"],
                      constraints=["پیاده‌روی کم"])
    answer = await route._verified_access_answer(
        "سرویس بهداشتی نزدیک میدان درکه و مسیر کم‌پیاده‌روی تأیید شده؟", goal, Graph())
    assert "نمی‌توانم سرویس نزدیک" in answer
    assert "فاصلهٔ پیاده‌روی" in answer
    assert "پناهگاه" not in answer


@pytest.mark.asyncio
async def test_chat_explains_named_alternatives_without_map_results(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")

    class Graph:
        synced_calls = []
        shown_calls = []

        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return None

        async def upsert_travel_goal(self, *args):
            pass

        async def sync_session_features(self, user_id, session_id, state):
            assert user_id.startswith("user_") and session_id.startswith("session_")
            assert state["trip_types"] == [] or "کمپ" in state["accommodation"]
            Graph.synced_calls.append((user_id, session_id))

        async def get_previous_recommendations(self, user_id, session_id):
            return {"ids": set(), "names": set()}

        async def get_recent_session_features(self, user_id, session_id):
            return {}

        async def record_shown_destinations(self, user_id, session_id, names):
            Graph.shown_calls.append((user_id, session_id, names))

        async def find_destinations_mentioned_in_text(self, reply):
            assert "آبشار لاتون" in reply and "دریاچه نئور" in reply
            return ["آبشار لاتون", "دریاچه نئور"]

        async def get_destination_details(self, name):
            return {"name": name, "categories": ["طبیعت"], "trip_types": ["کمپ"]}

        async def get_user_history(self, user_id, exclude_session_id=None):
            return {"liked": []}

    class Agent:
        async def run(self, *args, **kwargs):
            return SimpleNamespace(output="برای کمپ: آبشار لاتون یا دریاچه نئور را پیشنهاد می‌کنم.",
                                   all_messages=lambda: [], new_messages=lambda: [])

    class Memory:
        _client = SimpleNamespace(_ensure_connected=lambda: object())
        short_term = SimpleNamespace(
            get_conversation=None,
            add_message=None,
        )
        long_term = SimpleNamespace(get_preferences_for=None)
        reasoning = SimpleNamespace()

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=[])

    async def add_message(*args, **kwargs):
        return SimpleNamespace(id="m1")

    async def preferences(*args):
        return []

    async def plan(*args):
        return turns.TurnDecision("plan")

    async def goal(*args):
        return TravelGoal(objective="آخر هفته کمپ بریم", accommodation=["کمپ"])

    async def no_screen(*args, **kwargs):
        return None

    Memory.short_term.get_conversation = conversation
    Memory.short_term.add_message = add_message
    Memory.long_term.get_preferences_for = preferences
    monkeypatch.setattr(route, "get_memory_client", Memory)
    monkeypatch.setattr(route, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(route, "TripGraphRepository", Graph)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: Agent())
    monkeypatch.setattr(route, "get_neshan_client", lambda: None)
    monkeypatch.setattr(route, "understand_turn", plan)
    monkeypatch.setattr(route, "update_travel_goal", goal)
    monkeypatch.setattr(route, "screen_short_trip", no_screen)
    async def no_writer(payload):
        return None
    monkeypatch.setattr(route, "write_explanation", no_writer)

    response = await route.chat(ChatRequest(message="میخوایم آخر هفته با بچه‌ها بریم کمپ"),
                                Response(), browser_user_id=None)
    assert response.reply.startswith("سلام!\n\n### چرا این سفر را پیشنهاد کردم؟")
    assert "**مسیر 1:** آبشار لاتون" in response.reply
    assert "**مسیر 2:** دریاچه نئور" in response.reply
    assert response.itinerary is None and response.itineraries == []
    assert Graph.synced_calls == [(response.user_id, response.session_id)]
    assert Graph.shown_calls == [(response.user_id, response.session_id,
                                  ["آبشار لاتون", "دریاچه نئور"])]


@pytest.mark.asyncio
async def test_why_these_routes_answers_from_last_actual_plans_not_generic_model(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")
    goal = TravelGoal(duration="سه روز", prior_insights={"moods": ["ماجراجویانه"]},
                      last_proposed_routes=[["آبشار الف", "دریاچه ب"], ["جنگل ج"]])

    class Graph:
        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return goal.model_dump()

        async def upsert_travel_goal(self, *args):
            return 1

        async def get_destination_details(self, name):
            return {"name": name, "categories": ["کوهنوردی"],
                    "description": f"طبیعت و مسیر بازدید در {name}."}

        async def get_user_history(self, user_id):
            return {"liked": []}

    async def preferences(*args):
        return []

    async def message(**kwargs):
        return SimpleNamespace(id="m1")

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=[])

    async def writer(payload):
        assert [[s["name"] for s in plan["stops"]] for plan in payload["plans"]] == goal.last_proposed_routes
        return {"intro": "از درخواست ماجراجویانهٔ قبلی‌ات یادم هست.",
                "plans": [{"index": 1, "why": "آبشار الف و دریاچه ب را برای دیدن طبیعت در یک مسیر گذاشتم."},
                          {"index": 2, "why": "جنگل ج مسیر دیگری برای همان سفر سه‌روزه است."}]}

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             short_term=SimpleNamespace(get_conversation=conversation, add_message=message),
                             long_term=SimpleNamespace(get_preferences_for=preferences))
    monkeypatch.setattr(route, "get_memory_client", lambda: memory)
    monkeypatch.setattr(route, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(route, "TripGraphRepository", Graph)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: pytest.fail("Do not ask model to invent plan reasons"))
    monkeypatch.setattr(route, "write_explanation", writer)
    async def classify(*args):
        return turns.TurnDecision("answer")
    monkeypatch.setattr(route, "understand_turn", classify)

    response = await route.chat(ChatRequest(message="چرا این برنامه ها رو بهم دادی؟", session_id="s1"),
                                Response(), browser_user_id="user-a")
    assert "آبشار الف و دریاچه ب" in response.reply
    assert "جنگل ج" in response.reply and "درخواست ماجراجویانه" in response.reply
    assert response.itinerary is None


@pytest.mark.asyncio
async def test_vehicle_question_uses_selected_place_without_new_itinerary(monkeypatch):
    route = importlib.import_module("src.api.routes.chat")

    class Graph:
        def __init__(self, *args):
            pass

        async def get_travel_goal(self, *args):
            return TravelGoal(duration="سه روز", destination_names=["آبشار برجی کلا"]).model_dump()

        async def get_destination_details(self, name):
            if name == "آبشار برجی کلا":
                return {"name": name, "vehicles": ["سواری", "آفرود"], "physical_readiness": "متوسط"}
            return None

    async def conversation(*args, **kwargs):
        return SimpleNamespace(messages=[])

    async def add_message(*args, **kwargs):
        return None

    memory = SimpleNamespace(_client=SimpleNamespace(_ensure_connected=lambda: object()),
                             short_term=SimpleNamespace(get_conversation=conversation, add_message=add_message))
    monkeypatch.setattr(route, "get_memory_client", lambda: memory)
    monkeypatch.setattr(route, "TripGraphRepository", Graph)
    monkeypatch.setattr(route, "get_embedding_provider", lambda: None)
    monkeypatch.setattr(route, "get_trip_planner_agent", lambda: pytest.fail("No new trip should be generated"))

    response = await route.chat(ChatRequest(message="با چه ماشینی برم اونجا؟", session_id="s1"),
                                Response(), browser_user_id="user-a")
    assert "آبشار برجی کلا" in response.reply and "سواری" in response.reply
    assert "آمادگی بدنی متوسط" in response.reply
    assert response.itinerary is None and response.itineraries == []
