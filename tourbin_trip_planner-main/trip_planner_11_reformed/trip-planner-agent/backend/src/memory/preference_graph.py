"""Controlled feature vocabulary and evidence-backed, idempotent user preferences."""

import hashlib
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FEATURE_LABELS = (
    "Category", "Season", "TripType", "Vehicle", "FacilitiesLevel", "TravelTime",
    "RoadType", "PhysicalReadiness", "TravelStyle", "Activity", "Environment",
    "FoodPreference", "AccommodationType", "DistanceRange",
)
# Only these relationships are actually populated for Destination nodes.
# A vocabulary node alone (e.g. Environment:خلوت) is not destination evidence.
DESTINATION_FEATURE_RELATIONS = {
    "Category": "HAS_CATEGORY",
    "Season": "BEST_SEASON",
    "TripType": "SUITABLE_FOR",
    "Vehicle": "ACCESSIBLE_BY",
    "FacilitiesLevel": "HAS_FACILITIES_LEVEL",
    "TravelTime": "HAS_TRAVEL_TIME",
    "RoadType": "HAS_ROAD_TYPE",
    "PhysicalReadiness": "REQUIRES_READINESS",
    "DistanceRange": "HAS_DISTANCE_RANGE",
}


def destination_feature_id(value: str) -> bool:
    """Only controlled labels backed by destination relationships are checkable."""
    return (isinstance(value, str) and
            value.partition(":")[0] in
            {label.lower() for label in DESTINATION_FEATURE_RELATIONS})
NEW_VALUES = {
    "FacilitiesLevel": ("کم", "متوسط", "زیاد"),
    "TravelTime": ("کوتاه", "متوسط", "طولانی"),
    "RoadType": ("آسان", "متوسط", "دشوار"),
    "PhysicalReadiness": ("کم", "متوسط", "زیاد"),
    "TravelStyle": ("آرام", "ماجراجویانه", "فشرده", "اقتصادی", "لوکس"),
    "Activity": ("عکاسی", "پیاده‌روی", "کافه‌گردی", "خرید"),
    "Environment": ("خلوت", "شلوغ", "شهری", "روستایی"),
    "FoodPreference": ("غذای محلی", "غذای خیابانی", "رستوران سنتی"),
    "AccommodationType": ("هتل", "اقامتگاه بوم‌گردی", "کمپ"),
    "DistanceRange": ("نزدیک", "متوسط", "دور"),
}
FEATURE_DESCRIPTIONS = {
    "Category": "نوع و ماهیت مقصد؛ مانند طبیعت یا مکان تاریخی",
    "Season": "فصل ترجیحی سفر",
    "TripType": "نوع فعالیت سفر، مانند کمپ یا بازدید",
    "Vehicle": "وسیله نقلیه ترجیحی کاربر",
    "FacilitiesLevel": "سطح کلی امکانات؛ نه وجود امکانات خاص",
    "TravelTime": "مدت مسیر از تهران: کوتاه تا ۲ ساعت، متوسط تا ۵ ساعت، طولانی بیشتر",
    "RoadType": "نوع و دشواری جاده دسترسی",
    "PhysicalReadiness": "ترجیح درجه سختی فعالیت؛ نه تشخیص توانایی جسمانی کاربر",
    "DistanceRange": "فاصله از تهران: نزدیک تا ۱۰۰ کیلومتر، متوسط تا ۳۰۰ کیلومتر، دور بیشتر",
    "TravelStyle": "سبک پایدار سفر",
    "Activity": "فعالیت مورد علاقه در سفر",
    "Environment": "فضای مورد علاقه سفر",
    "FoodPreference": "نوع غذای مورد علاقه سفر",
    "AccommodationType": "نوع اقامت ترجیحی",
}


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).translate(
        str.maketrans({"ي": "ی", "ك": "ک", "\u200c": " "})
    ).strip().split()).casefold()


def feature_id(label: str, name: str) -> str:
    if label not in FEATURE_LABELS:
        raise ValueError("Unknown feature type")
    digest = hashlib.sha256(f"{label}:{normalized(name)}".encode()).hexdigest()[:24]
    return f"{label.lower()}:{digest}"


def travel_time_bucket(hours: float | None) -> str | None:
    if hours is None or hours < 0:
        return None
    return "کوتاه" if hours <= 2 else "متوسط" if hours <= 5 else "طولانی"


def distance_bucket(km: float | None) -> str | None:
    if km is None or km < 0:
        return None
    return "نزدیک" if km <= 100 else "متوسط" if km <= 300 else "دور"


def facilities_bucket(value: str | None) -> str | None:
    if not value:
        return None
    value = normalized(value)
    # "دارد" says nothing about the *level*; never guess a value for it.
    return next((level for level in ("کم", "متوسط", "زیاد") if level in value), None)


def road_bucket(value: str | None) -> str | None:
    if not value:
        return None
    value = normalized(value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    if "سان" in value:
        return "آسان"
    if "سخت" in value or "دشوار" in value:
        return "دشوار"
    return "متوسط" if "متوسط" in value else None


class PreferenceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    positive_increment: Literal[0, 1]
    negative_increment: Literal[0, 1]
    positive_evidence_message_ids: list[str] = Field(default_factory=list)
    negative_evidence_message_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_evidence(self):
        if (not self.positive_increment and not self.negative_increment
                or bool(self.positive_evidence_message_ids) != bool(self.positive_increment)
                or bool(self.negative_evidence_message_ids) != bool(self.negative_increment)
                or set(self.positive_evidence_message_ids) & set(self.negative_evidence_message_ids)):
            raise ValueError("Each increment needs its own evidence")
        return self


class PreferenceBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    updates: list[PreferenceUpdate] = Field(max_length=100)

    @model_validator(mode="after")
    def unique_features(self):
        if len({u.node_id for u in self.updates}) != len(self.updates):
            raise ValueError("Duplicate feature update")
        return self


async def ensure_preference_ontology(driver) -> None:
    """Add only controlled nodes/edges; retain all legacy scalar properties."""
    async with driver.session() as session:
        for label in FEATURE_LABELS:
            result = await session.run(
                f"CREATE CONSTRAINT pref_{label.lower()}_key IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.preference_key IS UNIQUE")
            await result.consume()
        for label in FEATURE_LABELS[:4]:
            result = await session.run(f"MATCH (n:{label}) WHERE n.name IS NOT NULL "
                                       "RETURN n.name AS name, elementId(n) AS element_id")
            rows = [r.data() async for r in result]
            for row in rows:
                result = await session.run(
                    f"MATCH (n:{label}) WHERE elementId(n) = $element_id "
                    "SET n.preference_key = $key, n.preference_id = $id",
                    element_id=row["element_id"], key=normalized(row["name"]),
                    id=feature_id(label, row["name"]))
                await result.consume()
        for label, names in NEW_VALUES.items():
            for name in names:
                result = await session.run(
                    f"MERGE (n:{label} {{preference_key: $key}}) "
                    "SET n.name = $name, n.preference_id = $id",
                    key=normalized(name), name=name, id=feature_id(label, name))
                await result.consume()
        # These categories describe routes *from Tehran*; no inference is made
        # for destinations whose numeric source values are absent or ambiguous.
        result = await session.run(
            "MATCH (d:Destination) WHERE NOT d:OsmFeature "
            "RETURN d.id AS id, d.facilities_level AS facilities, "
            "d.travel_time_hours AS hours, d.distance_km AS km, "
            "d.road_type AS road, d.physical_readiness AS readiness")
        rows = [r.data() async for r in result]
        rels = {
            "FacilitiesLevel": "HAS_FACILITIES_LEVEL",
            "TravelTime": "HAS_TRAVEL_TIME",
            "RoadType": "HAS_ROAD_TYPE",
            "PhysicalReadiness": "REQUIRES_READINESS",
            "DistanceRange": "HAS_DISTANCE_RANGE",
        }
        for row in rows:
            values = {
                "FacilitiesLevel": facilities_bucket(row["facilities"]),
                "TravelTime": travel_time_bucket(row["hours"]),
                "RoadType": road_bucket(row["road"]),
                "PhysicalReadiness": row["readiness"] if row["readiness"] in NEW_VALUES["PhysicalReadiness"] else None,
                "DistanceRange": distance_bucket(row["km"]),
            }
            for label, rel in rels.items():
                # Re-running the migration also removes an obsolete bin when a
                # source property changes, without touching unrelated edges.
                result = await session.run(
                    f"MATCH (d:Destination {{id: $id}}) "
                    f"OPTIONAL MATCH (d)-[old:{rel}]->(n:{label}) "
                    "WHERE $key IS NULL OR n.preference_key <> $key DELETE old",
                    id=row["id"], key=normalized(values[label]) if values[label] else None)
                await result.consume()
                name = values[label]
                if name is not None:
                    result = await session.run(
                        f"MATCH (d:Destination {{id: $id}}), "
                        f"(n:{label} {{preference_key: $key}}) MERGE (d)-[:{rel}]->(n)",
                        id=row["id"], key=normalized(name))
                    await result.consume()


async def available_features(driver) -> list[dict]:
    async with driver.session() as session:
        result = await session.run(
            "MATCH (n) WHERE n.preference_id IS NOT NULL "
            "AND any(label IN labels(n) WHERE label IN $labels) "
            "RETURN n.preference_id AS id, n.name AS name, "
            "head([label IN labels(n) WHERE label IN $labels]) AS label "
            "ORDER BY label, name", labels=list(FEATURE_LABELS))
        return [{**row.data(), "description": FEATURE_DESCRIPTIONS[row["label"]]}
                async for row in result]


async def user_preferences(driver, user_id: str) -> list[dict]:
    async with driver.session() as session:
        result = await session.run(
            "MATCH (u:User {identifier: $user_id})-[r:HAS_PREFERENCE]->(n) "
            "WHERE n.preference_id IS NOT NULL "
            "RETURN n.preference_id AS id, n.name AS name, "
            "r.positive_count AS positive_count, r.negative_count AS negative_count, "
            "r.score AS score, toString(r.updated_at) AS updated_at "
            "ORDER BY r.updated_at DESC", user_id=user_id)
        return [row.data() async for row in result]


async def session_messages(driver, user_id: str, session_id: str) -> list[dict]:
    async with driver.session() as session:
        result = await session.run(
            "MATCH (u:User {identifier: $user_id})-[:OWNS_CHAT]->"
            "(chat:ChatSession {session_id: $session_id}) "
            "MATCH (u)-[:HAS_CONVERSATION]->(c:Conversation {session_id: chat.session_id})"
            "-[:HAS_MESSAGE]->(m:Message) "
            "WHERE m.role IN ['user', 'assistant'] "
            "RETURN m.id AS message_id, m.role AS role, m.content AS content "
            "ORDER BY m.timestamp, m.id", user_id=user_id, session_id=session_id)
        return [row.data() async for row in result]


async def apply_updates(driver, user_id: str, session_id: str,
                        batch: PreferenceBatch) -> int:
    """Validate *all* updates first, then apply evidence and counters atomically."""
    async def transaction(tx):
        result = await tx.run("MATCH (u:User {identifier: $user_id}) "
                              "SET u.preference_write_lock = 1 RETURN u", user_id=user_id)
        if not await result.single():
            raise ValueError("Unknown user")
        valid_messages = await tx.run(
            "MATCH (u:User {identifier: $user_id})-[:OWNS_CHAT]->"
            "(chat:ChatSession {session_id: $session_id}) "
            "MATCH (u)-[:HAS_CONVERSATION]->(c:Conversation {session_id: chat.session_id})"
            "-[:HAS_MESSAGE]->(m:Message {role: 'user'}) RETURN m.id AS id",
            user_id=user_id, session_id=session_id)
        message_ids = {r["id"] async for r in valid_messages}
        catalog = await tx.run(
            "MATCH (n) WHERE n.preference_id IN $ids "
            "AND any(label IN labels(n) WHERE label IN $labels) "
            "RETURN n.preference_id AS id, count(n) AS count",
            ids=[u.node_id for u in batch.updates], labels=list(FEATURE_LABELS))
        allowed = {r["id"]: r["count"] for r in [row async for row in catalog]}
        for update in batch.updates:
            evidence = update.positive_evidence_message_ids + update.negative_evidence_message_ids
            if (allowed.get(update.node_id) != 1 or not set(evidence) <= message_ids
                    or len(evidence) != len(set(evidence))):
                raise ValueError("Invalid feature or conversation evidence")
        changed = 0
        for update in batch.updates:
            for polarity, evidence in (
                ("positive", update.positive_evidence_message_ids),
                ("negative", update.negative_evidence_message_ids),
            ):
                if not evidence:
                    continue
                # At most one count per feature/polarity/conversation even
                # after retries or new messages added to the same conversation.
                existing = await tx.run(
                    "MATCH (n {preference_id: $node_id})-[e:HAS_EVIDENCE]->(:Message) "
                    "WHERE e.user_id = $user_id AND e.session_id = $session_id "
                    "AND e.polarity = $polarity RETURN count(e) AS count",
                    node_id=update.node_id, user_id=user_id,
                    session_id=session_id, polarity=polarity)
                if (await existing.single())["count"]:
                    continue
                for message_id in evidence:
                    result = await tx.run(
                        "MATCH (n {preference_id: $node_id}), (m:Message {id: $message_id}) "
                        "MERGE (n)-[e:HAS_EVIDENCE {user_id: $user_id, "
                        "session_id: $session_id, polarity: $polarity}]->(m) "
                        "RETURN e", node_id=update.node_id, message_id=message_id,
                        user_id=user_id, session_id=session_id, polarity=polarity)
                    await result.consume()
                result = await tx.run(
                    "MATCH (u:User {identifier: $user_id}), "
                    "(n {preference_id: $node_id}) "
                    "MERGE (u)-[r:HAS_PREFERENCE]->(n) "
                    "ON CREATE SET r.positive_count = 0, r.negative_count = 0 "
                    "SET r.positive_count = r.positive_count + $positive, "
                    "r.negative_count = r.negative_count + $negative, "
                    "r.updated_at = datetime() "
                    "WITH r SET r.score = r.positive_count - r.negative_count",
                    user_id=user_id, node_id=update.node_id,
                    positive=int(polarity == "positive"), negative=int(polarity == "negative"))
                await result.consume()
                changed += 1
        return changed

    async with driver.session() as session:
        return await session.execute_write(transaction)
