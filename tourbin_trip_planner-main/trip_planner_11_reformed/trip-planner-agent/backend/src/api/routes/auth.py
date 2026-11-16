"""Account authentication backed by the same Neo4j database as trip memory."""

import hashlib
import hmac
import logging
import re
import secrets
import time
from dataclasses import dataclass
from uuid import uuid4

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from src.config import get_settings
from src.memory.client import get_memory_client
from src.memory.preference_extractor import finalize_preferences

router = APIRouter()
logger = logging.getLogger(__name__)
AUTH_COOKIE = "tourbin_auth"
AUTH_AGE = 60 * 60 * 24 * 30
ITERATIONS = 600_000


class Credentials(BaseModel):
    identifier: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=1024)


@dataclass(frozen=True)
class Identity:
    user_id: str
    token_hash: str


def driver():
    memory = get_memory_client()
    if memory is None:
        raise HTTPException(503, "Account storage is unavailable")
    try:
        return memory._client._ensure_connected()
    except Exception as exc:
        raise HTTPException(503, "Account storage is unavailable") from exc


def normalize(identifier: str) -> tuple[str, str]:
    value = identifier.strip().lower()
    if "@" in value:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise HTTPException(422, "Invalid email address")
        return "email", value
    value = re.sub(r"[\s()-]", "", value)
    if not re.fullmatch(r"\+?[0-9]{10,15}", value):
        raise HTTPException(422, "Use an email address or a phone number with country code")
    return "phone", value


def password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    try:
        salt_hex, _ = stored.split(":")
        return hmac.compare_digest(password_hash(password, bytes.fromhex(salt_hex)), stored)
    except (ValueError, TypeError, AttributeError):
        return False


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def conversation_title(message: str) -> str:
    """Use the first actual request as a stable Persian-readable chat title."""
    text = " ".join(message.split()).strip(" .،؟?!")
    text = re.sub(r"^(?:سلام|درود|صبح بخیر|وقت بخیر|hi|hello)(?:[!،,\s]+|$)", "", text,
                  flags=re.IGNORECASE).strip()
    if not text:
        return "گفتگوی سفر"
    if len(text) <= 52:
        return text
    return text[:52].rsplit(" ", 1)[0].strip() + "…"


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(AUTH_COOKIE, token, max_age=AUTH_AGE, httponly=True,
                        secure=get_settings().auth_cookie_secure, samesite="lax", path="/")


async def require_user(token: str | None = Cookie(default=None, alias=AUTH_COOKIE)) -> Identity:
    if not token or len(token) != 64:
        raise HTTPException(401, "Please sign in")
    async with driver().session() as session:
        result = await session.run(
            "MATCH (s:AuthSession {token_hash: $hash})-[:AUTHENTICATES]->(u:User) "
            "WHERE s.expires_at > $now RETURN u.identifier AS user_id",
            hash=token_digest(token), now=int(time.time()))
        row = await result.single()
    if not row:
        raise HTTPException(401, "Please sign in")
    return Identity(row["user_id"], token_digest(token))


async def issue_session(response: Response, user_id: str) -> None:
    token = secrets.token_hex(32)
    async with driver().session() as session:
        await session.run(
            "MATCH (u:User {identifier: $user_id}) "
            "CREATE (s:AuthSession {token_hash: $hash, expires_at: $expires})"
            "-[:AUTHENTICATES]->(u)",
            user_id=user_id, hash=token_digest(token), expires=int(time.time()) + AUTH_AGE)
    set_auth_cookie(response, token)


@router.post("/auth/register", status_code=201)
async def register(credentials: Credentials, response: Response):
    kind, value = normalize(credentials.identifier)
    user_id = f"account_{uuid4().hex}"
    db = driver()
    try:
        async with db.session() as session:
            for query in (
                "CREATE CONSTRAINT account_email IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
                "CREATE CONSTRAINT account_phone IF NOT EXISTS FOR (u:User) REQUIRE u.phone IS UNIQUE",
            ):
                result = await session.run(query)
                await result.consume()
            # An account has exactly one login identifier. A random, server-issued
            # identifier cannot overlap with old browser-cookie identities.
            result = await session.run(
                f"CREATE (u:User {{identifier: $id, {kind}: $value, password_hash: $password}}) "
                "RETURN u.identifier AS id",
                id=user_id, value=value, password=password_hash(credentials.password))
            await result.consume()
    except Exception as exc:
        # Unique constraint violations (including concurrent registrations).
        if getattr(exc, "code", "") in (
            "Neo.ClientError.Schema.ConstraintValidationFailed",
            "Neo.ClientError.Schema.ConstraintValidationFailure",
        ):
            raise HTTPException(409, "Account already exists") from exc
        raise
    await issue_session(response, user_id)
    return {"user_id": user_id}


@router.post("/auth/login")
async def login(credentials: Credentials, response: Response):
    kind, value = normalize(credentials.identifier)
    async with driver().session() as session:
        result = await session.run(
            f"MATCH (u:User {{{kind}: $value}}) "
            "RETURN u.identifier AS id, u.password_hash AS hash, "
            "u.locked_until AS locked_until",
            value=value)
        row = await result.single()
        if row and (row["locked_until"] or 0) > int(time.time()):
            raise HTTPException(429, "Too many attempts; try again later")
        if not row or not verify_password(credentials.password, row["hash"]):
            if row:
                result = await session.run(
                    "MATCH (u:User {identifier: $id}) "
                    "SET u.failed_logins = coalesce(u.failed_logins, 0) + 1 "
                    "SET u.locked_until = CASE WHEN u.failed_logins >= 5 "
                    "THEN $until ELSE coalesce(u.locked_until, 0) END",
                    id=row["id"], until=int(time.time()) + 900)
                await result.consume()
            raise HTTPException(401, "Invalid credentials")
        result = await session.run(
            "MATCH (u:User {identifier: $id}) "
            "SET u.failed_logins = 0, u.locked_until = 0", id=row["id"])
        await result.consume()
    await issue_session(response, row["id"])
    return {"user_id": row["id"]}


@router.get("/auth/me")
async def me(identity: Identity = Depends(require_user)):
    return {"user_id": identity.user_id}


@router.post("/auth/logout")
async def logout(response: Response, identity: Identity = Depends(require_user)):
    async with driver().session() as session:
        await session.run("MATCH (s:AuthSession {token_hash: $hash}) DETACH DELETE s",
                          hash=identity.token_hash)
    response.delete_cookie(AUTH_COOKIE, path="/")
    return {"ok": True}


async def claim_session(user_id: str, session_id: str, *, create: bool = False) -> None:
    """Never load conversation messages until ownership has been checked."""
    async with driver().session() as session:
        if create:
            result = await session.run(
                "MATCH (u:User {identifier: $user_id}) "
                "CREATE (u)-[:OWNS_CHAT]->(c:ChatSession "
                "{session_id: $session_id, created_at: $now}) "
                "RETURN c.session_id AS id", user_id=user_id, session_id=session_id,
                now=int(time.time()))
            await result.consume()
        else:
            result = await session.run(
                "MATCH (u:User {identifier: $user_id})-[:OWNS_CHAT]->"
                "(c:ChatSession {session_id: $session_id}) RETURN c.session_id AS id",
                user_id=user_id, session_id=session_id)
            if not await result.single():
                raise HTTPException(403, "This conversation is not yours")


@router.get("/auth/history")
async def history(identity: Identity = Depends(require_user)):
    async with driver().session() as session:
        result = await session.run(
            "MATCH (u:User {identifier: $user_id})-[:OWNS_CHAT]->(c:ChatSession) "
            "MATCH (u)-[:HAS_CONVERSATION]->(conv:Conversation {session_id: c.session_id}) "
            "MATCH (conv)-[:HAS_MESSAGE]->(m:Message {role: 'user'}) "
            "WITH c, m ORDER BY m.timestamp ASC "
            "WITH c, collect(m.content)[..10] AS user_messages "
            "RETURN c.session_id AS session_id, user_messages "
            "ORDER BY c.created_at DESC LIMIT 50", user_id=identity.user_id)
        rows = [r.data() async for r in result]
    return [{"session_id": row["session_id"],
             "title": next((title for text in row["user_messages"]
                            if (title := conversation_title(text)) != "گفتگوی سفر"),
                           "گفتگوی سفر")} for row in rows]


@router.get("/auth/history/{session_id}")
async def history_messages(session_id: str, identity: Identity = Depends(require_user)):
    await claim_session(identity.user_id, session_id)
    async with driver().session() as session:
        result = await session.run(
            "MATCH (u:User {identifier: $user_id})-[:HAS_CONVERSATION]->"
            "(conv:Conversation {session_id: $session_id})-[:HAS_MESSAGE]->(m:Message) "
            "WHERE m.role IN ['user', 'assistant'] "
            "RETURN m.role AS role, m.content AS content "
            "ORDER BY m.timestamp ASC, m.id ASC",
            user_id=identity.user_id, session_id=session_id)
        messages = [row.data() async for row in result]
    return {"session_id": session_id, "messages": messages}


@router.post("/auth/history/{session_id}/finalize")
async def finalize_conversation(session_id: str, identity: Identity = Depends(require_user)):
    """Explicit end-of-conversation hook; safe to retry on the same session."""
    await claim_session(identity.user_id, session_id)
    try:
        updated = await finalize_preferences(driver(), identity.user_id, session_id)
    except Exception as exc:
        logger.warning("Could not extract preferences for session %s", session_id, exc_info=True)
        raise HTTPException(502, "Preference extraction unavailable; please retry later") from exc
    return {"updated": updated}
