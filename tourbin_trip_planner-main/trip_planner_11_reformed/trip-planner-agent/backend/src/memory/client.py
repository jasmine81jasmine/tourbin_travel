"""Memory client factory and lifecycle management.

Reuses neo4j-agent-memory (the library behind Lenny's Memory) for the three
generic memory layers -- short-term conversation, long-term
preferences/entities, and reasoning traces. The tourism knowledge graph
itself (Destination, City, Category, ...) is queried separately through
`src.memory.graph.TripGraphRepository`, which shares the same Neo4j driver.
"""

import logging

from neo4j_agent_memory import ExtractionConfig, ExtractorType, MemoryClient, MemorySettings, Neo4jConfig
from neo4j_agent_memory.llm import from_provider
from neo4j_agent_memory.memory.long_term import DeduplicationConfig

from src.config import get_settings

logger = logging.getLogger(__name__)

_memory_client: MemoryClient | None = None
_memory_connected: bool = False
_embedding_provider = None


def build_embedding_provider():
    """Build the shared provider used by memory and destination retrieval."""
    settings = get_settings()
    embed_kwargs: dict = {}
    if settings.embedding_model.startswith("openai/"):
        embed_kwargs["api_key"] = settings.openai_api_key.get_secret_value() or "not-needed"
        if settings.embedding_base_url:
            embed_kwargs["api_base"] = settings.embedding_base_url
        embed_kwargs["dimensions"] = settings.embedding_dimensions
    return from_provider(settings.embedding_model, kind="embedding", **embed_kwargs)


async def init_memory_client() -> MemoryClient | None:
    global _memory_client, _memory_connected, _embedding_provider

    if _memory_client is not None:
        return _memory_client

    settings = get_settings()

    _embedding_provider = build_embedding_provider()

    llm_kwargs: dict = {}
    if settings.llm_model.startswith("openai/"):
        llm_kwargs["api_key"] = settings.llm_api_key.get_secret_value() or "not-needed"
        if settings.llm_base_url:
            llm_kwargs["api_base"] = settings.llm_base_url
    elif settings.llm_model.startswith("anthropic/") and settings.anthropic_api_key:
        llm_kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()
    llm_provider = from_provider(settings.llm_model, kind="llm", **llm_kwargs)

    # Extraction runs purely through the configured LLM (whichever provider
    # you set LLM_MODEL/LLM_BASE_URL to -- AvalAI, university, etc.), rather
    # than the default 3-stage pipeline (spaCy + GLiNER + LLM fallback).
    # Reasons:
    #   - spaCy needs a separately-installed English model (en_core_web_sm)
    #     that isn't installed here, so that stage was silently no-op'ing.
    #   - GLiNER downloads a ~1.7GB local model on first use and only
    #     supports a fixed menu of *generic* domain presets (poleo, podcast,
    #     news, scientific, business, entertainment, medical, legal) -- none
    #     of which is "trip planning", so `gliner_schema="custom"` was an
    #     invalid value that silently fell back to generic labels anyway.
    #   - The LLM stage alone needs no local downloads, already runs through
    #     whichever provider is configured, and -- unlike the other two --
    #     lets us hand it real trip-domain entity type names in plain text.
    extraction_config = ExtractionConfig(
        extractor_type=ExtractorType.LLM,
        # NOTE: the extractor turns each entity_type string into a real,
        # additional Neo4j label (PascalCased -- "VEHICLE" -> :Vehicle),
        # stamped onto its own generic memory-graph nodes. Several of the
        # obvious trip-domain names collide 1:1 with labels the destinations
        # ETL already owns (see etl/constraints.cypher: Destination,
        # Province, City, Category, Season, TripType, Vehicle, Landmark),
        # each with its own uniqueness constraint -- so extracting e.g. a
        # vehicle mention crashes with a ConstraintValidationFailed the
        # moment its name collides with an existing destination-graph node.
        # Every type below is deliberately named to avoid that overlap.
        entity_types=[
            "DESTINATION_MENTION",  # a specific place mentioned, e.g. "دهستان لفور"
            "PROVINCE_MENTION",
            "CITY_MENTION",
            "TRIP_CATEGORY",        # e.g. nature, waterfall, historical, ecotourism
            "TRIP_SEASON",          # بهار/تابستان/پاییز/زمستان
            "TRAVEL_STYLE",         # family, friends, solo, romantic, adventure, ...
            "VEHICLE_MENTION",
            "LANDMARK_MENTION",
            "DATE_OR_DURATION",     # "next weekend", "2 days", specific dates
            "GROUP_SIZE",           # "3 friends", "family of 4", "با همسرم"
            "BUDGET",
            "PREFERENCE",           # free-form taste, e.g. "prefers quiet places"
            "PERSON",               # travelers/companions mentioned by name or role
        ],
        extract_relations=True,
        extract_preferences=True,
    )

    memory_settings = MemorySettings(
        neo4j=Neo4jConfig(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
        ),
        embedding=_embedding_provider,
        llm=llm_provider,
        extraction=extraction_config,
    )

    _memory_client = MemoryClient(memory_settings)

    try:
        await _memory_client.connect()
        _memory_connected = True
        _memory_client.long_term._deduplication = DeduplicationConfig(
            auto_merge_threshold=settings.dedup_auto_merge_threshold,
            flag_threshold=settings.dedup_flag_threshold,
            use_fuzzy_matching=True,
        )
        logger.info("Connected to Neo4j memory graph")
    except Exception as e:
        logger.warning("Failed to connect to Neo4j memory graph: %s", e)
        _memory_connected = False

    return _memory_client


def get_memory_client() -> MemoryClient | None:
    if not _memory_connected:
        return None
    return _memory_client


def get_embedding_provider():
    return _embedding_provider


def is_memory_connected() -> bool:
    return _memory_connected


async def close_memory_client() -> None:
    global _memory_client, _memory_connected, _embedding_provider
    if _memory_client is not None and _memory_connected:
        await _memory_client.close()
    _memory_client = None
    _memory_connected = False
    _embedding_provider = None
