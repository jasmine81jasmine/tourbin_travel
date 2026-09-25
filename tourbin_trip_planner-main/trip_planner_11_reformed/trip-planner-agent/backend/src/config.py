"""Application configuration settings."""

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(
        default="neo4j",
        validation_alias=AliasChoices("NEO4J_USERNAME", "NEO4J_USER", "neo4j_username", "neo4j_user"),
    )
    neo4j_password: SecretStr = Field(default=SecretStr("password"))

    # LLM / embedding providers (same convention as neo4j-agent-memory)
    llm_model: str = Field(
        default="openai/gpt-4.1-mini",
        description="'openai/<model>' or 'anthropic/<model>'. AvalAI's models "
        "(gpt-4.1-mini, etc.) go through the openai/ prefix since it's an "
        "OpenAI-compatible proxy -- see llm_base_url below.",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("LLM_API_KEY", "AVALAI_API_KEY", "OPENAI_API_KEY"),
        description="Key for whichever provider llm_model points at.",
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_BASE_URL", "AVALAI_BASE_URL"),
        description="Override for OpenAI-compatible proxies, e.g. https://api.avalai.ir/v1",
    )

    embedding_model: str = Field(default="openai/text-embedding-3-small")
    embedding_dimensions: int = Field(
        default=768,
        description="Must match whatever `dimensions=` you actually request from "
        "the embedding API, and etl/constraints.cypher's vector index size. "
        "text-embedding-3-small defaults to 1536 unless you explicitly ask for "
        "fewer, as AvalAI's sample script does (dimensions=768).",
    )
    # Separate from llm_api_key on purpose: AvalAI's sample only demonstrates
    # chat completions, so embeddings default to going straight to OpenAI
    # unless you also set EMBEDDING_BASE_URL (e.g. if AvalAI proxies /embeddings too).
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    embedding_base_url: str | None = Field(default=None)
    anthropic_api_key: SecretStr | None = Field(default=None)

    # --- University (FUM) LLM + embedding gateway ---
    # The university's APIs are not OpenAI-wire-compatible (single `query`
    # string, no tool calling, different embedding response shape), so a
    # local adapter (src/adapters/university_proxy.py) translates between
    # them and the OpenAI wire format. These settings configure *that*
    # adapter's outbound calls to the real university endpoints; they're
    # only read when you actually route LLM_BASE_URL / EMBEDDING_BASE_URL
    # at the adapter (see .env.example).
    university_llm_url: str = Field(
        default="http://fumllm.um.ac.ir/llm/query_unified",
        validation_alias=AliasChoices("UNIVERSITY_LLM_URL"),
    )
    university_llm_server_id: int = Field(
        default=1,
        validation_alias=AliasChoices("UNIVERSITY_LLM_SERVER_ID"),
    )
    university_embedding_url: str = Field(
        default="https://ai-gateway.um.ac.ir/ai-embedding/api/embeddings",
        validation_alias=AliasChoices("UNIVERSITY_EMBEDDING_URL"),
    )

    # Preference / reasoning memory behaviour
    dedup_auto_merge_threshold: float = Field(default=0.92)
    dedup_flag_threshold: float = Field(default=0.82)

    # Neshan maps API (geocoding, routing, TSP ordering, isochrone, nearby
    # search) used to ground trip plans in real distances/times. Optional:
    # every call degrades gracefully to DB data + haversine estimates when
    # this key is unset, so leaving it blank never breaks the chat endpoint.
    neshan_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NESHAN_API_KEY"),
    )
    neshan_base_url: str = Field(
        default="https://api.neshan.org",
        validation_alias=AliasChoices("NESHAN_BASE_URL"),
    )

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=True)
    cors_origins_str: str = Field(default="http://localhost:3000", alias="cors_origins")

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_str.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
