"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.adapters import university_proxy
from src.api.routes import chat
from src.config import get_settings
from src.memory.client import close_memory_client, init_memory_client, is_memory_connected

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_settings = get_settings()
if _settings.openai_api_key.get_secret_value():
    os.environ["OPENAI_API_KEY"] = _settings.openai_api_key.get_secret_value()
if _settings.anthropic_api_key and _settings.anthropic_api_key.get_secret_value():
    os.environ["ANTHROPIC_API_KEY"] = _settings.anthropic_api_key.get_secret_value()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_memory_client()
    yield
    await close_memory_client()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Trip Planner Agent",
        description="LLM trip-planning agent on top of a Neo4j tourism knowledge graph",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/api", tags=["chat"])
    # OpenAI-wire-format shim in front of the university LLM/embedding
    # gateway -- only hit when LLM_BASE_URL / EMBEDDING_BASE_URL in .env
    # point here (see .env.example "Option B"). See adapters/university_proxy.py.
    app.include_router(university_proxy.router, prefix="/adapters/university", tags=["adapters"])

    @app.get("/", include_in_schema=False)
    async def frontend():
        return FileResponse(Path(__file__).resolve().parents[2] / "index.html")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "memory_connected": is_memory_connected()}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=settings.debug)
