from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.app.api.routes import educator, health, index, mindmap, search, verify
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.db.postgres import Postgres
from backend.app.db.qdrant import QdrantStore
from backend.app.db.redis import RedisCache
from backend.app.retrieval.embedder import Embedder


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.postgres = Postgres(
            database_url=settings.database_url,
            min_size=settings.postgres_min_size,
            max_size=settings.postgres_max_size,
        )
        qdrant_api_key = (
            settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        )
        openai_api_key = (
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        )
        self.qdrant = QdrantStore(settings.qdrant_url_str, qdrant_api_key)
        self.redis = RedisCache(settings.redis_url)
        self.embedder = Embedder(
            model=settings.openai_embedding_model,
            api_key=openai_api_key,
            vector_size=settings.qdrant_vector_size,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = structlog.get_logger(__name__)
    state = AppState(settings)
    app.state.services = state

    await state.postgres.connect()
    logger.info("application_started", app_env=settings.app_env)
    try:
        yield
    finally:
        await state.postgres.close()
        await state.qdrant.close()
        await state.redis.close()
        logger.info("application_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Allied Health AI Mind Map API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def bind_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.clear_contextvars()
        return response

    @app.exception_handler(UnexpectedResponse)
    async def qdrant_exception_handler(_request: Request, exc: UnexpectedResponse) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": "Vector store request failed", "error": str(exc)},
        )

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(verify.router)
    app.include_router(mindmap.router)
    app.include_router(index.router)
    app.include_router(educator.router)
    return app


app = create_app()
