"""FastAPI application factory.

Run locally with ``meetingminutes serve`` or ``uvicorn meetingminutes.api.app:app``.
The engine is created in the lifespan handler and shared via ``app.state`` so tests can point
the same app at a throw-away database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from meetingminutes import __version__
from meetingminutes.api.routes import router
from meetingminutes.api.search import router as search_router
from meetingminutes.config import Settings, get_settings
from meetingminutes.db.session import make_engine, make_session_factory
from meetingminutes.embeddings.base import Embedder, build_embedder

_UNSET = object()


def create_app(
    settings: Settings | None = None, embedder: Embedder | None | object = _UNSET
) -> FastAPI:
    """Build the app. Pass ``embedder`` explicitly (including ``None``) to bypass model loading."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url, echo=settings.db_echo)
        app.state.engine = engine
        app.state.session_factory = make_session_factory(engine)
        # Model load happens here, not at import, so `meetingminutes --help` stays instant.
        app.state.embedder = build_embedder(settings) if embedder is _UNSET else embedder
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(title="MeetingMinutes", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)
    app.include_router(search_router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        async with app.state.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
