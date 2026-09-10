from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator, Iterator

import numpy as np
import pytest
from alembic.config import Config

from alembic import command
from meetingminutes.config import Settings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tone(seconds: float, sample_rate: int = 16_000, freq: float = 440.0, amp: float = 0.3):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, sample_rate: int = 16_000):
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


@pytest.fixture
def sr() -> int:
    return 16_000


# --- database -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A throw-away Postgres with pgvector, migrated to head.

    Set MM_TEST_DATABASE_URL to use an existing server instead of the embedded one.
    """
    url = os.environ.get("MM_TEST_DATABASE_URL")
    server = None
    if not url:
        pgserver = pytest.importorskip("pgserver")
        server = pgserver.get_server(tempfile.mkdtemp(prefix="mm-pg-"))
        # pgserver hands back a libpq URI; SQLAlchemy needs the asyncpg dialect spelled out.
        url = server.get_uri().replace("postgresql://", "postgresql+asyncpg://", 1)
        url = url.replace("postgres:@", "postgres@")

    cfg = Config(os.path.join(ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(ROOT, "alembic"))
    cfg.cmd_opts = type("Opts", (), {"x": [f"url={url}"]})()
    command.upgrade(cfg, "head")
    try:
        yield url
    finally:
        if server is not None:
            server.cleanup()


@pytest.fixture
def settings(database_url: str) -> Settings:
    return Settings(database_url=database_url)


class HashEmbedder:
    """Deterministic bag-of-words embedder: shared words -> nearby vectors. No model needed."""

    model_name = "test-hash-v1"
    dim = 384

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            v[hash(word) % self.dim] += 1.0  # PYTHONHASHSEED only changes *which* slot is used
        norm = float(np.linalg.norm(v))
        return (v / norm if norm else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str):
        return self._vec(text)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
async def app(settings: Settings, embedder):
    from sqlalchemy import text

    from meetingminutes.api.app import create_app

    application = create_app(settings, embedder=embedder)
    async with application.router.lifespan_context(application):
        async with application.state.engine.begin() as conn:
            await conn.execute(text("TRUNCATE segments, meetings"))
        yield application


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:  # noqa: F821
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
