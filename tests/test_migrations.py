"""The hand-written migration must match the ORM models exactly."""

from __future__ import annotations

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy.ext.asyncio import create_async_engine

from meetingminutes.db.models import Base


async def test_no_drift_between_models_and_migrations(database_url: str):
    engine = create_async_engine(database_url)

    def _diff(sync_conn):
        ctx = MigrationContext.configure(sync_conn, opts={"compare_type": True})
        return compare_metadata(ctx, Base.metadata)

    try:
        async with engine.connect() as conn:
            diff = await conn.run_sync(_diff)
    finally:
        await engine.dispose()
    assert diff == [], diff
