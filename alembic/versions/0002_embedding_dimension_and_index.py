"""fix embedding dimension, add embedding_model and HNSW index

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-10
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

DIM = 384


def upgrade() -> None:
    # No rows carry embeddings before this migration, so the cast is safe.
    op.execute(f"ALTER TABLE segments ALTER COLUMN embedding TYPE vector({DIM})")
    op.add_column("segments", sa.Column("embedding_model", sa.String(200), nullable=True))
    op.execute(
        "CREATE INDEX ix_segments_embedding_hnsw ON segments "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX ix_segments_embedding_hnsw")
    op.drop_column("segments", "embedding_model")
    op.execute("ALTER TABLE segments ALTER COLUMN embedding TYPE vector")
