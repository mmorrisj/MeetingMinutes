"""Embedding interface. Backends turn text into unit-length float vectors."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from meetingminutes.config import Settings

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed stored text (transcript segments)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Some models want a different prefix for queries."""
        ...


def build_embedder(settings: Settings) -> Embedder | None:
    """Construct the configured embedder, or ``None`` when embeddings are disabled/unavailable.

    ``None`` degrades gracefully: segments are stored without vectors and ``/search`` returns
    503 until an embedder is available; ``POST /embeddings/backfill`` fills the gap later.
    """
    if not settings.embeddings_enabled:
        return None
    try:
        from meetingminutes.embeddings.fastembed_backend import FastEmbedEmbedder  # noqa: PLC0415
    except ImportError:
        log.warning(
            "fastembed is not installed; segments will be stored without embeddings. "
            "Run `pip install 'meetingminutes[embed]'`."
        )
        return None
    embedder = FastEmbedEmbedder(settings.embedding_model)
    if embedder.dim != settings.embedding_dim:
        raise RuntimeError(
            f"{settings.embedding_model!r} produces {embedder.dim}-d vectors but the schema is "
            f"vector({settings.embedding_dim}). Changing models with a different dimension needs "
            "a migration that alters segments.embedding."
        )
    return embedder
