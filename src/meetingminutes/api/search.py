"""Semantic search over segments plus the embedding backfill endpoint."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from meetingminutes.api import schemas
from meetingminutes.api.routes import Session
from meetingminutes.db.models import Meeting, Segment
from meetingminutes.embeddings.base import Embedder

router = APIRouter(tags=["search"])


def _embedder_or_503(request: Request) -> Embedder:
    embedder = request.app.state.embedder
    if embedder is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "embeddings are not configured on this server (install meetingminutes[embed])",
        )
    return embedder


async def embed_new_segments(
    session: AsyncSession, embedder: Embedder | None, segments: list[Segment]
) -> None:
    """Attach vectors to freshly created segments. No-op without an embedder."""
    if embedder is None or not segments:
        return
    vectors = await run_in_threadpool(embedder.embed_documents, [s.text for s in segments])
    for seg, vec in zip(segments, vectors, strict=True):
        seg.embedding = vec
        seg.embedding_model = embedder.model_name


@router.get("/search", response_model=schemas.SearchResponse)
async def search(
    request: Request,
    session: Session,
    q: str = Query(min_length=1, max_length=1000),
    limit: int = Query(10, ge=1, le=100),
    meeting_id: uuid.UUID | None = None,
) -> schemas.SearchResponse:
    embedder = _embedder_or_503(request)
    qvec = await run_in_threadpool(embedder.embed_query, q)

    distance = Segment.embedding.cosine_distance(qvec).label("distance")
    stmt = (
        select(Segment, Meeting.title, distance)
        .join(Meeting, Meeting.id == Segment.meeting_id)
        .where(Segment.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )
    if meeting_id is not None:
        stmt = stmt.where(Segment.meeting_id == meeting_id)

    rows = (await session.execute(stmt)).all()
    hits = [
        schemas.SearchHit(
            segment_id=seg.id,
            meeting_id=seg.meeting_id,
            meeting_title=title,
            text=seg.text,
            start_seconds=seg.start_seconds,
            end_seconds=seg.end_seconds,
            score=1.0 - float(dist),
        )
        for seg, title, dist in rows
    ]
    return schemas.SearchResponse(query=q, hits=hits)


@router.post("/embeddings/backfill", response_model=schemas.BackfillResult)
async def backfill_embeddings(
    request: Request,
    session: Session,
    batch_size: int = Query(256, ge=1, le=2000),
    max_segments: int = Query(10_000, ge=1, le=1_000_000),
) -> schemas.BackfillResult:
    """Embed segments that have no vector, or one from a different model.

    Runs synchronously in batches; call repeatedly (or raise ``max_segments``) for big backlogs.
    """
    embedder = _embedder_or_503(request)
    needs_work = (Segment.embedding.is_(None)) | (Segment.embedding_model != embedder.model_name)
    embedded = 0
    while embedded < max_segments:
        rows = list(
            await session.scalars(
                select(Segment)
                .where(needs_work)
                .order_by(Segment.created_at)
                .limit(min(batch_size, max_segments - embedded))
            )
        )
        if not rows:
            break
        vectors = await run_in_threadpool(embedder.embed_documents, [s.text for s in rows])
        for seg, vec in zip(rows, vectors, strict=True):
            await session.execute(
                update(Segment)
                .where(Segment.id == seg.id)
                .values(embedding=vec, embedding_model=embedder.model_name)
            )
        await session.commit()
        embedded += len(rows)

    remaining = await session.scalar(
        select(Segment.id).where(needs_work).limit(1)  # cheap "is anything left" probe
    )
    return schemas.BackfillResult(embedded=embedded, remaining=remaining is not None)
