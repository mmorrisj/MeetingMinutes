from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from meetingminutes.api import schemas
from meetingminutes.db.models import Meeting, Segment, utcnow

router = APIRouter(prefix="/meetings", tags=["meetings"])


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session


Session = Annotated[AsyncSession, Depends(get_session)]


async def _load_meeting(session: AsyncSession, meeting_id: uuid.UUID) -> Meeting:
    meeting = await session.get(Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "meeting not found")
    return meeting


@router.post("", response_model=schemas.MeetingOut, status_code=status.HTTP_201_CREATED)
async def create_meeting(body: schemas.MeetingCreate, session: Session) -> Meeting:
    meeting = Meeting(**body.model_dump(exclude_none=True))
    session.add(meeting)
    await session.commit()
    await session.refresh(meeting)
    return meeting


@router.get("", response_model=schemas.MeetingList)
async def list_meetings(
    session: Session,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> schemas.MeetingList:
    total = await session.scalar(select(func.count()).select_from(Meeting))
    rows = await session.scalars(
        select(Meeting).order_by(Meeting.started_at.desc()).limit(limit).offset(offset)
    )
    return schemas.MeetingList(items=list(rows), total=total or 0)


@router.get("/{meeting_id}", response_model=schemas.MeetingDetail)
async def get_meeting(meeting_id: uuid.UUID, session: Session) -> Meeting:
    meeting = await session.scalar(
        select(Meeting).where(Meeting.id == meeting_id).options(selectinload(Meeting.segments))
    )
    if meeting is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "meeting not found")
    return meeting


@router.post("/{meeting_id}/segments", response_model=schemas.BatchResult)
async def append_segments(
    meeting_id: uuid.UUID, body: schemas.SegmentBatch, session: Session
) -> schemas.BatchResult:
    meeting = await _load_meeting(session, meeting_id)
    if meeting.ended_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "meeting has already ended")
    session.add_all(Segment(meeting_id=meeting.id, **seg.model_dump()) for seg in body.segments)
    await session.commit()
    return schemas.BatchResult(inserted=len(body.segments))


@router.post("/{meeting_id}/end", response_model=schemas.MeetingOut)
async def end_meeting(meeting_id: uuid.UUID, session: Session) -> Meeting:
    meeting = await _load_meeting(session, meeting_id)
    if meeting.ended_at is None:  # idempotent: ending twice keeps the first timestamp
        meeting.ended_at = utcnow()
        await session.commit()
        await session.refresh(meeting)
    return meeting
