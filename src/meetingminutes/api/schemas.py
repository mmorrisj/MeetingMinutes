"""Request/response models for the HTTP API. Kept separate from the ORM models on purpose."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MeetingCreate(BaseModel):
    title: str | None = Field(None, max_length=300)
    source_device: str | None = Field(None, max_length=300)
    started_at: datetime | None = None


class SegmentIn(BaseModel):
    text: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    confidence: float | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def _end_after_start(self) -> SegmentIn:
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be >= start_seconds")
        return self


class SegmentBatch(BaseModel):
    segments: list[SegmentIn] = Field(min_length=1, max_length=1000)


class SegmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float | None


class MeetingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    source_device: str | None
    started_at: datetime
    ended_at: datetime | None
    created_at: datetime


class MeetingDetail(MeetingOut):
    segments: list[SegmentOut]


class MeetingList(BaseModel):
    items: list[MeetingOut]
    total: int


class BatchResult(BaseModel):
    inserted: int
