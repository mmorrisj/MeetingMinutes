"""Transcriber interface and the segment type it produces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

from meetingminutes.audio.chunker import AudioChunk


@dataclass(frozen=True, slots=True)
class Segment:
    """A stretch of recognised speech with absolute (session-relative) timestamps."""

    text: str
    start_seconds: float
    end_seconds: float
    confidence: float | None = None  # backend-specific; None when unavailable

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, chunk: AudioChunk) -> list[Segment]:
        """Return segments for ``chunk`` with times offset by ``chunk.start_seconds``."""
        ...
