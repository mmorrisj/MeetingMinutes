"""Where transcribed segments go. Each sink is small so they compose."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Protocol

from meetingminutes.transcribe.base import Segment


class Sink(Protocol):
    def write(self, segments: Iterable[Segment]) -> None: ...
    def close(self) -> None: ...


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ConsoleSink:
    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream or sys.stdout

    def write(self, segments: Iterable[Segment]) -> None:
        for seg in segments:
            self._stream.write(f"[{_fmt_time(seg.start_seconds)}] {seg.text}\n")
        self._stream.flush()

    def close(self) -> None:
        pass


class JsonlSink:
    """One JSON object per segment. Append-only, so a crash never loses earlier lines."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def write(self, segments: Iterable[Segment]) -> None:
        for seg in segments:
            self._fh.write(json.dumps(seg.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class MultiSink:
    def __init__(self, *sinks: Sink) -> None:
        self._sinks = sinks

    def write(self, segments: Iterable[Segment]) -> None:
        batch = list(segments)
        for sink in self._sinks:
            sink.write(batch)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()


def session_transcript_path(output_dir: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"meeting-{stamp}.jsonl"
