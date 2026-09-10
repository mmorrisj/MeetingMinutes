"""Where transcribed segments go. Each sink is small so they compose."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Protocol

import httpx

from meetingminutes.transcribe.base import Segment

log = logging.getLogger(__name__)


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


class ApiSink:
    """Stream segments to the MeetingMinutes API.

    Creates the meeting on construction and ends it on ``close()``. Segments are posted per
    batch (one transcribed chunk); a failed POST is retried a few times and then logged and
    dropped rather than stalling transcription, since the JSONL sink still has every line.
    """

    def __init__(
        self,
        base_url: str,
        *,
        title: str | None = None,
        source_device: str | None = None,
        client: httpx.Client | None = None,
        retries: int = 3,
    ) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=10.0)
        self._retries = retries
        body = {"title": title, "source_device": source_device}
        resp = self._client.post("/meetings", json=body)
        resp.raise_for_status()
        self.meeting_id: str = resp.json()["id"]

    def write(self, segments: Iterable[Segment]) -> None:
        batch = [seg.to_dict() for seg in segments]
        if not batch:
            return
        self._post_with_retry(f"/meetings/{self.meeting_id}/segments", {"segments": batch})

    def close(self) -> None:
        try:
            self._post_with_retry(f"/meetings/{self.meeting_id}/end", None)
        finally:
            self._client.close()

    def _post_with_retry(self, path: str, json_body: dict | None) -> None:
        delay = 0.5
        for attempt in range(1, self._retries + 1):
            try:
                resp = self._client.post(path, json=json_body)
                resp.raise_for_status()
                return
            except httpx.HTTPError as exc:
                if attempt == self._retries:
                    log.error("giving up on POST %s after %d attempts: %s", path, attempt, exc)
                    return
                log.warning("POST %s failed (attempt %d): %s", path, attempt, exc)
                time.sleep(delay)
                delay *= 2
